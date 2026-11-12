import re
import time

from .generation import MIN_EVIDENCE_CHARS, MIN_EVIDENCE_SCORE


RECALL_KS = (1, 3, 5)
ABSTENTION_THRESHOLDS = tuple(round(value / 100, 2) for value in range(5, 100, 5))
RETRIEVAL_CONFIGURATIONS = (
    ("BM25 only", "bm25", False),
    ("Dense only", "dense", False),
    ("Hybrid (BM25 + dense, RRF)", "hybrid", False),
    ("Hybrid + cross-encoder rerank", "hybrid", True),
)
STOP_WORDS = {
    "about", "after", "also", "and", "are", "been", "between", "from",
    "have", "into", "more", "must", "not", "shall", "such", "than",
    "that", "the", "their", "there", "these", "this", "under", "what",
    "when", "where", "which", "with",
}


def make_auto_benchmark(index, max_answerable=13, unanswerable_count=3):
    examples = []
    seen_questions = set()
    for chunk in index.chunks:
        text = chunk.get("text", "")
        if len(text) < 80:
            continue
        terms = [
            word.lower()
            for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text)
            if word.lower() not in STOP_WORDS
        ]
        section = chunk.get("section")
        if section:
            topic = " ".join(terms[:3])
            question = f"What does clause {section} state about {topic}?" if topic else (
                f"What does clause {section} state?"
            )
        else:
            if not terms:
                continue
            question = f"What does the document say about {' '.join(terms[:5])}?"
        if question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append({
            "question": question,
            "answerable": True,
            "expected_pages": {(str(chunk["document"]), int(chunk["page"]))},
        })
        if len(examples) >= max_answerable:
            break

    corpus = " ".join(chunk.get("text", "") for chunk in index.chunks).lower()
    for number in range(1, unanswerable_count + 1):
        probe = f"lexisync_eval_probe_{number:03d}"
        while probe in corpus:
            number += unanswerable_count
            probe = f"lexisync_eval_probe_{number:03d}"
        examples.append({
            "question": f"What value is assigned to the identifier {probe}?",
            "answerable": False,
            "expected_pages": set(),
        })
    return examples


def _page_recall(retrieved, relevant_pages):
    retrieved_pages = {
        (str(item["document"]), int(item["page"])) for item in retrieved
    }
    return len(retrieved_pages & relevant_pages) / len(relevant_pages)


def evaluate_index(index, benchmark, rerank=True):
    recall_sums = {
        config[0]: {cutoff: 0.0 for cutoff in RECALL_KS}
        for config in RETRIEVAL_CONFIGURATIONS
    }
    retrieval_latencies = {config[0]: [] for config in RETRIEVAL_CONFIGURATIONS}
    calibration_examples = []
    recall_total = 0

    for example in benchmark:
        question = example["question"]
        answerable = example["answerable"]
        relevant_pages = example["expected_pages"]
        if answerable and relevant_pages:
            recall_total += 1

        retrieved_by_config = {}
        for name, mode, use_reranker in RETRIEVAL_CONFIGURATIONS:
            started = time.perf_counter()
            retrieved = index.retrieve(
                question, max(RECALL_KS), mode=mode, rerank=use_reranker
            )
            retrieval_latencies[name].append(time.perf_counter() - started)
            retrieved_by_config[name] = retrieved
            if answerable and relevant_pages:
                for cutoff in RECALL_KS:
                    recall_sums[name][cutoff] += _page_recall(
                        retrieved[:cutoff], relevant_pages
                    )

        pipeline_name = (
            "Hybrid + cross-encoder rerank" if rerank
            else "Hybrid (BM25 + dense, RRF)"
        )
        pipeline_evidence = retrieved_by_config[pipeline_name]
        score = max(
            (float(item.get("score", 0)) for item in pipeline_evidence),
            default=0.0,
        )
        has_sufficient_text = any(
            len(item.get("text", "")) >= MIN_EVIDENCE_CHARS for item in pipeline_evidence
        )
        calibration_examples.append((answerable, score, has_sufficient_text))

    answerable_count = sum(answerable for answerable, _, _ in calibration_examples)
    unanswerable_count = len(calibration_examples) - answerable_count
    abstention_rows = []
    for threshold in ABSTENTION_THRESHOLDS:
        predictions = [
            score < threshold or not has_sufficient_text
            for _, score, has_sufficient_text in calibration_examples
        ]
        correct = sum(
            prediction == (not example[0])
            for prediction, example in zip(predictions, calibration_examples)
        )
        false_abstentions = sum(
            prediction
            for prediction, example in zip(predictions, calibration_examples)
            if example[0]
        )
        abstention_rows.append({
            "threshold": threshold,
            "abstention_accuracy": correct / len(benchmark) if benchmark else None,
            "false_abstain_rate": (
                false_abstentions / answerable_count if answerable_count else None
            ),
        })

    bm25_loaded = getattr(index, "bm25", None) is not None
    dense_loaded = (
        getattr(index, "model", None) is not None
        and getattr(index, "embeddings", None) is not None
    )
    reranker_loaded = getattr(index, "cross", None) is not None
    recall_rows = []
    for name, _, _ in RETRIEVAL_CONFIGURATIONS:
        models_loaded = {
            "BM25 only": bm25_loaded,
            "Dense only": dense_loaded,
            "Hybrid (BM25 + dense, RRF)": bm25_loaded and dense_loaded,
            "Hybrid + cross-encoder rerank": (
                bm25_loaded and dense_loaded and reranker_loaded
            ),
        }[name]
        recall_rows.append({
            "Configuration": name,
            "Models loaded": models_loaded,
            **{
                f"Recall@{cutoff}": (
                    recall_sums[name][cutoff] / recall_total if recall_total else None
                )
                for cutoff in RECALL_KS
            },
            "Latency (ms)": (
                1000 * sum(retrieval_latencies[name]) / len(retrieval_latencies[name])
                if retrieval_latencies[name] else None
            ),
        })

    pipeline_name = (
        "Hybrid + cross-encoder rerank" if rerank
        else "Hybrid (BM25 + dense, RRF)"
    )
    active_threshold = MIN_EVIDENCE_SCORE
    correctly_classified = 0
    for answerable, score, has_sufficient_text in calibration_examples:
        predicted_abstention = score < active_threshold or not has_sufficient_text
        correctly_classified += predicted_abstention == (not answerable)

    return {
        "questions_used": len(benchmark),
        "full_pipeline_recall_at_5": (
            recall_sums[pipeline_name][5] / recall_total if recall_total else None
        ),
        "mean_latency_ms": (
            1000 * sum(retrieval_latencies[pipeline_name])
            / len(retrieval_latencies[pipeline_name])
            if retrieval_latencies[pipeline_name] else 0
        ),
        "abstention_accuracy": (
            correctly_classified / len(benchmark) if benchmark else None
        ),
        "recall_questions": recall_total,
        "recall_by_configuration": recall_rows,
        "abstention_calibration": abstention_rows,
        "calibration_answerable_questions": answerable_count,
        "calibration_unanswerable_questions": unanswerable_count,
    }
