import streamlit as st
from pathlib import Path
import pandas as pd
import json, time, os, tempfile
from lexisync.ingestion import ingest_pdf
from lexisync.retrieval import HybridIndex
from lexisync.generation import answer_question
from lexisync.evaluation import evaluate_index, make_auto_benchmark
from lexisync.analytics import page_anomaly_report

st.set_page_config(page_title="LexiSyncAI", page_icon="⚖️", layout="wide")
st.title("LexiSyncAI")
st.caption("Evidence-grounded legal document analyst · Groq API · Hybrid retrieval · Verifiable page citations")

with st.sidebar:
    st.header("Workspace")
    case_name = st.text_input("Case / matter name", "Demo legal matter")
    st.divider()
    st.subheader("Model settings")
    groq_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY",""), type="password",
                             help="Get a key at https://console.groq.com/keys. It is used only for answer generation.")
    model = st.selectbox("Groq model", ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"], index=0)
    st.caption("The key is not saved to disk. Set GROQ_API_KEY in your environment for persistent local configuration.")
    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Candidate passages", 3, 20, 8)
    use_reranker = st.checkbox("Cross-encoder reranking", True)
    st.caption("First run downloads the embedding and reranker models.")

if "docs" not in st.session_state: st.session_state.docs=[]
if "index" not in st.session_state: st.session_state.index=None
if "last_result" not in st.session_state: st.session_state.last_result=None

tabs=st.tabs(["Documents & indexing","Ask LexiSyncAI","Evaluation","Ablations & diagnostics","Architecture & guide"])

with tabs[0]:
    st.subheader("1. Create a case and upload public legal PDFs")
    st.info("Assignment scope: upload 2–5 public PDFs (contracts, statutes, judgments or regulatory filings). Do not upload confidential/client material.")
    uploads=st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    c1,c2=st.columns([1,3])
    with c1:
        do_index=st.button("Extract & index documents", type="primary", disabled=not uploads)
    with c2:
        st.caption("Digital PDFs are parsed in reading order. Image-only pages are flagged and sent through OCR when Tesseract is available.")
    if do_index:
        if len(uploads)<1:
            st.error("Upload at least one PDF.")
        elif len(uploads)>5:
            st.error("The assignment specifies a bundle of 2–5 public PDFs. Please upload no more than five.")
        else:
            docs=[]
            bar=st.progress(0)
            for i,u in enumerate(uploads):
                data=u.getvalue()
                with st.spinner(f"Extracting {u.name}…"):
                    parsed=ingest_pdf(data, u.name)
                docs.extend(parsed)
                bar.progress((i+1)/len(uploads))
            st.session_state.docs=docs
            st.session_state.index=None
            st.session_state.eval_report=None
            st.session_state.eval_signature=None
            st.success(f"Extracted {len(docs)} page records from {len(uploads)} document(s).")
            with st.spinner("Building BM25 + dense embedding index…"):
                st.session_state.index=HybridIndex(docs)
                st.session_state.index.build()
            st.success("Index ready: BM25 + transformer embeddings + optional cross-encoder reranker.")
    if st.session_state.docs:
        df=pd.DataFrame([{"Document":d["document"],"Page":d["page"],"Characters":len(d["text"]),
                          "Scanned/OCR":d.get("ocr_used",False),"Tables":len(d.get("tables",[])),
                          "Sections detected":", ".join(d.get("sections",[])[:4])} for d in st.session_state.docs])
        st.subheader("Detected document structure")
        st.dataframe(df, use_container_width=True, hide_index=True)
        scanned=sum(bool(d.get("ocr_used")) for d in st.session_state.docs)
        st.metric("Pages indexed",len(st.session_state.docs))
        st.metric("Pages OCR processed",scanned)
        with st.expander("View extracted page text and table data"):
            for d in st.session_state.docs:
                st.markdown(f"**{d['document']} — page {d['page']}**")
                st.text(d["text"][:5000] or "[No text extracted]")
                if d.get("tables"): st.write("Extracted tables:",d["tables"][:3])

with tabs[1]:
    st.subheader("2. Ask a question grounded only in uploaded documents")
    question=st.text_area("Legal / compliance question", placeholder="Does the liability cap in Section 7 exceed the total indemnity obligations in Section 9?", height=90)
    ask=st.button("Generate cited answer", type="primary", disabled=not (question.strip() and st.session_state.index))
    if ask:
        start=time.perf_counter()
        with st.spinner("Retrieving evidence, reranking passages and verifying support…"):
            result=answer_question(question, st.session_state.index, groq_key, model, top_k, use_reranker)
        result["latency_seconds"]=round(time.perf_counter()-start,2)
        st.session_state.last_result=result
    r=st.session_state.last_result
    if r:
        st.divider()
        status=r.get("confidence","INSUFFICIENT EVIDENCE")
        st.subheader("Answer")
        color={"HIGH":"🟢","MEDIUM":"🟡","LOW":"🟠","INSUFFICIENT EVIDENCE":"🔴"}.get(status,"🔴")
        st.markdown(f"**Confidence: {color} {status}** · **Latency:** {r.get('latency_seconds','—')} sec · **Abstained:** {'Yes' if r.get('abstained') else 'No'}")
        st.write(r["answer"])
        if r.get("explanation"): st.caption(r["explanation"])
        if r.get("conflicts"): st.warning("Potential cross-document conflict(s): "+ " | ".join(r["conflicts"]))
        st.subheader("Evidence and citations")
        if r.get("citations"):
            for i,c in enumerate(r["citations"],1):
                with st.expander(f"[{i}] {c['document']} · page {c['page']} · {c.get('section') or 'Section not detected'} · relevance {c.get('score',0):.3f}", expanded=True):
                    st.write(c["text"])
                    if c.get("bbox"): st.caption(f"Region (PDF points): {c['bbox']}")
                    if c.get("tables"): st.write("Table evidence:",c["tables"])
        else: st.info("No sufficiently reliable supporting passage was retrieved.")
        st.caption("Every citation points to an extracted page and passage. Always verify against the source PDF before relying on legal conclusions.")
        st.download_button("Download answer + evidence (JSON)", json.dumps(r,indent=2,ensure_ascii=False), file_name="lexisync_answer.json", mime="application/json")

with tabs[2]:
    st.subheader("3. Evaluation — automatic retrieval benchmark")
    st.write(
        "The benchmark uses queries derived from indexed passages with known source pages, "
        "plus synthetic absent-identifier probes for abstention calibration."
    )
    if not st.session_state.index:
        st.info("Upload and index documents in the Documents & indexing tab to run the evaluation.")
    elif st.button("Run evaluation",type="primary"):
        benchmark=make_auto_benchmark(st.session_state.index)
        with st.spinner(f"Evaluating {len(benchmark)} automatic benchmark queries…"):
            st.session_state.eval_report=evaluate_index(
                st.session_state.index,benchmark,use_reranker
            )
            st.session_state.eval_signature=(id(st.session_state.index),use_reranker)
    if (
        st.session_state.get("eval_report")
        and st.session_state.get("eval_signature")
        == (id(st.session_state.index),use_reranker)
    ):
        rep=st.session_state.eval_report
        st.caption(
            "Synthetic retrieval diagnostic; does not call Groq or replace a human-labeled QA "
            "evaluation. Latency covers retrieval and the evidence gate."
        )
        cols=st.columns(4)
        recall=rep["full_pipeline_recall_at_5"]
        abstention=rep["abstention_accuracy"]
        cols[0].metric("Questions used",rep["questions_used"])
        cols[1].metric("Recall@5 (full pipeline)",f"{recall:.1%}" if recall is not None else "N/A")
        cols[2].metric("Mean latency (full)",f"{rep['mean_latency_ms']:.0f} ms")
        cols[3].metric("Abstention accuracy",f"{abstention:.0%}" if abstention is not None else "N/A")
        st.subheader("Recall@k per configuration")
        if rep["recall_questions"]:
            st.caption(
                f"Page Recall@k over {rep['recall_questions']} automatic queries with known "
                "source pages."
            )
        else:
            st.warning("No page-grounded benchmark queries could be created from the indexed documents.")
        recall_table=pd.DataFrame(rep["recall_by_configuration"])
        for column in ("Recall@1","Recall@3","Recall@5"):
            recall_table[column]=recall_table[column].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else "N/A"
            )
        recall_table["Latency (ms)"]=recall_table["Latency (ms)"].map(
            lambda value: f"{value:.1f}" if pd.notna(value) else "N/A"
        )
        st.dataframe(recall_table,use_container_width=True,hide_index=True)
        st.subheader("Abstention-gate calibration (threshold sweep)")
        st.caption(
            "Threshold predictions abstain when the top full-pipeline retrieval score is below "
            "the threshold or the evidence text is too short. "
            f"{rep['calibration_answerable_questions']} answerable and "
            f"{rep['calibration_unanswerable_questions']} synthetic unanswerable probes were included."
        )
        calibration_table=pd.DataFrame(rep["abstention_calibration"])
        for column in ("abstention_accuracy","false_abstain_rate"):
            calibration_table[column]=calibration_table[column].map(
                lambda value: (
                    f"{value:.3f}".rstrip("0").rstrip(".") if pd.notna(value) else "N/A"
                )
            )
        st.dataframe(calibration_table,use_container_width=True,hide_index=True)
    elif st.session_state.index:
        st.info("Click Run evaluation to calculate recall, latency, and abstention metrics from the indexed documents.")

with tabs[3]:
    st.subheader("4. Technical diagnostics and bonus features")
    if st.session_state.docs:
        st.markdown("**Page anomaly / extraction diagnostics**")
        st.dataframe(pd.DataFrame(page_anomaly_report(st.session_state.docs)),use_container_width=True,hide_index=True)
    if st.session_state.index and question if 'question' in locals() else False:
        pass
    st.markdown("**Retrieval ablation**")
    st.write("Compare BM25-only, dense-only, and hybrid retrieval on the same query. Use the passages and ranks to explain why exact clause-number retrieval benefits from BM25.")
    q=st.text_input("Ablation query",placeholder="Find clause 7.2",key="ablation_q")
    if st.button("Compare retrieval configurations",disabled=not (st.session_state.index and q.strip())):
        out=st.session_state.index.ablate(q,top_k)
        for mode,items in out.items():
            st.markdown(f"**{mode.upper()}**")
            st.write([f"{x['document']} p.{x['page']} — {x['text'][:240]}" for x in items])
    st.markdown("**Implemented extended features**")
    st.markdown("- OCR fallback for scanned pages (Tesseract; scanned pages are flagged)\n- PDF table extraction and table-aware evidence snippets\n- Page/region bounding-box metadata for citations\n- Cross-document amount/date/percentage discrepancy flagging\n- Draft → verifier → one bounded retry loop\n- BM25-only / dense-only / hybrid retrieval ablation\n- Lightweight page anomaly diagnostics")

with tabs[4]:
    st.subheader("System architecture")
    st.code("""PDFs
  │
  ├─ PyMuPDF: blocks, reading order, page metadata, bounding boxes
  ├─ OCR fallback: Tesseract for image-only pages
  ├─ pdfplumber: table extraction
  ▼
Section-aware page chunks
  ├─ BM25 lexical index (exact clause IDs e.g. 7.2)
  ├─ Sentence-Transformer dense embeddings
  ▼
Reciprocal-rank fusion → Cross-Encoder reranker
  ▼
Evidence sufficiency / abstention gate
  ▼
Groq-hosted LLM (answer only from evidence)
  ├─ Citation validation against retrieved passage IDs
  ├─ Verifier pass + one retry
  └─ Conflict / discrepancy flags
  ▼
Answer + confidence + page/section/region citations + evaluation""",language="text")
    st.markdown("**Run locally**")
    st.code("cp .env.example .env  # add GROQ_API_KEY\\ndocker compose up --build",language="bash")
    st.markdown("**Important limitations**: this is a research/demo assistant, not legal advice. OCR and table extraction quality depends on document quality. Figures/charts are not interpreted by a vision-language model in this version; a chart-only answer should abstain.")
