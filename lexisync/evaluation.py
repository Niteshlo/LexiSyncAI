import pandas as pd, time
from .generation import answer_question

def make_starter_dataset():
    return pd.DataFrame([
        {"question":"What does clause 7.2 say?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"What is the governing law stated in the uploaded agreement?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"Which party must maintain insurance?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"What is the notice period for termination?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"What is the liability cap?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"What indemnities are listed?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"When does the contract commence?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"Which court has jurisdiction?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"What confidentiality obligations apply?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"Can the agreement be assigned without consent?","expected_answer":"Replace with a fact from your PDF","answerable":True,"expected_pages":""},
        {"question":"What is the vendor's stock ticker?","expected_answer":"Not in uploaded documents","answerable":False,"expected_pages":""},
        {"question":"What is the CEO's home address?","expected_answer":"Not in uploaded documents","answerable":False,"expected_pages":""},
        {"question":"What was the weather on the signing date?","expected_answer":"Not in uploaded documents","answerable":False,"expected_pages":""},
        {"question":"What is the exact bank balance of the buyer?","expected_answer":"Not in uploaded documents","answerable":False,"expected_pages":""},
        {"question":"What is the personal phone number of the signatory?","expected_answer":"Not in uploaded documents","answerable":False,"expected_pages":""},
    ])

def evaluate_dataset(df,index,key,model,k=8,rerank=True):
    required={"question","expected_answer","answerable"}
    if not required.issubset(df.columns): raise ValueError("CSV must include question, expected_answer, answerable")
    rows=[]; lat=[]; abstain_correct=0; unans=0; citations=0; total_cites=0
    recall_hits={"bm25":0,"dense":0,"hybrid":0}; recall_total=0
    for _,r in df.iterrows():
        t=time.perf_counter()
        ans=answer_question(str(r.question),index,key,model,k,rerank)
        elapsed=time.perf_counter()-t; lat.append(elapsed)
        expected=bool(r.answerable) if isinstance(r.answerable,bool) else str(r.answerable).strip().lower() in ("true","1","yes","y")
        if not expected:
            unans+=1; abstain_correct+=int(ans["abstained"])
        cs=ans.get("citations",[]); total_cites+=len(cs)
        if "expected_pages" in df.columns and str(r.get("expected_pages","")).strip():
            pages={int(x) for x in str(r.expected_pages).replace(";",",").split(",") if x.strip().isdigit()}
            citations+=sum(1 for c in cs if c["page"] in pages)
            # Retrieval recall@k against manually labelled relevant page numbers.
            for mode in ("bm25","dense","hybrid"):
                found=index.retrieve(str(r.question),k,mode=mode,rerank=(mode=="hybrid" and rerank))
                if any(x["page"] in pages for x in found): recall_hits[mode]+=1
            recall_total+=1
        rows.append({"question":r.question,"answer":ans["answer"],"expected_answer":r.expected_answer,
                     "answerable":expected,"abstained":ans["abstained"],"confidence":ans["confidence"],
                     "latency_seconds":round(elapsed,2),"citation_count":len(cs)})
    # Correctness requires manual review; no automated semantic score is misrepresented as truth.
    report={"Questions":len(rows),"Unanswerable abstention accuracy":f"{abstain_correct}/{unans}" if unans else "N/A",
            "Citation precision (page labels; requires expected_pages)":f"{citations}/{total_cites}" if "expected_pages" in df.columns else "Add expected_pages for scoring",
            "Mean latency (s)":round(sum(lat)/len(lat),2) if lat else 0,
            "Recall@k BM25":f"{recall_hits['bm25']}/{recall_total}" if recall_total else "Add expected_pages",
            "Recall@k dense":f"{recall_hits['dense']}/{recall_total}" if recall_total else "Add expected_pages",
            "Recall@k hybrid":f"{recall_hits['hybrid']}/{recall_total}" if recall_total else "Add expected_pages",
            "Manual answer correctness":"To be scored by reviewer"}
    return report,rows
