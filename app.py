import streamlit as st
from pathlib import Path
import pandas as pd
import json, time, os, tempfile
from lexisync.ingestion import ingest_pdf
from lexisync.retrieval import HybridIndex
from lexisync.generation import answer_question
from lexisync.evaluation import evaluate_dataset, make_starter_dataset
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
    st.subheader("3. Evaluation — held-out questions")
    st.write("Upload a CSV with columns: question, expected_answer, answerable (true/false), optional expected_pages. A starter dataset is available below; replace its examples with questions grounded in your own uploaded PDFs.")
    starter=make_starter_dataset()
    st.download_button("Download evaluation CSV template", starter.to_csv(index=False).encode(), "evaluation_template.csv","text/csv")
    eval_file=st.file_uploader("Upload held-out evaluation CSV",type=["csv"],key="evalcsv")
    if eval_file and st.session_state.index:
        data=pd.read_csv(eval_file)
        if st.button("Run evaluation",type="primary"):
            with st.spinner("Running retrieval and answer evaluation…"):
                report,rows=evaluate_dataset(data,st.session_state.index,groq_key,model,top_k,use_reranker)
            st.session_state.eval_report=report; st.session_state.eval_rows=rows
    if st.session_state.get("eval_report"):
        rep=st.session_state.eval_report
        cols=st.columns(4)
        for col,(label,val) in zip(cols,rep.items()):
            col.metric(label,str(val))
        st.dataframe(pd.DataFrame(st.session_state.eval_rows),use_container_width=True)
        st.download_button("Download evaluation results",pd.DataFrame(st.session_state.eval_rows).to_csv(index=False).encode(),"evaluation_results.csv","text/csv")
    else:
        st.info("The assignment requires at least 15 held-out questions, including at least 3 unanswerable questions. Report manual answer correctness, citation precision, abstention behaviour and latency.")

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
