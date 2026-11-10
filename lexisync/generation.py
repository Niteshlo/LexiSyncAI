import os, json, re
from openai import OpenAI

ABSTAIN="The documents do not contain enough evidence to answer this."
def _conf(evidence):
    if not evidence: return "INSUFFICIENT EVIDENCE"
    best=max((x.get("score",0) for x in evidence),default=0)
    # Conservative gate: require meaningful lexical/dense relevance and non-trivial text.
    if best<0.48 or max(len(x["text"]) for x in evidence)<80: return "INSUFFICIENT EVIDENCE"
    if best>=0.72: return "HIGH"
    if best>=0.58: return "MEDIUM"
    return "LOW"

def _conflicts(evidence):
    # Surface same metric/amount with different values across separate documents.
    vals={}
    for e in evidence:
        for m in re.finditer(r'(?i)(revenue|liability cap|indemnity|fine|penalty|turnover|term)\D{0,35}((?:€|EUR|USD|\$|£)?\s?\d[\d,.]*(?:\s?(?:million|m|billion|bn|%))?)',e["text"]):
            key=m.group(1).lower(); val=re.sub(r'\s+','',m.group(2).lower())
            vals.setdefault(key,{}).setdefault(val,set()).add(e["document"])
    return [f"{k}: {', '.join(v)} appears in different document(s)" for k,v in vals.items() if len(v)>1]

def answer_question(question,index,api_key,model,k=8,use_reranker=True):
    evidence=index.retrieve(question,k,rerank=use_reranker)
    conf=_conf(evidence)
    if conf=="INSUFFICIENT EVIDENCE":
        return {"answer":ABSTAIN,"explanation":"The retrieval sufficiency gate did not find passages strong enough to support a reliable answer. No legal conclusion has been inferred.",
                "confidence":conf,"abstained":True,"citations":[],"conflicts":[]}
    if not api_key:
        return {"answer":"Add your Groq API key in the sidebar to generate an answer. Retrieved evidence is shown below; no answer has been generated.",
                "explanation":"Evidence retrieval succeeded, but generation requires a Groq API key.","confidence":"LOW",
                "abstained":False,"citations":_citations(evidence),"conflicts":_conflicts(evidence)}
    client=OpenAI(api_key=api_key,base_url="https://api.groq.com/openai/v1")
    context=[]
    for i,e in enumerate(evidence):
        context.append(f"[E{i+1}] Document: {e['document']} | Page: {e['page']} | Section: {e.get('section')}\n{e['text'][:5000]}")
    prompt=f"""You are LexiSyncAI, a cautious legal document analyst. Answer ONLY from the evidence below.
If evidence does not establish the answer, say exactly: "{ABSTAIN}".
Every factual claim must cite evidence IDs like [E1] and include document, page and section/clause when available.
Distinguish facts from inference. For conflicting sources, describe the conflict and do not silently choose one.
Never invent a clause, page, quotation, amount or governing law.
Return concise plain-language answer and a short explanation.

QUESTION: {question}

EVIDENCE:
{chr(10).join(context)}"""
    try:
        resp=client.chat.completions.create(model=model,messages=[{"role":"system","content":"Be precise, evidence-bound, and willing to abstain."},{"role":"user","content":prompt}],temperature=0.1,max_tokens=1200)
        draft=resp.choices[0].message.content or ABSTAIN
        # Verifier pass: ask model to remove unsupported claims; bounded to one pass.
        verify=f"""Verify this draft strictly against the evidence. If any claim lacks direct support or citation, revise it to abstain or qualify it. Do not add facts.
DRAFT: {draft}
QUESTION: {question}
EVIDENCE:
{chr(10).join(context)}
Return only the corrected answer."""
        v=client.chat.completions.create(model=model,messages=[{"role":"user","content":verify}],temperature=0,max_tokens=1000)
        final=v.choices[0].message.content or draft
        abstained=ABSTAIN.lower() in final.lower() or "not contain enough evidence" in final.lower()
        return {"answer":final,"explanation":"Answer generated from retrieved passages and passed through one bounded evidence-verification pass.",
                "confidence":"INSUFFICIENT EVIDENCE" if abstained else conf,"abstained":abstained,
                "citations":_citations(evidence),"conflicts":_conflicts(evidence)}
    except Exception as ex:
        return {"answer":f"Groq API error: {type(ex).__name__}: {ex}","explanation":"Check the API key, selected model, connectivity and account limits.",
                "confidence":"LOW","abstained":False,"citations":_citations(evidence),"conflicts":_conflicts(evidence)}

def _citations(evidence):
    out=[]
    for e in evidence:
        out.append({"document":e["document"],"page":e["page"],"section":e.get("section"),"text":e["text"][:3500],
                    "score":e.get("rerank_score",e.get("score",0)),"bbox":e.get("bbox",[]),"tables":e.get("tables",[])[:2]})
    return out
