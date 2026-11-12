import re, math
from collections import defaultdict
from rank_bm25 import BM25Okapi
import numpy as np

def tokenize(s):
    # Preserve dotted clause identifiers as atomic tokens (e.g. 7.2).
    return re.findall(r'\d+(?:\.\d+)+|[a-zA-Z]+|[€$£]?\d+(?:[.,]\d+)?%?',s.lower())

class HybridIndex:
    def __init__(self,pages):
        self.pages=pages; self.chunks=[]; self.bm25=None; self.embeddings=None; self.model=None; self.cross=None
    def build(self):
        for p in self.pages:
            text=p["text"]
            # Section-aware chunking: split at headings; retain page and heading metadata.
            parts=re.split(r'(?im)(?=^(?:(?:section|article|clause)\s+)?\d+(?:\.\d+){0,5}\s+)',text)
            if len(parts)<=1: parts=[text]
            for part in parts:
                part=part.strip()
                if len(part)<35: continue
                sec=None
                m=re.search(r'(?im)^\s*(?:(?:section|article|clause)\s+)?(\d+(?:\.\d+){0,5})\b',part)
                if m: sec=m.group(1)
                self.chunks.append({"document":p["document"],"page":p["page"],"section":sec,
                                    "text":part[:7000],"tables":p.get("tables",[]),"bbox":p.get("bbox",[])[:8]})
        if not self.chunks: raise ValueError("No searchable text found. Check PDF extraction or OCR installation.")
        self.bm25=BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings=self.model.encode([c["text"] for c in self.chunks],normalize_embeddings=True,show_progress_bar=False)
    def _dense(self,q):
        v=self.model.encode([q],normalize_embeddings=True)[0]
        return np.asarray(self.embeddings)@v
    def retrieve(self,q,k=8,mode="hybrid",rerank=True):
        bt=np.asarray(self.bm25.get_scores(tokenize(q)))
        dt=self._dense(q)
        bi=np.argsort(bt)[::-1][:max(k*3,k)]
        di=np.argsort(dt)[::-1][:max(k*3,k)]
        if mode=="bm25": ids=list(bi[:k])
        elif mode=="dense": ids=list(di[:k])
        else:
            ranks=defaultdict(float)
            for rank,idx in enumerate(bi,1): ranks[int(idx)]+=1/(60+rank)
            for rank,idx in enumerate(di,1): ranks[int(idx)]+=1/(60+rank)
            ids=[i for i,_ in sorted(ranks.items(),key=lambda x:x[1],reverse=True)[:max(k*3,k)]]
        for idx in ids:
            self.chunks[idx]["bm25_score"]=float(bt[idx]); self.chunks[idx]["dense_score"]=float(dt[idx])
            self.chunks[idx]["score"]=float((bt[idx]/(abs(bt).max()+1e-9)+1)/2*.5+(dt[idx]+1)/2*.5)
        if rerank and mode=="hybrid" and len(ids)>1:
            try:
                if self.cross is None:
                    from sentence_transformers import CrossEncoder
                    self.cross=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                pairs=[[q,self.chunks[i]["text"][:2500]] for i in ids]
                scores=self.cross.predict(pairs)
                ids=[x for _,x in sorted(zip(scores,ids),reverse=True)]
                for s,i in zip(sorted(scores,reverse=True),ids): self.chunks[i]["rerank_score"]=float(s)
            except Exception:
                pass
        return [dict(self.chunks[i]) for i in ids[:k]]
    def ablate(self,q,k=8):
        return {m:self.retrieve(q,k,mode=m,rerank=False) for m in ("bm25","dense","hybrid")}
