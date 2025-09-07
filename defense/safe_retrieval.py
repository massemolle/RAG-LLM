import json, os, re
from rank_bm25 import BM25Okapi

STOP = {
    "the","a","an","is","are","of","in","on","to","for","and","or","what","how","why","where","when",
    "do","does","did","with","from","by","at","as","it","this","that","these","those","be","can"
}

def _keywords(q: str):
    toks = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", q.lower())
    return [t for t in toks if len(t) >= 4 and t not in STOP]

def _kw_hits(text: str, kws):
    hits = 0
    for k in kws:
        if re.search(rf"\b{re.escape(k)}\b", text, flags=re.IGNORECASE):
            hits += 1
    return hits

class SafeIndex:
    def __init__(self, path="./safe_index/index.jsonl"):
        self.path = path
        self.records = []
        self.tokens = []
        self.bm25 = None
        self.reload()

    def reload(self):
        self.records.clear()
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    self.records.append(json.loads(line))
        texts = [r["text"] for r in self.records]
        self.tokens = [t.lower().split() for t in texts] if texts else []
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

    def query(self, q: str, k=5, min_rel=0.35, min_kw=1, max_chunks=4):
        """
        Returns top chunks with score & rel. Gating:
        - relative BM25 >= min_rel of top
        - contains at least min_kw query keywords (word-boundary)
        """
        if not self.bm25:
            return []

        q_tokens = q.lower().split()
        scores = self.bm25.get_scores(q_tokens)          # NumPy array

        # ---- robust emptiness checks for NumPy/list ----
        if scores is None:
            return []
        size = getattr(scores, "size", None)
        if size is not None and size == 0:
            return []
        try:
            # convert to Python list for safe ops (max(), indexing)
            scores = list(scores)
        except Exception:
            return []

        if len(scores) == 0:
            return []

        # widen the candidate pool to avoid early pruning
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(1, k*4)]
        max_s = float(max(scores)) if scores else 1.0
        if max_s == 0.0:
            return []

        kws = _keywords(q)

        results = []
        for i in top_idx:
            r = self.records[i]
            s = float(scores[i])
            rel = s / max_s
            txt = r["text"]
            if rel < float(min_rel):
                continue
            if kws and _kw_hits(txt, kws) < int(min_kw):
                continue
            results.append({
                "text": txt,
                "meta": {
                    "doc": r["doc"], "chunk": r["chunk"], "doc_sha": r["doc_sha"],
                    "collection": r["collection"], "score": s, "rel": rel
                }
            })
            if len(results) >= int(max_chunks):
                break
        return results
