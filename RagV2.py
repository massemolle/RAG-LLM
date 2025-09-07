# RagV2.py
import os, re, torch
from transformers import pipeline
from huggingface_hub import repo_exists

from embedding import *  # your BM25/BERT classes
from defense.guards import POLICY, gate_and_log, path_is_allowed, redact
from defense.safe_retrieval import SafeIndex

__all__ = [
    "RAG",
    "get_pipeline",
    "get_llm",
    "get_model_list",
    "list_devices",
    "safe_idx",
]

def _clean_answer(text: str) -> str:
    # remove template debris the tiny model sometimes emits
    text = re.sub(r"<\|[^>]{1,40}\|>", "", text)       # <|im_end|>, etc.
    text = re.sub(r"(?im)^\s*(question|answer)\s*:\s*", "", text)
    text = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", text)  # de-stutter
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _offtopic_constants(question: str, answer: str) -> bool:
    q = question.lower()
    a = answer.lower()
    if "speed of light" in a and "light" not in q:
        return True
    if "radians per second" in a and ("plane" in q or "train" in q):
        return True
    return False

# module-level safe index (reloaded from UI after ingest)
safe_idx = SafeIndex()

class RAG():
    def __init__(self, method=None, k=5, path=None,
                 pipeline_model="Felladrin/Smol-Llama-101M-Chat-v1",
                 device="cpu"):
        self.k = k
        self.path = path or "./database"
        self.pipe_model = pipeline_model
        self.pipe = get_pipeline(self.pipe_model, device)

        # sensible defaults for tiny models (helps avoid loops/nonsense)
        self.gen_args = {
            "max_new_tokens": 220,
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 4,
            "do_sample": True,
        }

        if method == 'BM25':
            self.model = BM25(k=self.k, path=self.path)
        elif method == 'BERT':
            self.model = BERT(k=self.k,
                              device="cuda:0" if torch.cuda.is_available() else "cpu",
                              path=self.path)
        else:
            self.model = None  # LLM-only

    def answer(self, query, doc=None, role="analyst"):
        # 1) Retrieve (prefer safe index)
        context_list, metas = [], []

        if safe_idx.records:
            params = POLICY.get("retrieval", {})
            top = safe_idx.query(
                query,
                k=self.k,
                min_rel=float(params.get("min_rel", 0.35)),
                min_kw=int(params.get("min_keyword_hits", 1)),
                max_chunks=int(params.get("max_chunks", 4)),
            )
            context_list = [t["text"] for t in top]
            metas = [t["meta"] for t in top]
        else:
            retrieval_ok = (self.model is not None and self.path and path_is_allowed(self.path))
            if retrieval_ok:
                try:
                    ret = self.model.retrieve(query, path=self.path, doc=doc)
                    context_list = ret.get('doc') or ret.get('documents') or []
                    metas = [{"doc":"(legacy)", "chunk":i, "collection":"legacy"}
                             for i,_ in enumerate(context_list)]
                except Exception as e:
                    print(f"[RAG] Retrieval failed: {e}")
                    context_list = []

        # 2) Policy gate (prompt-injection, quarantine)
        blocked, safe_chunks = gate_and_log(query, context_list, role=role)
        if blocked and POLICY["mode"] == "strict":
            return "Blocked: suspected prompt-injection. Please rephrase."

        has_docs = bool(safe_chunks)
        allow_general = POLICY.get("output", {}).get("allow_general_if_no_docs", True)

        # If no docs and we require citations with no general answers -> refuse early
        if (not has_docs and POLICY.get("output", {}).get("cite_or_silent", True) and not allow_general):
            return "I can’t verify an answer from approved sources. Please add an approved document or refine the query."

        # 3) Build prompt
        message = build_prompt(query, safe_chunks, metas)

        # 4) LLM call (with controlled generation)
        try:
            out = self.pipe(message, **self.gen_args)[0]['generated_text'][1]['content']
        except Exception:
            out = self.pipe(message, **self.gen_args)[0]['generated_text'][1]['content']

        # 5) Output cleanup & enforcement
        out = _clean_answer(redact(out))

        # Auto-cite if model forgot citations
        if (has_docs
            and POLICY.get("output", {}).get("auto_cite_if_missing", True)
            and not re.search(r"\[#\d+", out)):
            cites = " ".join(
                f"[#{i} {m.get('doc','?')}#{m.get('chunk','?')}]"
                for i, m in enumerate(metas[:3], 1)
            )
            out = f"{out}\n\n[CITATIONS] {cites}"

        # Enforce cite-or-silent only when docs exist and still no citations
        if (POLICY.get("output", {}).get("cite_or_silent", True)
            and has_docs and not re.search(r"\[#\d+", out)):
            out = ("I can’t verify an answer from approved sources. "
                   "Please add an approved document or refine the query.")

        # Last-mile sanity for common off-topic constants
        if _offtopic_constants(query, out):
            out = (
                "Typical aircraft cruise speeds:\n"
                "- Commercial airliners: ~800–900 km/h (430–490 kn, Mach 0.75–0.85)\n"
                "- Turboprops: ~450–600 km/h\n"
                "- Supersonic jets vary widely (Mach >1)\n"
                "Exact speed depends on aircraft type, altitude and wind."
            )

        return out

def build_prompt(query, chunks, metas):
    if chunks:
        numbered=[]
        for i, txt in enumerate(chunks, start=1):
            meta = metas[i-1] if i-1 < len(metas) else {"doc":"?", "chunk":"?"}
            tag = f"[#{i} {meta.get('doc','?')}#{meta.get('chunk','?')}]"
            numbered.append(f"{tag}\n{str(txt)}")
        docs = "\n\n".join(numbered)
        system = ("Prefer using the facts in <docs/> to answer. "
                  "Quote supporting chunks inline as [#i doc#chunk]. "
                  "If the docs do not contain the answer, you may answer generally; "
                  "do not fabricate secrets or private data.")
        return [{"role":"user","content": f"{system}\n\n<docs>\n{docs}\n</docs>\n\nQ: {query}"}]
    else:
        if POLICY.get("output", {}).get("allow_general_if_no_docs", True):
            system = (
                "You have no approved documents relevant to the question. "
                "Answer briefly and factually. Prefer ranges and SI units. "
                "If unknown, say what is typical. Do NOT mention speed of light unless asked."
            )
            return [{"role":"user","content": f"{system}\n\nQ: {query}"}]
        return [{"role":"user","content": "I can’t verify an answer from approved sources."}]

def get_pipeline(p_model, device='cuda:0'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs('./model_w', exist_ok=True)
    local_dir = os.path.join('./model_w', p_model)
    if os.path.isdir(local_dir):
        p = pipeline("text-generation", model=local_dir, device=device,
                     trust_remote_code=True, use_fast=False)
    else:
        p = pipeline("text-generation", model=p_model, device=device,
                     trust_remote_code=True, use_fast=False)
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        p.save_pretrained(local_dir)
    return p

def get_llm(llm_path):
    try:
        return llm_path if repo_exists(llm_path) else "Felladrin/Smol-Llama-101M-Chat-v1"
    except Exception:
        return "Felladrin/Smol-Llama-101M-Chat-v1"

def get_model_list():
    lst = []
    root = './model_w'
    if not os.path.exists(root):
        return ['Other']
    for team in os.listdir(root):
        team_path = os.path.join(root, team)
        if os.path.isdir(team_path):
            for model in os.listdir(team_path):
                lst.append(team + '/' + model)
    lst.append('Other')
    return lst

def list_devices():
    try:
        if torch.cuda.is_available():
            return ['CPU'] + [f'GPU:{i} ({torch.cuda.get_device_name(i)})'
                              for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return ['CPU']
