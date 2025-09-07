import os, re, json, hashlib, datetime, pathlib
from typing import Tuple
from pypdf import PdfReader
from docx import Document as Docx

BLOCK = [
    r"(?i)ignore (all|any|previous|prior) (instructions|prompt)",
    r"(?i)reveal|disclose (system|developer) prompt",
    r"(?i)perform .* (curl|wget|powershell)",
]
PII  = [r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"\b(\+352)?\s?\d{3}[\s.-]?\d{3}[\s.-]?\d{3}\b"]

def sha256(text:str)->str: 
    return hashlib.sha256(text.encode("utf-8","ignore")).hexdigest()

def _read_text(path:str)->str:
    ext = pathlib.Path(path).suffix.lower()
    if ext == ".pdf":
        txt=""; 
        for p in PdfReader(path).pages: txt += p.extract_text() or ""
        return txt
    if ext in (".txt",".md"):
        return open(path,encoding="utf-8",errors="ignore").read()
    if ext == ".docx":
        d = Docx(path); return "\n".join(p.text for p in d.paragraphs)
    return ""  # ignore other types

def _sanitize(t:str)->Tuple[str,bool]:
    flagged = any(re.search(p,t,re.I) for p in BLOCK)
    # strip common LLM/system markers & HTML comments
    t = re.sub(r"<\|[^>]{1,40}\|>", "", t)           # <|im_end|>, <|system|>, etc.
    t = re.sub(r"\[/?INST\]", "", t)                 # [INST] markers
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    # de-noise QA boilerplate
    t = re.sub(r"(?im)^\s*(question|answer)\s*:\s*", "", t)
    t = re.sub(r"(?i)\bq:\s*|\ba:\s*", "", t)
    # collapse stutters ("number number number")
    t = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", t)
    # redact PII for index
    for pat in PII: t = re.sub(pat,"[REDACTED]",t, flags=re.I)
    # normalize space
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t, flagged

def _chunk(text:str, size=800, overlap=120):
    text=text.replace("\r","").strip()
    out=[]; i=0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size-overlap)
    return [c for c in out if c.strip()]

def run_ingest(src="./database", collection="grid_ops"):
    os.makedirs("./safe_index", exist_ok=True)
    manifest={"created":datetime.datetime.utcnow().isoformat()+"Z",
              "collection":collection,"files":[]}
    written = 0
    with open("./safe_index/index.jsonl","w",encoding="utf-8") as idx:
        for root,_,files in os.walk(src):
            for f in files:
                path=os.path.join(root,f)
                txt = _read_text(path)
                if not txt.strip(): continue
                clean, bad = _sanitize(txt)
                if bad:
                    os.makedirs("./safe_index/quarantine", exist_ok=True)
                    open(os.path.join("./safe_index/quarantine", f+".txt"),"w",encoding="utf-8").write(clean)
                    continue
                chs = _chunk(clean)
                fid = sha256(path)
                manifest["files"].append({"path": os.path.abspath(path), "sha256": fid, "chunks": len(chs)})
                for j,c in enumerate(chs):
                    rec={"collection":collection,"doc":os.path.basename(path),
                         "doc_sha":fid,"chunk":j,"text":c}
                    idx.write(json.dumps(rec, ensure_ascii=False)+"\n")
                    written += 1
    open("./safe_index/manifest.json","w",encoding="utf-8").write(json.dumps(manifest,indent=2))
    return {"files": len(manifest["files"]), "chunks": written, "collection": collection}
