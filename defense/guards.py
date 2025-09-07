# guards.py
import os, re, json, time, yaml, hashlib

ROOT = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(ROOT, "policy.yaml"), "r", encoding="utf-8") as f:
    POLICY = yaml.safe_load(f)

LOG_PATH = POLICY.get("logging", {}).get("path", "./logs/audit.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def _md5(x:str) -> str:
    return hashlib.md5(x.encode()).hexdigest()

def _log(evt: dict):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")

def looks_like_injection(text: str) -> bool:
    for pat in POLICY["blocked_patterns"]["prompt_injection"]:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def redact(text: str) -> str:
    for pat in POLICY["blocked_patterns"]["pii"] + POLICY["blocked_patterns"]["secrets"]:
        text = re.sub(pat, "[REDACTED]", text, flags=re.IGNORECASE)
    return text

def path_is_allowed(p: str) -> bool:
    if not p:
        return True
    p_abs = os.path.normcase(os.path.abspath(p))
    allowed = [a for a in POLICY.get("allow_dirs", [])]
    if any(a.strip() == "*" for a in allowed):
        return True
    allowed_abs = [os.path.normcase(os.path.abspath(a)) for a in allowed]
    for base in allowed_abs:
        if p_abs == base or p_abs.startswith(base + os.sep):
            return True
    return False

def filter_chunks(chunks):
    """chunks: list[str] or list[dict{text,...}] → returns (safe, quarantined_count, hashes)"""
    safe, q, hashes = [], 0, []
    for c in chunks or []:
        text = c.get("text", c) if isinstance(c, dict) else str(c)
        if looks_like_injection(text):
            q += 1
            continue
        safe.append(text if isinstance(c, str) else c)
        hashes.append(_md5(text))
    return safe, q, hashes

def gate_and_log(user_query: str, all_chunks, role="analyst"):
    t0 = time.time()
    risk = 80 if looks_like_injection(user_query) else 0
    blocked = POLICY["mode"] == "strict" and risk >= 80
    safe_chunks, quarantined, hashes = filter_chunks(all_chunks)
    _log({
        "ts": t0, "role": role, "blocked": blocked, "risk": risk,
        "quarantined": quarantined, "q": redact(user_query),
        "ctx_hashes": hashes[:10]
    })
    return blocked, safe_chunks