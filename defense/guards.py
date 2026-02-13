"""
Policy Enforcement Guards
=========================

Provides query gating, PII redaction, and prompt-injection detection
for the RAG pipeline. Loads policy configuration from ``policy.yaml``.

Functions:
    gate_and_log    -- Evaluate a user query + retrieved chunks against policy.
    redact          -- Redact PII from text (delegates to utils.pii).
    looks_like_injection -- Check for prompt injection patterns.
    filter_chunks   -- Filter retrieved chunks for injections.
    reload_policy   -- Hot-reload policy.yaml without restart.
"""

import os, re, json, time, yaml, hashlib

ROOT = os.path.dirname(os.path.dirname(__file__))

def _load_policy():
    """Load policy from YAML file"""
    with open(os.path.join(ROOT, "policy.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def reload_policy():
    """Reload POLICY from disk - call this after saving policy.yaml"""
    global POLICY, LOG_PATH
    POLICY = _load_policy()
    LOG_PATH = POLICY.get("logging", {}).get("path", "./logs/audit.jsonl")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    return POLICY

def update_cite_or_silent(value: bool):
    """Update cite_or_silent in memory (without file I/O)"""
    global POLICY
    POLICY.setdefault("output", {})["cite_or_silent"] = value

# Initial load
POLICY = _load_policy()
LOG_PATH = POLICY.get("logging", {}).get("path", "./logs/audit.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def _md5(x: str) -> str:
    """MD5 hash — delegates to shared utils.text."""
    try:
        from utils.text import md5_hash
        return md5_hash(x)
    except ImportError:
        return hashlib.md5(x.encode()).hexdigest()

def _log(evt: dict):
    """Append event to audit JSONL log — delegates to shared logger."""
    try:
        from utils.audit_logger import log_audit
        log_audit(evt, log_path=LOG_PATH)
    except ImportError:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")

def looks_like_injection(text: str) -> bool:
    """Check for prompt injection using the shared pattern registry."""
    try:
        from utils.injection_patterns import looks_like_injection as _shared_check
        return _shared_check(text)
    except ImportError:
        # Fallback to policy-based patterns
        for pat in POLICY["blocked_patterns"]["prompt_injection"]:
            if re.search(pat, text, flags=re.IGNORECASE):
                return True
        return False

def redact(text: str) -> str:
    """Redact PII from text using the shared unified PII module."""
    try:
        from utils.pii import redact_pii
        return redact_pii(text)
    except ImportError:
        # Fallback to policy-based patterns if utils not available
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
    """Log the query and filter chunks for injections. Never blocks (use guardrails for that)."""
    t0 = time.time()
    risk = 80 if looks_like_injection(user_query) else 0
    safe_chunks, quarantined, hashes = filter_chunks(all_chunks)
    _log({
        "ts": t0, "role": role, "blocked": False, "risk": risk,
        "quarantined": quarantined, "q": redact(user_query),
        "ctx_hashes": hashes[:10]
    })
    return False, safe_chunks