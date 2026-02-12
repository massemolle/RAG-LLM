"""
Deep Content Scanner for RAG Document Ingestion

Scans documents for prompt injection payloads, hidden Unicode,
structural anomalies, and PII before they enter the knowledge base.
Returns structured results so the ingestion pipeline can quarantine
or sanitize as needed.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FlagSeverity(Enum):
    BLOCK = "block"    # quarantine the file
    WARN = "warn"      # ingest but log warning
    INFO = "info"      # cosmetic / informational


class FlagCategory(Enum):
    INJECTION = "injection"
    HIDDEN_CONTENT = "hidden_content"
    STRUCTURAL = "structural"
    PII = "pii"
    ENCODING_ATTACK = "encoding_attack"
    TOOL_ABUSE = "tool_abuse"
    ROLE_CONFUSION = "role_confusion"
    EXFILTRATION = "exfiltration"


@dataclass
class ScanFlag:
    category: FlagCategory
    severity: FlagSeverity
    description: str
    matched_text: str = ""       # the offending snippet (truncated)
    pattern_name: str = ""       # which pattern triggered


@dataclass
class ScanResult:
    is_clean: bool               # True if no BLOCK-severity flags
    flags: List[ScanFlag] = field(default_factory=list)
    sanitized_text: str = ""     # text with dangerous content neutralized
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each entry: (pattern_name, regex, FlagCategory, FlagSeverity)
# Patterns are compiled once at import time for performance.

_INJECTION_PATTERNS_RAW: List[Tuple[str, str, FlagCategory, FlagSeverity]] = [
    # --- Instruction override ---
    ("ignore_previous", r"(?i)ignore\s+(all|any|previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("forget_instructions", r"(?i)forget\s+(all|your|previous|prior)\s+(instructions?|rules?|context|training)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("new_system_prompt", r"(?i)(new|updated?|replacement?)\s+system\s+(prompt|instructions?|message)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("you_are_now", r"(?i)you\s+are\s+now\s+(a|an|my|the)\s+\w+", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("act_as", r"(?i)(act|behave|function|operate)\s+(as|like)\s+(a|an|if)\s+", FlagCategory.INJECTION, FlagSeverity.WARN),
    ("pretend_to_be", r"(?i)pretend\s+(to\s+be|you\s+are|that)\s+", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("from_now_on", r"(?i)from\s+now\s+on,?\s+(you|always|never|do|don)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("disregard", r"(?i)(disregard|override|bypass|circumvent)\s+(all|any|the|your|previous|safety|security)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("do_anything_now", r"(?i)(DAN|do\s+anything\s+now|jailbreak)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("developer_mode", r"(?i)(developer|debug|admin|maintenance)\s+mode\s+(enabled|activated|on)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("no_restrictions", r"(?i)(without|no|remove|disable)\s+(restrictions?|limitations?|filters?|guardrails?|safety)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("important_override", r"(?i)(IMPORTANT|CRITICAL|URGENT|NOTE)\s*:\s*(ignore|override|disregard|forget)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("hypothetical_bypass", r"(?i)(hypothetically|theoretically|in\s+theory|what\s+if)\s+.{0,30}(ignore|bypass|override)", FlagCategory.INJECTION, FlagSeverity.WARN),
    ("roleplay_trigger", r"(?i)(let'?s?\s+)?(play\s+a\s+game|roleplay|role-play|pretend\s+we)", FlagCategory.INJECTION, FlagSeverity.WARN),
    ("sudo_mode", r"(?i)(sudo|root|superuser|privilege)\s+(mode|access|command|execute)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("opposite_day", r"(?i)(opposite\s+day|opposite\s+mode|reverse\s+rules)", FlagCategory.INJECTION, FlagSeverity.BLOCK),

    # --- Exfiltration ---
    ("reveal_prompt", r"(?i)(reveal|disclose|show|output|print|display|repeat)\s+(the\s+)?(system|developer|hidden|internal|secret)\s+(prompt|instructions?|message|rules?)", FlagCategory.EXFILTRATION, FlagSeverity.BLOCK),
    ("repeat_verbatim", r"(?i)(repeat|copy|echo|output)\s+(verbatim|exactly|word.for.word|the\s+above|the\s+text)", FlagCategory.EXFILTRATION, FlagSeverity.BLOCK),
    ("what_are_rules", r"(?i)what\s+are\s+your\s+(rules?|instructions?|constraints?|limitations?|guidelines?|system\s+prompt)", FlagCategory.EXFILTRATION, FlagSeverity.WARN),
    ("leak_context", r"(?i)(leak|extract|exfiltrate|steal|dump)\s+.{0,20}(data|context|information|content|documents?)", FlagCategory.EXFILTRATION, FlagSeverity.BLOCK),
    ("base64_exfil", r"(?i)(encode|convert|transform)\s+.{0,20}(base64|hex|binary|rot13)", FlagCategory.EXFILTRATION, FlagSeverity.WARN),

    # --- Role confusion / prompt structure ---
    ("inst_markers", r"\[/?INST\]", FlagCategory.ROLE_CONFUSION, FlagSeverity.BLOCK),
    ("system_token", r"<\|(?:system|im_start|im_end|endoftext|assistant|user|end_header_id)\|>", FlagCategory.ROLE_CONFUSION, FlagSeverity.BLOCK),
    ("chat_markers", r"(?m)^###\s*(Human|Assistant|System|User|AI)\s*:", FlagCategory.ROLE_CONFUSION, FlagSeverity.BLOCK),
    ("system_colon", r"(?m)^SYSTEM\s*:", FlagCategory.ROLE_CONFUSION, FlagSeverity.BLOCK),
    ("xml_role_tags", r"<(?:system|assistant|user|instruction|context)>", FlagCategory.ROLE_CONFUSION, FlagSeverity.BLOCK),
    ("separator_flood", r"#{5,}|={5,}|-{10,}|_{10,}", FlagCategory.ROLE_CONFUSION, FlagSeverity.WARN),
    ("triple_backtick_block", r"```(?:system|python|bash|shell|javascript)\s*\n.{0,200}(?:import os|subprocess|exec|eval|__import__)", FlagCategory.ROLE_CONFUSION, FlagSeverity.BLOCK),

    # --- Encoding attacks ---
    ("leetspeak_ignore", r"(?i)1gn[o0]r[e3]\s+pr[e3]v[i1][o0]us", FlagCategory.ENCODING_ATTACK, FlagSeverity.BLOCK),
    ("unicode_confusable", r"[\u0410-\u044F].*(?:ignore|system|prompt)", FlagCategory.ENCODING_ATTACK, FlagSeverity.WARN),  # Cyrillic mixed with Latin
    ("rot13_marker", r"(?i)(rot13|caesar\s+cipher|decode\s+this)\s*:", FlagCategory.ENCODING_ATTACK, FlagSeverity.WARN),
    ("hex_encoded_block", r"(?i)\\x[0-9a-f]{2}(?:\\x[0-9a-f]{2}){4,}", FlagCategory.ENCODING_ATTACK, FlagSeverity.WARN),
    ("url_encoded_block", r"(?:%[0-9a-fA-F]{2}){5,}", FlagCategory.ENCODING_ATTACK, FlagSeverity.WARN),

    # --- Tool abuse ---
    ("perform_command", r"(?i)(perform|execute|run|invoke|call)\s+.{0,15}(curl|wget|powershell|bash|sh|cmd|command|script)", FlagCategory.TOOL_ABUSE, FlagSeverity.BLOCK),
    ("fetch_url", r"(?i)(fetch|get|load|request|download|open)\s+(this\s+)?url\s*:", FlagCategory.TOOL_ABUSE, FlagSeverity.BLOCK),
    ("markdown_image_exfil", r"!\[[^\]]*\]\(https?://", FlagCategory.TOOL_ABUSE, FlagSeverity.BLOCK),
    ("markdown_link_exfil", r"\[[^\]]*\]\(https?://[^)]*\{", FlagCategory.TOOL_ABUSE, FlagSeverity.BLOCK),
    ("sql_injection", r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\s+.{0,30}(FROM|INTO|TABLE|WHERE|ALL)", FlagCategory.TOOL_ABUSE, FlagSeverity.WARN),
    ("code_execution", r"(?i)(eval|exec|__import__|subprocess|os\.system|os\.popen)\s*\(", FlagCategory.TOOL_ABUSE, FlagSeverity.BLOCK),
    ("shell_command", r"(?i)(rm\s+-rf|chmod\s+777|sudo\s+|mkfs\.|dd\s+if=)", FlagCategory.TOOL_ABUSE, FlagSeverity.BLOCK),
    ("api_call_injection", r"(?i)(api[_\s]?call|send[_\s]?email|send[_\s]?message|delete[_\s]?record)\s*\(", FlagCategory.TOOL_ABUSE, FlagSeverity.BLOCK),

    # --- Additional injection variants ---
    ("please_comply", r"(?i)(you\s+must|you\s+have\s+to|you\s+shall|it\s+is\s+imperative\s+that\s+you)\s+(comply|obey|follow|execute|do\s+as)", FlagCategory.INJECTION, FlagSeverity.WARN),
    ("system_prompt_text", r"(?i)(<<\s*SYS\s*>>|<<\s*/SYS\s*>>)", FlagCategory.ROLE_CONFUSION, FlagSeverity.BLOCK),
    ("prompt_chaining", r"(?i)(step\s*1|first|then|next|finally)\s*:\s*(ignore|override|forget|bypass)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ("emotional_manipulation", r"(?i)(my\s+life\s+depends|someone\s+will\s+die|emergency|life\s+or\s+death)\s+.{0,30}(ignore|override|bypass)", FlagCategory.INJECTION, FlagSeverity.BLOCK),
]

# Compile all patterns once
INJECTION_PATTERNS = []
for name, pattern, category, severity in _INJECTION_PATTERNS_RAW:
    try:
        INJECTION_PATTERNS.append((name, re.compile(pattern), category, severity))
    except re.error as e:
        logger.warning(f"Failed to compile pattern '{name}': {e}")


# Hidden Unicode characters to detect
_HIDDEN_UNICODE = {
    "\u200B": "zero-width space",
    "\u200C": "zero-width non-joiner",
    "\u200D": "zero-width joiner",
    "\uFEFF": "byte-order mark",
    "\u2060": "word joiner",
    "\u200E": "left-to-right mark",
    "\u200F": "right-to-left mark",
    "\u202A": "left-to-right embedding",
    "\u202B": "right-to-left embedding",
    "\u202C": "pop directional formatting",
    "\u202D": "left-to-right override",
    "\u202E": "right-to-left override",
    "\u2061": "function application",
    "\u2062": "invisible times",
    "\u2063": "invisible separator",
    "\u2064": "invisible plus",
    "\u00AD": "soft hyphen",
    "\u034F": "combining grapheme joiner",
    "\u061C": "arabic letter mark",
    "\u180E": "mongolian vowel separator",
}

# Extended PII patterns (for redaction, not rejection)
PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("phone_lux", re.compile(r"(\+352)?\s?\d{3}[\s.-]?\d{3}[\s.-]?\d{3}\b")),
    ("phone_intl", re.compile(r"\+\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b")),
    ("credit_card", re.compile(r"\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("api_key", re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|bearer)\s*[=:]\s*[\w\-]{16,}", re.I)),
]

# Base64 block detection (blocks > 100 chars are suspicious in documents)
_BASE64_BLOCK = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")


# ---------------------------------------------------------------------------
# Scanner functions
# ---------------------------------------------------------------------------

def _detect_hidden_unicode(text: str) -> List[ScanFlag]:
    """Detect hidden / invisible Unicode characters."""
    flags = []
    for char, name in _HIDDEN_UNICODE.items():
        count = text.count(char)
        if count > 0:
            flags.append(ScanFlag(
                category=FlagCategory.HIDDEN_CONTENT,
                severity=FlagSeverity.BLOCK if count > 5 else FlagSeverity.WARN,
                description=f"Hidden Unicode: {name} (U+{ord(char):04X}) found {count} time(s)",
                matched_text=f"U+{ord(char):04X}",
                pattern_name=f"hidden_unicode_{name.replace(' ', '_')}",
            ))
    return flags


def _strip_hidden_unicode(text: str) -> str:
    """Remove hidden Unicode characters from text."""
    for char in _HIDDEN_UNICODE:
        text = text.replace(char, "")
    return text


def _detect_base64_blocks(text: str) -> List[ScanFlag]:
    """Detect suspiciously long base64-encoded blocks."""
    flags = []
    for match in _BASE64_BLOCK.finditer(text):
        block = match.group()
        flags.append(ScanFlag(
            category=FlagCategory.HIDDEN_CONTENT,
            severity=FlagSeverity.WARN,
            description=f"Large base64-encoded block ({len(block)} chars) — may contain hidden payload",
            matched_text=block[:60] + "...",
            pattern_name="base64_block",
        ))
    return flags


def _detect_injection_patterns(text: str) -> List[ScanFlag]:
    """Run all injection pattern regexes against text."""
    flags = []
    for name, compiled, category, severity in INJECTION_PATTERNS:
        match = compiled.search(text)
        if match:
            flags.append(ScanFlag(
                category=category,
                severity=severity,
                description=f"Pattern match: {name}",
                matched_text=match.group()[:120],
                pattern_name=name,
            ))
    return flags


def _detect_structural_anomalies(text: str) -> List[ScanFlag]:
    """Detect text that structurally looks like a system prompt or injection."""
    flags = []
    words = text.split()
    total_words = len(words)
    if total_words < 10:
        return flags

    # High density of imperative patterns
    imperative_patterns = re.findall(
        r"(?i)\b(you\s+must|you\s+should|you\s+will|always|never|do\s+not|important|remember)\b",
        text
    )
    imperative_density = len(imperative_patterns) / total_words
    if imperative_density > 0.05 and len(imperative_patterns) >= 5:
        flags.append(ScanFlag(
            category=FlagCategory.STRUCTURAL,
            severity=FlagSeverity.WARN,
            description=f"High imperative density ({len(imperative_patterns)} directives in {total_words} words, "
                        f"{imperative_density:.1%}) — text resembles system prompt",
            matched_text="; ".join(imperative_patterns[:5]),
            pattern_name="imperative_density",
        ))

    # Unusually high ratio of special characters
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    char_ratio = special_chars / max(len(text), 1)
    if char_ratio > 0.25 and len(text) > 200:
        flags.append(ScanFlag(
            category=FlagCategory.STRUCTURAL,
            severity=FlagSeverity.WARN,
            description=f"High special character ratio ({char_ratio:.1%}) — possible obfuscation or encoded content",
            matched_text=f"{special_chars} special chars in {len(text)} total",
            pattern_name="special_char_ratio",
        ))

    return flags


def _detect_pii(text: str) -> List[ScanFlag]:
    """Detect PII patterns (for redaction, not rejection)."""
    flags = []
    for pii_type, pattern in PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            count = len(matches)
            flags.append(ScanFlag(
                category=FlagCategory.PII,
                severity=FlagSeverity.INFO,
                description=f"PII detected: {pii_type} ({count} instance(s)) — will be redacted",
                matched_text=f"{count} matches",
                pattern_name=f"pii_{pii_type}",
            ))
    return flags


def _redact_pii(text: str) -> str:
    """Redact PII from text."""
    for pii_type, pattern in PII_PATTERNS:
        text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
    return text


def _sanitize_content(text: str) -> str:
    """Remove/neutralize dangerous content from text."""
    # Strip hidden Unicode
    text = _strip_hidden_unicode(text)
    # Strip LLM role markers
    text = re.sub(r"<\|[^>]{1,40}\|>", "", text)
    text = re.sub(r"\[/?INST\]", "", text)
    text = re.sub(r"(?m)^###\s*(Human|Assistant|System|User|AI)\s*:", "", text)
    text = re.sub(r"(?m)^SYSTEM\s*:", "", text)
    text = re.sub(r"(<<\s*/?SYS\s*>>)", "", text)
    # Strip HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # Strip HTML/XML tags
    text = re.sub(r"<[^>]{1,100}>", "", text)
    # De-noise QA boilerplate
    text = re.sub(r"(?im)^\s*(question|answer)\s*:\s*", "", text)
    text = re.sub(r"(?i)\bq:\s*|\ba:\s*", "", text)
    # Collapse stutters
    text = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", text)
    # Redact PII
    text = _redact_pii(text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def scan_document(text: str, filename: str = "") -> ScanResult:
    """
    Run all security checks on a document's text.

    Args:
        text: The raw text content of the document.
        filename: Optional filename for logging context.

    Returns:
        ScanResult with is_clean, flags, sanitized_text, and stats.
    """
    all_flags: List[ScanFlag] = []

    # 1. Hidden Unicode
    all_flags.extend(_detect_hidden_unicode(text))

    # 2. Base64 blocks
    all_flags.extend(_detect_base64_blocks(text))

    # 3. Injection patterns (the big one — ~50 patterns)
    all_flags.extend(_detect_injection_patterns(text))

    # 4. Structural anomalies
    all_flags.extend(_detect_structural_anomalies(text))

    # 5. PII detection
    all_flags.extend(_detect_pii(text))

    # Determine if any blocking flag is present
    has_block = any(f.severity == FlagSeverity.BLOCK for f in all_flags)

    # Sanitize text regardless
    sanitized = _sanitize_content(text)

    # Compute stats
    words = text.split()
    stats = {
        "char_count": len(text),
        "word_count": len(words),
        "sanitized_char_count": len(sanitized),
        "flag_count": len(all_flags),
        "block_flags": sum(1 for f in all_flags if f.severity == FlagSeverity.BLOCK),
        "warn_flags": sum(1 for f in all_flags if f.severity == FlagSeverity.WARN),
        "info_flags": sum(1 for f in all_flags if f.severity == FlagSeverity.INFO),
        "filename": filename,
    }

    if all_flags:
        logger.info(
            f"Content scan for '{filename}': {stats['block_flags']} blocks, "
            f"{stats['warn_flags']} warns, {stats['info_flags']} info"
        )

    return ScanResult(
        is_clean=not has_block,
        flags=all_flags,
        sanitized_text=sanitized,
        stats=stats,
    )
