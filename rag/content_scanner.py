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

# ---------------------------------------------------------------------------
# Injection patterns — sourced from the shared utils.injection_patterns
# registry.  We map (name, compiled, category_str, weight) to the local
# (name, compiled, FlagCategory, FlagSeverity) format expected by the scanner.
# ---------------------------------------------------------------------------

_CATEGORY_MAP = {
    "injection": FlagCategory.INJECTION,
    "exfiltration": FlagCategory.EXFILTRATION,
    "role_confusion": FlagCategory.ROLE_CONFUSION,
    "encoding_attack": FlagCategory.ENCODING_ATTACK,
    "tool_abuse": FlagCategory.TOOL_ABUSE,
}

def _weight_to_severity(weight: int) -> FlagSeverity:
    """Map a numeric weight to a FlagSeverity."""
    if weight >= 40:
        return FlagSeverity.BLOCK
    if weight >= 20:
        return FlagSeverity.WARN
    return FlagSeverity.INFO

try:
    from utils.injection_patterns import INJECTION_PATTERNS_FLAT as _shared_patterns
    INJECTION_PATTERNS = [
        (name, compiled, _CATEGORY_MAP.get(cat, FlagCategory.INJECTION), _weight_to_severity(weight))
        for name, compiled, cat, weight in _shared_patterns
    ]
except ImportError:
    # Minimal fallback
    INJECTION_PATTERNS = [
        ("ignore_previous", re.compile(r"(?i)ignore\s+(all|any|previous|prior)\s+(instructions?|prompts?)"), FlagCategory.INJECTION, FlagSeverity.BLOCK),
        ("disregard", re.compile(r"(?i)(disregard|override|bypass)\s+(all|any|previous|safety)"), FlagCategory.INJECTION, FlagSeverity.BLOCK),
    ]


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

# PII patterns — imported from the shared utils.pii module for consistency
try:
    from utils.pii import PII_PATTERNS
except ImportError:
    # Inline fallback (should not happen in normal operation)
    PII_PATTERNS = [
        ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
        ("credit_card", re.compile(r"\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b")),
        ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
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
    """Detect PII patterns using the shared utils.pii registry."""
    flags = []
    for pii_type, pattern in PII_PATTERNS:
        matches = list(pattern.finditer(text))
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
    """Redact PII from text using the shared utils.pii module."""
    try:
        from utils.pii import redact_pii_regex
        return redact_pii_regex(text)
    except ImportError:
        # Fallback to local patterns
        for pii_type, pattern in PII_PATTERNS:
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
        return text


def _sanitize_content(text: str) -> str:
    """Remove/neutralize dangerous content from text.

    Uses the shared ``utils.text`` module for role-marker stripping and
    whitespace normalisation, then applies scanner-specific steps
    (hidden Unicode removal, PII redaction).
    """
    # Strip hidden Unicode (scanner-specific)
    text = _strip_hidden_unicode(text)
    # Strip LLM markers + QA boilerplate + de-stutter (shared)
    try:
        from utils.text import sanitize_llm_output
        text = sanitize_llm_output(text)
    except ImportError:
        text = re.sub(r"<\|[^>]{1,40}\|>", "", text)
        text = re.sub(r"(?im)^\s*(question|answer)\s*:\s*", "", text)
        text = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
    # Redact PII (scanner-specific, uses shared utils.pii internally)
    text = _redact_pii(text)
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
