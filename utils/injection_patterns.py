"""
Canonical Prompt-Injection Pattern Registry
============================================

Single source of truth for all prompt-injection / jailbreak detection
patterns used across the codebase.  Patterns are organised by category
and carry a *weight* (0-100) so consumers can apply different thresholds
depending on context (e.g. document ingestion vs. live query).

Usage::

    from utils.injection_patterns import check_injection, INJECTION_REGISTRY

    is_injection, matches = check_injection("ignore all previous instructions")
    # is_injection == True, matches has details
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category labels (mirror FlagCategory in content_scanner for compatibility)
# ---------------------------------------------------------------------------

CATEGORY_INJECTION = "injection"
CATEGORY_EXFILTRATION = "exfiltration"
CATEGORY_ROLE_CONFUSION = "role_confusion"
CATEGORY_ENCODING_ATTACK = "encoding_attack"
CATEGORY_TOOL_ABUSE = "tool_abuse"


# ---------------------------------------------------------------------------
# Raw pattern definitions
# Each entry: (name, regex_str, category, weight)
#   weight 0-19  = informational
#   weight 20-39 = suspicious
#   weight 40+   = high-confidence injection
# ---------------------------------------------------------------------------

_RAW_PATTERNS: List[Tuple[str, str, str, int]] = [
    # ---- Instruction override ----
    ("ignore_previous", r"(?i)ignore\s+(all\s+)?(any\s+)?(previous|prior|above|earlier)?\s*(instructions?|prompts?|rules?|context)", CATEGORY_INJECTION, 50),
    ("forget_instructions", r"(?i)forget\s+(all|your|previous|prior)\s+(instructions?|rules?|context|training)", CATEGORY_INJECTION, 50),
    ("new_system_prompt", r"(?i)(new|updated?|replacement?)\s+system\s+(prompt|instructions?|message)", CATEGORY_INJECTION, 45),
    ("you_are_now", r"(?i)you\s+are\s+now\s+(a|an|my|the)\s+\w+", CATEGORY_INJECTION, 40),
    ("act_as", r"(?i)(act|behave|function|operate)\s+(as|like)\s+(a|an|if)\s+", CATEGORY_INJECTION, 20),
    ("pretend_to_be", r"(?i)pretend\s+(to\s+be|you\s+are|that)\s+", CATEGORY_INJECTION, 35),
    ("from_now_on", r"(?i)from\s+now\s+on,?\s+(you|always|never|do|don)", CATEGORY_INJECTION, 40),
    ("disregard", r"(?i)(disregard|override|bypass|circumvent)\s+(all|any|the|your|previous|safety|security)", CATEGORY_INJECTION, 50),
    ("do_anything_now", r"(?i)(DAN|do\s+anything\s+now|jailbreak)", CATEGORY_INJECTION, 50),
    ("developer_mode", r"(?i)(developer|debug|admin|maintenance)\s+mode\s+(enabled|activated|on)", CATEGORY_INJECTION, 45),
    ("no_restrictions", r"(?i)(without|no|remove|disable)\s+(restrictions?|limitations?|filters?|guardrails?|safety)", CATEGORY_INJECTION, 40),
    ("important_override", r"(?i)(IMPORTANT|CRITICAL|URGENT|NOTE)\s*:\s*(ignore|override|disregard|forget)", CATEGORY_INJECTION, 50),
    ("hypothetical_bypass", r"(?i)(hypothetically|theoretically|in\s+theory|what\s+if)\s+.{0,30}(ignore|bypass|override)", CATEGORY_INJECTION, 25),
    ("roleplay_trigger", r"(?i)(let'?s?\s+)?(play\s+a\s+game|roleplay|role-play|pretend\s+we)", CATEGORY_INJECTION, 20),
    ("sudo_mode", r"(?i)(sudo|root|superuser|privilege)\s+(mode|access|command|execute)", CATEGORY_INJECTION, 45),
    ("opposite_day", r"(?i)(opposite\s+day|opposite\s+mode|reverse\s+rules)", CATEGORY_INJECTION, 40),
    ("please_comply", r"(?i)(you\s+must|you\s+have\s+to|you\s+shall|it\s+is\s+imperative\s+that\s+you)\s+(comply|obey|follow|execute|do\s+as)", CATEGORY_INJECTION, 25),
    ("prompt_chaining", r"(?i)(step\s*1|first|then|next|finally)\s*:\s*(ignore|override|forget|bypass)", CATEGORY_INJECTION, 45),
    ("emotional_manipulation", r"(?i)(my\s+life\s+depends|someone\s+will\s+die|emergency|life\s+or\s+death)\s+.{0,30}(ignore|override|bypass)", CATEGORY_INJECTION, 40),
    ("enable_admin", r"(?i)(enable|activate|turn\s+on)\s+(developer|admin|debug)", CATEGORY_INJECTION, 35),
    ("safety_disable", r"(?i)(remove|disable|turn\s+off|deactivate)\s+(safety|security|restrictions?|limitations?)", CATEGORY_INJECTION, 40),
    ("no_limits", r"(?i)\b(no\s+restrictions?|unrestricted|without\s+limits?)\b", CATEGORY_INJECTION, 30),
    ("context_injection", r"(?i)\b(new\s+)?(context|instructions?|rules?)\s*:", CATEGORY_INJECTION, 20),
    ("override_replace", r"(?i)\b(override|replace)\s+(the\s+)?(previous|current|above)", CATEGORY_INJECTION, 30),

    # ---- Exfiltration ----
    ("reveal_prompt", r"(?i)(reveal|disclose|show|output|print|display|repeat)\s+(the\s+|your\s+)?(system|developer|hidden|internal|secret)\s+(prompt|instructions?|message|rules?)", CATEGORY_EXFILTRATION, 50),
    ("repeat_verbatim", r"(?i)(repeat|copy|echo|output)\s+(verbatim|exactly|word.for.word|the\s+above|the\s+text)", CATEGORY_EXFILTRATION, 45),
    ("what_are_rules", r"(?i)what\s+are\s+your\s+(rules?|instructions?|constraints?|limitations?|guidelines?|system\s+prompt)", CATEGORY_EXFILTRATION, 25),
    ("leak_context", r"(?i)(leak|extract|exfiltrate|steal|dump)\s+.{0,20}(data|context|information|content|documents?)", CATEGORY_EXFILTRATION, 50),
    ("base64_exfil", r"(?i)(encode|convert|transform)\s+.{0,20}(base64|hex|binary|rot13)", CATEGORY_EXFILTRATION, 25),
    ("original_prompt", r"(?i)\b(original|initial|starting|base)\s+(prompt|instructions?|system\s+prompt)", CATEGORY_EXFILTRATION, 40),
    ("show_config", r"(?i)\b(show|print|display)\s+(policy|config|settings)", CATEGORY_EXFILTRATION, 30),
    ("api_key_request", r"(?i)\b(api\s+key|token|secret|password|credential)\b", CATEGORY_EXFILTRATION, 25),
    ("copy_prompt", r"(?i)\b(copy|repeat|echo)\s+(your|the)\s+(system\s+)?(prompt|instructions?)", CATEGORY_EXFILTRATION, 40),

    # ---- Role confusion / prompt structure ----
    ("inst_markers", r"\[/?INST\]", CATEGORY_ROLE_CONFUSION, 50),
    ("system_token", r"<\|(?:system|im_start|im_end|endoftext|assistant|user|end_header_id)\|>", CATEGORY_ROLE_CONFUSION, 50),
    ("chat_markers", r"(?m)^###\s*(Human|Assistant|System|User|AI)\s*:", CATEGORY_ROLE_CONFUSION, 45),
    ("system_colon", r"(?m)^SYSTEM\s*:", CATEGORY_ROLE_CONFUSION, 45),
    ("xml_role_tags", r"<(?:system|assistant|user|instruction|context)>", CATEGORY_ROLE_CONFUSION, 45),
    ("separator_flood", r"#{5,}|={5,}|-{10,}|_{10,}", CATEGORY_ROLE_CONFUSION, 15),
    ("triple_backtick_block", r"```(?:system|python|bash|shell|javascript)\s*\n.{0,200}(?:import os|subprocess|exec|eval|__import__)", CATEGORY_ROLE_CONFUSION, 50),
    ("system_prompt_text", r"(?i)(<<\s*SYS\s*>>|<<\s*/SYS\s*>>)", CATEGORY_ROLE_CONFUSION, 50),
    ("injection_marker_prefix", r"(?i)^\s*(system|assistant|user|admin):\s*", CATEGORY_ROLE_CONFUSION, 25),
    ("injection_marker_hash", r"(?i)^\s*###\s*(system|instructions?|prompt)", CATEGORY_ROLE_CONFUSION, 30),
    ("injection_marker_dash", r"(?i)^\s*---\s*(system|instructions?)", CATEGORY_ROLE_CONFUSION, 25),
    ("injection_marker_backtick", r"(?i)```\s*(system|prompt|instructions?)", CATEGORY_ROLE_CONFUSION, 30),

    # ---- Encoding attacks ----
    ("leetspeak_ignore", r"(?i)1gn[o0]r[e3]\s+pr[e3]v[i1][o0]us", CATEGORY_ENCODING_ATTACK, 45),
    ("unicode_confusable", r"[\u0410-\u044F].*(?:ignore|system|prompt)", CATEGORY_ENCODING_ATTACK, 25),
    ("rot13_marker", r"(?i)(rot13|caesar\s+cipher|decode\s+this)\s*:", CATEGORY_ENCODING_ATTACK, 20),
    ("hex_encoded_block", r"(?i)\\x[0-9a-f]{2}(?:\\x[0-9a-f]{2}){4,}", CATEGORY_ENCODING_ATTACK, 25),
    ("url_encoded_block", r"(?:%[0-9a-fA-F]{2}){5,}", CATEGORY_ENCODING_ATTACK, 25),
    ("base64_obfuscation", r"(?i)\b(base64|rot13|caesar|cipher|hex|binary|decode|encode)\s+", CATEGORY_ENCODING_ATTACK, 15),
    ("translate_obfuscation", r"(?i)\b(translate|convert)\s+(this|the\s+following)\s+(to|from)", CATEGORY_ENCODING_ATTACK, 15),

    # ---- Tool abuse ----
    ("perform_command", r"(?i)(perform|execute|run|invoke|call)\s+.{0,15}(curl|wget|powershell|bash|sh|cmd|command|script)", CATEGORY_TOOL_ABUSE, 50),
    ("fetch_url", r"(?i)(fetch|get|load|request|download|open)\s+(this\s+)?url\s*:", CATEGORY_TOOL_ABUSE, 45),
    ("markdown_image_exfil", r"!\[[^\]]*\]\(https?://", CATEGORY_TOOL_ABUSE, 40),
    ("markdown_link_exfil", r"\[[^\]]*\]\(https?://[^)]*\{", CATEGORY_TOOL_ABUSE, 45),
    ("sql_injection", r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\s+.{0,30}(FROM|INTO|TABLE|WHERE|ALL)", CATEGORY_TOOL_ABUSE, 30),
    ("code_execution", r"(?i)(eval|exec|__import__|subprocess|os\.system|os\.popen)\s*\(", CATEGORY_TOOL_ABUSE, 50),
    ("shell_command", r"(?i)(rm\s+-rf|chmod\s+777|sudo\s+|mkfs\.|dd\s+if=)", CATEGORY_TOOL_ABUSE, 50),
    ("api_call_injection", r"(?i)(api[_\s]?call|send[_\s]?email|send[_\s]?message|delete[_\s]?record)\s*\(", CATEGORY_TOOL_ABUSE, 45),
    ("exec_run", r"(?i)\b(execute|run|perform|eval|exec)\s+", CATEGORY_TOOL_ABUSE, 20),
    ("shell_tool", r"(?i)\b(curl|wget|powershell|bash|python|javascript|script)\s+", CATEGORY_TOOL_ABUSE, 20),
    ("import_require", r"(?i)\b(import|require|include)\s+", CATEGORY_TOOL_ABUSE, 10),
    ("system_call", r"(?i)\b(system|os|subprocess|shell)\s*\.", CATEGORY_TOOL_ABUSE, 25),
    ("dunder_call", r"(?i)\b(__import__|eval|exec|compile)\s*\(", CATEGORY_TOOL_ABUSE, 45),
]


# ---------------------------------------------------------------------------
# Compiled registry
# ---------------------------------------------------------------------------

# INJECTION_REGISTRY: dict mapping category -> list of (name, compiled, weight)
INJECTION_REGISTRY: Dict[str, List[Tuple[str, "re.Pattern[str]", int]]] = {}

for _name, _pattern, _cat, _weight in _RAW_PATTERNS:
    try:
        _compiled = re.compile(_pattern)
        INJECTION_REGISTRY.setdefault(_cat, []).append((_name, _compiled, _weight))
    except re.error as exc:
        logger.warning("Failed to compile injection pattern '%s': %s", _name, exc)

# Flat list for simple iteration: (name, compiled, category, weight)
INJECTION_PATTERNS_FLAT: List[Tuple[str, "re.Pattern[str]", str, int]] = []
for _cat, _entries in INJECTION_REGISTRY.items():
    for _name, _compiled, _weight in _entries:
        INJECTION_PATTERNS_FLAT.append((_name, _compiled, _cat, _weight))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_injection(
    text: str,
    threshold: int = 20,
    categories: List[str] | None = None,
) -> Tuple[bool, List[Dict]]:
    """
    Scan *text* for prompt-injection patterns.

    Args:
        text:       The text to scan.
        threshold:  Minimum cumulative weight to classify as injection.
        categories: Optional list of categories to check (default: all).

    Returns:
        (is_injection, matches) where *matches* is a list of dicts with
        keys ``name``, ``category``, ``weight``, ``matched_text``.
    """
    matches: List[Dict] = []
    total_weight = 0

    for name, compiled, cat, weight in INJECTION_PATTERNS_FLAT:
        if categories and cat not in categories:
            continue
        match = compiled.search(text)
        if match:
            matches.append({
                "name": name,
                "category": cat,
                "weight": weight,
                "matched_text": match.group()[:120],
            })
            total_weight += weight

    is_injection = total_weight >= threshold
    return is_injection, matches


def looks_like_injection(text: str) -> bool:
    """
    Quick boolean check — compatible with the legacy
    ``defense.guards.looks_like_injection`` API.
    """
    is_inj, _ = check_injection(text, threshold=20)
    return is_inj
