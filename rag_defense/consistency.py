"""
Cross-Chunk Consistency Checker

Detects potential contradictions between retrieved chunks using
keyword-based negation heuristics. Flags are informational warnings
only — they never block.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyFlag:
    chunk_a_idx: int
    chunk_b_idx: int
    description: str
    snippet_a: str
    snippet_b: str
    severity: str = "warning"  # always warning, never block


# Negation words that flip the meaning of an assertion
_NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "cannot", "can't", "won't", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "shouldn't", "wouldn't", "couldn't"}

# Strong assertion verbs
_ASSERTION_PATTERN = re.compile(
    r"(?i)\b(\w+(?:\s+\w+)?)\s+(is|are|was|were|has|have|had|will|should|must|can)\s+(not\s+)?(.{3,40}?)(?:\.|,|;|$)"
)

# Numeric value pattern: "X is/are/= <number>"
_NUMERIC_PATTERN = re.compile(
    r"(?i)\b(\w+(?:\s+\w+){0,2})\s+(?:is|are|was|were|=|equals?)\s+(\d+[\d,.]*%?)"
)


def _extract_assertions(text: str) -> List[dict]:
    """
    Extract simple subject-predicate assertions from text.

    Returns list of dicts with: subject, predicate, negated (bool), span.
    """
    assertions = []
    for m in _ASSERTION_PATTERN.finditer(text):
        subject = m.group(1).strip().lower()
        verb = m.group(2).strip().lower()
        neg = m.group(3) is not None
        predicate = m.group(4).strip().lower()

        # Also check for negation word at start of predicate
        first_word = predicate.split()[0] if predicate.split() else ""
        if first_word in _NEGATION_WORDS:
            neg = True
            predicate = " ".join(predicate.split()[1:])

        if len(subject) < 2 or len(predicate) < 2:
            continue

        assertions.append({
            "subject": subject,
            "verb": verb,
            "predicate": predicate,
            "negated": neg,
            "span": m.group(0)[:80],
        })

    return assertions


def _extract_numerics(text: str) -> List[dict]:
    """
    Extract numeric assertions like "temperature is 25" or "cost = 100".
    """
    numerics = []
    for m in _NUMERIC_PATTERN.finditer(text):
        subject = m.group(1).strip().lower()
        value = m.group(2).strip()
        numerics.append({
            "subject": subject,
            "value": value,
            "span": m.group(0)[:80],
        })
    return numerics


def flag_inconsistencies(
    chunks: List[str],
    query: str = "",
) -> List[ConsistencyFlag]:
    """
    Check retrieved chunks for potential contradictions.

    Uses heuristic assertion extraction and negation detection.
    Results are informational warnings only.

    Args:
        chunks: List of chunk texts as retrieved.
        query: The user query (for context, not currently used in logic).

    Returns:
        List of ConsistencyFlag (warning severity, never blocks).
    """
    if len(chunks) < 2:
        return []

    flags: List[ConsistencyFlag] = []

    # Extract assertions and numerics per chunk
    chunk_assertions = [_extract_assertions(c) for c in chunks]
    chunk_numerics = [_extract_numerics(c) for c in chunks]

    # Compare each pair of chunks
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            # --- Check assertion contradictions ---
            for a_i in chunk_assertions[i]:
                for a_j in chunk_assertions[j]:
                    # Same subject, similar predicate, opposite negation
                    if (a_i["subject"] == a_j["subject"]
                            and a_i["predicate"] == a_j["predicate"]
                            and a_i["negated"] != a_j["negated"]):
                        flags.append(ConsistencyFlag(
                            chunk_a_idx=i,
                            chunk_b_idx=j,
                            description=(
                                f"Possible contradiction on '{a_i['subject']}': "
                                f"chunk {i+1} says '{a_i['span']}' vs "
                                f"chunk {j+1} says '{a_j['span']}'"
                            ),
                            snippet_a=a_i["span"],
                            snippet_b=a_j["span"],
                        ))

            # --- Check numeric contradictions ---
            for n_i in chunk_numerics[i]:
                for n_j in chunk_numerics[j]:
                    if (n_i["subject"] == n_j["subject"]
                            and n_i["value"] != n_j["value"]):
                        flags.append(ConsistencyFlag(
                            chunk_a_idx=i,
                            chunk_b_idx=j,
                            description=(
                                f"Numeric disagreement on '{n_i['subject']}': "
                                f"chunk {i+1} says {n_i['value']} vs "
                                f"chunk {j+1} says {n_j['value']}"
                            ),
                            snippet_a=n_i["span"],
                            snippet_b=n_j["span"],
                        ))

    if flags:
        logger.info(f"Consistency check found {len(flags)} potential contradiction(s)")

    return flags
