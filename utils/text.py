"""
Text Processing Utilities
=========================

Shared functions for text sanitization, hashing, file I/O, and chunking
that were previously duplicated across ``RagV2.py``,
``rag/content_scanner.py``, ``rag/ingest.py``, and
``defense/guards.py``.

Usage::

    from utils.text import sanitize_llm_output, sha256_hash, read_document, chunk_text
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text sanitization
# ---------------------------------------------------------------------------

def strip_role_markers(text: str) -> str:
    """
    Remove common LLM role / template markers from *text*.

    Handles ``<|im_end|>``, ``[INST]``, ``### Human:``, ``SYSTEM:``,
    ``<< SYS >>``, HTML comments, and generic XML/HTML tags.
    """
    # LLM special tokens
    text = re.sub(r"<\|[^>]{1,40}\|>", "", text)
    text = re.sub(r"\[/?INST\]", "", text)
    # Chat-style role prefixes
    text = re.sub(r"(?m)^###\s*(Human|Assistant|System|User|AI)\s*:", "", text)
    text = re.sub(r"(?m)^SYSTEM\s*:", "", text)
    # Llama-style system tags
    text = re.sub(r"(<<\s*/?SYS\s*>>)", "", text)
    # HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # Generic HTML/XML tags (up to 100 chars to avoid runaway)
    text = re.sub(r"<[^>]{1,100}>", "", text)
    return text


def sanitize_llm_output(text: str) -> str:
    """
    Clean up raw LLM output: strip role markers, QA boilerplate,
    de-stutter repeated words, and normalise whitespace.

    This is the consolidated version of ``RagV2._clean_answer`` and
    ``rag.content_scanner._sanitize_content`` (minus PII
    redaction which is handled separately by ``utils.pii``).
    """
    text = strip_role_markers(text)
    # QA boilerplate
    text = re.sub(r"(?im)^\s*(question|answer)\s*:\s*", "", text)
    text = re.sub(r"(?i)\bq:\s*|\ba:\s*", "", text)
    # De-stutter (e.g. "the the" -> "the")
    text = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", text)
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text* (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def md5_hash(text: str) -> str:
    """Return the MD5 hex digest of *text* (UTF-8 encoded)."""
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_document(path: str) -> str:
    """
    Extract plain text from a supported document file.

    Supported formats: ``.pdf``, ``.docx``, ``.txt``, ``.md``.
    Returns an empty string for unsupported formats.
    """
    ext = pathlib.Path(path).suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            txt = ""
            for page in PdfReader(path).pages:
                txt += page.extract_text() or ""
            return txt
        except Exception as exc:
            logger.warning("Failed to read PDF %s: %s", path, exc)
            return ""

    if ext in (".txt", ".md"):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except Exception as exc:
            logger.warning("Failed to read text file %s: %s", path, exc)
            return ""

    if ext == ".docx":
        try:
            from docx import Document as Docx
            doc = Docx(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:
            logger.warning("Failed to read DOCX %s: %s", path, exc)
            return ""

    return ""  # unsupported format


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """
    Split *text* into overlapping chunks.

    Args:
        text:    The text to split.
        size:    Maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    text = text.replace("\r", "").strip()
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += max(1, size - overlap)
    return [c for c in chunks if c.strip()]
