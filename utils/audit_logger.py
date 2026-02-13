"""
Unified Audit / Security JSONL Logger
======================================

Consolidates the "append JSON line to a JSONL file" pattern that was
previously copy-pasted in ``defense/guards.py``,
``nvidia_nemo/config/actions.py``, and
``nvidia_nemo/guardrails_integration.py``.

Usage::

    from utils.audit_logger import log_audit, log_security

    log_audit({"event": "query", "user": "analyst", "blocked": False})
    log_security({"event": "injection_attempt", "query": "..."})
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
_DEFAULT_AUDIT_PATH = "./logs/audit.jsonl"
_DEFAULT_SECURITY_PATH = "./logs/security.jsonl"


def _append_jsonl(event: Dict[str, Any], path: str) -> None:
    """Append a single JSON object as a line to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def log_audit(
    event: Dict[str, Any],
    log_path: Optional[str] = None,
) -> None:
    """
    Append *event* to the audit JSONL log.

    Automatically adds a ``timestamp`` field if not present.

    Args:
        event:    Dict to serialise as one JSON line.
        log_path: Override the default audit log path.
    """
    if "timestamp" not in event and "ts" not in event:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    path = log_path or _DEFAULT_AUDIT_PATH
    try:
        _append_jsonl(event, path)
    except Exception as exc:
        logger.warning("Failed to write audit log to %s: %s", path, exc)


def log_security(
    event: Dict[str, Any],
    log_path: Optional[str] = None,
) -> None:
    """
    Append *event* to the security JSONL log.

    Automatically adds a ``timestamp`` field if not present.

    Args:
        event:    Dict to serialise as one JSON line.
        log_path: Override the default security log path.
    """
    if "timestamp" not in event and "ts" not in event:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    path = log_path or _DEFAULT_SECURITY_PATH
    try:
        _append_jsonl(event, path)
    except Exception as exc:
        logger.warning("Failed to write security log to %s: %s", path, exc)
