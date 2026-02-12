"""
OCSF (Open Cybersecurity Schema Framework) Mapper

Converts GuardResult and PipelineTiming objects into standardized
OCSF v1.3.0 Security Finding (class_uid 2001) and Detection Finding
(class_uid 2004) events.

These OCSF events are attached to OpenTelemetry spans and Langfuse
traces for structured security observability, and optionally written
to a local JSONL file for later SIEM ingestion.

Reference: https://schema.ocsf.io/1.3.0/
"""

import json
import logging
import os
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# OCSF constants
OCSF_VERSION = "1.3.0"
PRODUCT_NAME = "RAG-LLM Guardrails"
PRODUCT_VENDOR = "Enovos"

# Severity mapping: GuardResult.severity -> OCSF severity_id
# OCSF severity_id: 0=Unknown, 1=Informational, 2=Low, 3=Medium, 4=High, 5=Critical, 6=Fatal
_SEVERITY_MAP = {
    "allowed": 1,       # Informational
    "review": 3,        # Medium
    "escalate": 3,      # Medium
    "blocked": 4,       # High
}

# OCSF activity_id: 0=Unknown, 1=Create, 2=Update
_ACTIVITY_MAP = {
    True: 1,   # triggered -> Create
    False: 0,  # not triggered -> Unknown
}

# OCSF status_id: 0=Unknown, 1=Success (new), 2=Failure
_STATUS_MAP = {
    "allowed": 1,
    "review": 1,
    "escalate": 1,
    "blocked": 2,
}


def _epoch_ms(dt: Optional[datetime] = None) -> int:
    """Return current time or given datetime as epoch milliseconds."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _base_metadata() -> Dict[str, Any]:
    """Common OCSF metadata block."""
    return {
        "version": OCSF_VERSION,
        "product": {
            "name": PRODUCT_NAME,
            "vendor_name": PRODUCT_VENDOR,
            "lang": "en",
        },
        "logged_time": _epoch_ms(),
    }


def guard_result_to_ocsf(
    guard_name: str,
    severity_value: str,
    reason: str,
    triggered: bool,
    request_id: str = "",
    guardrails_mode: str = "complete",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Convert a single guard evaluation result to an OCSF Security Finding
    (class_uid 2001, category_uid 2 — Findings).

    Args:
        guard_name: Guard identifier (e.g. "llm-guard", "input-sentimental").
        severity_value: One of "allowed", "blocked", "review", "escalate".
        reason: Human-readable explanation from the guard.
        triggered: Whether the guard was actively evaluated.
        request_id: Pipeline request ID for correlation.
        guardrails_mode: Active mode ("off", "classic", "complete").
        timestamp: Event time (defaults to now).

    Returns:
        OCSF-compliant dict (Security Finding, class_uid 2001).
    """
    severity_lower = severity_value.lower() if severity_value else "allowed"
    ts = _epoch_ms(timestamp)

    return {
        "class_uid": 2001,
        "class_name": "Security Finding",
        "category_uid": 2,
        "category_name": "Findings",
        "severity_id": _SEVERITY_MAP.get(severity_lower, 0),
        "activity_id": _ACTIVITY_MAP.get(triggered, 0),
        "status_id": _STATUS_MAP.get(severity_lower, 0),
        "time": ts,
        "finding_info": {
            "title": f"Guard: {guard_name}",
            "uid": request_id,
            "desc": (reason or "")[:1000],
            "analytic": {
                "type": guardrails_mode,
                "name": guard_name,
            },
        },
        "metadata": _base_metadata(),
    }


def pipeline_timing_to_ocsf(
    request_id: str,
    query_preview: str,
    total_ms: float,
    was_blocked: bool,
    final_result: str,
    layers: Optional[List[Dict[str, Any]]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Convert a full pipeline timing summary to an OCSF Detection Finding
    (class_uid 2004, category_uid 2 — Findings).

    Args:
        request_id: Unique pipeline request ID.
        query_preview: First ~50 chars of the user query.
        total_ms: Total pipeline latency in milliseconds.
        was_blocked: Whether the pipeline blocked the request.
        final_result: "ALLOWED", "BLOCKED", "REVIEW", etc.
        layers: Optional list of layer dicts with name, duration_ms, result.
        start_time: Pipeline start time.
        end_time: Pipeline end time.

    Returns:
        OCSF-compliant dict (Detection Finding, class_uid 2004).
    """
    result_lower = final_result.lower() if final_result else "allowed"

    # Build evidences from layer breakdown
    evidences = []
    if layers:
        for layer in layers:
            evidences.append({
                "name": layer.get("name", "unknown"),
                "duration_ms": round(layer.get("duration_ms", 0), 2),
                "result": layer.get("result", "UNKNOWN"),
            })

    event = {
        "class_uid": 2004,
        "class_name": "Detection Finding",
        "category_uid": 2,
        "category_name": "Findings",
        "severity_id": _SEVERITY_MAP.get(result_lower, 0),
        "activity_id": 1,  # Create
        "status_id": 2 if was_blocked else 1,
        "time": _epoch_ms(start_time),
        "duration": round(total_ms, 2),
        "finding_info": {
            "title": "Guardrails Pipeline Evaluation",
            "uid": request_id,
            "desc": f"Query: {query_preview} | Result: {final_result} | {total_ms:.0f}ms",
        },
        "evidences": evidences,
        "metadata": _base_metadata(),
    }

    if end_time:
        event["end_time"] = _epoch_ms(end_time)

    return event


# ---------------------------------------------------------------------------
# Convenience wrappers that accept the actual dataclass objects
# (avoids importing guardrails types here — keeps the module dependency-free)
# ---------------------------------------------------------------------------

def guard_results_to_ocsf_list(
    guard_results: List[Any],
    request_id: str = "",
    guardrails_mode: str = "complete",
    timestamp: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a list of GuardResult objects to OCSF Security Finding dicts.
    Accepts any object with .guard_name, .severity.value, .reason, .triggered.
    """
    findings = []
    for r in guard_results:
        try:
            findings.append(guard_result_to_ocsf(
                guard_name=r.guard_name,
                severity_value=r.severity.value if hasattr(r.severity, "value") else str(r.severity),
                reason=r.reason,
                triggered=r.triggered,
                request_id=request_id,
                guardrails_mode=guardrails_mode,
                timestamp=timestamp,
            ))
        except Exception as e:
            logger.debug(f"OCSF mapping skipped for guard {getattr(r, 'guard_name', '?')}: {e}")
    return findings


def pipeline_timing_obj_to_ocsf(timing: Any) -> Dict[str, Any]:
    """
    Convert a PipelineTiming dataclass to an OCSF Detection Finding dict.
    Accepts any object with .request_id, .query_preview, .total_ms,
    .was_blocked, .final_result, .layers, .start_time, .end_time.
    """
    layers_data = []
    for layer in getattr(timing, "layers", []):
        layers_data.append({
            "name": layer.layer_name,
            "duration_ms": layer.duration_ms,
            "result": layer.result or "UNKNOWN",
        })

    return pipeline_timing_to_ocsf(
        request_id=timing.request_id,
        query_preview=timing.query_preview,
        total_ms=timing.total_ms,
        was_blocked=timing.was_blocked,
        final_result=timing.final_result,
        layers=layers_data,
        start_time=timing.start_time,
        end_time=timing.end_time,
    )


# ---------------------------------------------------------------------------
# Local JSONL emitter (POC fallback — no SIEM required)
# ---------------------------------------------------------------------------

_DEFAULT_OCSF_LOG = os.path.join("logs", "ocsf_events.jsonl")


def emit_ocsf_event(event: Dict[str, Any], path: str = _DEFAULT_OCSF_LOG) -> None:
    """
    Append a single OCSF event as one JSON line to a local JSONL file.

    Args:
        event: OCSF-compliant dict.
        path: File path for the JSONL log (default: logs/ocsf_events.jsonl).
    """
    try:
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:
        logger.debug(f"Failed to write OCSF event to {path}: {e}")
