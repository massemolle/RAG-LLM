# NVIDIA NeMo Guardrails — RAG Integration

This directory contains the production guardrails stack for the RAG chatbot: multi-layer input/output security, NeMo flows, policy framework, timing metrics, and observability (Langfuse/OpenTelemetry).

---

## Architecture Overview

The pipeline uses **speculative parallel execution**:

- **Line A (parallel):** Input guards (Layer 0 → Layer 1 → Layer 2 “NeMo”).
- **Line B (parallel):** LLM generation.
- When the LLM responds, **output guards** run in parallel.
- **LLM Judge (Layer 3)** runs only on escalation from input guards; otherwise it is skipped.

**Layers:**

| Layer | Name | Role |
|-------|------|------|
| 0 | Embedding similarity | Semantic attack-pattern detection (optional) |
| 1 | LLM Guard | Defensive scanners (prompt injection, toxicity, PII, etc.) |
| 2 | NeMo (input guardrails) | 2-layer: deterministic patterns + NeMo jailbreak heuristics |
| 3 | LLM Judge | Escalation-only model-based judge |
| — | Output guards | Topic, PII, grounding, integrity, etc. |

All layers are timed; metrics are exposed to the UI and to Langfuse/OpenTelemetry.

---

## Directory Layout

```
nvidia_nemo/
├── config/
│   ├── config.yml      # NeMo Guardrails config (models, rails, flows)
│   ├── config.py       # Config loading helpers
│   └── actions.py      # NeMo custom actions (RAG retrieve, validate_chunk, logging)
├── rails/              # NeMo Colang flows
│   ├── input_rails.co
│   ├── output_rails.co
│   ├── rag_flow.co
│   ├── tool_safety.co
│   ├── pii_handling.co
│   ├── jailbreak_detection.co
│   └── monitoring.co
├── policy_matrix.yml   # Policy matrix (content × intent → decision)
├── __init__.py
├── README.md           # This file
│
├── enhanced_guardrails.py   # Main pipeline: layers 0–3, parallel guards, LLM call
├── guardrails_integration.py # NeMo LLMRails + RAG (GuardedRAG, initialize_guardrails)
├── guardrails_wrapper.py    # GuardrailsWrapper + GuardrailsStatus for UI
├── timing_metrics.py       # PipelineTiming, LayerTiming, per-guard breakdown
│
├── attack_embeddings.py     # Layer 0: attack-pattern embeddings & similarity
├── llm_guard_integration.py # Layer 1: llm-guard scan_input / scan_output
├── jailbreak_heuristics.py  # NeMo-style length/perplexity, prefix/suffix (Layer 2)
├── pii_detection.py         # PII detection/redaction (Presidio or regex)
├── content_classifier.py    # ContentCategory detection from text
├── policy_framework.py      # Policy matrix, IntentClass, ContentCategory, PolicyDecision
├── unified_input_security.py # Intent-based guardrails using policy matrix
├── structured_guardrails.py  # Structured guards with severity and logging
├── retrieval_rails_integration.py # Chunk sanitization for RAG retrieval
├── production_hardening.py   # Caching, rate limiting, model routing
│
└── test_guardrails.py       # Tests for guardrails behavior
```

---

## File Reference

### Core pipeline and integration

- **`enhanced_guardrails.py`** — Central orchestration: runs Layer 0 (optional), Layer 1 (LLM Guard), Layer 2 (NeMo 2-layer: deterministic + jailbreak heuristics), then LLM call; runs input guards and LLM in parallel, then output guards. Hosts `guard_input_security_3layer` (actually 2-layer now), output guards (topic, PII, grounding, integrity), and the single LLM Judge step on escalation. Uses `timing_metrics` for per-layer and per-guard timings.
- **`guardrails_integration.py`** — NeMo `LLMRails` setup and RAG wiring: `initialize_guardrails()`, `GuardedRAG`, custom actions (e.g. `retrieve_from_rag`, `validate_chunk`, logging). Connects `config/` and `rails/` to the RAG pipeline.
- **`guardrails_wrapper.py`** — `GuardrailsWrapper` and `GuardrailsStatus`: status for UI (triggered guards, jailbreak/PII flags, grounding, risk score). Used by the Streamlit app for display.

### Timing and observability

- **`timing_metrics.py`** — `PipelineTiming`, `LayerTiming`, `GuardrailsTimer`, `record_timing`, `get_timing_stats`. Tracks wall-clock per layer and per-guard; exposes “bottleneck” and “SKIPPED (no escalation)” for the LLM Judge. Consumed by the UI and by Langfuse/OpenTelemetry.

### Layers and guards

- **`attack_embeddings.py`** — Layer 0: attack-pattern embedding DB and similarity check (`check_attack_similarity`). Used when enabled for semantic attack detection.
- **`llm_guard_integration.py`** — Layer 1: `llm-guard` (`scan_prompt`, `scan_output`) for prompt injection, toxicity, secrets, invisible text, etc. Wraps `scan_input_text` / `scan_output_text`.
- **`jailbreak_heuristics.py`** — NeMo-style heuristics: length-per-perplexity and prefix/suffix perplexity (GPT-2). Used inside Layer 2 (NeMo) in `enhanced_guardrails`.
- **`pii_detection.py`** — PII detection and redaction (Presidio if available, else regex). Used in input/output flows and retrieval.
- **`content_classifier.py`** — Classifies text into `ContentCategory` (e.g. prompt_injection, jailbreak, violence, PII). Used by the policy framework.
- **`policy_framework.py`** — `ContentCategory`, `IntentClass`, `ResponseMode`, `PolicyMatrix`, `PolicyDecision`, `IntentClassifier`. Loads `policy_matrix.yml` and drives intent-based decisions.
- **`unified_input_security.py`** — Applies the policy matrix to input: content + intent → `PolicyDecision` / `GuardResult`. Can be used as an alternative or complement to the layered pipeline.
- **`structured_guardrails.py`** — Structured guard runner with severity and standardized logging (`GuardResult`, log lines). Used for consistent guard interface and logs.

### Retrieval and production

- **`retrieval_rails_integration.py`** — Sanitizes RAG chunks (instruction stripping, secret redaction). Used by retrieval flows and `validate_chunk`-style actions.
- **`production_hardening.py`** — `GuardrailsCache`, rate limiter, model router; optional caching, per-IP/session limits, and escalation throttling.

### Config and rails

- **`config/config.yml`** — NeMo Guardrails: models, instructions, rails (input, output, retrieval, execution, monitoring), jailbreak thresholds.
- **`config/actions.py`** — Custom NeMo actions: RAG retrieve, chunk validation, logging, security/audit writes, alerts. Used by Colang flows in `rails/`.
- **`rails/*.co`** — Colang flows for input, output, RAG, tools, PII, jailbreak, monitoring. Referenced by `config.yml`.

### Tests

- **`test_guardrails.py`** — Tests for guardrails behavior (layers, blocking, timing, etc.).

---

## Usage (high level)

- The **Streamlit app** (`streamv3.py`) and **RAG entrypoint** (`RagV2.py`) use `enhanced_guardrails` for the full pipeline (input guards || LLM → output guards, LLM Judge on escalation).
- Observability is via **Langfuse and OpenTelemetry**; the old `logs/` folder and local `.log` files are obsolete.
- For a “wrap RAG with NeMo only” usage, `GuardedRAG` and `initialize_guardrails()` in `guardrails_integration.py` provide the classic NeMo+RAG integration.

---

## References

- [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails)
- [Guardrails process](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-process.html)
- [Configuration guide](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/configuration-guide.html)
