# RAG-LLM Assistant

Secure Retrieval-Augmented Generation assistant with multi-layer guardrails,
data classification, and comprehensive observability.

**State of the art (2026):** All defense schemes described below are **implemented and active** in the codebase. This document is the single source of truth for what is working.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamv3.py --server.port 8502
```

Set `BEAMSTUDIO_API_KEY` in your environment for cloud LLM access.

---

## Project Structure

```
RAG-LLM/
├── streamv3.py                 # Streamlit UI (main entry point)
├── RagV2.py                    # Core RAG orchestrator
├── embedding.py                # BM25 / BERT embedding classes
├── llm_client.py               # LLM client abstraction (local + BeamStudio)
├── pipeline_maker.py           # Pipeline factory utilities
├── policy.yaml                 # Central policy configuration
│
├── rag/                        # RAG data & security module
│   ├── __init__.py             # Public API re-exports
│   ├── content_scanner.py      # Deep ingestion-time threat scanning
│   ├── classification.py       # Data classification (5 levels)
│   ├── safe_retrieval.py       # BM25-based retrieval over sanitized index
│   ├── ingest.py               # Secure ingestion pipeline
│   ├── index_versioning.py     # Snapshot and rollback for knowledge base
│   ├── retrieval_monitor.py    # Anomaly detection on retrieval patterns
│   ├── consistency.py          # Cross-chunk contradiction flagging
│   ├── data/                   # Raw documents for ingestion
│   └── index/                  # Sanitized JSONL index (gitignored)
│
├── defense/                    # Chunk-level safety guards
│   ├── guards.py               # Chunk injection filtering, PII redaction, audit logging
│   └── safe_retrieval.py       # Backward-compat re-export -> rag/
│
├── nvidia_nemo/                # Multi-layer guardrails engine
│   ├── enhanced_guardrails.py  # Orchestrator (3 modes: off/classic/complete)
│   ├── structured_guardrails.py# Structured guardrails with severity
│   ├── guardrails_wrapper.py   # UI wrapper for guardrails status
│   ├── pii_detection.py        # PII detection (Presidio + regex)
│   ├── attack_embeddings.py    # Layer 0: embedding similarity
│   ├── llm_guard_integration.py# Layer 1: LLM Guard
│   ├── jailbreak_heuristics.py # Statistical jailbreak detection
│   ├── policy_framework.py     # Policy matrix & intent classification
│   ├── content_classifier.py   # Content category detection
│   ├── production_hardening.py # Caching, rate limiting, model routing
│   ├── timing_metrics.py       # Performance timing for guardrails
│   ├── config/                 # NeMo guardrails configuration
│   └── rails/                  # NeMo Colang rail definitions
│
├── utils/                      # Shared utility library
│   ├── __init__.py             # Public API re-exports
│   ├── pii.py                  # Unified PII detection & redaction
│   ├── injection_patterns.py   # Canonical injection pattern registry
│   ├── text.py                 # Text sanitization, hashing, file I/O
│   └── audit_logger.py         # Structured JSONL audit/security logging
│
├── observability/              # Observability integrations
│   ├── langfuse_integration.py # Langfuse tracing and logging
│   ├── opentelemetry_integration.py # OpenTelemetry integration
│   └── ocsf_mapper.py          # OCSF event mapping
│
├── model/                      # Legacy model implementations
│   ├── BERT.py                 # BERT embedding model
│   └── database.py             # Document processing utilities
│
├── tests/                      # Test suite
│   ├── test_pii.py             # PII detection tests
│   ├── test_injection_patterns.py # Injection pattern tests
│   ├── test_text_utils.py      # Text utility tests
│   ├── test_content_scanner.py # Content scanner tests
│   ├── test_classification.py  # Data classification tests
│   ├── test_ingest.py          # Ingestion pipeline tests
│   ├── test_guardrails_layers.py # Guardrails layer tests
│   └── test_policy_matrix.py   # Policy matrix tests
│
├── ingest_safe.py              # Backward-compat stub -> rag/ingest.py
├── .streamlit/config.toml      # Streamlit configuration (HTTPS)
├── .ssl/                       # Self-signed certificates (gitignored)
└── .gitignore
```

---

## UI Controls

The top bar has 3 controls:

| Control | Type | Purpose |
|---------|------|---------|
| **Safe mode** | Checkbox | Disables tool use, restricts to retrieval-only answers (upcoming) |
| **Restrict to documents only** | Checkbox | ON: only answer from indexed docs. OFF: answer any question |
| **Guardrails** | Dropdown | Off / Classic / Complete (see below) |

---

## Defense Architecture (2026 State of the Art)

Everything in this section is **implemented and working** in the current codebase. Code paths are given so you can verify or extend behavior.

### Overview

Defenses are organized in four tiers: **input guardrails** (before the LLM), **output guardrails** (after the LLM), **RAG-specific defenses** (ingestion and retrieval), and **global safeguards** (rate limiting, audit, observability).

```
User query
    │
    ├─► [1] INPUT GUARDRAILS (parallel)
    │       Embedding similarity, LLM Guard, NeMo, Topic taxonomy,
    │       Input security/sentimental/topic (LLM judges, multi-turn aware)
    │       → BLOCK / ESCALATE → LLM Judge (only if escalated)
    │
    ├─► [2] RAG RETRIEVAL (SafeIndex only)
    │       Chunk-level injection filter (defense/guards.py)
    │       Retrieval monitor, consistency checks (rag/)
    │
    ├─► [3] LLM GENERATION
    │       build_prompt() in RagV2.py (system prompts from constants)
    │
    └─► [4] OUTPUT GUARDRAILS (parallel)
            Topic, Global, Differential, Integrity, IP,
            LLM Guard output (toxicity/sensitive), System prompt leakage
            → BLOCK / ALLOW → response to user
```

### 1. Input Guardrails (Code: `nvidia_nemo/enhanced_guardrails.py`)

| Guard | Mode | What it does | Code reference |
|-------|------|--------------|----------------|
| **embedding-similarity** | Complete | Semantic similarity to known attack patterns (optional Layer 0) | `guard_embedding_similarity()`, `attack_embeddings.py` |
| **llm-guard** | Complete | Prompt injection, toxicity, secrets, invisible text (Layer 1) | `guard_llm_guard()`, `llm_guard_integration.scan_input_text()` |
| **topic-taxonomy** | Complete | Topic classification vs allowed domains | `guard_topic_taxonomy()` |
| **input-security** | Complete / Classic | NeMo 2-layer (deterministic + jailbreak heuristics) or policy matrix; escalation to LLM Judge | `guard_input_security_3layer()`, `_guard_input_security_policy_framework()` |
| **input-sentimental** | Classic / Complete | LLM judge: frustration, anger, human handoff requests | `guard_input_sentimental()`, `_classic_llm_judge("input_sentimental")` |
| **input-topic** | Classic / Complete | LLM judge: Enovos/energy relevance of the query | `guard_input_topic()`, `_classic_llm_judge("input_topic")` |

**Multi-turn awareness:** The three input LLM judges receive a sliding window of the last 5 user messages per session (`_session_history`, `_format_conversation_history()`). Prompts use a `{conversation_history}` placeholder; see `ENOVOS_CLASSIC_PROMPTS` in `enhanced_guardrails.py`.

### 2. Output Guardrails (Code: `nvidia_nemo/enhanced_guardrails.py`)

| Guard | Mode | What it does | Code reference |
|-------|------|--------------|----------------|
| **output-topic** | Classic / Complete | LLM judge: response relevance to Enovos/energy | `guard_output_topic()`, `_classic_llm_judge("output_topic")` |
| **output-global** | Classic / Complete | LLM judge: hallucination, policy, toxicity, PII, internal refs, free-service promises | `guard_output_global()`, `_classic_llm_judge("output_global")` |
| **output-differential** | Complete | Detects contradictions vs previous or expected content | `guard_output_differential()` |
| **output-integrity** | Complete | Grounding and citation consistency | `guard_output_integrity()` |
| **output-ip** | Complete | Intellectual property and source misuse | `guard_output_ip()` |
| **output-llm-guard** | Classic / Complete | LLM Guard output scanners: toxicity, sensitive data | `guard_output_llm_guard()`, `llm_guard_integration.scan_output_text()` |
| **output-prompt-leakage** | Classic / Complete | Substring + Jaccard similarity vs known system prompts (`RagV2.SYSTEM_PROMPTS`) | `guard_output_prompt_leakage()` |

System prompts used for leakage detection are defined as constants in `RagV2.py`: `SYSTEM_PROMPT_RAG`, `SYSTEM_PROMPT_GENERAL`, `SYSTEM_PROMPT_STRICT`.

### 3. RAG-Specific Defenses

| Component | What it does | Code reference |
|-----------|--------------|----------------|
| **Ingestion scanner** | Deep scan before indexing: hidden Unicode, base64, injection patterns (60+ from `utils/injection_patterns.py`), structural anomalies, PII. Block → quarantine; warn → ingest with flag. | `rag/content_scanner.py` (`scan_document()`), `rag/ingest.py` |
| **Data classification** | 5 levels (public → secret). Classified/secret rejected at ingestion. Folder-based inference or explicit UI. | `rag/classification.py` (`classify_document()`, `is_ingestible()`) |
| **Chunk filtering** | Retrieved chunks filtered for injection patterns before being sent to the LLM (no user-facing block mode; guardrails handle blocking). | `defense/guards.py` (`filter_chunks()`, `looks_like_injection()`), `RagV2.answer()` |
| **Retrieval monitor** | Per-document retrieval frequency in a sliding window; flags anomalies (e.g. one doc fetched far more than others). | `rag/retrieval_monitor.py` (`get_retrieval_monitor().record()`) |
| **Consistency check** | Cross-chunk contradiction flagging (keyword/numeric heuristics). Informational only, never blocks. | `rag/consistency.py` (`flag_inconsistencies()`), `RagV2.answer()` |
| **Index versioning** | Snapshot before overwrite, rollback from UI. | `rag/index_versioning.py` (`list_versions()`, `rollback_index()`) |
| **Citation enforcement** | "Restrict to documents only" (cite_or_silent): refuse when no docs or when LLM does not cite. | `RagV2.py` (early exit + post-generation check), `policy.yaml` → `output.cite_or_silent` |

### 4. Global Safeguards

| Mechanism | What it does | Code reference |
|-----------|--------------|----------------|
| **Global rate limit** | 60 queries per minute globally (configurable). Applied at start of `answer()` (guardrails on) and in the guardrails-off branch in the UI. | `nvidia_nemo/production_hardening.py` (`GlobalRateLimiter`, `check_global_limit()`), `enhanced_guardrails.answer()`, `streamv3.py` |
| **Audit logging** | Every query + chunks (redacted) and risk/quarantine counts written to JSONL. | `defense/guards.py` (`gate_and_log()` → `_log()`), `utils/audit_logger.py` |
| **OCSF mapping** | Guard results and pipeline timing mapped to OCSF Security/Detection Finding events; attached to OpenTelemetry/Langfuse and optional local JSONL. | `observability/ocsf_mapper.py` |

### 5. Shared Defense Primitives (Code: `utils/`)

| Module | Purpose |
|--------|---------|
| `utils/pii.py` | Canonical PII patterns (email, phone, SSN, IBAN, etc.) and redaction; used by guardrails, RAG, and audit. |
| `utils/injection_patterns.py` | 60+ prompt-injection patterns with weights and categories; used by content scanner and `defense/guards.py`. |
| `utils/text.py` | Text sanitization, hashing (SHA-256, MD5), document reading, chunking. |
| `utils/audit_logger.py` | Structured JSONL audit and security logging. |

### 6. Observability (Code: `observability/`)

Traces (retrieval, generation, each guard) are sent to **Langfuse** via **OpenTelemetry** OTLP. Guard results and timing are mapped to **OCSF** for security observability. See `observability/README.md` and `observability/ocsf_mapper.py`.

---

## Guardrails Modes

| Mode | Description | Latency |
|------|-------------|---------|
| **Off** | No guardrails, direct LLM API call | Fastest |
| **Classic** | LLM-as-judge only (5 judges: 3 input, 2 output) | Fast (~500-800ms) |
| **Complete** | Full multi-layer pipeline with speculative parallel execution | Thorough (~1-2s) |

### Complete Mode Architecture (Speculative Parallel)

In Complete mode, **fast guards** and **LLM judges** run in parallel simultaneously:

```
┌──────────────────────────────────────────────────────────────┐
│                   PARALLEL EXECUTION                          │
│  ┌─────────────────────────┐  ┌────────────────────────────┐ │
│  │ Fast Guards (~500ms)    │  │ LLM Judges (speculative)   │ │
│  │ - Embedding similarity  │  │ - Input sentimental        │ │
│  │ - LLM Guard            │  │ - Input security           │ │
│  │ - NeMo Guardrails      │  │ - Input topic              │ │
│  │ - Topic taxonomy       │  │                            │ │
│  └─────────────────────────┘  └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Fast Guards Result? │
              └─────────────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   All ALLOWED   Any BLOCKED   Any ESCALATE
       │             │             │
       ▼             ▼             ▼
    VALID         BLOCKED      Use LLM Judge
 (ignore LLM)                   Results
```

---

## Shared Utilities (`utils/`)

The `utils/` package consolidates logic that was previously duplicated
across 7+ files:

- **`utils/pii`** -- Unified PII detection and redaction (Presidio + regex).
  Superset of all patterns: email, phone, credit card, SSN, IBAN, IP, passport, API key.
- **`utils/injection_patterns`** -- Canonical registry of 60+ prompt-injection
  patterns with weighted scoring and category filtering.
- **`utils/text`** -- Text sanitization, SHA-256/MD5 hashing, document reading
  (PDF/DOCX/TXT), and overlapping chunking.
- **`utils/audit_logger`** -- Structured JSONL audit and security logging.

---

## Data Classification

Documents are classified at ingestion time into 5 levels:

| Level | Ingestible | Description |
|-------|-----------|-------------|
| `public` | Yes | Openly available |
| `entity_internal` | Yes | Internal to the entity (default) |
| `group_internal` | Yes | Internal to the group |
| `classified` | **No** | Restricted access -- rejected at ingestion |
| `secret` | **No** | Top secret -- rejected at ingestion |

Classification is inferred from folder names or set explicitly via the UI.

---

## LLM Models

### BeamStudio (Cloud)

| Model | Description | Notes |
|-------|-------------|-------|
| **gpt-5.1** | Latest reasoning model (default) | High token limit, best quality |
| **gpt-4o** | Fast, efficient model | Good balance of speed/quality |
| **gpt-5-mini** | Lightweight reasoning model | Temperature must be 1 (API constraint) |

### Local LLMs

Local models are supported via HuggingFace Transformers pipelines. See
`llm_client.py` for the `LocalLLMClient` implementation.

---

## HTTPS / Deployment

### Self-Signed Certificates (Development)

Self-signed certificates are pre-generated in `.ssl/` and configured in
`.streamlit/config.toml`. To regenerate:

```bash
openssl req -x509 -newkey rsa:4096 \
  -keyout .ssl/private.key \
  -out .ssl/certchain.pem \
  -days 365 -nodes \
  -subj "/CN=localhost/O=RAG-LLM/C=LU"
```

### Production

Use a reverse proxy (nginx/Caddy) with TLS termination in front of
Streamlit running on HTTP.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Embedding Functions

All embedding functions are in `embedding.py` as classes with `retrieve`
and `process` methods. BM25 is the default (fastest). See the file for
the template interface.
