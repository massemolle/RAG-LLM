# Observability & Monitoring

Production-ready OpenTelemetry integration with Langfuse for comprehensive tracing, monitoring, and observability of the RAG-LLM system.

## Architecture

### System Overview

```
┌─────────────────┐
│   streamv3.py   │  ← Entry point: Initializes OpenTelemetry
└────────┬────────┘
         │
         ├─→ Initializes OpenTelemetry with Langfuse OTLP endpoint
         │
         ▼
┌─────────────────────────┐
│  OpenTelemetry SDK      │
│  (OTLP HTTP Exporter)   │
└────────┬────────────────┘
         │
         ├─→ Sends traces via OTLP protocol
         │
         ▼
┌─────────────────────────┐
│  Langfuse Cloud         │
│  /api/public/otel       │
└─────────────────────────┘
```

### Component Flow

```
User Query (streamv3.py)
    │
    ├─→ EnhancedStructuredGuardrails.answer()
    │   │
    │   ├─→ Creates root OpenTelemetry trace
    │   │   (via observability.opentelemetry_integration)
    │   │
    │   ├─→ Guardrails evaluation (5 guards)
    │   │   ├─→ input-sentimental
    │   │   ├─→ input-security (3-layer defense)
    │   │   │   ├─→ Layer A: Deterministic patterns
    │   │   │   ├─→ Layer B: NeMo heuristics
    │   │   │   └─→ Layer C: LLM judge
    │   │   ├─→ input-topic
    │   │   ├─→ output-topic
    │   │   └─→ output-global
    │   │
    │   └─→ RAG.answer() (RagV2.py)
    │       │
    │       ├─→ Retrieval operation
    │       │   └─→ Creates retrieval span
    │       │
    │       └─→ LLM generation
    │           └─→ Creates generation span
    │
    └─→ All spans exported to Langfuse via OTLP
```

## Technology Stack

### Core Components

1. **OpenTelemetry SDK** (`opentelemetry-api`, `opentelemetry-sdk`)
   - Standard observability framework
   - Provides tracing, metrics, and logging APIs
   - CNCF standard for distributed tracing

2. **OTLP HTTP Exporter** (`opentelemetry-exporter-otlp-proto-http`)
   - Exports traces to Langfuse via HTTP/protobuf
   - Uses Langfuse's native `/api/public/otel/v1/traces` endpoint
   - No conversion needed - native OpenTelemetry protocol

3. **Langfuse** (Cloud or Self-hosted)
   - Observability platform for LLM applications
   - Receives OpenTelemetry traces via OTLP endpoint
   - Provides dashboard, metrics, and analytics

### File Structure

```
observability/
├── __init__.py                    # Package initialization
├── README.md                       # This file
├── opentelemetry_integration.py   # Core OpenTelemetry setup & helpers
└── langfuse_integration.py        # Fallback direct Langfuse SDK (if OTEL unavailable)
```

## Code Integration

### How Components Connect

#### 1. Initialization (`streamv3.py`)

```python
# At application startup
from observability.opentelemetry_integration import initialize_opentelemetry

# Initialize OpenTelemetry with Langfuse OTLP endpoint
initialize_opentelemetry(
    service_name="rag-llm-system",
    service_version=os.getenv("APP_RELEASE", "1.0.0")
)
```

**What happens:**
- Reads `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from environment
- Configures OTLP HTTP exporter to `https://cloud.langfuse.com/api/public/otel/v1/traces`
- Sets up Basic Auth with base64-encoded API keys
- Creates global tracer provider
- Registers batch span processor

#### 2. Main Query Flow (`enhanced_guardrails.py`)

```python
def answer(self, query: str, role: str, user_id: str, session_id: str, trace_name: str):
    # Create root trace
    from observability.opentelemetry_integration import get_tracer, set_span_attribute
    
    tracer = get_tracer()
    otel_trace = tracer.start_as_current_span(
        trace_name,
        attributes={
            "langfuse.trace.name": trace_name,
            "langfuse.user.id": user_id,
            "langfuse.session.id": session_id,
            "langfuse.trace.tags": json.dumps(["rag", "guardrails", role])
        }
    )
    
    # ... guardrails evaluation ...
    
    # ... RAG retrieval and generation ...
    
    # Flush spans at end
    from observability.opentelemetry_integration import flush
    flush()
```

**What happens:**
- Creates root span for entire query
- Sets trace-level attributes (user, session, tags)
- All child spans automatically inherit trace context
- Flushes spans to Langfuse at end

#### 3. Retrieval Operations (`RagV2.py`)

```python
def answer(self, query: str, role: str):
    # Create retrieval span
    from observability.opentelemetry_integration import create_trace_span, set_span_attribute
    
    with create_trace_span(
        "retrieval",
        attributes={"span.type": "retriever", "span.name": "document_retrieval"}
    ) as retrieval_span:
        # ... retrieval logic ...
        set_span_attribute("input", json.dumps({"query": query}))
        set_span_attribute("output", json.dumps({"documents": docs}))
```

**What happens:**
- Creates nested span under root trace
- Automatically inherits trace context
- Logs retrieval input/output
- Span ends when context exits

#### 4. LLM Generation (`RagV2.py`)

```python
# Create generation span
with create_trace_span(
    "llm_generation",
    attributes={
        "span.type": "generation",
        "span.name": "llm_response",
        "model": self.pipe_model
    }
) as gen_span:
    # ... LLM call ...
    set_span_attribute("input", json.dumps({"prompt": prompt}))
    set_span_attribute("output", json.dumps({"response": response}))
    set_span_attribute("latency_ms", elapsed_ms)
```

**What happens:**
- Creates generation span (special type for LLM calls)
- Langfuse automatically recognizes `span.type: "generation"` and maps to Generation observation
- Logs model, input, output, latency

#### 5. Guardrails Evaluation (`enhanced_guardrails.py`)

```python
# Each guard creates a span
with create_trace_span(
    f"guard_{guard_name}",
    attributes={
        "span.type": "guard",
        "span.name": guard_name,
        "guard.severity": severity.value,
        "guard.triggered": triggered
    }
) as guard_span:
    # ... guard evaluation ...
    set_span_attribute("guard.reason", reason)
```

**What happens:**
- Each guard (input-sentimental, input-security, etc.) creates a span
- Spans are nested under root trace
- Severity, reason, and triggered status are logged

## What Gets Logged

### Trace-Level Attributes

Set on root span, appear on entire trace in Langfuse:

- **`langfuse.trace.name`**: Trace name (e.g., "rag_query_with_guardrails")
- **`langfuse.user.id`**: User identifier (anonymized)
- **`langfuse.session.id`**: Session identifier
- **`langfuse.trace.tags`**: Array of tags (e.g., ["rag", "guardrails", "analyst"])
- **`service.name`**: Service identifier ("rag-llm-system")
- **`service.version`**: Application version

### Observation-Level Attributes

Set on individual spans, appear on observations in Langfuse:

#### Retrieval Spans
- **`span.type`**: "retriever"
- **`input`**: Query text (JSON)
- **`output`**: Retrieved documents (JSON)
- **`retrieval.method`**: "safe_index" or "legacy"
- **`document.count`**: Number of documents retrieved

#### Generation Spans
- **`span.type`**: "generation"
- **`model`**: LLM model name
- **`input`**: Prompt text (JSON)
- **`output`**: LLM response (JSON)
- **`latency_ms`**: Generation time in milliseconds
- **`has_citations`**: Boolean indicating if response has citations

#### Guard Spans
- **`span.type`**: "guard"
- **`guard.name`**: Guard identifier (e.g., "input-security")
- **`guard.severity`**: "allowed", "review", or "blocked"
- **`guard.triggered`**: Boolean
- **`guard.reason`**: Explanation of decision
- **`layer.severity`**: For input-security guard (Layer A/B/C results)

#### Layer Spans (for input-security)
- **`span.type`**: "layer"
- **`layer.name`**: "layer_a", "layer_b", or "layer_c"
- **`layer.severity`**: Layer result
- **`layer.categories`**: Detected categories (JSON)
- **`layer.cached`**: Boolean (if result was cached)

## Langfuse Attribute Mapping

Langfuse automatically maps OpenTelemetry attributes to its data model:

### Trace Mapping
- `langfuse.trace.name` → Trace name
- `langfuse.user.id` → User ID
- `langfuse.session.id` → Session ID
- `langfuse.trace.tags` → Tags array
- `langfuse.trace.metadata.*` → Top-level metadata keys (filterable)

### Observation Mapping
- `span.type: "generation"` → Generation observation
- `span.type: "retriever"` → Retriever observation
- `span.type: "guard"` → Guard observation
- `model` → Model name (for generations)
- `input` → Input data
- `output` → Output data
- `langfuse.observation.metadata.*` → Top-level metadata keys (filterable)

## Setup & Configuration

### 1. Install Dependencies

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

### 2. Configure Environment Variables

Create `.env` file or set environment variables:

```bash
# Required
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # alternative name
APP_RELEASE=v1.0.0  # Application version
```

### 3. Get Langfuse API Keys

1. Sign up at https://cloud.langfuse.com (or self-host)
2. Navigate to **Settings** → **API Keys**
3. Create a new API key pair
4. Copy public key (`pk-lf-...`) and secret key (`sk-lf-...`)

### 4. Verify Setup

The system automatically initializes on startup. Check Streamlit logs for:
- `✅ OpenTelemetry + Langfuse observability enabled` (success)
- `ℹ️ OpenTelemetry not configured` (missing API keys)

## Production Considerations

### Performance

- **Batch Processing**: Spans are batched and exported asynchronously
  - Batch size: 50 spans
  - Queue size: 100 spans
  - Export timeout: 5 seconds
- **Non-blocking**: Span creation doesn't block request processing
- **Automatic Flushing**: Spans are flushed at trace end and on application shutdown

### Reliability

- **Graceful Degradation**: If OpenTelemetry unavailable, falls back to direct Langfuse SDK
- **Error Handling**: Export errors are logged but don't break application flow
- **Retry Logic**: OTLP exporter includes built-in retry for transient failures

### Security

- **PII Protection**: All text is automatically anonymized before export
  - Email addresses, phone numbers, credit cards, API keys, etc.
- **Secure Transport**: Uses HTTPS for all communications
- **Authentication**: Basic Auth with base64-encoded API keys

### Monitoring

- **Trace Completeness**: All operations are traced (guards, retrieval, generation)
- **Span Hierarchy**: Proper parent-child relationships maintained
- **Attribute Consistency**: Standard attribute names across all spans

## Viewing Traces in Langfuse

1. **Access Dashboard**: https://cloud.langfuse.com (or your self-hosted instance)
2. **Navigate to Traces**: See all RAG queries with full hierarchy
3. **Filter & Search**: 
   - By user ID
   - By session ID
   - By trace name
   - By tags
   - By metadata keys
4. **View Details**: Click any trace to see:
   - Complete span hierarchy
   - Guard evaluations with severity
   - Layer A/B/C results
   - Retrieval operations
   - LLM generations with timing
   - All attributes and metadata

## Troubleshooting

### No Traces Appearing

1. **Check API Keys**: `echo $LANGFUSE_PUBLIC_KEY`
2. **Check Initialization**: Look for "OpenTelemetry initialized" in logs
3. **Check Endpoint**: Verify `LANGFUSE_HOST` is correct
4. **Check Network**: Ensure outbound HTTPS to Langfuse is allowed
5. **Wait for Export**: Spans are batched - may take a few seconds

### Missing Spans

- **Verify Instrumentation**: Check that spans are created in code
- **Check Flush**: Ensure `flush()` is called at trace end
- **Check Errors**: Look for export errors in logs

### Performance Issues

- **Reduce Batch Size**: Lower `max_export_batch_size` if needed
- **Increase Timeout**: Adjust `export_timeout_millis` for slow networks
- **Disable for Development**: Set `ENABLE_LANGFUSE_TRACING=false` if needed

## API Reference

### Core Functions

#### `initialize_opentelemetry()`
Initialize OpenTelemetry with Langfuse OTLP endpoint.

**Parameters:**
- `langfuse_public_key` (str, optional): Public key (or env var)
- `langfuse_secret_key` (str, optional): Secret key (or env var)
- `langfuse_host` (str, optional): Langfuse host URL (or env var)
- `service_name` (str): Service name (default: "rag-llm-system")
- `service_version` (str, optional): Service version

**Returns:** `bool` - True if initialized successfully

#### `get_tracer()`
Get OpenTelemetry tracer instance.

**Returns:** Tracer object or None

#### `create_trace_span(name, attributes, kind)`
Create a new span context manager.

**Parameters:**
- `name` (str): Span name
- `attributes` (dict, optional): Span attributes
- `kind` (SpanKind, optional): Span kind

**Returns:** Context manager (use with `with` statement)

#### `set_span_attribute(key, value)`
Set attribute on current active span.

**Parameters:**
- `key` (str): Attribute key
- `value` (Any): Attribute value (dict/list auto-converted to JSON)

#### `flush()`
Flush all pending spans to Langfuse.

**Note:** Called automatically at trace end, but can be called manually.

## Best Practices

1. **Use Langfuse Attribute Namespace**: Prefix attributes with `langfuse.*` for proper mapping
2. **Set Trace-Level Attributes Early**: User ID, session ID, tags should be on root span
3. **Use Appropriate Span Types**: `generation`, `retriever`, `guard` for proper Langfuse mapping
4. **Flush at Trace End**: Always call `flush()` after completing a trace
5. **Handle Errors Gracefully**: Don't let observability errors break application flow
6. **Monitor Export Errors**: Check logs for failed exports and adjust configuration

## Support

For issues or questions:
- Check Langfuse documentation: https://langfuse.com/docs
- OpenTelemetry Python docs: https://opentelemetry.io/docs/instrumentation/python/
- Langfuse OTLP integration: https://langfuse.com/docs/observability/opentelemetry
