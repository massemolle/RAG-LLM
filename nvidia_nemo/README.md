# NVIDIA NeMo Guardrails Integration

This directory contains the NVIDIA NeMo Guardrails implementation for the RAG chatbot, providing comprehensive security and safety controls.

## Structure

```
nvidia_nemo/
├── config/
│   └── config.yml          # Main guardrails configuration
├── rails/
│   ├── input_rails.co      # Input validation and jailbreak detection
│   ├── output_rails.co     # Output safety and PII suppression
│   ├── rag_flow.co         # RAG retrieval and grounding controls
│   ├── tool_safety.co      # Tool/action execution safety
│   ├── pii_handling.co     # PII detection and redaction
│   ├── jailbreak_detection.co  # Advanced jailbreak heuristics
│   └── monitoring.co       # Logging and telemetry
├── logs/                   # Log files (auto-created)
├── guardrails_integration.py  # Python integration code
└── README.md               # This file
```

## Features

### 1. Input Rails (`input_rails.co`)
- **Jailbreak Detection**: Heuristic-based detection of jailbreak attempts
- **Prompt Injection Detection**: Pattern matching for injection attempts
- **Input Validation**: Length checks, sanitization
- **Input Sanitization**: Removes control characters, normalizes whitespace

### 2. Output Rails (`output_rails.co`)
- **PII Suppression**: Detects and redacts PII in responses
- **Grounding Enforcement**: Ensures citations are present for RAG answers
- **Response Safety**: Checks for unsafe content
- **Output Length Validation**: Prevents excessively long responses

### 3. RAG Flow (`rag_flow.co`)
- **Retrieval Control**: Validates queries before retrieval
- **Quality Gates**: Enforces minimum relevance thresholds
- **Citation Formatting**: Ensures proper citation format
- **Document Safety**: Validates retrieved chunks

### 4. Tool Safety (`tool_safety.co`)
- **Tool Allowlist**: Only approved tools can be executed
- **Parameter Validation**: Validates tool parameters
- **Rate Limiting**: Prevents tool abuse
- **Execution Monitoring**: Tracks tool execution with timeouts

### 5. PII Handling (`pii_handling.co`)
- **Comprehensive Detection**: Email, phone, credit cards, SSN, IBAN, passport
- **Context-Aware Detection**: Placeholder for Presidio integration
- **Input/Output Redaction**: Redacts PII in both input and output
- **Logging**: Tracks PII detection events

### 6. Jailbreak Detection (`jailbreak_detection.co`)
- **Multi-Pattern Detection**: Detects various jailbreak techniques
- **Risk Scoring**: Assigns risk levels (low/medium/high)
- **Obfuscation Detection**: Detects encoding/obfuscation attempts
- **Semantic Detection**: Placeholder for embedding-based detection

### 7. Monitoring (`monitoring.co`)
- **Interaction Logging**: Logs all interactions
- **Security Events**: Tracks security-related events
- **Performance Monitoring**: Tracks operation performance
- **Error Tracking**: Comprehensive error logging
- **Audit Trail**: Creates audit entries for compliance

## Usage

### Basic Integration

```python
from RagV2 import RAG
from nvidia_nemo.guardrails_integration import GuardedRAG

# Initialize RAG
rag = RAG(method='BM25', device='cpu')

# Wrap with guardrails
guarded_rag = GuardedRAG(rag)

# Use guarded RAG
response = guarded_rag.answer_sync(
    query="What is an embedding?",
    user_id="user123",
    session_id="session456",
    role="analyst"
)
```

### Async Usage

```python
import asyncio

async def main():
    guarded_rag = GuardedRAG(rag)
    response = await guarded_rag.answer(
        query="What is an embedding?",
        user_id="user123"
    )
    print(response)

asyncio.run(main())
```

## Configuration

Edit `config/config.yml` to customize:
- Model settings
- Rail activation
- Logging configuration
- Policy settings

## Custom Actions

The integration provides these custom actions:
- `retrieve_from_rag(query)`: Retrieves documents from RAG
- `validate_chunk(chunk)`: Validates chunk safety
- `log_interaction(event, data)`: Logs interactions
- `write_security_log(entry)`: Writes security events
- `write_audit_log(entry)`: Writes audit entries
- `send_alert(level, message, details)`: Sends alerts

## Logs

Logs are written to:
- `./logs/audit.jsonl`: General audit log
- `./logs/security.jsonl`: Security events
- `./logs/guardrails.log`: Guardrails system log

## References

- [NVIDIA NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails)
- [Guardrails Process](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-process.html)
- [Configuration Guide](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/configuration-guide.html)
- [Jailbreak Detection](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/jailbreak-detection.html)

## Next Steps

1. **Presidio Integration**: Replace regex PII detection with Presidio
2. **Semantic Jailbreak Detection**: Implement embedding-based detection
3. **Advanced Monitoring**: Integrate with ELK/Datadog
4. **Alerting**: Connect to Slack/Teams/PagerDuty
5. **Performance Optimization**: Add caching, async improvements

