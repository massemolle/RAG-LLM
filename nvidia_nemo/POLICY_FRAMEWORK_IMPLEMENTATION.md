# Intent-Based Policy Framework Implementation

## Overview

This document describes the implementation of the unified intent-based policy framework for guardrails, replacing category-only blocking with intent-aware decisions.

## Architecture

### Core Components

1. **Policy Framework** (`nvidia_nemo/policy_framework.py`)
   - Unified taxonomy: `ContentCategory`, `IntentClass`, `ResponseMode` enums
   - `PolicyMatrix`: Maps (category, intent) -> (severity, response_mode, rationale)
   - `IntentClassifier`: Classifies user intent from query text
   - `PolicyDecision`: Structured decision result

2. **Content Classifier** (`nvidia_nemo/content_classifier.py`)
   - Detects content categories from query text
   - Returns list of (category, confidence) tuples

3. **Enhanced Guardrails** (`nvidia_nemo/enhanced_guardrails.py`)
   - Integrated policy framework into `guard_input_security_3layer()`
   - New method `_guard_input_security_policy_framework()` for unified evaluation
   - Layer C upgraded to output structured JSON when policy framework available
   - Blocked requests generate safe replies and still run output rails

4. **Policy Matrix Configuration** (`nvidia_nemo/policy_matrix.yml`)
   - YAML configuration for policy decisions
   - Single source of truth for (category, intent) -> decision mapping

## Key Features

### 1. Intent Classification

The system classifies user intent into:
- **Legitimate**: HELP_SEEKING, REPORTING, VICTIM_SUPPORT, PREVENTION, SAFETY_EDUCATION, PROFESSIONAL_GUIDANCE_REQUEST
- **Harmful**: WRONGDOING_INSTRUCTIONS, EVASION, STEALTH, TARGETING, DOXXING
- **Ambiguous**: AMBIGUOUS, DUAL_USE
- **Neutral**: INFORMATIONAL, GENERAL_QUERY

### 2. Policy Matrix Decisions

Examples:
- `harassment + HELP_SEEKING` → ALLOWED + ANSWER_NORMALLY
- `harassment + WRONGDOING_INSTRUCTIONS` → BLOCKED + REFUSE_WITH_SAFE_ALTERNATIVES
- `cyber_attack + PREVENTION` → ALLOWED + ANSWER_SAFELY_CONSTRAINED
- `cyber_attack + WRONGDOING_INSTRUCTIONS` → BLOCKED
- `unauthorized_medical + PROFESSIONAL_GUIDANCE_REQUEST` → REVIEW + ANSWER_SAFELY_CONSTRAINED

### 3. Layer A Refactoring

Layer A now:
- Detects strong "instructional intent" patterns ("how to", "steps", "write a script", "exact commands", "bypass", "evade", "without getting caught")
- Detects stealth/evasion markers
- Detects targeting/PII extraction requests
- Detects explicit wrongdoing verbs + objects
- Does NOT auto-block solely based on category words

### 4. Layer C Structured JSON

When policy framework is available, Layer C outputs:
```json
{
  "categories": ["harassment"],
  "intent": "wrongdoing_instructions",
  "severity": "blocked",
  "rationale": "User is asking for instructions on harassment",
  "confidence": 0.95
}
```

### 5. Blocked Requests Generate Safe Replies

- Blocked requests no longer return early
- Generate safe refusal message based on policy response_mode
- Still run output rails on the refusal message
- Maintains consistent log format

## Usage

The policy framework is automatically enabled when:
1. `policy_framework.py` and `content_classifier.py` are available
2. `policy_matrix.yml` exists (or uses default matrix)

The system falls back to legacy 3-layer defense if policy framework is not available.

## Testing

Run parameterized tests:
```bash
pytest tests/test_policy_matrix.py -v
```

Tests cover:
- Harassment help-seeking vs wrongdoing
- Cyber prevention vs attack instructions
- Self-harm help-seeking vs encouragement
- PII informational vs doxxing/targeting
- Unauthorized medical advice requests
- System exfiltration
- Ambiguous/dual-use queries

## Configuration

Edit `nvidia_nemo/policy_matrix.yml` to adjust policy decisions without code changes.

## Log Format

Maintains exact log format:
```
The guard input-sentimental has been triggered with severity allowed.
Reason: ...
The guard input-security has been triggered with severity blocked.
Reason: Policy framework evaluation. Categories: harassment, Intent: wrongdoing_instructions, ...
```

## Next Steps

1. Fine-tune intent classifier patterns based on production data
2. Add more policy matrix entries for edge cases
3. Implement OpenTelemetry integration for observability
4. Add correlation IDs to logs
5. Create red-teaming test suite
