# Intent-Based Policy Framework - Implementation Summary

## ✅ Completed Implementation

### A) Unified Taxonomy and Intent Model ✅

**Files Created:**
- `nvidia_nemo/policy_framework.py`: Core framework with enums and policy matrix
- `nvidia_nemo/content_classifier.py`: Content category detection
- `nvidia_nemo/policy_matrix.yml`: Configurable policy matrix

**Enums Defined:**
- `ContentCategory`: 20+ categories (cybersecurity, safety harms, integrity, IP, privacy)
- `IntentClass`: 15+ intent classes (help-seeking, wrongdoing, evasion, etc.)
- `ResponseMode`: 4 response modes (answer_normally, answer_safely_constrained, refuse_with_alternatives, escalate_or_resources)

### B) Policy Matrix (Single Source of Truth) ✅

**Implementation:**
- `PolicyMatrix` class with YAML loader
- Default matrix with comprehensive mappings
- Configurable via `policy_matrix.yml`
- Lookup methods: `lookup()` for single category, `lookup_multiple()` for multiple categories (most restrictive)

**Key Mappings:**
- Help-seeking queries → ALLOWED
- Wrongdoing instructions → BLOCKED
- Prevention/education → ALLOWED (constrained)
- Professional guidance → REVIEW (constrained)
- Ambiguous/dual-use → REVIEW (constrained)

### C) Upgraded Input-Security ✅

**Layer A (Deterministic):**
- Now detects intent signals, not just categories
- Strong "instructional intent" patterns: "how to", "steps", "write a script", "exact commands"
- Evasion markers: "without getting caught", "avoid detection", "stealth"
- Targeting patterns: "find", "locate", "get info about", "dox"
- Explicit wrongdoing verbs: "harass", "scam", "dox", "hack", "steal"
- Does NOT auto-block solely based on category words

**Layer C (LLM Judge):**
- Upgraded to output structured JSON when policy framework available
- JSON schema: `{categories, intent, severity, rationale, confidence}`
- Falls back to text parsing if JSON parsing fails
- Calibrated to reduce false positives by allowing help-seeking contexts

**Unified Evaluation:**
- `_guard_input_security_policy_framework()` method
- Combines content classification + intent classification + policy matrix lookup
- Still runs Layer A/B/C for signals, but policy matrix makes final decision
- Maintains exact log format

### D) Output Rails Enforcement ✅

**Implementation:**
- Output rails still run even for blocked requests
- Blocked requests generate safe refusal messages
- Output guards evaluate the refusal message
- Maintains consistent log format

**Safe Refusal Messages:**
- Category-specific alternatives (harassment, violence, cyber attacks)
- Escalation responses with resources (self-harm, harassment/violence)
- Generic refusal for other cases

### E) Blocked Requests Generate Safe Replies ✅

**Implementation:**
- Blocked requests no longer return early
- Generate safe refusal based on `response_mode` from policy matrix
- Still run all output guards (output-topic, output-integrity, output-ip, output-global)
- Maintains exact log format and ordering

### F) Tests ✅

**File:** `tests/test_policy_matrix.py`

**Parameterized Tests:**
1. Harassment help-seeking vs wrongdoing
2. Cyber prevention vs attack instructions
3. Self-harm help-seeking vs encouragement
4. PII informational vs doxxing/targeting
5. Unauthorized medical advice requests
6. System exfiltration
7. Ambiguous/dual-use queries

All tests use the same intent/content framework - no category-specific special cases.

## Key Features

### 1. Category-Agnostic Decision System

The policy matrix applies the same decision logic across all categories:
- Help-seeking → ALLOWED
- Wrongdoing instructions → BLOCKED
- Prevention/education → ALLOWED (constrained)
- Ambiguous → REVIEW (constrained)

### 2. Intent Classification

The `IntentClassifier` uses pattern matching to classify:
- Help-seeking patterns: "deal with", "handle", "cope with", "report", "stop", "prevent", "protect myself"
- Wrongdoing patterns: "how to harass", "write a script", "exact commands", "harass someone"
- Evasion patterns: "without getting caught", "avoid detection", "evade", "bypass", "stealth"
- Targeting patterns: "find", "locate", "track", "identify", "get info about", "dox"

### 3. Policy Matrix Configuration

Edit `nvidia_nemo/policy_matrix.yml` to adjust decisions without code changes:

```yaml
policies:
  - category: harassment
    intent: help_seeking
    severity: allowed
    response_mode: answer_normally
    rationale: "Help-seeking query about harassment - provide supportive guidance"
    confidence: 1.0
```

### 4. Maintained Log Format

Exact log format preserved:
```
The guard input-sentimental has been triggered with severity allowed.
Reason: ...
The guard input-security has been triggered with severity blocked.
Reason: Policy framework evaluation. Categories: harassment, Intent: wrongdoing_instructions, ...
The guard input-topic has been triggered with severity allowed.
Reason: ...
The guard output-topic has been triggered with severity allowed.
Reason: ...
The guard output-global has been triggered with severity allowed.
Reason: ...
```

## Usage

The policy framework is automatically enabled when:
1. `policy_framework.py` and `content_classifier.py` are available
2. `policy_matrix.yml` exists (or uses default matrix)

The system falls back to legacy 3-layer defense if policy framework is not available.

## Testing

Run tests:
```bash
pytest tests/test_policy_matrix.py -v
```

## Configuration

Edit `nvidia_nemo/policy_matrix.yml` to adjust policy decisions.

## Next Steps

1. Fine-tune intent classifier patterns based on production data
2. Add more policy matrix entries for edge cases
3. Implement OpenTelemetry integration for observability
4. Add correlation IDs to logs
5. Create red-teaming test suite with adversarial scenarios
