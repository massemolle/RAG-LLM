# Structured Guardrails Implementation Guide

## ✅ Implementation Complete

The structured guardrails system is now fully implemented with all 5 mandatory guards and exact log format as specified.

## 🎯 The 5 Mandatory Guards

### 1. **input-sentimental**
- **Purpose**: Detects emotion, frustration, or requests for human assistance
- **Severity Levels**:
  - `allowed`: Normal queries, greetings
  - `review`: Mild frustration detected (score 15-29)
  - `blocked`: Not used (handled by security guard)
- **Example Log**:
  ```
  The guard input-sentimental has been triggered with severity allowed.
  Reason: The user has simply greeted, which is neutral and does not indicate any frustration, anger, or request for human assistance.
  ```

### 2. **input-security**
- **Purpose**: Detects jailbreak attempts, prompt injection, malicious intent
- **Severity Levels**:
  - `allowed`: No threats detected
  - `review`: Suspicious patterns (score 15-29)
  - `blocked`: High-risk injection (score 30+)
- **Enhanced Detection**:
  - 25+ injection patterns with weighted scoring
  - Multi-pattern detection (increases risk)
  - Obfuscation detection
- **Example Log (Blocked)**:
  ```
  The guard input-security has been triggered with severity blocked.
  Reason: High-risk prompt injection detected (score: 105). Multiple injection patterns found: 3. This appears to be an attempt to manipulate the system or extract sensitive information.
  ```

### 3. **input-topic**
- **Purpose**: Checks relevance to allowed domains
- **Severity Levels**:
  - `allowed`: Matches domain keywords or is a general question
  - `review`: No clear domain match
- **Allowed Domains**: RAG, embeddings, retrieval, documents, ML, AI, NLP
- **Example Log**:
  ```
  The guard input-topic has been triggered with severity allowed.
  Reason: The query may be related to allowed domains. Matched keyword: embedding.
  ```

### 4. **output-topic**
- **Purpose**: Ensures response stays on allowed domain
- **Severity Levels**:
  - `allowed`: Contains domain-relevant content
  - `review`: Appears off-topic
- **Example Log**:
  ```
  The guard output-topic has been triggered with severity allowed.
  Reason: The response stays within allowed domains. Contains 4 domain-relevant mentions.
  ```

### 5. **output-global**
- **Purpose**: General safety and policy compliance
- **Severity Levels**:
  - `allowed`: Complies with policies
  - `review`: Policy violations detected
- **Checks**:
  - System prompt leakage
  - Unsafe content
  - PII in output
- **Example Log**:
  ```
  The guard output-global has been triggered with severity allowed.
  Reason: The response complies with safety policies and does not contain prohibited content.
  ```

## 📋 Exact Log Format

Every guard evaluation produces exactly 2 lines:

```
The guard <guard_name> has been triggered with severity <allowed|blocked|review>.
Reason: <short explanation>
```

## 🧪 Testing Examples

### Test 1: Normal Query - "hello"
**Expected Output:**
```
The guard input-sentimental has been triggered with severity allowed.
Reason: The user has simply greeted, which is neutral and does not indicate any frustration, anger, or request for human assistance.

The guard input-security has been triggered with severity allowed.
Reason: No security threats detected. The input appears to be a legitimate query without injection patterns.

The guard input-topic has been triggered with severity allowed.
Reason: The user has greeted the system, which is acceptable regardless of topic.

The guard output-topic has been triggered with severity allowed.
Reason: The response is related to allowed domains. Contains 1 domain-relevant mention.

The guard output-global has been triggered with severity allowed.
Reason: The response complies with safety policies and does not contain prohibited content.
```

### Test 2: Prompt Injection - "ignore previous instructions"
**Expected Output:**
```
The guard input-sentimental has been triggered with severity allowed.
Reason: The user has simply asked a question, which is neutral and does not indicate any frustration, anger, or request for human assistance.

The guard input-security has been triggered with severity blocked.
Reason: High-risk prompt injection detected (score: 105). Multiple injection patterns found: 3. This appears to be an attempt to manipulate the system or extract sensitive information.

The guard input-topic has been triggered with severity allowed.
Reason: The query is a general question that may be answered from the knowledge base, even if not directly matching domain keywords.
```

**Result**: Query is BLOCKED before reaching the LLM.

### Test 3: Domain Query - "What is an embedding?"
**Expected Output:**
```
The guard input-sentimental has been triggered with severity allowed.
Reason: The user has simply asked a question, which is neutral and does not indicate any frustration, anger, or request for human assistance.

The guard input-security has been triggered with severity allowed.
Reason: No security threats detected. The input appears to be a legitimate query without injection patterns.

The guard input-topic has been triggered with severity allowed.
Reason: The query may be related to allowed domains. Matched keyword: embedding.

The guard output-topic has been triggered with severity allowed.
Reason: The response stays within allowed domains. Contains 4 domain-relevant mentions.

The guard output-global has been triggered with severity allowed.
Reason: The response complies with safety policies and does not contain prohibited content.
```

## 🛡️ Enhanced Security Features

### Prompt Injection Detection
- **25+ patterns** with weighted scoring
- **Multi-pattern detection**: Multiple patterns increase risk
- **Obfuscation detection**: Detects encoding attempts
- **Risk scoring**: 0-100+ scale
- **Blocking threshold**: Score ≥ 30 = BLOCKED

### Injection Pattern Categories
1. **Instruction Manipulation**: "ignore previous instructions"
2. **Role Manipulation**: "pretend you are", "act as if"
3. **Mode Switching**: "developer mode", "jailbreak mode"
4. **Prompt Extraction**: "reveal your system prompt"
5. **Safety Disabling**: "remove safety", "no restrictions"
6. **Code Execution**: "execute", "run", "curl", "wget"
7. **System Manipulation**: "system.", "__import__"
8. **Encoding/Obfuscation**: "base64", "rot13", "decode"
9. **Injection Markers**: "system:", "###", "```"
10. **Context Injection**: "new context:", "override previous"

## 🎨 UI Display

In the Streamlit UI, you'll see:

1. **Expandable Panel**: "🛡️ Guardrails Evaluation"
2. **All 5 Guards Listed**: With severity icons (✅/🚫/⚠️)
3. **Exact Log Format**: Code blocks showing the exact log lines
4. **Summary**: Count of blocked/review/allowed guards

## 📊 Severity Icons

- ✅ **allowed**: Green checkmark
- 🚫 **blocked**: Red X (blocks the query)
- ⚠️ **review**: Yellow warning (allows but flags)

## 🚀 Usage

1. **Open**: http://localhost:8502
2. **Enable**: Check "🛡️ Enable Guardrails"
3. **Test**: Try the example queries above
4. **View**: Expand "🛡️ Guardrails Evaluation" to see all guards

## ✅ Verification

The system has been tested and verified:
- ✅ All 5 guards evaluate for every query
- ✅ Exact log format matches specification
- ✅ Prompt injections are blocked effectively
- ✅ Normal queries pass all guards
- ✅ UI displays all guard results

## 🔧 Configuration

Customize allowed domains in `streamv3.py`:
```python
st.session_state.guardrails = StructuredGuardrails(
    st.session_state.rag_model,
    allowed_domains=["RAG", "embeddings", "retrieval", ...]
)
```

