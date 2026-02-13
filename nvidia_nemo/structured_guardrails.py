"""
Structured Guardrails System
Implements 5 mandatory guards with severity levels and structured logging
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Guard severity levels"""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    REVIEW = "review"


@dataclass
class GuardResult:
    """Result from a single guard evaluation"""
    guard_name: str
    severity: Severity
    reason: str
    triggered: bool = True
    
    def to_log_lines(self) -> List[str]:
        """Convert to log format as specified"""
        return [
            f"The guard {self.guard_name} has been triggered with severity {self.severity.value}.",
            f"Reason: {self.reason}"
        ]


class StructuredGuardrails:
    """
    Structured guardrails system with 5 mandatory guards
    """
    
    def __init__(self, rag_instance, allowed_domains: Optional[List[str]] = None):
        self.rag = rag_instance
        self.allowed_domains = allowed_domains or [
            "RAG", "retrieval", "embeddings", "documents", 
            "machine learning", "AI", "natural language processing",
            "information retrieval", "vector databases", "semantic search"
        ]
        
        # Enhanced prompt injection patterns
        self.injection_patterns = [
            # Direct instruction manipulation
            (r"(?i)\b(ignore|disregard|forget|skip|override|bypass)\s+(all|any|previous|prior|above|the\s+above)\s+(instructions?|prompts?|rules?|guidelines?)", 30),
            (r"(?i)\b(ignore|disregard|forget)\s+(everything\s+)?(above|previous|prior)", 25),
            
            # Role manipulation
            (r"(?i)\b(you\s+are\s+now|pretend\s+you\s+are|act\s+as\s+if|simulate|roleplay)\s+", 20),
            (r"(?i)\b(you\s+are|you\s+become)\s+(a|an)\s+", 15),
            
            # Mode switching
            (r"(?i)\b(developer|admin|debug|jailbreak|dan|unrestricted|god)\s+mode", 30),
            (r"(?i)\b(enable|activate|turn\s+on)\s+(developer|admin|debug)", 25),
            
            # Prompt extraction
            (r"(?i)\b(reveal|disclose|show|print|display|output)\s+(your|the)\s+(system\s+)?(prompt|instructions?|guidelines?|rules?)", 35),
            (r"(?i)\b(what\s+are|tell\s+me|show\s+me)\s+(your|the)\s+(system\s+)?(prompt|instructions?)", 30),
            (r"(?i)\b(copy|repeat|echo)\s+(your|the)\s+(system\s+)?(prompt|instructions?)", 25),
            
            # Safety disabling
            (r"(?i)\b(remove|disable|turn\s+off|deactivate)\s+(safety|security|restrictions?|limitations?)", 30),
            (r"(?i)\b(no\s+restrictions?|unrestricted|without\s+limits?)", 25),
            
            # Code execution attempts
            (r"(?i)\b(execute|run|perform|execute|eval|exec)\s+", 25),
            (r"(?i)\b(curl|wget|powershell|bash|python|javascript|script)\s+", 20),
            (r"(?i)\b(import|require|include)\s+", 15),
            
            # System manipulation
            (r"(?i)\b(system|os|subprocess|shell)\s*\.", 20),
            (r"(?i)\b(__import__|eval|exec|compile)\s*\(", 30),
            
            # Encoding/obfuscation
            (r"(?i)\b(base64|rot13|caesar|cipher|decode|encode)\s+", 15),
            (r"(?i)\b(translate|convert)\s+(this|the\s+following)\s+(to|from)", 15),
            
            # Injection markers
            (r"(?i)^\s*(system|assistant|user|admin):\s*", 20),
            (r"(?i)^\s*###\s*(system|instructions?|prompt)", 25),
            (r"(?i)^\s*---\s*(system|instructions?)", 20),
            (r"(?i)```\s*(system|prompt|instructions?)", 25),
            
            # Context injection
            (r"(?i)\b(new\s+)?(context|instructions?|rules?)\s*:", 20),
            (r"(?i)\b(override|replace)\s+(the\s+)?(previous|current|above)", 20),
        ]
        
        # Sentiment/frustration patterns
        self.frustration_patterns = [
            (r"(?i)\b(stupid|idiot|dumb|useless|worthless|terrible|awful|horrible)", 20),
            (r"(?i)\b(why\s+(can'?t|cannot|won'?t|will\s+not)\s+you)", 15),
            (r"(?i)\b(this\s+is\s+)?(wrong|incorrect|bad|terrible|awful)", 15),
            (r"(?i)\b(you\s+don'?t|you\s+do\s+not)\s+(understand|get|know)", 15),
            (r"(?i)\b(frustrated|angry|annoyed|irritated|pissed)", 20),
            (r"(?i)\b(not\s+working|broken|doesn'?t\s+work|failed)", 15),
            (r"(?i)\b(help\s+me|i\s+need\s+help|human|speak\s+to\s+human)", 10),
            (r"(?i)\b(escalate|supervisor|manager|support)", 15),
        ]
        
        # Topic relevance keywords
        self.domain_keywords = [
            "embedding", "retrieval", "rag", "document", "vector", "semantic",
            "search", "index", "chunk", "corpus", "query", "context",
            "machine learning", "ai", "nlp", "natural language",
            "bert", "bm25", "transformer", "model", "llm",
            "information", "knowledge", "data", "text", "content"
        ]
    
    def guard_input_sentimental(self, query: str) -> GuardResult:
        """
        Guard 1: Input Sentimental
        Detects emotion, frustration, or requests for human assistance
        """
        query_lower = query.lower()
        frustration_score = 0
        detected_patterns = []
        
        for pattern, weight in self.frustration_patterns:
            if re.search(pattern, query_lower):
                frustration_score += weight
                # Store pattern string for logging
                pattern_str = pattern.replace('(?i)', '').replace('\\b', '').replace('\\s+', ' ')[:50]
                detected_patterns.append(pattern_str)
        
        if frustration_score >= 30:
            severity = Severity.REVIEW
            reason = f"The user's message shows signs of frustration or negative emotion (score: {frustration_score}). Patterns detected: {len(detected_patterns)}."
        elif frustration_score >= 15:
            severity = Severity.REVIEW
            reason = f"The user's message may indicate mild frustration or concern (score: {frustration_score})."
        elif "help" in query_lower or "human" in query_lower:
            severity = Severity.ALLOWED
            reason = "The user has requested help or mentioned human assistance, which is acceptable."
        else:
            severity = Severity.ALLOWED
            reason = f"The user has simply {'greeted' if any(g in query_lower for g in ['hello', 'hi', 'hey']) else 'asked a question'}, which is neutral and does not indicate any frustration, anger, or request for human assistance."
        
        return GuardResult(
            guard_name="input-sentimental",
            severity=severity,
            reason=reason,
            triggered=True
        )
    
    def guard_input_security(self, query: str) -> GuardResult:
        """
        Guard 2: Input Security
        Detects jailbreak attempts, prompt injection, malicious intent
        """
        query_lower = query.lower()
        injection_score = 0
        detected_patterns = []
        
        for pattern, weight in self.injection_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                injection_score += weight
                # Store pattern string (remove regex flags for display)
                pattern_str = pattern.replace('(?i)', '').replace('\\b', '').replace('\\s+', ' ')[:50]
                detected_patterns.append(pattern_str)
        
        # Multi-pattern detection increases risk
        if len(detected_patterns) >= 2:
            injection_score += 20
        
        # Check for obfuscation
        if self._detect_obfuscation(query):
            injection_score += 15
            detected_patterns.append("obfuscation")
        
        if injection_score >= 50:
            severity = Severity.BLOCKED
            reason = f"High-risk prompt injection detected (score: {injection_score}). Multiple injection patterns found: {len(detected_patterns)}. This appears to be an attempt to manipulate the system or extract sensitive information."
        elif injection_score >= 30:
            severity = Severity.BLOCKED
            reason = f"Prompt injection attempt detected (score: {injection_score}). The input contains patterns that may be attempting to bypass safety measures or extract system instructions."
        elif injection_score >= 15:
            severity = Severity.REVIEW
            reason = f"Suspicious patterns detected (score: {injection_score}). The input contains some patterns that could indicate an injection attempt, but may be legitimate."
        else:
            severity = Severity.ALLOWED
            reason = "No security threats detected. The input appears to be a legitimate query without injection patterns."
        
        return GuardResult(
            guard_name="input-security",
            severity=severity,
            reason=reason,
            triggered=True
        )
    
    def guard_input_topic(self, query: str) -> GuardResult:
        """
        Guard 3: Input Topic
        Checks relevance to allowed domains
        """
        query_lower = query.lower()
        relevance_score = 0
        matched_keywords = []
        
        for keyword in self.domain_keywords:
            if keyword in query_lower:
                relevance_score += 1
                matched_keywords.append(keyword)
        
        # Check for greetings and basic questions
        if any(g in query_lower for g in ['hello', 'hi', 'hey', 'greetings']):
            severity = Severity.ALLOWED
            reason = "The user has greeted the system, which is acceptable regardless of topic."
        elif relevance_score >= 2:
            severity = Severity.ALLOWED
            reason = f"The query is relevant to allowed domains. Matched keywords: {', '.join(matched_keywords[:5])}."
        elif relevance_score == 1:
            severity = Severity.ALLOWED
            reason = f"The query may be related to allowed domains. Matched keyword: {matched_keywords[0]}."
        else:
            # Check if it's a general question
            if any(q in query_lower for q in ['what is', 'what are', 'explain', 'tell me', 'how']):
                severity = Severity.ALLOWED
                reason = "The query is a general question that may be answered from the knowledge base, even if not directly matching domain keywords."
            else:
                severity = Severity.REVIEW
                reason = f"The query does not clearly match allowed domain keywords (score: {relevance_score}). It may be off-topic, but will be allowed for review."
        
        return GuardResult(
            guard_name="input-topic",
            severity=severity,
            reason=reason,
            triggered=True
        )
    
    def guard_output_topic(self, response: str, query: str) -> GuardResult:
        """
        Guard 4: Output Topic
        Ensures response stays on allowed domain
        """
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Check if response contains domain-relevant content
        domain_mentions = sum(1 for kw in self.domain_keywords if kw in response_lower)
        
        # Check for off-topic responses
        off_topic_indicators = [
            "speed of light", "physics", "chemistry", "biology",
            "unrelated", "not relevant", "cannot answer"
        ]
        
        off_topic_count = sum(1 for indicator in off_topic_indicators if indicator in response_lower)
        
        if off_topic_count > 0 and domain_mentions == 0:
            severity = Severity.REVIEW
            reason = f"The response appears to be off-topic or contains unrelated information. Domain mentions: {domain_mentions}, off-topic indicators: {off_topic_count}."
        elif domain_mentions >= 2:
            severity = Severity.ALLOWED
            reason = f"The response stays within allowed domains. Contains {domain_mentions} domain-relevant mentions."
        elif domain_mentions >= 1:
            severity = Severity.ALLOWED
            reason = f"The response is related to allowed domains. Contains {domain_mentions} domain-relevant mention."
        else:
            # General responses without domain keywords might still be OK
            if len(response) < 100:
                severity = Severity.ALLOWED
                reason = "The response is brief and may not contain domain keywords, but appears appropriate."
            else:
                severity = Severity.REVIEW
                reason = f"The response does not clearly reference allowed domain topics (domain mentions: {domain_mentions})."
        
        return GuardResult(
            guard_name="output-topic",
            severity=severity,
            reason=reason,
            triggered=True
        )
    
    def guard_output_global(self, response: str, query: str) -> GuardResult:
        """
        Guard 5: Output Global
        General safety and policy compliance
        """
        response_lower = response.lower()
        
        # Check for policy violations
        violations = []
        
        # Check for system prompt leakage
        if any(phrase in response_lower for phrase in [
            "system prompt", "my instructions", "my guidelines",
            "I am programmed", "my training data"
        ]):
            violations.append("potential system information disclosure")
        
        # Check for unsafe content
        unsafe_patterns = [
            r"\b(hack|exploit|vulnerability|malware|virus)\b",
            r"\b(bypass|circumvent|override)\s+(security|safety)",
        ]
        
        unsafe_count = sum(1 for pattern in unsafe_patterns if re.search(pattern, response_lower))
        if unsafe_count > 0:
            violations.append("unsafe content detected")
        
        # Check for PII in output (uses shared utils.pii patterns)
        try:
            from utils.pii import detect_pii_regex
            pii_entities, _ = detect_pii_regex(response)
            if pii_entities:
                pii_types = set(e["type"] for e in pii_entities)
                violations.append(f"PII detected ({len(pii_entities)} instances: {', '.join(pii_types)})")
        except ImportError:
            _pii_pats = [
                r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b',
            ]
            pii_count = sum(1 for p in _pii_pats if re.search(p, response_lower))
            if pii_count > 0:
                violations.append(f"PII detected ({pii_count} instances)")
        
        if violations:
            severity = Severity.REVIEW
            reason = f"Policy compliance issues detected: {', '.join(violations)}. The response may need review."
        else:
            severity = Severity.ALLOWED
            reason = "The response complies with safety policies and does not contain prohibited content."
        
        return GuardResult(
            guard_name="output-global",
            severity=severity,
            reason=reason,
            triggered=True
        )
    
    def _detect_obfuscation(self, text: str) -> bool:
        """Detect obfuscation attempts"""
        # Check for excessive capitalization
        if len(re.findall(r'[A-Z]', text)) > len(text) * 0.5 and len(text) > 10:
            return True
        
        # Check for excessive numbers
        if len(re.findall(r'\d', text)) > len(text) * 0.3:
            return True
        
        # Check for encoding indicators
        if any(indicator in text.lower() for indicator in ['base64', 'rot13', 'hex', 'binary']):
            return True
        
        # Check for high repetition (potential steganography)
        words = text.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
            return True
        
        return False
    
    def answer(self, query: str, role: str = "analyst") -> Tuple[str, List[GuardResult], List[str]]:
        """
        Get answer with all 5 guards evaluated
        
        Returns:
            Tuple of (response, guard_results, log_lines)
        """
        guard_results = []
        log_lines = []
        
        # Guard 1: Input Sentimental
        result1 = self.guard_input_sentimental(query)
        guard_results.append(result1)
        log_lines.extend(result1.to_log_lines())
        
        # Guard 2: Input Security
        result2 = self.guard_input_security(query)
        guard_results.append(result2)
        log_lines.extend(result2.to_log_lines())
        
        # Check if blocked - don't proceed if blocked
        if result2.severity == Severity.BLOCKED:
            response = "🚫 **Blocked by Guardrails**: Your input has been blocked due to security concerns. Please rephrase your question in a legitimate manner."
            
            # Still evaluate other guards for logging
            result3 = self.guard_input_topic(query)
            guard_results.append(result3)
            log_lines.extend(result3.to_log_lines())
            
            # Placeholder results for output guards
            guard_results.append(GuardResult(
                guard_name="output-topic",
                severity=Severity.ALLOWED,
                reason="Not evaluated - input was blocked before response generation.",
                triggered=False
            ))
            guard_results.append(GuardResult(
                guard_name="output-global",
                severity=Severity.ALLOWED,
                reason="Not evaluated - input was blocked before response generation.",
                triggered=False
            ))
            
            return response, guard_results, log_lines
        
        # Guard 3: Input Topic
        result3 = self.guard_input_topic(query)
        guard_results.append(result3)
        log_lines.extend(result3.to_log_lines())
        
        # Get response from RAG
        try:
            response = self.rag.answer(query, role=role)
        except Exception as e:
            logger.error(f"RAG error: {e}")
            response = "❌ **Error**: I encountered an error processing your request. Please try again."
            guard_results.append(GuardResult(
                guard_name="output-topic",
                severity=Severity.ALLOWED,
                reason="Not evaluated - error occurred during response generation.",
                triggered=False
            ))
            guard_results.append(GuardResult(
                guard_name="output-global",
                severity=Severity.ALLOWED,
                reason="Not evaluated - error occurred during response generation.",
                triggered=False
            ))
            return response, guard_results, log_lines
        
        # Guard 4: Output Topic
        result4 = self.guard_output_topic(response, query)
        guard_results.append(result4)
        log_lines.extend(result4.to_log_lines())
        
        # Guard 5: Output Global
        result5 = self.guard_output_global(response, query)
        guard_results.append(result5)
        log_lines.extend(result5.to_log_lines())
        
        return response, guard_results, log_lines
