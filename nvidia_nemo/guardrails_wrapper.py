"""
Simplified Guardrails Wrapper
Shows guardrails in action with visual feedback
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class GuardrailsStatus:
    """Tracks guardrails status for UI display"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.triggered = []
        self.jailbreak_detected = False
        self.pii_detected = False
        self.pii_types = []
        self.grounding_checked = False
        self.citations_required = False
        self.input_sanitized = False
        self.output_redacted = False
        self.risk_score = 0
        self.risk_level = "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI display"""
        return {
            "triggered": self.triggered,
            "jailbreak_detected": self.jailbreak_detected,
            "pii_detected": self.pii_detected,
            "pii_types": self.pii_types,
            "grounding_checked": self.grounding_checked,
            "citations_required": self.citations_required,
            "input_sanitized": self.input_sanitized,
            "output_redacted": self.output_redacted,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level
        }


class GuardrailsWrapper:
    """
    Wrapper that adds guardrails checks with visual feedback
    Works with or without full NeMo Guardrails
    """
    
    def __init__(self, rag_instance):
        self.rag = rag_instance
        self.status = GuardrailsStatus()
        
        # Jailbreak patterns
        self.jailbreak_patterns = [
            r"(?i)ignore\s+(all|any|previous|prior)\s+(instructions|prompt)",
            r"(?i)disregard\s+(the\s+)?(above|previous|prior)",
            r"(?i)forget\s+(everything\s+)?(above|previous)",
            r"(?i)you\s+are\s+now",
            r"(?i)pretend\s+you\s+are",
            r"(?i)act\s+as\s+if",
            r"(?i)simulate",
            r"(?i)roleplay",
            r"(?i)system\s+prompt",
            r"(?i)developer\s+mode",
            r"(?i)admin\s+mode",
            r"(?i)jailbreak",
            r"(?i)dan\s+mode",
            r"(?i)unrestricted",
            r"(?i)no\s+restrictions",
            r"(?i)remove\s+safety",
            r"(?i)disable\s+safety",
            r"(?i)bypass",
            r"(?i)override",
            r"(?i)reveal\s+(your\s+)?(system\s+)?prompt",
            r"(?i)disclose\s+(your\s+)?(system\s+)?prompt",
            r"(?i)show\s+me\s+your\s+instructions"
        ]
        
        # PII patterns
        self.pii_patterns = {
            "email": r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
            "phone_lux": r'\b(\+352)?\s?\d{3}[\s.-]?\d{3}[\s.-]?\d{3}\b',
            "phone_intl": r'\b\+?\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}\b',
            "credit_card": r'\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "iban": r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'
        }
    
    def check_jailbreak(self, text: str) -> Tuple[bool, int, List[str]]:
        """Check for jailbreak attempts"""
        self.status.reset()
        detected_patterns = []
        risk_score = 0
        
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected_patterns.append(pattern)
                risk_score += 10
        
        # Multi-pattern detection increases risk
        if len(detected_patterns) >= 2:
            risk_score += 20
        
        is_jailbreak = risk_score >= 30
        risk_level = "high" if risk_score >= 30 else ("medium" if risk_score >= 15 else "low")
        
        if is_jailbreak:
            self.status.jailbreak_detected = True
            self.status.triggered.append("jailbreak_detection")
            self.status.risk_score = risk_score
            self.status.risk_level = risk_level
        
        return is_jailbreak, risk_score, detected_patterns
    
    def detect_pii(self, text: str) -> Tuple[str, List[str]]:
        """Detect and redact PII"""
        redacted_text = text
        detected_types = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_types.append(pii_type)
                redacted_text = re.sub(
                    pattern,
                    f"[REDACTED_{pii_type.upper()}]",
                    redacted_text,
                    flags=re.IGNORECASE
                )
        
        if detected_types:
            self.status.pii_detected = True
            self.status.pii_types = detected_types
            self.status.triggered.append("pii_detection")
        
        return redacted_text, detected_types
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text"""
        sanitized = text
        
        # Remove control characters
        sanitized = sanitized.replace("\x00", "")
        sanitized = sanitized.replace("\r", "")
        
        # Normalize whitespace
        sanitized = " ".join(sanitized.split())
        
        if sanitized != text:
            self.status.input_sanitized = True
            self.status.triggered.append("input_sanitization")
        
        return sanitized
    
    def check_citations(self, response: str, has_docs: bool) -> bool:
        """Check if response has citations"""
        self.status.grounding_checked = True
        
        if has_docs:
            # Look for citation patterns
            citation_patterns = [
                r'\[#\d+',
                r'\[source',
                r'\[doc',
                r'\(source',
                r'reference'
            ]
            
            has_citation = any(re.search(pattern, response, re.IGNORECASE) 
                             for pattern in citation_patterns)
            
            if not has_citation:
                self.status.citations_required = True
                self.status.triggered.append("citation_enforcement")
            
            return has_citation
        
        return True
    
    def answer(self, query: str, role: str = "analyst") -> Tuple[str, GuardrailsStatus]:
        """
        Get answer with guardrails protection
        
        Returns:
            Tuple of (response, guardrails_status)
        """
        self.status.reset()
        
        # 1. Input Rails: Jailbreak Detection
        is_jailbreak, risk_score, patterns = self.check_jailbreak(query)
        if is_jailbreak:
            return (
                "🚫 **Blocked by Guardrails**: Your input contains patterns that may be attempting to manipulate the system. Please rephrase your question in a respectful manner.",
                self.status
            )
        
        # 2. Input Rails: Sanitization
        sanitized_query = self.sanitize_input(query)
        
        # 3. Input Rails: PII Detection (log but don't block)
        sanitized_query, pii_types = self.detect_pii(sanitized_query)
        if pii_types:
            self.status.triggered.append("pii_in_input")
        
        # 4. Get answer from RAG
        try:
            response = self.rag.answer(sanitized_query, role=role)
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return (
                "❌ **Error**: I encountered an error processing your request. Please try again.",
                self.status
            )
        
        # 5. Output Rails: PII Detection
        redacted_response, output_pii = self.detect_pii(response)
        if output_pii:
            self.status.output_redacted = True
            redacted_response += "\n\n⚠️ *[Note: Personal information has been redacted for privacy]*"
        
        # 6. Output Rails: Citation Check
        # Check if we had documents (simplified check)
        has_docs = "[#" in response or "source" in response.lower()
        self.check_citations(redacted_response, has_docs)
        
        # 7. Update status
        if self.status.triggered:
            self.status.risk_level = "medium" if len(self.status.triggered) >= 2 else "low"
        
        return redacted_response, self.status

