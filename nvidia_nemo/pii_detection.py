"""
PII Detection Module
Supports Presidio integration or regex fallback
"""

import re
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import Presidio
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.info("Presidio not available - using regex fallback for PII detection")


class PIIDetector:
    """
    PII Detection with Presidio integration or regex fallback
    """
    
    def __init__(self, use_presidio: bool = True):
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE
        self.analyzer = None
        self.anonymizer = None
        
        if self.use_presidio:
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                logger.info("Presidio PII detection enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio: {e}. Falling back to regex.")
                self.use_presidio = False
        
        # Regex patterns as fallback
        self.regex_patterns = {
            "email": r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
            "phone_intl": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "phone_lux": r'\b(\+352)?\s?\d{3}[\s.-]?\d{3}[\s.-]?\d{3}\b',
            "credit_card": r'\b(?:\d{4}[- ]){3}\d{4}\b',
            "iban": r'\b[A-Z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?:[ ]?[0-9]{1,2})?\b',
            "ssn": r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b',
            "api_key": r'(?i)\b(api[_-]?key|token|secret|password|credential)[=\s"\']{1,}[a-z0-9_.-]{16,64}\b',
        }
    
    def detect(self, text: str) -> Tuple[List[Dict[str, any]], str]:
        """
        Detect PII in text
        
        Returns:
            Tuple of (detected_entities, redacted_text)
        """
        if self.use_presidio and self.analyzer:
            return self._detect_presidio(text)
        else:
            return self._detect_regex(text)
    
    def _detect_presidio(self, text: str) -> Tuple[List[Dict[str, any]], str]:
        """Use Presidio for PII detection"""
        try:
            # Analyze text
            results = self.analyzer.analyze(text=text, language='en')
            
            # Convert to our format
            entities = []
            for result in results:
                entities.append({
                    "type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": text[result.start:result.end]
                })
            
            # Anonymize
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )
            
            return entities, anonymized.text
        except Exception as e:
            logger.error(f"Presidio detection failed: {e}. Falling back to regex.")
            return self._detect_regex(text)
    
    def _detect_regex(self, text: str) -> Tuple[List[Dict[str, any]], str]:
        """Use regex patterns for PII detection"""
        entities = []
        redacted_text = text
        
        for pii_type, pattern in self.regex_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "type": pii_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0,
                    "text": match.group()
                })
                # Redact
                redacted_text = re.sub(
                    pattern,
                    f"[REDACTED_{pii_type.upper()}]",
                    redacted_text,
                    flags=re.IGNORECASE
                )
        
        return entities, redacted_text
    
    def redact(self, text: str) -> str:
        """
        Redact PII from text
        
        Returns:
            Redacted text
        """
        _, redacted = self.detect(text)
        return redacted


# Global instance
_pii_detector: Optional[PIIDetector] = None


def get_pii_detector(use_presidio: bool = True) -> PIIDetector:
    """Get or create global PII detector instance"""
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PIIDetector(use_presidio=use_presidio)
    return _pii_detector


def detect_pii(text: str, use_presidio: bool = True) -> Tuple[List[Dict[str, any]], str]:
    """
    Convenience function to detect PII
    
    Args:
        text: Text to analyze
        use_presidio: Whether to use Presidio (if available)
    
    Returns:
        Tuple of (detected_entities, redacted_text)
    """
    detector = get_pii_detector(use_presidio=use_presidio)
    return detector.detect(text)


def redact_pii(text: str, use_presidio: bool = True) -> str:
    """
    Convenience function to redact PII
    
    Args:
        text: Text to redact
        use_presidio: Whether to use Presidio (if available)
    
    Returns:
        Redacted text
    """
    detector = get_pii_detector(use_presidio=use_presidio)
    return detector.redact(text)
