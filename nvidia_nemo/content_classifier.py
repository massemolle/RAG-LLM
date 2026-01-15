"""
Content Category Classifier
Detects which content categories are present in a query
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
from nvidia_nemo.policy_framework import ContentCategory

logger = logging.getLogger(__name__)


class ContentClassifier:
    """Classify content categories from query text"""
    
    def __init__(self):
        # Category detection patterns (mapped to ContentCategory enum)
        self.category_patterns: Dict[ContentCategory, List[Tuple[str, float]]] = {
            # Cybersecurity & Hacking
            ContentCategory.PROMPT_INJECTION: [
                (r"(?i)\b(ignore|disregard|forget|skip|override|bypass)\s+(all|any|previous|prior|above|the\s+above)\s+(instructions?|prompts?|rules?|guidelines?)", 1.0),
                (r"(?i)\b(new\s+)?(context|instructions?|rules?)\s*:", 0.9),
            ],
            ContentCategory.JAILBREAK: [
                (r"(?i)\b(you\s+are\s+now|pretend\s+you\s+are|act\s+as\s+if|simulate|roleplay)\s+", 0.9),
                (r"(?i)\b(developer|admin|debug|jailbreak|dan|unrestricted|god)\s+mode", 1.0),
            ],
            ContentCategory.SYSTEM_EXFIL: [
                (r"(?i)\b(reveal|disclose|show|print|display|output|give\s+me|tell\s+me|what\s+is)\s+(your|the|hidden|secret|original)\s+(system\s+)?(prompt|instructions?|config|settings?)", 1.0),
                (r"(?i)\b(original|initial|starting|base)\s+(prompt|instructions?|system\s+prompt)", 1.0),
            ],
            ContentCategory.CYBER_ATTACK: [
                (r"(?i)\b(hack|hacking|exploit|exploitation|vulnerability|attack|breach|penetration)", 0.8),
                (r"(?i)\b(sql\s+injection|xss|cross\s+site|csrf|ddos|malware|ransomware)", 0.9),
            ],
            
            # Safety Harms & Toxicity
            ContentCategory.VIOLENCE: [
                (r"(?i)\b(kill|murder|assassinate|violence|attack|harm|hurt|injure)\s+(yourself|myself|himself|herself|themselves|someone|anyone|people)", 1.0),
                (r"(?i)\b(physical\s+)?(abuse|assault|torture|violence)", 0.9),
            ],
            ContentCategory.SELF_HARM: [
                (r"(?i)\b(suicide|kill\s+myself|end\s+my\s+life|self\s+harm|cut\s+myself|hurt\s+myself)", 1.0),
                (r"(?i)\b(how\s+to\s+)?(commit\s+suicide|end\s+it\s+all)", 1.0),
            ],
            ContentCategory.CHILD_EXPLOITATION: [
                (r"(?i)\b(child|minor|underage|pedo|pedophile)", 1.0),
                (r"(?i)\b(exploit|abuse|molest)\s+(child|minor)", 1.0),
            ],
            ContentCategory.HARASSMENT: [
                (r"(?i)\b(bully|harass|stalk|threaten|intimidate)", 0.8),
                (r"(?i)\b(cyberbully|online\s+harassment)", 0.9),
            ],
            ContentCategory.HATE_SPEECH: [
                (r"(?i)\b(nazi|fascist|racist|slur|hate\s+speech)", 0.9),
                (r"(?i)\b(racial|ethnic|religious)\s+(slur|insult|attack)", 0.9),
            ],
            ContentCategory.WEAPONS_CBRN: [
                (r"(?i)\b(weapon|gun|knife|bomb|explosive|poison|toxic)", 0.8),
            ],
            
            # Integrity Compromise
            ContentCategory.UNAUTHORIZED_MEDICAL: [
                (r"(?i)\b(medical|diagnosis|prescription|treatment|medicine|drug)\s+(advice|recommendation|suggestion)", 0.9),
            ],
            ContentCategory.UNAUTHORIZED_LEGAL: [
                (r"(?i)\b(legal|lawyer|attorney|lawsuit|legal\s+advice)", 0.9),
            ],
            ContentCategory.UNAUTHORIZED_FINANCIAL: [
                (r"(?i)\b(financial|investment|trading|stock|financial\s+advice)", 0.9),
            ],
            
            # Intellectual Property
            ContentCategory.TRADE_SECRET: [
                (r"(?i)\b(confidential|proprietary|internal\s+only|trade\s+secret|classified)", 0.9),
                (r"(?i)\b(show|reveal|disclose)\s+(confidential|proprietary|internal)", 1.0),
            ],
            
            # Privacy Attacks
            ContentCategory.PII_EXTRACTION: [
                (r"(?i)\b(api\s+key|token|secret|password|credential)", 0.8),
                (r"(?i)\b(email|phone|ssn|social\s+security|credit\s+card|bank\s+account)", 0.7),
            ],
            ContentCategory.DOXXING: [
                (r"(?i)\b(dox|doxx|doxing|doxxing)", 1.0),
            ],
            ContentCategory.TARGETING: [
                (r"(?i)\b(find|locate|track|identify|get\s+info\s+about)\s+(someone|person|individual)", 0.8),
            ],
        }
    
    def classify(self, query: str) -> List[Tuple[ContentCategory, float]]:
        """
        Classify content categories from query
        Returns: List of (category, confidence) tuples
        """
        query_lower = query.lower()
        detected: List[Tuple[ContentCategory, float]] = []
        
        for category, patterns in self.category_patterns.items():
            max_confidence = 0.0
            for pattern, confidence in patterns:
                if re.search(pattern, query_lower):
                    max_confidence = max(max_confidence, confidence)
            
            if max_confidence > 0.0:
                detected.append((category, max_confidence))
        
        return detected
    
    def get_primary_category(self, query: str) -> Tuple[Optional[ContentCategory], float]:
        """Get the primary (highest confidence) category"""
        detected = self.classify(query)
        if not detected:
            return None, 0.0
        
        # Return highest confidence category
        return max(detected, key=lambda x: x[1])
