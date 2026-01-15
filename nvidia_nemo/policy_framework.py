"""
Unified Policy Framework: Intent vs Content Classification
Implements category-agnostic guardrails based on intent classification
"""

import re
import json
import yaml
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# UNIFIED TAXONOMY
# ============================================================================

class ContentCategory(Enum):
    """Content domain/harm categories"""
    # Cybersecurity & Hacking
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_EXFIL = "system_exfil"
    CYBER_ATTACK = "cyber_attack"
    
    # Safety Harms & Toxicity (16 subtypes)
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    CHILD_EXPLOITATION = "child_exploitation"
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    ANIMAL_ABUSE = "animal_abuse"
    DISINFORMATION = "disinformation"
    SEXUAL_EXPLICIT = "sexual_explicit"
    WEAPONS_CBRN = "weapons_cbrn"
    
    # Integrity Compromise
    HALLUCINATION = "hallucination"
    UNAUTHORIZED_MEDICAL = "unauthorized_medical"
    UNAUTHORIZED_LEGAL = "unauthorized_legal"
    UNAUTHORIZED_FINANCIAL = "unauthorized_financial"
    
    # Intellectual Property
    COPYRIGHT_VIOLATION = "copyright_violation"
    TRADE_SECRET = "trade_secret"
    
    # Privacy Attacks
    PII_EXTRACTION = "pii_extraction"
    DOXXING = "doxxing"
    TARGETING = "targeting"


class IntentClass(Enum):
    """User intent classification"""
    # Legitimate intents
    HELP_SEEKING = "help_seeking"
    REPORTING = "reporting"
    VICTIM_SUPPORT = "victim_support"
    PREVENTION = "prevention"
    SAFETY_EDUCATION = "safety_education"
    PROFESSIONAL_GUIDANCE_REQUEST = "professional_guidance_request"
    
    # Harmful intents
    WRONGDOING_INSTRUCTIONS = "wrongdoing_instructions"
    EVASION = "evasion"
    STEALTH = "stealth"
    TARGETING = "targeting"
    DOXXING = "doxxing"
    PERSONAL_DATA_EXTRACTION = "personal_data_extraction"
    
    # Ambiguous
    AMBIGUOUS = "ambiguous"
    DUAL_USE = "dual_use"
    
    # Neutral
    INFORMATIONAL = "informational"
    GENERAL_QUERY = "general_query"


class ResponseMode(Enum):
    """Response handling mode"""
    ANSWER_NORMALLY = "answer_normally"
    ANSWER_SAFELY_CONSTRAINED = "answer_safely_constrained"
    REFUSE_WITH_SAFE_ALTERNATIVES = "refuse_with_safe_alternatives"
    ESCALATE_OR_RESOURCES = "escalate_or_resources"


class Severity(Enum):
    """Guard severity levels (kept for compatibility)"""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    REVIEW = "review"


# ============================================================================
# POLICY MATRIX
# ============================================================================

@dataclass
class PolicyDecision:
    """Policy decision from matrix lookup"""
    severity: Severity
    response_mode: ResponseMode
    rationale: str
    confidence: float = 1.0


class PolicyMatrix:
    """Unified policy matrix: (category, intent) -> decision"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.matrix: Dict[Tuple[str, str], PolicyDecision] = {}
        self.default_severity = Severity.REVIEW
        self.default_response_mode = ResponseMode.ANSWER_SAFELY_CONSTRAINED
        
        if config_path:
            self.load_from_yaml(config_path)
        else:
            self._load_default_matrix()
    
    def _load_default_matrix(self):
        """Load default policy matrix"""
        # Help-seeking queries: ALLOWED
        for category in [ContentCategory.HARASSMENT, ContentCategory.VIOLENCE, 
                         ContentCategory.SELF_HARM, ContentCategory.CYBER_ATTACK]:
            self.matrix[(category.value, IntentClass.HELP_SEEKING.value)] = PolicyDecision(
                severity=Severity.ALLOWED,
                response_mode=ResponseMode.ANSWER_NORMALLY,
                rationale=f"Help-seeking query about {category.value} - provide supportive guidance"
            )
            self.matrix[(category.value, IntentClass.REPORTING.value)] = PolicyDecision(
                severity=Severity.ALLOWED,
                response_mode=ResponseMode.ESCALATE_OR_RESOURCES,
                rationale=f"Reporting query about {category.value} - provide resources and escalation"
            )
            self.matrix[(category.value, IntentClass.VICTIM_SUPPORT.value)] = PolicyDecision(
                severity=Severity.ALLOWED,
                response_mode=ResponseMode.ESCALATE_OR_RESOURCES,
                rationale=f"Victim support query about {category.value} - provide resources"
            )
        
        # Prevention/Education: ALLOWED (constrained)
        for category in [ContentCategory.CYBER_ATTACK, ContentCategory.VIOLENCE,
                         ContentCategory.HARASSMENT]:
            self.matrix[(category.value, IntentClass.PREVENTION.value)] = PolicyDecision(
                severity=Severity.ALLOWED,
                response_mode=ResponseMode.ANSWER_SAFELY_CONSTRAINED,
                rationale=f"Prevention/education query about {category.value} - provide high-level guidance"
            )
            self.matrix[(category.value, IntentClass.SAFETY_EDUCATION.value)] = PolicyDecision(
                severity=Severity.ALLOWED,
                response_mode=ResponseMode.ANSWER_SAFELY_CONSTRAINED,
                rationale=f"Safety education query about {category.value} - provide educational content"
            )
        
        # Wrongdoing instructions: BLOCKED
        for category in [ContentCategory.HARASSMENT, ContentCategory.VIOLENCE,
                         ContentCategory.CYBER_ATTACK, ContentCategory.SELF_HARM]:
            self.matrix[(category.value, IntentClass.WRONGDOING_INSTRUCTIONS.value)] = PolicyDecision(
                severity=Severity.BLOCKED,
                response_mode=ResponseMode.REFUSE_WITH_SAFE_ALTERNATIVES,
                rationale=f"Wrongdoing instruction query about {category.value} - block and provide safe alternatives"
            )
        
        # Evasion/Stealth: BLOCKED
        for category in ContentCategory:
            self.matrix[(category.value, IntentClass.EVASION.value)] = PolicyDecision(
                severity=Severity.BLOCKED,
                response_mode=ResponseMode.REFUSE_WITH_SAFE_ALTERNATIVES,
                rationale=f"Evasion intent detected for {category.value} - block"
            )
            self.matrix[(category.value, IntentClass.STEALTH.value)] = PolicyDecision(
                severity=Severity.BLOCKED,
                response_mode=ResponseMode.REFUSE_WITH_SAFE_ALTERNATIVES,
                rationale=f"Stealth intent detected for {category.value} - block"
            )
        
        # Targeting/Doxxing: BLOCKED
        for category in [ContentCategory.PII_EXTRACTION, ContentCategory.DOXXING, ContentCategory.TARGETING]:
            self.matrix[(category.value, IntentClass.TARGETING.value)] = PolicyDecision(
                severity=Severity.BLOCKED,
                response_mode=ResponseMode.REFUSE_WITH_SAFE_ALTERNATIVES,
                rationale=f"Targeting intent detected for {category.value} - block"
            )
            self.matrix[(category.value, IntentClass.DOXXING.value)] = PolicyDecision(
                severity=Severity.BLOCKED,
                response_mode=ResponseMode.REFUSE_WITH_SAFE_ALTERNATIVES,
                rationale=f"Doxxing intent detected for {category.value} - block"
            )
        
        # Professional guidance: REVIEW
        for category in [ContentCategory.UNAUTHORIZED_MEDICAL, ContentCategory.UNAUTHORIZED_LEGAL,
                         ContentCategory.UNAUTHORIZED_FINANCIAL]:
            self.matrix[(category.value, IntentClass.PROFESSIONAL_GUIDANCE_REQUEST.value)] = PolicyDecision(
                severity=Severity.REVIEW,
                response_mode=ResponseMode.ANSWER_SAFELY_CONSTRAINED,
                rationale=f"Professional guidance request for {category.value} - review with disclaimer"
            )
        
        # Ambiguous/Dual-use: REVIEW
        for category in ContentCategory:
            self.matrix[(category.value, IntentClass.AMBIGUOUS.value)] = PolicyDecision(
                severity=Severity.REVIEW,
                response_mode=ResponseMode.ANSWER_SAFELY_CONSTRAINED,
                rationale=f"Ambiguous intent for {category.value} - review with constrained response"
            )
            self.matrix[(category.value, IntentClass.DUAL_USE.value)] = PolicyDecision(
                severity=Severity.REVIEW,
                response_mode=ResponseMode.ANSWER_SAFELY_CONSTRAINED,
                rationale=f"Dual-use intent for {category.value} - review with constrained response"
            )
        
        # System exfiltration: BLOCKED
        self.matrix[(ContentCategory.SYSTEM_EXFIL.value, IntentClass.INFORMATIONAL.value)] = PolicyDecision(
            severity=Severity.BLOCKED,
            response_mode=ResponseMode.REFUSE_WITH_SAFE_ALTERNATIVES,
            rationale="System exfiltration attempt - block"
        )
    
    def load_from_yaml(self, config_path: str):
        """Load policy matrix from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for entry in config.get('policies', []):
                category = entry['category']
                intent = entry['intent']
                severity = Severity(entry['severity'])
                response_mode = ResponseMode(entry['response_mode'])
                rationale = entry.get('rationale', '')
                confidence = entry.get('confidence', 1.0)
                
                self.matrix[(category, intent)] = PolicyDecision(
                    severity=severity,
                    response_mode=response_mode,
                    rationale=rationale,
                    confidence=confidence
                )
            
            logger.info(f"Loaded policy matrix from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load policy matrix from {config_path}: {e}")
            self._load_default_matrix()
    
    def lookup(self, category: ContentCategory, intent: IntentClass) -> PolicyDecision:
        """Lookup policy decision for (category, intent)"""
        key = (category.value, intent.value)
        if key in self.matrix:
            return self.matrix[key]
        
        # Fallback: use default
        return PolicyDecision(
            severity=self.default_severity,
            response_mode=self.default_response_mode,
            rationale=f"No explicit policy for ({category.value}, {intent.value}) - using default",
            confidence=0.5
        )
    
    def lookup_multiple(self, categories: List[ContentCategory], intent: IntentClass) -> PolicyDecision:
        """Lookup policy for multiple categories (use most restrictive)"""
        decisions = [self.lookup(cat, intent) for cat in categories]
        
        # Priority: BLOCKED > REVIEW > ALLOWED
        severity_priority = {Severity.BLOCKED: 3, Severity.REVIEW: 2, Severity.ALLOWED: 1}
        most_restrictive = max(decisions, key=lambda d: severity_priority.get(d.severity, 0))
        
        # Combine rationales
        rationales = [d.rationale for d in decisions if d.rationale]
        combined_rationale = " | ".join(rationales) if rationales else most_restrictive.rationale
        
        return PolicyDecision(
            severity=most_restrictive.severity,
            response_mode=most_restrictive.response_mode,
            rationale=combined_rationale,
            confidence=min(d.confidence for d in decisions)
        )


# ============================================================================
# INTENT CLASSIFIER
# ============================================================================

class IntentClassifier:
    """Classify user intent from query text"""
    
    def __init__(self):
        # Help-seeking patterns
        self.help_seeking_patterns = [
            (r"(?i)\b(how\s+to\s+)?(deal\s+with|handle|cope\s+with|respond\s+to|report|stop|prevent|protect\s+myself|protect\s+against)", 1.0),
            (r"(?i)\b(i\s+am\s+being|i\s+am\s+a\s+victim|i\s+need\s+help|i\s+need\s+support)", 1.0),
            (r"(?i)\b(what\s+should\s+i\s+do|what\s+can\s+i\s+do|where\s+can\s+i\s+get\s+help)", 0.9),
            (r"(?i)\b(help|support|resources|hotline|helpline)", 0.7),
        ]
        
        # Reporting patterns
        self.reporting_patterns = [
            (r"(?i)\b(report|file\s+a\s+complaint|report\s+to|tell\s+authorities|contact\s+police)", 1.0),
            (r"(?i)\b(how\s+to\s+report|where\s+to\s+report|who\s+to\s+contact)", 0.9),
        ]
        
        # Wrongdoing instruction patterns
        self.wrongdoing_patterns = [
            (r"(?i)\b(how\s+to|how\s+can\s+i|ways?\s+to|steps?\s+to|tutorial|guide|instructions?)\s+(harass|bully|stalk|threaten|attack|hack|steal|scam|dox|exploit)", 1.0),
            (r"(?i)\b(write\s+a\s+script|create\s+a\s+tool|build\s+a\s+weapon|make\s+a\s+bomb)", 1.0),
            (r"(?i)\b(exact\s+commands?|specific\s+steps?|detailed\s+instructions?)", 0.8),
            (r"(?i)\b(harass|bully|attack|hack|steal|scam|dox|exploit)\s+(someone|anyone|people|colleague)", 0.9),
        ]
        
        # Evasion/Stealth patterns
        self.evasion_patterns = [
            (r"(?i)\b(without\s+getting\s+caught|avoid\s+detection|evade|bypass|stealth|hidden|secret)", 1.0),
            (r"(?i)\b(how\s+to\s+avoid|how\s+to\s+evade|how\s+to\s+bypass)", 0.9),
        ]
        
        # Targeting/Doxxing patterns
        self.targeting_patterns = [
            (r"(?i)\b(find|locate|track|identify|get\s+info\s+about|personal\s+information\s+about)\s+(someone|person|individual)", 0.9),
            (r"(?i)\b(dox|doxx|doxing|doxxing|personal\s+data|private\s+information)", 1.0),
            (r"(?i)\b(address|phone|email|ssn|social\s+security)\s+(of|for|about)", 0.8),
        ]
        
        # Prevention/Education patterns
        self.prevention_patterns = [
            (r"(?i)\b(how\s+to\s+prevent|how\s+to\s+protect|how\s+to\s+defend|security\s+best\s+practices)", 0.9),
            (r"(?i)\b(what\s+is|explain|tell\s+me\s+about|information\s+about)\s+(cybersecurity|security|safety)", 0.7),
        ]
        
        # Professional guidance patterns
        self.professional_guidance_patterns = [
            (r"(?i)\b(should\s+i|what\s+should\s+i|advice|recommendation|guidance)\s+(about|regarding|for)\s+(medical|legal|financial)", 0.8),
            (r"(?i)\b(medical|legal|financial)\s+(advice|recommendation|guidance|consultation)", 0.9),
        ]
    
    def classify(self, query: str) -> Tuple[IntentClass, float]:
        """
        Classify intent from query
        Returns: (intent_class, confidence)
        """
        query_lower = query.lower()
        scores: Dict[IntentClass, float] = {}
        
        # Check help-seeking
        for pattern, weight in self.help_seeking_patterns:
            if re.search(pattern, query_lower):
                scores[IntentClass.HELP_SEEKING] = scores.get(IntentClass.HELP_SEEKING, 0) + weight
        
        # Check reporting
        for pattern, weight in self.reporting_patterns:
            if re.search(pattern, query_lower):
                scores[IntentClass.REPORTING] = scores.get(IntentClass.REPORTING, 0) + weight
        
        # Check wrongdoing instructions (high priority)
        for pattern, weight in self.wrongdoing_patterns:
            if re.search(pattern, query_lower):
                scores[IntentClass.WRONGDOING_INSTRUCTIONS] = scores.get(IntentClass.WRONGDOING_INSTRUCTIONS, 0) + weight
        
        # Check evasion
        for pattern, weight in self.evasion_patterns:
            if re.search(pattern, query_lower):
                scores[IntentClass.EVASION] = scores.get(IntentClass.EVASION, 0) + weight
        
        # Check targeting
        for pattern, weight in self.targeting_patterns:
            if re.search(pattern, query_lower):
                scores[IntentClass.TARGETING] = scores.get(IntentClass.TARGETING, 0) + weight
        
        # Check prevention
        for pattern, weight in self.prevention_patterns:
            if re.search(pattern, query_lower):
                scores[IntentClass.PREVENTION] = scores.get(IntentClass.PREVENTION, 0) + weight
        
        # Check professional guidance
        for pattern, weight in self.professional_guidance_patterns:
            if re.search(pattern, query_lower):
                scores[IntentClass.PROFESSIONAL_GUIDANCE_REQUEST] = scores.get(IntentClass.PROFESSIONAL_GUIDANCE_REQUEST, 0) + weight
        
        # Determine intent (highest score wins, but harmful intents take priority)
        if not scores:
            return IntentClass.INFORMATIONAL, 0.5
        
        # Priority order: harmful > help-seeking > neutral
        harmful_intents = [IntentClass.WRONGDOING_INSTRUCTIONS, IntentClass.EVASION, 
                          IntentClass.TARGETING, IntentClass.DOXXING]
        for intent in harmful_intents:
            if intent in scores and scores[intent] > 0.5:
                return intent, min(scores[intent], 1.0)
        
        # Otherwise, highest score
        best_intent = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_intent[1], 1.0)
        
        # If confidence is low, mark as ambiguous
        if confidence < 0.6:
            return IntentClass.AMBIGUOUS, confidence
        
        return best_intent[0], confidence
