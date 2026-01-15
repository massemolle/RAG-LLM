"""
Unified Input Security using Policy Framework
Implements intent-based guardrails with policy matrix
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from nvidia_nemo.enhanced_guardrails import Severity, GuardResult
from nvidia_nemo.policy_framework import (
    ContentCategory, IntentClass, ResponseMode, PolicyMatrix, PolicyDecision
)

logger = logging.getLogger(__name__)


class UnifiedInputSecurity:
    """Unified input security using policy framework"""
    
    def __init__(self, policy_matrix: PolicyMatrix, content_classifier, intent_classifier):
        self.policy_matrix = policy_matrix
        self.content_classifier = content_classifier
        self.intent_classifier = intent_classifier
    
    def evaluate(self, query: str, layer_a_result=None, layer_b_result=None, 
                 layer_c_result=None) -> GuardResult:
        """
        Evaluate input security using policy framework
        Returns GuardResult with severity and reason
        """
        # Step 1: Classify content categories
        detected_categories = self.content_classifier.classify(query)
        category_enums = [cat for cat, conf in detected_categories if conf > 0.5]
        
        # Step 2: Classify intent
        intent_class, intent_confidence = self.intent_classifier.classify(query)
        
        # Step 3: Lookup policy decision
        if category_enums:
            # Use most restrictive decision across all categories
            policy_decision = self.policy_matrix.lookup_multiple(category_enums, intent_class)
        else:
            # No category detected - check if intent alone is harmful
            if intent_class in [IntentClass.WRONGDOING_INSTRUCTIONS, IntentClass.EVASION, 
                               IntentClass.TARGETING, IntentClass.DOXXING]:
                policy_decision = PolicyDecision(
                    severity=Severity.BLOCKED,
                    response_mode=ResponseMode.REFUSE_WITH_SAFE_ALTERNATIVES,
                    rationale=f"Harmful intent detected ({intent_class.value}) without specific category",
                    confidence=intent_confidence
                )
            else:
                policy_decision = PolicyDecision(
                    severity=Severity.ALLOWED,
                    response_mode=ResponseMode.ANSWER_NORMALLY,
                    rationale="No harmful categories or intents detected",
                    confidence=1.0
                )
        
        # Step 4: Build reason string
        category_names = [cat.value for cat in category_enums] if category_enums else ["none"]
        reason_parts = [
            f"Policy framework evaluation.",
            f"Categories: {', '.join(category_names)}",
            f"Intent: {intent_class.value} (confidence: {intent_confidence:.2f})",
            f"Policy decision: {policy_decision.severity.value}",
            f"Response mode: {policy_decision.response_mode.value}",
            f"Rationale: {policy_decision.rationale}"
        ]
        
        # Add layer contributions if available
        layer_contributions = []
        if layer_a_result:
            layer_contributions.append(f"Layer A: {layer_a_result[1][:100]}")
        if layer_b_result:
            layer_contributions.append(f"Layer B: {layer_b_result[1][:100]}")
        if layer_c_result:
            layer_contributions.append(f"Layer C: {layer_c_result[1][:100]}")
        
        if layer_contributions:
            reason_parts.append(f"Layer contributions: {' | '.join(layer_contributions)}")
        
        reason = " | ".join(reason_parts)
        
        return GuardResult(
            guard_name="input-security",
            severity=policy_decision.severity,
            reason=reason,
            triggered=True,
            layers_triggered=["policy_framework"] + (["layer_a", "layer_b", "layer_c"] if layer_contributions else [])
        )
