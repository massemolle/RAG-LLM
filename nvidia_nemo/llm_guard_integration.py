"""
LLM Guard Integration for Defensive Scanning

This module integrates the llm-guard library (protectai/llm-guard) for
comprehensive input and output scanning including:
- Prompt injection detection
- Toxicity detection
- PII/Secrets detection
- Invisible text removal

Layer 1 in the guardrails pipeline.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Check if llm-guard is available
LLM_GUARD_AVAILABLE = False
try:
    from llm_guard import scan_prompt, scan_output
    from llm_guard.input_scanners import (
        PromptInjection,
        Toxicity,
        Secrets,
        InvisibleText,
    )
    from llm_guard.output_scanners import (
        Toxicity as OutputToxicity,
        Sensitive,
        NoRefusal,
    )
    LLM_GUARD_AVAILABLE = True
    logger.info("LLM Guard library loaded successfully")
except ImportError as e:
    logger.warning(f"LLM Guard not available: {e}. Install with: pip install llm-guard")


class ScannerType(Enum):
    """Types of scanners available"""
    INVISIBLE_TEXT = "invisible_text"
    SECRETS = "secrets"
    PROMPT_INJECTION = "prompt_injection"
    TOXICITY = "toxicity"
    OUTPUT_TOXICITY = "output_toxicity"
    SENSITIVE = "sensitive"
    NO_REFUSAL = "no_refusal"


@dataclass
class ScanResult:
    """Result of a single scanner"""
    scanner_name: str
    is_valid: bool
    risk_score: float
    sanitized_text: str
    details: Dict = field(default_factory=dict)


@dataclass
class LLMGuardResult:
    """Aggregated result of all LLM Guard scans"""
    is_safe: bool
    sanitized_text: str
    scan_results: List[ScanResult]
    failed_scanners: List[str]
    total_risk_score: float
    details: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/UI"""
        return {
            "is_safe": self.is_safe,
            "failed_scanners": self.failed_scanners,
            "total_risk_score": self.total_risk_score,
            "scanner_results": {
                r.scanner_name: {
                    "valid": r.is_valid,
                    "risk": r.risk_score
                }
                for r in self.scan_results
            }
        }


class LLMGuardWrapper:
    """
    Wrapper class for LLM Guard scanners with configurable thresholds
    and scanner selection.
    """
    
    def __init__(
        self,
        enable_prompt_injection: bool = True,
        enable_toxicity: bool = True,
        enable_secrets: bool = True,
        enable_invisible_text: bool = True,
        prompt_injection_threshold: float = 0.5,
        toxicity_threshold: float = 0.7,
        fail_fast: bool = True,
        lazy_load: bool = True
    ):
        """
        Initialize LLM Guard wrapper.
        
        Args:
            enable_prompt_injection: Enable prompt injection scanner
            enable_toxicity: Enable toxicity scanner
            enable_secrets: Enable secrets scanner
            enable_invisible_text: Enable invisible text scanner
            prompt_injection_threshold: Threshold for prompt injection (0-1)
            toxicity_threshold: Threshold for toxicity (0-1)
            fail_fast: Stop scanning after first failure
            lazy_load: Load models on first use (saves memory)
        """
        self.enable_prompt_injection = enable_prompt_injection
        self.enable_toxicity = enable_toxicity
        self.enable_secrets = enable_secrets
        self.enable_invisible_text = enable_invisible_text
        self.prompt_injection_threshold = prompt_injection_threshold
        self.toxicity_threshold = toxicity_threshold
        self.fail_fast = fail_fast
        self.lazy_load = lazy_load
        
        self._input_scanners = None
        self._output_scanners = None
        self._initialized = False
        
    def _initialize_scanners(self):
        """Initialize scanners on first use"""
        if self._initialized or not LLM_GUARD_AVAILABLE:
            return
        
        logger.info("Initializing LLM Guard scanners...")
        
        # Build input scanners list (ordered by speed: fast first)
        self._input_scanners = []
        
        if self.enable_invisible_text:
            self._input_scanners.append(InvisibleText())
            logger.debug("Added InvisibleText scanner")
        
        if self.enable_secrets:
            self._input_scanners.append(Secrets(redact_mode="all"))
            logger.debug("Added Secrets scanner")
        
        if self.enable_prompt_injection:
            self._input_scanners.append(
                PromptInjection(threshold=self.prompt_injection_threshold)
            )
            logger.debug(f"Added PromptInjection scanner (threshold={self.prompt_injection_threshold})")
        
        if self.enable_toxicity:
            self._input_scanners.append(
                Toxicity(threshold=self.toxicity_threshold)
            )
            logger.debug(f"Added Toxicity scanner (threshold={self.toxicity_threshold})")
        
        # Build output scanners
        self._output_scanners = [
            OutputToxicity(threshold=self.toxicity_threshold),
            Sensitive(),
        ]
        
        self._initialized = True
        logger.info(f"Initialized {len(self._input_scanners)} input scanners and {len(self._output_scanners)} output scanners")
    
    def scan_input(self, text: str) -> LLMGuardResult:
        """
        Scan input text with all enabled scanners.
        
        Args:
            text: The input text to scan
            
        Returns:
            LLMGuardResult with aggregated scan results
        """
        if not LLM_GUARD_AVAILABLE:
            logger.warning("LLM Guard not available, skipping input scan")
            return LLMGuardResult(
                is_safe=True,
                sanitized_text=text,
                scan_results=[],
                failed_scanners=[],
                total_risk_score=0.0,
                details={"warning": "LLM Guard not installed"}
            )
        
        self._initialize_scanners()
        
        if not self._input_scanners:
            return LLMGuardResult(
                is_safe=True,
                sanitized_text=text,
                scan_results=[],
                failed_scanners=[],
                total_risk_score=0.0,
                details={"info": "No input scanners configured"}
            )
        
        try:
            # Run all scanners
            sanitized_text, results_valid, results_score = scan_prompt(
                self._input_scanners,
                text,
                fail_fast=self.fail_fast
            )
            
            # Build individual scan results
            scan_results = []
            failed_scanners = []
            total_risk = 0.0
            
            for scanner in self._input_scanners:
                scanner_name = scanner.__class__.__name__
                is_valid = results_valid.get(scanner_name, True)
                risk_score = results_score.get(scanner_name, 0.0)
                
                scan_results.append(ScanResult(
                    scanner_name=scanner_name,
                    is_valid=is_valid,
                    risk_score=risk_score,
                    sanitized_text=sanitized_text,
                    details={}
                ))
                
                if not is_valid:
                    failed_scanners.append(scanner_name)
                
                total_risk = max(total_risk, risk_score)
            
            is_safe = len(failed_scanners) == 0
            
            return LLMGuardResult(
                is_safe=is_safe,
                sanitized_text=sanitized_text,
                scan_results=scan_results,
                failed_scanners=failed_scanners,
                total_risk_score=total_risk,
                details={
                    "scanners_run": len(self._input_scanners),
                    "fail_fast": self.fail_fast
                }
            )
            
        except Exception as e:
            logger.error(f"LLM Guard scan failed: {e}")
            return LLMGuardResult(
                is_safe=True,  # Fail open to avoid blocking on errors
                sanitized_text=text,
                scan_results=[],
                failed_scanners=[],
                total_risk_score=0.0,
                details={"error": str(e)}
            )
    
    def scan_output(self, prompt: str, response: str) -> LLMGuardResult:
        """
        Scan output text with output scanners.
        
        Args:
            prompt: The original prompt
            response: The LLM response to scan
            
        Returns:
            LLMGuardResult with aggregated scan results
        """
        if not LLM_GUARD_AVAILABLE:
            logger.warning("LLM Guard not available, skipping output scan")
            return LLMGuardResult(
                is_safe=True,
                sanitized_text=response,
                scan_results=[],
                failed_scanners=[],
                total_risk_score=0.0,
                details={"warning": "LLM Guard not installed"}
            )
        
        self._initialize_scanners()
        
        if not self._output_scanners:
            return LLMGuardResult(
                is_safe=True,
                sanitized_text=response,
                scan_results=[],
                failed_scanners=[],
                total_risk_score=0.0,
                details={"info": "No output scanners configured"}
            )
        
        try:
            # Run output scanners
            sanitized_text, results_valid, results_score = scan_output(
                self._output_scanners,
                prompt,
                response,
                fail_fast=self.fail_fast
            )
            
            # Build individual scan results
            scan_results = []
            failed_scanners = []
            total_risk = 0.0
            
            for scanner in self._output_scanners:
                scanner_name = scanner.__class__.__name__
                is_valid = results_valid.get(scanner_name, True)
                risk_score = results_score.get(scanner_name, 0.0)
                
                scan_results.append(ScanResult(
                    scanner_name=scanner_name,
                    is_valid=is_valid,
                    risk_score=risk_score,
                    sanitized_text=sanitized_text,
                    details={}
                ))
                
                if not is_valid:
                    failed_scanners.append(scanner_name)
                
                total_risk = max(total_risk, risk_score)
            
            is_safe = len(failed_scanners) == 0
            
            return LLMGuardResult(
                is_safe=is_safe,
                sanitized_text=sanitized_text,
                scan_results=scan_results,
                failed_scanners=failed_scanners,
                total_risk_score=total_risk,
                details={
                    "scanners_run": len(self._output_scanners),
                    "fail_fast": self.fail_fast
                }
            )
            
        except Exception as e:
            logger.error(f"LLM Guard output scan failed: {e}")
            return LLMGuardResult(
                is_safe=True,
                sanitized_text=response,
                scan_results=[],
                failed_scanners=[],
                total_risk_score=0.0,
                details={"error": str(e)}
            )
    
    def get_status(self) -> Dict:
        """Get status of LLM Guard integration"""
        return {
            "available": LLM_GUARD_AVAILABLE,
            "initialized": self._initialized,
            "input_scanners": len(self._input_scanners) if self._input_scanners else 0,
            "output_scanners": len(self._output_scanners) if self._output_scanners else 0,
            "config": {
                "prompt_injection": self.enable_prompt_injection,
                "toxicity": self.enable_toxicity,
                "secrets": self.enable_secrets,
                "invisible_text": self.enable_invisible_text,
                "fail_fast": self.fail_fast
            }
        }


# Fallback implementation when LLM Guard is not installed
class FallbackLLMGuard:
    """
    Fallback scanner using regex patterns when llm-guard is not installed.
    Less accurate but provides basic protection.
    """
    
    import re
    
    # Injection patterns — sourced from shared utils.injection_patterns
    # when available, with a minimal inline fallback.
    try:
        from utils.injection_patterns import INJECTION_PATTERNS_FLAT as _shared
        INJECTION_PATTERNS = [p.pattern for _, p, _, _ in _shared[:20]]
    except ImportError:
        INJECTION_PATTERNS = [
            r'ignore\s+(all\s+)?previous\s+instructions',
            r'disregard\s+(everything|all|your)',
            r'forget\s+(your|all|previous)',
            r'you\s+are\s+now\s+\w+',
            r'pretend\s+(to\s+be|you\s+are)',
            r'act\s+as\s+if',
            r'system\s*:?\s*override',
            r'</?(system|instructions|prompt)>',
        ]
    
    SECRET_PATTERNS = [
        r'[a-zA-Z0-9]{20,}',  # Long alphanumeric strings (potential API keys)
        r'sk-[a-zA-Z0-9]{32,}',  # OpenAI-style keys
        r'AKIA[A-Z0-9]{16}',  # AWS access keys
        r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',  # Private keys
    ]
    
    TOXICITY_WORDS = [
        'kill', 'murder', 'hate', 'attack', 'destroy', 'harm',
        'racist', 'sexist', 'slur',
    ]
    
    def __init__(self):
        self.injection_regex = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self.secret_regex = [re.compile(p) for p in self.SECRET_PATTERNS]
    
    def scan_input(self, text: str) -> LLMGuardResult:
        """Basic fallback scanning"""
        failed_scanners = []
        risk_score = 0.0
        
        # Check for injection patterns
        for regex in self.injection_regex:
            if regex.search(text):
                failed_scanners.append("FallbackPromptInjection")
                risk_score = max(risk_score, 0.8)
                break
        
        # Check for secrets
        for regex in self.secret_regex:
            if regex.search(text):
                failed_scanners.append("FallbackSecrets")
                risk_score = max(risk_score, 0.7)
                break
        
        # Basic toxicity check
        text_lower = text.lower()
        if any(word in text_lower for word in self.TOXICITY_WORDS):
            # Only flag if multiple toxic words or in aggressive context
            toxic_count = sum(1 for word in self.TOXICITY_WORDS if word in text_lower)
            if toxic_count >= 2:
                failed_scanners.append("FallbackToxicity")
                risk_score = max(risk_score, 0.6)
        
        return LLMGuardResult(
            is_safe=len(failed_scanners) == 0,
            sanitized_text=text,
            scan_results=[],
            failed_scanners=failed_scanners,
            total_risk_score=risk_score,
            details={"fallback": True, "reason": "llm-guard not installed"}
        )
    
    def scan_output(self, prompt: str, response: str) -> LLMGuardResult:
        """Basic fallback output scanning"""
        return self.scan_input(response)  # Reuse input scanning for basic check


# Singleton instance
_llm_guard_instance: Optional[LLMGuardWrapper] = None


def get_llm_guard(
    prompt_injection_threshold: float = 0.5,
    toxicity_threshold: float = 0.7
) -> LLMGuardWrapper:
    """Get or create the singleton LLM Guard instance"""
    global _llm_guard_instance
    if _llm_guard_instance is None:
        if LLM_GUARD_AVAILABLE:
            _llm_guard_instance = LLMGuardWrapper(
                prompt_injection_threshold=prompt_injection_threshold,
                toxicity_threshold=toxicity_threshold
            )
        else:
            _llm_guard_instance = FallbackLLMGuard()
    return _llm_guard_instance


def scan_input_text(text: str) -> LLMGuardResult:
    """Convenience function to scan input text"""
    guard = get_llm_guard()
    return guard.scan_input(text)


def scan_output_text(prompt: str, response: str) -> LLMGuardResult:
    """Convenience function to scan output text"""
    guard = get_llm_guard()
    return guard.scan_output(prompt, response)
