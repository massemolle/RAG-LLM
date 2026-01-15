"""
NeMo Guardrails Jailbreak Detection Heuristics
Implements Length per Perplexity and Prefix/Suffix Perplexity
"""

import re
import logging
import torch
from typing import Tuple, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

# Global model cache
_perplexity_model = None
_perplexity_tokenizer = None


def get_perplexity_model():
    """Get or load GPT2 model for perplexity calculation"""
    global _perplexity_model, _perplexity_tokenizer
    
    if _perplexity_model is None:
        try:
            model_name = "gpt2-large"  # NeMo uses gpt2-large
            logger.info(f"Loading {model_name} for perplexity calculation...")
            _perplexity_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            _perplexity_model = GPT2LMHeadModel.from_pretrained(model_name)
            _perplexity_model.eval()
            
            # Set pad token
            if _perplexity_tokenizer.pad_token is None:
                _perplexity_tokenizer.pad_token = _perplexity_tokenizer.eos_token
            
            logger.info(f"{model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load perplexity model: {e}")
            return None, None
    
    return _perplexity_model, _perplexity_tokenizer


def calculate_perplexity(text: str, model=None, tokenizer=None) -> Optional[float]:
    """
    Calculate perplexity of text using GPT2 model
    Returns None if calculation fails
    """
    if model is None or tokenizer is None:
        model, tokenizer = get_perplexity_model()
        if model is None:
            return None
    
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Calculate perplexity
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    except Exception as e:
        logger.error(f"Perplexity calculation failed: {e}")
        return None


def length_per_perplexity(text: str, threshold: float = 89.79) -> Tuple[bool, float, Optional[float]]:
    """
    NeMo Heuristic 1: Length per Perplexity
    Detects jailbreaks by comparing text length to perplexity
    
    High ratio (length/perplexity) indicates suspicious text:
    - Normal text: low perplexity, reasonable length → low ratio
    - Jailbreak: often has high perplexity (garbled/encoded) or unusual patterns → high ratio
    
    Args:
        text: Input text to check
        threshold: Threshold value (default from NeMo: 89.79)
    
    Returns:
        (is_jailbreak, ratio, perplexity)
    """
    if not text or len(text.strip()) == 0:
        return False, 0.0, None
    
    perplexity = calculate_perplexity(text)
    if perplexity is None:
        return False, 0.0, None
    
    # Use character length (NeMo uses character count)
    text_length = len(text)
    
    # Calculate ratio: length / perplexity
    # NeMo's threshold of 89.79 means: if length/perplexity > 89.79, it's suspicious
    # This catches cases where text is unusually long relative to its perplexity
    # (normal text has moderate perplexity, jailbreaks might have unusual ratios)
    ratio = text_length / perplexity if perplexity > 0 else float('inf')
    
    # Check for very high perplexity (garbled/encoded text) - but be more conservative
    # Very high perplexity (>1000) often indicates encoded or garbled text
    # Only flag if text is substantial (not just a short query)
    if perplexity > 1000 and text_length > 50:
        return True, ratio, perplexity
    
    # Check for very low perplexity with substantial length (might be repetitive or suspicious)
    # But be lenient - short queries can have low perplexity naturally
    if perplexity < 5 and text_length > 200:
        return True, ratio, perplexity
    
    # For short queries, be more lenient
    if text_length < 50:
        # Only flag if ratio is extremely high (likely not a real issue for short queries)
        is_jailbreak = ratio > (threshold * 2)  # Double the threshold for short queries
    else:
        is_jailbreak = ratio > threshold
    
    return is_jailbreak, ratio, perplexity


def prefix_suffix_perplexity(text: str, threshold: float = 1845.65) -> Tuple[bool, float, Optional[float], Optional[float]]:
    """
    NeMo Heuristic 2: Prefix and Suffix Perplexity
    Detects jailbreaks by comparing prefix/suffix perplexity
    
    Jailbreak attempts often have:
    - Normal prefix (to bypass initial checks)
    - Suspicious suffix (injection payload)
    - Or vice versa
    
    Args:
        text: Input text to check
        threshold: Threshold value (default from NeMo: 1845.65)
    
    Returns:
        (is_jailbreak, ratio, prefix_perplexity, suffix_perplexity)
    """
    if not text or len(text.strip()) == 0:
        return False, 0.0, None, None
    
    # Split into prefix and suffix (first and last 50% of tokens/words)
    words = text.split()
    if len(words) < 4:
        # Too short, use full text for both
        prefix_text = text
        suffix_text = text
    else:
        mid = len(words) // 2
        prefix_text = " ".join(words[:mid])
        suffix_text = " ".join(words[mid:])
    
    prefix_perp = calculate_perplexity(prefix_text)
    suffix_perp = calculate_perplexity(suffix_text)
    
    if prefix_perp is None or suffix_perp is None:
        return False, 0.0, prefix_perp, suffix_perp
    
    # Calculate ratio: max(prefix_perp, suffix_perp) / min(prefix_perp, suffix_perp)
    # NeMo's threshold of 1845.65 is very high, so this catches extreme cases
    # High ratio indicates large difference between prefix and suffix (suspicious)
    if prefix_perp == 0 or suffix_perp == 0:
        # Avoid division by zero - if either is zero, use a safe default
        ratio = 0.0
    else:
        # Use the ratio of max to min (this is what NeMo actually uses)
        ratio = max(prefix_perp, suffix_perp) / min(prefix_perp, suffix_perp)
    
    # For very short queries (like "hello", "what is X?"), perplexity can vary naturally
    # Technical terms, proper nouns, and short phrases can have high perplexity
    # Be very lenient with short queries - skip prefix/suffix check entirely
    if len(words) <= 6:
        # For queries with 6 words or less, only flag if BOTH parts have extremely high perplexity
        # This catches truly garbled text, not just technical terms
        if prefix_perp > 2000 and suffix_perp > 2000:
            return True, ratio, prefix_perp, suffix_perp
        # Otherwise, allow short queries through (they're likely legitimate)
        is_jailbreak = False
        return is_jailbreak, ratio, prefix_perp, suffix_perp
    
    # Check for very high perplexity in either part (garbled/encoded)
    # But be conservative - only flag if it's clearly abnormal
    if (prefix_perp > 1000 or suffix_perp > 1000) and ratio > 50:
        # One part is much higher than the other AND one is very high
        return True, ratio, prefix_perp, suffix_perp
    
    # Check for large absolute difference (indicates mixed normal/suspicious text)
    # This is a key indicator: normal queries have similar prefix/suffix perplexity
    # But be more conservative - only flag extreme cases
    abs_diff = abs(prefix_perp - suffix_perp)
    avg_perp = (prefix_perp + suffix_perp) / 2
    if avg_perp > 0:
        relative_diff = abs_diff / avg_perp
        # If difference is more than 500% of average AND absolute diff > 500, it's suspicious
        if relative_diff > 5.0 and abs_diff > 500:
            return True, ratio, prefix_perp, suffix_perp
    
    # The threshold of 1845.65 is extremely high - only flag truly extreme cases
    # This catches cases where one part has extremely high perplexity relative to the other
    is_jailbreak = ratio > threshold
    
    return is_jailbreak, ratio, prefix_perp, suffix_perp


def check_jailbreak_heuristics(
    text: str,
    length_per_perplexity_threshold: float = 89.79,
    prefix_suffix_perplexity_threshold: float = 1845.65
) -> Tuple[bool, dict]:
    """
    Check both jailbreak heuristics
    
    Returns:
        (is_jailbreak, details_dict)
    """
    details = {
        "length_per_perplexity": {},
        "prefix_suffix_perplexity": {}
    }
    
    # Heuristic 1: Length per Perplexity
    is_jailbreak_1, ratio_1, perp_1 = length_per_perplexity(text, length_per_perplexity_threshold)
    details["length_per_perplexity"] = {
        "is_jailbreak": is_jailbreak_1,
        "ratio": ratio_1,
        "perplexity": perp_1,
        "threshold": length_per_perplexity_threshold
    }
    
    # Heuristic 2: Prefix/Suffix Perplexity
    is_jailbreak_2, ratio_2, perp_prefix, perp_suffix = prefix_suffix_perplexity(
        text, prefix_suffix_perplexity_threshold
    )
    details["prefix_suffix_perplexity"] = {
        "is_jailbreak": is_jailbreak_2,
        "ratio": ratio_2,
        "prefix_perplexity": perp_prefix,
        "suffix_perplexity": perp_suffix,
        "threshold": prefix_suffix_perplexity_threshold
    }
    
    # If either heuristic detects jailbreak, flag it
    is_jailbreak = is_jailbreak_1 or is_jailbreak_2
    
    return is_jailbreak, details

