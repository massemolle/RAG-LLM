"""
LLM Client Abstraction Layer

Provides a unified interface for different LLM backends:
- LocalLLMClient: Hugging Face transformers pipeline (local models)
- BeamStudioClient: BeamStudio API (GPT-5.1, GPT-4o, etc.)

All clients support OpenTelemetry tracing for observability.
"""

import os
import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# OpenTelemetry imports (optional)
try:
    from observability.opentelemetry_integration import (
        create_trace_span, set_span_attribute, OPENTELEMETRY_AVAILABLE
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt or messages.
        
        Args:
            prompt: Plain text prompt (for text-generation models)
            messages: List of chat messages [{"role": "user", "content": "..."}]
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the model."""
        pass
    
    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (e.g., 'local', 'beamstudio')."""
        pass


class LocalLLMClient(LLMClient):
    """
    Local LLM client using Hugging Face transformers pipeline.
    Wraps existing get_pipeline functionality.
    """
    
    def __init__(
        self,
        model_name: str = "Felladrin/Smol-Llama-101M-Chat-v1",
        device: str = "cpu",
        pipeline_instance=None
    ):
        """
        Initialize local LLM client.
        
        Args:
            model_name: Hugging Face model ID or local path
            device: Device to run on ('cpu', 'cuda:0', etc.)
            pipeline_instance: Optional pre-initialized pipeline
        """
        self._model_name = model_name
        self._device = device
        self._pipeline = pipeline_instance
        
        # Default generation args for local models
        self.default_gen_args = {
            "max_new_tokens": 220,
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 4,
            "do_sample": True,
        }
        
        if self._pipeline is None:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the Hugging Face pipeline."""
        import torch
        from transformers import pipeline as hf_pipeline
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        os.makedirs('./model_w', exist_ok=True)
        local_dir = os.path.join('./model_w', self._model_name)
        
        try:
            if os.path.isdir(local_dir):
                self._pipeline = hf_pipeline(
                    "text-generation",
                    model=local_dir,
                    device=device,
                    trust_remote_code=True,
                    use_fast=False
                )
            else:
                self._pipeline = hf_pipeline(
                    "text-generation",
                    model=self._model_name,
                    device=device,
                    trust_remote_code=True,
                    use_fast=False
                )
                os.makedirs(os.path.dirname(local_dir), exist_ok=True)
                self._pipeline.save_pretrained(local_dir)
            logger.info(f"Initialized local LLM: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            self._pipeline = None
    
    @property
    def pipeline(self):
        """Get the underlying pipeline instance."""
        return self._pipeline
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def provider(self) -> str:
        return "local"
    
    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Generate text using the local pipeline."""
        if self._pipeline is None:
            raise RuntimeError("LLM pipeline not initialized")
        
        start_time = datetime.now()
        span = None
        
        # Start OpenTelemetry span if available
        if OPENTELEMETRY_AVAILABLE:
            try:
                span = create_trace_span(
                    "llm_generation",
                    attributes={
                        "llm.provider": self.provider,
                        "llm.model": self._model_name,
                        "llm.prompt_length": len(prompt),
                        "span.type": "llm"
                    }
                )
                span.__enter__()
            except Exception as e:
                logger.debug(f"Failed to create OTEL span: {e}")
        
        try:
            # Merge default args with provided kwargs
            gen_args = {**self.default_gen_args, **kwargs}
            
            # Remove args not supported by HF pipeline
            gen_args.pop('max_completion_tokens', None)
            
            # Call pipeline
            result = self._pipeline(prompt, **gen_args)
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    full_text = result[0]['generated_text']
                    # Remove input prompt from output
                    if full_text.startswith(prompt):
                        output = full_text[len(prompt):].strip()
                    else:
                        output = full_text.strip()
                else:
                    output = str(result[0])
            else:
                output = str(result)
            
            # Add span attributes
            if span:
                try:
                    set_span_attribute(span, "llm.completion_length", len(output))
                    set_span_attribute(span, "llm.latency_ms", (datetime.now() - start_time).total_seconds() * 1000)
                except Exception:
                    pass
            
            return output
            
        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            if span:
                try:
                    set_span_attribute(span, "error", str(e))
                except Exception:
                    pass
            raise
        finally:
            if span:
                try:
                    span.__exit__(None, None, None)
                except Exception:
                    pass


class BeamStudioClient(LLMClient):
    """
    BeamStudio API client for cloud-hosted LLMs (GPT-5.1, GPT-4o, etc.)
    Uses Azure OpenAI compatible API format.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """
        Initialize BeamStudio client.
        
        Args:
            api_key: API key (or use BEAMSTUDIO_API_KEY env var)
            base_url: Base URL (or use BEAMSTUDIO_BASE_URL env var)
            model: Model name (or use BEAMSTUDIO_MODEL env var)
            api_version: API version (or use BEAMSTUDIO_API_VERSION env var)
        """
        self._api_key = api_key or os.getenv("BEAMSTUDIO_API_KEY")
        self._base_url = base_url or os.getenv("BEAMSTUDIO_BASE_URL", "https://beamstudio.private.uat.enocloud.eu")
        self._model = model or os.getenv("BEAMSTUDIO_MODEL", "gpt-5.1")
        self._api_version = api_version or os.getenv("BEAMSTUDIO_API_VERSION", "2025-04-01-preview")
        
        if not self._api_key:
            logger.warning("BeamStudio API key not configured. Set BEAMSTUDIO_API_KEY environment variable.")
        
        # Default generation parameters for API
        # Note: Reasoning models (gpt-5.1, o1, etc.) use tokens for internal chain-of-thought
        # before producing output, so we need higher limits (typically 500-2000+ for reasoning alone)
        # Complex multi-part queries can consume 1500+ reasoning tokens
        self.default_gen_args = {
            "temperature": 0.7,
            "max_completion_tokens": 4000,  # Very high for complex reasoning queries
        }
        
        # Store last token usage for display
        self.last_usage = None
        
        logger.info(f"Initialized BeamStudio client: model={self._model}, base_url={self._base_url}")
    
    @property
    def model_name(self) -> str:
        return f"beamstudio/{self._model}"
    
    @property
    def provider(self) -> str:
        return "beamstudio"
    
    def _build_endpoint(self) -> str:
        """Build the API endpoint URL."""
        return f"{self._base_url}/v1/llm/openai/deployments/{self._model}/chat/completions"
    
    def _convert_prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Convert a plain text prompt to chat messages format."""
        return [
            {"role": "system", "content": "You are a helpful RAG assistant. Answer questions based on the provided context. Cite sources when available."},
            {"role": "user", "content": prompt}
        ]
    
    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize messages for Azure OpenAI API.
        Only adds system message if none exists.
        """
        if not messages:
            return [{"role": "system", "content": "You are a helpful assistant."}]
        
        # If messages already have a system role, use them as-is
        has_system = any(m.get('role') == 'system' for m in messages)
        if has_system:
            return messages
        
        # Add default system message only if missing
        return [{"role": "system", "content": "You are a helpful assistant."}] + messages
    
    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Generate text using the BeamStudio API."""
        if not self._api_key:
            raise RuntimeError("BeamStudio API key not configured. Set BEAMSTUDIO_API_KEY environment variable.")
        
        start_time = datetime.now()
        span = None
        
        # Start OpenTelemetry span if available
        if OPENTELEMETRY_AVAILABLE:
            try:
                span = create_trace_span(
                    "llm_generation",
                    attributes={
                        "llm.provider": self.provider,
                        "llm.model": self._model,
                        "llm.prompt_length": len(prompt),
                        "llm.api_endpoint": self._build_endpoint(),
                        "span.type": "llm"
                    }
                )
                span.__enter__()
            except Exception as e:
                logger.debug(f"Failed to create OTEL span: {e}")
        
        try:
            # Use provided messages or convert prompt
            if messages is None:
                messages = self._convert_prompt_to_messages(prompt)
            else:
                # Normalize messages to ensure proper format
                messages = self._normalize_messages(messages)
            
            # Build request
            endpoint = self._build_endpoint()
            headers = {
                "Content-Type": "application/json",
                "api-key": self._api_key,
            }
            
            # Merge default args with provided kwargs
            gen_args = {**self.default_gen_args}
            
            # Map common parameter names from HuggingFace to OpenAI format
            if 'max_new_tokens' in kwargs:
                gen_args['max_completion_tokens'] = kwargs.pop('max_new_tokens')
            if 'max_completion_tokens' in kwargs:
                gen_args['max_completion_tokens'] = kwargs.pop('max_completion_tokens')
            if 'temperature' in kwargs:
                gen_args['temperature'] = kwargs.pop('temperature')
            if 'top_p' in kwargs:
                gen_args['top_p'] = kwargs.pop('top_p')
            
            # Filter out HuggingFace-specific parameters that Azure OpenAI doesn't support
            unsupported_params = [
                'repetition_penalty', 'no_repeat_ngram_size', 'do_sample',
                'pad_token_id', 'eos_token_id', 'num_beams', 'early_stopping',
                'length_penalty', 'num_return_sequences', 'use_cache'
            ]
            for param in unsupported_params:
                kwargs.pop(param, None)
                gen_args.pop(param, None)
            
            # Models that only support temperature=1 (BeamStudio API returns 400 otherwise)
            if self._model and "gpt-5-mini" in self._model.lower():
                gen_args["temperature"] = 1
            
            # Build request body with only supported parameters
            body = {
                "messages": messages,
                **gen_args
            }
            
            # Add API version as query parameter
            params = {"api-version": self._api_version}
            
            # Log the full request for debugging
            logger.info(f"[BEAMSTUDIO_REQUEST] endpoint={endpoint}, model={self._model}")
            logger.info(f"[BEAMSTUDIO_REQUEST] messages count={len(messages)}")
            for i, msg in enumerate(messages):
                content_preview = msg.get('content', '')[:100]
                logger.info(f"[BEAMSTUDIO_REQUEST] message[{i}] role={msg.get('role')}, content={content_preview}...")
            logger.info(f"[BEAMSTUDIO_REQUEST] gen_args={gen_args}")
            
            # Make API call
            response = requests.post(
                endpoint,
                headers=headers,
                params=params,
                json=body,
                timeout=60
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"BeamStudio API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                if span:
                    try:
                        set_span_attribute(span, "error", error_msg)
                        set_span_attribute(span, "http.status_code", response.status_code)
                    except Exception:
                        pass
                raise RuntimeError(error_msg)
            
            # Parse response
            data = response.json()
            
            # Log raw response for debugging
            logger.info(f"BeamStudio raw API response keys: {data.keys() if isinstance(data, dict) else type(data)}")
            
            # Log prompt filter results (Azure content filter on INPUT)
            if 'prompt_filter_results' in data:
                logger.info(f"BeamStudio prompt_filter_results: {data['prompt_filter_results']}")
            
            if 'choices' in data:
                logger.info(f"BeamStudio choices count: {len(data['choices'])}")
                if len(data['choices']) > 0:
                    choice = data['choices'][0]
                    logger.info(f"BeamStudio first choice keys: {choice.keys() if isinstance(choice, dict) else type(choice)}")
                    
                    # Log content filter results if present
                    if 'content_filter_results' in choice:
                        logger.info(f"BeamStudio content_filter_results: {choice['content_filter_results']}")
                    
                    if 'message' in choice:
                        msg = choice['message']
                        logger.info(f"BeamStudio message keys: {msg.keys() if isinstance(msg, dict) else type(msg)}")
                        
                        # Log refusal if present
                        if 'refusal' in msg:
                            refusal_val = msg['refusal']
                            logger.info(f"BeamStudio refusal field: '{refusal_val}' (type={type(refusal_val)})")
                        
                        if 'content' in msg:
                            content_preview = msg['content'][:200] if msg['content'] else "(empty)"
                            logger.info(f"BeamStudio content preview: {content_preview}")
            
            # Extract generated text from OpenAI-format response
            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                if 'message' in choice:
                    msg = choice['message']
                    
                    # Check for refusal (Azure content filter)
                    refusal = msg.get('refusal')
                    if refusal:
                        logger.warning(f"BeamStudio API REFUSED to answer: {refusal}")
                        # Return a helpful message about the refusal
                        output = f"The model declined to answer: {refusal}"
                    elif 'content' in msg:
                        output = msg['content']
                        if not output:
                            # Check content_filter_results for why it might be empty
                            filter_results = choice.get('content_filter_results', {})
                            if filter_results:
                                logger.warning(f"BeamStudio content filter results: {filter_results}")
                            logger.warning("BeamStudio API returned empty content string (no refusal message)")
                            # Try to provide a fallback response
                            output = ""
                    else:
                        output = str(choice)
                        logger.warning(f"BeamStudio response missing content in message: {msg}")
                else:
                    output = str(choice)
                    logger.warning(f"BeamStudio response missing message structure: {choice}")
            else:
                output = str(data)
                logger.warning(f"BeamStudio response missing choices: {data}")
            
            # Extract and store token usage
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.last_usage = None
            if 'usage' in data:
                usage = data['usage']
                self.last_usage = {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'reasoning_tokens': usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0) if 'completion_tokens_details' in usage else 0,
                    'latency_ms': latency_ms,
                    'finish_reason': data['choices'][0].get('finish_reason', 'unknown') if data.get('choices') else 'unknown'
                }
                # Clear, visible logging of token usage
                logger.info(f"[TOKEN_USAGE] prompt={self.last_usage['prompt_tokens']}, completion={self.last_usage['completion_tokens']}, "
                           f"reasoning={self.last_usage['reasoning_tokens']}, total={self.last_usage['total_tokens']}, "
                           f"finish_reason={self.last_usage['finish_reason']}, latency={latency_ms:.0f}ms")
            
            # Add span attributes for observability
            if span:
                try:
                    set_span_attribute(span, "llm.completion_length", len(output))
                    set_span_attribute(span, "llm.latency_ms", latency_ms)
                    set_span_attribute(span, "http.status_code", response.status_code)
                    
                    # Token usage if available
                    if self.last_usage:
                        set_span_attribute(span, "llm.tokens.prompt", self.last_usage['prompt_tokens'])
                        set_span_attribute(span, "llm.tokens.completion", self.last_usage['completion_tokens'])
                        set_span_attribute(span, "llm.tokens.total", self.last_usage['total_tokens'])
                        set_span_attribute(span, "llm.tokens.reasoning", self.last_usage['reasoning_tokens'])
                except Exception:
                    pass
            
            logger.debug(f"BeamStudio API response: {len(output)} chars, latency={latency_ms:.0f}ms")
            return output
            
        except requests.exceptions.RequestException as e:
            logger.error(f"BeamStudio API request failed: {e}")
            if span:
                try:
                    set_span_attribute(span, "error", str(e))
                except Exception:
                    pass
            raise RuntimeError(f"BeamStudio API request failed: {e}")
        except Exception as e:
            logger.error(f"BeamStudio generation failed: {e}")
            if span:
                try:
                    set_span_attribute(span, "error", str(e))
                except Exception:
                    pass
            raise
        finally:
            if span:
                try:
                    span.__exit__(None, None, None)
                except Exception:
                    pass


def get_llm_client(
    provider: str = "local",
    model_name: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: 'local' for Hugging Face, 'beamstudio' for BeamStudio API
        model_name: Model name/ID (provider-specific)
        device: Device for local models
        **kwargs: Additional provider-specific arguments
        
    Returns:
        LLMClient instance
    """
    if provider == "local":
        return LocalLLMClient(
            model_name=model_name or "Felladrin/Smol-Llama-101M-Chat-v1",
            device=device,
            **kwargs
        )
    elif provider == "beamstudio":
        return BeamStudioClient(
            model=model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Convenience function to list available BeamStudio models
def get_beamstudio_models() -> List[str]:
    """Return list of available BeamStudio models (gpt-5.1 first as default)."""
    return ["gpt-5.1", "gpt-4o", "gpt-5-mini"]


def is_beamstudio_configured() -> bool:
    """Check if BeamStudio API is configured."""
    api_key = os.getenv("BEAMSTUDIO_API_KEY")
    return api_key is not None and api_key != "your-api-key-here"
