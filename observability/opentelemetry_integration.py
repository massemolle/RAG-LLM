"""
OpenTelemetry Integration for Langfuse Export
Uses native OpenTelemetry OTLP HTTP exporter to send traces to Langfuse
"""

import os
import logging
import base64
from typing import Optional, Any

logger = logging.getLogger(__name__)

# OpenTelemetry imports
OPENTELEMETRY_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry not available - install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http")
    trace = None

# Global tracer
_tracer: Optional[Any] = None


def initialize_opentelemetry(
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_host: Optional[str] = None,
    service_name: str = "rag-llm-system",
    service_version: Optional[str] = None
) -> bool:
    """
    Initialize OpenTelemetry with Langfuse OTLP endpoint
    
    Langfuse has native OpenTelemetry support via the /api/public/otel endpoint.
    This function configures the OTLP HTTP exporter to send traces directly to Langfuse.
    
    Args:
        langfuse_public_key: Langfuse public key (or LANGFUSE_PUBLIC_KEY env var)
        langfuse_secret_key: Langfuse secret key (or LANGFUSE_SECRET_KEY env var)
        langfuse_host: Langfuse host URL (or LANGFUSE_HOST/LANGFUSE_BASE_URL env var, default: https://cloud.langfuse.com)
        service_name: Service name for traces
        service_version: Service version
    
    Returns:
        True if initialized successfully
    """
    global _tracer
    
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available")
        return False
    
    try:
        # Get Langfuse credentials
        public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = langfuse_host or os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        
        if not public_key or not secret_key:
            logger.warning("Langfuse keys not configured - set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
            return False
        
        # Construct OTLP endpoint
        # Langfuse OTLP endpoint for traces is /api/public/otel/v1/traces
        # According to Langfuse docs, use signal-specific endpoint for traces
        if host.endswith('/'):
            host = host.rstrip('/')
        otlp_endpoint = f"{host}/api/public/otel/v1/traces"
        
        # Create Basic Auth header
        # Format: "pk-lf-xxx:sk-lf-xxx" base64 encoded
        auth_string = f"{public_key}:{secret_key}"
        auth_bytes = auth_string.encode('utf-8')
        auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        # Create resource with service information
        resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version or os.getenv("APP_RELEASE", "unknown"),
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python"
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Create OTLP HTTP exporter for Langfuse
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            headers={
                "Authorization": f"Basic {auth_b64}"
            }
        )
        
        # Add span processor with batching
        span_processor = BatchSpanProcessor(
            exporter,
            max_queue_size=100,
            export_timeout_millis=5000,
            max_export_batch_size=50
        )
        tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Get tracer
        _tracer = trace.get_tracer(__name__)
        
        logger.info(f"OpenTelemetry initialized with Langfuse OTLP endpoint: {otlp_endpoint}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)
        return False


def get_tracer():
    """Get OpenTelemetry tracer"""
    return _tracer


def get_current_span():
    """Get current active span"""
    if not OPENTELEMETRY_AVAILABLE:
        return None
    return trace.get_current_span()


def create_trace_span(
    name: str,
    attributes: Optional[dict] = None,
    kind: Optional[Any] = None
):
    """
    Create a new span context manager
    
    Args:
        name: Span name
        attributes: Span attributes (use langfuse.* namespace for Langfuse-specific attributes)
        kind: Span kind (trace.SpanKind.SERVER, CLIENT, etc.)
    
    Returns:
        Span context manager (use with 'with' statement)
    """
    if not _tracer:
        from contextlib import nullcontext
        return nullcontext()
    
    if not OPENTELEMETRY_AVAILABLE:
        from contextlib import nullcontext
        return nullcontext()
    
    # Set default span kind if not provided
    if kind is None:
        from opentelemetry.trace import SpanKind
        kind = SpanKind.INTERNAL
    
    # Convert attributes to proper format
    span_attrs = {}
    if attributes:
        for key, value in attributes.items():
            # Convert complex types to JSON strings
            if isinstance(value, (dict, list)):
                import json
                span_attrs[key] = json.dumps(value)
            else:
                span_attrs[key] = value
    
    span = _tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=span_attrs
    )
    
    return span


def add_span_event(
    name: str,
    attributes: Optional[dict] = None
):
    """Add event to current span"""
    span = get_current_span()
    if span:
        span.add_event(name, attributes=attributes or {})


def set_span_attribute(key: str, value: Any):
    """Set attribute on current span"""
    span = get_current_span()
    if span:
        # Convert value to JSON-serializable format
        if isinstance(value, (dict, list)):
            import json
            value = json.dumps(value)
        span.set_attribute(key, value)


def set_span_status(status_code: StatusCode, description: Optional[str] = None):
    """Set status on current span"""
    span = get_current_span()
    if span:
        span.set_status(Status(status_code, description))


def flush():
    """Flush all spans to Langfuse via OTLP"""
    if OPENTELEMETRY_AVAILABLE and _tracer:
        try:
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, 'force_flush'):
                tracer_provider.force_flush(timeout_millis=5000)
            elif hasattr(tracer_provider, '_span_processors'):
                for processor in getattr(tracer_provider, '_span_processors', []):
                    if hasattr(processor, 'force_flush'):
                        processor.force_flush(timeout_millis=5000)
        except Exception:
            pass
