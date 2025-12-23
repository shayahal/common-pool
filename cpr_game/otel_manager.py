"""OpenTelemetry manager for CPR game tracing.

Provides single source of truth for all tracing using OpenTelemetry.
Traces are exported via OTLP to configured receivers (Langfuse, LangSmith, etc.).
"""

from typing import Dict, Optional, Any
import logging
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from .config import CONFIG
from .logger_setup import get_logger

logger = get_logger(__name__)


class OTelManager:
    """Manager for OpenTelemetry tracing (single source of truth).
    
    Initializes OTel SDK and configures exporters to send traces
    to configured receivers (Langfuse, LangSmith, etc.).
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize OpenTelemetry manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config if config is not None else CONFIG
        
        # Check if OTel is enabled
        if not self.config.get("otel_enabled", True):
            logger.info("OpenTelemetry is disabled in configuration")
            self.tracer = None
            self.tracer_provider = None
            return
        
        # Get OTel configuration
        service_name = self.config.get("otel_service_name", "cpr-game")
        endpoint = self.config.get("otel_endpoint", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))
        protocol = self.config.get("otel_protocol", os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"))
        
        # Resource attributes
        resource_attrs = {
            "service.name": service_name,
            "service.version": self.config.get("otel_service_version", "1.0.0"),
        }
        
        # Add custom resource attributes from config
        custom_attrs = self.config.get("otel_resource_attributes", {})
        resource_attrs.update(custom_attrs)
        
        resource = Resource.create(resource_attrs)
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Configure exporter based on protocol
        if protocol == "grpc":
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                insecure=endpoint.startswith("http://") or "localhost" in endpoint
            )
        else:
            # HTTP protocol
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPExporter
            exporter = HTTPExporter(
                endpoint=endpoint,
            )
        
        # Add span processor
        span_processor = BatchSpanProcessor(exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(service_name)
        
        # Instrument LangChain if available
        try:
            from opentelemetry.instrumentation.langchain import LangChainInstrumentor
            langchain_instrumentor = LangChainInstrumentor()
            langchain_instrumentor.instrument(tracer_provider=self.tracer_provider)
            logger.info("✓ LangChain auto-instrumentation enabled")
        except ImportError:
            logger.debug("LangChain instrumentation not available - install opentelemetry-instrumentation-langchain for auto-instrumentation")
        except Exception as e:
            logger.warning(f"Could not enable LangChain auto-instrumentation: {e}")
        
        logger.info(f"✓ OpenTelemetry initialized: service={service_name}, endpoint={endpoint}")
    
    def get_tracer(self):
        """Get the OTel tracer.
        
        Returns:
            Tracer instance or None if OTel is disabled
        """
        return self.tracer
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """Start a new span.
        
        Args:
            name: Span name
            attributes: Span attributes
            **kwargs: Additional span options
            
        Returns:
            Span context manager
        """
        if self.tracer is None:
            # Return a no-op context manager if OTel is disabled
            from contextlib import nullcontext
            return nullcontext()
        
        span = self.tracer.start_as_current_span(name, attributes=attributes, **kwargs)
        return span
    
    def flush(self):
        """Flush all pending spans."""
        if self.tracer_provider:
            try:
                self.tracer_provider.force_flush()
            except Exception as e:
                logger.warning(f"Error flushing OTel spans: {e}")
    
    def shutdown(self):
        """Shutdown tracer provider."""
        if self.tracer_provider:
            try:
                self.tracer_provider.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down OTel tracer provider: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()

