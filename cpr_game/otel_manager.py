"""OpenTelemetry manager for CPR game tracing.

Provides single source of truth for all tracing using OpenTelemetry.
Traces are exported via OTLP to configured receivers (Langfuse, LangSmith, etc.).
"""

from typing import Dict, Optional, Any
import logging
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from .config import CONFIG
from .logger_setup import get_logger

logger = get_logger(__name__)

# Suppress verbose OTLP exporter connection errors
# These errors occur when the collector is not running, which is acceptable
_otlp_logger = logging.getLogger("opentelemetry.exporter.otlp.proto.grpc.exporter")
_otlp_logger.setLevel(logging.CRITICAL)  # Only show critical errors, suppress UNAVAILABLE errors
_otlp_http_logger = logging.getLogger("opentelemetry.exporter.otlp.proto.http.exporter")
_otlp_http_logger.setLevel(logging.CRITICAL)


class GracefulOTLPExporter(SpanExporter):
    """Wrapper around OTLP exporter that handles connection failures.
    
    NO SILENT FAILURES - all errors are logged at ERROR level.
    """
    
    def __init__(self, exporter: SpanExporter, endpoint: str):
        """Initialize exporter wrapper.
        
        Args:
            exporter: The underlying OTLP exporter
            endpoint: The endpoint being used (for logging)
        """
        self.exporter = exporter
        self.endpoint = endpoint
        self._failure_count = 0
    
    def export(self, spans):
        """Export spans with error logging.
        
        NO SILENT FAILURES - all failures are logged at ERROR level.
        """
        if not spans:
            return SpanExportResult.SUCCESS
        
        # Log export attempt for debugging (INFO level so it's visible)
        logger.info(f"Exporting {len(spans)} span(s) to {self.endpoint}")
        for span in spans:
            trace_id = span.context.trace_id if span.context.trace_id else None
            trace_id_str = f"{trace_id:x}" if trace_id else "none"
            logger.debug(f"  - Span: {span.name} (trace_id: {trace_id_str})")
        
        try:
            result = self.exporter.export(spans)
            if result == SpanExportResult.SUCCESS:
                logger.info(f"Successfully exported {len(spans)} span(s) to {self.endpoint}")
                # Reset failure count on success
                if self._failure_count > 0:
                    logger.info(f"OpenTelemetry export to {self.endpoint} recovered after {self._failure_count} failures")
                    self._failure_count = 0
            elif result == SpanExportResult.FAILURE:
                self._failure_count += 1
                # Log every failure at ERROR level - NO SILENT FAILURES
                logger.error(
                    f"OpenTelemetry export to {self.endpoint} FAILED (failure #{self._failure_count}). "
                    f"Traces are NOT being exported. "
                    f"Check if collector is running: 'docker ps | grep otel-collector' or 'docker-compose ps otel-collector'. "
                    f"To start: 'docker-compose up -d otel-collector'. "
                    f"To disable: set OTEL_ENABLED=false"
                )
            return result
        except Exception as e:
            self._failure_count += 1
            # Log every exception at ERROR level with full stack trace - NO SILENT FAILURES
            logger.error(
                f"OpenTelemetry export to {self.endpoint} EXCEPTION (failure #{self._failure_count}): {e}. "
                f"Traces are NOT being exported. "
                f"Check if collector is running: 'docker ps | grep otel-collector' or 'docker-compose ps otel-collector'. "
                f"To start: 'docker-compose up -d otel-collector'. "
                f"To disable: set OTEL_ENABLED=false",
                exc_info=True
            )
            # NOTE: OpenTelemetry SpanExporter.export() interface requires returning SpanExportResult,
            # NOT raising exceptions. However, we log ALL failures at ERROR level with full stack trace.
            # The failure is NOT silent - it's logged and visible. The OTel SDK will handle the failure result.
            # If you need to stop execution on export failures, check _failure_count or monitor ERROR logs.
            return SpanExportResult.FAILURE
    
    def shutdown(self):
        """Shutdown the underlying exporter."""
        try:
            self.exporter.shutdown()
        except Exception as e:
            logger.error(
                f"Error shutting down OpenTelemetry exporter for {self.endpoint}: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to shutdown OpenTelemetry exporter: {e}") from e
    
    def force_flush(self, timeout_millis: int = 30000):
        """Force flush the underlying exporter."""
        try:
            return self.exporter.force_flush(timeout_millis)
        except Exception as e:
            logger.error(
                f"Error flushing OpenTelemetry exporter for {self.endpoint}: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to flush OpenTelemetry exporter: {e}") from e


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
        endpoint = self.config.get("otel_endpoint", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"))  # HTTP port - exporter will append /v1/traces automatically
        protocol = self.config.get("otel_protocol", os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"))  # Langfuse requires HTTP
        
        # Check if collector is expected to be running (when using localhost endpoint)
        if endpoint.startswith("http://localhost") or endpoint.startswith("http://127.0.0.1"):
            # Check if collector is running (best effort - don't fail if check fails)
            try:
                import subprocess
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=otel-collector", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    logger.info(f"OTel collector container detected: {result.stdout.strip()}")
                else:
                    logger.warning(
                        f"OTel collector does not appear to be running. "
                        f"Traces will not be exported to Langfuse/LangSmith. "
                        f"Start with: 'docker-compose up -d otel-collector'"
                    )
            except Exception as e:
                logger.error(f"Could not check if OTel collector is running: {e}", exc_info=True)
                raise RuntimeError(f"Failed to check OTel collector status: {e}") from e
        
        # Check Langfuse API keys if Langfuse is expected
        otel_receiver = self.config.get("otel_receiver", os.getenv("OTEL_RECEIVER", "both")).lower()
        if otel_receiver in ("langfuse", "both"):
            langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
            langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
            if not langfuse_public_key or not langfuse_secret_key:
                logger.error(
                    f"LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY are not set. "
                    f"Traces will NOT be exported to Langfuse. "
                    f"Set these environment variables or add them to .env file."
                )
        
        # Resource attributes
        resource_attrs = {
            "service.name": service_name,
            "service.version": self.config.get("otel_service_version", "1.0.0"),
        }
        
        # Add custom resource attributes from config
        custom_attrs = self.config.get("otel_resource_attributes", {})
        resource_attrs.update(custom_attrs)
        
        resource = Resource.create(resource_attrs)
        
        # Check if we should reuse existing provider BEFORE creating a new one
        # This prevents "Overriding of current TracerProvider is not allowed" warnings
        # when multiple games run in parallel
        # IMPORTANT: Only reuse if it's an SDK provider (not ProxyTracerProvider) so we can add span processors
        from opentelemetry.trace import NoOpTracerProvider, ProxyTracerProvider
        from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
        
        current_provider = trace.get_tracer_provider()
        reuse_provider = False
        
        # Only reuse if it's an actual SDK provider (not ProxyTracerProvider or NoOpTracerProvider)
        if isinstance(current_provider, SDKTracerProvider):
            # Provider already exists and is an SDK provider, reuse it
            logger.debug("Reusing existing SDK tracer provider")
            self.tracer_provider = current_provider
            self._provider_owned = False  # We didn't create it, so don't shutdown
            reuse_provider = True
        else:
            # Create new tracer provider (either NoOpTracerProvider or ProxyTracerProvider exists)
            logger.debug(f"Creating new tracer provider (current is {type(current_provider).__name__})")
            self.tracer_provider = TracerProvider(resource=resource)
            self._provider_owned = True
        
        # Configure exporter based on protocol
        # IMPORTANT: Langfuse only supports HTTP/protobuf, not gRPC
        if protocol == "grpc":
            logger.warning(
                "gRPC protocol is configured but Langfuse only supports HTTP/protobuf. "
                "Switching to HTTP protocol. Set OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf to avoid this warning."
            )
            protocol = "http/protobuf"
            # Update endpoint to use HTTP port (exporter will append /v1/traces automatically)
            if endpoint.endswith(":4317"):
                endpoint = endpoint.replace(":4317", ":4318")
            # Remove /v1/traces if present - HTTP exporter will add it automatically
            if endpoint.endswith("/v1/traces"):
                endpoint = endpoint[:-10]  # Remove "/v1/traces"
            elif endpoint.endswith("/v1/traces/"):
                endpoint = endpoint[:-11]  # Remove "/v1/traces/"
        
        # HTTP protocol (required for Langfuse)
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPExporter
        # IMPORTANT: When endpoint is passed explicitly, the exporter does NOT append /v1/traces automatically
        # We need to either pass None (to use default) or include /v1/traces in the endpoint
        # The exporter's _append_trace_path is only called when using default/env var, not when passing endpoint explicitly
        # So we need to append /v1/traces ourselves if not already present
        if not endpoint.endswith("/v1/traces"):
            if endpoint.endswith("/"):
                endpoint_with_path = endpoint + "v1/traces"
            else:
                endpoint_with_path = endpoint + "/v1/traces"
        else:
            endpoint_with_path = endpoint
        
        logger.info(f"Creating OTLP HTTP exporter with endpoint: {endpoint_with_path}")
        base_exporter = HTTPExporter(
            endpoint=endpoint_with_path,
        )
        # Log the actual endpoint the exporter is using
        logger.info(f"OTLP HTTP exporter configured with endpoint: {base_exporter._endpoint}")
        
        # Wrap exporter to handle connection failures gracefully
        exporter = GracefulOTLPExporter(base_exporter, endpoint)
        
        # Add OTLP span processor (for collector/LangFuse/LangSmith)
        # Use shorter export interval to send traces faster (default is 5s)
        span_processor = BatchSpanProcessor(
            exporter,
            max_queue_size=2048,
            export_timeout_millis=30000,
            schedule_delay_millis=500  # Send batches every 500ms instead of default 5s
        )
        
        # Set global tracer provider if we created a new one (must be set before adding processors)
        if not reuse_provider:
            try:
                trace.set_tracer_provider(self.tracer_provider)
                logger.debug("Set new tracer provider as global")
            except Exception as e:
                logger.error(f"Failed to set tracer provider: {e}", exc_info=True)
                raise RuntimeError(f"Failed to set tracer provider: {e}") from e
                    logger.error(f"Failed to get existing tracer provider: {final_error}", exc_info=True)
                    raise
        
        # Always add span processor to the provider (whether we created it or reused it)
        if isinstance(self.tracer_provider, SDKTracerProvider):
            self.tracer_provider.add_span_processor(span_processor)
            self._span_processor = span_processor
            logger.debug(f"Added span processor to tracer provider (type: {type(self.tracer_provider).__name__})")
        else:
            # If provider is ProxyTracerProvider, try to get the actual SDK provider
            from opentelemetry.trace import ProxyTracerProvider
            if isinstance(self.tracer_provider, ProxyTracerProvider):
                # Try to get the actual provider from the proxy
                try:
                    actual_provider = trace.get_tracer_provider()
                    if isinstance(actual_provider, SDKTracerProvider):
                        actual_provider.add_span_processor(span_processor)
                        self._span_processor = span_processor
                        logger.debug(f"Added span processor to actual SDK provider behind ProxyTracerProvider")
                    else:
                        logger.warning(
                            f"Actual provider is not SDK provider (type: {type(actual_provider).__name__}), "
                            f"cannot add span processor. Traces may not be exported!"
                        )
                        self._span_processor = None
                except Exception as e:
                    logger.error(f"Failed to get actual provider from ProxyTracerProvider: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to get actual provider from ProxyTracerProvider: {e}") from e
            else:
                logger.error(
                    f"Tracer provider is not SDK provider (type: {type(self.tracer_provider).__name__}), "
                    f"cannot add span processor. Traces will NOT be exported!"
                )
                self._span_processor = None
        
        # Get tracer
        self.tracer = trace.get_tracer(service_name)
        
        # Instrument LangChain if available
        try:
            from opentelemetry.instrumentation.langchain import LangChainInstrumentor
            langchain_instrumentor = LangChainInstrumentor()
            langchain_instrumentor.instrument(tracer_provider=self.tracer_provider)
            logger.info("[OK] LangChain auto-instrumentation enabled")
        except ImportError:
            # LangChain instrumentation is optional, so ImportError is acceptable
            logger.debug("LangChain instrumentation not available - install opentelemetry-instrumentation-langchain for auto-instrumentation")
        except Exception as e:
            error_msg = f"Could not enable LangChain auto-instrumentation: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
        logger.info(f"[OK] OpenTelemetry initialized: service={service_name}, endpoint={endpoint}")
    
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
        try:
            # First try to flush via span processor (works even if provider is reused)
            if hasattr(self, '_span_processor') and self._span_processor:
                try:
                    self._span_processor.force_flush(timeout_millis=5000)
                    logger.debug("Flushed spans via span processor")
                except Exception as e:
                    logger.error(f"Span processor flush failed: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to flush span processor: {e}") from e
            
            # Also try provider flush if we own it
            if self.tracer_provider and self._provider_owned:
                try:
                    # Check if provider has force_flush method (ProxyTracerProvider doesn't)
                    if hasattr(self.tracer_provider, 'force_flush'):
                        self.tracer_provider.force_flush(timeout_millis=5000)
                        logger.debug("Flushed spans via tracer provider")
                except AttributeError:
                    # Provider doesn't have force_flush - that's okay
                    logger.debug("Tracer provider doesn't support force_flush")
                except Exception as e:
                    logger.error(f"Tracer provider flush failed: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to flush tracer provider: {e}") from e
            
            # If provider is a ProxyTracerProvider, try to get the actual SDK provider
            from opentelemetry.trace import ProxyTracerProvider
            if isinstance(self.tracer_provider, ProxyTracerProvider):
                try:
                    # ProxyTracerProvider delegates to the actual provider
                    # Try to get the actual provider and flush its processors
                    actual_provider = trace.get_tracer_provider()
                    if hasattr(actual_provider, '_span_processors'):
                        for processor in actual_provider._span_processors:
                            try:
                                processor.force_flush(timeout_millis=5000)
                                logger.debug("Flushed spans via processor from ProxyTracerProvider")
                            except Exception as e:
                                logger.error(f"Processor flush failed: {e}", exc_info=True)
                                raise RuntimeError(f"Failed to flush processor from ProxyTracerProvider: {e}") from e
                except Exception as e:
                    logger.error(f"Failed to flush via ProxyTracerProvider: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to flush via ProxyTracerProvider: {e}") from e
            
            # If provider is reused, try to get the SDK provider and flush its processors
            if self.tracer_provider and not self._provider_owned:
                try:
                    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
                    if isinstance(self.tracer_provider, SDKTracerProvider):
                        # Flush all span processors
                        for processor in self.tracer_provider._span_processors:
                            if hasattr(processor, 'force_flush'):
                                processor.force_flush(timeout_millis=5000)
                        logger.debug("Flushed spans via SDK tracer provider processors")
                except Exception as e:
                    logger.error(f"Failed to flush via SDK provider: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to flush via SDK provider: {e}") from e
        except Exception as e:
            logger.error(f"Error flushing OTel spans: {e}", exc_info=True)
            raise RuntimeError(f"Failed to flush OTel spans: {e}") from e
    
    def shutdown(self):
        """Shutdown tracer provider."""
        if self.tracer_provider and self._provider_owned:
            try:
                # Check if provider has shutdown method (ProxyTracerProvider doesn't)
                if hasattr(self.tracer_provider, 'shutdown'):
                    self.tracer_provider.shutdown()
                else:
                    # ProxyTracerProvider or similar - can't shutdown directly
                    logger.debug("Tracer provider doesn't support shutdown (likely ProxyTracerProvider)")
            except AttributeError:
                # Provider doesn't have shutdown - that's okay
                logger.debug("Tracer provider doesn't support shutdown")
            except Exception as e:
                logger.error(f"Error shutting down OTel tracer provider: {e}", exc_info=True)
                raise RuntimeError(f"Failed to shutdown OTel tracer provider: {e}") from e
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()

