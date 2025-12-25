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
        
        try:
            result = self.exporter.export(spans)
            if result == SpanExportResult.SUCCESS:
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
            # Return failure but don't raise - allows application to continue
            # But errors are ALWAYS logged at ERROR level
            return SpanExportResult.FAILURE
    
    def shutdown(self):
        """Shutdown the underlying exporter."""
        try:
            self.exporter.shutdown()
        except Exception as e:
            # Log shutdown errors - never fail silently
            # Don't re-raise during cleanup to avoid crashing the application
            logger.error(
                f"Error shutting down OpenTelemetry exporter for {self.endpoint}: {e}",
                exc_info=True
            )
    
    def force_flush(self, timeout_millis: int = 30000):
        """Force flush the underlying exporter."""
        try:
            return self.exporter.force_flush(timeout_millis)
        except Exception as e:
            # Log flush errors - never fail silently
            logger.error(
                f"Error flushing OpenTelemetry exporter for {self.endpoint}: {e}",
                exc_info=True
            )
            # Return False to indicate failure, but error is logged
            return False


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
                # Don't fail initialization if docker check fails
                logger.debug(f"Could not check if OTel collector is running: {e}")
        
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
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        
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
        span_processor = BatchSpanProcessor(exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # Add FalkorDB exporter if enabled
        # NO SILENT FAILURES - if FalkorDB is enabled, it MUST work
        if self.config.get("falkordb_enabled", False):
            try:
                from .falkordb_exporter import FalkorDBExporter
                falkordb_exporter = FalkorDBExporter(
                    host=self.config.get("falkordb_host", "localhost"),
                    port=self.config.get("falkordb_port", 6379),
                    username=self.config.get("falkordb_username"),
                    password=self.config.get("falkordb_password"),
                    group_id=self.config.get("falkordb_group_id", "cpr-game-traces"),
                    enabled=self.config.get("falkordb_enabled", True),
                    max_retries=self.config.get("falkordb_max_retries", 10),  # Increased for rate limits
                    base_retry_delay=self.config.get("falkordb_base_retry_delay", 5.0),  # Increased base delay
                    max_retry_delay=self.config.get("falkordb_max_retry_delay", 300.0),  # 5 minutes max delay
                    export_timeout=self.config.get("falkordb_export_timeout", 60 * 60.0),  # 60 minutes default
                    episode_rate_limit=self.config.get("falkordb_episode_rate_limit", 1.0)  # Throttle episode additions
                )
                # Only add processor if exporter is actually enabled
                if falkordb_exporter.enabled:
                    falkordb_processor = BatchSpanProcessor(falkordb_exporter)
                    self.tracer_provider.add_span_processor(falkordb_processor)
                    logger.info("FalkorDB exporter enabled and registered")
                else:
                    # If enabled in config but exporter is disabled, this is a FATAL error
                    error_msg = (
                        "FalkorDB is enabled in config but exporter initialization failed. "
                        "This is a FATAL error - FalkorDB export will not work. "
                        "Check logs for details."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            except ImportError as e:
                # Missing dependency is a fatal error if FalkorDB is enabled
                error_msg = (
                    f"FalkorDB is enabled but required dependencies are missing: {e}. "
                    "Install with: pip install graphiti-core[falkordb]"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            except Exception as e:
                # Any other error is fatal if FalkorDB is enabled
                error_str = str(e).lower()
                if "api_key" in error_str or "api key" in error_str:
                    error_msg = (
                        f"FalkorDB is enabled but OPENAI_API_KEY is missing. "
                        f"This is a FATAL error. Set OPENAI_API_KEY environment variable. Error: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                else:
                    error_msg = f"FalkorDB is enabled but initialization failed. This is a FATAL error: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg) from e
        
        # Track if we created the provider ourselves (for flush/shutdown)
        self._provider_owned = True
        
        # Set global tracer provider only if one doesn't exist
        # This prevents "Overriding of current TracerProvider is not allowed" warnings
        # when multiple games run in parallel
        # IMPORTANT: If FalkorDB is enabled, we MUST use our own provider to ensure
        # the exporter is registered, because ProxyTracerProvider doesn't support add_span_processor
        try:
            current_provider = trace.get_tracer_provider()
            # Check if it's a real provider (not NoOpTracerProvider)
            # NoOpTracerProvider doesn't have resource attribute set properly
            from opentelemetry.trace import NoOpTracerProvider, ProxyTracerProvider
            from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
            
            # If FalkorDB is enabled, we need our own provider (not ProxyTracerProvider)
            falkordb_enabled = self.config.get("falkordb_enabled", False)
            is_proxy = isinstance(current_provider, ProxyTracerProvider)
            is_sdk = isinstance(current_provider, SDKTracerProvider)
            
            if not isinstance(current_provider, NoOpTracerProvider):
                if falkordb_enabled and (is_proxy or not is_sdk):
                    # We need our own provider for FalkorDB - ProxyTracerProvider can't add processors
                    logger.info("Creating new tracer provider for FalkorDB export (cannot use ProxyTracerProvider)")
                    # Don't reuse - use our own provider (FalkorDB exporter already added above)
                    trace.set_tracer_provider(self.tracer_provider)
                    # Skip the "reused provider" logic since we're using our own - FalkorDB exporter already added
                else:
                    # Provider already exists and is suitable, reuse it
                    logger.debug("Reusing existing tracer provider")
                    self.tracer_provider = current_provider
                    self._provider_owned = False  # We didn't create it, so don't flush/shutdown
                    
                    # IMPORTANT: Even when reusing a provider, we need to ensure FalkorDB exporter is added
                    # because the existing provider might not have it (e.g., from a previous game that failed to init)
                    if falkordb_enabled:
                        # Check if FalkorDB exporter is already added by checking span processors
                        # Note: This is a best-effort check - we'll try to add it anyway
                        has_falkordb = False
                        try:
                            from .falkordb_exporter import FalkorDBExporter
                            # Check if we already have a FalkorDB processor
                            try:
                                # Try to access span processors - different tracer providers have different attributes
                                span_processors = None
                                if hasattr(self.tracer_provider, '_span_processors'):
                                    span_processors = self.tracer_provider._span_processors
                                elif hasattr(self.tracer_provider, 'span_processors'):
                                    span_processors = self.tracer_provider.span_processors
                                elif hasattr(self.tracer_provider, '_sdk_span_processor'):
                                    # Some providers wrap processors
                                    span_processors = [self.tracer_provider._sdk_span_processor]
                                
                                if span_processors:
                                    for processor in span_processors:
                                        exporter = None
                                        if hasattr(processor, '_span_exporter'):
                                            exporter = processor._span_exporter
                                        elif hasattr(processor, 'span_exporter'):
                                            exporter = processor.span_exporter
                                        
                                        if exporter and isinstance(exporter, FalkorDBExporter):
                                            has_falkordb = True
                                            logger.info("FalkorDB exporter already present in reused provider")
                                            break
                            except Exception as e:
                                logger.debug(f"Could not check for existing FalkorDB exporter: {e}")
                                # Assume it's not there and try to add it
                        except Exception as e:
                            # If we can't check or add the exporter, this is a FATAL error
                            error_msg = f"Failed to ensure FalkorDB exporter is registered: {e}. This is a FATAL error."
                            logger.error(error_msg, exc_info=True)
                            raise RuntimeError(error_msg) from e
                        
                        if not has_falkordb:
                            # Add FalkorDB exporter to the reused provider
                            # NO SILENT FAILURES - if FalkorDB is enabled, it MUST be added
                            try:
                                logger.info("Adding FalkorDB exporter to reused tracer provider")
                                falkordb_exporter = FalkorDBExporter(
                                    host=self.config.get("falkordb_host", "localhost"),
                                    port=self.config.get("falkordb_port", 6379),
                                    username=self.config.get("falkordb_username"),
                                    password=self.config.get("falkordb_password"),
                                    group_id=self.config.get("falkordb_group_id", "cpr-game-traces"),
                                    enabled=self.config.get("falkordb_enabled", True),
                                    max_retries=self.config.get("falkordb_max_retries", 10),  # Increased for rate limits
                                    base_retry_delay=self.config.get("falkordb_base_retry_delay", 5.0),  # Increased base delay
                                    max_retry_delay=self.config.get("falkordb_max_retry_delay", 300.0),  # 5 minutes max delay
                                    export_timeout=self.config.get("falkordb_export_timeout", 60 * 60.0),  # 60 minutes default
                                    episode_rate_limit=self.config.get("falkordb_episode_rate_limit", 1.0)  # Throttle episode additions
                                )
                                if not falkordb_exporter.enabled:
                                    error_msg = "FalkorDB exporter initialization failed when adding to reused provider - FATAL error"
                                    logger.error(error_msg)
                                    raise RuntimeError(error_msg)
                                
                                falkordb_processor = BatchSpanProcessor(falkordb_exporter)
                                # ProxyTracerProvider doesn't have add_span_processor - need to get underlying provider
                                actual_provider = self.tracer_provider
                                if hasattr(self.tracer_provider, '_delegate'):
                                    # ProxyTracerProvider wraps the actual provider
                                    actual_provider = self.tracer_provider._delegate
                                elif hasattr(self.tracer_provider, '_provider'):
                                    actual_provider = self.tracer_provider._provider
                                
                                if hasattr(actual_provider, 'add_span_processor'):
                                    actual_provider.add_span_processor(falkordb_processor)
                                    logger.info("FalkorDB exporter added to reused tracer provider")
                                else:
                                    error_msg = (
                                        f"Cannot add span processor to provider type {type(actual_provider)}. "
                                        "FalkorDB export will NOT work. This is a FATAL error."
                                    )
                                    logger.error(error_msg)
                                    raise RuntimeError(error_msg)
                            except Exception as e:
                                # If we can't add the exporter, this is a FATAL error
                                error_msg = f"Failed to add FalkorDB exporter to reused provider: {e}. This is a FATAL error."
                                logger.error(error_msg, exc_info=True)
                                raise RuntimeError(error_msg) from e
            else:
                # No real provider, set ours
                trace.set_tracer_provider(self.tracer_provider)
        except Exception:
            # If anything goes wrong, try to set our provider
            # This might raise a warning, but it's better than failing
            try:
                trace.set_tracer_provider(self.tracer_provider)
            except Exception:
                # If setting fails (e.g., provider already set), just reuse existing
                self.tracer_provider = trace.get_tracer_provider()
                self._provider_owned = False  # We didn't create it
        
        # Get tracer
        self.tracer = trace.get_tracer(service_name)
        
        # Instrument LangChain if available
        try:
            from opentelemetry.instrumentation.langchain import LangChainInstrumentor
            langchain_instrumentor = LangChainInstrumentor()
            langchain_instrumentor.instrument(tracer_provider=self.tracer_provider)
            logger.info("✓ LangChain auto-instrumentation enabled")
        except ImportError:
            # LangChain instrumentation is optional, so ImportError is acceptable
            logger.debug("LangChain instrumentation not available - install opentelemetry-instrumentation-langchain for auto-instrumentation")
        except Exception as e:
            error_msg = f"Could not enable LangChain auto-instrumentation: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
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
        if self.tracer_provider and self._provider_owned:
            try:
                # Check if provider has force_flush method (ProxyTracerProvider doesn't)
                if hasattr(self.tracer_provider, 'force_flush'):
                    self.tracer_provider.force_flush()
                else:
                    # ProxyTracerProvider or similar - can't flush directly
                    logger.debug("Tracer provider doesn't support force_flush (likely ProxyTracerProvider)")
            except AttributeError:
                # Provider doesn't have force_flush - that's okay
                logger.debug("Tracer provider doesn't support force_flush")
            except Exception as e:
                # Log warning but don't raise - flushing is best effort
                logger.warning(f"Error flushing OTel spans: {e}")
    
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
                # Log warning but don't raise - shutdown is best effort
                logger.warning(f"Error shutting down OTel tracer provider: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()

