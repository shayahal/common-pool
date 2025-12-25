#!/usr/bin/env python3
"""Send a test trace through the OpenTelemetry collector to Langfuse."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
import time

# Configure to send to local collector (which forwards to Langfuse)
endpoint = "http://localhost:4318/v1/traces"

print(f"Sending test trace to collector at {endpoint}")
print("Collector will forward to Langfuse with Basic Auth...")

# Create exporter
exporter = OTLPSpanExporter(endpoint=endpoint)

# Create tracer provider
resource = Resource.create({
    "service.name": "test-service",
    "service.version": "1.0.0"
})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Create a test span
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("test-trace-from-python") as span:
    span.set_attribute("test.attribute", "test-value")
    span.set_attribute("test.message", "This is a test trace sent through the collector")
    time.sleep(0.1)

# Force flush to ensure it's sent
provider.force_flush(timeout_millis=5000)

print("[SUCCESS] Trace sent! Check Langfuse dashboard.")
print("Monitor collector logs: docker-compose logs otel-collector --follow")

