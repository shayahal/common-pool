# Logging System Guide

This document explains the logging systems in the CPR Game codebase and when to use each.

## Overview

The codebase uses **two logging systems** that serve different purposes:

1. **Application Logging** (`logger_setup.py`) - For development and debugging
2. **OpenTelemetry Tracing** (`logging_manager.py`) - For observability and API metrics

## 1. Application Logging (`logger_setup.py`)

### Purpose
Development and debugging. Use for error messages, info logs, and debugging output during development.

### Output Destinations
- **`logs/cpr_game.log`**: File with all log levels (DEBUG and above)
- **Console/stdout**: Real-time output (INFO level and above)

### Usage

```python
from cpr_game.logger_setup import get_logger

logger = get_logger(__name__)

logger.debug("Detailed debugging information")
logger.info("General informational message")
logger.warning("Warning message")
logger.error("Error message with details")
```

### When to Use
- Logging errors during development
- Debugging game logic issues
- Tracking application flow
- Development-time diagnostics

### Configuration
Logging is automatically configured when `GameRunner.setup_game()` is called. The log directory can be customized via the `log_dir` config parameter (default: `"logs"`).

## 2. OpenTelemetry Tracing (`logging_manager.py`)

### Purpose
Distributed tracing for research, observability, and API metrics tracking. All traces are sent to configured receivers (Langfuse or LangSmith) via OpenTelemetry Protocol (OTLP).

### Output Destination
- **OpenTelemetry traces** → Configured receiver(s) (Langfuse/LangSmith)
- **API metrics** (tokens, costs, latency) are captured as OpenTelemetry span attributes

### Features
- **Thread-based tracing** (LangSmith model): Each game is a thread, each player action is a separate trace
- API metrics tracking (tokens, costs, latency) automatically captured via span attributes
- Distributed tracing across game execution
- Research analysis and debugging
- View all player actions in a game as a conversation thread in LangSmith

### Configuration

Set the `OTEL_RECEIVER` environment variable to control which receiver(s) receive traces:

```bash
# Use only Langfuse
OTEL_RECEIVER=langfuse

# Use only LangSmith
OTEL_RECEIVER=langsmith

# Use both (default)
OTEL_RECEIVER=both
```

### When to Use
- Tracking LLM API calls and metrics (costs, tokens, latency)
- Analyzing game execution traces
- Research data collection
- Production monitoring
- Comparing agent behaviors

### Viewing Traces

**Langfuse:**
1. Go to https://cloud.langfuse.com
2. Navigate to your project
3. View traces, spans, and LLM generations
4. Analyze API costs and token usage

**LangSmith:**
1. Go to https://smith.langchain.com
2. Navigate to your project
3. View traces and debug LLM calls
4. Analyze prompt performance

## Configuration Guide

### Environment Variables

```bash
# OpenTelemetry Configuration
OTEL_SERVICE_NAME=cpr-game
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_ENABLED=true
OTEL_SERVICE_VERSION=1.0.0

# Receiver Selection: "langfuse", "langsmith", or "both"
OTEL_RECEIVER=both

# Langfuse (required if OTEL_RECEIVER includes "langfuse")
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# LangSmith (required if OTEL_RECEIVER includes "langsmith")
LANGSMITH_API_KEY=ls-...
LANGSMITH_PROJECT=cpr-game
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### OpenTelemetry Collector Configuration

If using the OpenTelemetry Collector (recommended), configure `otel-collector-config.yaml`:

1. Set `OTEL_RECEIVER` environment variable to your desired receiver(s)
2. Comment/uncomment exporters in `otel-collector-config.yaml` based on `OTEL_RECEIVER`:
   - If `OTEL_RECEIVER=langfuse`: Comment out LangSmith exporter
   - If `OTEL_RECEIVER=langsmith`: Comment out Langfuse exporter
   - If `OTEL_RECEIVER=both`: Keep both exporters (default)
3. Update the `exporters` list in `service.pipelines.traces` accordingly

See `otel-collector-config.yaml` for detailed comments.

## Comparison: Choosing the Right Logger

| Use Case | Logger to Use | Why |
|----------|---------------|-----|
| Debugging code errors | Application Logger | Quick, local, detailed |
| Tracking API costs | OpenTelemetry | Automatic tracking in spans |
| Development diagnostics | Application Logger | Immediate feedback |
| Research analysis | OpenTelemetry | Structured, queryable traces |
| Production monitoring | OpenTelemetry | Centralized, scalable |
| Local development logs | Application Logger | Simple, no setup needed |

## Migration from api_logger

Previously, API metrics were logged to `logs/api_calls.log` via `api_logger.py`. This has been replaced by OpenTelemetry tracing, which captures the same metrics (tokens, costs, latency) as span attributes in Langfuse/LangSmith.

**Benefits of OpenTelemetry:**
- Single source of truth for all observability
- Better integration with analysis tools
- Structured data for querying and analysis
- No separate log file to manage
- Automatic correlation with game traces

## Troubleshooting

### Application Logs Not Appearing
- Check that `GameRunner.setup_game()` was called
- Verify log directory permissions
- Check log level settings

### OpenTelemetry Traces Not Appearing
- Verify `OTEL_ENABLED=true`
- Check `OTEL_EXPORTER_OTLP_ENDPOINT` is set correctly
- Ensure collector is running (if using collector)
- Verify API keys are set for selected receiver(s)
- Check that `OTEL_RECEIVER` matches collector configuration

### API Metrics Missing
- API metrics are captured automatically via OpenTelemetry spans
- View them in Langfuse/LangSmith dashboards
- Check that LLM agents are using real API calls (not mock agents)

## Additional Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [LangSmith Documentation](https://docs.smith.langchain.com/)

