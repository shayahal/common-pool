# Langfuse Export Troubleshooting

## Current Issues Found

1. **OTel Collector is NOT running**
   - The collector needs to be running to forward traces to Langfuse
   - Check: `docker ps | grep otel-collector`
   - Start: `docker-compose up -d otel-collector`

2. **Langfuse API Keys are NOT set**
   - `LANGFUSE_PUBLIC_KEY` is not set
   - `LANGFUSE_SECRET_KEY` is not set
   - These are required for the collector to authenticate with Langfuse

3. **Error Suppression Removed**
   - Previously, export failures were only logged once and then suppressed
   - Now ALL failures are logged at ERROR level every time
   - You will see errors in logs if export fails

## How to Fix

### Step 1: Set Langfuse API Keys

Add to your `.env` file:
```bash
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

Get these from: https://cloud.langfuse.com/settings

### Step 2: Start OTel Collector

```bash
docker-compose up -d otel-collector
```

Verify it's running:
```bash
docker ps | grep otel-collector
```

Check collector logs:
```bash
docker-compose logs otel-collector
```

### Step 3: Verify Configuration

The collector configuration is in `otel-collector-config.yaml`:
- Receives traces on `localhost:4317` (gRPC) or `localhost:4318` (HTTP)
- Forwards to Langfuse at `https://cloud.langfuse.com/api/public/otel`
- Requires `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` environment variables

### Step 4: Run an Experiment

After starting the collector and setting API keys, run an experiment and check:

1. **Collector logs** - should show traces being received and forwarded:
   ```bash
   docker-compose logs -f otel-collector
   ```

2. **Application logs** - should show export success (or errors if collector is down):
   ```bash
   grep "OpenTelemetry export" logs/error.log
   ```

3. **Langfuse UI** - check https://cloud.langfuse.com for traces

## What Changed

### Before (Silent Failures)
- Export failures were logged once, then suppressed
- Subsequent failures were logged at DEBUG level only
- Easy to miss that traces weren't being exported

### After (NO SILENT FAILURES)
- **ALL export failures are logged at ERROR level every time**
- Failure count is tracked and logged
- Collector status is checked at initialization
- API keys are validated at initialization
- Clear error messages explain what's wrong and how to fix it

## Error Messages You'll See

If collector is not running:
```
ERROR - OpenTelemetry export to http://localhost:4317 FAILED (failure #1). 
Traces are NOT being exported. 
Check if collector is running: 'docker ps | grep otel-collector' 
To start: 'docker-compose up -d otel-collector'
```

If API keys are missing:
```
ERROR - LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY are not set. 
Traces will NOT be exported to Langfuse. 
Set these environment variables or add them to .env file.
```

## Architecture

```
Application (Python)
    └── OpenTelemetry SDK
        └── OTLP Exporter → Collector (localhost:4317)
            └── Collector → Langfuse (https://cloud.langfuse.com/api/public/otel)
```

The collector acts as a proxy that:
1. Receives traces from your application
2. Processes/batches them
3. Forwards them to Langfuse with authentication headers

## Next Steps

1. Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to `.env`
2. Start collector: `docker-compose up -d otel-collector`
3. Run an experiment
4. Check collector logs for forwarding activity
5. Check Langfuse UI for traces

