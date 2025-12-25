# FalkorDB Integration Setup Guide

This guide explains how to set up and use FalkorDB for storing OpenTelemetry traces from the CPR game.

## Overview

The integration sends all OpenTelemetry traces to a local FalkorDB instance using Graphiti, which automatically extracts entities and relationships from the trace data, creating a queryable knowledge graph.

## Prerequisites

1. Docker installed and running
2. Python dependencies installed (already done)

## Step 1: Start FalkorDB

FalkorDB will **automatically start** when needed by the application. However, you can also manage it manually:

### Automatic Startup

FalkorDB is automatically started when:
- The FalkorDB exporter is initialized (if `FALKORDB_ENABLED=true`)
- The application detects FalkorDB is not running

### Manual Management

You can manage FalkorDB using the management script:

```bash
# Check status
python manage_falkordb.py status

# Start FalkorDB
python manage_falkordb.py start

# Stop FalkorDB
python manage_falkordb.py stop

# Restart FalkorDB
python manage_falkordb.py restart
```

### Using Docker Compose

FalkorDB is also available in `docker-compose.yml`:

```bash
# Start FalkorDB (and other services)
docker compose up -d falkordb

# Stop FalkorDB
docker compose stop falkordb

# View status
docker compose ps falkordb
```

### Manual Docker Command (Alternative)

If you prefer to use Docker directly:

```bash
docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest
```

Verify it's running:

```bash
docker ps | grep falkordb
```

You should see the container running. The web UI is available at http://localhost:3000 (if available).

## Step 2: Configure Environment (Optional)

By default, the integration uses:
- Host: `localhost`
- Port: `6379`
- Group ID: `cpr-game-traces`

You can customize these via environment variables:

```bash
export FALKORDB_ENABLED=true
export FALKORDB_HOST=localhost
export FALKORDB_PORT=6379
export FALKORDB_GROUP_ID=cpr-game-traces
```

## Step 3: Run the Test

Test the integration:

```bash
python test_falkordb_integration.py
```

This will:
1. Check FalkorDB connection
2. Run a short test game
3. Verify traces are stored in FalkorDB

## Step 4: Run a Real Game

Run a full game - traces will automatically be sent to FalkorDB:

```bash
python main.py
```

Or use the experiment runner:

```bash
python experiments/run_experiment.py
```

## Querying Traces in FalkorDB

You can query traces using Graphiti's search functionality. The traces are stored as episodes with:
- Span names (e.g., "game_setup", "round_1", "player_0_action")
- Attributes (game.id, round.number, player.id, etc.)
- Events (prompts, responses, actions)
- Timing information

Example queries (via Graphiti Python API):

```python
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver

falkor_driver = FalkorDriver(host='localhost', port='6379')
graphiti = Graphiti(graph_driver=falkor_driver)

# Search for game traces
results = await graphiti.search("game trace")
for r in results:
    print(r.fact)
```

## Architecture

```
Application (Python)
    ├── OpenTelemetry SDK
    │   ├── OTLP Exporter → Collector → LangFuse/LangSmith
    │   └── FalkorDB Exporter → Graphiti → FalkorDB
    │
    └── Traces stored as Graphiti episodes
        └── Automatic entity/relationship extraction
            └── Queryable knowledge graph
```

## Troubleshooting

### FalkorDB Connection Failed

- Ensure Docker is running: `docker ps`
- Check status: `python manage_falkordb.py status`
- Start FalkorDB: `python manage_falkordb.py start`
- Or use docker-compose: `docker compose up -d falkordb`
- Check if container exists: `docker ps -a | grep falkordb`
- Start existing container: `docker start falkordb` (if it exists)
- Or create new: `docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest`

### No Traces Found

- Check that `FALKORDB_ENABLED=true` in environment or config
- Verify traces are being generated (check application logs)
- Wait a few seconds after game completion for traces to be processed
- Check FalkorDB logs: `docker logs falkordb`

### Graphiti Import Error

- Install: `pip install "graphiti-core[falkordb]"`
- Verify: `python -c "from graphiti_core import Graphiti; print('OK')"`

## Configuration

The FalkorDB exporter is automatically enabled if:
1. `FALKORDB_ENABLED=true` (default: true)
2. Graphiti is installed
3. FalkorDB is accessible (will be started automatically if not running)

You can disable it by setting `FALKORDB_ENABLED=false`.

## Stopping FalkorDB

To stop FalkorDB when not needed:

```bash
# Using the management script (recommended)
python manage_falkordb.py stop

# Using docker-compose
docker compose stop falkordb

# Using docker directly
docker stop falkordb
```

**Note:** FalkorDB will automatically start again when the application needs it (if `FALKORDB_ENABLED=true`).

## Next Steps

- Explore traces in FalkorDB using Graphiti queries
- Build custom analysis tools using the knowledge graph
- Export traces for further analysis
- Set up Grafana or other visualization tools (if supported)

