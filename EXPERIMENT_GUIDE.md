# Experiment Guide: Running Experiments and Viewing Results

## Overview

This guide explains how to run experiments and monitor results in real-time using both the Streamlit experiment app and DuckDB UI.

## Process for Running an Experiment

### Step 1: Create an Experiment

You have two options:

#### Option A: Using the Streamlit App (Recommended)

```bash
streamlit run experiment_app.py
```

1. Open the "📝 Define Experiment" tab
2. Enter an experiment name
3. Add players (persona + model combinations)
4. Configure experiment parameters
5. Click "💾 Save Experiment"

#### Option B: Using Python Script

```bash
python create_and_run_experiment.py
```

This creates an experiment definition. To run it, use the Streamlit app or command line (see Step 2).

### Step 2: Run the Experiment

#### Option A: Using the Streamlit App

1. Go to the "📋 Experiments List" tab
2. Select your experiment
3. Click "▶️ Run Experiment"
4. Configure options:
   - **Use Mock Agents**: Check to avoid API calls (faster, for testing)
   - **Max Workers**: Number of parallel games (default: 10)
5. Click "🚀 Start Experiment"

The app will show progress as games complete.

#### Option B: Using Command Line

```bash
# With real LLM agents
python main.py --experiment-id exp_12345678

# With mock agents (no API calls, faster for testing)
python main.py --experiment-id exp_12345678 --use-mock

# With custom worker count
python main.py --experiment-id exp_12345678 --use-mock --max-workers 5
```

Replace `exp_12345678` with your actual experiment ID.

### Step 3: View Results

#### Option A: Streamlit App Results Tab

1. Open the "📊 Results" tab in the Streamlit app
2. Select your experiment
3. View:
   - Aggregate statistics (tragedy rate, avg rounds, etc.)
   - Individual game results table
   - Download results as JSON or CSV

The results tab will automatically refresh as new games complete if you're viewing while an experiment is running.

#### Option B: Query Script

```bash
python query_experiments.py
```

This provides an interactive interface to query experiment results.

## Can You Use DuckDB UI and Experiment App Simultaneously?

**Yes! Both can now operate concurrently without lock conflicts.**

### DuckDB Concurrent Access

The system now uses a hybrid connection strategy that enables concurrent access:

1. **Read-only connections** - Used for all SELECT queries (experiment app queries, DuckDB UI)
2. **Write connections** - Created only when needed for INSERT/UPDATE/DELETE operations
3. **Multiple processes can read simultaneously** - No lock conflicts between readers
4. **Writes are serialized** - Only one write operation at a time (as expected)

### How It Works

- **Experiment App**: Uses read-only connections for all queries (list experiments, view results, etc.)
- **Write Operations**: Creates temporary write connections only when saving/deleting experiments
- **DuckDB UI**: Can use read-only mode for queries while experiments run
- **No Lock Conflicts**: Multiple readers can access the database simultaneously

### Recommended Workflow

#### Scenario 1: Running Experiment from Streamlit App (✅ Recommended)

**Best option - no lock conflicts:**
1. Start Streamlit app: `streamlit run experiment_app.py`
2. Go to "📋 Experiments List" tab
3. Select experiment and click "▶️ Run Experiment"
4. Experiment runs in the same process, so no lock conflicts
5. Monitor results in "📊 Results" tab as games complete
6. **You can keep DuckDB UI open simultaneously** - it will see updates in real-time

#### Scenario 2: Running Experiment via Command Line + Monitoring

✅ **Now fully supported - no need to close apps:**
1. Start Streamlit app: `streamlit run experiment_app.py` (keep it running)
2. Run experiment: `python main.py --experiment-id exp_123 --use-mock`
3. Monitor results in Streamlit app in real-time (no restart needed)
4. OR use DuckDB UI simultaneously for SQL queries

#### Scenario 3: DuckDB UI + Running Experiment

✅ **Fully supported - concurrent access:**
- Run experiment via command line: `python main.py --experiment-id exp_123`
- Start DuckDB UI: `duckdb data/game_results.duckdb -ui`
- **Important**: Use read-only mode in DuckDB UI (only SELECT queries)
- Experiment can write while UI reads - no conflicts!
- Streamlit app can also be running simultaneously

### Starting DuckDB UI

To view the database directly:

```bash
# Option 1: Start DuckDB UI from command line
duckdb data/game_results.duckdb -ui

# Option 2: From within Python/DuckDB connection
# In Python REPL or script:
import duckdb
conn = duckdb.connect("data/game_results.duckdb")
conn.execute("CALL start_ui();")
```

This opens a web interface in your browser at `http://localhost:3000` (default).

### Useful SQL Queries for DuckDB UI

While monitoring experiments, you can run queries like:

```sql
-- View all experiments
SELECT * FROM experiments ORDER BY created_at DESC;

-- View recent game results
SELECT 
    e.name as experiment_name,
    er.game_id,
    er.total_rounds,
    er.final_resource_level,
    er.tragedy_occurred,
    er.timestamp
FROM experiment_results er
JOIN experiments e ON e.experiment_id = er.experiment_id
ORDER BY er.timestamp DESC
LIMIT 50;

-- Count games per experiment
SELECT 
    e.name,
    COUNT(er.game_id) as game_count,
    SUM(CASE WHEN er.tragedy_occurred THEN 1 ELSE 0 END) as tragedies
FROM experiments e
LEFT JOIN experiment_results er ON e.experiment_id = er.experiment_id
GROUP BY e.name
ORDER BY e.created_at DESC;

-- View player performance in a specific experiment
SELECT 
    egp.persona,
    egp.model,
    COUNT(*) as games_played,
    AVG(egp.total_reward) as avg_reward,
    MIN(egp.total_reward) as min_reward,
    MAX(egp.total_reward) as max_reward
FROM experiment_game_players egp
WHERE egp.experiment_id = 'exp_12345678'  -- Replace with your experiment ID
GROUP BY egp.persona, egp.model
ORDER BY avg_reward DESC;
```

## Real-Time Monitoring Strategy

### Option A: All-in-One (✅ Recommended - No Lock Issues)

1. **Single Terminal**: Streamlit app
   ```bash
   streamlit run experiment_app.py
   ```
   - Create experiments in "📝 Define Experiment" tab
   - Run experiments in "📋 Experiments List" tab  
   - View results in "📊 Results" tab
   - **Everything works without lock conflicts!**
   - Can run DuckDB UI simultaneously for SQL queries

### Option B: Separate Processes (✅ Now Fully Supported)

1. **Terminal 1**: Run experiment
   ```bash
   python main.py --experiment-id exp_123 --use-mock
   ```

2. **Terminal 2**: DuckDB UI for SQL queries (read-only)
   ```bash
   duckdb data/game_results.duckdb -ui
   ```
   - Use for custom SQL analysis
   - Refresh queries manually to see new results
   - **Only use SELECT queries** (read-only)
   - **Can run simultaneously with experiment** - no lock conflicts!

3. **Terminal 3**: Streamlit app (can run simultaneously)
   ```bash
   streamlit run experiment_app.py
   ```
   - View results in real-time as experiment runs
   - **Can run while `main.py` is running** - no conflicts!
   - Uses read-only connections for queries
   - Creates write connections only when saving/deleting

### Important Notes

- **Concurrent reads**: Multiple processes can read simultaneously (no locks)
- **Serialized writes**: Only one write operation at a time (as expected)
- **Streamlit app**: Uses read-only connections for queries, write connections only for saves
- **DuckDB UI**: Should use read-only mode (SELECT queries only) for concurrent access
- **Best practice**: Use Streamlit app's built-in experiment runner (no conflicts)
- The Streamlit app caches read-only database connections, click "🔄 Refresh" to see updates
- DuckDB UI queries are real-time but require manual refresh

## Troubleshooting

### Database Lock Errors

**If you see this error (should be rare now):**
```
IO Error: Could not set lock on file ".../game_results.duckdb": Conflicting lock is held
```

**This typically only happens when:**
- Multiple processes try to write simultaneously (e.g., two experiments running at once)
- A write operation is interrupted and connection not properly closed

**Solutions:**

1. **Wait for writes to complete** - If an experiment is actively writing, wait for it to finish
   - Write operations are now serialized automatically
   - The system includes retry logic for temporary lock conflicts

2. **Check for other processes** holding write locks:
   ```bash
   # On macOS/Linux:
   lsof data/game_results.duckdb
   # Kill processes if needed (only if stuck)
   ```

3. **Use read-only connections** - The Streamlit app now uses read-only connections for queries
   - You can keep it open while experiments run
   - Only write operations (save/delete) create write connections temporarily

4. **Run experiments from Streamlit app** - Recommended approach with no lock conflicts

### Results Not Appearing

1. **Refresh the Streamlit app** - Click "🔄 Refresh" button
2. **Check experiment status** - Ensure it's actually running
3. **Check logs** - Look at `logs/cpr_game.log` for errors
4. **Verify experiment ID** - Make sure you're viewing the correct experiment

### Performance Considerations

- **Multiple concurrent reads**: Fully supported - multiple processes can read simultaneously
- **Reads during writes**: Safe, DuckDB handles this well with read-only connections
- **Write operations**: Serialized automatically (one at a time) with retry logic
- **Connection management**: Read-only connections are cached, write connections created on-demand
- **Large result sets**: Streamlit app paginates, DuckDB UI handles large queries well

