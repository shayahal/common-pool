# Database Structure Documentation

This document describes the database schema for the Common Pool Resource Game experiment system.

## Overview

The database uses **DuckDB** and is stored at `data/game_results.duckdb`. It contains 4 main tables organized into 2 systems:

1. **OLD SYSTEM (Legacy)**: `game_results` - Stores individual player results
2. **NEW SYSTEM (Experiment Management)**: `experiments`, `experiment_players`, `experiment_results` - Manages experiments with player pools

---

## Table: `game_results`

### Purpose
Stores individual player results for each game. Used by the legacy system when running games directly.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | TEXT | Unique game identifier |
| `player_id` | INTEGER | Player number (0, 1, 2, ...) |
| `persona` | TEXT | Player persona type (e.g., "rational_selfish", "cooperative") |
| `model` | TEXT | LLM model used (e.g., "gpt-3.5-turbo") |
| `total_reward` | DOUBLE | Final cumulative payoff for this player |
| `experiment_id` | TEXT | Optional link to experiment (nullable) |
| `timestamp` | TIMESTAMP | When the game was run |

**Primary Key**: `(game_id, player_id)`

### Example Data

```
game_id          | player_id | persona          | model          | total_reward | experiment_id | timestamp
-----------------|-----------|------------------|----------------|--------------|---------------|------------------
"demo_game"      | 0         | rational_selfish | gpt-3.5-turbo  | 1000.0       | NULL          | 2025-12-17...
"demo_game"      | 1         | cooperative      | gpt-3.5-turbo  | 850.0        | NULL          | 2025-12-17...
"quick_test"     | 0         | aggressive       | gpt-3.5-turbo  | 1200.0       | NULL          | 2025-12-17...
```

### Used By
- `GameRunner.save_game_results()`
- `quick_experiment.py`
- `simple_example.py`

### Notes
- One row per player per game
- Stores granular player-level data
- Can optionally link to experiments via `experiment_id`

---

## Table: `experiments`

### Purpose
Stores experiment definitions - what you want to test. This is the main entry point for the new experiment management system.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | TEXT (PK) | Unique experiment identifier (e.g., "exp_123abc") |
| `name` | TEXT | Human-readable experiment name |
| `status` | TEXT | Current status: `"pending"`, `"running"`, `"completed"`, or `"failed"` |
| `created_at` | TIMESTAMP | When the experiment was created |
| `n_players` | INTEGER | Number of players in the player pool |
| `max_steps` | INTEGER | Maximum number of rounds per game |
| `initial_resource` | INTEGER | Initial resource level at game start |
| `regeneration_rate` | DOUBLE | Resource regeneration multiplier per round |
| `max_extraction` | INTEGER | Maximum extraction per player per round |
| `max_fishes` | INTEGER | Maximum resource capacity |
| `number_of_games` | INTEGER | Total number of games to run in this experiment |
| `number_of_players_per_game` | INTEGER | Number of players randomly selected for each game |

**Primary Key**: `experiment_id`

### Parameter Columns

All parameters are stored as individual columns for easier querying:

- `n_players`: Size of the player pool
- `max_steps`: Maximum rounds per game
- `initial_resource`: Starting resource level
- `regeneration_rate`: Resource regeneration multiplier
- `max_extraction`: Maximum extraction per player
- `max_fishes`: Resource capacity limit
- `number_of_games`: Total games in experiment
- `number_of_players_per_game`: Players per game (randomly selected from pool)

### Example Data

```
experiment_id | name                    | status    | created_at          | n_players | max_steps | ...
--------------|-------------------------|-----------|---------------------|-----------|-----------|----
"exp_123"     | "6 Players GPT-3.5 Test"| completed | 2025-12-17 10:00:00 | 6         | 50        | ...
"exp_456"     | "Small Test"            | pending   | 2025-12-17 11:00:00 | 2         | 20        | ...
```

### Relationships
- One-to-many with `experiment_players`
- One-to-many with `experiment_results`

---

## Table: `experiment_players`

### Purpose
Defines the player pool for each experiment. Each experiment can have multiple players (persona + model combinations).

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | TEXT (FK) | Links to `experiments.experiment_id` |
| `player_index` | INTEGER | Position in the player pool (0, 1, 2, ...) |
| `persona` | TEXT | Player persona type |
| `model` | TEXT | LLM model for this player |

**Primary Key**: `(experiment_id, player_index)`  
**Foreign Key**: `experiment_id` → `experiments.experiment_id`

### Example Data

For a 6-player experiment:

```
experiment_id | player_index | persona          | model
--------------|--------------|------------------|------------------
"exp_123"     | 0            | rational_selfish | gpt-3.5-turbo
"exp_123"     | 1            | cooperative      | gpt-3.5-turbo
"exp_123"     | 2            | aggressive       | gpt-3.5-turbo
"exp_123"     | 3            | conservative     | gpt-3.5-turbo
"exp_123"     | 4            | opportunistic    | gpt-3.5-turbo
"exp_123"     | 5            | altruistic       | gpt-3.5-turbo
```

### Notes
- Defines the available player pool for an experiment
- Players are randomly selected from this pool for each game
- Multiple players can share the same persona or model

---

## Table: `experiment_results`

### Purpose
Stores game result summaries for each experiment. Each experiment can have many games, each with one result summary.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | TEXT (FK) | Links to `experiments.experiment_id` |
| `game_id` | TEXT | Unique game identifier (e.g., "exp_123_game_0001") |
| `summary` | TEXT (JSON) | Game summary as JSON string |
| `timestamp` | TIMESTAMP | When the game was run |
| `winning_player_id` | INTEGER | Player ID (index) with the highest cumulative payoff |
| `winning_payoff` | DOUBLE | The highest cumulative payoff value |
| `cumulative_payoff_sum` | DOUBLE | Sum of all players' cumulative payoffs |
| `total_rounds` | INTEGER | Total number of rounds played in the game |
| `final_resource_level` | DOUBLE | Final resource level at game end |
| `tragedy_occurred` | BOOLEAN | Whether the resource was depleted (tragedy occurred) |

**Primary Key**: `(experiment_id, game_id)`  
**Foreign Key**: `experiment_id` → `experiments.experiment_id`

### Summary JSON Structure

The `summary` column contains a JSON object with:

```json
{
  "total_rounds": 25,
  "final_resource_level": 50.5,
  "tragedy_occurred": false,
  "avg_cooperation_index": 0.75,
  "cumulative_payoffs": [100.0, 95.0, 110.0, 105.0],
  "gini_coefficient": 0.15,
  "sustainability_score": 0.85
}
```

### Example Data

```
experiment_id | game_id              | summary (JSON)           | timestamp          | winning_player_id | winning_payoff | cumulative_payoff_sum | total_rounds | final_resource_level | tragedy_occurred
--------------|----------------------|--------------------------|-------------------|-------------------|---------------|----------------------|--------------|---------------------|-----------------
"exp_123"     | "exp_123_game_0001"  | {"total_rounds": 25, ...}| 2025-12-17...     | 2                 | 110.0         | 410.0                | 25           | 50.5                 | false
"exp_123"     | "exp_123_game_0002"  | {"total_rounds": 20, ...}| 2025-12-17...     | 0                 | 120.0         | 380.0                | 20           | 30.2                 | false
"exp_123"     | "exp_123_game_0003"  | {"total_rounds": 30, ...}| 2025-12-17...     | 1                 | 95.0          | 350.0                | 30           | 0.0                  | true
```

### Notes
- One row per game per experiment
- Stores aggregated game-level summaries
- Includes metrics for all players in the game
- The new columns (`winning_player_id`, `winning_payoff`, `cumulative_payoff_sum`, `total_rounds`, `final_resource_level`, `tragedy_occurred`) are extracted from the `summary` JSON for easier querying and analysis
- These columns are automatically populated when saving results via `save_experiment_result()`

---

## Entity Relationship Diagram

```
┌──────────────────┐
│   experiments    │
├──────────────────┤
│ experiment_id (PK)│
│ name             │
│ status           │
│ created_at       │
│ parameters (JSON)│
└────────┬─────────┘
         │
         │ 1
         │
         │ many
         │
    ┌────┴────────────────────┐
    │                         │
┌───▼────────────┐   ┌────────▼──────────────┐
│experiment_     │   │ experiment_results    │
│  players       │   ├───────────────────────┤
├────────────────┤   │ experiment_id (FK)    │
│experiment_id(FK)│   │ game_id              │
│player_index    │   │ summary (JSON)        │
│persona         │   │ timestamp             │
│model           │   └───────────────────────┘
└────────────────┘

┌──────────────────┐
│  game_results    │  (Independent, legacy system)
├──────────────────┤
│ game_id          │
│ player_id        │
│ persona          │
│ model            │
│ total_reward     │
│ experiment_id    │  (optional link)
│ timestamp        │
└──────────────────┘
```

---

## Data Flow

### Old System Workflow

```
1. Run game directly
   GameRunner.run_episode()
   
2. Save results
   → game_results table
   → One row per player per game
```

### New System Workflow

```
1. Create Experiment
   experiment_app.py or create_and_run_experiment.py
   → experiments table (experiment definition)
   → experiment_players table (player pool)
   
2. Run Experiment
   main.py --experiment-id <id>
   → For each game:
      - Randomly select players from pool
      - Run game
      - Save summary to experiment_results table
```

---

## Key Concepts

### Experiment vs Game

- **Experiment**: A configuration you want to test (e.g., "6 players, all GPT-3.5, 10 games")
- **Game**: One instance of running that experiment (e.g., game 1, game 2, ..., game 10)

### Player Pool vs Players per Game

- **Player Pool** (`experiment_players`): All available players for the experiment (e.g., 6 players)
- **Players per Game** (`parameters.number_of_players_per_game`): How many players participate in each game (e.g., 4 players)
- **Random Assignment**: Each game randomly selects `number_of_players_per_game` players from the pool

### Data Granularity

- **`game_results`**: Player-level data (one row per player per game)
- **`experiment_results`**: Game-level summaries (one row per game, includes all players)

---

## Common SQL Queries

### Get all experiments

```sql
SELECT * FROM experiments 
ORDER BY created_at DESC;
```

### Get players for an experiment

```sql
SELECT * FROM experiment_players 
WHERE experiment_id = 'exp_123'
ORDER BY player_index;
```

### Get all game results for an experiment

```sql
SELECT * FROM experiment_results 
WHERE experiment_id = 'exp_123'
ORDER BY timestamp;
```

### Get experiment with player count and game count

```sql
SELECT 
    e.experiment_id,
    e.name,
    e.status,
    COUNT(DISTINCT ep.player_index) as num_players,
    COUNT(er.game_id) as num_games
FROM experiments e
LEFT JOIN experiment_players ep ON e.experiment_id = ep.experiment_id
LEFT JOIN experiment_results er ON e.experiment_id = er.experiment_id
GROUP BY e.experiment_id, e.name, e.status;
```

### Query by parameter values

```sql
SELECT 
    experiment_id,
    name,
    number_of_games,
    max_steps
FROM experiments
WHERE max_steps > 50;
```

### Compare old vs new system for same experiment

```sql
SELECT 
    'game_results' as source,
    COUNT(*) as row_count
FROM game_results 
WHERE experiment_id = 'exp_123'
UNION ALL
SELECT 
    'experiment_results' as source,
    COUNT(*) as row_count
FROM experiment_results 
WHERE experiment_id = 'exp_123';
```

---

## Database Operations

### Using DatabaseManager Methods

```python
from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG

db = DatabaseManager(
    db_path=CONFIG.get("db_path", "data/game_results.duckdb"),
    enabled=True
)

# Create experiment
db.save_experiment(
    experiment_id="exp_123",
    name="My Experiment",
    players=[{"persona": "cooperative", "model": "gpt-3.5-turbo"}],
    parameters={"max_steps": 50, "number_of_games": 10, ...}
)

# Load experiment
experiment = db.load_experiment("exp_123")

# List all experiments
experiments = db.list_experiments()

# Save game result
db.save_experiment_result(
    experiment_id="exp_123",
    game_id="exp_123_game_0001",
    summary={"total_rounds": 25, ...}
)

# Get experiment results
results = db.get_experiment_results("exp_123")

# Custom query
results = db.query_results(
    "SELECT * FROM experiments WHERE status = ?",
    ["completed"]
)
```

---

## Notes

- **DuckDB limitation**: Only one write connection allowed at a time. Close Streamlit app before running `main.py`.
- **JSON storage**: Only `experiment_results.summary` is stored as JSON. Parameters are stored as individual columns.
- **Foreign keys**: `experiment_players` and `experiment_results` reference `experiments` via foreign keys.
- **Cascade delete**: When an experiment is deleted, associated players and results are also deleted (manual cascade implemented).
- **Backward compatibility**: The `load_experiment()` method still returns parameters as a dictionary for backward compatibility with existing code.

---

## Migration Notes

The `game_results` table has an optional `experiment_id` column that can link to the new system. This allows gradual migration from the old system to the new system while maintaining backward compatibility.

