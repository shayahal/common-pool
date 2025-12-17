# Common Pool Resource Game - RL Environment Specification

## Project Overview

Build a Reinforcement Learning environment for the Common Pool Resource (CPR) game with LLM-based players, Langfuse logging, and visual presentation of game dynamics.

**Research Focus:** Tragedy avoidance and emergent communication patterns

---

## Configuration Constants

All constants should be easily configurable through a `config.py` or YAML file.

### Game Parameters

```python
# Core Game Settings
N_PLAYERS = 2
MAX_STEPS = 100
INITIAL_RESOURCE = 1000.0
REGENERATION_RATE = 2.0  # Resource doubles each round (before extraction)
MIN_RESOURCE = 0.0  # Resource depletion threshold

# Action Space
MIN_EXTRACTION = 0.0
MAX_EXTRACTION = 100.0  # Maximum a player can extract per round
ACTION_TYPE = "continuous"

# Reward Parameters
EXTRACTION_VALUE = 1.0  # Value per unit extracted
DEPLETION_PENALTY = -1000.0  # Penalty when resource fully depleted
SUSTAINABILITY_BONUS = 10.0  # Bonus for keeping resource above threshold
SUSTAINABILITY_THRESHOLD = 500.0  # Resource level for sustainability bonus

# Game Mechanics
MOVE_TYPE = "simultaneous"  # Players act at the same time
TERMINATION_CONDITIONS = ["max_steps", "resource_depletion"]
```

### LLM Configuration

```python
# Model Settings
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500
LLM_TIMEOUT = 30  # seconds

# Agent Personas (first N_PLAYERS will be used)
PLAYER_PERSONAS = [
    "rational_selfish",  # Player 0: Maximizes individual payoff
    "cooperative",       # Player 1: Prioritizes group welfare
    # Leave additional slots empty for neutral agents
]

# Persona System Prompts
PERSONA_PROMPTS = {
    "rational_selfish": """You are a rational, self-interested player who aims to maximize your own payoff.
You understand game theory and will extract resources strategically to maximize your individual gains.
You may cooperate if it serves your long-term interests, but your primary goal is personal profit.""",
    
    "cooperative": """You are a cooperative player who values group welfare and sustainability.
You aim to maintain the resource for long-term benefit of all players.
You prefer fair distribution and will try to establish trust and cooperation patterns.""",
    
    "": """You are a player in a resource management game. 
Make decisions that you think are best given the situation."""
}

# Prompt Engineering
INCLUDE_HISTORY_ROUNDS = 5  # How many past rounds to include in context
SHOW_OTHER_PLAYERS_ACTIONS = True  # Full observability
ALLOW_REASONING_OUTPUT = True  # Let LLM explain its thinking
```

### Langfuse Configuration

```python
# API Settings (set via environment variables)
LANGFUSE_PUBLIC_KEY = "pk-lf-..."  # os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = "sk-lf-..."  # os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = "https://cloud.langfuse.com"

# Trace Settings
LOG_LEVEL = "detailed"  # "minimal", "standard", "detailed"
LOG_LLM_PROMPTS = True
LOG_LLM_RESPONSES = True
LOG_GAME_STATE = True
LOG_REASONING = True

# Metrics to Track Per Round
ROUND_METRICS = [
    "resource_level",
    "total_extraction",
    "individual_extractions",
    "individual_payoffs",
    "cooperation_index",
]

# Metrics to Track Per Game
GAME_METRICS = [
    "total_rounds",
    "final_resource_level",
    "cumulative_payoffs",
    "tragedy_occurred",  # Boolean: did resource deplete?
    "avg_cooperation_index",
    "gini_coefficient",  # Payoff inequality measure
    "sustainability_score",  # % of rounds above threshold
]
```

### Visualization Configuration

```python
# Dashboard Settings
VISUALIZATION_TOOL = "streamlit"  # or "gradio"
UPDATE_MODE = "realtime"  # "realtime" or "post_game"
REFRESH_RATE = 0.5  # seconds between updates

# Chart Configuration
CHART_HEIGHT = 400
CHART_WIDTH = 800
SHOW_GRID = True

# Plots to Display
PLOTS = {
    "resource_over_time": True,
    "individual_extractions": True,
    "cumulative_payoffs": True,
    "cooperation_index": True,
    "reasoning_log": True,  # Text display of LLM reasoning
}

# Styling
PLAYER_COLORS = ["#FF6B6B", "#4ECDC4"]
RESOURCE_COLOR = "#2ECC71"
THRESHOLD_COLOR = "#E74C3C"
BACKGROUND_COLOR = "#FFFFFF"
```

---

## Architecture Components

### 1. CPR Environment (`cpr_environment.py`)

**Gymnasium-compatible RL environment**

**Key Methods:**
- `__init__(config)` - Initialize with config dict
- `reset()` - Reset to initial state, return observation
- `step(actions)` - Process simultaneous actions, return (obs, rewards, done, info)
- `render()` - Return visualization data
- `_compute_rewards(actions)` - Reward function
- `_compute_cooperation_index()` - Measure cooperation level
- `_get_observation()` - Build observation dict for agents

**State Representation:**
```python
state = {
    "current_resource": float,
    "current_step": int,
    "extraction_history": List[np.array],  # Last N rounds
    "payoff_history": List[np.array],      # Last N rounds
    "player_cumulative_payoffs": np.array,
}
```

**Observation (Full Observability):**
```python
observation = {
    "resource_level": float,
    "step": int,
    "my_recent_extractions": List[float],
    "other_players_recent_extractions": List[List[float]],
    "my_cumulative_payoff": float,
    "other_players_cumulative_payoffs": List[float],
}
```

**Reward Function:**
```
reward_i = extraction_i * EXTRACTION_VALUE

if resource > SUSTAINABILITY_THRESHOLD:
    reward_i += SUSTAINABILITY_BONUS

if resource <= MIN_RESOURCE:
    reward_i += DEPLETION_PENALTY / N_PLAYERS
```

**Resource Dynamics:**
```
R(t+1) = max(R(t) * REGENERATION_RATE - sum(extractions), MIN_RESOURCE)
```

---

### 2. LLM Agent (`llm_agent.py`)

**Direct LLM policy (no RL component yet)**

**Key Methods:**
- `__init__(player_id, persona, config)` - Setup agent
- `act(observation, context)` - Return extraction amount
- `_build_prompt(observation)` - Construct LLM prompt
- `_parse_action(llm_response)` - Extract number from text
- `_update_memory(observation, action, reward)` - Store history

**Prompt Structure:**
```
SYSTEM: {persona_prompt}

USER:
=== Common Pool Resource Game - Round {step} ===

Resource Status:
- Current resource level: {resource}
- Sustainable threshold: {threshold}
- Your goal: Balance personal gain with sustainability

Your History (last {N} rounds):
- Round {n-2}: Extracted {x}, Earned {r}
- Round {n-1}: Extracted {x}, Earned {r}

Other Player's History:
- Round {n-2}: Extracted {x}
- Round {n-1}: Extracted {x}

Current Standings:
- Your total earnings: {payoff}
- Other player's total earnings: {payoff}

Instructions:
1. Analyze the situation
2. Decide how much to extract this round (0-100)
3. Explain your reasoning
4. State your action as "EXTRACT: <number>"

Your response:
```

**Action Parsing:**
- Look for pattern: `EXTRACT: {number}`
- Fallback: Extract last number from response
- Validation: Clip to [MIN_EXTRACTION, MAX_EXTRACTION]

---

### 3. Langfuse Integration (`logging_manager.py`)

**Tracing hierarchy:**
```
game_trace (top level)
├── round_0_span
│   ├── player_0_generation
│   ├── player_1_generation
│   └── round_0_scores
├── round_1_span
│   ├── player_0_generation
│   ├── player_1_generation
│   └── round_1_scores
└── game_summary
```

**Key Methods:**
- `start_game_trace(game_id, config)` - Initialize top-level trace
- `log_round(round_num, observations, actions, rewards)` - Log round span
- `log_generation(player_id, prompt, response, action)` - Log LLM call
- `log_metrics(metric_dict)` - Add scores to current span
- `end_game_trace(summary)` - Finalize and compute game-level metrics

**Custom Scores:**
```python
scores = {
    "cooperation_index": float,  # 0-1
    "sustainability_score": float,  # % rounds above threshold
    "tragedy_occurred": bool,
    "payoff_fairness": float,  # 1 - gini coefficient
}
```

---

### 4. Visualization Dashboard (`dashboard.py`)

**Streamlit-based real-time visualization**

**Layout:**
```
[Header: Game Status]
├── Resource Level: {current} / {initial}
├── Round: {step} / {max_steps}
└── Status: [Running/Depleted/Complete]

[Chart 1: Resource Over Time]
- Line plot with sustainability threshold

[Chart 2: Player Extractions]
- Multi-line plot (one per player)

[Chart 3: Cumulative Payoffs]
- Multi-line plot showing earnings

[Chart 4: Cooperation Index]
- Line plot tracking cooperation metric

[Panel: LLM Reasoning Log]
- Expandable text showing latest reasoning from each agent

[Controls]
- Pause/Resume
- Step Forward
- Reset
```

**Key Methods:**
- `initialize_dashboard()` - Setup Streamlit layout
- `update_charts(game_state)` - Refresh all visualizations
- `display_reasoning(player_id, text)` - Show LLM output
- `show_metrics(metrics_dict)` - Display summary stats

---

### 5. Game Runner (`game_runner.py`)

**Main execution loop**

**Key Methods:**
- `setup_game(config)` - Initialize environment, agents, logging
- `run_episode()` - Execute one full game
- `run_tournament(n_games)` - Run multiple games
- `export_results()` - Save data for analysis

**Execution Flow:**
```
1. Load configuration
2. Initialize CPR environment
3. Create LLM agents with personas
4. Setup Langfuse trace
5. Initialize dashboard
6. For each round:
   a. Get observations for each agent
   b. LLM agents decide actions (logged to Langfuse)
   c. Environment executes step
   d. Update dashboard
   e. Log metrics
7. End game trace
8. Display final summary
```

---

## File Structure

```
cpr_game/
├── config.py                 # All configurable constants
├── cpr_environment.py        # Gymnasium environment
├── llm_agent.py             # LLM-based agent
├── logging_manager.py        # Langfuse integration
├── dashboard.py             # Streamlit visualization
├── game_runner.py           # Main execution script
├── utils.py                 # Helper functions
├── requirements.txt         # Dependencies
└── experiments/
    ├── run_experiment.py    # Experiment scripts
    └── analysis.py          # Post-hoc analysis
```

---

## Dependencies

```txt
# Core
gymnasium==0.29.1
numpy>=1.24.0

# LLM
openai>=1.0.0
langchain>=0.1.0  # Optional: for complex prompting

# Logging
langfuse>=2.0.0

# Visualization
streamlit>=1.31.0
plotly>=5.18.0
pandas>=2.0.0

# Analysis
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
```

---

## Usage Example

```python
from config import CONFIG
from game_runner import GameRunner

# Initialize
runner = GameRunner(CONFIG)

# Run single game with visualization
runner.run_episode(visualize=True)

# Run tournament (multiple games)
results = runner.run_tournament(n_games=10)

# Analyze results
runner.export_results("results.json")
```

---

## Evaluation Metrics

### Tragedy Avoidance
- **Resource Survival Rate:** % of games where resource > 0 at end
- **Average Final Resource:** Mean resource level at termination
- **Depletion Round:** When resource hits 0 (if applicable)

### Emergent Communication
- **Cooperation Index:** Variance in extraction rates (lower = more coordinated)
- **Turn-Taking Patterns:** Detect if players alternate high/low extraction
- **Convergence:** Do extraction rates stabilize over rounds?
- **LLM Reasoning Analysis:** Text analysis of strategic mentions of cooperation

### Fairness
- **Gini Coefficient:** Inequality in cumulative payoffs
- **Payoff Difference:** Absolute difference between players

### Strategy Evolution
- **Extraction Trend:** Are players extracting more or less over time?
- **Response to Other:** Cross-correlation between players' actions
- **Persona Fidelity:** Does behavior match intended persona?

---

## Future Extensions (Post-RL Integration)

- Add RL policy layer (PPO, DQN) trained on rewards
- Hybrid LLM reasoning + RL action selection
- Multi-agent RL with LLM communication channels
- Vary information asymmetry (partial observability)
- Add communication phase between rounds
- Test with more players (3-5)
- Experiment with different regeneration models

---

## Key Design Decisions Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Regeneration Model | 2x multiplication | Simple, fast growth to test cooperation |
| Player Count | 2 | Simplest multi-agent case |
| Personas | Selfish + Cooperative | Clear strategic contrast |
| LLM Model | GPT-3.5-turbo | Cost-effective for experiments |
| Observation Space | Full info | Baseline before partial observability |
| Move Type | Simultaneous | Classic CPR game structure |
| Termination | 100 steps OR depletion | Long enough to see patterns |
| Focus | Tragedy avoidance + communication | Core research questions |

---

## Notes for Claude Code

- All constants should be in `config.py` for easy modification
- Use type hints throughout
- Add comprehensive docstrings
- Include unit tests for environment dynamics
- Make logging optional (can disable Langfuse for faster iteration)
- Dashboard should work both realtime and in replay mode
- Provide example config files for different experiment setups