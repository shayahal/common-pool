# Common Pool Resource Game - RL Environment

A Reinforcement Learning environment for studying the Common Pool Resource (CPR) problem with LLM-based agents. Built for research on tragedy avoidance and emergent communication patterns in multi-agent systems.

## Features

- **Gymnasium-compatible environment** for future RL integration
- **LLM-based agents** with customizable personas (GPT-3.5/4)
- **OpenTelemetry integration** for standardized tracing (single source of truth)
- **Langfuse & LangSmith** as OTel receivers for LLM-specific analysis
- **Streamlit dashboard** for real-time visualization
- **Comprehensive metrics** (cooperation index, Gini coefficient, sustainability score)
- **Experiment framework** for systematic research
- **Full test coverage** with pytest

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with your API keys and OpenTelemetry configuration:

```bash
# OpenAI API Key (required for LLM agents)
OPENAI_API_KEY=sk-...

# OpenTelemetry Configuration (Single Source of Truth)
OTEL_SERVICE_NAME=cpr-game
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_SERVICE_VERSION=1.0.0
OTEL_ENABLED=true

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

**Note**: 
- Set `OTEL_RECEIVER` to "langfuse", "langsmith", or "both" to control which receiver(s) receive traces
- If using an OpenTelemetry Collector (recommended for production), set `OTEL_EXPORTER_OTLP_ENDPOINT` to your collector endpoint (default: `http://localhost:4317`)
- Configure the collector's `otel-collector-config.yaml` to route to the selected receiver(s) based on `OTEL_RECEIVER`

### Run a Quick Game

```python
from cpr_game import GameRunner

# Use mock agents (no API calls needed)
runner = GameRunner(use_mock_agents=True)
runner.setup_game()
summary = runner.run_episode(verbose=True)

print(f"Tragedy occurred: {summary['tragedy_occurred']}")
print(f"Cooperation index: {summary['avg_cooperation_index']:.3f}")
```

### Run with Real LLM Agents

```python
from cpr_game import GameRunner

# Requires OPENAI_API_KEY environment variable
runner = GameRunner(use_mock_agents=False)
runner.setup_game()
summary = runner.run_episode(verbose=True)
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
common-pool/
├── cpr_game/
│   ├── config.py              # Configuration parameters
│   ├── cpr_environment.py     # Gymnasium environment
│   ├── llm_agent.py          # LLM-based agents
│   ├── otel_manager.py        # OpenTelemetry manager (single source)
│   ├── logging_manager.py     # OTel-based tracing integration
│   ├── dashboard.py          # Streamlit visualization
│   ├── game_runner.py        # Main orchestration
│   └── utils.py              # Helper functions
├── otel-collector-config.yaml # OTel Collector configuration (optional)
├── experiments/
│   ├── run_experiment.py     # Experiment scripts
│   └── analysis.py           # Analysis tools
├── tests/
│   ├── test_environment.py
│   ├── test_agents.py
│   └── test_utils.py
└── requirements.txt
```

## Game Mechanics

### Resource Dynamics

```
R(t+1) = max(R(t) * regeneration_rate - sum(extractions), 0)
```

Default: Resource doubles each round (regeneration_rate = 2.0)

### Reward Function

```python
reward = extraction  # (value per unit is 1)

if resource >= sustainability_threshold:
    reward += sustainability_bonus

if resource <= 0:
    reward += depletion_penalty / n_players
```

### Termination

Game ends when:
1. Resource depleted (tragedy)
2. Maximum steps reached (100 by default)

## Configuration

Edit `cpr_game/config.py` or pass custom config dict:

```python
from cpr_game import GameRunner
from cpr_game.config import CONFIG

custom_config = CONFIG.copy()
custom_config['n_players'] = 3
custom_config['max_steps'] = 50
custom_config['regeneration_rate'] = 1.5

runner = GameRunner(custom_config)
```

### Key Parameters

- `N_PLAYERS`: Number of players (default: 2)
- `MAX_STEPS`: Maximum rounds (default: 100)
- `INITIAL_RESOURCE`: Starting resource (default: 1000)
- `REGENERATION_RATE`: Resource growth rate (default: 2.0)
- `SUSTAINABILITY_THRESHOLD`: Bonus threshold (default: 500)

### Agent Personas

Available personas:
- `rational_selfish`: Maximizes individual payoff
- `cooperative`: Prioritizes group welfare
- `""`: Neutral agent

## Running Experiments

### Persona Comparison

```bash
cd experiments
python run_experiment.py
```

This runs multiple experiments:
- Persona matchups (selfish vs cooperative)
- Regeneration rate sensitivity
- Player count scaling
- Sustainability threshold variations

### Analyze Results

```bash
cd experiments
python analysis.py
```

Generates:
- Summary statistics
- Comparison plots
- Aggregated metrics across experiments

## Visualization Dashboard

```python
# (Streamlit integration - work in progress)
# The dashboard.py module is ready for Streamlit app integration
```

## Research Metrics

### Tragedy Avoidance
- Resource survival rate
- Average final resource
- Depletion round (if applicable)

### Emergent Communication
- Cooperation index (extraction variance)
- Turn-taking patterns
- LLM reasoning analysis

### Fairness
- Gini coefficient (payoff inequality)
- Payoff difference between players

## Logging

The codebase uses a multi-tier logging system:

### Log Destinations

1. **`logs/cpr_game.log`** - Main application log
   - All Python logger calls (logger.info(), logger.error(), etc.)
   - DEBUG level and above
   - Text format with timestamps

2. **Console/stdout** - Application output
   - INFO level and above
   - Real-time feedback during game execution

3. **OpenTelemetry** - Distributed tracing and observability
   - All traces exported via OTLP (OpenTelemetry Protocol)
   - Traces sent to configured receivers based on `OTEL_RECEIVER`:
     - **Langfuse**: LLM-specific analysis and visualization
     - **LangSmith**: LangChain debugging and prompt evaluation
     - **Both**: Send to both receivers for comparison
   - API metrics (tokens, costs, latency) are tracked via OpenTelemetry spans
   - View API metrics in Langfuse/LangSmith dashboards
   - Requires `OTEL_EXPORTER_OTLP_ENDPOINT` (collector or direct endpoint)
   - Langfuse/LangSmith API keys required for authentication
   - Used for research analysis, debugging, and monitoring

### Log Levels

- **DEBUG**: Detailed debugging information (file only)
- **INFO**: General informational messages (file + console)
- **WARNING**: Warning messages about potential issues
- **ERROR**: Error messages with exception details

### Configuration

Logging is configured automatically when `GameRunner.setup_game()` is called.
The log directory can be customized via the `log_dir` config parameter (default: `"logs"`).

For detailed logging system documentation, see [docs/LOGGING.md](docs/LOGGING.md).

### OpenTelemetry Architecture

The project uses OpenTelemetry as the **single source of truth** for all tracing:

```
Application (Python)
    ├── OpenTelemetry SDK
    │   ├── LangChain Auto-instrumentation
    │   └── Custom spans for game logic
    │
    └── OTLP Exporter → OpenTelemetry Collector (optional)
        │
        └── Exporters:
            ├── OTLP → Langfuse (LLM analysis)
            ├── OTLP → LangSmith (LangChain debugging)
            └── OTLP → Long-term Storage (optional)
```

**Benefits:**
- Single instrumentation point eliminates inconsistencies
- Standard OTel format works with any compatible backend
- Easy to add/remove receivers without code changes
- Consistent trace data across all platforms

**Setup Options:**

1. **Direct Export** (Simplest): Set `OTEL_EXPORTER_OTLP_ENDPOINT` to Langfuse/LangSmith endpoints directly
2. **Collector Pattern** (Recommended): Use OpenTelemetry Collector for flexible routing and processing

See `otel-collector-config.yaml` for collector configuration example.

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=cpr_game tests/

# Specific test file
pytest tests/test_environment.py -v
```

### Adding New Personas

Edit `cpr_game/config.py`:

```python
PLAYER_PERSONAS = [
    "rational_selfish",
    "cooperative",
    "your_new_persona",  # Add here
]

PERSONA_PROMPTS = {
    "your_new_persona": """Your custom system prompt here...""",
}
```

## Future Extensions

- RL policy layer (PPO, DQN)
- Hybrid LLM reasoning + RL actions
- Communication channels between agents
- Partial observability experiments
- More players (3-5)
- Alternative regeneration models

## Citation

If you use this code in your research, please cite:

```
@software{cpr_game_2024,
  title={Common Pool Resource Game - RL Environment},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/common-pool}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.
