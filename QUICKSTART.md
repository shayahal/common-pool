# Quick Start Guide - CPR Game

Get up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Run Your First Game (No API Keys Needed!)

```bash
python example.py
```

This runs 6 examples demonstrating all major features using **mock agents** (no API keys required).

## Step 3: Try the Quick Game Function

Create a file `quick_test.py`:

```python
from cpr_game.game_runner import quick_game

# Run a quick 20-round game with mock agents
summary = quick_game(
    n_players=2,
    max_steps=20,
    use_mock=True,
    verbose=True
)

print(f"\nFinal Results:")
print(f"  Tragedy: {summary['tragedy_occurred']}")
print(f"  Cooperation: {summary['avg_cooperation_index']:.3f}")
print(f"  Final Resource: {summary['final_resource_level']:.1f}")
```

Run it:
```bash
python quick_test.py
```

## Step 4: Run Tests

```bash
pytest tests/ -v
```

You should see 45+ tests pass!

## Step 5: Run Experiments

```bash
cd experiments
python run_experiment.py
python analysis.py
```

This generates JSON results and analysis plots.

---

## Using Real LLM Agents

### 1. Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Langfuse** (optional): https://cloud.langfuse.com

### 2. Set Environment Variables

Create `.env` file:
```bash
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

### 3. Run with Real Agents

```python
from cpr_game import GameRunner

# Real LLM agents
runner = GameRunner(use_mock_agents=False)
runner.setup_game()
summary = runner.run_episode(verbose=True)
```

---

## Customization Examples

### Change Game Parameters

```python
from cpr_game import GameRunner
from cpr_game.config import CONFIG

config = CONFIG.copy()
config['max_steps'] = 30
config['regeneration_rate'] = 1.5
config['initial_resource'] = 500

runner = GameRunner(config, use_mock_agents=True)
runner.setup_game()
runner.run_episode()
```

### Test Different Personas

```python
config = CONFIG.copy()
config['player_personas'] = ['rational_selfish', 'cooperative']

runner = GameRunner(config, use_mock_agents=True)
runner.setup_game()
runner.run_episode()
```

### Run a Tournament

```python
runner = GameRunner(use_mock_agents=True)
results = runner.run_tournament(n_games=10)

# Save results
runner.export_results("my_tournament.json")
```

---

## Common Issues

### Import Error: No module named 'gymnasium'

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### OpenAI API Error

**Solution**: Use mock agents or set `OPENAI_API_KEY`
```python
runner = GameRunner(use_mock_agents=True)  # No API key needed
```

### Langfuse Connection Error

**Solution**: Ensure Langfuse API keys are set correctly
```python
# Langfuse is always required - set environment variables:
# LANGFUSE_PUBLIC_KEY=your_public_key
# LANGFUSE_SECRET_KEY=your_secret_key
runner = GameRunner(use_mock_agents=True)
```

---

## What's Next?

- 📖 Read `README.md` for full documentation
- 🔬 Check `IMPLEMENTATION_SUMMARY.md` for technical details
- 🧪 Explore `experiments/` for research examples
- 💻 Customize `cpr_game/config.py` for your experiments
- 📊 Use `experiments/analysis.py` to analyze results

---

## Project Structure

```
common-pool/
├── cpr_game/           # Main package
├── experiments/        # Research scripts
├── tests/              # Unit tests
├── example.py          # Usage examples
├── README.md           # Full documentation
└── requirements.txt    # Dependencies
```

---

## Need Help?

- Check `example.py` for working code
- Run tests to verify setup: `pytest tests/ -v`
- Review `IMPLEMENTATION_SUMMARY.md` for details
- All modules have comprehensive docstrings

---

**You're ready to start researching the tragedy of the commons! 🌳**
