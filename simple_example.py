#!/usr/bin/env python
"""Simple CPR Game Example - Works immediately, no API keys needed!"""

from cpr_game import GameRunner
from cpr_game.config import CONFIG

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           CPR Game - Simple Working Example                              ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

print("Example 1: Basic 20-round Game")
print("=" * 75)

# Create runner with mock agents (no API keys needed!)
runner = GameRunner(use_mock_agents=True)
runner.setup_game("demo_game")

# Run game without visualization (to avoid Streamlit warnings)
summary = runner.run_episode(visualize=False, verbose=False)

# Display results
print(f"\n📊 RESULTS:")
print(f"  Total Rounds: {summary['total_rounds']}")
print(f"  Final Resource: {summary['final_resource_level']:.2f}")
print(f"  Tragedy Occurred: {'Yes' if summary['tragedy_occurred'] else 'No ✓'}")
print(f"  Cooperation Index: {summary['avg_cooperation_index']:.3f}/1.000")
print(f"  Payoff Inequality (Gini): {summary['gini_coefficient']:.3f}")
print(f"  Sustainability: {summary['sustainability_score']:.1%}")

print(f"\n  Player 0 Earnings: {summary['cumulative_payoffs'][0]:.2f}")
print(f"  Player 1 Earnings: {summary['cumulative_payoffs'][1]:.2f}")

print("\n" + "=" * 75)
print("Example 2: Shorter Game with Custom Settings")
print("=" * 75)

# Custom configuration
config = CONFIG.copy()
config['max_steps'] = 30
config['regeneration_rate'] = 1.5
config['initial_resource'] = 500

runner2 = GameRunner(config, use_mock_agents=True)
runner2.setup_game("custom_game")
summary2 = runner2.run_episode(visualize=False, verbose=False)

print(f"\n📊 RESULTS (30 rounds, 1.5x regen, 500 initial):")
print(f"  Final Resource: {summary2['final_resource_level']:.2f}")
print(f"  Tragedy: {'Yes' if summary2['tragedy_occurred'] else 'No ✓'}")
print(f"  Cooperation: {summary2['avg_cooperation_index']:.3f}")

print("\n" + "=" * 75)
print("Example 3: Tournament (5 games)")
print("=" * 75)

runner3 = GameRunner(use_mock_agents=True, )
results = []

for i in range(5):
    runner3.setup_game(f"tournament_{i}")
    result = runner3.run_episode(visualize=False, verbose=False)
    results.append(result)
    status = "DEPLETED" if result['tragedy_occurred'] else "SUSTAINED"
    print(f"  Game {i+1}: {result['total_rounds']:3d} rounds, {status:10s}, Coop: {result['avg_cooperation_index']:.3f}")

# Aggregate stats
import numpy as np
tragedy_rate = sum(1 for r in results if r['tragedy_occurred']) / len(results)
avg_coop = np.mean([r['avg_cooperation_index'] for r in results])

print(f"\n  Tournament Summary:")
print(f"    Tragedy Rate: {tragedy_rate:.1%}")
print(f"    Avg Cooperation: {avg_coop:.3f}")

print("\n" + "=" * 75)
print("Example 4: Using Environment Directly")
print("=" * 75)

from cpr_game.cpr_environment import CPREnvironment
import numpy as np

env = CPREnvironment()
obs, info = env.reset()

print(f"\n  Initial resource: {info['resource']:.1f}")

# Run 5 rounds with random actions
for i in range(5):
    actions = np.random.uniform(0, 30, size=2)
    obs, rewards, terminated, truncated, info = env.step(actions)

    print(f"  Round {i+1}: Resource={info['resource']:7.1f}, "
          f"Extractions=[{actions[0]:.1f}, {actions[1]:.1f}], "
          f"Coop={info['cooperation_index']:.2f}")

    if terminated or truncated:
        break

print("\n" + "=" * 75)
print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 75)

print("""
🎯 What You Just Saw:
  ✅ Basic game execution (100 rounds)
  ✅ Custom configuration
  ✅ Tournament mode (multiple games)
  ✅ Direct environment usage

📚 Next Steps:
  1. Try with real LLM agents: Set OPENAI_API_KEY and use use_mock_agents=False
  2. Enable Langfuse logging: Set LANGFUSE keys
  3. Run experiments: cd experiments && python run_experiment.py
  4. Run tests: pytest tests/ -v

💡 The implementation is complete and fully functional!
""")
