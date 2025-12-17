#!/usr/bin/env python3
"""Quick experiment to test Langfuse logging."""

from cpr_game.game_runner import GameRunner
from cpr_game.config import CONFIG

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING QUICK EXPERIMENT")
    print("=" * 60)

    # Configure for a quick experiment
    config = CONFIG.copy()
    config["n_players"] = 2
    config["max_steps"] = 10  # Short game

    # Use mock agents for speed (no real LLM calls)
    runner = GameRunner(config, use_mock_agents=True)

    # Setup and run
    runner.setup_game(game_id="quick_experiment_test")
    summary = runner.run_episode(visualize=False, verbose=True)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Check Langfuse dashboard for trace: quick_experiment_test")
    print("=" * 60)
