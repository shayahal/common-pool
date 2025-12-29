#!/usr/bin/env python3
"""Test if game creates and sends traces."""

import sys
import os
import time
sys.path.insert(0, '.')

from cpr_game.game_runner import GameRunner
from cpr_game.config import CONFIG

# Create minimal config
config = CONFIG.copy()
config["n_players"] = 2
config["max_steps"] = 2  # Very short game
config["use_mock_agents"] = True

print("Creating game runner...")
runner = GameRunner(config=config, use_mock_agents=True)

print("Setting up game...")
game_id = runner.setup_game()
print(f"Game ID: {game_id}")

if runner.logger:
    print(f"Logger tracer: {'Active' if runner.logger.tracer else 'None'}")
    print(f"Game ID in logger: {runner.logger.game_id}")
else:
    print("Logger is None!")
    sys.exit(1)

print("\nRunning short game...")
summary = runner.run_episode(visualize=False, verbose=False)

print(f"\nGame completed: {summary.get('total_rounds', 0)} rounds")
print(f"Generation data: {len(runner.logger.get_generation_data())} generations")

# Wait a moment for traces to be sent
print("\nWaiting 2 seconds for traces to be flushed...")
time.sleep(2)

print("\nCheck Langfuse dashboard now.")
print("If no traces appear, check collector logs:")
print("  docker-compose logs otel-collector --tail 50")

