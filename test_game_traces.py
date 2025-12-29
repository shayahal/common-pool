#!/usr/bin/env python3
"""Test if game creates and flushes traces properly."""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from cpr_game.game_runner import GameRunner
from cpr_game.config import CONFIG

# Create minimal config for testing
config = CONFIG.copy()
config["n_players"] = 2
config["max_steps"] = 2  # Very short game
config["use_mock_agents"] = True  # Use mock agents to avoid API calls

print("Creating game runner...")
runner = GameRunner(config=config, use_mock_agents=True)

print("Setting up game...")
game_id = runner.setup_game()
print(f"Game ID: {game_id}")
print(f"Tracer exists: {runner.logger.tracer is not None if runner.logger else False}")

if not runner.logger or not runner.logger.tracer:
    print("ERROR: OpenTelemetry is not enabled!")
    sys.exit(1)

print("\nRunning game episode...")
summary = runner.run_episode(visualize=False, verbose=False)

print(f"\nGame completed:")
print(f"  Rounds: {summary.get('total_rounds', 0)}")
print(f"  Generations logged: {len(runner.logger.get_generation_data())}")
print(f"  Round metrics: {len(runner.logger.get_round_metrics())}")

print("\nWaiting 2 seconds for traces to be exported...")
time.sleep(2)

print("\nCheck Langfuse dashboard for traces:")
print("  https://cloud.langfuse.com")
print("\nCheck collector logs:")
print("  docker-compose logs otel-collector --tail 50")

