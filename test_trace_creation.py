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

print("=" * 60)
print("Testing Game Trace Creation")
print("=" * 60)

print("\n1. Creating game runner...")
runner = GameRunner(config=config, use_mock_agents=True)

print("\n2. Setting up game...")
game_id = runner.setup_game()
print(f"   Game ID: {game_id}")
print(f"   Logger exists: {runner.logger is not None}")
print(f"   Tracer exists: {runner.logger.tracer is not None if runner.logger else False}")

if runner.logger and runner.logger.tracer:
    print("   ✓ OpenTelemetry is enabled")
else:
    print("   ✗ OpenTelemetry is DISABLED")
    sys.exit(1)

print("\n3. Running game episode...")
print("   (This should create traces and flush them)")
summary = runner.run_episode(visualize=False, verbose=False)

print(f"\n4. Game completed:")
print(f"   Rounds: {summary.get('total_rounds', 0)}")
print(f"   Generations logged: {len(runner.logger.get_generation_data())}")
print(f"   Round metrics: {len(runner.logger.get_round_metrics())}")

print("\n5. Waiting for traces to be exported...")
time.sleep(2)  # Give collector time to forward traces

print("\n" + "=" * 60)
print("Check Langfuse dashboard for traces")
print("=" * 60)
print("\nIf no traces appear, check:")
print("1. Collector logs: docker-compose logs otel-collector --tail 50")
print("2. Game logs for errors")
print("3. Langfuse dashboard: https://cloud.langfuse.com")

