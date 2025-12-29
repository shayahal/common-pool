#!/usr/bin/env python3
"""Test if game creates traces properly."""

import sys
import os
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
print(f"Logger: {runner.logger}")
print(f"Tracer: {runner.logger.tracer if runner.logger else None}")

if runner.logger and runner.logger.tracer:
    print("✓ OpenTelemetry is enabled")
else:
    print("✗ OpenTelemetry is DISABLED - traces won't be created")
    print("  Check OTEL_ENABLED environment variable")

print("\nRunning short game...")
summary = runner.run_episode(visualize=False, verbose=False)

print(f"\nGame completed: {summary.get('total_rounds', 0)} rounds")
print(f"Final resource: {summary.get('final_resource_level', 0)}")

if runner.logger:
    print(f"\nGeneration data count: {len(runner.logger.get_generation_data())}")
    print(f"Round metrics count: {len(runner.logger.get_round_metrics())}")

print("\nCheck Langfuse dashboard for traces.")
print("If no traces appear, check:")
print("1. OTEL_ENABLED=true in .env")
print("2. Collector is running: docker ps | grep otel-collector")
print("3. Collector logs: docker-compose logs otel-collector --tail 50")

