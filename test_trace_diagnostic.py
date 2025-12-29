#!/usr/bin/env python3
"""Diagnostic test for trace export."""

import sys
import os
import time
sys.path.insert(0, '.')

from cpr_game.game_runner import GameRunner
from cpr_game.config import CONFIG
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

# Create minimal config
config = CONFIG.copy()
config["n_players"] = 2
config["max_steps"] = 1  # Very short game
config["use_mock_agents"] = True

print("Creating game runner...")
runner = GameRunner(config=config, use_mock_agents=True)

print("Setting up game...")
game_id = runner.setup_game()
print(f"Game ID: {game_id}")

if runner.logger and runner.logger.otel_manager:
    otel_mgr = runner.logger.otel_manager
    print(f"\nOTel Manager Diagnostics:")
    print(f"  Tracer: {'Active' if otel_mgr.tracer else 'None'}")
    print(f"  Provider type: {type(otel_mgr.tracer_provider).__name__}")
    print(f"  Provider owned: {otel_mgr._provider_owned}")
    print(f"  Span processor: {'Set' if hasattr(otel_mgr, '_span_processor') and otel_mgr._span_processor else 'None'}")
    
    # Check provider span processors
    if isinstance(otel_mgr.tracer_provider, SDKTracerProvider):
        processors = getattr(otel_mgr.tracer_provider, '_span_processors', [])
        print(f"  Provider has {len(processors)} span processor(s)")
        for i, proc in enumerate(processors):
            print(f"    Processor {i}: {type(proc).__name__}")

print("\nRunning short game...")
summary = runner.run_episode(visualize=False, verbose=False)

print(f"\nGame completed: {summary.get('total_rounds', 0)} rounds")
print(f"Generation data: {len(runner.logger.get_generation_data())} generations")

# Wait for traces
print("\nWaiting 3 seconds for traces to be flushed and sent...")
time.sleep(3)

print("\nCheck collector logs:")
print("  docker-compose logs otel-collector --tail 50")

