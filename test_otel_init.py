#!/usr/bin/env python3
"""Test OpenTelemetry initialization."""

import sys
import os
sys.path.insert(0, '.')

from cpr_game.otel_manager import OTelManager
from cpr_game.config import CONFIG

print("Checking OpenTelemetry configuration...")
print(f"OTEL_ENABLED (env): {os.getenv('OTEL_ENABLED', 'not set')}")
print(f"OTEL_ENABLED (config): {CONFIG.get('otel_enabled')}")
print(f"OTEL_ENDPOINT: {CONFIG.get('otel_endpoint')}")
print(f"OTEL_PROTOCOL: {CONFIG.get('otel_protocol')}")

try:
    print("\nInitializing OTelManager...")
    manager = OTelManager(CONFIG)
    if manager.tracer is None:
        print("[X] Tracer is None - OpenTelemetry is disabled or failed to initialize")
    else:
        print("[OK] Tracer is active - OpenTelemetry is working")
        print(f"  Tracer name: {manager.tracer}")
except Exception as e:
    print(f"[ERROR] Error initializing OTelManager: {e}")
    import traceback
    traceback.print_exc()

