#!/usr/bin/env python3
"""Test sending traces directly to Langfuse using their SDK."""

import os
import sys

# Load env vars from .env file
def load_env_file(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

load_env_file('.env')
load_env_file('.env.local')

# Check env vars
pk = os.environ.get('LANGFUSE_PUBLIC_KEY', '')
sk = os.environ.get('LANGFUSE_SECRET_KEY', '')
print(f"LANGFUSE_PUBLIC_KEY: {pk[:20] if pk else 'NOT SET'}...")
print(f"LANGFUSE_SECRET_KEY: {sk[:20] if sk else 'NOT SET'}...")

if not pk or not sk:
    print("ERROR: Langfuse credentials not set!")
    sys.exit(1)

# Test Langfuse SDK
from langfuse import Langfuse

print("\nInitializing Langfuse client...")
lf = Langfuse(
    public_key=pk,
    secret_key=sk,
    host="https://cloud.langfuse.com",
    debug=True
)

print("Creating span (trace)...")
with lf.start_as_current_span(name="test-trace-from-script") as span:
    print(f"Trace ID: {lf.get_current_trace_id()}")
    
    print("Creating generation...")
    with lf.start_as_current_generation(
        name="test-generation",
        input="What is 2+2?",
        output="4",
        model="test-model"
    ) as gen:
        print(f"Generation ID: {lf.get_current_observation_id()}")

print("Flushing...")
lf.flush()

print("\n[SUCCESS] Trace sent to Langfuse!")
print("Check your dashboard: https://cloud.langfuse.com")

