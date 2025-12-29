#!/usr/bin/env python3
"""Test Langfuse OTel endpoint directly."""

import os
import requests

# Load env vars
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

auth = os.environ.get('LANGFUSE_AUTH_HEADER', '')
print(f"Auth header: {auth[:40]}...")

# Test the endpoint
url = 'https://cloud.langfuse.com/api/public/otel/v1/traces'
print(f"\nTesting endpoint: {url}")

# Try with empty payload first
r = requests.post(url, headers={
    'Authorization': auth,
    'Content-Type': 'application/x-protobuf'
}, data=b'')

print(f"Status: {r.status_code}")
print(f"Response: {r.text[:500] if r.text else '(empty)'}")
print(f"Headers: {dict(r.headers)}")

