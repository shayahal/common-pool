#!/usr/bin/env python3
"""Compute Langfuse Basic Auth header from .env file and update docker-compose environment."""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

public_key = os.getenv('LANGFUSE_PUBLIC_KEY', '')
secret_key = os.getenv('LANGFUSE_SECRET_KEY', '')

if not public_key or not secret_key:
    print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env file")
    exit(1)

# Compute Basic Auth header
auth_string = f"{public_key}:{secret_key}"
auth_b64 = base64.b64encode(auth_string.encode()).decode()
auth_header = f"Basic {auth_b64}"

print(f"Computed LANGFUSE_AUTH_HEADER: Basic <base64-encoded>")
print(f"Length: {len(auth_header)} characters")

# Write to .env.local for container environment
env_local_path = Path(__file__).parent / '.env.local'
with open(env_local_path, 'w', encoding='utf-8') as f:
    f.write(f"LANGFUSE_AUTH_HEADER={auth_header}\n")
    f.write(f"LANGFUSE_AUTH_B64={auth_b64}\n")

# Also append to .env for docker-compose variable substitution
# (docker-compose reads .env by default for ${} substitution)
env_path = Path(__file__).parent / '.env'
env_content = env_path.read_text(encoding='utf-8') if env_path.exists() else ''

# Remove old LANGFUSE_AUTH_HEADER if present
lines = [line for line in env_content.split('\n') if not line.startswith('LANGFUSE_AUTH_HEADER=')]

# Add new one
lines.append(f"LANGFUSE_AUTH_HEADER={auth_header}")

# Write back
with open(env_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
    if not lines[-1].endswith('\n'):
        f.write('\n')

print(f"Written to {env_local_path}")
print(f"Updated {env_path}")
print("Now run: docker-compose restart otel-collector")

