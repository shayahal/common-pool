#!/usr/bin/env python3
"""Check if traces should be spans based on metadata."""

import pandas as pd
import json

df = pd.read_csv('data/1-exp-10-games-traces.csv')

print("=" * 70)
print("SPAN ANALYSIS")
print("=" * 70)

# Check metadata for span indicators
player_actions = df[df['name'].str.contains('player_.*_action', na=False, regex=True)]
print(f"\nPlayer action rows: {len(player_actions)}")

if len(player_actions) > 0:
    sample = player_actions.iloc[0]
    if pd.notna(sample['metadata']):
        try:
            meta = json.loads(sample['metadata'])
            if 'attributes' in meta:
                attrs = json.loads(meta['attributes'])
                print("\nMetadata attributes relevant to spans:")
                print(f"  langfuse.observation.type: {attrs.get('langfuse.observation.type', 'NOT FOUND')}")
                print(f"  trace_id: {attrs.get('trace_id', 'NOT FOUND')}")
                print(f"  duration_ms: {attrs.get('duration_ms', 'NOT FOUND')}")
                print(f"  end_time: {attrs.get('end_time', 'NOT FOUND')}")
                print(f"  game.id: {attrs.get('game.id', 'NOT FOUND')}")
                print(f"  round.number: {attrs.get('round.number', 'NOT FOUND')}")
        except Exception as e:
            print(f"Error parsing metadata: {e}")

# Check if we can identify parent traces
print("\n=== TRACE STRUCTURE ===")
print("Checking trace_id patterns in metadata...")
trace_ids = set()
for idx, row in player_actions.head(10).iterrows():
    if pd.notna(row['metadata']):
        try:
            meta = json.loads(row['metadata'])
            if 'attributes' in meta:
                attrs = json.loads(meta['attributes'])
                trace_id = attrs.get('trace_id')
                if trace_id:
                    trace_ids.add(trace_id)
                    print(f"  Row {idx}: trace_id={trace_id[:50]}...")
        except:
            pass

print(f"\nUnique trace_ids found: {len(trace_ids)}")

# Check sessionId vs trace_id
print("\n=== SESSION vs TRACE ===")
sample = player_actions.iloc[0]
print(f"  sessionId (from CSV): {sample['sessionId']}")
if pd.notna(sample['metadata']):
    try:
        meta = json.loads(sample['metadata'])
        if 'attributes' in meta:
            attrs = json.loads(meta['attributes'])
            print(f"  trace_id (from metadata): {attrs.get('trace_id', 'NOT FOUND')}")
            print(f"  session_id (from metadata): {attrs.get('session_id', 'NOT FOUND')}")
            print(f"  game.id (from metadata): {attrs.get('game.id', 'NOT FOUND')}")
    except:
        pass

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

