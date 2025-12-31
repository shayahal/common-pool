#!/usr/bin/env python3
"""Analyze CSV structure to identify all missing fields."""

import pandas as pd
import json

df = pd.read_csv('data/1-exp-10-games-traces.csv')

print("=" * 70)
print("CSV ANALYSIS")
print("=" * 70)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Check name patterns
print("\n=== NAME PATTERNS ===")
player_action = df['name'].str.contains('player_.*_action', na=False, regex=True).sum()
game_summary = df['name'].str.contains('game_summary', na=False).sum()
print(f"  player_X_action: {player_action}")
print(f"  game_summary: {game_summary}")
print(f"  Other: {len(df) - player_action - game_summary}")

# Check input/output
print("\n=== INPUT/OUTPUT ===")
print(f"  Rows with input: {df['input'].notna().sum()}")
print(f"  Rows with output: {df['output'].notna().sum()}")
print(f"  Rows with both: {(df['input'].notna() & df['output'].notna()).sum()}")

# Check game_summary rows for populated metrics
print("\n=== GAME SUMMARY ROWS ===")
summary_rows = df[df['name'].str.contains('game_summary', na=False)]
print(f"  Found {len(summary_rows)} game_summary rows")
if len(summary_rows) > 0:
    row = summary_rows.iloc[0]
    print("\n  Populated columns in game_summary:")
    for col in df.columns:
        val = row[col]
        if pd.notna(val) and str(val) != 'nan' and str(val).strip() != '':
            val_str = str(val)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"    {col}: {val_str}")

# Check metadata structure
print("\n=== METADATA STRUCTURE ===")
if df['metadata'].notna().sum() > 0:
    sample_meta = df[df['metadata'].notna()].iloc[0]['metadata']
    try:
        meta_dict = json.loads(sample_meta)
        print("  Top-level keys:", list(meta_dict.keys()))
        if 'attributes' in meta_dict:
            try:
                attrs = json.loads(meta_dict['attributes'])
                print("  Attributes keys:", list(attrs.keys())[:20])
            except:
                print("  Attributes is not JSON string")
    except:
        print("  Metadata is not valid JSON")

# Check which columns have non-null values
print("\n=== COLUMN POPULATION ===")
print("  Columns with >0 non-null values:")
for col in sorted(df.columns):
    non_null = df[col].notna().sum()
    if non_null > 0:
        pct = (non_null / len(df)) * 100
        print(f"    {col}: {non_null} ({pct:.1f}%)")

# Check round_metrics and game_summary output
print("\n=== ROUND METRICS ROWS ===")
round_metrics = df[df['name'].str.contains('round_.*_metrics', na=False, regex=True)]
print(f"  Found {len(round_metrics)} round_metrics rows")
if len(round_metrics) > 0:
    row = round_metrics.iloc[0]
    print(f"  Sample name: {row['name']}")
    if pd.notna(row.get('output')):
        try:
            output = json.loads(row['output'])
            if isinstance(output, dict):
                print(f"  Output keys: {list(output.keys())[:10]}")
        except:
            print(f"  Output (first 200 chars): {str(row['output'])[:200]}")

print("\n=== GAME SUMMARY OUTPUT ===")
summary = df[df['name'].str.contains('game_summary', na=False)]
if len(summary) > 0:
    row = summary.iloc[0]
    if pd.notna(row.get('output')):
        try:
            output_str = row['output']
            # Remove quotes if present
            if output_str.startswith('"') and output_str.endswith('"'):
                output_str = output_str[1:-1]
            output = json.loads(output_str)
            if isinstance(output, dict):
                print(f"  Output keys: {list(output.keys())[:30]}")
        except Exception as e:
            print(f"  Error parsing output: {e}")
            print(f"  Output (first 500 chars): {str(row['output'])[:500]}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

