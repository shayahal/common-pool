#!/usr/bin/env python3
"""Check for ID duplications and what text goes into RAG."""

from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.config import get_config
import pandas as pd
import json

print("=" * 70)
print("ID DUPLICATION ANALYSIS")
print("=" * 70)

# Check CSV
df = pd.read_csv('data/1-exp-10-games-traces.csv')
print(f"\nCSV: Total rows: {len(df)}")
print(f"CSV: Unique IDs: {df['id'].nunique()}")
print(f"CSV: Duplicate IDs: {len(df) - df['id'].nunique()}")

if len(df) != df['id'].nunique():
    print("\nDuplicate IDs in CSV:")
    dupes = df[df.duplicated(subset=['id'], keep=False)].sort_values('id')
    print(dupes[['id', 'name', 'sessionId']].head(10).to_string())

# Check database
m = Neo4jManager(get_config())

print("\n=== DATABASE ID ANALYSIS ===")
# Check Trace IDs
r = m.execute_query('MATCH (t:Trace) WITH t.id as id, collect(t) as traces WHERE size(traces) > 1 RETURN id, size(traces) as cnt LIMIT 10')
print(f"Trace ID duplicates: {len(r)}")
if r:
    for row in r:
        print(f"  {row['id']}: {row['cnt']} occurrences")

# Check all node types
print("\n=== NODE COUNTS ===")
r = m.execute_query('MATCH (n) RETURN labels(n)[0] as type, count(*) as cnt ORDER BY cnt DESC')
for row in r:
    print(f"  {row['type']}: {row['cnt']}")

# Check ID uniqueness per type
print("\n=== ID UNIQUENESS PER TYPE ===")
for node_type in ['Trace', 'Session', 'Span', 'Generation']:
    r = m.execute_query(f'MATCH (n:{node_type}) RETURN count(DISTINCT n.id) as unique_ids, count(n) as total')
    if r:
        row = r[0]
        if row['total'] > 0:
            print(f"  {node_type}: {row['unique_ids']} unique IDs, {row['total']} total nodes")
            if row['unique_ids'] != row['total']:
                print(f"    WARNING: {row['total'] - row['unique_ids']} duplicate IDs!")

print("\n" + "=" * 70)
print("RAG TEXT CONTENT ANALYSIS")
print("=" * 70)

# Check what text fields exist
print("\n=== TEXT FIELDS IN DATABASE ===")
r = m.execute_query('''
    MATCH (g:Generation)
    WHERE g.prompt IS NOT NULL OR g.response IS NOT NULL OR g.reasoning IS NOT NULL
    RETURN count(*) as cnt
''')
print(f"Generations with text: {r[0]['cnt'] if r else 0}")

r = m.execute_query('''
    MATCH (t:Trace)
    WHERE t.input IS NOT NULL OR t.output IS NOT NULL
    RETURN count(*) as cnt
''')
print(f"Traces with text: {r[0]['cnt'] if r else 0}")

r = m.execute_query('''
    MATCH (s:Span)
    WHERE s.input IS NOT NULL OR s.output IS NOT NULL
    RETURN count(*) as cnt
''')
print(f"Spans with text: {r[0]['cnt'] if r else 0}")

# Sample text content
print("\n=== SAMPLE TEXT CONTENT ===")
r = m.execute_query('''
    MATCH (t:Trace)
    WHERE t.input IS NOT NULL
    RETURN t.id, t.name, left(t.input, 150) as input_preview, left(t.output, 150) as output_preview
    LIMIT 3
''')
print("Sample Trace text:")
for row in r:
    print(f"  {row['t.name']}:")
    print(f"    Input: {row['input_preview']}...")
    print(f"    Output: {row['output_preview']}...")

# Check what GraphRAG would extract
print("\n=== WHAT GRAPHRAG EXTRACTS ===")
print("Current GraphRAG extraction logic:")
print("  1. Generation entities: prompt, response, reasoning")
print("  2. Trace entities: input, output")
print("  3. Error entities: message")
print("\nCurrent state:")
print(f"  Generation entities: {m.execute_query('MATCH (g:Generation) RETURN count(*) as cnt')[0]['cnt']}")
print(f"  Trace entities with input: {m.execute_query('MATCH (t:Trace) WHERE t.input IS NOT NULL RETURN count(*) as cnt')[0]['cnt']}")
print(f"  Trace entities with output: {m.execute_query('MATCH (t:Trace) WHERE t.output IS NOT NULL RETURN count(*) as cnt')[0]['cnt']}")

# Check CSV input/output
print("\n=== CSV INPUT/OUTPUT CONTENT ===")
player_actions = df[df['name'].str.contains('player_.*_action', na=False, regex=True)]
print(f"Player action rows: {len(player_actions)}")
print(f"  With input: {player_actions['input'].notna().sum()}")
print(f"  With output: {player_actions['output'].notna().sum()}")

if len(player_actions) > 0:
    sample = player_actions.iloc[0]
    print("\nSample player action input/output:")
    input_val = str(sample['input'])
    output_val = str(sample['output'])
    print(f"  Input length: {len(input_val)} chars")
    print(f"  Input preview: {input_val[:200]}...")
    print(f"  Output length: {len(output_val)} chars")
    print(f"  Output preview: {output_val[:200]}...")

m.close()
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

