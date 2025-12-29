"""Diagnostic script to understand ID assignment issues."""

import csv
import json
from langfuse_graphrag.csv_parser import LangfuseCSVParser
from langfuse_graphrag.extractor import EntityExtractor

# Parse CSV
parser = LangfuseCSVParser('data/1-exp-10-games-traces.csv')
csv_type = parser.detect_csv_type()
print(f"Detected CSV type: {csv_type}")

records = list(parser.parse())
print(f"Parsed {len(records)} records")

# Show first few records
print("\n=== First 3 parsed records ===")
for i, record in enumerate(records[:3]):
    print(f"\nRecord {i+1}:")
    print(f"  _csv_type: {record.get('_csv_type')}")
    print(f"  id: {record.get('id')}")
    print(f"  session_id: {record.get('session_id')}")
    print(f"  sessionId: {record.get('sessionId')}")
    print(f"  name: {record.get('name')}")

# Extract entities
extractor = EntityExtractor()
entities = extractor.extract_entities(records)

print("\n=== Extracted entities ===")
print(f"Sessions: {len(entities.get('Session', []))}")
print(f"Traces: {len(entities.get('Trace', []))}")

# Show first few traces
print("\n=== First 5 Trace entities ===")
for i, trace in enumerate(entities.get('Trace', [])[:5]):
    print(f"\nTrace {i+1}:")
    print(f"  id: {trace.get('id')}")
    print(f"  trace_id: {trace.get('trace_id')}")
    print(f"  session_id: {trace.get('session_id')}")
    print(f"  name: {trace.get('name')}")
    print(f"  game_id: {trace.get('game_id')}")
    print(f"  round: {trace.get('round')}")

# Show first few sessions
print("\n=== First 5 Session entities ===")
for i, session in enumerate(entities.get('Session', [])[:5]):
    print(f"\nSession {i+1}:")
    print(f"  id: {session.get('id')}")
    print(f"  session_id: {session.get('session_id')}")
    print(f"  name: {session.get('name')}")

# Check for duplicate trace IDs
trace_ids = [t.get('id') for t in entities.get('Trace', [])]
print(f"\n=== Trace ID analysis ===")
print(f"Total traces: {len(trace_ids)}")
print(f"Unique trace IDs: {len(set(trace_ids))}")
if len(trace_ids) != len(set(trace_ids)):
    from collections import Counter
    duplicates = {k: v for k, v in Counter(trace_ids).items() if v > 1}
    print(f"Duplicate trace IDs: {len(duplicates)}")
    print(f"Sample duplicates: {list(duplicates.items())[:5]}")

# Check session ID usage
print(f"\n=== Session ID analysis ===")
session_ids = [s.get('id') for s in entities.get('Session', [])]
print(f"Total sessions: {len(session_ids)}")
print(f"Unique session IDs: {len(set(session_ids))}")

# Check how many traces per session
from collections import defaultdict
traces_by_session = defaultdict(list)
for trace in entities.get('Trace', []):
    session_id = trace.get('session_id')
    if session_id:
        traces_by_session[session_id].append(trace.get('id'))

print(f"\n=== Traces per session ===")
for session_id, trace_list in list(traces_by_session.items())[:10]:
    print(f"Session {session_id}: {len(trace_list)} traces")

