# ID Duplication and RAG Text Content Analysis

## ID Duplication Status

✅ **No ID duplications found:**
- CSV: 698 rows, 698 unique IDs (0 duplicates)
- Database Trace nodes: 698 unique IDs, 698 total nodes (0 duplicates)
- Database Session nodes: 10 unique IDs, 10 total nodes (0 duplicates)

**Conclusion:** ID handling is correct. No issues with duplicate IDs.

## RAG Text Content Analysis

### Current State

**What GraphRAG Currently Extracts:**
1. **Generation entities**: `prompt`, `response`, `reasoning` fields
   - **Current count: 0** (no Generation entities exist!)
   
2. **Trace entities**: `input`, `output` fields
   - **Current count: 543 traces with input, 552 with output**
   - This is where the actual LLM content is currently stored
   
3. **Error entities**: `message` field
   - Not checked (likely 0)

### What Text Goes Into RAG Chunks

Based on `extract_text_from_entities()` in `graphrag_indexer.py`:

**Current extraction (from Trace entities):**
- **Input text**: Full prompt text from `trace.input` field
  - Example: `"System: Aggressive player. Prioritizes immediate gains...\n\nYou are playing a Common Pool Resource game..."`
  - Length: ~977 characters per input
  - **Issue**: Text may have surrounding quotes that need stripping
  
- **Output text**: LLM response from `trace.output` field
  - Example: `"I will extract the maximum amount possible to maximize my payoff in this round."`
  - Length: ~79 characters per output
  - **Issue**: Text may have surrounding quotes that need stripping

**What SHOULD be extracted (after fixes):**
1. **From Span entities** (when created):
   - `input` → prompt text (with prefix "Span Input: ")
   - `output` → response text (with prefix "Span Output: ")
   - `reasoning` from metadata → reasoning text (with prefix "Reasoning: ")

2. **From Generation entities** (when created):
   - `prompt` → prompt text (with prefix "Prompt: ")
   - `response` → response text (with prefix "Response: ")
   - `reasoning` → reasoning text (with prefix "Reasoning: ")

3. **From Trace entities** (fallback):
   - `input` → prompt text (with prefix "Trace Input: ")
   - `output` → response text (with prefix "Trace Output: ")

### Text Processing Pipeline

1. **Extraction**: Gets raw text from entity fields
2. **Deduplication**: Uses MD5 hash of raw text to avoid duplicates
3. **Chunking**: Splits long texts at semantic boundaries (paragraphs/sentences)
4. **Prefixing**: Adds context prefix (e.g., "Prompt: ", "Response: ")
5. **Entity Extraction**: LLM extracts semantic entities from chunks
6. **Embedding**: Generates embeddings for semantic entities
7. **Clustering**: Creates communities from entity embeddings

### Issues Identified

1. **Quoted Strings**: Input/output fields may have surrounding quotes that should be stripped
   - CSV shows: `"System: Aggressive player..."`
   - Need to check if quotes are preserved in database or stripped

2. **No Generation Entities**: Currently 0 Generation entities, so GraphRAG only processes Trace content
   - After fix: Should have ~543 Generation entities from Spans

3. **No Span Entities**: Currently 0 Span entities
   - After fix: Should have ~543 Span entities
   - GraphRAG should prioritize Span content over Trace content

4. **Missing Reasoning**: Reasoning text is in metadata but not extracted as separate field
   - Should extract `metadata.attributes.reasoning` for GraphRAG

### What Text Will Be in RAG After Fixes

**Expected content:**
- **~543 prompts** (from Span/Generation input/prompt fields)
- **~543 responses** (from Span/Generation output/response fields)  
- **~543 reasoning texts** (from metadata.attributes.reasoning)
- **Game summary outputs** (JSON, but may extract text from it)
- **Round metrics outputs** (JSON, but may extract text from it)

**Total unique text chunks:** ~1,600-2,000 (after deduplication and chunking)

**Text prefixes for context:**
- "Prompt: " - LLM prompts
- "Response: " - LLM responses  
- "Reasoning: " - Player reasoning
- "Span Input: " - Span-level inputs (fallback)
- "Span Output: " - Span-level outputs (fallback)
- "Trace Input: " - Trace-level inputs (fallback)
- "Trace Output: " - Trace-level outputs (fallback)

