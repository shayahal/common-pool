# FalkorDB Troubleshooting Guide

## Issue: No Data in FalkorDB

### Root Cause
The FalkorDB exporter was creating async tasks in the wrong event loop, causing exports to fail silently. This has been fixed.

### Solution Applied
1. **Fixed Event Loop Binding**: The exporter now uses a dedicated background event loop thread to handle async exports, ensuring Graphiti/FalkorDB connections are bound to the correct event loop.

2. **Improved Error Handling**: Added proper error handling and logging to track export progress.

3. **Better Logging**: Export progress is now logged at INFO level so you can see when spans are being exported.

### Verification

#### Check if Exporter is Enabled
```bash
.venv\Scripts\Activate.ps1
python check_falkordb_exporter.py
```

#### Test Export Functionality
```bash
.venv\Scripts\Activate.ps1
python test_falkordb_export.py
```

This will:
- Create test spans
- Export them to FalkorDB
- Verify they're stored (note: Graphiti search may take time to index)

### Important Notes

1. **OpenAI API Key Required**: Graphiti requires `OPENAI_API_KEY` to extract entities from traces. Make sure it's set in your environment.

2. **Export Takes Time**: Each episode export takes 20-30 seconds because Graphiti calls OpenAI to extract entities and relationships. This is normal.

3. **Search Indexing Delay**: After episodes are added, Graphiti's search may take additional time to index them. Episodes are stored immediately, but search results may lag.

4. **Check Logs**: Look for these messages in your logs:
   - `"Exporting X spans to FalkorDB"` - Export started
   - `"Adding episode X/Y"` - Episode being added
   - `"Successfully added episode"` - Episode stored
   - `"Exported X spans to FalkorDB"` - Export completed

### Querying FalkorDB Directly

If Graphiti search isn't finding episodes, you can query FalkorDB directly using Cypher:

```python
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver

falkor_driver = FalkorDriver(host='localhost', port='6379')
graphiti = Graphiti(graph_driver=falkor_driver)

# Query all episodes
results = await graphiti.search("episode")
```

Or use FalkorDB's web UI at http://localhost:3000 to run Cypher queries directly.

### Common Issues

1. **"No traces found"**: 
   - Check that `FALKORDB_ENABLED=true` in your config
   - Verify FalkorDB is running: `docker ps | grep falkordb`
   - Check logs for export messages
   - Wait longer for indexing (30+ seconds after export)

2. **"Event loop binding error"**: 
   - This should be fixed now, but if you see it, restart your application

3. **"OPENAI_API_KEY required"**: 
   - Set the environment variable: `export OPENAI_API_KEY=sk-...`
   - Or add to `.env` file

### Next Steps

1. Run a real game/experiment and check the logs for export messages
2. Wait 30-60 seconds after the game completes for exports to finish
3. Query FalkorDB to verify traces are stored
4. If still not working, check the error logs for specific error messages

