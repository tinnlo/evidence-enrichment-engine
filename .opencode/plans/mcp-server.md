# MCP Server for Evidence Enrichment Engine

## Goal
Expose enriched and synthesized datasets as an MCP server for agent calling using the official `mcp` Python SDK with `FastMCP`.

## Stage 1: Foundation
**Goal**: Add dependency, create server skeleton, wire up transports
**Status**: Complete

### Changes:
1. **`pyproject.toml`** — Add `mcp` optional dependency group:
   ```toml
   mcp = [
     "mcp>=1.0.0",
   ]
   ```
   Add script entrypoint:
   ```toml
   evidence-enrich-mcp = "evidence_enrichment.mcp_server:main"
   ```

2. **`evidence_enrichment/mcp_server.py`** (NEW) — FastMCP server skeleton:
   - Create `FastMCP("Evidence Enrichment Engine")` instance
   - Import existing models, coordinator, settings
   - Helper to resolve replay/output paths from settings
   - `main()` function with transport selection (stdio default, streamable-http flag)

### Success Criteria:
- Server starts with `python -m evidence_enrichment.mcp_server`
- MCP Inspector connects successfully

---

## Stage 2: Resources
**Goal**: Implement read-only data access endpoints
**Status**: Complete

### Resources to implement in `mcp_server.py`:
1. `evidence://bundles` — List all replay bundle filenames
2. `evidence://bundles/{name}` — Return a specific bundle's JSON content
3. `evidence://results/latest` — Return latest demo_result.json if it exists

### Success Criteria:
- MCP Inspector shows all 3 resources
- Bundle listing returns all 6 replay files
- Bundle detail returns valid JSON

---

## Stage 3: Tools
**Goal**: Implement action endpoints wrapping pipeline functionality
**Status**: Complete

### Tools to implement in `mcp_server.py`:

1. **`run_enrichment_pipeline`**
   - Params: `entity_name: str`, `entity_id: str`, `field: str = "hq_country"`, `mode: str = "replay"`, `replay_bundle: str | None = None`
   - Runs `EvidenceCoordinator.run()` with the given entity
   - Returns full `PipelineRunResult` (Pydantic model -> structured output)

2. **`get_synthesis_result`**
   - Params: `entity_name: str`, `entity_id: str`, `field: str = "hq_country"`, `mode: str = "replay"`
   - Runs pipeline, extracts only synthesis + decision info
   - Returns a custom `SynthesisSummary` Pydantic model

3. **`list_replay_scenarios`**
   - No params
   - Scans `examples/replay/` directory
   - Returns list of scenario names with descriptions derived from filenames

4. **`get_evidence_claims`**
   - Params: `entity_name: str`, `entity_id: str`, `field: str = "hq_country"`, `mode: str = "replay"`
   - Runs pipeline, extracts only the fact claims
   - Returns list of `FactClaim` objects

5. **`compare_scenarios`**
   - Params: `scenario_a: str`, `scenario_b: str`
   - Runs pipeline for both replay bundles
   - Returns side-by-side comparison dict

### Success Criteria:
- All 5 tools appear in MCP Inspector
- `run_enrichment_pipeline` returns valid PipelineRunResult JSON
- `get_synthesis_result` returns compact synthesis summary
- `compare_scenarios` returns comparison with both results

---

## Stage 4: Prompt + CLI
**Goal**: Add prompt template and CLI integration
**Status**: Complete

### Changes:
1. **Prompt in `mcp_server.py`**:
   - `analyze_entity` prompt — guides agent through evidence analysis workflow

2. **`evidence_enrichment/cli.py`** — Add `mcp` subcommand:
   ```python
   @app.command("mcp")
   def mcp_command(transport: str = "stdio"):
       from evidence_enrichment.mcp_server import mcp
       mcp.run(transport=transport)
   ```

### Success Criteria:
- `evidence-enrich mcp` launches MCP server on stdio
- `evidence-enrich mcp --transport streamable-http` launches HTTP server
- Prompt appears in MCP Inspector

---

## Stage 5: Tests
**Goal**: Unit tests for MCP tools and resources
**Status**: Complete

### New file: `tests/test_mcp_server.py`
- Test resource listing returns correct bundle count
- Test bundle detail resource returns valid JSON
- Test `list_replay_scenarios` tool
- Test `run_enrichment_pipeline` tool with replay mode
- Test `get_synthesis_result` tool returns expected fields
- Test `get_evidence_claims` tool returns claims list
- Test `compare_scenarios` tool

### Success Criteria:
- All tests pass with `pytest tests/test_mcp_server.py`

---

## Stage 6: Polish
**Goal**: Update public API and docs
**Status**: Complete

### Changes:
1. **`evidence_enrichment/__init__.py`** — Add MCP server to `__all__`
2. **`README.md`** — Add MCP section with:
   - What it is
   - How to install (`pip install .[mcp]`)
   - How to run (`evidence-enrich mcp`)
   - How to connect (Claude Desktop, OpenCode, MCP Inspector)
   - Available tools/resources list

### Success Criteria:
- README documents MCP feature
- Public API exports are clean

---

## Dependencies
- `mcp>=1.0.0` (new, optional group)
- No other new dependencies

## Files Modified
| File | Type |
|---|---|
| `evidence_enrichment/mcp_server.py` | NEW |
| `tests/test_mcp_server.py` | NEW |
| `pyproject.toml` | EDIT |
| `evidence_enrichment/cli.py` | EDIT |
| `evidence_enrichment/__init__.py` | EDIT |
| `README.md` | EDIT |
