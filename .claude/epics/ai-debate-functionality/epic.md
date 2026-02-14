---
name: ai-debate-functionality
status: backlog
created: 2026-02-14T00:10:21Z
progress: 0%
prd: .claude/prds/ai-debate-functionality.md
github: https://github.com/jobu711/option_alpha_v2/issues/90
---

# Epic: ai-debate-functionality

## Overview
Move AI debates from automatic pipeline phase 5 to user-initiated on-demand via dashboard multi-select. The scan pipeline drops from 6 phases to 5 (removing `ai_debate`). Users select tickers via checkboxes and click "Debate Selected" to trigger debates via a new HTMX POST endpoint. The `ai/` module architecture is unchanged — only the trigger mechanism moves from orchestrator to web route.

## Architecture Decisions
- **No changes to `ai/` module**: `DebateManager`, agents, clients, context all stay as-is per CLAUDE.md constraint
- **Standard HTMX request/response**: No WebSocket for debate progress — single POST returns all results when complete (PRD NFR-3)
- **No new database schema**: Reuse existing `ai_theses` table with DELETE-then-INSERT for "always fresh" semantics (FR-5)
- **Minimal JavaScript**: Only vanilla JS needed for checkbox state management and collecting selected symbols into the HTMX request body
- **`_scan_running` guard reused**: The existing module-level `_scan_running` flag in `routes.py` guards the debate endpoint (409 if scan active)

## Technical Approach

### Backend Changes
1. **`pipeline/orchestrator.py`**: Remove `_phase_ai_debate()` call from `run_scan()`, remove `"ai_debate"` from `PHASE_NAMES`, remove `get_client`/`DebateManager` imports. Keep `debate_results` parameter in `_phase_persist()` but always pass empty list. Update phase indices for persist (4 instead of 5).
2. **`web/routes.py`**: Remove `"ai_debate"` from the phases list in `trigger_scan()`. Add new `POST /debate` endpoint that accepts `{"symbols": [...]}`, validates symbols against latest scan, runs `DebateManager.run_debate()` per ticker, persists results, and returns `_debate_results.html` partial.

### Frontend Changes
3. **`_candidates_table.html`**: Add checkbox column (first column) with `name="debate_symbol"` carrying ticker symbol as value. Add "Select All" toggle in header.
4. **`dashboard.html`**: Add "Debate Selected" button with count indicator, wired via `hx-post="/debate"`. Add `#debate-results` container div for HTMX response target. Small inline `<script>` for collecting checked symbols into request body and updating button state/count.
5. **New `_debate_results.html`**: Partial template rendering debate cards — each shows bull/bear/risk analysis, conviction badge, direction badge, recommended action. Fallback results visually distinct with muted styling.

### Persistence
- Before inserting new debate results, DELETE existing `ai_theses` rows for the same `(scan_run_id, ticker)` combo to ensure "always fresh" semantics without schema changes.

## Implementation Strategy
- **Phase 1** (#91, #93): Backend — remove debate from pipeline, add `/debate` endpoint
- **Phase 2** (#95, #92, #94): Frontend — checkbox UI, debate button, results partial
- **Phase 3** (#96): Tests — update existing pipeline tests, add debate endpoint tests

## Task Breakdown Preview
- [ ] #91: Remove AI debate phase from scan pipeline orchestrator
- [ ] #93: Add POST `/debate` endpoint to web routes
- [ ] #95: Add checkbox multi-select to candidates table template
- [ ] #92: Add debate button, results container, and JS wiring to dashboard
- [ ] #94: Create `_debate_results.html` partial template
- [ ] #96: Update tests for 5-phase pipeline and new debate endpoint

## Dependencies
- **Internal (files touched)**:
  - `src/option_alpha/pipeline/orchestrator.py` — remove phase 5, update PHASE_NAMES and phase indices
  - `src/option_alpha/web/routes.py` — update `/scan` phases, add `/debate` endpoint
  - `src/option_alpha/web/templates/_candidates_table.html` — add checkbox column
  - `src/option_alpha/web/templates/dashboard.html` — add debate button + results area
  - `src/option_alpha/web/templates/_debate_results.html` — new partial (create)
  - `tests/test_integration.py` — update pipeline tests, add debate tests
- **External**: None (reuses existing `ai/` module, `persistence/repository.py`, Ollama/Claude backends)

## Success Criteria (Technical)
- `pytest tests/` passes with all existing tests updated for 5-phase pipeline
- `POST /debate` returns 409 when scan is running, 400 for invalid symbols
- Debate results render inline via HTMX without page reload
- Debate results persisted to `ai_theses` table and visible on `/ticker/{symbol}` detail page
- Scan completes without former debate phase delay (5 phases in progress UI)

## Tasks Created
- [ ] #91 - Remove AI debate phase from scan pipeline orchestrator (parallel: true)
- [ ] #93 - Add POST /debate endpoint to web routes (parallel: false, depends: #91)
- [ ] #95 - Add checkbox multi-select to candidates table template (parallel: true)
- [ ] #92 - Add debate button, results container, and JS wiring to dashboard (parallel: false, depends: #93, #95)
- [ ] #94 - Create _debate_results.html partial template (parallel: true)
- [ ] #96 - Update tests for 5-phase pipeline and new debate endpoint (parallel: false, depends: #91, #93, #94)

Total tasks: 6
Parallel tasks: 3 (#91, #95, #94)
Sequential tasks: 3 (#93, #92, #96)
Estimated total effort: 9-14 hours

## Estimated Effort
- **6 issues**, small-to-medium scope each
- Issues 1-2 (backend) can be done first as foundation
- Issues 3-5 (frontend) depend on issue 2 for the endpoint
- Issue 6 (tests) should be last to validate everything
- Critical path: Issue 1 → Issue 2 → Issues 3-5 (parallel) → Issue 6
