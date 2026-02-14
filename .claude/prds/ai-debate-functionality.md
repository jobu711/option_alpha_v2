---
name: ai-debate-functionality
description: Move AI debates from automatic batch pipeline to user-initiated on-demand per-ticker via dashboard multi-select
status: backlog
created: 2026-02-14T00:03:29Z
---

# PRD: ai-debate-functionality

## Executive Summary

The AI debate system currently runs automatically as pipeline phase 5, debating the top 10 candidates after every scan. This is slow, resource-intensive, and wasteful — the user often only cares about 1-3 tickers with clear directional signals. This PRD moves AI debates from an automatic batch phase to a user-initiated on-demand action triggered via checkbox multi-select on the dashboard. The scan pipeline drops from 6 phases to 5, and debates are launched independently when the user is ready.

## Problem Statement

### Current Pain Points

1. **Resource waste**: Every scan automatically debates 10 tickers, even when the user only cares about 2-3. Each debate requires 3 sequential LLM calls (Bull → Bear → Risk), so 10 tickers = 30 LLM calls.

2. **Scan time bloat**: The AI debate phase (phase 5) is the slowest phase in the pipeline. It adds significant wall-clock time to every scan, delaying the user's ability to see scores and options recommendations.

3. **No user control**: The user has no say in which tickers get debated. The system auto-selects the top 10 by composite score, but the user may want to debate a ticker ranked #15 that has a compelling catalyst, or skip a top-5 ticker that's clearly neutral.

4. **Wasted on neutral tickers**: Many top-scored tickers have a NEUTRAL direction — debating them is unproductive since there's no clear thesis to evaluate.

### Why Now

The AI debate system was recently rewritten to use official SDKs (epic `aidebate-rewrite`) and is now stable and reliable. With the underlying reliability solved, the UX problem is now the priority: let users choose when and what to debate.

## User Stories

### US-1: Select Tickers for Debate

**As a** user viewing scan results on the dashboard,
**I want** to select specific tickers via checkboxes and click "Debate Selected",
**so that** I only spend LLM resources on tickers I actually want to analyze.

**Acceptance Criteria:**
- Each row in the candidates table has a checkbox column
- A "Debate Selected" button appears when 1+ checkboxes are checked
- The button is disabled when no checkboxes are checked
- Selected count is displayed (e.g., "Debate 3 Selected")

### US-2: On-Demand Debate Execution

**As a** user who clicked "Debate Selected",
**I want** debates to run for my selected tickers and results to appear inline on the dashboard,
**so that** I can see the analysis without navigating away from my workflow.

**Acceptance Criteria:**
- Clicking "Debate Selected" triggers an HTMX POST with the selected ticker symbols
- A loading indicator appears for each selected ticker while its debate runs
- Debate results render inline (HTMX partial swap) as each ticker completes
- The user can continue viewing the dashboard while debates run (non-blocking UI)
- Each debate always runs fresh (no caching of prior results)

### US-3: Scan Without Debate Phase

**As a** user running a scan,
**I want** the scan to complete faster by skipping the automatic AI debate phase,
**so that** I see scores and options data quickly and decide which tickers to debate myself.

**Acceptance Criteria:**
- The scan pipeline has 5 phases: Data Fetch → Scoring → Catalysts → Options → Persist
- The `ai_debate` phase is completely removed from the pipeline
- Scan progress UI reflects 5 phases (no "ai_debate" pending phase)
- Scan completion time is reduced by the former debate phase duration
- Existing debate results in the database from prior scans are still displayed on ticker detail pages

### US-4: Debate Results Display

**As a** user who triggered a debate,
**I want** to see bull/bear/risk analysis and the final thesis inline on the dashboard,
**so that** I can make an informed trading decision.

**Acceptance Criteria:**
- Debate results show: bull thesis, bear thesis, risk synthesis, conviction (1-10), direction, and recommended action
- Results are persisted to the `ai_theses` table (linked to the current scan run)
- Ticker detail page (`/ticker/{symbol}`) also shows the debate result if one exists
- If a debate fails, the fallback result is shown with a clear indicator that it's a fallback

### US-5: Debate While Scan is Idle

**As a** user,
**I want** to trigger debates only when no scan is currently running,
**so that** debate LLM calls don't compete with any future pipeline LLM usage.

**Acceptance Criteria:**
- "Debate Selected" button is disabled with a tooltip when a scan is running
- If a scan starts while debates are in progress, debates continue to completion (no cancellation)
- Clear visual distinction between scan-in-progress and debate-in-progress states

## Requirements

### Functional Requirements

#### FR-1: Remove AI Debate from Scan Pipeline

**Changes to `pipeline/orchestrator.py`:**
- Remove `_phase_ai_debate()` method from `ScanOrchestrator`
- Remove `ai_debate` from `PHASE_NAMES` list (becomes 5 phases)
- Remove `debate_results` from `run_scan()` flow
- Keep `debate_results` parameter in `_phase_persist()` but pass empty list
- Update `ScanResult` to have `top_n_debated=0` by default
- Remove imports of `get_client` and `DebateManager` from orchestrator

**Changes to `web/routes.py`:**
- Remove `ai_debate` from the phases list in `trigger_scan()`
- Progress UI shows 5 phases instead of 6

#### FR-2: New Debate API Endpoint

**New POST endpoint: `/debate`**
- Accepts JSON body: `{"symbols": ["AAPL", "TSLA", "NVDA"]}`
- Validates that symbols exist in the latest scan's ticker scores
- Instantiates `LLMClient` via `get_client()` and `DebateManager`
- Runs debates for requested symbols only
- Persists results to `ai_theses` table (linked to latest scan run)
- Returns HTMX partial with debate results for inline rendering

**Guards:**
- Returns 409 if a scan is currently running
- Returns 400 if no symbols provided or symbols not in latest scan
- Returns 503 if LLM backend health check fails

#### FR-3: Dashboard Multi-Select UI

**Changes to `_candidates_table.html`:**
- Add checkbox column as the first column in the table
- Each checkbox carries the ticker symbol as its value
- Add "Select All" / "Deselect All" toggle in the header

**New UI elements on `dashboard.html`:**
- "Debate Selected" button (above or below the candidates table)
- Selected count indicator: "N selected"
- Button disabled state when count is 0 or scan is running
- Loading state while debates are in progress

**HTMX wiring:**
- Button triggers `hx-post="/debate"` with selected symbols collected via JavaScript
- Response targets a debate results container on the page
- `hx-indicator` shows spinner during the request

#### FR-4: Debate Results Partial Template

**New template: `_debate_results.html`**
- Renders debate results for 1+ tickers
- Each ticker shows: bull thesis, bear thesis, risk synthesis, conviction badge, direction badge, recommended action
- Compact card layout that fits within the dashboard flow
- Fallback results are visually distinct (muted styling, "[FALLBACK]" label)

#### FR-5: Persist On-Demand Debate Results

- On-demand debate results are saved to the same `ai_theses` table
- Linked to the current (latest) scan run via `scan_run_id`
- If a debate already exists for that ticker+scan, it is replaced (always fresh)
- Uses existing `save_ai_theses()` from `persistence/repository.py`

### Non-Functional Requirements

#### NFR-1: Performance
- Scans complete faster (no debate phase overhead)
- Individual on-demand debates complete in the same time as before (per-ticker LLM latency unchanged)
- Dashboard remains responsive while debates run (async background processing)

#### NFR-2: Backward Compatibility
- Existing debate results in the database are still displayed
- Ticker detail page works the same (shows debate if one exists, regardless of how it was triggered)
- Export report includes debate data if available
- No database schema changes needed

#### NFR-3: Simplicity
- Minimal JavaScript — rely on HTMX for interactivity
- Small amount of vanilla JS for checkbox state management and symbol collection
- No new WebSocket channels — use standard HTMX request/response

#### NFR-4: Error Handling
- Individual ticker debate failures don't prevent other selected tickers from completing
- Clear error messages for: LLM backend unavailable, ticker not in scan, scan running
- Fallback results rendered with visual distinction

## Success Criteria

1. **Scan speed**: Scan pipeline completes without the former debate phase delay
2. **User control**: Users can select and debate any subset of scan candidates
3. **Inline results**: Debate results appear on the dashboard via HTMX without page reload
4. **Persistence**: On-demand debate results are saved and visible on ticker detail pages
5. **Reliability**: On-demand debates have the same >90% success rate as the former automatic debates

## Constraints & Assumptions

### Constraints
- Must use existing HTMX + Jinja2 patterns (no new JS frameworks)
- Must preserve the `ai/` module architecture (CLAUDE.md warning)
- Must work with both Ollama and Claude backends
- `DebateManager` API unchanged — only the trigger mechanism changes
- No database schema migration (reuse existing `ai_theses` table)

### Assumptions
- The `DebateManager.run_debates()` method works correctly (proven by `aidebate-rewrite` epic)
- Users will typically select 1-5 tickers for debate (not all candidates)
- The LLM backend is configured and healthy before the user triggers debates
- HTMX can handle the response time of sequential LLM debates (may need timeout tuning on the HTMX request)

## Out of Scope

- Real-time per-agent progress (WebSocket streaming of Bull → Bear → Risk steps)
- Debate result caching or staleness detection
- Automatic pre-filtering of neutral tickers (user makes the choice)
- Changes to the debate agent prompts or flow (Bull → Bear → Risk stays as-is)
- Changes to `ai/clients.py`, `ai/agents.py`, `ai/context.py`, or `ai/debate.py`
- New LLM backend support
- Debate comparison or historical tracking across scans
- Batch debate queue or scheduling

## Dependencies

### Internal
- `src/option_alpha/pipeline/orchestrator.py` — remove phase 5
- `src/option_alpha/pipeline/progress.py` — update phase list if hardcoded
- `src/option_alpha/web/routes.py` — new `/debate` endpoint, update `/scan` phases
- `src/option_alpha/web/templates/_candidates_table.html` — add checkboxes
- `src/option_alpha/web/templates/dashboard.html` — add debate button + results container
- `src/option_alpha/web/templates/_debate_results.html` — new partial (debate results display)
- `src/option_alpha/web/templates/_progress.html` — update to 5 phases
- `src/option_alpha/persistence/repository.py` — may need upsert logic for replacing existing debate results
- `tests/test_integration.py` — update pipeline tests (no debate phase)
- `tests/test_web.py` or new test file — test new `/debate` endpoint

### External
- No new external dependencies
- Existing `ollama` and `anthropic` SDK dependencies unchanged
- LLM backend must be running for debates to work (same as before)
