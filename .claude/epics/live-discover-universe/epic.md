---
name: live-discover-universe
status: completed
created: 2026-02-14T15:54:23Z
completed: 2026-02-14
progress: 100%
prd: .claude/prds/live-discover-universe.md
github: https://github.com/jobu711/option_alpha_v2/issues/107
---

# Epic: live-discover-universe

## Overview

Add a live discovery engine that fetches the CBOE optionable securities CSV, validates new candidates via yfinance, auto-adds them with an "Auto-Discovered" tag, deactivates stale tickers, and fixes the broken `last_scanned_at` column. The implementation leverages existing `universe_service` functions (`add_tickers`, `toggle_ticker`, `get_full_universe`) and the proven `_filter_via_yfinance` batch pattern to minimize new code.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CBOE fetch library | `httpx` (sync `httpx.get`) | Already a project dependency (used in AI clients). Simpler than `aiohttp` for a one-shot CSV download. |
| Discovery engine async vs sync | `async def run_discovery()` | Matches orchestrator pattern; runs in FastAPI `BackgroundTask`. yfinance calls are sync but wrapped. |
| Stale detection trigger | Inside `run_discovery()` | Single atomic operation: add new + prune stale in one run, recorded in one `discovery_runs` row. |
| Reuse vs. new validation code | Inline adaptation of `_filter_via_yfinance` pattern | Same MultiIndex parsing logic from `universe.py:151-211`, but returns `(passed, failed)` tuple instead of just `passed`. Avoids import coupling. |
| Concurrency guard | Module-level `_discovery_running` flag | Same pattern as `_scan_running` / `_debate_running` in `web/routes.py`. |
| `last_scanned_at` fix location | `orchestrator._phase_persist()` | Single batch UPDATE after `save_ticker_scores()` — minimal change, no new phase needed. |

## Technical Approach

### Backend: Core Discovery Engine (`data/discovery.py`)

New module with three public functions:

- **`async run_discovery(conn, settings, on_progress)`** — Full pipeline: fetch CBOE CSV → dedup against DB + failure cache → validate via yfinance → add new tickers → detect/deactivate stale → record run stats.
- **`should_run_discovery(conn, settings)`** — Check if last completed run is older than `universe_refresh_interval_days`.
- **`get_last_discovery_run(conn)`** — Return most recent `discovery_runs` row as dict.

Internal helpers:
- `_fetch_cboe_optionable(url)` — `httpx.get()` + `csv.DictReader` + symbol filtering
- `_validate_via_yfinance(candidates, settings)` — Adapted batch pattern returning `(passed, failed)`
- `_get_cached_failures(conn, ttl_hours)` — Query `discovery_failures` within TTL
- `_cache_failures(conn, symbols, reason)` — INSERT OR REPLACE into `discovery_failures`
- `_detect_stale_tickers(conn, threshold_days)` — Single SQL query + `toggle_ticker()` calls

### Backend: Config Changes (`config.py`)

Add 5 settings after line 51 (after `min_avg_volume`):

```python
# --- Universe discovery ---
cboe_optionable_url: str = "https://www.cboe.com/us/options/symboldir/equity_index_options/?download=csv"
universe_refresh_interval_days: int = 7
discovery_batch_size: int = 100
stale_ticker_threshold_days: int = 90
failure_cache_ttl_hours: int = 24
```

These align with keys already present in `config.json` (`universe_refresh_interval_days`, `failure_cache_ttl_hours`).

### Backend: Migration (`persistence/migrations/005_discovery.sql`)

Two new tables + seed "Auto-Discovered" tag:
- **`discovery_runs`** — `id, started_at, completed_at, cboe_symbols_fetched, new_tickers_added, stale_tickers_deactivated, status, error_message`
- **`discovery_failures`** — `symbol (PK), reason, failed_at` with index on `failed_at`
- INSERT OR IGNORE the "Auto-Discovered" preset tag into `universe_tags`

### Backend: Orchestrator Fix (`pipeline/orchestrator.py`)

In `_phase_persist()`, after line 409 (`save_ticker_scores`), add a batch UPDATE:

```python
scored_symbols = [ts.symbol for ts in ticker_scores]
if scored_symbols:
    placeholders = ",".join("?" for _ in scored_symbols)
    conn.execute(
        f"UPDATE universe_tickers SET last_scanned_at = datetime('now') "
        f"WHERE symbol IN ({placeholders})",
        scored_symbols,
    )
    conn.commit()
```

### Backend: REST Endpoints (`web/universe_routes.py`)

- **`POST /api/universe/refresh`** — Guard with `_discovery_running` flag. Run `run_discovery()` via `BackgroundTasks`. Return HTMX partial or JSON status.
- **`GET /api/universe/discovery-status`** — Return `{running, last_run}` from `get_last_discovery_run()`.

### Frontend: Universe Dashboard (`templates/universe/universe.html`)

Add "Refresh Universe" button before "+ Add Ticker" (line 36):

```html
<button class="btn btn-secondary"
        hx-post="/api/universe/refresh"
        hx-target="#discovery-status"
        hx-swap="innerHTML"
        hx-indicator="#refresh-spinner">
    Refresh Universe
</button>
<span id="refresh-spinner" class="htmx-indicator">Refreshing...</span>
<div id="discovery-status"></div>
```

### Testing (`tests/test_discovery.py`)

Mock `httpx.get` for CBOE CSV and `yf.download` for yfinance. Use in-memory SQLite with migrations applied. Key test cases:

- CBOE CSV parsing (valid symbols, filtered non-equity)
- yfinance validation (price/volume filtering, batch handling)
- Failure cache (within TTL skipped, expired re-validated)
- Stale detection (NULL scan dates, old scan dates, grace period)
- `should_run_discovery` (never run, overdue, recent)
- End-to-end `run_discovery` (dedup, tagging, run recording)
- `last_scanned_at` updated by orchestrator persist phase

## Implementation Strategy

### Development Order

Tasks are ordered to build foundational pieces first, with each task independently testable:

1. **Config + Migration** — Foundation that everything else depends on
2. **`last_scanned_at` fix** — Small, independent, immediately useful
3. **Core discovery engine** — Largest piece, no web dependency
4. **REST endpoints** — Thin layer over discovery engine
5. **UI button** — Thin layer over REST endpoint
6. **Tests** — Validates everything end-to-end

### Risk Mitigation

- **CBOE CSV format change**: `_fetch_cboe_optionable` catches parse errors and records them in `discovery_runs`. The URL is a setting, not hardcoded.
- **yfinance rate limiting**: Reuses existing batch size (100) and sequential batching. Discovery runs infrequently (weekly).
- **Empty universe**: `toggle_ticker()` SAVEPOINT safety already prevents this — no new code needed.

## Task Breakdown Preview

- [ ] Task 1: Add discovery settings to `config.py` and create `005_discovery.sql` migration
- [ ] Task 2: Fix `last_scanned_at` — add batch UPDATE in orchestrator `_phase_persist()`
- [ ] Task 3: Create `data/discovery.py` — CBOE fetch, yfinance validation, failure cache, stale detection, run recording
- [ ] Task 4: Add `POST /api/universe/refresh` and `GET /api/universe/discovery-status` endpoints to `universe_routes.py`
- [ ] Task 5: Add "Refresh Universe" button + status indicator to `universe.html`
- [ ] Task 6: Create `tests/test_discovery.py` — full test suite with mocked externals
- [ ] Task 7: Run full test suite (`pytest tests/`) and fix any regressions

## Dependencies

### External
- **CBOE optionable securities CSV** — public, no auth, stable format
- **yfinance** — existing dependency, used for validation
- **httpx** — existing dependency, used for CSV fetch

### Internal (existing code reused)
- `universe_service.add_tickers(conn, symbols, tags, source)` — insert discoveries
- `universe_service.toggle_ticker(conn, symbol, active)` — deactivate stale (SAVEPOINT safety)
- `universe_service.get_full_universe(conn)` — dedup existing tickers
- `persistence/database.py:initialize_db()` — auto-applies new migration
- `_filter_via_yfinance()` pattern in `data/universe.py:151-211` — adapted for discovery

## Success Criteria (Technical)

| Criteria | Verification |
|----------|-------------|
| `005_discovery.sql` applied on startup | `initialize_db()` runs it; `discovery_runs` and `discovery_failures` tables exist |
| `last_scanned_at` populated after scan | Run scan, query `SELECT last_scanned_at FROM universe_tickers WHERE last_scanned_at IS NOT NULL` — count matches scored tickers |
| Discovery adds new tickers | Run refresh, verify new rows in `universe_tickers` with `source='discovered'` and "Auto-Discovered" tag |
| Stale tickers deactivated | After discovery, verify old never-scanned tickers have `is_active=0` |
| Failure cache prevents re-validation | Run discovery twice, second run should skip previously failed symbols |
| No regressions | `pytest tests/` — all existing tests pass |
| New test coverage | `pytest tests/test_discovery.py -v` — 14+ tests pass |

## Estimated Effort

- **Tasks**: 7
- **New code**: ~250 lines (`discovery.py`) + ~50 lines (config/migration/endpoints/template) + ~200 lines (tests)
- **Modified code**: ~15 lines across `config.py`, `orchestrator.py`, `universe_routes.py`, `universe.html`
- **Critical path**: Task 1 (config/migration) → Task 3 (discovery engine) → Task 4 (endpoints) → Task 6 (tests)

## Tasks Created
- [ ] #108 - Add discovery settings and database migration (parallel: false)
- [ ] #109 - Fix last_scanned_at in orchestrator persist phase (parallel: true)
- [ ] #110 - Create core discovery engine module (parallel: false, depends: #108)
- [ ] #111 - Add discovery REST endpoints to universe routes (parallel: false, depends: #108, #110)
- [ ] #112 - Add Refresh Universe button to dashboard UI (parallel: false, depends: #111)
- [ ] #113 - Create discovery test suite (parallel: false, depends: #108, #109, #110)
- [ ] #114 - Run full test suite and fix regressions (parallel: false, depends: all)

Total tasks: 7
Parallel tasks: 1 (#109 can run alongside #108)
Sequential tasks: 6
Estimated total effort: 11.5 hours
