---
name: live-discover-universe
description: Auto-discover optionable tickers from CBOE, validate via yfinance, deactivate stale symbols, and fix last_scanned_at tracking
status: backlog
created: 2026-02-14T15:41:18Z
---

# PRD: live-discover-universe

## Executive Summary

Replace the static seed-only ticker universe with a live discovery engine that fetches the authoritative CBOE optionable securities list, validates candidates via yfinance, auto-adds new tickers with an "Auto-Discovered" tag, detects and deactivates stale/delisted tickers, and makes the existing `last_scanned_at` column functional. This keeps the universe fresh without manual intervention.

## Problem Statement

The ticker universe is populated from hardcoded Python lists (~702 tickers) seeded once into SQLite on first run. Stale or delisted tickers are only removed via hand-written SQL migrations. The `last_scanned_at` column exists in `universe_tickers` but is **never updated** — no code writes to it. This causes three concrete problems:

1. **New optionable tickers are never discovered automatically** — the universe stagnates from day one.
2. **Delisted/dead tickers accumulate silently** — yfinance returns no data, the pipeline skips them quietly, wasting fetch budget and polluting the universe.
3. **The "last scanned" filter in the Universe UI is non-functional** — every ticker shows "Never" because the column is always NULL.

This is important now because the universe has already required multiple manual cleanup migrations (`003_cleanup_stale_tickers.sql`, `004_remove_cday_wrk.sql`), and the problem will only worsen as markets evolve.

## User Stories

### US-1: Automatic Universe Refresh
**As a** user managing my scan universe,
**I want** a "Refresh Universe" button on the Universe dashboard
**So that** I can discover new optionable tickers and prune stale ones without writing SQL.

**Acceptance Criteria:**
- A "Refresh Universe" button appears on the Universe dashboard toolbar
- Clicking it triggers a background discovery run
- A status indicator shows progress/completion
- Guard prevents concurrent discovery runs (409 if already running)
- New tickers appear in the table tagged "Auto-Discovered"
- Stale tickers (>90 days without a successful scan) are deactivated

### US-2: Discovery Audit Trail
**As a** user,
**I want** to see when discovery last ran and what it found
**So that** I can trust the universe is being maintained.

**Acceptance Criteria:**
- `GET /api/universe/discovery-status` returns last run timestamp, counts (new added, stale deactivated, CBOE symbols fetched), and running state
- `discovery_runs` table records every run with stats and status

### US-3: Functional Last-Scanned Tracking
**As a** user filtering tickers by scan recency,
**I want** `last_scanned_at` to actually update when tickers are scored
**So that** the "Last Scanned" column and staleness detection work correctly.

**Acceptance Criteria:**
- After a successful scan, `last_scanned_at` is updated for all scored tickers
- The Universe UI "Last Scanned" column shows real dates instead of "Never"
- Stale detection uses `last_scanned_at` (or `created_at` for never-scanned tickers) to identify inactive symbols

### US-4: Failure Caching
**As a** system operator,
**I want** symbols that fail yfinance validation to be cached
**So that** repeated discovery runs don't waste time re-validating thousands of non-equity symbols (index options, warrants, etc.).

**Acceptance Criteria:**
- Failed symbols are cached in `discovery_failures` with reason and timestamp
- Cached failures are skipped on subsequent runs within the TTL (default 24h)
- Expired failures are re-validated automatically

## Requirements

### Functional Requirements

#### FR-1: CBOE Optionable List Fetch
- Download CSV from `https://www.cboe.com/us/options/symboldir/equity_index_options/?download=csv`
- Parse with `csv.DictReader`, extract stock symbols
- Filter: `symbol.isalpha() and 1 <= len(symbol) <= 5` (skip index options `$SPX`, warrants `SPAK+`, units with `.`)

#### FR-2: Candidate Deduplication
- Subtract existing DB tickers (`get_full_universe(conn)`) from CBOE list
- Subtract cached failures (within TTL) from candidates
- Only validate genuinely new symbols

#### FR-3: yfinance Validation
- Batch `yf.download(period="5d")` in groups of `discovery_batch_size` (default 100)
- Reuse the MultiIndex parsing pattern from `data/universe.py:_filter_via_yfinance()`
- Filter by existing `min_price` and `min_avg_volume` settings
- Return validated tickers and cache failures

#### FR-4: Auto-Add Discoveries
- Call existing `universe_service.add_tickers(conn, symbols, tags=["auto-discovered"], source="discovered")`
- New tickers are active by default
- Tagged "Auto-Discovered" for easy filtering

#### FR-5: Stale Ticker Detection
- Query active tickers where:
  - `last_scanned_at IS NULL AND created_at < threshold` (old, never scanned), OR
  - `last_scanned_at < threshold` (not scanned recently)
- Default threshold: 90 days (`stale_ticker_threshold_days`)
- Deactivate via `toggle_ticker(conn, symbol, active=False)` (leverages existing SAVEPOINT safety to prevent emptying universe)

#### FR-6: Last-Scanned-At Update
- In orchestrator `_phase_persist()`, after `save_ticker_scores()`, batch-UPDATE `last_scanned_at = datetime('now')` for all scored symbols
- Single UPDATE with `IN (...)` clause (no N+1)

#### FR-7: REST Endpoints
- `POST /api/universe/refresh` — trigger discovery (409 if already running)
- `GET /api/universe/discovery-status` — return `{running: bool, last_run: dict|null}`

#### FR-8: Interval-Based Readiness Check
- `should_run_discovery(conn)` returns `True` if no completed runs or last run older than `universe_refresh_interval_days` (default 7)
- Used by UI to indicate when refresh is recommended (not for auto-trigger)

### Non-Functional Requirements

#### NFR-1: Performance
- CBOE CSV fetch: single HTTP GET, <5s expected
- yfinance validation: batched to avoid rate-limiting, reuses existing batch size patterns
- Stale detection: single SQL query, no iteration
- `last_scanned_at` update: single batch UPDATE per scan

#### NFR-2: Reliability
- Discovery failures don't crash the app — errors are caught and recorded in `discovery_runs`
- Failure cache prevents repeated wasted API calls
- Existing SAVEPOINT safety in `toggle_ticker` prevents emptying the universe during stale pruning

#### NFR-3: Security
- Binds to 127.0.0.1 only (existing constraint)
- CBOE URL is a configurable setting (no hardcoded external URLs in business logic)
- No credentials required for CBOE CSV (public endpoint)

#### NFR-4: Testability
- All external calls (httpx, yfinance) are mockable
- In-memory SQLite for test isolation
- Core engine is a pure function (`run_discovery(conn, settings)`) with no global state

## Success Criteria

| Metric | Target |
|--------|--------|
| New tickers discovered on first run | >0 (validates CBOE integration works) |
| `last_scanned_at` populated after scan | 100% of scored tickers |
| Stale tickers auto-deactivated | All tickers >90 days without scan |
| Failure cache hit rate on 2nd run | >80% (most CBOE non-equity symbols cached) |
| Existing test suite | All pass (no regressions) |
| New test coverage | 14+ test cases covering all discovery paths |

## Constraints & Assumptions

### Constraints
- **CBOE CSV availability**: If CBOE changes their URL or format, the fetch step will fail gracefully and record the error
- **yfinance rate limits**: Batch size of 100 and existing fetch concurrency settings mitigate this
- **SQLite single-writer**: Discovery must not run concurrently with scan pipeline (existing `_scan_running` pattern applies)

### Assumptions
- CBOE CSV format remains stable (columns: Company Name, Stock Symbol, DPM, Post/Station)
- `symbol.isalpha() and len <= 5` is sufficient to filter equity tickers from index/warrant/unit symbols
- yfinance 5-day download is sufficient to validate a ticker is active and liquid
- 90-day stale threshold is conservative enough to avoid false positives from market holidays or temporary data gaps

## Out of Scope

- **Scheduled/automatic discovery** — this PRD covers on-demand only (user clicks button). Scheduled runs (e.g., weekly cron) can be added later using `should_run_discovery()` as the readiness check
- **Discovery source plugins** — only CBOE CSV; no Finviz, social sentiment, or unusual activity feeds
- **WebSocket progress streaming** — discovery status is polled, not pushed (unlike the scan pipeline)
- **UI for managing failure cache** — failures auto-expire via TTL; no manual cache management UI
- **Auto-triggering discovery before scan** — discovery is independent of the scan pipeline

## Dependencies

### External
- **CBOE optionable securities CSV** — public endpoint, no auth required
- **yfinance** — already a project dependency, used for validation
- **httpx** — already a project dependency (used in AI clients), used for CBOE fetch

### Internal
- `universe_service.add_tickers()` — existing function for inserting tickers with tags
- `universe_service.toggle_ticker()` — existing function with SAVEPOINT safety
- `universe_service.get_full_universe()` — existing function for dedup
- `persistence/database.py` — migration runner (auto-applies `005_discovery.sql`)
- `pipeline/orchestrator.py` — `_phase_persist()` for `last_scanned_at` fix

## Changes Summary

| # | File | Action | Description |
|---|------|--------|-------------|
| 1 | `src/option_alpha/config.py` | MODIFY | Add 5 discovery settings after "Data pre-filters" section |
| 2 | `src/option_alpha/persistence/migrations/005_discovery.sql` | CREATE | `discovery_runs` + `discovery_failures` tables, "Auto-Discovered" tag |
| 3 | `src/option_alpha/data/discovery.py` | CREATE | Core discovery engine (~200 lines) |
| 4 | `src/option_alpha/pipeline/orchestrator.py` | MODIFY | Update `last_scanned_at` after successful scoring |
| 5 | `src/option_alpha/web/universe_routes.py` | MODIFY | Add `POST /api/universe/refresh` + status endpoint |
| 6 | `src/option_alpha/web/templates/universe/universe.html` | MODIFY | Add "Refresh Universe" button + status indicator |
| 7 | `tests/test_discovery.py` | CREATE | 14+ test cases with mocked externals |
