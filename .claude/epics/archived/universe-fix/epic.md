---
name: universe-fix
status: completed
created: 2026-02-12T13:01:57Z
updated: 2026-02-12T13:40:00Z
completed: 2026-02-12T13:40:00Z
progress: 100%
prd: .claude/prds/universe-fix.md
github: https://github.com/jobu711/option_alpha_v2/issues/55
---

# Epic: universe-fix

## Overview

Rebuild the ticker universe system to enforce open interest validation, remove stale/delisted tickers, and add scheduled weekly refresh. The core change is adding an OI gate (100+ total) to `universe_refresh.py` and wiring APScheduler into the FastAPI lifespan for automated maintenance. Existing `universe.py` consumers and `universe_data.json` schema remain unchanged.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Modify `universe_refresh.py` in-place | Already has the SEC EDGAR + yfinance pipeline; add OI check as a new validation step rather than rewriting |
| Atomic write with backup | Write to `universe_data.json.tmp` then rename; preserve `.bak` so a failed refresh never corrupts the live file |
| APScheduler (BackgroundScheduler) | Lightweight, no external process needed, integrates cleanly with FastAPI lifespan; already used pattern in similar Python projects |
| ETFs exempt from OI gate | ETFs are curated and working; validate optionability only, skip OI threshold |
| Configurable `min_universe_oi` setting | Add to `Settings` class alongside existing `min_open_interest` (which is for options filtering, not universe filtering) |
| Rate-limited batch validation | yfinance per-ticker OI checks are slow; use `time.sleep` pacing at ~1800 req/hr to avoid throttling |

## Technical Approach

### Modified Modules

**`config.py`** — Add new settings:
- `min_universe_oi: int = 100` — OI threshold for universe inclusion
- `universe_refresh_schedule: str = "sat"` — Day of week for scheduled refresh

**`universe_refresh.py`** — Core changes:
- Add `_validate_open_interest(tickers) -> list[str]` step between optionability check and metadata enrichment
- For each ticker: fetch options chain via `yf.Ticker(symbol).options`, sum OI across nearest expiration, compare to threshold
- Separate stock vs ETF paths: stocks get OI gate, ETFs get optionability-only validation
- Atomic write: write to `.tmp`, backup current to `.bak`, rename `.tmp` to live
- Add universe size warnings (below 500 or above 1500 stocks)
- Log per-ticker pass/fail with reason
- Add rate limiting between yfinance calls

**`web/app.py`** — Lifespan changes:
- Import and start APScheduler `BackgroundScheduler` in lifespan
- Schedule `refresh_universe()` weekly on configured day
- Shut down scheduler on app teardown

**`web/routes.py`** — Enhance existing endpoint:
- `POST /api/universe/refresh` already exists; add `?regenerate=true` query param to trigger full SEC EDGAR rebuild vs. validation-only refresh

### No Changes Required

- `universe.py` — Reads `universe_data.json` unchanged; no schema changes
- `universe_data.json` schema — Same fields: `symbol`, `name`, `sector`, `market_cap_tier`, `asset_type`
- Pipeline orchestrator — Consumes universe via `get_scan_universe()` as before
- Watchlist system — Unaffected

## Implementation Strategy

Work bottom-up: settings first, then core OI validation logic, then integrate into refresh pipeline, then scheduled execution, then tests.

## Task Breakdown

- [ ] **Task 1: Add universe OI settings to config** — Add `min_universe_oi` (default 100) and `universe_refresh_schedule` (default "sat") to `Settings` class in `config.py`
- [ ] **Task 2: Add OI validation to universe refresh** — Implement `_validate_open_interest()` in `universe_refresh.py` that checks total OI for each stock ticker (ETFs exempt); add rate limiting; integrate into `_do_refresh()` pipeline between optionability and enrichment steps
- [ ] **Task 3: Implement atomic writes and ETF preservation** — Backup current `universe_data.json` before overwrite; write to `.tmp` then rename; separate ETF entries before regeneration and merge back after; add universe size warnings to logs and `universe_meta.json`
- [ ] **Task 4: Add full regeneration mode** — Add `regenerate: bool` param to `refresh_universe()` (default False = validate existing tickers only; True = full SEC EDGAR rebuild); wire to existing `POST /api/universe/refresh?regenerate=true` endpoint
- [ ] **Task 5: Add APScheduler for weekly refresh** — Add `apscheduler` dependency; start `BackgroundScheduler` in FastAPI lifespan; schedule weekly refresh on configured day; clean shutdown on app teardown
- [ ] **Task 6: Tests for OI validation, atomic writes, and scheduled refresh** — Unit tests for `_validate_open_interest()`, atomic write/backup behavior, ETF preservation, size warnings, settings; integration test for full refresh pipeline with mocked yfinance

## Dependencies

- **External**: SEC EDGAR API (existing), yfinance (existing), APScheduler (new — `pip install apscheduler`)
- **Internal**: `config.py` (Settings), `universe_refresh.py`, `universe.py`, `web/app.py`, `web/routes.py`
- **Task ordering**: Task 1 before Tasks 2-5; Task 2 before Task 3; Tasks 2-5 before Task 6

## Success Criteria (Technical)

- All stock tickers in regenerated `universe_data.json` have 100+ total OI (verified by test)
- ETF entries preserved unchanged through regeneration
- Failed refresh leaves previous `universe_data.json` intact (`.bak` exists)
- `universe_meta.json` updated with ticker count, timestamp, and size warnings
- APScheduler runs weekly without manual intervention
- Existing tests continue to pass (no downstream breakage)
- New tests cover OI validation, atomic writes, ETF preservation, scheduler setup

## Estimated Effort

- **6 tasks**, incremental and testable
- Each task is a focused change to 1-2 files
- Critical path: Task 1 → Task 2 → Task 3 → Task 6
- Tasks 4 and 5 can be parallelized after Task 2

## Tasks Created
- [ ] #57 - Add universe OI settings to config (parallel: false)
- [ ] #60 - Add OI validation to universe refresh (parallel: false)
- [ ] #61 - Implement atomic writes and ETF preservation (parallel: false)
- [ ] #56 - Add full regeneration mode (parallel: true)
- [ ] #58 - Add APScheduler for weekly refresh (parallel: true)
- [ ] #59 - Tests for universe refresh overhaul (parallel: false)

Total tasks: 6
Parallel tasks: 2 (#56, #58 — can run concurrently after #60)
Sequential tasks: 4 (#57 → #60 → #61, then #59 after all)
Estimated total effort: 19 hours
