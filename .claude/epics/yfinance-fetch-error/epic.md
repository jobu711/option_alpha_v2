---
name: yfinance-fetch-error
status: backlog
created: 2026-02-11T13:35:05Z
progress: 0%
prd: .claude/prds/yfinance-fetch-error.md
github: https://github.com/jobu711/option_alpha_v2/issues/12
---

# Epic: yfinance-fetch-error

## Overview

Reduce yfinance fetch errors from hundreds per scan to near-zero by: cleaning ~20 delisted tickers from universe lists, adding a JSON-based failure cache with configurable TTL, throttling concurrent requests to avoid Yahoo 401 rate limits, classifying errors by type (delisted/rate-limited/network), and exposing retry/batch parameters on the settings page.

## Architecture Decisions

- **Failure cache format: JSON file** — Stored at `data/cache/_failures.json`. Aligns with existing cache patterns in `cache.py` (parquet + JSON). Keeps `data/` layer independent of `persistence/` (SQLite). Simple key-value store for ~20-50 entries doesn't warrant a database table.

- **Error classification: string enum in models.py** — `FetchErrorType(str, Enum)` with values DELISTED, RATE_LIMITED, NETWORK, INSUFFICIENT_DATA, UNKNOWN. Stored in failure cache entries to enable differentiated logging. Using a simple flat TTL for cache expiry (not per-type TTL) to keep implementation simple — the error type is informational for logs/dashboard.

- **Throttling: fixed inter-batch delay** — Stagger concurrent requests with `time.sleep()` based on batch index. Simple and predictable. Reduce default workers from 4 to 2. Adaptive backoff deferred to future enhancement.

- **Configurable params via Settings** — Add 5 new fields to the existing Pydantic `Settings` class. Backwards-compatible via defaults. Passed through orchestrator to fetcher rather than reading global settings inside fetcher.

## Technical Approach

### Data Layer (`data/`)
- **universe.py**: Remove 12 tickers from `SP500_CORE`, 9 from `POPULAR_OPTIONS` (including duplicate RIVN). Pure list edits.
- **cache.py**: Add 4 functions (`load_failure_cache`, `record_failures`, `clear_failure_cache`, `get_failure_cache_stats`) after existing `clear_cache`. All I/O wrapped in try/except. Uses fixed filename `_failures.json` (not date-based).
- **fetcher.py**: Accept configurable `max_retries`, `retry_delays`, `batch_size`, `max_workers` as parameters. Add `batch_idx` to `_process_batch` with `time.sleep()` stagger. Modify `_download_batch` to classify errors by parsing exception messages. Add `FetchErrorType` enum to `models.py`.

### Config Layer
- **config.py**: Add `fetch_max_retries`, `fetch_retry_delays`, `fetch_batch_size`, `fetch_max_workers`, `failure_cache_ttl_hours` fields with sensible defaults.
- **models.py**: Add `FetchErrorType` enum.

### Pipeline Layer
- **orchestrator.py**: In `_phase_data_fetch`, load failure cache before fetch, filter `to_fetch`, pass config params to `fetch_batch()`, record failures after fetch.

### Web Layer
- **routes.py**: Add `get_failure_cache_stats()` call in `dashboard()` context. Add `POST /clear-failure-cache` endpoint (follows `reset_settings` pattern). Parse new fetch config fields in `save_settings`.
- **dashboard.html**: Show "N tickers skipped" in freshness stats section when failures exist.
- **settings.html**: Add "Data Fetch" section with inputs for batch_size, max_workers, max_retries, failure_cache_ttl_hours, and a "Clear Failure Cache" button.

## Implementation Strategy

**Ordering**: Config/models first (foundation) → data layer (cache + fetcher) → orchestrator integration → web UI → tests. Each task can be validated independently.

**Risk mitigation**:
- All failure cache I/O is try/except guarded — corruption never blocks scans
- New Settings fields have defaults matching current behavior (except workers: 4→2)
- Existing 547+ tests must pass after each task

## Task Breakdown Preview

- [ ] Task 1: **Remove delisted tickers from universe.py** — Remove ~20 confirmed delisted/acquired/merged/invalid tickers from SP500_CORE and POPULAR_OPTIONS lists. Fix duplicate RIVN. (1 file)
- [ ] Task 2: **Add fetch config fields and error enum** — Add 5 new fields to Settings class in config.py. Add FetchErrorType enum to models.py. (2 files)
- [ ] Task 3: **Implement failure cache functions** — Add load_failure_cache, record_failures, clear_failure_cache, get_failure_cache_stats to cache.py. JSON-based with TTL eviction and corruption resilience. (1 file)
- [ ] Task 4: **Refactor fetcher with throttling and configurable params** — Accept settings-driven batch_size, max_workers, max_retries, retry_delays. Add inter-batch delay via batch_idx. Classify errors by parsing yfinance exceptions. (1 file)
- [ ] Task 5: **Integrate failure cache into orchestrator** — Filter known failures before fetch, record new failures after fetch, log skip counts. Pass fetch config from settings to fetch_batch. (1 file)
- [ ] Task 6: **Update dashboard and settings UI** — Dashboard: show skipped ticker count. Settings: add "Data Fetch" section with 4 config inputs + "Clear Failure Cache" button. Add POST /clear-failure-cache route. (3 files)
- [ ] Task 7: **Add tests for failure cache, error classification, and fetcher** — ~15-20 tests covering all new functions, error types, TTL eviction, corruption handling, and orchestrator integration with mocked fetcher. (1 new file)

## Dependencies

- **No external dependencies added** — all implementation uses existing stdlib + yfinance
- **Internal ordering**: Task 2 (config/enum) must precede Tasks 3-6. Task 1 is independent.
- **Existing tests**: Must pass after each task (regression gate)

## Success Criteria (Technical)

- All 547+ existing tests pass after implementation
- New test file adds ~15-20 passing tests
- Universe reduced by ~20 tickers (from ~702 to ~682)
- `fetch_batch()` accepts configurable params and no longer uses module-level constants
- Failure cache persists across scans and auto-evicts after TTL
- Settings page shows and persists all 5 new config fields
- Dashboard shows failure cache stats when failures exist
- Error logs include classification (DELISTED/RATE_LIMITED/NETWORK/etc.)

## Estimated Effort

- **7 tasks** across 9 files (1 new test file)
- Tasks are small and focused — most are single-file changes
- Largest task: Task 4 (fetcher refactor) touches the most logic
- Critical path: Task 2 → Tasks 3/4 (parallel) → Task 5 → Task 6

## Tasks Created
- [ ] #13 - Remove delisted tickers from universe.py (parallel: true)
- [ ] #18 - Add fetch config fields and error enum (parallel: true)
- [ ] #19 - Implement failure cache functions (parallel: true, after #18)
- [ ] #14 - Refactor fetcher with throttling and configurable params (parallel: true, after #18)
- [ ] #15 - Integrate failure cache into orchestrator (parallel: false, after #19+#14)
- [ ] #16 - Update dashboard and settings UI (parallel: false, after #18+#19)
- [ ] #17 - Add tests for failure cache, error classification, and fetcher (parallel: false, after #19-#16)

Total tasks: 7
Parallel tasks: 4 (#13, #18, #19, #14)
Sequential tasks: 3 (#15, #16, #17)
Estimated total effort: 12-19 hours
