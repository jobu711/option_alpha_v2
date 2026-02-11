---
name: yfinance-fetch-error
description: Reduce yfinance fetch errors through delisted ticker cleanup, failure caching, adaptive throttling, error classification, and configurable retry
status: backlog
created: 2026-02-11T13:35:05Z
---

# PRD: yfinance-fetch-error

## Executive Summary

The Option Alpha scanner experiences hundreds of avoidable errors per scan run due to three compounding problems: delisted/invalid tickers in the static universe lists, Yahoo Finance rate limiting from aggressive concurrent requests, and no memory of previous failures. This PRD defines a comprehensive solution that cleans the ticker universe, adds intelligent failure caching, implements adaptive throttling, classifies errors by type, and makes retry/batch parameters configurable — transforming a noisy, wasteful scan into a reliable, efficient one.

## Problem Statement

Running a full scan produces hundreds of errors from distinct causes that compound each other:

1. **Delisted/invalid tickers** (~20 symbols) in the static universe lists — ATVI (acquired by MSFT), SIVB/FRC (collapsed banks), DWAC (merged), DISH (merged with T-Mobile), etc. These are retried every scan because failures aren't cached.

2. **Yahoo Finance rate limiting** — 4 concurrent workers fire batches simultaneously with no throttle. After the first wave, Yahoo returns 401 "Invalid Crumb" / "Unauthorized" errors, causing entire batches to return 0/50 tickers. Logs show batches 5+ consistently returning 0 results.

3. **No error classification** — All failures are logged as generic errors. There is no distinction between "this ticker doesn't exist" (permanent), "Yahoo rate-limited us" (transient), and "network timeout" (retryable). This makes diagnosis difficult and prevents intelligent retry strategies.

4. **Hardcoded retry/batch parameters** — Retry count (3), backoff delays ([1, 2, 4]s), batch size (50), and worker count (4) are all hardcoded constants. Users cannot tune these for their environment.

The compounding effect: 702 tickers in 15 batches x 4 workers x 3 retries with backoff = massive wasted API calls, with the same delisted tickers failing every scan.

**Why now**: This is the most common source of noise in scan output and wastes significant time per scan. Fixing it directly improves scan reliability and user experience.

## User Stories

### US-1: Clean scan output
**As a** user running a scan, **I want** the scanner to skip known-invalid tickers **so that** my scan completes without hundreds of avoidable errors.

**Acceptance Criteria:**
- Confirmed delisted/acquired/merged tickers are removed from `universe.py`
- Dot-notation tickers that yfinance can't resolve (BF.B, BRK.B) are removed
- Duplicate entries (e.g., RIVN appearing twice) are removed
- Total universe size decreases by ~20 tickers
- All existing tests still pass after removals

### US-2: Failure caching across scans
**As a** user running repeated scans, **I want** tickers that failed on the last scan to be skipped automatically **so that** subsequent scans are faster and produce fewer errors.

**Acceptance Criteria:**
- Tickers that fail to return data are recorded in a failure cache
- On the next scan, cached failures are skipped (with a log message)
- Failure cache entries expire after a configurable TTL (default 7 days)
- Failure cache file corruption never blocks a scan
- Dashboard shows count of skipped tickers
- Settings page has a "Clear Failure Cache" button to force retry

### US-3: Avoid rate limiting
**As a** user, **I want** the scanner to throttle requests to avoid Yahoo Finance rate limits **so that** batches don't return 0 results due to 401 errors.

**Acceptance Criteria:**
- Default workers reduced from 4 to 2
- Inter-batch delay staggers concurrent requests
- Batches no longer consistently return 0/50 results
- Scan still completes in a reasonable timeframe

### US-4: Error classification
**As a** user or developer, **I want** fetch errors to be classified by type **so that** I can distinguish permanent failures (delisted) from transient ones (rate limit, network) and take appropriate action.

**Acceptance Criteria:**
- Errors are classified into categories: `DELISTED` (ticker doesn't exist), `RATE_LIMITED` (HTTP 401/429), `NETWORK` (timeout, connection error), `INSUFFICIENT_DATA` (ticker exists but < 5 data points), `UNKNOWN`
- Failure cache records the error type alongside the ticker
- Different error types can have different cache TTLs (e.g., DELISTED = 30 days, RATE_LIMITED = 1 hour)
- Logs include error classification for easier debugging

### US-5: Configurable retry and batch parameters
**As a** user, **I want** to tune retry count, backoff delays, batch size, and worker count from the settings page **so that** I can optimize for my network environment.

**Acceptance Criteria:**
- `max_retries`, `retry_delays`, `batch_size`, `max_workers`, and `failure_cache_ttl_hours` are configurable via `Settings` class
- Settings page exposes these under a "Data Fetch" section
- Defaults match current behavior (except workers: 4 -> 2)
- Changes persist via `config.json`

## Requirements

### Functional Requirements

#### FR-1: Universe cleanup
- Remove ~20 confirmed delisted/acquired/merged tickers from `SP500_CORE` and `POPULAR_OPTIONS` in `universe.py`
- Remove dot-notation tickers that yfinance cannot resolve (`BF.B`, `BRK.B`)
- Remove duplicate entries (e.g., `RIVN` on line 87)
- Remove cross-list duplicates where the same ticker appears in multiple lists (dedup handled by `get_full_universe()` already, but clean source lists for maintainability)

**Tickers to remove from SP500_CORE:**
ATVI, SIVB, FRC, DISH, CTLT, DFS, LUMN, PXD, SBNY, PEAK, BF.B, BRK.B

**Tickers to remove from POPULAR_OPTIONS:**
DWAC, CZOO, AMPS, BLDE, LTHM, ANNA, DTST, HYMC, duplicate RIVN

#### FR-2: Failure cache
- Store failure data in `data/cache/_failures.json` with structure:
  ```json
  {"TICKER": {"last_failure": "ISO-datetime", "count": 3, "error_type": "DELISTED"}}
  ```
- Implement four functions in `data/cache.py`:
  - `load_failure_cache(settings, ttl_hours) -> dict` — Load, evict expired, return
  - `record_failures(requested, returned, settings, error_type) -> int` — Compute diff, write/update, return count
  - `clear_failure_cache(settings) -> bool` — Delete file
  - `get_failure_cache_stats(settings) -> dict` — Return summary for dashboard
- Integrate into `_phase_data_fetch` in `orchestrator.py`: filter before fetch, record after fetch

#### FR-3: Adaptive throttling
- Reduce `max_workers` default from 4 to 2
- Add inter-batch delay in `_process_batch` to stagger concurrent requests
- Pass `batch_idx` via `enumerate(batches)` in executor submission

#### FR-4: Error classification
- Define `FetchErrorType` enum: `DELISTED`, `RATE_LIMITED`, `NETWORK`, `INSUFFICIENT_DATA`, `UNKNOWN`
- Parse yfinance exceptions and HTTP status codes to classify errors:
  - 401/403/"Invalid Crumb" -> `RATE_LIMITED`
  - Ticker not found / empty data -> `DELISTED` or `INSUFFICIENT_DATA`
  - Timeout / ConnectionError -> `NETWORK`
- Log error type alongside ticker symbol
- Store error type in failure cache for differentiated TTL behavior

#### FR-5: Configurable retry/batch settings
- Add to `Settings` class in `config.py`:
  - `fetch_max_retries: int = 3`
  - `fetch_retry_delays: list[float] = [1.0, 2.0, 4.0]`
  - `fetch_batch_size: int = 50`
  - `fetch_max_workers: int = 2`
  - `failure_cache_ttl_hours: int = 168` (7 days)
- Pass these from orchestrator into `fetch_batch()` and `retry_with_backoff()`
- Expose on settings page under "Data Fetch" section

#### FR-6: Dashboard and settings UI
- Dashboard: show "N tickers skipped (cached failures)" in freshness stats
- Settings page: "Data Fetch" section with:
  - Failure cache TTL (hours)
  - Max retries
  - Batch size
  - Max workers
  - "Clear Failure Cache" button

### Non-Functional Requirements

- **Resilience**: Failure cache corruption (invalid JSON, missing file, permission error) must never block a scan. All I/O wrapped in try/except with fallback to empty state.
- **Performance**: Throttling should not more than double scan time compared to current (broken) behavior. The goal is reliability, not speed.
- **Backwards compatibility**: Existing `config.json` files without new fields should continue to work via Pydantic defaults.
- **Testing**: All new code must have comprehensive tests. No live API calls in tests.
- **Logging**: All new behavior must be logged at appropriate levels (INFO for skipped tickers, DEBUG for cache operations, WARNING for errors).

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Fetch errors per scan | Hundreds | < 20 (only genuinely unavailable tickers) |
| Batches returning 0/50 | 5+ per scan | 0 |
| Second-scan speed (cached failures) | Same as first | Faster (skip known failures) |
| Error diagnosis time | Manual log reading | Classified by type in logs and dashboard |
| Retry/batch tunability | Edit source code | Settings page |

## Constraints & Assumptions

**Constraints:**
- Must use existing yfinance library (no alternative data providers in scope)
- Must maintain Python 3.11+ compatibility
- Cannot add new external dependencies
- Must preserve existing test suite (547+ tests passing)

**Assumptions:**
- The delisted ticker list in the plan is accurate as of February 2026
- Yahoo Finance rate limiting triggers around 4+ concurrent batch downloads
- Reducing to 2 workers with staggered delays will be sufficient to avoid 401s
- 7-day default TTL for failure cache is appropriate (balances freshness vs efficiency)

## Out of Scope

- **Auto-validating the ticker universe** — Periodically checking if tickers are still valid/tradeable is a separate feature
- **Alternative data providers** — Switching away from yfinance
- **Real-time rate limit detection** — Dynamically adjusting concurrency mid-scan based on HTTP response codes (could be a future enhancement)
- **Historical failure analytics** — Tracking failure trends over time
- **Ticker universe auto-update** — Automatically pulling current S&P 500 / popular options lists

## Dependencies

- **Internal**: `config.py` (Settings class), `data/cache.py` (cache layer), `data/fetcher.py` (fetch logic), `pipeline/orchestrator.py` (integration), `web/routes.py` + templates (UI)
- **External**: yfinance library behavior (HTTP error codes, batch download semantics)
- **Testing**: pytest, existing test fixtures and mocking patterns

## Alternative Approaches Considered

### Failure cache: SQLite vs JSON
The existing stack uses SQLite for persistence. Storing failures there would be more robust and queryable. However, JSON was chosen because:
- The data layer (`data/`) should remain independent of the persistence layer (`persistence/`)
- JSON is simpler for a small key-value store (~20-50 entries)
- File-based cache aligns with existing parquet/JSON caching patterns in `cache.py`

### Throttling: Fixed delays vs adaptive backoff
The plan uses fixed inter-batch delays. An alternative is adaptive backoff that increases delays when 401 errors are detected. The fixed approach was chosen for simplicity, but the error classification system (FR-4) lays groundwork for adaptive throttling as a future enhancement.

### yfinance session-based rate limiting
yfinance supports custom `requests.Session` objects with `requests-ratelimiter` for automatic rate limiting. This would add an external dependency, so manual throttling was chosen instead.

## Files Modified

| File | Change |
|------|--------|
| `src/option_alpha/data/universe.py` | Remove ~20 delisted tickers, fix duplicates |
| `src/option_alpha/config.py` | Add fetch config fields + failure_cache_ttl_hours |
| `src/option_alpha/data/cache.py` | Add 4 failure cache functions |
| `src/option_alpha/data/fetcher.py` | Add error classification, reduce workers, add throttling, accept configurable params |
| `src/option_alpha/pipeline/orchestrator.py` | Integrate failure cache, pass config to fetcher |
| `src/option_alpha/web/routes.py` | Dashboard stats + clear failure cache endpoint |
| `src/option_alpha/web/templates/dashboard.html` | Show skipped ticker count |
| `src/option_alpha/web/templates/settings.html` | Data Fetch config section + clear cache button |
| `tests/test_failure_cache.py` (new) | ~15-20 tests for failure cache + error classification |
