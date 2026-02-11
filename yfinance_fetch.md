# Plan: Reduce yfinance Fetch Errors

## Context

Running a scan produces hundreds of errors from two distinct causes:

1. **Delisted/invalid tickers** in the static universe lists — ATVI (acquired by MSFT), SIVB/FRC (collapsed), DWAC (merged), DISH (merged with T-Mobile), CTLT (acquired), BLDE, CZOO, AMPS, etc. These are retried every scan because failures aren't cached.

2. **Yahoo Finance rate limiting** — 4 concurrent workers fire batches simultaneously with no throttle. After the first wave, Yahoo returns 401 "Invalid Crumb" / "Unauthorized" errors, causing entire batches to return 0/50 tickers. The logs show batches 5+ consistently returning 0 results.

Both problems compound: 702 tickers in 15 batches × 4 workers × 3 retries with backoff = massive wasted API calls.

---

## Changes (3 parts)

### Part 1: Remove known-delisted tickers from universe.py

**File**: `src/option_alpha/data/universe.py`

Remove these confirmed delisted/acquired/merged tickers from `SP500_CORE`:
- `ATVI` — acquired by Microsoft (2023)
- `SIVB` — Silicon Valley Bank, collapsed (2023)
- `FRC` — First Republic Bank, collapsed (2023)
- `DISH` — merged with EchoStar/T-Mobile
- `CTLT` — acquired by Novo Holdings (2024)
- `DFS` — acquired by Capital One (2024)
- `LUMN` — restructured, trades as LUMN but no options data
- `PXD` — acquired by ExxonMobil (2024)
- `SBNY` — Signature Bank, collapsed (2023)
- `PEAK` — merged into Healthpeak Properties (DOC)
- `BF.B`, `BRK.B` — dot-notation tickers that yfinance can't resolve

Remove from `POPULAR_OPTIONS`:
- `DWAC` — merged into Trump Media (DJT)
- `CZOO` — Cazoo, delisted
- `AMPS` — delisted
- `BLDE` — delisted
- `LTHM` — acquired by Arcadium Lithium
- `ANNA` — not a real ticker
- `DTST` — very low volume, not optionable
- `HYMC` — delisted

Also remove duplicate `RIVN` on line 87 and duplicate entries that appear in multiple lists (e.g., `ENPH`, `GNRC`, `TTD` in both SP500_CORE and POPULAR_OPTIONS).

---

### Part 2: Add failure cache (negative cache)

**Purpose**: Tickers that fail to fetch are remembered so they aren't re-attempted every scan. Expires after a configurable TTL (default 7 days).

#### 2a. Config setting

**File**: `src/option_alpha/config.py` (after line 55, `db_path` field)

Add:
```python
failure_cache_ttl_hours: int = 168  # 7 days
```

#### 2b. Failure cache functions

**File**: `src/option_alpha/data/cache.py` (after `clear_cache` at line 254)

Add 4 functions. Data stored in `data/cache/_failures.json`:
```json
{"BADTICKER": {"last_failure": "2026-02-11T14:30:00", "count": 3}}
```

- **`load_failure_cache(settings, ttl_hours) -> dict`** — Read file, evict entries older than TTL, return dict of failed tickers. Returns `{}` if file missing/corrupt.
- **`record_failures(requested, returned, settings) -> int`** — Compute `failed = set(requested) - set(returned)`. Write/update entries. Return count.
- **`clear_failure_cache(settings) -> bool`** — Delete `_failures.json`.
- **`get_failure_cache_stats(settings) -> dict`** — Return `{"total_failed": N}` for dashboard.

Uses fixed filename (not date-based like `save_json`) because failure data must persist across days. All I/O wrapped in try/except so corruption never blocks scans.

#### 2c. Orchestrator integration

**File**: `src/option_alpha/pipeline/orchestrator.py`

**Import** (line 30): Add `load_failure_cache, record_failures` to existing import.

**In `_phase_data_fetch`** (between lines 213 and 220):

After computing `to_fetch`, filter out known failures:
```python
failure_cache = load_failure_cache(self.settings, self.settings.failure_cache_ttl_hours)
if failure_cache:
    to_fetch = [t for t in to_fetch if t not in failure_cache]
```

After `fetch_batch` returns, record new failures:
```python
record_failures(to_fetch, fetched, self.settings)
```

---

### Part 3: Throttle fetcher to avoid Yahoo 401 rate limits

**File**: `src/option_alpha/data/fetcher.py`

The current setup fires all 15 batches across 4 workers simultaneously, triggering Yahoo's rate limiter. Two changes:

#### 3a. Reduce default workers from 4 to 2

**Line 154**: Change `max_workers: int = 4` to `max_workers: int = 2`

#### 3b. Add inter-batch delay

**In `_process_batch`** (line 184): Add a small sleep before the yfinance download to stagger requests:

```python
def _process_batch(batch_idx: int, batch: list[str]) -> dict[str, TickerData]:
    if batch_idx > 0:
        time.sleep(2 * batch_idx % max_workers)  # stagger concurrent batches
    ...
```

Pass `batch_idx` via `enumerate(batches)` in the executor submission at line 200-202.

This spaces out requests so Yahoo doesn't see a burst of 4 simultaneous batch downloads.

---

### Part 4: Dashboard / Settings UI

#### 4a. Show failure count on dashboard

**File**: `src/option_alpha/web/routes.py` (line 133)
- Call `get_failure_cache_stats(settings)` and pass to template context.

**File**: `src/option_alpha/web/templates/dashboard.html` (after line 82)
- Show `"N tickers skipped"` in the freshness stats if failures exist.

#### 4b. Clear button on settings page

**File**: `src/option_alpha/web/routes.py`
- Add POST `/clear-failure-cache` endpoint (follows `reset_settings` pattern).

**File**: `src/option_alpha/web/templates/settings.html` (before line 147)
- Add "Data Cache" section with a button to clear the failure cache.

---

### Part 5: Tests

**New file**: `tests/test_failure_cache.py`

~15 tests covering:
- `load_failure_cache` — empty, valid, TTL eviction, corrupt file
- `record_failures` — new failures, count increment, all-succeed
- `clear_failure_cache` — existing and nonexistent file
- `get_failure_cache_stats` — empty and populated
- Orchestrator integration — verify `fetch_batch` skips known failures

---

## Files Modified

| File | Change |
|------|--------|
| `src/option_alpha/data/universe.py` | Remove ~20 delisted tickers, fix duplicate |
| `src/option_alpha/config.py:55` | Add `failure_cache_ttl_hours` field |
| `src/option_alpha/data/cache.py:254+` | Add 4 failure cache functions |
| `src/option_alpha/data/fetcher.py:154,184` | Reduce workers to 2, add inter-batch delay |
| `src/option_alpha/pipeline/orchestrator.py:30,213` | Integrate failure cache |
| `src/option_alpha/web/routes.py` | Dashboard stats + clear endpoint |
| `src/option_alpha/web/templates/dashboard.html:82` | Show skipped count |
| `src/option_alpha/web/templates/settings.html:147` | Clear cache button |
| `tests/test_failure_cache.py` (new) | ~15 test methods |

## Verification

1. `pytest tests/` — all existing 547+ tests pass
2. `pytest tests/test_failure_cache.py` — new tests pass
3. Run a scan (`python -m option_alpha`) — observe:
   - No delisted ticker errors for removed symbols
   - Logs show "Skipping N known-failed tickers" on second scan
   - Batches no longer return 0/50 due to throttling
4. Dashboard shows skipped count after first scan
5. Settings page "Clear Failure Cache" resets for next scan
