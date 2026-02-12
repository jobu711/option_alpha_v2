---
name: universe-expansion
status: completed
created: 2026-02-11T14:32:12Z
completed: 2026-02-11T16:30:00Z
progress: 100%
prd: .claude/prds/universe-expansion.md
github: https://github.com/jobu711/option_alpha_v2/issues/20
---

# Epic: Universe Expansion

## Overview

Expand the ticker universe from ~580 hardcoded Python lists to ~3,000 dynamically-maintained optionable stocks. The core change replaces the static `SP500_CORE`, `POPULAR_OPTIONS`, and `OPTIONABLE_ETFS` lists in `data/universe.py` with a file-backed data store (`universe_data.json`) containing ticker metadata (symbol, name, sector, market-cap tier). A weekly auto-refresh mechanism keeps the universe current from free data sources (SEC EDGAR + Nasdaq listings + yfinance validation). The web UI gains a universe preset selector, sector filters, custom watchlists, and scan time estimation. The pipeline orchestrator is updated to accept a dynamic universe based on user selections.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Universe data format | JSON file (`universe_data.json`) | Human-readable, easy to diff, small at ~3k records (~2-3 MB). Parquet overkill for metadata. |
| Refresh data source | SEC EDGAR `company_tickers.json` + yfinance optionability check | Free, no API key needed, reliable. Nasdaq FTP is a fallback. |
| Sector source | yfinance `Ticker.info["sector"]` | Available for free, GICS-compatible, fetched during refresh. |
| Market-cap tiers | Derived from yfinance `Ticker.info["marketCap"]` during refresh | Changes weekly; tier boundaries: Large >$10B, Mid $2-10B, Small $300M-2B, Micro <$300M. |
| Preset implementation | Filter functions over the single universe data file | No separate lists per preset — just metadata queries. Simple, no data duplication. |
| Watchlist storage | `watchlists.json` alongside `config.json` | Keeps watchlists separate from settings. Simple file-based CRUD. |
| Default universe | Full (~3,000) | Matches original PRD target. Users can narrow via presets for faster scans. |
| Universe selector persistence | Fields in `config.json` via Settings class | Consistent with existing config pattern. |

## Technical Approach

### Data Layer Changes (`src/option_alpha/data/`)

**`universe.py` refactor:**
- Replace hardcoded lists with `load_universe_data()` that reads `universe_data.json`
- `universe_data.json` schema: `[{"symbol": "AAPL", "name": "Apple Inc", "sector": "Technology", "market_cap_tier": "large", "asset_type": "stock"}, ...]`
- Retain `get_full_universe()` API (returns list of symbols) for backward compatibility
- Add `get_scan_universe(presets, sectors, watchlist_tickers)` that applies preset + sector filters and merges watchlist tickers
- Ship baseline `universe_data.json` in `src/option_alpha/data/` with the package

**`universe_refresh.py` (new):**
- Fetch SEC EDGAR `company_tickers.json` → extract all US-listed tickers
- Batch-validate optionability via `yf.Ticker(symbol).options` (check for non-empty expiration list)
- Fetch sector + market-cap from `yf.Ticker(symbol).info` for validated tickers
- Write updated `universe_data.json` with diff log (tickers added/removed)
- Store refresh metadata (timestamp, counts) in `universe_meta.json`
- Fallback to existing file if refresh fails

### Config Changes (`src/option_alpha/config.py`)

Add to `Settings`:
```python
# Universe selection
universe_presets: list[str] = ["full"]  # sp500, midcap, smallcap, etfs, full
universe_sectors: list[str] = []  # empty = all sectors
universe_refresh_interval_days: int = 7
```

### Watchlist Layer

**`watchlists.json`** (new, alongside config.json):
```json
{
  "watchlists": {
    "my-techs": ["PLTR", "NET", "CRWD"],
    "earnings-week": ["AAPL", "MSFT"]
  },
  "active_watchlist": null
}
```

Simple file-based CRUD managed via new functions in a `watchlists.py` module or within `universe.py`.

### Pipeline Integration (`src/option_alpha/pipeline/orchestrator.py`)

- `_phase_data_fetch` currently calls `get_full_universe()` directly (line 208)
- Change to call `get_scan_universe()` which respects preset/sector/watchlist config
- No other phase changes needed — scoring, catalysts, options, AI all operate on arbitrary ticker lists

### Web UI Changes (`src/option_alpha/web/`)

**Dashboard (`dashboard.html`):**
- Add universe preset multi-select chips above scan button (S&P 500, Mid-Cap, Small-Cap, ETFs, Full)
- Add sector filter chips/checkboxes (11 GICS sectors)
- Add scan time estimate text (e.g., "~500 tickers, est. 3 min")
- Add universe health stats bar (total tickers, cache %, last refresh)

**Settings (`settings.html`):**
- Add watchlist management section: create, edit, delete named watchlists
- Add manual universe refresh button with status display

**Routes (`routes.py`):**
- `GET /api/universe/stats` — universe size, presets, cache stats, last refresh
- `POST /api/universe/refresh` — trigger manual refresh
- `GET /api/watchlists` — list all watchlists
- `POST /api/watchlists` — create/update watchlist
- `DELETE /api/watchlists/{name}` — delete watchlist
- `POST /api/scan` — update to accept preset/sector/watchlist params

### Performance for 3k Scale

- Increase `fetch_batch_size` default from 20 → 50 (yfinance handles this fine for cached tickers)
- Increase `fetch_max_workers` default from 2 → 4
- The existing 18-hour Parquet cache means most of ~3,000 tickers hit cache on subsequent scans
- The existing failure cache (24h TTL) prevents retrying dead tickers
- No new performance infrastructure needed — the existing batching/caching already scales

## Task Breakdown Preview

- [ ] **Task 1: Universe data store + baseline** — Create `universe_data.json` schema, generate ~3,000-ticker baseline from SEC EDGAR + yfinance, refactor `universe.py` to load from file (FR-1.1–1.5)
- [ ] **Task 2: Dynamic refresh** — Add `universe_refresh.py` with SEC EDGAR fetch, optionability validation, sector/market-cap enrichment, diff logging, weekly auto-trigger (FR-2.1–2.7)
- [ ] **Task 3: Presets + sector filters** — Add `get_scan_universe(presets, sectors)` to `universe.py`, add preset/sector config fields to `Settings`, implement filter logic (FR-3.1–3.5, FR-4.1–4.4)
- [ ] **Task 4: Custom watchlists** — Add `watchlists.json` persistence, CRUD functions, ticker validation, integration with `get_scan_universe()` (FR-5.1–5.6)
- [ ] **Task 5: Pipeline integration** — Update `orchestrator.py` `_phase_data_fetch` to use `get_scan_universe()` instead of `get_full_universe()`, pass preset/sector/watchlist from settings (FR-7 partial)
- [ ] **Task 6: Web UI — universe controls + watchlists** — Dashboard preset selector, sector filter chips, scan time estimate, watchlist management in settings, universe health stats, new API routes (FR-6.1–6.6)
- [ ] **Task 7: Performance tuning** — Increase default batch size/workers, validate 15-minute target at 3k scale, add parallel cache loading if needed (FR-7.1–7.5)
- [ ] **Task 8: Tests** — Update `test_universe.py`, add tests for refresh, presets, sector filters, watchlists, pipeline integration with dynamic universe

## Dependencies

### External
- **SEC EDGAR** (`https://www.sec.gov/files/company_tickers.json`) — ticker discovery (free, no key)
- **yfinance** — optionability validation, sector/market-cap metadata (already a dependency)

### Internal (task ordering)
- Task 1 (data store) must complete before Tasks 2, 3, 4, 5
- Task 3 (presets/filters) must complete before Task 5 (pipeline) and Task 6 (UI)
- Task 4 (watchlists) must complete before Task 6 (UI)
- Task 5 (pipeline) must complete before Task 7 (performance)
- Tasks 6, 7, 8 can partially overlap

### Existing Code Impact
- `data/universe.py` — Major refactor (replace hardcoded lists with file loading)
- `config.py` — Add ~5 new settings fields
- `pipeline/orchestrator.py` — Change 1 line (universe source) + pass-through config
- `web/routes.py` — Add 5-6 new endpoints
- `web/templates/dashboard.html` — Add universe selector section
- `web/templates/settings.html` — Add watchlist management section
- `tests/test_universe.py` — Major rewrite for new data structures

## Success Criteria (Technical)

| Criteria | Target | Validation |
|----------|--------|------------|
| Baseline universe ships with package | ~2,800-3,500 tickers in `universe_data.json` | File exists, `len(load_universe_data()) >= 2800` |
| Backward compatibility | `get_full_universe()` returns all tickers as sorted list | Existing test passes unchanged |
| Refresh works | SEC EDGAR fetch + yfinance validation produces valid universe | Integration test with mocked responses |
| Presets filter correctly | `get_scan_universe(["sp500"])` returns ~500 tickers | Unit test |
| Sector filters work | `get_scan_universe(sectors=["Technology"])` returns tech-only | Unit test |
| Watchlist CRUD | Create, read, update, delete watchlists persisted to file | Unit test |
| Pipeline uses dynamic universe | Orchestrator scans only selected preset/sector tickers | Integration test |
| Full scan ≤ 15 minutes | 3,000 tickers with warm cache completes in time | Manual benchmark |
| All existing tests pass | No regressions | `pytest tests/` |

## Estimated Effort

- **8 tasks** covering data store, refresh, presets, watchlists, pipeline, UI, performance, tests
- **Critical path**: Task 1 → Task 3 → Task 5 → Task 7 (data store → filters → pipeline → perf)
- **Parallelizable**: Tasks 2 (refresh), 4 (watchlists), 8 (tests) can run alongside the critical path
- **Largest task**: Task 1 (baseline generation — requires fetching metadata for ~3,000 tickers from yfinance, may take significant wall-clock time for the initial run)
- **Simplifications applied**:
  - No new database tables — universe metadata is file-based JSON
  - No new caching layer — reuses existing Parquet cache infrastructure
  - No new performance infrastructure — existing batch/cache system scales to 3k with config tuning
  - Presets are just metadata filters, not separate maintained lists

## Tasks Created

- [ ] #21 - Universe data store and baseline generation (parallel: false) — L, 12-16h
- [ ] #24 - Dynamic universe refresh (parallel: true) — M, 8-10h
- [ ] #27 - Universe presets and sector filters (parallel: true) — S, 4-6h
- [ ] #28 - Custom watchlist management (parallel: true) — S, 4-6h
- [ ] #22 - Pipeline integration with dynamic universe (parallel: false) — S, 2-3h
- [ ] #23 - Web UI universe controls and watchlist management (parallel: false) — L, 10-14h
- [ ] #25 - Performance tuning for 3k-scale universe (parallel: true) — S, 3-4h
- [ ] #26 - Tests for universe expansion (parallel: true) — M, 8-10h

Total tasks: 8
Parallel tasks: 5 (#24, #27, #28, #25, #26)
Sequential tasks: 3 (#21, #22, #23)
Estimated total effort: 51-69 hours

### Dependency Graph

```
#21 (data store)
 ├── #24 (refresh)        [parallel]
 ├── #27 (presets/filters) [parallel]
 │    └── #22 (pipeline)
 │         ├── #23 (web UI) ← also depends on #28
 │         └── #25 (perf)   [parallel]
 ├── #28 (watchlists)     [parallel]
 └── #26 (tests)          [parallel, progressive]
```
