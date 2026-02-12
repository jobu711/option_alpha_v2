---
name: universe-expansion
description: Expand ticker universe from ~580 hardcoded tickers to ~3,000 optionable stocks with dynamic discovery, preset/sector filters, and custom watchlists
status: backlog
created: 2026-02-11T14:18:54Z
---

# PRD: Universe Expansion

## Executive Summary

Expand Option Alpha's ticker universe from ~580 hardcoded tickers to ~3,000+ dynamically-maintained optionable stocks. Replace the static Python lists in `data/universe.py` with a hybrid system: a large shipped baseline list plus weekly auto-refresh from free data sources. Add a universe selector to the web UI with market-cap tier presets (S&P 500, Mid-cap, Small-cap, Full) and sector filters (Tech, Healthcare, Energy, etc.), along with support for user-defined custom watchlists. The full ~3,000 universe becomes the new default scan target.

**Value Proposition:** Catch 5x more breakout opportunities by scanning the full optionable stock universe, while giving traders granular control over what they scan and how long it takes.

## Problem Statement

The current universe is ~580 hardcoded tickers across three manually-curated Python lists (`SP500_CORE`, `POPULAR_OPTIONS`, `OPTIONABLE_ETFS`). This creates several problems:

1. **Missed opportunities** — Only ~19% of the ~3,000+ optionable US stocks are covered. High-probability setups in mid/small-cap names are invisible.
2. **Stale data** — Hardcoded lists go stale as companies IPO, get acquired, delist, or gain/lose options markets. Updates require code changes and redeployment.
3. **No user control** — Traders cannot focus scans on specific sectors, market-cap tiers, or personal watchlists. It's all-or-nothing with the single static universe.
4. **PRD gap** — The original PRD targets "3,000+ optionable stocks" but the implementation delivers ~580.

**Why now:** The data layer, caching, and failure management are mature and battle-tested. The fetcher already handles batching, retries, and failure caching. Expanding the universe is now a matter of sourcing more tickers and adding UI controls — the infrastructure is ready.

## User Stories

### Primary Persona: Active Retail Options Trader

Same persona as the core PRD — trades options 3-5 times/week, wants data-driven setups, comfortable with browser-based tools.

### User Stories

**UE-1: Full Universe Scan**
> As a trader, I want the default scan to cover all ~3,000 optionable US stocks so I never miss a high-scoring setup in a name I hadn't thought to watch.

Acceptance Criteria:
- Default scan processes ~3,000 tickers (after price/volume filtering)
- Scan completes within 15 minutes for the full universe
- Results include tickers from all market-cap tiers, not just large-caps
- Failure rate stays below 5% of total universe

**UE-2: Universe Presets**
> As a trader, I want to select a scan universe preset (e.g., "S&P 500 Only" or "Mid-Cap") so I can run a faster, focused scan when I know what segment I want to trade.

Acceptance Criteria:
- Universe selector dropdown visible on the dashboard scan controls
- Available presets: S&P 500 (~500), Mid-Cap (~600), Small-Cap (~900), ETFs (~100), Full (~3,000)
- Selecting a preset updates the expected scan time estimate
- Preset selection persists across sessions (saved in config)
- Can combine presets (e.g., S&P 500 + ETFs)

**UE-3: Sector Filters**
> As a trader, I want to filter the universe by sector (Technology, Healthcare, Energy, etc.) so I can focus on sectors with upcoming catalysts or macro tailwinds.

Acceptance Criteria:
- Sector filter chips/checkboxes displayed alongside universe preset selector
- Standard GICS sectors: Technology, Healthcare, Financials, Consumer Discretionary, Consumer Staples, Industrials, Energy, Materials, Utilities, Real Estate, Communication Services
- Sector filters combine with universe presets (e.g., "S&P 500 + Technology only")
- Sector metadata stored per ticker in the universe data
- Filter selection persists across sessions

**UE-4: Custom Watchlist**
> As a trader, I want to paste a list of tickers to add to my scan so I can include specific names I'm watching that might not be in the default universe.

Acceptance Criteria:
- Text input on dashboard to paste comma/space/newline-separated tickers
- Custom tickers are validated (must exist on yfinance, must be optionable)
- Custom watchlist is saved and reusable across sessions
- Can name and manage multiple watchlists
- Custom watchlist can be used alone or combined with presets

**UE-5: Universe Auto-Refresh**
> As a trader, I want the ticker universe to stay current automatically so I don't miss newly optionable stocks or waste time on delisted ones.

Acceptance Criteria:
- Universe master list refreshes weekly from an external source
- Refresh happens automatically on first scan after 7 days since last refresh
- Can trigger manual refresh from the settings page
- Refresh logs show: tickers added, tickers removed, reason for changes
- Stale universe warning if refresh fails for 30+ days

**UE-6: Universe Health Dashboard**
> As a trader, I want to see universe statistics so I understand how healthy my scan data is.

Acceptance Criteria:
- Universe stats visible on dashboard: total tickers, active after filtering, cache coverage %, failure rate %
- Last refresh date displayed
- Breakdown by preset/sector shown in settings or dedicated page

## Requirements

### Functional Requirements

#### FR-1: Universe Data Store
- **FR-1.1:** Replace hardcoded Python lists with a structured data file (`universe.json` or `universe.parquet`) containing ticker, company name, sector, market cap tier, and optionability flag
- **FR-1.2:** Ship a baseline universe of ~3,000 optionable US stocks in the package
- **FR-1.3:** Include GICS sector classification for each ticker
- **FR-1.4:** Include market-cap tier classification: Large-cap (>$10B), Mid-cap ($2-10B), Small-cap ($300M-2B), Micro-cap (<$300M)
- **FR-1.5:** Maintain backward compatibility — existing `get_full_universe()` API returns all tickers as before

#### FR-2: Dynamic Universe Refresh
- **FR-2.1:** Weekly auto-refresh of the universe master list
- **FR-2.2:** Source optionable tickers from free data (SEC EDGAR company tickers, Nasdaq traded listings, or yfinance screening)
- **FR-2.3:** Validate refreshed tickers: confirm optionability, remove delisted/suspended
- **FR-2.4:** Diff reporting: log tickers added/removed on each refresh
- **FR-2.5:** Fallback to cached/shipped baseline if refresh source is unavailable
- **FR-2.6:** Manual refresh trigger from settings UI
- **FR-2.7:** Store refresh metadata: last refresh timestamp, source, tickers added/removed count

#### FR-3: Universe Presets
- **FR-3.1:** Define named presets: `sp500`, `midcap`, `smallcap`, `etfs`, `full`
- **FR-3.2:** Presets are based on market-cap tier and/or asset type classification in the universe data
- **FR-3.3:** Presets are combinable (union of selected presets)
- **FR-3.4:** Selected preset(s) persisted in `config.json`
- **FR-3.5:** `get_scan_universe()` function returns tickers matching current preset + filter configuration

#### FR-4: Sector Filters
- **FR-4.1:** GICS sector filter: 11 standard sectors
- **FR-4.2:** Sector filters combine with presets via intersection (e.g., S&P 500 AND Technology)
- **FR-4.3:** Sector filter selection persisted in `config.json`
- **FR-4.4:** "All sectors" is the default (no filtering)

#### FR-5: Custom Watchlists
- **FR-5.1:** Users can create named watchlists of arbitrary tickers
- **FR-5.2:** Watchlist input: paste comma/space/newline-separated tickers in the UI
- **FR-5.3:** Ticker validation on input: check existence via yfinance, warn on invalid tickers
- **FR-5.4:** Watchlists stored in `config.json` or a separate `watchlists.json`
- **FR-5.5:** Watchlists can be selected as a scan universe (alone or combined with presets)
- **FR-5.6:** CRUD operations: create, rename, edit, delete watchlists

#### FR-6: Web UI — Universe Controls
- **FR-6.1:** Universe preset selector (multi-select chips or checkboxes) on the scan dashboard
- **FR-6.2:** Sector filter controls (chips or checkboxes) on the scan dashboard
- **FR-6.3:** Estimated scan time display based on selected universe size
- **FR-6.4:** Custom watchlist management page (accessible from settings or dashboard)
- **FR-6.5:** Universe health stats: total tickers, post-filter count, cache hit %, failure %, last refresh date
- **FR-6.6:** Quick-add: paste tickers inline on the dashboard to add to current scan without saving a watchlist

#### FR-7: Performance Optimization for Scale
- **FR-7.1:** Adaptive batch sizing: increase batch size for cached tickers, decrease for fresh fetches
- **FR-7.2:** Parallel cache loading: load Parquet cache files concurrently for faster startup
- **FR-7.3:** Smart fetch ordering: prioritize tickers most likely to pass scoring threshold (based on historical scores)
- **FR-7.4:** Progressive results: start displaying scored results while later batches are still fetching
- **FR-7.5:** Tiered rate limiting: respect yfinance rate limits with dynamic backoff based on error rate
- **FR-7.6:** Cache warming: background pre-fetch of universe data outside of scan runs (optional)

### Non-Functional Requirements

#### NFR-1: Performance
- Full universe scan (~3,000 tickers) completes within 15 minutes
- Preset scans scale linearly: S&P 500 (~500 tickers) in ~3 minutes, Mid-cap (~600) in ~3.5 minutes
- Universe selector/filter UI interactions are instant (<100ms)
- Universe refresh completes within 5 minutes

#### NFR-2: Reliability
- Universe refresh failures fall back to the last-known-good list — never leave the user with an empty universe
- Graceful degradation: if sector/market-cap metadata is missing for some tickers, include them in "Unknown" category rather than excluding
- Weekly refresh retries up to 3 times before falling back

#### NFR-3: Storage
- Universe data file: <5 MB for ~3,000 tickers with metadata
- Parquet cache for full universe: ~300-500 MB (manageable for local tool)
- Cache eviction: remove Parquet files for tickers no longer in the universe after 30 days
- Configurable cache directory location

#### NFR-4: Data Quality
- Universe refresh validates all tickers: active, not suspended, has options market
- Market-cap tier and sector classification accuracy >95%
- Stale ticker detection: flag tickers with no trading activity in 30+ days

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Universe size | 2,800-3,500 optionable tickers | Count of validated tickers in universe data |
| Full scan time | < 15 minutes | Pipeline timing logs |
| Preset scan time | Proportional to ticker count (±20%) | Pipeline timing logs |
| Refresh success rate | > 95% of weekly refreshes succeed | Refresh metadata logs |
| Sector coverage | All 11 GICS sectors represented | Universe data validation |
| Data freshness | Universe master list < 7 days old | Refresh timestamp check |
| Scan completion rate | > 95% of universe tickers processed | Logged success/failure counts |
| User workflow | Select universe + start scan < 10 seconds | Manual verification |

## Constraints & Assumptions

### Constraints
- **yfinance rate limiting** — 3,000 tickers at current batch sizes would take ~25 minutes without caching. Must leverage cache aggressively and optimize fetch strategy.
- **No paid data sources** — Must source universe data from free APIs/datasets (SEC EDGAR, Nasdaq listings, yfinance itself). Data quality may be imperfect.
- **Sector classification** — Free sources may not provide GICS sectors directly; may need to map from SIC codes or scrape from yfinance `info` field.
- **Market-cap classification** — Market caps change daily; tier boundaries are approximate. Classifications refresh weekly alongside universe.

### Assumptions
- Most tickers in the ~3,000 universe will have cached data after the first full scan, making subsequent scans significantly faster
- yfinance rate limits are manageable at ~3,000 tickers with 18-hour cache freshness (only ~300-500 tickers need fresh fetch per scan)
- SEC EDGAR or Nasdaq listings provide a reliable free source for discovering optionable tickers
- Users prefer broader default coverage over faster default scan times

## Out of Scope

- **International markets** — US exchanges only (NYSE, NASDAQ, AMEX). No international stock options.
- **Real-time universe updates** — Weekly refresh is sufficient; no intraday discovery of newly listed options.
- **Penny stocks** — Sub-$5 stocks remain filtered out by default (existing `min_price` setting).
- **Options universe expansion** — This PRD covers the _equity_ universe. Options chain expansion (multi-leg, weeklies, etc.) is a separate feature.
- **Paid data feed integration** — No Bloomberg, Polygon, or Tradier integrations. Free sources only.
- **Smart universe suggestions** — No ML-based "suggested tickers" or "similar to your watchlist" features.
- **Collaborative watchlists** — No sharing/importing watchlists from other users. Single-user tool.

## Dependencies

### External Dependencies
- **SEC EDGAR Company Tickers** (`https://www.sec.gov/files/company_tickers.json`) — Free, no API key, updated daily. Source for discovering US-listed companies.
- **Nasdaq Traded Listings** (`ftp://ftp.nasdaqtrader.com/symboldirectory/`) — Free listings of all Nasdaq and NYSE-traded securities.
- **yfinance** — Used to validate optionability (check if `ticker.options` returns expiration dates) and fetch sector/market-cap metadata from `ticker.info`.

### Internal Dependencies
- **Data layer** (`data/fetcher.py`, `data/cache.py`) — Must handle 5x more tickers. May need batch size and worker tuning.
- **Config system** (`config.py`) — Must persist universe preset selection, sector filters, and custom watchlists.
- **Pipeline orchestrator** (`pipeline/orchestrator.py`) — Must accept dynamic universe input instead of calling `get_full_universe()` directly.
- **Web UI** (`web/`) — Must add universe selector controls to the scan dashboard and watchlist management page.
- **Scoring engine** — No changes needed; already operates on arbitrary ticker lists.
- **Existing tests** — Must update `test_universe.py` for new data structures and APIs.

## Technical Approach (High-Level)

### Phase 1: Universe Data Store
Replace hardcoded lists with `data/universe_data.json` containing ticker metadata (symbol, name, sector, market_cap_tier, asset_type). Ship a ~3,000-ticker baseline with the package. Update `universe.py` to load from file instead of Python lists.

### Phase 2: Dynamic Refresh
Add `universe_refresh.py` that fetches SEC EDGAR + Nasdaq listings, cross-references with yfinance for optionability/metadata, and updates the universe data file. Run weekly on scan trigger or manually from settings.

### Phase 3: Presets & Filters
Add preset/sector filter logic to `universe.py`. New `get_scan_universe(presets, sectors)` function. Persist selections in config.

### Phase 4: Custom Watchlists
Add watchlist CRUD to config/persistence layer. Validation via yfinance. UI for management.

### Phase 5: UI Controls
Add universe selector, sector filter chips, scan time estimator, and watchlist management to the web dashboard.

### Phase 6: Performance Optimization
Tune batch sizes, add parallel cache loading, implement smart fetch ordering, and ensure 15-minute scan time for full universe.
