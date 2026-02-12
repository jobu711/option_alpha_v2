---
name: universe-fix
description: Rebuild ticker universe with OI filtering, delisted ticker removal, and weekly auto-refresh
status: backlog
created: 2026-02-12T12:56:12Z
---

# PRD: universe-fix

## Executive Summary

Full overhaul of the option scanner's ticker universe system. The current 687-ticker universe contains stale, delisted, and low-liquidity tickers that waste API calls and slow down scans. This project regenerates the stock universe from scratch with strict open interest filtering (100+ total OI), preserves existing ETFs, and implements weekly automated refresh to maintain data quality. Every ticker in the universe must be active, optionable, and have meaningful options activity.

## Problem Statement

The ticker universe (`universe_data.json`) is a static curated list that has degraded over time. Key gaps:

1. **No open interest validation** — A ticker with options expirations but zero OI passes all current checks. The refresh only verifies a ticker *has* expiration dates, not that anyone is actually trading those options.
2. **Stale/delisted tickers** — Delisted or changed tickers remain in the universe. When the scan hits these, yfinance throws errors. The pipeline catches them gracefully (doesn't crash) but wastes API calls and slows throughput.
3. **No automated maintenance** — The refresh mechanism exists but doesn't run on a schedule. The universe drifts out of date between manual interventions.

**Why now**: As the scanner matures, input data quality is the biggest bottleneck. Fixing the universe is higher leverage than improving any downstream phase.

## User Stories

### Scanner Operator
- **As a scanner operator**, I want every ticker in my universe to have real options liquidity, so scans don't waste time on dead tickers.
  - *Acceptance criteria*: All stock tickers have 100+ total OI across their options chains. Tickers below threshold are excluded.
- **As a scanner operator**, I want the universe to stay current automatically, so I don't need to manually curate the ticker list.
  - *Acceptance criteria*: Weekly refresh validates all tickers and removes any that fail. No manual intervention needed.
- **As a scanner operator**, I want delisted tickers removed before they reach the scan pipeline, so I don't see yfinance errors in my scan logs.
  - *Acceptance criteria*: Tickers that fail data fetch during refresh are immediately removed from `universe_data.json`.

### Developer
- **As a developer**, I want the universe refresh to be reliable and recoverable, so a failed refresh doesn't corrupt my ticker list.
  - *Acceptance criteria*: If refresh fails partway, previous `universe_data.json` is preserved. Partial ticker failures don't block the full refresh.
- **As a developer**, I want to trigger a full regeneration manually, so I can rebuild the universe on demand.
  - *Acceptance criteria*: Regeneration available via CLI and API endpoint.

## Requirements

### Functional Requirements

#### Open Interest Filtering
- **OI-01**: Universe refresh validates total open interest across all strikes/expirations for each ticker
- **OI-02**: Tickers with total OI below 100 are excluded from `universe_data.json`
- **OI-03**: OI threshold is configurable in settings (default: 100)
- **OI-04**: OI validation runs during both initial regeneration and weekly refresh

#### Delisted Ticker Removal
- **DELIST-01**: Tickers that fail API data fetch are immediately removed from universe
- **DELIST-02**: Tickers returning no price data or empty DataFrames are flagged and removed
- **DELIST-03**: Removal happens during refresh cycle, not during scans (scan uses cached universe)

#### ETF Preservation
- **ETF-01**: Existing ETF entries in `universe_data.json` survive stock universe regeneration
- **ETF-02**: ETFs are separately validated for optionability but exempt from OI threshold
- **ETF-03**: ETF list is identifiable by `asset_type` field in `universe_data.json` schema

#### Universe Size Management
- **SIZE-01**: After filtering, universe targets 500-1500 stock tickers
- **SIZE-02**: If universe falls below 500, system logs a warning (does not auto-lower threshold)
- **SIZE-03**: If universe exceeds 1500, system logs a warning (does not auto-raise threshold)
- **SIZE-04**: Universe size is reported in refresh metadata (`universe_meta.json`)

#### Scheduled Refresh
- **SCHED-01**: Weekly refresh runs automatically via APScheduler within FastAPI lifespan
- **SCHED-02**: Refresh executes the full pipeline: fetch tickers -> validate optionability -> check OI -> enrich metadata -> write JSON
- **SCHED-03**: Refresh runs on weekends when markets are closed
- **SCHED-04**: Refresh interval is configurable in settings (default: 7 days)

#### Failure Recovery
- **FAIL-01**: If refresh fails partway through, previous `universe_data.json` is preserved
- **FAIL-02**: Partial failures (some tickers fail validation) don't block the entire refresh
- **FAIL-03**: Refresh logs errors for failed tickers with reason (rate limited, no data, API error)
- **FAIL-04**: Rate limiting is enforced during batch validation (conservative 1800 req/hr with backoff)

#### Full Regeneration
- **REGEN-01**: One-time regeneration rebuilds the entire stock universe from SEC EDGAR source
- **REGEN-02**: Regeneration applies all filters: optionability, OI threshold, price/volume
- **REGEN-03**: Regeneration preserves ETF entries unchanged
- **REGEN-04**: Output maintains existing `universe_data.json` schema (`symbol`, `name`, `sector`, `market_cap_tier`, `asset_type`)
- **REGEN-05**: Regeneration can be triggered manually via CLI or API endpoint

### Non-Functional Requirements

- **Performance**: Weekly refresh can take time but should be reasonably efficient (not hours). Rate limiting at 1800 req/hr with backoff.
- **Compatibility**: Must produce the same `universe_data.json` schema so downstream consumers (`universe.py`, pipeline) work unchanged.
- **Reliability**: Atomic writes — previous universe preserved if refresh fails. Partial ticker failures don't corrupt the full dataset.
- **Observability**: Refresh logs ticker-level pass/fail with reasons. Universe size and last refresh time tracked in `universe_meta.json`.

## Success Criteria

- Zero delisted ticker errors during scans after universe regeneration
- All stock tickers in universe have 100+ total open interest
- Universe size between 500-1500 stock tickers (plus existing ETFs)
- Weekly refresh runs unattended without manual intervention
- Refresh completes without corrupting universe data on partial failures
- No changes required to downstream scan pipeline code

## Constraints & Assumptions

- **Data sources**: Free APIs only (SEC EDGAR + yfinance). Open to alternatives if faster/better, but no paid subscriptions.
- **ETFs**: Existing ETF entries preserved during stock regeneration.
- **Schema**: `universe_data.json` format unchanged (`symbol`, `name`, `sector`, `market_cap_tier`, `asset_type`).
- **Python**: 3.10+ async/await patterns, FastAPI web layer.
- **Assumption**: 100 OI threshold provides meaningful liquidity without being too restrictive. May need tuning after initial regeneration.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time universe updates during scans | Weekly cadence sufficient; real-time adds complexity and API costs |
| Paid data APIs | Prefer free sources; SEC EDGAR + yfinance stack works |
| Scan pipeline changes | This project fixes the input, not the processing |
| Historical OI time-series tracking | High complexity, requires database schema changes |
| ML-based filtering | Over-engineering; simple rules are transparent and debuggable |
| Full options chain storage | Massive storage, staleness issues; only store metadata |
| Custom universe profiles | Multi-config adds complexity; single profile sufficient for v1 |
| Penny stock inclusion | Below $5 = unreliable data and manipulation risk |
| Watchlist/manual ticker management | That system works fine as-is |

## Dependencies

- **SEC EDGAR API** — Source for US-listed ticker discovery
- **yfinance** — Optionability validation, OI data, price/volume data, metadata enrichment
- **APScheduler** — Weekly scheduled refresh (new dependency)
- **Existing modules**: `universe.py`, `universe_refresh.py`, `config.py` (Settings), `universe_data.json`, `universe_meta.json`

## v2 (Deferred)

Tracked but not in current roadmap:

- **TIER-01/02**: Multi-tier OI thresholds (bronze: 100+, silver: 500+, gold: 2000+) with adaptive auto-adjustment
- **LIQ-01/02**: 30-day average volume and bid-ask spread as additional liquidity metrics
- **CTRL-01/02/03**: Manual whitelist/blacklist that persists across refresh cycles
- **MON-01/02**: Dashboard widget for universe health and historical size tracking

## Key Decisions

| Decision | Rationale | Status |
|----------|-----------|--------|
| Full regeneration of stock universe | Too many bad tickers to fix incrementally | Pending |
| OI threshold of 100+ total | Meaningful liquidity without being too restrictive | Pending |
| Weekly refresh cadence | Balance between freshness and API usage | Pending |
| Immediate removal on validation failure | No quarantine needed — clean universe is the priority | Pending |
| Preserve ETFs, regenerate stocks only | ETF list is curated and working fine | Pending |
| Open to alternative data sources | Current per-ticker yfinance validation is slow | Pending |
