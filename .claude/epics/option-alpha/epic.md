---
name: option-alpha
status: backlog
created: 2026-02-11T00:51:59Z
progress: 0%
prd: .claude/prds/option-alpha.md
github: https://github.com/jobu711/option_alpha_v2/issues/1
---

# Epic: Option Alpha v1.0

## Overview

Build a local Python web application that scans ~3,000 optionable stocks daily for breakout setups, scores them technically, runs multi-agent LLM debate, and recommends specific options contracts. The app is a FastAPI server with Jinja2+HTMX frontend, SQLite persistence, and Ollama/Claude LLM backends.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Project layout | `src/option_alpha/` with hatchling | Modern Python packaging, clean imports |
| Web framework | FastAPI + Jinja2 + HTMX | Server-rendered, no JS build step, WebSocket support for progress |
| Database | SQLite with WAL mode | Zero-config, sufficient for single-user local tool |
| Data caching | Parquet (OHLCV) + JSON (metadata) | Fast columnar reads for price data |
| Scoring method | Weighted geometric mean | Penalizes weak individual scores vs arithmetic mean |
| AI debate flow | Sequential Bull -> Bear -> Risk | Bear needs bull thesis; risk needs both |
| Structured LLM output | instructor + Pydantic | Reliable extraction with retry/fallback |
| Options Greeks | py_vollib with manual BSM fallback | Handles edge cases in volatility calculation |
| Interactivity | HTMX + Alpine.js | Minimal JS, server-driven updates |

## Technical Approach

### Package Structure

```
src/option_alpha/
  __init__.py
  __main__.py           # Entry point: python -m option_alpha
  config.py             # Pydantic settings, defaults, config persistence
  models.py             # Shared Pydantic models (TickerScore, TradeThesis, etc.)
  data/
    fetcher.py          # yfinance batch download with threading
    universe.py         # Ticker universe management, pre-filtering
    cache.py            # Parquet/JSON caching layer
  scoring/
    indicators.py       # BB width, ATR, RSI, OBV, SMA alignment, rel volume
    normalizer.py       # Percentile normalization across universe
    composite.py        # Weighted geometric mean scoring
  catalysts/
    earnings.py         # Earnings date fetching, proximity scoring
  options/
    chains.py           # Options chain fetching and filtering
    greeks.py           # Greeks calculation (py_vollib + BSM fallback)
    recommender.py      # Contract recommendation (delta-based strike, DTE)
  ai/
    clients.py          # Ollama and Claude API clients
    agents.py           # Bull, Bear, Risk agent implementations
    context.py          # Context builder (~2000 token prompt curation)
    debate.py           # Debate pipeline orchestration
  pipeline/
    orchestrator.py     # Phase sequencing, progress reporting
    progress.py         # WebSocket progress broadcaster
  persistence/
    database.py         # SQLite connection, WAL mode, migrations
    repository.py       # CRUD operations for scan results
  web/
    app.py              # FastAPI app factory
    routes.py           # Route handlers (dashboard, detail, settings, export)
    websocket.py        # WebSocket endpoint for scan progress
    templates/          # Jinja2 templates
    static/             # CSS, minimal JS
  backtest/
    runner.py           # Historical backtesting engine
    metrics.py          # Win rate, avg return, statistical summary
```

### Data Flow

1. **Fetch** -> yfinance batch download -> Parquet cache
2. **Score** -> pandas-ta indicators -> percentile normalize -> geometric mean composite
3. **Catalysts** -> earnings dates -> exponential decay scoring -> merge into composite
4. **Options** -> top 50 chains -> Greeks calc -> liquidity filter -> contract recommendation
5. **AI Debate** -> top 10 context build -> Bull -> Bear -> Risk -> structured thesis
6. **Persist** -> SQLite store scan results, scores, theses
7. **Display** -> FastAPI serves dashboard with ranked table, detail views

### Key Integration Points

- **yfinance** for all market data (OHLCV, options chains, earnings)
- **FRED API** for risk-free rate (with configurable fallback)
- **Ollama** (local) or **Claude API** (cloud) for LLM debate
- **WebSocket** for real-time scan progress to frontend

## Implementation Strategy

Development follows the internal dependency chain from the PRD: data layer first, then scoring, catalysts, options, AI, persistence, dashboard, pipeline orchestration, and backtesting. Each phase is independently testable with mock data.

### Testing Approach
- Unit tests with mocked external APIs (yfinance, Ollama, FRED)
- Integration tests for the full pipeline with fixture data
- pytest as test framework

### Risk Mitigation
- yfinance rate limits: batched downloads with threading + retry/backoff
- LLM failures: conservative defaults (no_trade, conviction=3)
- Partial scan failures: continue processing remaining tickers, log errors

## Task Breakdown

- [ ] **Task 1: Project scaffolding & data layer** - Set up hatchling src-layout, dependencies, config system (Pydantic settings with YAML/JSON persistence), yfinance batch fetcher with threading, Parquet/JSON caching, ticker universe management, pre-filtering (FR-1.1 through FR-1.6)
- [ ] **Task 2: Technical scoring engine** - Calculate BB width, ATR, RSI, OBV, SMA alignment, relative volume via pandas-ta; percentile normalization across universe; weighted geometric mean composite scoring; full transparency storage of raw/normalized/weight/contribution (FR-2.1 through FR-2.6)
- [ ] **Task 3: Catalyst detection & options analysis** - Earnings date fetching, exponential decay proximity scoring, IV crush warning; options chain fetching for top 50, Greeks via py_vollib with BSM fallback, liquidity filtering, delta-based contract recommendation, FRED risk-free rate (FR-3.1 through FR-3.4, FR-4.1 through FR-4.6)
- [ ] **Task 4: AI multi-agent debate** - Ollama/Claude client abstraction, Bull/Bear/Risk agent implementations with structured Pydantic output via instructor, context builder (~2000 tokens), debate pipeline for top 10, retry/fallback with conservative defaults (FR-5.1 through FR-5.10)
- [ ] **Task 5: SQLite persistence layer** - Database setup with WAL mode, schema design (scan runs, ticker scores, AI theses), migrations, CRUD repository, historical query API for score trends (FR-8.1 through FR-8.4)
- [ ] **Task 6: Pipeline orchestrator** - Sequential phase execution (data -> scoring -> catalysts -> options -> AI -> persist), progress reporting model, partial failure handling, phase timing/logging (FR-7.1 through FR-7.4)
- [ ] **Task 7: Web dashboard - core views** - FastAPI app with Jinja2 templates, HTMX interactivity, scan trigger with WebSocket progress, ranked candidates table (sortable, color-coded), ticker detail panel (score breakdown, options, Greeks, AI debate), market regime header (FR-6.1 through FR-6.7, FR-6.10)
- [ ] **Task 8: Web dashboard - settings & export** - Settings page for weight/threshold/AI config with persistence, HTML report export (self-contained, date-stamped), data freshness indicator (FR-6.8, FR-6.9, US-5, US-7)
- [ ] **Task 9: Backtesting engine** - Historical data fetching, scoring engine replay over date range, breakout success measurement, statistical summary (win rate, avg return, max drawdown), configurable parameters, dashboard integration (FR-9.1 through FR-9.6)
- [ ] **Task 10: Integration testing & polish** - End-to-end pipeline test with fixtures, `__main__.py` entry point (zero-config startup), error messaging (Ollama not running, missing API keys), final validation against all acceptance criteria

## Dependencies

### External Service Dependencies
- **yfinance API** - Stock data, options chains, earnings (free, no key)
- **FRED API** - Risk-free rate (free, optional key)
- **Ollama** - Local LLM (user must install separately)
- **Anthropic Claude API** - Optional cloud LLM (requires API key)

### Python Package Dependencies
- fastapi, uvicorn, jinja2 (web stack)
- yfinance, pandas, numpy (data layer)
- pandas-ta-classic (technical indicators)
- py_vollib (options Greeks)
- instructor, pydantic (structured LLM output)
- httpx (async HTTP for Ollama/Claude)
- pytest (testing)

### Internal Task Dependencies
- Task 2 depends on Task 1 (scoring needs data layer)
- Task 3 depends on Tasks 1+2 (catalysts/options need data + scoring)
- Task 4 depends on Tasks 2+3 (AI debate needs scoring + options data)
- Task 5 is independent (can parallel with Tasks 2-4)
- Task 6 depends on Tasks 1-5 (orchestrates all phases)
- Tasks 7-8 depend on Tasks 5+6 (dashboard displays persisted results)
- Task 9 depends on Tasks 1+2 (backtesting reuses scoring engine)
- Task 10 depends on all prior tasks

## Success Criteria (Technical)

| Metric | Target |
|--------|--------|
| Scan completion rate | > 95% of tickers processed per run |
| Full scan duration | < 15 minutes for 3,000+ tickers |
| Dashboard page load | < 2 seconds |
| Ticker detail load | < 500ms |
| AI debate completion | > 90% of top-10 get valid structured theses |
| Score transparency | 100% of scores have per-indicator breakdown |
| Test coverage | Unit tests for all modules with mocked externals |

## Estimated Effort

- **Overall:** ~8-10 focused implementation sessions
- **Critical path:** Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 6 -> Task 7
- **Parallelizable:** Task 5 (persistence) can be built alongside Tasks 2-4
- **Highest risk:** AI debate quality (Task 4) and yfinance reliability (Task 1)

## Tasks Created
- [ ] #4 - Project scaffolding & data layer (parallel: false)
- [ ] #7 - Technical scoring engine (parallel: false)
- [ ] #10 - Catalyst detection & options analysis (parallel: false)
- [ ] #11 - AI multi-agent debate system (parallel: false)
- [ ] #2 - SQLite persistence layer (parallel: true)
- [ ] #5 - Pipeline orchestrator (parallel: false)
- [ ] #8 - Web dashboard - core views (parallel: false)
- [ ] #3 - Web dashboard - settings & export (parallel: false)
- [ ] #6 - Backtesting engine (parallel: true)
- [ ] #9 - Integration testing & polish (parallel: false)

Total tasks: 10
Parallel tasks: 2 (#2, #6)
Sequential tasks: 8
Estimated total effort: 72-92 hours
