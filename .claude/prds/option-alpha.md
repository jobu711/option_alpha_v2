---
name: option-alpha
description: AI-powered local web app that scans optionable stocks daily for breakout setups, scores them technically, runs LLM debate, and recommends options contracts
status: backlog
created: 2026-02-11T00:29:41Z
---

# PRD: Option Alpha v1.0

## Executive Summary

Option Alpha is an AI-powered local Python web application that scans all optionable stocks daily for catalyst-driven breakout setups. It scores candidates using a consolidation/squeeze-weighted technical analysis system, then uses multi-agent LLM debate (Bull/Bear/Risk agents) to produce trade theses and specific options contract recommendations with full Greek analysis.

The tool runs as a local FastAPI web app with a browser-based UI, providing an interactive dashboard for triggering scans, viewing ranked candidates, drilling into individual tickers, and reviewing AI-generated trade theses. Historical results are persisted in SQLite for day-over-day comparison, and static HTML reports can be generated for archiving and sharing.

**Value Proposition:** Replace hours of manual pre-market screening with a single automated scan that surfaces the highest-probability options setups with AI-generated conviction ratings.

## Problem Statement

Options traders face an overwhelming universe of 3,000+ optionable stocks daily. Identifying high-probability breakout setups requires:

1. **Screening thousands of tickers** for technical consolidation/squeeze patterns
2. **Cross-referencing catalysts** (earnings, events) with technical setups
3. **Analyzing options chains** for the best contract (strike, expiration, Greeks)
4. **Forming a trade thesis** weighing bull/bear arguments and risk

This process takes 2-3 hours manually and is prone to recency bias, confirmation bias, and missed opportunities. A systematic, AI-augmented approach can surface better setups faster while forcing structured analysis of both sides of every trade.

**Why now:** Local LLM capabilities (Ollama) have reached sufficient quality for structured financial analysis, and cloud APIs (Claude) provide an optional upgrade path. Python's data ecosystem (yfinance, pandas, py_vollib) makes the full pipeline feasible as a single local tool.

## User Stories

### Primary Persona: Active Retail Options Trader

An active retail trader who trades options 3-5 times per week, primarily directional plays on technical breakouts with catalyst backing. Comfortable with terminal/browser tools, wants data-driven setups rather than gut feeling.

### User Stories

**US-1: Morning Scan**
> As a trader, I want to open the web dashboard and trigger a full universe scan so I can see today's top breakout candidates before market open.

Acceptance Criteria:
- Can trigger scan from the web UI with a single click
- Scan runs all phases (data fetch, scoring, catalyst, options, AI debate)
- Progress is displayed in real-time (current phase, tickers processed)
- Scan completes within 15 minutes for 3,000+ tickers
- Results are displayed automatically when scan finishes

**US-2: Ranked Candidates Review**
> As a trader, I want to see candidates ranked by composite score with color-coded direction indicators so I can quickly identify the best setups.

Acceptance Criteria:
- Candidates displayed in a sortable table ranked by composite score
- Color coding: green (bullish), red (bearish), yellow (neutral)
- Key columns visible: ticker, composite score, direction, catalyst date, conviction
- Can sort by any column
- Top 10 candidates highlighted

**US-3: Ticker Deep-Dive**
> As a trader, I want to drill into a specific ticker to see the full score breakdown, options recommendations, and AI debate so I can make an informed trade decision.

Acceptance Criteria:
- Click a ticker to open detail view
- Score breakdown shows each indicator's raw value, normalized score, weight, and contribution
- Options section shows recommended contract (strike, expiry, direction) with all Greeks
- AI debate section shows Bull thesis, Bear counter-arguments, and Risk agent synthesis
- Conviction score (1-10) and trade/no-trade recommendation clearly displayed

**US-4: Historical Comparison**
> As a trader, I want to see how a ticker's score has changed over recent days so I can identify emerging setups and strengthening patterns.

Acceptance Criteria:
- Ticker detail shows score history (last 7-30 days where available)
- Visual trend indicator (score rising/falling/stable)
- Previous scan results accessible from the dashboard

**US-5: HTML Report Export**
> As a trader, I want to export today's scan results as a static HTML report so I can save it for review or share with trading partners.

Acceptance Criteria:
- One-click export from the dashboard
- HTML report includes: ranked table, top 10 detail breakdowns, AI debate summaries
- Report is self-contained (no external dependencies)
- Saved with date-stamped filename

**US-6: Backtesting Validation**
> As a trader, I want to backtest the scoring system against historical data so I can validate that high scores correlate with actual breakouts.

Acceptance Criteria:
- Can run backtest over configurable date range
- Measures: win rate (did price move X% in direction within Y days), average return, max drawdown
- Results displayed with statistical summary
- Can compare different scoring weight configurations

**US-7: Configuration**
> As a trader, I want to adjust scoring weights, filter thresholds, and AI settings from the web UI so I can tune the system without editing config files.

Acceptance Criteria:
- Settings page accessible from dashboard
- Can modify: indicator weights, minimum score threshold, DTE range, liquidity filters
- Can toggle AI backend (Ollama / Claude API)
- Changes persist across sessions
- Can reset to defaults

## Requirements

### Functional Requirements

#### FR-1: Data Layer
- **FR-1.1:** Fetch daily OHLCV data for 3,000+ optionable stocks via yfinance
- **FR-1.2:** Batch download with threading for performance
- **FR-1.3:** Cache OHLCV data in Parquet format, metadata in JSON
- **FR-1.4:** Pre-filter tickers by minimum volume, price, and optionability
- **FR-1.5:** Curated ticker universe maintained as configuration
- **FR-1.6:** Error handling with retry/backoff for API failures

#### FR-2: Technical Scoring Engine
- **FR-2.1:** Calculate indicators: Bollinger Band width, ATR, RSI, OBV, SMA alignment, relative volume
- **FR-2.2:** Squeeze/consolidation detection via weighted composite of BB + ATR + volume signals
- **FR-2.3:** Configurable scoring weights per indicator
- **FR-2.4:** Percentile normalization (0-100) across the universe per indicator
- **FR-2.5:** Full transparency: store raw, normalized, weight, and contribution per indicator per ticker
- **FR-2.6:** Weighted geometric mean for composite scoring (penalizes poor individual scores)

#### FR-3: Catalyst Detection
- **FR-3.1:** Fetch next earnings date per ticker via yfinance
- **FR-3.2:** Catalyst proximity scoring with exponential decay (3/7/14 day windows)
- **FR-3.3:** IV crush warning flag for earnings within 7 days
- **FR-3.4:** Catalyst weight integrated into composite score (default weight: 0.25)

#### FR-4: Options Analysis
- **FR-4.1:** Fetch options chains for top 50 scored candidates
- **FR-4.2:** Calculate Greeks (delta, gamma, theta, vega) via py_vollib with manual BSM fallback
- **FR-4.3:** Liquidity filtering: open interest > 100, bid-ask spread < 10% of mid, volume > 0
- **FR-4.4:** Contract recommendation: delta-based strike selection, 30-60 DTE targeting
- **FR-4.5:** Direction derived from technical thesis (bullish = calls, bearish = puts)
- **FR-4.6:** Risk-free rate from FRED API with configurable fallback default

#### FR-5: AI Multi-Agent Debate
- **FR-5.1:** Ollama client integration with connection verification
- **FR-5.2:** Optional Claude API integration (configurable via API key)
- **FR-5.3:** Bull agent: structured bullish analysis from scoring + options data
- **FR-5.4:** Bear agent: counter-arguments given bull thesis
- **FR-5.5:** Risk agent: synthesis, conviction score (1-10), trade/no-trade recommendation
- **FR-5.6:** Context builder: curate ~2000 token prompt from scoring + options data
- **FR-5.7:** Structured output via Pydantic models (TradeThesis, AgentResponse)
- **FR-5.8:** Retry/fallback: 3 retries on parse failure, plain-text fallback
- **FR-5.9:** Conservative defaults on failure: no_trade, conviction=3
- **FR-5.10:** Top-10 candidate analysis pipeline

#### FR-6: Web Dashboard
- **FR-6.1:** FastAPI backend with Jinja2 templates and HTMX for interactivity
- **FR-6.2:** Scan trigger button with real-time progress display
- **FR-6.3:** Ranked candidates table (sortable, color-coded by direction)
- **FR-6.4:** Ticker detail panel: score breakdown, options chain, Greeks, AI debate
- **FR-6.5:** Market regime header (VIX level, SPY trend)
- **FR-6.6:** Data freshness indicator (when was last scan)
- **FR-6.7:** Score history visualization per ticker (from SQLite)
- **FR-6.8:** Settings page for configuration management
- **FR-6.9:** Static HTML report export (self-contained, date-stamped)
- **FR-6.10:** Color coding: green (bullish), red (bearish), yellow (neutral/caution)

#### FR-7: Pipeline Orchestration
- **FR-7.1:** Orchestrator runs all phases in sequence: data -> scoring -> catalysts -> options -> AI -> results
- **FR-7.2:** Real-time progress reporting via WebSocket to dashboard
- **FR-7.3:** Partial failure handling: continue with available data if a phase partially fails
- **FR-7.4:** Phase timing and logging

#### FR-8: Persistence & History
- **FR-8.1:** SQLite database for storing scan results
- **FR-8.2:** Store per-ticker scores, recommendations, and AI theses per scan run
- **FR-8.3:** Historical query API: score trends, previous recommendations
- **FR-8.4:** Database migrations for schema changes

#### FR-9: Backtesting
- **FR-9.1:** Fetch historical OHLCV data for backtesting period
- **FR-9.2:** Run scoring engine against historical data
- **FR-9.3:** Measure breakout success: did price move X% in predicted direction within Y days
- **FR-9.4:** Statistical summary: win rate, average return, Sharpe-like metric
- **FR-9.5:** Configurable parameters: lookback period, breakout threshold, holding period
- **FR-9.6:** Results displayed in dashboard with visual charts

### Non-Functional Requirements

#### NFR-1: Performance
- Full universe scan (3,000+ tickers) completes within 15 minutes
- Dashboard page load under 2 seconds
- Ticker detail panel loads under 500ms
- AI debate for top 10 tickers completes within 5 minutes (Ollama) / 2 minutes (Claude)

#### NFR-2: Reliability
- Graceful handling of yfinance API rate limits and outages
- Scan continues if individual tickers fail (logs errors, processes remainder)
- AI fallback to conservative defaults on LLM failures
- SQLite WAL mode for concurrent read/write safety

#### NFR-3: Security
- Local-only by default (binds to 127.0.0.1)
- API keys stored in environment variables or local config, never in code
- No external data transmission except to data APIs (yfinance, FRED) and optional LLM API

#### NFR-4: Usability
- Zero-config startup: `python -m option_alpha` opens browser to dashboard
- All configuration accessible from web UI
- Clear error messages when Ollama is not running or API keys are missing

#### NFR-5: Maintainability
- Python package with clean module separation per phase
- Pydantic models for all data structures
- Type hints throughout
- Comprehensive test suite with mock-based testing for deterministic CI

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Scan completion rate | > 95% of tickers processed per run | Logged success/failure counts |
| Scan duration | < 15 minutes for full universe | Pipeline timing logs |
| Scoring transparency | 100% of scores have per-indicator breakdown | Unit test validation |
| AI debate completion | > 90% of top-10 get valid structured theses | Parse success rate logging |
| Backtest win rate | > 55% for top-scored candidates (validation, not guarantee) | Backtest results |
| Dashboard load time | < 2 seconds | Manual verification |
| User workflow time | < 5 minutes from scan start to trade decision | User feedback |

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Rich data/ML ecosystem |
| Package manager | hatchling (src-layout) | Modern Python packaging |
| Web framework | FastAPI | Async support, WebSockets for progress |
| Templates | Jinja2 + HTMX | Server-rendered, no JS build step |
| Styling | Alpine.js (minimal JS) | Lightweight interactivity |
| Data fetching | yfinance | Free, comprehensive stock/options data |
| Technical indicators | pandas-ta-classic | Handles edge cases, well-maintained |
| Greeks | py_vollib | BSM calculations with dividend support |
| LLM (local) | Ollama | Free, private, runs locally |
| LLM (cloud) | Anthropic Claude API | Optional higher-quality analysis |
| Structured LLM output | instructor | Pydantic model extraction with retry |
| Database | SQLite | Zero-config, file-based, sufficient for local use |
| Caching | Parquet (OHLCV), JSON (metadata) | Fast columnar reads for price data |
| Testing | pytest | Standard Python testing |

## Constraints & Assumptions

### Constraints
- **Data source:** yfinance is free but rate-limited and occasionally unreliable; no paid data feed
- **Historical IV:** yfinance lacks historical implied volatility data; IV percentile ranking deferred
- **Local compute:** AI debate quality depends on user's hardware (Ollama) or optional API key (Claude)
- **Market hours:** Data is end-of-day; no real-time intraday scanning

### Assumptions
- User has Python 3.11+ installed
- User has Ollama installed and running (or provides Claude API key)
- User has reliable internet for yfinance data fetching
- User runs the tool before market open (pre-market workflow)
- 3,000+ optionable tickers is a sufficient universe (covers all major US exchanges)

## Out of Scope

The following are explicitly **not** included in v1.0:

- **Real-time streaming data** - This is an end-of-day tool, not a live trading platform
- **Order execution / broker integration** - No automated trade placement
- **IV percentile ranking** - Blocked by lack of historical IV data from yfinance
- **Multi-leg options strategies** - v1.0 focuses on single-leg directional plays (calls/puts)
- **Options P&L simulation** - Backtesting covers signal accuracy only, not options-specific P&L
- **Portfolio tracking** - No tracking of open positions or portfolio-level risk
- **Mobile app** - Browser-based web app only
- **Multi-user support** - Single-user local tool
- **Paid data feeds** - yfinance only; no Bloomberg, Polygon, or similar integrations
- **Intraday scanning** - No support for scanning during market hours

## Dependencies

### External Dependencies
- **yfinance API** - Stock data, options chains, earnings calendar (free, no key required)
- **FRED API** - Risk-free rate for Greeks calculation (free, optional API key)
- **Ollama** - Local LLM inference (must be installed separately)
- **Anthropic API** - Optional cloud LLM (requires API key from console.anthropic.com)

### Python Dependencies
- FastAPI, uvicorn, Jinja2 (web stack)
- yfinance, pandas, numpy (data layer)
- pandas-ta-classic (technical indicators)
- py_vollib (options Greeks)
- instructor, pydantic (structured LLM output)
- httpx (async HTTP for Ollama/Claude)
- sqlite3 (stdlib, persistence)
- pytest (testing)

### Internal Dependencies (Phase Order)
1. Data Layer (foundation for everything)
2. Technical Scoring (requires data)
3. Catalyst Detection (requires data + scoring)
4. Options Analysis (requires data + scoring)
5. AI Debate (requires scoring + options data)
6. Persistence Layer (stores results from all phases)
7. Web Dashboard (displays all phase outputs)
8. Pipeline Orchestration (wires all phases)
9. Backtesting (requires scoring engine + historical data)

## Appendix: Key Design Decisions

| Decision | Choice | Alternative Considered | Rationale |
|----------|--------|----------------------|-----------|
| Composite scoring | Weighted geometric mean | Arithmetic mean | Penalizes poor individual scores; a ticker must be strong across multiple indicators |
| Catalyst scoring | Exponential decay | Linear decay | More realistic: imminent catalysts are disproportionately more important |
| AI debate flow | Sequential (Bull -> Bear -> Risk) | Parallel agents | Bear agent needs bull thesis to counter-argue; risk agent needs both |
| Parse failure default | no_trade, conviction=3 | Skip ticker | Conservative approach prevents false confidence from bad LLM output |
| Frontend | Server-rendered (Jinja2 + HTMX) | React SPA | Simpler stack, no JS build step, sufficient for local single-user tool |
| Database | SQLite | PostgreSQL | Zero-config, file-based, appropriate for local single-user workload |
