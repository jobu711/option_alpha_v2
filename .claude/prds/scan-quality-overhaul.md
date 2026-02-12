---
name: scan-quality-overhaul
description: Expand technical indicators and enrich AI debate context for deeper, more actionable analysis
status: backlog
created: 2026-02-12T01:41:46Z
---

# PRD: scan-quality-overhaul

## Executive Summary

The scan pipeline currently uses 6 technical indicators (BB width, ATR%, RSI, OBV trend, SMA alignment, relative volume) and feeds a minimal ~2000-token context to AI debate agents. This produces shallow, generic debate output that lacks actionable specificity. This overhaul adds 8-10 new indicators across volume, momentum, volatility, and trend categories, and enriches the debate context with sector/market awareness, options flow analysis, and explicit risk quantification requirements.

## Problem Statement

1. **Limited indicator coverage**: Only 6 indicators drive scoring. Missing momentum oscillators (Stochastic RSI, Williams %R), volatility comparisons (Keltner Channels, historical vs implied vol), and trend structure signals (ADX, Ichimoku) that traders rely on for conviction.

2. **Shallow debate output**: Agent prompts are generic ("find the strongest reasons to be optimistic"). They don't instruct agents to reference specific indicator values, quantify risk/reward, or consider market regime. The context builder doesn't include sector benchmarks or options flow data.

3. **Direction signal relies on only 2 indicators**: `determine_direction()` uses only RSI + SMA alignment. Adding ADX for trend strength and momentum confirmation would reduce false signals.

## User Stories

### Trader reviewing scan results
**As a** trader reviewing the top-ranked tickers,
**I want** the composite score to reflect a broader set of technical signals,
**So that** I can trust the ranking represents genuinely strong setups rather than artifacts of a narrow indicator set.

**Acceptance criteria:**
- At least 12 indicators contribute to composite score
- Scoring weights are configurable per-indicator via settings
- New indicators are visible in the score breakdown on the dashboard

### Trader reading AI debate output
**As a** trader reading the bull/bear/risk debate,
**I want** agents to reference specific data values, provide price targets, and quantify risk/reward,
**So that** I get actionable trade ideas rather than generic commentary.

**Acceptance criteria:**
- Bull/Bear agents cite specific indicator values from context
- Risk agent outputs specific entry price, stop-loss, and profit target
- Risk agent calculates risk/reward ratio based on options data
- Debate context includes sector performance relative to SPY

### Trader evaluating options positioning
**As a** trader evaluating recommended options contracts,
**I want** the AI debate to consider unusual options activity and IV context,
**So that** I understand whether the options market agrees with the technical thesis.

**Acceptance criteria:**
- Context includes put/call ratio when available
- Context includes IV rank or IV percentile vs historical
- Risk agent comments on whether options pricing supports the thesis

## Requirements

### Functional Requirements

#### FR1: New Technical Indicators

**Volume-based:**
- VWAP deviation (% distance from VWAP)
- Accumulation/Distribution line trend (slope over 20 bars, like OBV trend)

**Momentum:**
- Stochastic RSI (K and D lines, 14-period)
- Williams %R (14-period)
- Rate of Change (ROC, 12-period)

**Volatility:**
- Keltner Channel width (20-period, 1.5x ATR)
- Historical volatility (20-day annualized) vs implied volatility ratio (when IV available from options data)

**Trend/Structure:**
- ADX (14-period, trend strength 0-100)
- Supertrend (10-period, 3x ATR multiplier)

All indicators follow existing pattern: pure pandas/numpy in `indicators.py`, return `float` or `float("nan")` on insufficient data.

#### FR2: Updated Scoring Weights

Add new indicators to `DEFAULT_SCORING_WEIGHTS` in `config.py` and `INDICATOR_WEIGHT_MAP` in `composite.py`. Suggested initial weights:

| Indicator | Weight | Rationale |
|-----------|--------|-----------|
| bb_width | 0.12 | Squeeze detection (existing) |
| atr_percent | 0.08 | Volatility context (existing) |
| rsi | 0.08 | Momentum (existing) |
| obv_trend | 0.06 | Volume confirmation (existing) |
| sma_alignment | 0.08 | Trend (existing) |
| relative_volume | 0.06 | Volume context (existing) |
| catalyst_proximity | 0.15 | Earnings catalyst (existing) |
| stoch_rsi | 0.06 | Momentum oscillator (new) |
| williams_r | 0.04 | Overbought/oversold (new) |
| roc | 0.04 | Price momentum (new) |
| adx | 0.08 | Trend strength (new) |
| keltner_width | 0.05 | Squeeze confirmation (new) |
| vwap_deviation | 0.05 | Intraday context (new) |
| ad_trend | 0.05 | Smart money flow (new) |

Weights sum to 1.0. All configurable via `config.json`.

#### FR3: Enhanced Direction Signal

Update `determine_direction()` to incorporate ADX:
- Current: RSI + SMA alignment only
- New: RSI + SMA alignment + ADX confirmation
- ADX > 25 strengthens directional conviction
- ADX < 20 biases toward NEUTRAL regardless of RSI/SMA

#### FR4: Enriched Debate Context

Expand `build_context()` in `ai/context.py` to include:

1. **Sector context**: Add sector name, sector 1-week and 1-month return vs SPY (requires fetching sector ETF data — XLK, XLF, XLE, etc.)
2. **Expanded indicator details**: Include all new indicator values with interpretive labels (e.g., "ADX: 32.4 — moderate trend")
3. **Options flow summary**: Put/call ratio, IV rank (percentile of current IV vs 52-week range), unusual volume flags
4. **Risk parameters**: ATR-based stop-loss distance, recent support/resistance levels from Bollinger Bands

Target context size: ~2500-3000 tokens (up from ~2000).

#### FR5: Improved Agent Prompts

**Bull agent**: Instruct to cite at least 3 specific data points, identify the strongest confirming indicator, and state a price target with timeframe.

**Bear agent**: Instruct to quantify downside risk in dollar/percentage terms, identify the weakest indicator in the bull case, and cite a specific risk scenario.

**Risk agent**: Instruct to output:
- Specific entry price (or range)
- Stop-loss level (ATR-based)
- Profit target (based on technical levels)
- Risk/reward ratio
- Position sizing suggestion (as % of portfolio, conservative)
- Whether options IV supports or undermines the thesis

### Non-Functional Requirements

- **Performance**: New indicators must not add > 2 seconds to scoring phase for 500 tickers
- **Backward compatibility**: Existing config.json files work without changes (new weights get defaults)
- **Test coverage**: Each new indicator needs unit tests matching existing pattern (min data, NaN handling, known-value verification)
- **Context size**: Enriched context must stay under 4000 tokens to fit small LLM context windows (llama3.1:8b has 8K context)

## Success Criteria

- **Indicator diversity**: Composite score uses 12+ indicators (up from 6)
- **Debate specificity**: Risk agent output includes numeric price targets and stop-loss levels in >80% of debates
- **Direction accuracy**: Manual review of 20 tickers shows direction signal matches visual chart assessment in >70% of cases (up from current baseline — establish baseline first)
- **User satisfaction**: Debate output is actionable enough to inform a real trade decision without external chart analysis

## Constraints & Assumptions

- **Assumes OHLCV data is sufficient**: All new indicators use standard OHLCV columns already fetched by yfinance. No additional API calls needed for indicators themselves.
- **Sector data requires additional fetches**: Sector ETF prices (XLK, XLF, etc.) need fetching. Can be cached with existing parquet cache infrastructure.
- **IV rank requires historical IV data**: Need 52-week IV history for IV rank calculation. May need to store rolling IV values or approximate from options chain snapshots.
- **Small LLM limitations**: llama3.1:8b may not consistently produce structured risk quantification. Prompts should include explicit examples. Consider recommending a larger model (llama3.1:70b or Claude) for best results.
- **No live data**: All indicators computed from cached daily OHLCV. VWAP uses daily approximation (typical price × volume), not intraday.

## Out of Scope

- **Real-time / intraday indicators**: No streaming data or sub-daily analysis
- **Machine learning scoring**: No replacing geometric mean with ML-based ranking
- **Custom indicator builder**: No user-defined indicator formulas
- **Backtesting integration**: Not validating indicator effectiveness via backtesting in this PRD
- **Alternative LLM backends**: No adding new AI providers (OpenAI, Gemini, etc.)
- **Chart image analysis**: No sending chart screenshots to vision models

## Dependencies

- **pandas / numpy**: Already in dependencies, sufficient for all new indicators
- **scipy.stats**: Already used in normalizer for percentile ranking
- **yfinance**: Already fetches OHLCV. Sector ETF data uses same fetcher
- **Existing test infrastructure**: pytest with mocking patterns already established
- **Options chain data**: Already fetched in pipeline phase 4; IV data available from `OptionsRecommendation` model
