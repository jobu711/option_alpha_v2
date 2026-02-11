---
name: fix-business-logic
description: Fix systematic neutral bias across scoring, options, and AI debate pipeline phases
status: backlog
created: 2026-02-11T21:38:56Z
---

# PRD: fix-business-logic

## Executive Summary

The Option Alpha scanner's 6-phase pipeline is producing **100% neutral results** for every ticker, regardless of actual market conditions. Root-cause analysis reveals a cascading failure originating in the data fetch layer and amplified by overly strict business logic in scoring, options filtering, and AI debate phases. This PRD defines the fixes, diagnostic tooling, and validation criteria needed to restore directional signal accuracy.

## Problem Statement

**Every scan returns neutral** — even for obviously trending stocks (e.g., RSI 70 with strong uptrend). This renders the entire scanner unusable.

### Root Causes Identified

| # | Phase | Root Cause | Impact |
|---|-------|-----------|--------|
| 1 | **Data Fetch** | `fetcher.py` fetches 6 months (~128 bars); `sma_direction()` requires >= 200 bars and returns "neutral" when data is insufficient | **ALL tickers classified neutral** — this is the primary smoking gun |
| 2 | **Scoring** | `determine_direction()` requires BOTH RSI AND SMA alignment with strict SMA ordering (20 > 50 > 200) — no tolerance, no partial signals | Even with sufficient data, many legitimately trending stocks score neutral |
| 3 | **Options** | `recommend_contract()` returns `None` for NEUTRAL direction — no options recommendation generated | Zero options recommendations produced |
| 4 | **AI Debate** | Risk agent prompt says "be conservative: when in doubt, recommend no trade"; all fallbacks default to neutral; no conviction rubric | AI layer adds further neutral bias on top of already-broken upstream signals |

### Why This Matters Now

The scanner produces zero actionable output. Users cannot identify any bullish or bearish setups. The tool is effectively non-functional for its core purpose.

## User Stories

### US-1: Scanner Returns Directional Signals
**As a** trader using the scanner,
**I want** scans to correctly classify tickers as bullish, bearish, or neutral based on technical indicators,
**So that** I can identify actionable trade opportunities.

**Acceptance Criteria:**
- Tickers with RSI > 55 and price above rising SMAs are classified BULLISH
- Tickers with RSI < 45 and price below falling SMAs are classified BEARISH
- Mixed signals produce NEUTRAL — but not ALL signals
- A scan of 50+ tickers in a trending market produces at least some non-neutral results

### US-2: Options Recommendations Generated for Directional Tickers
**As a** trader reviewing scan results,
**I want** bullish/bearish tickers to include options contract recommendations,
**So that** I can evaluate specific trade setups.

**Acceptance Criteria:**
- Every BULLISH ticker gets a call recommendation (or explicit "no liquid options" reason)
- Every BEARISH ticker gets a put recommendation (or explicit reason)
- NEUTRAL tickers are clearly labeled as "no directional signal"

### US-3: AI Debate Produces Varied Verdicts
**As a** trader reviewing AI analysis,
**I want** the bull/bear/risk debate to produce honest assessments rather than defaulting to "no trade",
**So that** I get genuine AI-powered trade evaluation.

**Acceptance Criteria:**
- Risk agent produces bullish/bearish verdicts when technical signals are strong
- Conviction scores span the full 1-10 range based on signal strength
- Fallback behavior is clearly logged, not silent

### US-4: Pipeline Diagnostics for Score Transparency
**As a** developer or power user,
**I want** visibility into each phase's intermediate results,
**So that** I can diagnose silent failures and verify the pipeline is working correctly.

**Acceptance Criteria:**
- Each pipeline phase logs key decision points (e.g., "AAPL: RSI=65, SMA_dir=bullish -> BULLISH")
- Score breakdown is visible per-ticker in scan results
- Data sufficiency warnings are logged (e.g., "AAPL: only 128 bars, need 200 for SMA200")

## Requirements

### Functional Requirements

#### FR-1: Fix Data Fetch Period (Critical)
- **File**: `src/option_alpha/data/fetcher.py`
- Change default fetch period from `"6mo"` to `"1y"` (or configurable)
- Ensure all tickers receive >= 200 trading days of data for SMA200 calculation
- Add configurable `data_fetch_period` setting in `config.py`

#### FR-2: Relax Direction Classification Logic
- **File**: `src/option_alpha/scoring/composite.py`
- Make direction determination more nuanced:
  - Use available SMAs when fewer than 200 bars exist (e.g., SMA20 vs SMA50 only)
  - Allow RSI-only direction when SMA data is insufficient
  - Consider a weighted scoring approach instead of strict AND logic
- Add configurable direction thresholds (RSI bullish/bearish cutoffs)

#### FR-3: Improve SMA Direction Flexibility
- **File**: `src/option_alpha/scoring/indicators.py`
- When < 200 bars available, fall back to shorter SMAs (SMA20 vs SMA50)
- Add tolerance for near-alignment (e.g., SMAs within 0.5% count as aligned)
- Return a confidence level alongside direction (high/medium/low)

#### FR-4: Options Phase — Handle Weak Signals
- **File**: `src/option_alpha/options/recommender.py`
- Consider generating options recommendations for high-score NEUTRAL tickers (e.g., straddles/strangles for neutral-but-high-volatility setups)
- At minimum, log why each ticker was skipped

#### FR-5: Rebalance AI Debate Prompts
- **File**: `src/option_alpha/ai/agents.py`
- Remove or soften "be conservative" instruction in risk agent prompt
- Add conviction scoring rubric (e.g., "8-10: strong directional signal with multiple confirming indicators")
- Instruct risk agent to weight the pre-computed DIRECTION_SIGNAL from technical analysis
- Ensure fallback responses indicate failure, not a genuine "neutral" verdict

#### FR-6: Add Pipeline Diagnostics
- Add structured logging at key decision points in each phase
- Log direction classification reasoning per ticker
- Log data sufficiency warnings
- Log options filtering outcomes (why each ticker was included/excluded)
- Include score breakdown in scan result data model

### Non-Functional Requirements

#### NFR-1: Performance
- Increasing data fetch period from 6mo to 1y should not significantly impact scan time (yfinance fetches are already the bottleneck; 1y vs 6mo is marginal)
- Diagnostic logging must not impact scan throughput

#### NFR-2: Backward Compatibility
- Existing config.json files should work without changes (new settings use sensible defaults)
- Database schema unchanged — existing scan results remain valid
- Parquet cache should auto-refresh with longer period data

#### NFR-3: Testability
- All direction logic changes must have unit tests with known-bullish and known-bearish datasets
- AI prompt changes must have integration tests verifying non-neutral outputs

## Success Criteria

| Metric | Current State | Target |
|--------|--------------|--------|
| % of tickers classified non-neutral in trending market | 0% | >= 30% |
| Options recommendations generated per scan | 0 | >= 5 for top-50 scan |
| AI debate verdicts that are non-neutral | 0% | >= 40% when upstream direction is non-neutral |
| Data sufficiency (tickers with >= 200 bars) | 0% | 100% |
| Direction classification logged per ticker | No | Yes |

## Constraints & Assumptions

### Constraints
- Must remain compatible with Python 3.11+
- Cannot add new external dependencies (use existing stack)
- Must not break existing 745+ tests
- yfinance API rate limits apply — 1y data fetch is acceptable, 5y may throttle

### Assumptions
- yfinance reliably returns >= 200 trading days for `period="1y"`
- Most S&P 500 and large-cap tickers have sufficient options liquidity
- LLM backends (Ollama/Claude) can follow updated prompt instructions without model changes
- Existing scoring weights are reasonable — the issue is direction classification, not composite scores

## Out of Scope

- Changing the weighted geometric mean scoring algorithm
- Adding new technical indicators
- Modifying the pipeline phase order or architecture
- Changing the database schema
- Adding a web UI for diagnostics (logging only for now)
- Backtesting the direction classification accuracy
- Options strategy recommendations beyond single-leg calls/puts

## Dependencies

### Internal
- `config.py` Settings class — new configurable fields needed
- `models.py` — may need direction confidence field
- Test suite — all 23 test files must pass after changes

### External
- yfinance API — data availability for extended periods
- LLM backend (Ollama/Claude) — prompt responsiveness to updated instructions
