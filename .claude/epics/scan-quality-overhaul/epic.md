---
name: scan-quality-overhaul
status: backlog
created: 2026-02-12T11:06:40Z
progress: 0%
prd: .claude/prds/scan-quality-overhaul.md
github: https://github.com/jobu711/option_alpha_v2/issues/49
---

# Epic: scan-quality-overhaul

## Overview

Expand the scoring engine from 6 to ~15 technical indicators and enrich the AI debate context so agents produce specific, actionable analysis with price targets and risk/reward quantification. All new indicators use existing OHLCV data — no new API dependencies.

## Architecture Decisions

- **Pure pandas/numpy indicators**: Follow existing pattern in `indicators.py` — each function takes a DataFrame, returns `float` or `NaN`. No pandas_ta dependency for new indicators.
- **Backward-compatible config**: New weights added to `DEFAULT_SCORING_WEIGHTS` with defaults. Existing `config.json` files continue to work — missing keys get defaults.
- **No new models needed**: Existing `ScoreBreakdown`, `TickerScore`, `OptionsRecommendation` already carry all fields needed. Dashboard already renders breakdown dynamically.
- **Context enrichment in-place**: Expand `build_context()` with interpretive labels and options flow data already available from `OptionsRecommendation`. Sector name available from universe data. No new data fetching required for core enrichment.
- **Skip HV/IV ratio**: PRD notes this needs 52-week IV history we don't store. Defer to future work. The remaining 8 new indicators still exceed the 12-indicator target.

## Technical Approach

### Indicators (indicators.py)
Add 8 new functions following existing pattern: `_validate_ohlcv()`, min rows constant, return `float("nan")` on insufficient data. Update `compute_all_indicators()` to include all new indicators.

New indicators: `vwap_deviation`, `ad_trend`, `stoch_rsi`, `williams_r`, `roc`, `keltner_width`, `adx`, `supertrend`

### Scoring (config.py, composite.py, normalizer.py)
- Add 7 new entries to `DEFAULT_SCORING_WEIGHTS` (vwap_deviation, ad_trend, stoch_rsi, williams_r, roc, adx, keltner_width). Supertrend is directional (binary up/down), not weighted — used only in direction signal.
- Add corresponding entries to `INDICATOR_WEIGHT_MAP` in composite.py
- Add `keltner_width` to normalizer invert set (lower = tighter = better, like bb_width)
- Enhance `determine_direction()` to use ADX for trend strength confirmation

### AI Context (ai/context.py)
- Add interpretive labels for all indicators (e.g., "ADX: 32.4 — moderate trend")
- Add options flow summary: put/call ratio, IV context from existing `OptionsRecommendation`
- Add ATR-based risk parameters: stop-loss distance, BB-derived support/resistance
- Include sector name from universe metadata (already in `UniverseTicker.sector`)

### Agent Prompts (ai/agents.py)
- Bull: cite 3+ specific data points, identify strongest confirming indicator, state price target
- Bear: quantify downside in $/%, identify weakest bull indicator, cite specific risk scenario
- Risk: output entry price, stop-loss, profit target, risk/reward ratio, position sizing, IV assessment

## Implementation Strategy

Linear execution — each task builds on the previous:
1. Indicators first (foundation)
2. Scoring integration (wires indicators into pipeline)
3. AI context enrichment (uses new indicator data)
4. Agent prompts (uses enriched context)
5. Integration validation (end-to-end verification)

## Task Breakdown Preview

- [ ] Task 1: Implement 8 new technical indicators + unit tests
- [ ] Task 2: Wire scoring weights, composite mapping, and enhanced direction signal + tests
- [ ] Task 3: Enrich AI debate context with interpretive labels, options flow, and risk params + tests
- [ ] Task 4: Improve bull/bear/risk agent prompts per FR5 requirements + tests
- [ ] Task 5: End-to-end integration validation and backward compatibility verification

## Dependencies

- **pandas / numpy**: Already available, sufficient for all new indicators
- **scipy.stats**: Already used in normalizer
- **Existing OHLCV data**: All indicators use standard columns already fetched
- **OptionsRecommendation model**: Already carries IV, bid/ask, volume, OI — no changes needed

## Success Criteria (Technical)

- `compute_all_indicators()` returns 14 indicators (6 existing + 8 new)
- `DEFAULT_SCORING_WEIGHTS` has 14 entries summing to 1.0
- `determine_direction()` uses ADX to reduce false neutral signals
- `build_context()` output includes interpretive labels for all indicators
- Risk agent prompt explicitly requests entry/stop/target/R:R
- All existing tests pass without modification
- New tests cover each indicator (min data, NaN handling, known values)

## Estimated Effort

- Task 1 (indicators): Moderate — 8 functions following established pattern
- Task 2 (scoring): Small — config/mapping changes + direction signal update
- Task 3 (context): Moderate — context builder restructuring with new sections
- Task 4 (prompts): Small — prompt text updates
- Task 5 (integration): Small — validation and regression testing

## Tasks Created
- [ ] #50 - Implement 8 new technical indicators with unit tests (parallel: true)
- [ ] #51 - Wire scoring weights, composite mapping, and enhanced direction signal (parallel: false, depends: #50)
- [ ] #52 - Enrich AI debate context with interpretive labels and risk params (parallel: false, depends: #51)
- [ ] #53 - Improve bull/bear/risk agent prompts (parallel: false, depends: #52)
- [ ] #54 - End-to-end integration validation and backward compatibility (parallel: false, depends: #50-#53)

Total tasks: 5
Parallel tasks: 1 (Task 001 — foundation, no dependencies)
Sequential tasks: 4 (Tasks 002-005 — linear dependency chain)
Estimated total effort: 17-23 hours
