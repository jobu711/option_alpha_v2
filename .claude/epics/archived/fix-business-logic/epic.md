---
name: fix-business-logic
status: completed
created: 2026-02-11T22:00:15Z
completed: 2026-02-11T23:45:00Z
progress: 100%
prd: .claude/prds/fix-business-logic.md
github: https://github.com/jobu711/option_alpha_v2/issues/41
---

# Epic: fix-business-logic

## Overview

Fix cascading neutral bias across the scan pipeline. The root cause is a data fetch period of 6 months (~128 bars) that prevents SMA200 calculation, causing `sma_direction()` to always return "neutral". This propagates through scoring, options filtering, and AI debate — producing zero actionable output. The fix touches 4 pipeline phases: data fetch, scoring, options, and AI debate.

## Architecture Decisions

- **No new dependencies** — all fixes use existing pandas/numpy/pydantic stack
- **Configurable over hardcoded** — new thresholds added to `Settings` class so users can tune direction sensitivity
- **Graceful degradation over hard cutoffs** — SMA direction falls back to shorter-period SMAs when data is insufficient rather than returning "neutral"
- **Logging over UI** — diagnostics added via Python `logging` module at existing log levels; no web UI changes

## Technical Approach

### Phase 1: Data Layer Fix (Root Cause)

**Files**: `config.py`, `fetcher.py`, `orchestrator.py`

- Add `data_fetch_period: str = "1y"` to `Settings`
- Change `fetch_batch()` and `fetch_single()` defaults from `"6mo"` to `"1y"`
- Wire `settings.data_fetch_period` through `orchestrator._phase_data_fetch()` → `fetch_batch(period=...)`
- Existing parquet cache auto-refreshes (18-hour TTL) so stale 6mo data resolves on next scan

### Phase 2: Scoring Logic Fix

**Files**: `indicators.py`, `composite.py`

- **`sma_direction()`**: Remove hard 200-bar cutoff. Gracefully degrade:
  - >= 200 bars: use SMA20/50/200 (current behavior)
  - >= 50 bars: fall back to SMA20 vs SMA50
  - < 50 bars: return "neutral" (genuinely insufficient data)
- **`determine_direction()`**: Relax strict AND requirement:
  - Strong signal: RSI agrees with SMA direction → BULLISH/BEARISH
  - RSI-only signal: when SMA is "neutral" but RSI is extreme (>60 or <40), use RSI-only direction
  - Add configurable thresholds: `direction_rsi_bullish` (default 50), `direction_rsi_bearish` (default 50)

### Phase 3: Options Phase Logging

**File**: `recommender.py`

- Add info-level log line in `recommend_contract()` explaining why each ticker was skipped (neutral direction, no liquid contracts, no delta match)
- Add summary log in `recommend_for_scored_tickers()` showing direction distribution of input tickers

### Phase 4: AI Debate Prompt Rebalance

**File**: `agents.py`

- Soften risk agent prompt: replace "Be conservative: when in doubt, recommend no trade with low conviction" with balanced guidance that respects the pre-computed direction signal
- Add conviction scoring rubric to risk agent prompt (1-3: weak/conflicting signals, 4-6: moderate with mixed indicators, 7-10: strong with confirming technicals)
- Mark fallback responses with a flag/note so they're distinguishable from genuine neutral verdicts

### Phase 5: Pipeline Diagnostics

**Files**: `composite.py`, `indicators.py`, `orchestrator.py`

- Log direction classification reasoning per ticker: `"AAPL: RSI=65.2, SMA_dir=bullish(20>50>200) -> BULLISH"`
- Log data sufficiency: `"AAPL: 253 bars (sufficient for SMA200)"` or `"AAPL: 128 bars (using SMA20/50 fallback)"`
- Log phase-level summary: `"Scoring: 15 BULLISH, 8 BEARISH, 27 NEUTRAL out of 50 tickers"`

## Implementation Strategy

**Execution order**: Tasks are ordered by dependency — data fix first (unblocks everything), then scoring logic, then downstream phases, then tests.

**Risk mitigation**:
- Each task produces a testable increment
- Existing 745+ tests run after each task to catch regressions
- Direction logic changes are validated with synthetic DataFrames (known-bullish, known-bearish, edge cases)

## Task Breakdown Preview

- [ ] Task 1: Add `data_fetch_period` setting and wire through pipeline (config.py, fetcher.py, orchestrator.py)
- [ ] Task 2: Fix `sma_direction()` graceful degradation for insufficient data (indicators.py)
- [ ] Task 3: Relax `determine_direction()` to reduce false neutrals (composite.py)
- [ ] Task 4: Add diagnostic logging to scoring and direction classification (composite.py, indicators.py)
- [ ] Task 5: Add skip-reason logging to options recommender (recommender.py)
- [ ] Task 6: Rebalance AI risk agent prompt and conviction rubric (agents.py)
- [ ] Task 7: Update and add tests for all changed logic (tests/)

## Dependencies

### Internal
- `config.py` Settings class — Task 1 adds new field consumed by Tasks 2-3
- `indicators.py` — Task 2 changes output consumed by Task 3's `determine_direction()`
- Test suite — Task 7 depends on all implementation tasks (1-6)

### External
- yfinance API — must return >= 200 bars for `period="1y"` (verified: ~252 trading days/year)
- LLM backend — prompt changes (Task 6) rely on model's ability to follow updated instructions

## Success Criteria (Technical)

| Criteria | Validation |
|----------|-----------|
| All 745+ existing tests pass | `pytest tests/` |
| `sma_direction()` returns non-neutral for DataFrames with >= 50 bars and clear trend | New unit tests |
| `determine_direction()` returns BULLISH for RSI=65 + uptrending SMAs | New unit tests |
| Options recommender logs skip reasons at info level | Log inspection |
| Risk agent prompt no longer contains "be conservative" blanket instruction | Code review |
| Pipeline logs direction distribution summary after scoring phase | Log inspection |

## Tasks Created

- [x] #42 - Add data_fetch_period setting and wire through pipeline (parallel: false)
- [x] #44 - Fix sma_direction graceful degradation for insufficient data (parallel: false)
- [x] #46 - Relax determine_direction to reduce false neutrals (parallel: false)
- [x] #48 - Add diagnostic logging to scoring and direction classification (parallel: true)
- [x] #43 - Add skip-reason logging to options recommender (parallel: true)
- [x] #45 - Rebalance AI risk agent prompt and conviction rubric (parallel: true)
- [x] #47 - Update and add tests for all changed logic (parallel: false)

Total tasks: 7
Parallel tasks: 3 (#48, #43, #45)
Sequential tasks: 4 (#42 → #44 → #46, then #47 after all)
Estimated total effort: 8-10 hours

## Estimated Effort

- **7 tasks**, all in existing files (no new files needed)
- **Critical path**: Task 1 → Task 2 → Task 3 → Task 7
- Tasks 4, 5, 6 are independent of each other and can be parallelized after Task 1
