---
name: recommender-fix
description: Fix options recommender returning 0 results and AI debate echoing JSON schema
status: backlog
created: 2026-02-11T16:20:59Z
---

# PRD: recommender-fix

## Executive Summary

After resolving the yfinance Invalid Crumb 401 errors, two pre-existing pipeline issues surfaced that break the core scan workflow: the options recommender produces zero recommendations due to an overly strict bid-ask spread filter, and the AI debate phase fails because the Ollama llama3.1:8b model echoes back the raw JSON schema instead of generating structured responses. Both issues render the scan pipeline non-functional past the scoring phase.

## Problem Statement

**What problem are we solving?**

The Option Alpha pipeline has two blocking defects in its later phases:

1. **Options phase produces 0 recommendations.** The `max_bid_ask_spread_pct` default of 10% (`0.10`) filters out every option contract for any stock outside mega-cap tech. Materials, energy, healthcare, and mid-cap sectors routinely have bid-ask spreads of 15-40%+, meaning the filter eliminates 100% of contracts for the majority of the ticker universe.

2. **AI debate phase fails with schema echo.** When `OllamaClient.complete()` requests structured output, it appends the full Pydantic `model_json_schema()` dump (including `description`, `type`, `properties` metadata) as a prompt hint. The llama3.1:8b model treats this abstract schema as the expected output and echoes it back verbatim, causing a Pydantic validation error: `input_value={'description': 'Response...', 'type': 'object'}`.

**Why is this important now?**

These are the last two blockers preventing end-to-end scan execution. With the yfinance data fetch fixed, users can now fetch data and score tickers — but the pipeline still fails at the options and AI debate phases, making the application unusable for its core purpose.

## User Stories

### US-1: Options Recommendations for All Sectors
**As a** user scanning mid-cap or non-tech stocks,
**I want** the options recommender to return viable contracts,
**So that** I can evaluate option strategies across the full ticker universe.

**Acceptance Criteria:**
- Running a scan on materials/energy/healthcare sector stocks produces >0 option recommendations
- The default bid-ask spread threshold is 30% (`0.30`)
- Users can still override the threshold via `config.json` or `OPTION_ALPHA_MAX_BID_ASK_SPREAD_PCT` env var
- Existing tests pass without modification (tests use explicit values, not defaults)

### US-2: AI Debate Produces Structured Responses
**As a** user running the full pipeline with Ollama,
**I want** the AI debate phase to generate actual analysis,
**So that** I receive bull/bear/risk assessments instead of schema echoes.

**Acceptance Criteria:**
- Ollama structured output prompts use concrete JSON examples instead of abstract schema dumps
- `AgentResponse` and `TradeThesis` models produce valid structured output from llama3.1:8b
- The schema echo error (`input_value={'description': ...}`) no longer occurs
- Tests verify the new prompt hint format

## Requirements

### Functional Requirements

#### FR-1: Relax Default Bid-Ask Spread Filter
- **FR-1.1:** Change `max_bid_ask_spread_pct` default in `Settings` class (`src/option_alpha/config.py`, line 37) from `0.10` to `0.30`
- **FR-1.2:** Update the persisted value in `config.json` from `0.1` to `0.3`
- **FR-1.3:** No changes to filtering logic — only the default threshold value changes

#### FR-2: Replace Schema Dump with Concrete Examples in Ollama Prompt
- **FR-2.1:** In `OllamaClient.complete()` (`src/option_alpha/ai/clients.py`, lines 139-151), replace the `model_json_schema()` dump with a concrete example JSON hint
- **FR-2.2:** Add a helper function `_build_example_hint(response_model)` that generates a simple example JSON from the model's field names, types, and defaults — not the full JSON schema
- **FR-2.3:** The example hint for `AgentResponse` should look like: `{"role": "bull", "analysis": "Your analysis here", "key_points": ["point 1", "point 2"], "conviction": 7}`
- **FR-2.4:** The example hint for `TradeThesis` should look like: `{"symbol": "TICKER", "direction": "bullish", "conviction": 7, "entry_rationale": "...", "risk_factors": ["risk 1"], "recommended_action": "Buy TICKER 100C 30DTE"}`

### Non-Functional Requirements

- **NFR-1:** No new dependencies — changes use only existing stdlib and Pydantic APIs
- **NFR-2:** All existing tests must continue to pass
- **NFR-3:** Changes are backward-compatible — existing `config.json` files with explicit values are unaffected

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Option recommendations for mid-cap stocks | 0 contracts | >0 contracts per scan |
| AI debate structured output success rate (Ollama) | 0% (schema echo) | >90% valid responses |
| Full pipeline end-to-end completion | Fails at Phase 4 | Completes all 6 phases |
| Test suite pass rate | All pass | All pass (no regressions) |

## Constraints & Assumptions

**Constraints:**
- Must work with the existing `llama3.1:8b` model (no requirement to upgrade Ollama models)
- Must not break the `ClaudeClient` structured output path

**Assumptions:**
- The 30% bid-ask spread default is reasonable for the expanded ~3k ticker universe
- Concrete JSON examples will be sufficient to guide llama3.1:8b (no retry logic needed)
- The `_build_example_hint` helper can derive sensible placeholder values from Pydantic field types and defaults

## Out of Scope

- Adaptive per-sector bid-ask spread thresholds (future enhancement)
- Retry/fallback logic for invalid Ollama responses
- Switching to a larger Ollama model
- Changes to the `ClaudeClient` structured output implementation
- UI/dashboard changes

## Dependencies

- **Internal:** Depends on the yfinance fetch fix being in place (already merged)
- **External:** None — no new packages or API changes required

## Files to Modify

| File | Change |
|------|--------|
| `src/option_alpha/config.py` | Change `max_bid_ask_spread_pct` default from `0.10` to `0.30` |
| `config.json` | Update `max_bid_ask_spread_pct` from `0.1` to `0.3` |
| `src/option_alpha/ai/clients.py` | Replace schema dump with `_build_example_hint()` in `OllamaClient.complete()` |
| `tests/test_ai.py` | Update structured output test to verify new hint format |
