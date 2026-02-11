---
name: recommender-fix
status: backlog
created: 2026-02-11T16:28:21Z
progress: 0%
prd: .claude/prds/recommender-fix.md
github: https://github.com/jobu711/option_alpha_v2/issues/29
---

# Epic: recommender-fix

## Overview

Fix two blocking defects that prevent the scan pipeline from completing past Phase 4 (Options) and Phase 5 (AI Debate). The options liquidity filter is too strict for non-mega-cap stocks, and the Ollama structured output prompt causes small models to echo the JSON schema instead of generating responses.

## Architecture Decisions

- **Config-only fix for spread filter**: Change a single default value rather than introducing adaptive logic — keeps it simple and user-overridable
- **Example-based prompting over schema dumps**: Small LLMs respond far better to concrete JSON examples than abstract JSON Schema definitions. Build a generic helper that derives examples from Pydantic model fields so it works for any response model, not just the two known ones
- **No changes to ClaudeClient**: Claude handles JSON Schema prompts correctly — the fix is Ollama-specific, isolated to `OllamaClient.complete()`

## Technical Approach

### Fix 1: Relax Bid-Ask Spread Default

**Files:** `src/option_alpha/config.py`, `config.json`

Single-value change in two locations:
- `Settings.max_bid_ask_spread_pct` default: `0.10` → `0.30`
- `config.json` persisted value: `0.1` → `0.3`

No logic changes. Existing filtering in `options/chains.py` already uses this value dynamically. Tests use explicit values, so no test changes needed.

### Fix 2: Replace Schema Dump with Example Hint in OllamaClient

**File:** `src/option_alpha/ai/clients.py`

Replace lines 139-151 in `OllamaClient.complete()`:

**Current** (broken): Appends `model_json_schema()` dump containing `description`, `type`, `properties` metadata — llama3.1:8b echoes this verbatim.

**New**: Add a `_build_example_hint(response_model)` helper that:
1. Iterates over the model's `model_fields`
2. Generates a placeholder value per field type (`str` → `"..."`, `int` → `1`, `list[str]` → `["example"]`, `Optional[int]` → `5`, enum → first value)
3. Uses field defaults where available
4. Returns a prompt string like: `"\n\nRespond with a JSON object like this example:\n{...}"`

This produces concrete examples:
- `AgentResponse` → `{"role": "bull", "analysis": "Your analysis here", "key_points": ["point 1"], "conviction": 7}`
- `TradeThesis` → `{"symbol": "TICKER", "direction": "bullish", "conviction": 7, "entry_rationale": "...", "risk_factors": ["risk 1"], "recommended_action": "Buy TICKER 100C 30DTE"}`

### Test Updates

**File:** `tests/test_ai.py`

Update `TestOllamaClientRequest.test_complete_structured_output` to verify:
- The prompt hint contains an example JSON (not a schema definition)
- The hint does NOT contain schema-specific keys like `"description"`, `"properties"`, `"type": "object"`

## Implementation Strategy

This is a small, focused fix with no architectural risk. Both changes are independent and can be implemented in a single pass:

1. Change config default + config.json (1 minute)
2. Implement `_build_example_hint()` helper (core work)
3. Update `OllamaClient.complete()` to use new helper
4. Update test to verify new prompt format
5. Run full test suite to confirm no regressions

## Task Breakdown Preview

- [ ] Task 1: Update `max_bid_ask_spread_pct` default to 0.30 in `config.py` and `config.json`
- [ ] Task 2: Add `_build_example_hint()` helper and replace schema dump in `OllamaClient.complete()`
- [ ] Task 3: Update AI structured output test to verify example-based hint format
- [ ] Task 4: Run full test suite and verify no regressions

## Dependencies

- **Prerequisite**: yfinance fetch fix (already merged on `main`)
- **External**: None — no new packages required

## Success Criteria (Technical)

- `max_bid_ask_spread_pct` defaults to `0.30` in both `config.py` and `config.json`
- `OllamaClient.complete()` prompt hint contains a concrete JSON example, not a schema dump
- No `"description"`, `"properties"`, or `"type": "object"` in the Ollama prompt hint
- `pytest tests/` passes with 0 failures
- `_build_example_hint()` is generic and works for any Pydantic model, not just hardcoded for `AgentResponse`/`TradeThesis`

## Estimated Effort

- **4 tasks**, **4 files** modified
- Small, low-risk changes — no architectural impact

## Tasks Created
- [ ] #30 - Relax bid-ask spread default to 30% (parallel: true)
- [ ] #31 - Replace Ollama schema dump with example-based prompt hint (parallel: true)
- [ ] #32 - Update AI test to verify example-based hint format (parallel: false, depends: #31)
- [ ] #33 - Run full test suite and verify no regressions (parallel: false, depends: #30, #31, #32)

Total tasks: 4
Parallel tasks: 2 (#30, #31)
Sequential tasks: 2 (#32, #33)
Estimated total effort: 1.7 hours
