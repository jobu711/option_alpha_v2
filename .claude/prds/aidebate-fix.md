---
name: aidebate-fix
description: Fix AI debate reliability failures and overhaul the debate system for local Ollama usage
status: backlog
created: 2026-02-13T12:39:56Z
---

# PRD: aidebate-fix

## Executive Summary

The AI debate system (Bull/Bear/Risk multi-agent analysis) is completely non-functional on local Ollama with llama3.1:8b and 8-16GB hardware. All debates timeout and fall back to conservative defaults, meaning users never receive actual LLM-generated trade analysis. This overhaul will make the debate system reliable on modest local hardware first, then improve output quality and flexibility.

## Problem Statement

### Current Failure Mode
Every debate times out when using the default Ollama configuration (llama3.1:8b, 8-16GB RAM). The root causes are:

1. **Per-ticker timeout too aggressive**: `ai_per_ticker_timeout=60s` must cover 3 sequential agent calls (Bull -> Bear -> Risk), giving each agent only ~20s. An 8B model on CPU/limited VRAM often needs 30-60s per response for the ~3000-token context.

2. **Concurrency causes GPU/CPU contention**: `ai_debate_concurrency=3` launches 3 parallel debates, each running 3 sequential agents. On hardware with limited VRAM, concurrent Ollama requests queue and starve each other, compounding timeouts.

3. **Context too large for small models**: `build_context()` produces ~2500-3000 tokens of input. With the JSON schema hint appended by `OllamaClient.complete()`, the total prompt can exceed 4000 tokens. Small models struggle to produce structured JSON output from large prompts within timeout windows.

4. **Retry delays burn the per-ticker budget**: 5 retry attempts with delays `[1, 2, 4, 8, 16]` = 31s of sleep alone. If the first attempt times out at 20s and retries begin, the per-ticker 60s budget is already exhausted.

5. **Health check is expensive**: The health check generates a full completion (`"Say OK"`) which loads the model into VRAM. On cold start this can take 10-30s, leaving less headroom for actual debates.

### Impact
- 100% of debates produce fallback responses, making the entire AI debate phase useless
- Users see "[FALLBACK]" labels on every ticker, undermining confidence in the system
- The 6th pipeline phase (AI Debate) adds wall-clock time with zero value

## User Stories

### US-1: Local Ollama User (Primary)
**As a** user running Ollama locally on 8-16GB hardware,
**I want** the AI debate system to complete successfully for top-scored tickers,
**so that** I receive actual LLM-generated bull/bear/risk analysis instead of fallback defaults.

**Acceptance Criteria:**
- At least 80% of debates complete without timeout on llama3.1:8b with 16GB RAM
- Per-debate wall-clock time under 3 minutes with concurrency=1
- Fallback responses are only used for genuine model failures, not timeout starvation

### US-2: Configuration Awareness
**As a** user with varying hardware,
**I want** the system to auto-detect reasonable defaults based on my backend,
**so that** I don't need to manually tune 6+ timeout/concurrency settings.

**Acceptance Criteria:**
- Ollama backend defaults to concurrency=1, longer timeouts
- Claude API backend defaults to concurrency=3, shorter timeouts
- Users can still override all settings manually

### US-3: Prompt Efficiency
**As a** user running a small local model,
**I want** the prompts to be concise and focused,
**so that** the model can produce structured output within reasonable time limits.

**Acceptance Criteria:**
- Context prompt reduced to ~1500-2000 tokens (from ~3000)
- JSON example hints are compact (remove verbose schema dumps)
- Output quality remains comparable or improves

### US-4: Graceful Degradation
**As a** user,
**I want** partial results when some debates fail,
**so that** I still get value from debates that completed even if others timeout.

**Acceptance Criteria:**
- Completed debates are preserved even when the phase-level timeout fires
- Progress reporting shows which tickers completed vs failed
- Fallback results are clearly distinguishable from real LLM output

## Requirements

### Functional Requirements

#### FR-1: Backend-Aware Default Configuration
- Detect backend type (Ollama vs Claude) and apply appropriate defaults:
  - **Ollama**: `ai_debate_concurrency=1`, `ai_per_ticker_timeout=180`, `ai_request_timeout=120`, `ai_retry_delays=[2, 4]`
  - **Claude**: `ai_debate_concurrency=3`, `ai_per_ticker_timeout=60`, `ai_request_timeout=30`, `ai_retry_delays=[1, 2, 4]`
- Allow all values to be overridden in `config.json`

#### FR-2: Timeout Budget Management
- Implement a per-ticker time budget that is aware of the 3-agent sequence
- Each agent gets a fair share of the remaining budget (not a fixed split)
- If Bull agent takes 40s of a 180s budget, Bear gets up to 70s, Risk gets the remainder
- Abort early if remaining budget is insufficient for next agent

#### FR-3: Context Compression
- Reduce `build_context()` output to ~1500-2000 tokens:
  - Remove the formatted table headers/separators (use compact key-value format)
  - Only include top 5-6 most significant indicators (by weight or signal strength) instead of all
  - Condense options flow and risk parameters into single-line summaries
  - Remove the "Based on the above data..." trailing prompt (already in system prompt)

#### FR-4: Compact JSON Hints
- Replace verbose `model_json_schema()` dump in `ClaudeClient` with the compact `_build_example_hint()` approach already used in `OllamaClient`
- Reduce example hint size by using shorter placeholder values
- Consider a one-line schema hint for very small models

#### FR-5: Lighter Health Check
- Replace the full-generation health check (`"Say OK"`) with a simple API tag/model list check
- For Ollama: just verify `/api/tags` returns the expected model
- For Claude: just verify API key is valid with minimal token request (already does `max_tokens=1`)

#### FR-6: Reduce Retry Count for Ollama
- Default to 2 retries (instead of 5) for Ollama to avoid burning timeout budget
- Keep longer retry chains for Claude API where requests are faster

#### FR-7: Progress Visibility
- Emit per-agent progress (not just per-ticker) so users can see "Bull done, running Bear..."
- Log estimated time remaining based on observed per-agent durations

### Non-Functional Requirements

#### NFR-1: Performance
- Single debate (3 agents) should complete in under 180s on llama3.1:8b / 16GB
- Full debate phase for 10 tickers (concurrency=1) should complete in under 30 minutes
- No regression in Claude API performance

#### NFR-2: Backward Compatibility
- All existing config fields continue to work
- Users with custom `config.json` settings are not affected
- Fallback behavior remains conservative (no-trade on failure)

#### NFR-3: Testability
- All new timeout/budget logic must be unit-testable with mocked clients
- No changes to the test mocking patterns

## Success Criteria

1. **Reliability**: >80% of debates complete successfully on llama3.1:8b / 16GB RAM with default settings
2. **No regressions**: All 925+ existing tests pass
3. **Timeout elimination**: Zero timeout-induced fallbacks under normal Ollama operation
4. **Measurable improvement**: Log output shows actual LLM responses instead of "[FALLBACK]" labels

## Constraints & Assumptions

### Constraints
- Must work with Python 3.11+
- Cannot add new dependencies (use existing httpx, pydantic, asyncio)
- Must remain compatible with both Ollama and Claude backends
- Cannot change the Bull -> Bear -> Risk sequential agent order (Bear needs Bull's output, Risk needs both)

### Assumptions
- Users have Ollama installed and the model pulled (`ollama pull llama3.1:8b`)
- 8GB RAM is the minimum viable hardware for local Ollama
- The 8B parameter model is the baseline target; larger models will perform better
- Network latency to Ollama is negligible (localhost)

## Out of Scope

- Adding new LLM backends (e.g., OpenAI, vLLM, LMStudio)
- Changing the 3-agent debate architecture (Bull/Bear/Risk)
- Adding streaming support for LLM responses
- GPU detection or hardware profiling
- Model recommendation engine (suggesting which Ollama model to use)
- Changes to the scoring, options, or persistence phases

## Dependencies

- **Internal**: `config.py` (Settings), `ai/` module (all 4 files), `pipeline/orchestrator.py` (timeout integration)
- **External**: Ollama running locally with llama3.1:8b pulled
- **Testing**: Existing test suite in `tests/` (must all pass)
