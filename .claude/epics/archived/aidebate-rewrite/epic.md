---
name: aidebate-rewrite
status: completed
created: 2026-02-13T14:31:32Z
updated: 2026-02-13T15:30:00Z
completed: 2026-02-13T15:30:00Z
progress: 100%
prd: .claude/prds/aidebate-rewrite.md
github: https://github.com/jobu711/option_alpha_v2/issues/77
---

# Epic: aidebate-rewrite

## Overview

Replace the broken AI debate HTTP client layer with official provider SDKs (`anthropic`, `ollama`) while preserving the Bull/Bear/Risk debate structure, external interfaces, and all data models. The rewrite touches 3 files (`clients.py`, `agents.py`, `debate.py`) plus dependencies and tests. `context.py` and `models.py` are unchanged.

## Architecture Decisions

1. **Official SDKs over raw httpx**: The `anthropic` and `ollama` Python SDKs handle connection pooling, auth headers, retries, timeouts, and structured output — eliminating ~400 lines of brittle hand-rolled HTTP code.

2. **SDK-native structured output**: Claude uses tool-use (`response_model` via the SDK) for reliable JSON. Ollama uses the `format` parameter. No more manual `_extract_json_from_text()` parsing.

3. **Keep the LLMClient ABC**: Even with SDKs, wrap them behind the existing abstract base class so tests can mock at `complete()` level without touching SDK internals.

4. **Simplify retry to 1+1**: One attempt + one retry (2 total). Parse/validation retries immediately; network retries after a short delay. This prevents the retry spiral that burns timeout budgets.

5. **Equal time budget split**: Each of the 3 agents gets 1/3 of the per-ticker timeout. Simple, predictable, no complex dynamic reallocation.

6. **Preserve all external interfaces**: `get_client()`, `DebateManager.run_debate()`, `DebateManager.run_debates()`, progress callbacks, config fields — all unchanged. Pipeline, persistence, and web layers need zero modifications.

## Technical Approach

### clients.py (rewrite)

**Current**: 403 lines — raw httpx, manual JSON parsing, example hint builder, health checks via raw HTTP.

**New**:
- `OllamaClient`: wraps `ollama.AsyncClient`. `complete()` calls `client.chat()` with `format="json"`. Health check calls `client.list()` and checks model presence.
- `ClaudeClient`: wraps `anthropic.AsyncAnthropic`. `complete()` calls `client.messages.create()` with tool-use for structured output. Health check uses minimal `max_tokens=1` request.
- `get_client(config)`: factory unchanged — returns appropriate client based on `ai_backend` setting.
- `LLMClient` ABC signature preserved: `complete(messages, response_model)` and `health_check()`.
- Keep `_build_example_hint()` for Ollama JSON guidance (append to user message). Remove `_extract_json_from_text()` and `_parse_structured_output()`.
- `timeout` attribute preserved (orchestrator sets it via `client.timeout = ...`).

### agents.py (rewrite)

**Current**: 336 lines — 5-retry mega-function, httpx exception handling, 3 agent functions, 2 fallback functions.

**New**:
- Keep system prompts verbatim (BULL/BEAR/RISK_SYSTEM_PROMPT).
- Keep `_fallback_agent_response()` and `_fallback_thesis()` verbatim.
- Replace `_run_agent_with_retry()` with simpler `_call_with_retry()`: 1 attempt + 1 retry. Catches SDK-specific exceptions instead of httpx exceptions.
- `run_bull_agent()`, `run_bear_agent()`, `run_risk_agent()`: same signatures, same logic flow, just use the new retry wrapper.

### debate.py (rewrite)

**Current**: 292 lines — complex time budget (40%/50%/remaining), semaphore concurrency, phase timeout.

**New**:
- `DebateManager.__init__(client)` — unchanged.
- `run_debate()`: simplified time budget — each agent gets `per_ticker_timeout / 3`. Three sequential steps with `asyncio.wait_for()`.
- `run_debates()`: same semaphore pattern, same progress callback, same deterministic sort.
- Keep `_build_risk_response()` and `_fallback_debate_result()`.
- Same `ProgressCallback` type alias.

### pyproject.toml (update)

- Add `anthropic>=0.39.0` and `ollama>=0.4.0` to core dependencies.
- Remove `instructor` (no longer needed — was unused anyway).
- Keep `httpx` (used elsewhere in codebase).

### tests/test_ai.py (rewrite)

- Mock at `LLMClient.complete()` level (same as current approach).
- Replace httpx exception mocks with SDK exception mocks.
- Test new retry logic (1+1 instead of 5).
- Preserve coverage of: fallbacks, context building, debate orchestration, time budgets, concurrency, progress callbacks.

## Task Breakdown Preview

- [ ] Task 1: Add SDK dependencies — add `anthropic` and `ollama` to pyproject.toml, remove `instructor`, install
- [ ] Task 2: Rewrite clients.py — SDK-based OllamaClient, ClaudeClient, get_client(), health checks
- [ ] Task 3: Rewrite agents.py — simplified retry, same prompts, same agent functions, same fallbacks
- [ ] Task 4: Rewrite debate.py — simplified time budget, same DebateManager interface
- [ ] Task 5: Rewrite AI tests — update test_ai.py for new SDK-based implementation
- [ ] Task 6: Validate full test suite — run all 925+ tests, fix any regressions

## Dependencies

### Internal
- `src/option_alpha/ai/clients.py` — full rewrite
- `src/option_alpha/ai/agents.py` — full rewrite
- `src/option_alpha/ai/debate.py` — full rewrite
- `src/option_alpha/ai/context.py` — **unchanged**
- `src/option_alpha/models.py` — **unchanged**
- `src/option_alpha/config.py` — **unchanged** (existing fields sufficient)
- `src/option_alpha/pipeline/orchestrator.py` — **unchanged**
- `tests/test_ai.py` — full rewrite
- `pyproject.toml` — dependency update

### External
- `anthropic` Python SDK (new dependency)
- `ollama` Python SDK (new dependency)

## Success Criteria (Technical)

1. `pytest tests/` passes with 0 failures
2. At least one real debate completes on Ollama (manual smoke test)
3. At least one real debate completes on Claude (manual smoke test)
4. AI module LOC is reduced from current ~1350 lines
5. No `_extract_json_from_text()` or raw httpx calls remain in `ai/`

## Tasks Created

- [x] #78 - Add SDK dependencies (parallel: false)
- [x] #79 - Rewrite clients.py with official SDKs (parallel: false)
- [x] #80 - Rewrite agents.py with simplified retry (parallel: false)
- [x] #81 - Rewrite debate.py with simplified time budget (parallel: false)
- [x] #82 - Rewrite AI tests for SDK-based implementation (parallel: false)
- [x] #83 - Validate full test suite and clean up (parallel: false)

Total tasks: 6
Parallel tasks: 0
Sequential tasks: 6
Estimated total effort: 14.5 hours

## Estimated Effort

- 6 tasks, sequential dependency chain (#78 → #79 → #80 → #81 → #82 → #83)
- Task #79 (clients.py) is the critical path — SDK integration and structured output
- Tasks #80-#81 (agents, debate) are straightforward once clients work
- Task #82 (tests) is the largest by LOC but mechanically follows implementation
