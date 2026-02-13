---
name: aidebate-architecture-fix
status: backlog
created: 2026-02-13T15:09:15Z
progress: 0%
prd: .claude/prds/aidebate-architecture-fix.md
github: https://github.com/jobu711/option_alpha_v2/issues/62
updated: 2026-02-13T15:16:40Z
---

# Epic: aidebate-architecture-fix

## Overview

Comprehensive reliability, performance, and code quality overhaul of the AI debate system (`src/option_alpha/ai/`). This epic addresses network fragility, missing per-ticker isolation, non-deterministic output, wasted tokens, tightly-coupled indicator interpretation, dumb fallbacks, and opaque error handling — all while preserving the existing 3-agent single-round debate structure and two LLM backends.

## Architecture Decisions

- **No new dependencies**: All changes use existing stdlib + httpx + pydantic. Error categories use `StrEnum` (Python 3.11+), not a new package.
- **Additive model changes only**: New optional fields on `DebateResult` and new config settings with sensible defaults. Existing `config.json` files work without modification.
- **Data-driven thresholds over code**: Replace the 100-line `_interpret_indicator()` if/elif chain with a `INDICATOR_THRESHOLDS` dict. Adding a new indicator = adding a dict entry.
- **Structured error type**: New `AgentError` Pydantic model (not dataclass) to stay consistent with the codebase's Pydantic-everywhere pattern.
- **Jitter via stdlib random**: `random.uniform(0, 0.25 * delay)` for retry jitter — no new dependency needed.

## Technical Approach

### Files Modified

| File | Changes |
|------|---------|
| `src/option_alpha/config.py` | Add `ai_per_ticker_timeout`, `ai_health_check_timeout` settings |
| `src/option_alpha/models.py` | Add `composite_score`, `error` fields to `DebateResult`; add `AgentError` model with `ErrorCategory` enum |
| `src/option_alpha/ai/agents.py` | Jitter in retry backoff; validation-error retry with corrective hint; context-aware fallbacks accepting `TickerScore`; compressed system prompts |
| `src/option_alpha/ai/clients.py` | Ollama health check sends minimal completion; Claude health check validates API key; configurable health check timeout |
| `src/option_alpha/ai/context.py` | Replace `_interpret_indicator()` with `INDICATOR_THRESHOLDS` lookup table; merge Score Breakdown + Signal Summary; remove ticker symbol repetition |
| `src/option_alpha/ai/debate.py` | Per-ticker `asyncio.wait_for()`; sort results by `composite_score` desc after completion; populate `AgentError` on failures; phase summary log |
| `tests/test_ai.py` | New tests for all changes: jitter, per-ticker timeout, ordering, fallbacks, thresholds table, error categories, health checks |

### Key Implementation Details

**Per-ticker timeout (FR-1)**: In `debate.py:run_debates()`, wrap each `_run_single()` coroutine in `asyncio.wait_for(timeout=settings.ai_per_ticker_timeout)`. On `asyncio.TimeoutError`, return fallback with `AgentError(category=TIMEOUT)`. Phase-level timeout remains as backstop.

**Retry with jitter (FR-2)**: In `agents.py:_run_agent_with_retry()`, change default delays to `[1.0, 2.0, 4.0, 8.0, 16.0]`. Add `random.uniform(0, 0.25 * delay)` to each sleep. On `ValidationError`, append corrective hint to the last user message and retry immediately.

**Smarter health check (FR-3)**: `OllamaClient.health_check()` sends a minimal `/api/generate` request ("Say OK") after `/api/tags`. `ClaudeClient.health_check()` sends a minimal messages API call with `max_tokens=1`. Both use `ai_health_check_timeout` from config.

**Deterministic ordering (FR-4)**: Add `composite_score: float | None = None` to `DebateResult`. Populate it during debate construction. After `asyncio.gather()` completes, sort `results` by `composite_score` descending.

**Token-optimized prompts (FR-5)**: Compress Bull/Bear/Risk system prompts to <850 tokens combined. In `context.py`, mention ticker once in header, merge Score Breakdown + Signal Summary into a single "Indicators" section.

**Context-aware fallbacks (FR-6)**: Change `_fallback_agent_response(role)` → `_fallback_agent_response(role, ticker_score=None)`. When `ticker_score` provided: conviction = `max(2, min(8, round(score / 12.5)))`, direction = `ticker_score.direction`.

**Data-driven interpretation (FR-7)**: Define `INDICATOR_THRESHOLDS: dict[str, list[tuple[float, str]]]` at module level. `_interpret_indicator()` becomes a 10-line lookup. Unknown indicators return formatted raw value.

**Structured errors (FR-8)**: `ErrorCategory` StrEnum with NETWORK, PARSE, VALIDATION, TIMEOUT, UNKNOWN. `AgentError` model with `category`, `message`, `agent_role`, `attempt`. Log format: `[{category}] {role} for {ticker} (attempt {n}/{max}): {message}`.

## Implementation Strategy

**Development order**: Tasks are ordered by dependency — foundation (config/models) first, then isolated changes in parallel-safe order, tests last.

**Risk mitigation**: All new config fields have defaults matching or improving current behavior. Model changes are additive (optional fields). Each task is independently testable.

**Testing approach**: Each task includes inline test expectations. Final task adds comprehensive test coverage. Target: 950+ tests (up from 925+).

## Task Breakdown Preview

- [ ] Task 1: Add config settings and model foundations (config.py, models.py — new settings + AgentError + DebateResult fields)
- [ ] Task 2: Enhance health checks for both backends (clients.py — Ollama model-loaded check, Claude API key validation)
- [ ] Task 3: Add retry jitter and structured error reporting (agents.py — jitter backoff, validation hint retry, error categorization)
- [ ] Task 4: Add per-ticker timeout and deterministic result ordering (debate.py — asyncio.wait_for per ticker, sort results post-gather)
- [ ] Task 5: Implement context-aware fallback responses (agents.py — score-derived conviction/direction in fallbacks)
- [ ] Task 6: Replace indicator interpretation with data-driven thresholds (context.py — INDICATOR_THRESHOLDS dict)
- [ ] Task 7: Compress prompts and optimize context token usage (agents.py prompts, context.py — merge sections, remove redundancy)
- [ ] Task 8: Add comprehensive test coverage for all changes (tests/test_ai.py — timeout, ordering, fallbacks, thresholds, errors, health checks)

## Dependencies

### Internal (execution order)
- **Task 1** (config/models) must complete before Tasks 2-5 (they consume new settings/models)
- **Task 6** (thresholds) and **Task 7** (prompts) are independent of each other
- **Task 8** (tests) runs last, after all implementation tasks

### External
- **Ollama API** (`/api/tags`, `/api/generate`) — health check enhancement uses existing endpoints
- **Claude API** (`/v1/messages`) — health check sends minimal request to existing endpoint
- No new external dependencies

## Success Criteria (Technical)

| Metric | Current | Target |
|--------|---------|--------|
| Debate completion rate on VPN | ~60-70% | >90% (5 retries + jitter + per-ticker timeout) |
| Tokens per 10-ticker scan | ~120K-135K | <108K (compressed prompts + merged context sections) |
| Per-ticker p95 latency (Claude) | Unmeasured | <30s (per-ticker timeout default 60s) |
| Fallback usefulness | Generic neutral (conviction=3) | Score-derived direction + conviction |
| `_interpret_indicator()` lines | ~100 | <30 (data-driven lookup) |
| Error categories in logs | 2 (parse/network) | 5 (NETWORK, PARSE, VALIDATION, TIMEOUT, UNKNOWN) |
| Test count | 925+ | 950+ |

## Estimated Effort

- **8 tasks total**, ordered by dependency
- **Critical path**: Task 1 → Tasks 2-5 (parallel-safe) → Task 8
- **Risk items**: Prompt compression (FR-5) requires manual token counting to hit <850 target; health check changes need manual testing against live Ollama/Claude

## Tasks Created
- [ ] #63 - Add config settings and model foundations (parallel: false)
- [ ] #65 - Enhance health checks for both LLM backends (parallel: true)
- [ ] #67 - Add retry jitter and structured error reporting (parallel: true)
- [ ] #69 - Add per-ticker timeout and deterministic result ordering (parallel: true)
- [ ] #64 - Implement context-aware fallback responses (parallel: true)
- [ ] #66 - Replace indicator interpretation with data-driven thresholds (parallel: true)
- [ ] #68 - Compress prompts and optimize context token usage (parallel: false)
- [ ] #70 - Add comprehensive test coverage for all changes (parallel: false)

Total tasks: 8
Parallel tasks: 4 (can be worked on simultaneously after #63)
Sequential tasks: 4 (have dependencies)
Estimated total effort: 14-22 hours
