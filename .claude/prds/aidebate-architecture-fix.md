---
name: aidebate-architecture-fix
description: Comprehensive reliability, performance, and code quality overhaul of the AI debate system
status: backlog
created: 2026-02-13T14:48:46Z
---

# PRD: aidebate-architecture-fix

## Executive Summary

Overhaul the AI debate system (`src/option_alpha/ai/`) to improve network resilience, reduce token costs, fix non-deterministic behavior, and decouple tightly-coupled components. The current system suffers from shallow health checks, missing per-ticker timeouts, overly conservative fallbacks, and a context builder with hard-coded thresholds. These issues cause unreliable debate execution — especially on VPN or unstable network connections — and make the code harder to maintain.

This PRD preserves the existing 3-agent (Bull/Bear/Risk) single-round debate structure and the two LLM backends (Ollama, Claude). No new agents, backends, or debate rounds are introduced.

## Problem Statement

The AI debate phase (Phase 6 of the scan pipeline) has several architectural issues that degrade reliability and maintainability:

1. **Network fragility**: Users on VPN (e.g., ProtonVPN) experience frequent network errors during debates. The current retry logic (3 attempts with `[2.0, 4.0, 8.0]` second backoff) is insufficient for intermittent connectivity. The health check only verifies API reachability, not model readiness.

2. **Missing per-ticker isolation**: A single hung debate can block the entire phase for up to 600 seconds. There is no per-ticker timeout — only a phase-level `asyncio.wait_for()`.

3. **Non-deterministic output**: Concurrent debate results are appended in completion order, not score order. This produces inconsistent ordering across runs.

4. **Wasted tokens**: System prompts are ~1500 tokens combined. With ~2500-3000 tokens of context per ticker, each 3-agent debate consumes ~12K-13.5K tokens. A 10-ticker scan uses ~120K-135K tokens (~$0.30-0.40 on Claude).

5. **Tight coupling**: `context.py:_interpret_indicator()` has 100 lines of hard-coded thresholds. Any scoring indicator change requires edits here.

6. **Dumb fallbacks**: All fallback responses use conviction=3 and direction=NEUTRAL regardless of the pre-computed `TickerScore.direction` signal.

7. **Opaque error handling**: `asyncio.gather(..., return_exceptions=True)` swallows exceptions. Pydantic `ValidationError` is treated identically to `JSONDecodeError`, losing the opportunity to retry with hints.

This matters now because the system is actively in use and network errors are a recurring pain point.

## User Stories

### US-1: Reliable debates on unstable networks
**As a** user running scans over VPN,
**I want** the debate phase to gracefully handle intermittent network drops,
**So that** I get debate results for most tickers even when my connection is flaky.

**Acceptance Criteria:**
- Per-ticker timeout (configurable, default 60s) prevents one ticker from blocking the phase
- Retry count and backoff delays are configurable and default to more resilient values (e.g., 5 retries with jitter)
- Health check verifies the LLM model is loaded and responsive (not just that the API endpoint exists)
- Individual ticker failures produce fallback results without aborting other in-flight debates

### US-2: Deterministic debate output
**As a** developer reviewing scan results,
**I want** debate results to be ordered by composite score (descending),
**So that** results are consistent across runs regardless of concurrency timing.

**Acceptance Criteria:**
- `run_debates()` returns results sorted by `composite_score` descending
- Test coverage verifies ordering under concurrent execution

### US-3: Reduced token costs
**As a** user paying for Claude API calls,
**I want** the debate system to minimize token usage without sacrificing analysis quality,
**So that** my per-scan costs are lower.

**Acceptance Criteria:**
- System prompts are compressed (target: <900 tokens combined, down from ~1500)
- Context builder avoids redundant data (e.g., don't repeat ticker symbol in every section)
- Measurable: total tokens per 10-ticker scan reduced by at least 20%

### US-4: Context-aware fallbacks
**As a** user viewing debate results,
**I want** fallback responses to reflect the pre-computed scoring direction,
**So that** fallback results are more useful than a generic "neutral, conviction 3".

**Acceptance Criteria:**
- Fallback conviction is derived from `composite_score` (e.g., score > 70 = conviction 6, score < 30 = conviction 4)
- Fallback direction matches `TickerScore.direction`
- Fallback `entry_rationale` references the composite score and direction

### US-5: Maintainable context builder
**As a** developer adding new scoring indicators,
**I want** the context builder to not require hard-coded interpretation thresholds,
**So that** new indicators work without editing `_interpret_indicator()`.

**Acceptance Criteria:**
- Indicator interpretation thresholds are defined in a data structure (dict or config), not if/elif chains
- Adding a new indicator requires only adding an entry to the data structure
- Existing interpretations remain unchanged (backward compatible)

### US-6: Clear error reporting
**As a** developer debugging debate failures,
**I want** distinct error categories with actionable log messages,
**So that** I can quickly identify whether failures are network, parse, validation, or timeout errors.

**Acceptance Criteria:**
- Error types are categorized: `NETWORK`, `PARSE`, `VALIDATION`, `TIMEOUT`, `UNKNOWN`
- Each debate result includes an error field (None on success, error category + message on failure)
- Validation errors trigger a retry with a corrective hint appended to the prompt
- Log messages include the error category, ticker, agent role, and attempt number

## Requirements

### Functional Requirements

#### FR-1: Per-ticker timeout
- Add `ai_per_ticker_timeout` setting (default: 60 seconds) to `config.py`
- Wrap each `run_debate()` call in `asyncio.wait_for(timeout=ai_per_ticker_timeout)`
- On timeout, log a warning and return a fallback `DebateResult` for that ticker
- Phase-level timeout (`ai_debate_phase_timeout`) remains as a backstop

#### FR-2: Enhanced retry with jitter
- Change default `ai_retry_delays` to `[1.0, 2.0, 4.0, 8.0, 16.0]` (5 retries)
- Add random jitter (0-25% of delay) to each retry sleep to avoid thundering herd
- Differentiate retry behavior by error type:
  - **Parse errors** (`JSONDecodeError`): Retry immediately (no change)
  - **Validation errors** (`ValidationError`): Retry with corrective hint appended to prompt
  - **Network errors** (`TimeoutException`, `HTTPStatusError`): Retry with jitter backoff
  - **Other errors**: Retry with jitter backoff

#### FR-3: Smarter health check
- **Ollama**: After checking `/api/tags`, also send a minimal completion request (e.g., "Say OK") to verify model is loaded and responsive
- **Claude**: Verify API key is valid by checking response status from a lightweight request
- Add `ai_health_check_timeout` setting (default: 15 seconds)
- Health check failure includes a descriptive reason (e.g., "Model not loaded", "API key invalid", "Connection refused")

#### FR-4: Deterministic result ordering
- After all debates complete (or timeout), sort `results` list by `composite_score` descending before returning
- Add `composite_score` field to `DebateResult` model to enable sorting without external lookup

#### FR-5: Token-optimized prompts
- Compress system prompts: remove redundant phrasing, use concise bullet instructions
- Target: Bull prompt <250 tokens, Bear prompt <250 tokens, Risk prompt <350 tokens (combined <850)
- Remove ticker symbol repetition across context sections (mention once in header)
- Combine Score Breakdown and Signal Summary into a single compact section in context builder

#### FR-6: Context-aware fallback responses
- `_fallback_agent_response()` accepts `ticker_score: TickerScore` parameter
- Fallback conviction derived from `composite_score`: `max(2, min(8, round(score / 12.5)))`
- Fallback direction matches `ticker_score.direction`
- Fallback analysis includes: "Automated fallback based on composite score {score:.1f} ({direction})"

#### FR-7: Data-driven indicator interpretation
- Replace `_interpret_indicator()` if/elif chain with a threshold lookup table
- Define a `INDICATOR_THRESHOLDS` dict mapping indicator names to `list[tuple[threshold, label]]`
- Lookup function: iterate thresholds in order, return first matching label
- Unknown indicators return the raw value as a string (no KeyError)

#### FR-8: Structured error reporting
- Define an `AgentError` dataclass: `category` (enum: NETWORK, PARSE, VALIDATION, TIMEOUT, UNKNOWN), `message`, `agent_role`, `attempt`
- Add optional `error: AgentError | None` field to `DebateResult`
- Validation error retry: append "Your previous response had validation issues: {error_details}. Please fix." to the user message
- Log format: `[{category}] {agent_role} for {ticker} (attempt {n}/{max}): {message}`

### Non-Functional Requirements

#### NFR-1: Performance
- Per-ticker debate completion target: <30 seconds (p95) on Claude backend
- Phase-level timeout remains 600s as backstop
- No additional LLM calls beyond existing 3-agent flow (except health check warm-up)

#### NFR-2: Backward compatibility
- All new config settings have sensible defaults matching or improving current behavior
- Existing `config.json` files work without modification
- No changes to database schema or API responses
- `DebateResult` model changes are additive (new optional fields only)

#### NFR-3: Test coverage
- All new/modified functions must have unit tests
- Network error scenarios tested via mocked httpx responses
- Per-ticker timeout tested with artificially slow mock
- Fallback logic tested with various composite scores and directions
- Target: maintain or improve current test count (925+)

#### NFR-4: Observability
- All error categories logged at appropriate levels (WARNING for retries, ERROR for final failures)
- Debate phase summary log: "Completed {n}/{total} debates ({failures} failures) in {elapsed:.1f}s"
- Per-ticker timing logged at DEBUG level

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Debate completion rate on VPN | ~60-70% (estimated) | >90% |
| Tokens per 10-ticker scan | ~120K-135K | <108K (20% reduction) |
| Per-ticker p95 latency (Claude) | Unmeasured | <30s |
| Fallback usefulness | Generic neutral | Score-derived direction + conviction |
| Lines in `_interpret_indicator()` | ~100 | <30 (data-driven) |
| Error categories in logs | 2 (parse/network) | 5 (NETWORK, PARSE, VALIDATION, TIMEOUT, UNKNOWN) |
| Test count | 925+ | 950+ |

## Constraints & Assumptions

### Constraints
- **Python 3.11+**: Can use `asyncio.TaskGroup` if beneficial, but not required
- **No new dependencies**: All changes use existing stdlib + httpx + pydantic
- **Two backends only**: Ollama and Claude — no new LLM provider integrations
- **Three agents only**: Bull, Bear, Risk — no new agent roles
- **Single-round debates**: No multi-turn agent interactions

### Assumptions
- Network errors are the primary cause of debate failures (VPN-related)
- Token cost reduction of 20% is achievable through prompt compression without quality loss
- Existing test infrastructure (pytest + mocks) is sufficient for new test scenarios
- Users accept slightly longer phase times if individual ticker reliability improves (5 retries vs 3)

## Out of Scope

- **New LLM backends** (OpenAI, Groq, vLLM, etc.)
- **New agent roles** or configurable agent pipelines
- **Multi-round debates** (agents responding to each other iteratively)
- **Response caching across scans** — each scan runs fresh debates
- **Database schema changes** — no new tables or columns
- **Web UI changes** — debate results display remains unchanged
- **Prompt engineering** for quality improvements — only structural compression for token reduction
- **Streaming responses** from LLM backends
- **Rate limiting** for Claude API (managed externally)

## Dependencies

### Internal
- `src/option_alpha/config.py` — new settings fields
- `src/option_alpha/models.py` — additive fields on `DebateResult`
- `src/option_alpha/ai/agents.py` — retry logic, fallback logic, prompt text
- `src/option_alpha/ai/clients.py` — enhanced health checks
- `src/option_alpha/ai/context.py` — data-driven interpreter, token optimization
- `src/option_alpha/ai/debate.py` — per-ticker timeout, result ordering, error reporting
- `tests/test_ai.py` — new test cases

### External
- **Ollama API** (`/api/tags`, `/api/generate`) — health check enhancement depends on existing endpoints
- **Claude API** (`/v1/messages`) — no new endpoints needed
- **httpx** — existing dependency, no version change needed
