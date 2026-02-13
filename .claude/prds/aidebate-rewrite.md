---
name: aidebate-rewrite
description: Rewrite the AI debate system from scratch using official SDKs to fix pervasive network errors and timeouts
status: backlog
created: 2026-02-13T14:21:33Z
---

# PRD: aidebate-rewrite

## Executive Summary

The AI debate system (`src/option_alpha/ai/`) has never produced a successful debate in practice. Every LLM API call results in a network error or timeout, causing 100% fallback responses. The current implementation uses raw `httpx` calls with hand-rolled JSON parsing, retry logic, and connection management — all of which are fragile and hard to debug.

This PRD proposes a complete rewrite of the AI module using official provider SDKs (`anthropic` for Claude, `ollama` for Ollama). The rewrite keeps the same Bull/Bear/Risk 3-agent debate structure and the same external interface to the pipeline, but replaces all internal implementation with simpler, SDK-driven code that actually works.

## Problem Statement

### Current Failure Mode

Every API call to both Ollama and Claude backends produces network errors or timeouts. No debate has ever completed successfully. The symptoms:

1. **Network errors on every call**: The custom `httpx`-based clients in `clients.py` fail to establish or maintain connections reliably. Hand-rolled timeout handling, connection pooling, and error classification create compounding failure modes.

2. **JSON parsing failures**: Manual `_extract_json_from_text()` tries to parse 4 different JSON formats from raw LLM text output. When the connection does work, parsing often fails, triggering retries that burn the timeout budget.

3. **Retry logic makes it worse**: 5 retries with exponential backoff (`[1, 2, 4, 8, 16]` = 31s of sleep) guarantee that a single failed call exhausts the per-ticker timeout. The retry system designed to improve reliability actually ensures failure.

4. **Cascading timeouts**: Per-ticker timeout (60s) split across 3 sequential agents → per-agent budget of ~20s → insufficient for any real LLM call → timeout → retry → budget exhausted → fallback.

5. **403 lines of client code reimplementing what SDKs provide**: Connection management, authentication, structured output, error handling, and health checks are all hand-rolled in `clients.py` when the `anthropic` and `ollama` Python SDKs handle all of this correctly out of the box.

### Root Cause

The fundamental problem is not timeouts or configuration — it's that the HTTP client layer is broken. Tuning timeouts (as attempted in the `aidebate-fix` epic) cannot fix a client that fails to make successful API calls in the first place.

### Impact

- 100% of debates produce `[FALLBACK]` responses — the AI debate phase adds zero value
- The entire 6th pipeline phase is wasted wall-clock time
- Users have no confidence in the system's AI analysis capability

## User Stories

### US-1: Working Debates

**As a** user running any supported LLM backend,
**I want** the AI debate system to complete debates successfully,
**so that** I receive actual LLM-generated bull/bear/risk analysis for my top-scored tickers.

**Acceptance Criteria:**
- Debates complete without network errors when the LLM backend is running and reachable
- At least 90% of debates produce real LLM output (not fallback) under normal conditions
- Both Ollama and Claude backends work out of the box with minimal configuration

### US-2: Official SDK Integration

**As a** developer maintaining this codebase,
**I want** the LLM clients to use official provider SDKs,
**so that** connection management, auth, retries, and structured output are handled by well-tested library code instead of hand-rolled HTTP calls.

**Acceptance Criteria:**
- Claude client uses the `anthropic` Python SDK
- Ollama client uses the `ollama` Python SDK
- No raw `httpx` calls for LLM communication
- SDK versions pinned in `pyproject.toml`

### US-3: Structured Output

**As a** developer,
**I want** LLM responses to be reliably parsed into Pydantic models,
**so that** JSON parsing failures don't trigger cascading retries and timeouts.

**Acceptance Criteria:**
- Claude uses the Anthropic SDK's tool-use or JSON mode for structured output
- Ollama uses the Ollama SDK's `format` parameter for JSON output
- No manual `_extract_json_from_text()` regex parsing
- Validation errors are handled cleanly with a single retry, not 5

### US-4: Simple Retry Logic

**As a** user,
**I want** failed API calls to retry sensibly without burning the entire timeout budget,
**so that** transient errors recover quickly and permanent errors fail fast.

**Acceptance Criteria:**
- Maximum 2 retries (not 5) with short delays
- SDK-level retries preferred over application-level retries where available
- Total retry time for a single agent call is capped at a reasonable fraction of the per-ticker budget
- Network errors vs parse errors vs validation errors are distinguished using SDK exception types

### US-5: Health Check That Works

**As a** user,
**I want** the system to detect whether my LLM backend is available before starting debates,
**so that** I don't waste time on a debate phase that will fail.

**Acceptance Criteria:**
- Ollama health check: verify the API is reachable and the configured model is available (list models)
- Claude health check: verify the API key is valid with a minimal request
- Health check timeout is separate and short (5-10s)
- Unhealthy backend skips debate phase cleanly

### US-6: Preserved External Interface

**As a** pipeline orchestrator,
**I want** the rewritten AI module to expose the same external interface,
**so that** the pipeline, persistence, and web layers don't need changes.

**Acceptance Criteria:**
- `DebateManager` class with `run_debate()` and `run_debates()` methods
- Same `DebateResult`, `AgentResponse`, `TradeThesis` models (from `models.py`, unchanged)
- Same `get_client()` factory function signature
- Same config fields in `Settings` (no config.json migration needed)
- `build_context()` still returns a string prompt

## Requirements

### Functional Requirements

#### FR-1: Replace httpx Clients with Official SDKs

**Claude client:**
- Use `anthropic.AsyncAnthropic` client
- Use tool-use or `response_format` for structured JSON output
- Let the SDK handle auth, retries, and connection pooling
- Map SDK exceptions (`APIError`, `APITimeoutError`, `AuthenticationError`) to error categories

**Ollama client:**
- Use `ollama.AsyncClient`
- Use the `format` parameter for JSON-structured output
- Let the SDK handle connection management
- Map SDK exceptions to error categories

**Shared interface:**
- Keep the `LLMClient` abstract base class with `complete()` and `health_check()`
- `complete()` takes messages list and response model class, returns parsed Pydantic model
- `get_client(settings)` factory returns the appropriate client

#### FR-2: Simplified Agent Implementation

- Keep Bull/Bear/Risk sequential flow (Bear needs Bull output, Risk needs both)
- Each agent function: build messages → call client.complete() → return parsed model
- Remove the `_run_agent_with_retry()` mega-function — use a simple wrapper:
  - 1 attempt + 1 retry (2 total)
  - On parse/validation failure: retry once with corrective hint
  - On network failure: retry once after 2s delay
  - On second failure: return context-aware fallback
- System prompts: keep existing prompts, they're fine
- Fallback logic: keep context-aware fallbacks using `ticker_score.composite_score`

#### FR-3: Simplified Debate Orchestration

- Keep `DebateManager` with `run_debate()` and `run_debates()`
- Keep semaphore-gated concurrency (`ai_debate_concurrency`)
- Keep per-ticker timeout and phase-level timeout
- Simplify time budget: equal split across 3 agents (no dynamic reallocation)
  - If per-ticker is 180s, each agent gets 60s — simple and predictable
- Keep progress callbacks
- Keep deterministic ordering (sort by composite_score)

#### FR-4: Context Builder Unchanged

- `build_context()` was already improved in the `aidebate-fix` epic
- Keep the current implementation as-is (data-driven interpretation, top 6 indicators, compact format)
- No changes to `context.py`

#### FR-5: Health Check Using SDKs

- Ollama: `client.list()` to verify API reachable and model exists
- Claude: `client.messages.create()` with `max_tokens=1` (already minimal)
- Short timeout (5-10s), separate from request timeout
- Return bool — healthy or not

#### FR-6: Configuration Compatibility

- All existing `Settings` fields preserved: `ai_backend`, `ollama_model`, `claude_api_key`, `ai_retry_delays`, `ai_request_timeout`, `ai_per_ticker_timeout`, `ai_health_check_timeout`, `ai_debate_phase_timeout`, `ai_debate_concurrency`, `top_n_ai_debate`
- `get_effective_ai_settings()` continues to apply backend-aware defaults
- New SDK dependencies added to `pyproject.toml` in appropriate groups

### Non-Functional Requirements

#### NFR-1: Reliability
- >90% of debates complete successfully when backend is healthy
- Zero debates should fail due to client-side HTTP bugs
- Failures should only occur from genuine backend issues (model not loaded, API key invalid, backend down)

#### NFR-2: Simplicity
- Total AI module code should be smaller than current (~1350 lines)
- No manual HTTP request construction
- No manual JSON extraction from text
- Fewer error handling branches (SDKs handle common cases)

#### NFR-3: Testability
- All SDK calls wrapped behind `LLMClient` ABC — easy to mock
- Tests mock at the `LLMClient.complete()` level, not at httpx transport
- Existing test structure in `tests/test_ai.py` adapted to new implementation

#### NFR-4: Performance
- No regression in latency vs current implementation (SDK overhead is negligible)
- Connection reuse handled by SDK clients automatically

#### NFR-5: Backward Compatibility
- Pipeline orchestrator calls unchanged
- Config.json format unchanged
- Database schema unchanged
- Web layer unchanged

## Success Criteria

1. **Debates actually work**: At least 1 full debate (Bull → Bear → Risk) completes successfully with real LLM output on both Ollama and Claude backends
2. **Reliability**: >90% success rate under normal conditions (backend healthy, model loaded)
3. **No regressions**: All existing tests pass (adapted for new implementation)
4. **Code reduction**: AI module total LOC is reduced or comparable
5. **No [FALLBACK] spam**: Fallbacks only appear when backend is genuinely unavailable

## Constraints & Assumptions

### Constraints
- Python 3.11+
- Must add `anthropic` and `ollama` as new dependencies in `pyproject.toml`
- Must keep `LLMClient` ABC interface for testability
- Must keep Bull → Bear → Risk sequential agent order
- Cannot change `models.py` (DebateResult, AgentResponse, TradeThesis are shared)
- AI module scope only — no changes to pipeline, persistence, or web layers

### Assumptions
- The `anthropic` Python SDK supports async and structured output (tool-use)
- The `ollama` Python SDK supports async and JSON format parameter
- Network connectivity to LLM backends is functional (the issue is client code, not network)
- Users will install new SDK dependencies via `pip install -e ".[all]"`

## Out of Scope

- Adding new LLM backends (OpenAI, Gemini, etc.) — future work
- Streaming support for LLM responses
- Changing the 3-agent debate architecture
- Changes to `context.py` (already improved)
- Changes to `models.py` (shared data models)
- Changes to pipeline orchestrator, persistence, or web layers
- GPU detection or hardware profiling
- Prompt engineering improvements (keep existing prompts)

## Dependencies

### Internal
- `src/option_alpha/ai/` — all 4 implementation files rewritten (clients.py, agents.py, debate.py, __init__.py)
- `src/option_alpha/ai/context.py` — preserved unchanged
- `src/option_alpha/config.py` — may need minor updates if SDK config fields added
- `tests/test_ai.py` — must be rewritten to test new implementation

### External
- `anthropic` Python SDK (new dependency)
- `ollama` Python SDK (new dependency)
- Ollama running locally with configured model pulled
- Claude API key configured (for Claude backend)

### Testing
- All 925+ existing tests must pass
- AI tests rewritten to cover new SDK-based implementation
- Mock at `LLMClient.complete()` level
