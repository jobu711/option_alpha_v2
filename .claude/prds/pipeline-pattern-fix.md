---
name: pipeline-pattern-fix
description: Reorder pipeline to persist before AI Debate, and fix agent retry/timeout architecture to eliminate 25+ min scan overhead
status: backlog
created: 2026-02-11T18:18:55Z
---

# PRD: pipeline-pattern-fix

## Executive Summary

Two complementary fixes to the scan pipeline's AI Debate phase:

1. **Pipeline reordering**: Move Persist before AI Debate so scores and options are checkpointed to SQLite before the riskiest phase begins. Debate results are appended to the existing scan run afterward.
2. **Agent retry/timeout overhaul**: Fix the zero-delay retry loops, hardcoded timeouts, and sequential debate execution that combine to add 25+ minutes of overhead when LLM calls fail.

Together these changes eliminate data loss on debate failure and reduce worst-case debate phase duration from ~180 minutes to under 10 minutes.

## Problem Statement

### Problem 1: Data loss on debate failure

The pipeline currently runs: `Data Fetch → Scoring → Catalysts → Options → AI Debate → Persist`. Because Persist is the **final** phase, if AI Debate fails or times out, **all** previously computed results (ticker scores, catalyst-adjusted scores, options recommendations) are lost. Users must re-run the entire scan to recover data that was already successfully computed.

### Problem 2: Agent retry architecture adds 25+ minutes of overhead

The bull/bear/risk AI agents are routinely failing during scans due to compounding architectural issues:

**Zero retry delay** (`ai/agents.py:98-118`): The agent retry loops have **no sleep between attempts**. When a request times out after 120s, it immediately retries, hammering an already-overloaded LLM. Compare to `fetch_retry_delays: [1.0, 2.0, 4.0]` which the data fetch phase correctly implements.

**Fully sequential execution** (`ai/debate.py:59-69, 125-140`): All debates run in a strict `for` loop — one ticker at a time, and within each ticker: bull → bear → risk are `await`ed sequentially. There is no concurrency at either level.

**Hardcoded 120s timeouts** (`ai/clients.py`): Both OllamaClient and ClaudeClient use hardcoded 120-second timeouts. These are not configurable via Settings, unlike fetch timeouts.

**JSON parse failures waste retries** (`ai/clients.py:146-150`): Small LLMs (Ollama llama3.1:8b) frequently return malformed JSON. Parse errors are caught by the same broad `except Exception` as network errors, consuming a full retry attempt with no special handling or faster timeout.

**No circuit breaker** (`pipeline/orchestrator.py:440-456`): The pipeline doesn't verify LLM health before starting debates. If Ollama is hung, all 10 debates fail at maximum retry cost.

**The math**: Worst case per ticker = 3 agents × 3 retries × 120s timeout = **1,080s (18 min)**. With 10 tickers (`top_n_ai_debate` default), theoretical max is **3 hours**. Realistic scenario with 30-40% JSON failures on Ollama: **25-30 minutes**.

**Why now:** As the ticker universe grows and more LLM backends are tested, debate failures are increasingly common. Scores and options data are independently valuable even without AI debate results.

## User Stories

### US-1: Scan resilience on debate failure
**As a** user running a full scan,
**I want** my scores and options recommendations saved before AI debate starts,
**So that** if the debate phase fails, I still have actionable scan results.

**Acceptance Criteria:**
- Scores and options recs are visible in the dashboard after a scan where AI Debate failed
- The scan run is marked with a status that indicates partial completion (debate missing)
- No data loss for phases 1-4 when phase 5 (old) / phase 6 (new) fails

### US-2: Incremental scan results
**As a** user monitoring scan progress,
**I want** to see scores and options recs appear in the UI as soon as they're persisted,
**So that** I can start reviewing results while AI debate is still running.

**Acceptance Criteria:**
- Dashboard shows scores/options after Persist completes (before debate starts)
- Dashboard updates with debate results when they arrive
- Progress tracking accurately reflects the new phase order

### US-3: Debate results appended to existing scan
**As a** developer maintaining the pipeline,
**I want** debate results to update the existing scan run record,
**So that** there is a single scan run row per execution (not duplicates).

**Acceptance Criteria:**
- A single `scan_runs` row per pipeline execution
- Debate results (theses) are linked to the same `scan_run_id`
- `ScanRun.status` updates from partial to completed after debate succeeds
- `ScanRun.debates_completed` count is updated after debate phase

### US-4: Faster debate phase
**As a** user running a full scan,
**I want** the AI debate phase to complete in a reasonable time even when some LLM calls fail,
**So that** scans don't hang for 25+ minutes waiting on retries.

**Acceptance Criteria:**
- Debate phase completes in under 10 minutes even with 30-40% agent failure rate
- Failed agents use exponential backoff instead of instant retry
- Debate timeouts and retry counts are configurable via Settings

### US-5: LLM health gating
**As a** user running a scan with Ollama,
**I want** the pipeline to detect an unresponsive LLM before committing to 10 debates,
**So that** I don't waste time on a debate phase that will entirely fail.

**Acceptance Criteria:**
- A fast health check runs before the debate loop starts
- If the LLM is unreachable, the debate phase is skipped gracefully (not retried 30 times)
- The scan still completes with `PARTIAL` status (scores + options saved)

## Requirements

### Functional Requirements

#### FR-1: Pipeline phase reordering
Reorder the pipeline phases from:
```
1. Data Fetch  →  2. Scoring  →  3. Catalysts  →  4. Options  →  5. AI Debate  →  6. Persist
```
To:
```
1. Data Fetch  →  2. Scoring  →  3. Catalysts  →  4. Options  →  5. Persist  →  6. AI Debate
```

#### FR-2: Checkpoint persist (Phase 5)
- Save `ScanRun` with status `PARTIAL` (scores + options computed, debate pending)
- Persist all `TickerScore` records to SQLite
- Persist all `OptionsRecommendation` records (if a persistence path exists, or log them)
- Record phase timings and error messages accumulated so far
- Return the `scan_run_id` (database row ID) for use in Phase 6

#### FR-3: Post-debate update (Phase 6 tail)
- After AI Debate completes, update the existing `ScanRun` row:
  - Set `debates_completed` count
  - Update `status` to `COMPLETED` (or keep `PARTIAL` if debate had errors)
  - Update `duration_seconds` to include debate time
- Save `DebateResult` theses linked to the existing `scan_run_id`
- If AI Debate fails entirely, the scan run remains with status `PARTIAL` — no data is lost

#### FR-4: Progress tracking update
- Update `PHASE_NAMES` constant to reflect new order: `["data_fetch", "scoring", "catalysts", "options", "persist", "ai_debate"]`
- Phase indices in progress callbacks must match the new order
- WebSocket progress messages must reflect accurate phase sequencing

#### FR-5: ScanStatus semantics
- `PARTIAL` status means: scores and options saved, but AI debate incomplete or failed
- `COMPLETED` status means: all phases including debate finished successfully
- `FAILED` status means: a critical phase (data fetch, scoring) failed before any useful data was produced

#### FR-6: Agent retry delays with exponential backoff
- Add configurable retry delays to agent retry loops in `ai/agents.py`
- Default delays: `[2.0, 4.0, 8.0]` (exponential backoff matching fetch pattern)
- Add `ai_retry_delays: list[float]` to Settings with `OPTION_ALPHA_` env var prefix
- Each retry sleeps for the configured delay before re-attempting

#### FR-7: Configurable AI timeouts
- Move hardcoded 120s timeouts from OllamaClient/ClaudeClient into Settings
- Add `ai_request_timeout: int = 120` to Settings
- Add `ai_debate_phase_timeout: int = 600` to Settings (max total time for the debate phase, default 10 min)
- If debate phase timeout is reached, return whatever debates completed so far (partial results)

#### FR-8: Differentiate parse errors from network errors
- In agent retry loops, catch `json.JSONDecodeError` and `pydantic.ValidationError` separately from network/timeout errors
- Parse errors should retry with a shorter timeout or a modified prompt hint (e.g., "respond ONLY with valid JSON")
- Parse errors should log at DEBUG level (expected with small LLMs), not WARNING

#### FR-9: LLM health check before debate loop
- Before starting the debate `for` loop, perform a lightweight health check on the configured LLM client
- OllamaClient: call `/api/tags` endpoint (already exists as health check pattern)
- ClaudeClient: use a minimal completion request or API ping
- If health check fails, skip the debate phase entirely and log a clear error
- The scan continues with `PARTIAL` status (scores + options already persisted by FR-2)

#### FR-10: Concurrent debate execution
- Run debates for multiple tickers concurrently using `asyncio.gather` or `asyncio.TaskGroup`
- Add `ai_debate_concurrency: int = 3` to Settings (max concurrent debates)
- Within a single debate, bull → bear → risk must remain sequential (bear needs bull's thesis, risk needs both)
- Between tickers, debates are independent and can run in parallel

### Non-Functional Requirements

#### NFR-1: No regression in happy path
- When all phases succeed, the final `ScanResult` and persisted data must be identical to current behavior
- Existing tests must pass without modification (or with minimal test updates for phase ordering)

#### NFR-2: Backward-compatible database
- No schema changes required (existing `scan_runs`, `ticker_scores`, `ai_theses` tables unchanged)
- The persist phase already saves scores and theses — this change splits when those saves happen

#### NFR-3: Performance
- No measurable increase in total scan time for the happy path
- The extra database write (updating scan run after debate) is negligible (<10ms)

## Success Criteria

| Metric | Target |
|--------|--------|
| Scores preserved on debate failure | 100% of completed scores saved when AI Debate fails |
| Options recs preserved on debate failure | 100% of computed recs saved when AI Debate fails |
| Scan run status accuracy | `PARTIAL` when debate fails, `COMPLETED` when all succeed |
| Debate phase duration (happy path) | Under 5 minutes for 10 tickers with healthy LLM |
| Debate phase duration (30-40% failures) | Under 10 minutes (down from 25-30 min) |
| Debate phase duration (LLM down) | Under 30 seconds (health check fails fast, phase skipped) |
| Existing test suite | All 547+ tests pass (with phase-order adjustments where needed) |
| Happy-path scan result parity | Output identical to current implementation when all phases succeed |

## Constraints & Assumptions

### Constraints
- Must work with existing SQLite schema (no migrations)
- Must maintain the `ScanOrchestrator` class interface (`run_scan()` signature unchanged)
- `ScanResult` return model remains the same
- Python 3.11+ only

### Assumptions
- The `save_scan_run` / `save_ticker_scores` / `save_ai_theses` repository functions can be called at different times with the same `scan_run_id`
- The database connection can be opened, used for persist, closed, then reopened for debate persist (or held open across both phases)
- `ScanStatus.PARTIAL` already exists in the enum and is appropriate for "scores saved, debate pending"

## Out of Scope

- **Resume/retry on debate failure**: Automatically retrying a failed debate for an existing scan run (manual re-scan is fine)
- **Database schema changes**: No new tables or columns
- **UI changes**: Dashboard already handles `PARTIAL` status; no new UI work needed
- **Options recommendations persistence**: If options recs aren't currently persisted to SQLite, adding that persistence is a separate concern (but should be noted if discovered during implementation)
- **Streaming/incremental debate results**: Pushing individual debate results to the UI as they complete (future enhancement)
- **LLM fallback chains**: Automatically switching from Ollama to Claude API on failure (separate feature)

## Dependencies

### Internal
- `pipeline/orchestrator.py` — Phase reordering, debate phase timeout
- `pipeline/progress.py` — Phase names and progress tracking
- `persistence/repository.py` — Must support updating an existing scan run (verify `save_scan_run` can update, or add an `update_scan_run` function)
- `models.py` — `ScanStatus.PARTIAL` must be a valid status (already exists)
- `config.py` — New Settings fields: `ai_retry_delays`, `ai_request_timeout`, `ai_debate_phase_timeout`, `ai_debate_concurrency`
- `ai/agents.py` — Retry delay logic, error differentiation
- `ai/clients.py` — Configurable timeouts, health check methods
- `ai/debate.py` — Concurrent debate execution with `asyncio.gather`/`TaskGroup`
- `web/` routes and WebSocket handlers — Must handle new phase ordering in progress events

### External
- None — this is an internal refactor with no new dependencies

## Technical Notes

### Key files to modify (by area)

**Pipeline reordering:**
1. `src/option_alpha/pipeline/orchestrator.py` — Reorder phases, split persist into checkpoint + finalize
2. `src/option_alpha/persistence/repository.py` — Add `update_scan_run()` for post-debate status update

**Agent retry/timeout overhaul:**
3. `src/option_alpha/config.py` — Add `ai_retry_delays`, `ai_request_timeout`, `ai_debate_phase_timeout`, `ai_debate_concurrency` to Settings
4. `src/option_alpha/ai/agents.py` — Add `asyncio.sleep()` between retries using configured delays; separate `JSONDecodeError`/`ValidationError` handling from network errors
5. `src/option_alpha/ai/clients.py` — Accept timeout from Settings instead of hardcoded 120s; add `health_check()` method to base client
6. `src/option_alpha/ai/debate.py` — Replace sequential `for` loop with `asyncio.Semaphore`-gated concurrent execution; add phase-level timeout via `asyncio.wait_for`

**Tests:**
7. `tests/` — Update phase ordering assertions, add tests for retry delays, concurrent debates, health check gating

### Risk mitigation
- The `_phase_persist` method currently does all persistence in one shot; it needs to be split into "checkpoint" (scores + options) and "finalize" (debate results + status update)
- The `run_id` and database `scan_db_id` must be threaded from Phase 5 persist through to the post-debate update
- Concurrent debates must not overwhelm Ollama — the `ai_debate_concurrency` semaphore prevents this
- Bull → bear → risk ordering within a single debate MUST remain sequential (bear depends on bull's thesis, risk depends on both)
