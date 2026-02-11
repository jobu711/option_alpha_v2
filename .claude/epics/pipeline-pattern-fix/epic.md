---
name: pipeline-pattern-fix
status: backlog
created: 2026-02-11T18:37:46Z
progress: 0%
prd: .claude/prds/pipeline-pattern-fix.md
github: https://github.com/jobu711/option_alpha_v2/issues/34
---

# Epic: pipeline-pattern-fix

## Overview

Fix the scan pipeline's two critical defects: (1) data loss when the AI Debate phase fails, and (2) agent retry architecture that adds 25+ minutes of overhead on LLM failures. The fix reorders Persist before Debate for checkpointing, overhauls agent retry logic with exponential backoff, adds concurrent debate execution, and gates the debate phase on an LLM health check.

## Architecture Decisions

- **Persist-before-Debate reordering**: Scores and options are checkpointed to SQLite as phase 5. Debate becomes phase 6. A new `update_scan_run()` function finalizes the record after debate. This requires no schema changes — the existing tables support split writes.
- **Shared retry helper**: The three agent functions (`run_bull_agent`, `run_bear_agent`, `run_risk_agent`) contain nearly identical retry loops. Extract a common `_run_agent_with_retry()` coroutine that handles backoff delays, parse error differentiation, and logging — called by all three.
- **Semaphore-gated concurrency**: Use `asyncio.Semaphore(n)` + `asyncio.gather()` for inter-ticker parallelism in `debate.py`. Intra-ticker ordering (bull → bear → risk) stays sequential. Default concurrency of 3 avoids overwhelming Ollama.
- **Existing health_check() reuse**: Both `OllamaClient.health_check()` and `ClaudeClient.health_check()` already exist. The orchestrator just needs to call it before the debate loop — no new client code needed.
- **Settings pattern**: New AI fields follow the existing `fetch_*` pattern in `config.py` (e.g., `ai_retry_delays` mirrors `fetch_retry_delays`).

## Technical Approach

### Config Layer (`config.py`)
Add 4 new Settings fields following the existing `fetch_*` pattern:
- `ai_retry_delays: list[float]` — default `[2.0, 4.0, 8.0]`
- `ai_request_timeout: int` — default `120` (seconds)
- `ai_debate_phase_timeout: int` — default `600` (10 min)
- `ai_debate_concurrency: int` — default `3`

### Client Layer (`ai/clients.py`)
- Update `get_client()` factory to pass `timeout=settings.ai_request_timeout` to client constructors
- No other client changes needed (health checks already implemented)

### Agent Layer (`ai/agents.py`)
- Extract `_run_agent_with_retry(agent_fn, messages, client, ...)` helper that:
  - Sleeps `ai_retry_delays[attempt]` between retries
  - Catches `json.JSONDecodeError` / `pydantic.ValidationError` separately — logs at DEBUG, retries immediately (parse errors are fast, no need to wait)
  - Catches `httpx.TimeoutException` / `httpx.HTTPStatusError` — logs at WARNING, sleeps before retry
- All three agent functions call the shared helper instead of inline retry loops

### Debate Layer (`ai/debate.py`)
- Replace sequential `for` loop in `run_debates()` with `asyncio.Semaphore`-gated `asyncio.gather()`
- Add overall phase timeout via `asyncio.wait_for(gather, timeout=ai_debate_phase_timeout)`
- On timeout, return whatever debates completed so far

### Persistence Layer (`persistence/repository.py`)
- Add `update_scan_run(conn, scan_db_id, **fields)` — UPDATE query for `status`, `debates_completed`, `duration_seconds`, `error_message`

### Orchestrator (`pipeline/orchestrator.py`)
- Swap `PHASE_NAMES` order: `["data_fetch", "scoring", "catalysts", "options", "persist", "ai_debate"]`
- Split `_phase_persist` into `_phase_checkpoint` (phase 5: save scores + options, status=PARTIAL) and post-debate finalize logic
- Move AI Debate to phase 6; after debate, call `update_scan_run()` to set final status
- Add `client.health_check()` call at the start of `_phase_ai_debate`; if False, skip debates and log error

## Implementation Strategy

**Development order** (dependency-driven):
1. Config + clients first (no behavior change, just plumbing)
2. Agent retry overhaul (standalone, testable in isolation)
3. Concurrent debates (builds on retry fix)
4. Repository update function (standalone)
5. Orchestrator reorder + checkpoint (integrates everything)
6. Test updates last (verify all changes)

**Risk mitigation:**
- Each task produces a working codebase — no intermediate broken states
- The retry helper is extracted from existing working code, not written from scratch
- Concurrent debates are opt-in via `ai_debate_concurrency` (set to 1 to disable)

## Task Breakdown Preview

- [ ] Task 1: Add AI settings to `config.py` and wire timeout through `get_client()` factory
- [ ] Task 2: Extract shared `_run_agent_with_retry()` helper with backoff delays and error differentiation
- [ ] Task 3: Add semaphore-gated concurrent debate execution with phase timeout to `debate.py`
- [ ] Task 4: Add `update_scan_run()` to `repository.py`
- [ ] Task 5: Reorder pipeline phases — split persist into checkpoint + finalize, add health check gating
- [ ] Task 6: Update existing tests and add new tests for retry, concurrency, health check, and phase ordering

## Dependencies

### Internal (all within `src/option_alpha/`)
- `config.py` — Tasks 2-5 depend on Task 1 (new settings)
- `ai/agents.py` — Task 3 depends on Task 2 (retry helper)
- `persistence/repository.py` — Task 5 depends on Task 4 (`update_scan_run`)
- `pipeline/orchestrator.py` — Task 5 depends on Tasks 1-4

### External
- None — no new packages or services

## Success Criteria (Technical)

| Criteria | Validation |
|----------|------------|
| Zero data loss on debate failure | Integration test: kill LLM mid-debate, verify scores persisted |
| Retry backoff working | Unit test: mock client, assert `asyncio.sleep` called with `[2.0, 4.0, 8.0]` |
| Concurrent debates | Unit test: mock client, assert debates overlap in time |
| Health check gating | Unit test: mock `health_check()` returning False, assert debates skipped |
| Phase ordering correct | Unit test: assert `PHASE_NAMES` order and progress callback sequence |
| Happy-path parity | Existing integration tests pass unchanged (or with minimal ordering updates) |
| All 547+ tests pass | `pytest tests/` green |

## Estimated Effort

- **6 tasks**, each independently committable
- Tasks 1-4 are small (15-30 min each) — config, extraction, and plumbing
- Task 5 is medium (45-60 min) — orchestrator reorder with integration
- Task 6 is medium (30-45 min) — test updates
- **Total: ~3-4 hours of implementation**
- **Critical path**: Task 1 → Task 2 → Task 3, and Task 4 → Task 5

## Tasks Created
- [ ] #35 - Add AI settings to config and wire timeout through client factory (parallel: false)
- [ ] #36 - Extract shared agent retry helper with backoff and error differentiation (parallel: false, depends: #35)
- [ ] #37 - Add concurrent debate execution with phase timeout (parallel: false, depends: #35, #36)
- [ ] #38 - Add update_scan_run to repository (parallel: true)
- [ ] #39 - Reorder pipeline phases with checkpoint persist and health check gating (parallel: false, depends: #35, #37, #38)
- [ ] #40 - Update and add tests for retry, concurrency, health check, and phase ordering (parallel: false, depends: #35-#39)

Total tasks: 6
Parallel tasks: 1 (#38 can run alongside #35-#37)
Sequential tasks: 5
Estimated total effort: 3.25 hours
