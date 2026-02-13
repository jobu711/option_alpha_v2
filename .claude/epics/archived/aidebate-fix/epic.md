---
name: aidebate-fix
status: completed
created: 2026-02-13T12:41:46Z
updated: 2026-02-13T13:44:11Z
progress: 100%
prd: .claude/prds/aidebate-fix.md
github: https://github.com/jobu711/option_alpha_v2/issues/71
---

# Epic: aidebate-fix

## Overview

Fix the AI debate system so it reliably completes on local Ollama with llama3.1:8b / 8-16GB hardware. Currently 100% of debates timeout due to aggressive defaults (60s budget for 3 sequential agents, concurrency=3 causing GPU contention, ~3000-token context, 5 retries burning time). The fix changes defaults to be Ollama-friendly, compresses context, lightens health checks, and adds per-ticker time budgeting.

## Architecture Decisions

1. **Backend-aware effective defaults via helper method** — Rather than changing `Settings` field defaults (which would break users who haven't set `config.json` and expect current Claude behavior), add a `get_effective_ai_settings()` method that returns backend-appropriate values. Users who explicitly set values in `config.json` still override everything.

2. **Time budget passed via `asyncio.wait_for` per agent** — Instead of a complex budget-tracking class, use simple elapsed-time arithmetic in `DebateManager.run_debate()` to compute remaining time per agent and pass it as the `timeout` kwarg to `asyncio.wait_for()` wrapping each agent call.

3. **Context compression via top-N indicators** — Modify `build_context()` to select the top 6 indicators by weight (already available in `ScoreBreakdown.weight`), use compact key-value lines instead of padded table formatting. This is the highest-impact change for Ollama performance.

4. **Lightweight Ollama health check** — Replace the full `/api/generate` call with just checking `/api/tags` response for the configured model name. This avoids a cold model load during health check.

5. **Unify JSON hint approach** — `ClaudeClient` currently dumps the full JSON schema; switch it to use the same `_build_example_hint()` already used by `OllamaClient`.

## Technical Approach

### Files Modified

| File | Changes |
|------|---------|
| `src/option_alpha/config.py` | Add `get_effective_ai_settings()` helper; adjust Ollama defaults |
| `src/option_alpha/ai/clients.py` | Lighten Ollama health check; use `_build_example_hint()` in Claude client |
| `src/option_alpha/ai/context.py` | Compress `build_context()`: top-6 indicators, compact format, remove trailing prompt |
| `src/option_alpha/ai/debate.py` | Add per-agent time budget in `run_debate()`; use effective settings in `run_debates()` |
| `src/option_alpha/ai/agents.py` | Accept optional `timeout` param in agent functions for budget-aware calls |
| `tests/test_ai.py` | Update tests for new context format, health check behavior, budget logic |

### No New Files

All changes are modifications to existing files. No new modules, classes, or abstractions needed.

## Implementation Strategy

Changes are ordered to minimize risk and allow incremental testing:

1. **Config defaults first** — Safe change, no behavior change for Claude users
2. **Health check + JSON hints** — Small, isolated changes in `clients.py`
3. **Context compression** — Biggest impact on Ollama performance, most test updates
4. **Time budget + progress** — Core reliability fix in `debate.py`
5. **Tests last** — Update all tests to match new behavior, run full suite

## Task Breakdown Preview

- [ ] #72: Backend-aware config defaults (`config.py`) — FR-1, FR-6
- [ ] #73: Lighten Ollama health check + unify Claude JSON hints (`clients.py`) — FR-4, FR-5
- [ ] #74: Compress context builder (`context.py`) — FR-3
- [ ] #75: Per-agent time budget + effective settings in debate runner (`debate.py`, `agents.py`) — FR-2, FR-7
- [ ] #76: Update tests for all changes (`tests/test_ai.py`) — NFR-3

## Dependencies

- **Internal**: All changes are within `src/option_alpha/ai/` and `config.py`
- **No external dependencies added**
- **Test suite**: 925+ existing tests must pass after all changes

## Success Criteria (Technical)

1. All 925+ tests pass (including updated AI tests)
2. Default Ollama config: `concurrency=1`, `per_ticker_timeout=180`, `retry_delays=[2, 4]`
3. Context output is ~40-50% shorter (measured by character count in tests)
4. Ollama health check does NOT call `/api/generate`
5. Claude client uses `_build_example_hint()` instead of full JSON schema

## Tasks Created

- [ ] #72 - Backend-aware config defaults (parallel: true)
- [ ] #73 - Lighten health check and unify JSON hints (parallel: true)
- [ ] #74 - Compress context builder (parallel: true)
- [ ] #75 - Per-agent time budget and effective settings in debate runner (parallel: false, depends on #72)
- [ ] #76 - Comprehensive test updates (parallel: false, depends on #72-#75)

Total tasks: 5
Parallel tasks: 3 (#72, #73, #74 can run concurrently)
Sequential tasks: 2 (#75 after #72; #76 after all)
Estimated total effort: S + S + M + M + M

## Estimated Effort

- **5 tasks**, each completable in a single focused session
- **Critical path**: #72 (config) → #75 (debate runner) → #76 (tests)
- **Parallel batch**: #72, #73, #74 can all run concurrently
- **Risk**: Context compression (#74) requires the most test updates due to assertions on specific formatting strings
