# CLAUDE.md

> Think carefully and implement the most concise solution that changes as little code as possible.

## Project Overview

**Option Alpha** is an AI-powered local Python web application that scans optionable stocks for catalyst-driven breakout setups. It scores candidates using technical analysis, runs multi-agent LLM debate (Bull/Bear/Risk agents), and recommends options contracts with full Greek analysis.

## Tech Stack

- **Python 3.11+** with `hatchling` (src-layout: `src/option_alpha/`)
- **FastAPI** + Jinja2 + HTMX (server-rendered, no JS build step)
- **SQLite** with WAL mode for persistence
- **yfinance** for market data, **py_vollib** for Greeks
- **Ollama** (local) or **Anthropic Claude API** (cloud) for LLM backends
- **Pydantic** models throughout; structured LLM output via SDK-native tool use
- **Parquet** for OHLCV caching, **JSON** for metadata/config

## Package Structure

```
src/option_alpha/
  __main__.py       # Entry: python -m option_alpha (starts FastAPI on 127.0.0.1:8000)
  config.py         # Pydantic Settings, loads from config.json
  models.py         # Shared Pydantic models (TickerScore, TradeThesis, etc.)
  data/             # fetcher, universe, cache
  scoring/          # indicators, normalizer, composite (weighted geometric mean)
  catalysts/        # earnings proximity with exponential decay
  options/          # chains, greeks, recommender
  ai/               # clients, agents, context, debate
  pipeline/         # orchestrator, progress (WebSocket)
  persistence/      # database, repository
  web/              # app factory, routes, websocket, templates/, static/
  backtest/         # runner, metrics
```

## Testing

Always run tests before committing:
```bash
pytest tests/
```
- Tests use **pytest** with mocked external APIs (yfinance, Ollama, FRED)
- Test config: `pyproject.toml` -> `[tool.pytest.ini_options]` with `testpaths = ["tests"]`, `pythonpath = ["src"]`
- All test files are in `tests/` (flat structure, `test_*.py`)

## Code Style

- Follow existing patterns in the codebase
- Type hints throughout
- Pydantic models for all data structures
- Use `pathlib.Path` for file paths
- Config via `Settings.load()` from `config.py` (Pydantic BaseSettings with env prefix `OPTION_ALPHA_`)

## Configuration

- **Root config**: `config.json` (user-facing defaults)
- **Internal config**: `src/option_alpha/config.json` (extended scoring weights + AI/fetch settings)
- Settings support env vars with `OPTION_ALPHA_` prefix and `.env` file

## Key Design Decisions

- **Composite scoring**: Weighted geometric mean (penalizes poor individual scores)
- **AI debate**: Sequential Bull -> Bear -> Risk (each needs prior context)
- **AI clients**: SDK-based (`ollama` and `anthropic` packages) with tool use for structured output
- **Parse failure default**: `no_trade`, `conviction=3` (conservative)
- **Frontend**: Server-rendered Jinja2 + HTMX (no JS build step)
- **Security**: Binds to 127.0.0.1 only; API keys in env vars or config, never in code

## AI Debate System (PRESERVE THIS STRUCTURE)

> **WARNING**: The `ai/` module is stable after extensive iteration. Do not refactor, reorganize, or change the call signatures, structured output approach, or agent flow without explicit approval. Small bugs can be fixed, but the architecture must stay as-is.

### Module layout (`ai/`)
- **`clients.py`** — `LLMClient` ABC with `OllamaClient` and `ClaudeClient` implementations
- **`agents.py`** — Bull, Bear, Risk agent functions with system prompts and retry logic
- **`context.py`** — `build_context()` builds ~2000-token structured text from `TickerScore` + `OptionsRecommendation`
- **`debate.py`** — `DebateManager` orchestrates sequential Bull → Bear → Risk flow

### How Ollama structured output works
1. `OllamaClient.complete()` accepts an optional `response_model: type[BaseModel]`
2. When a model is provided, the Pydantic JSON schema is passed as `format=` to `ollama.AsyncClient.chat()` (Ollama's native structured output)
3. A concrete example hint is appended to the last user message via `_build_example_hint()` to improve compliance
4. The raw JSON text response is parsed with `json.loads()` then validated with `response_model.model_validate(data)`

### Agent flow (sequential, not parallel)
1. **Bull** receives ticker context → returns `AgentResponse` (analysis, key_points, conviction)
2. **Bear** receives context + bull's analysis → returns `AgentResponse` countering the bull case
3. **Risk** receives context + both analyses → returns `TradeThesis` (direction, conviction, entry_rationale, risk_factors, recommended_action)

### Retry and fallback
- `_call_with_retry()` in `agents.py`: one retry on parse/validation errors (appends fix hint), one retry with 2s delay on network errors
- On total failure: conservative fallback (`direction=neutral`, `conviction=3`, `recommended_action="No trade"`)
- Per-ticker timeout splits equally across 3 agents

### Key constraints — do not change
- Agents run **sequentially** (Bear needs Bull output, Risk needs both) — do not parallelize
- `_build_example_hint()` in `clients.py` is critical for Ollama compliance — do not remove
- `format=response_model.model_json_schema()` is how Ollama enforces JSON structure — do not switch to string prompting
- Claude client separates system messages from conversation (Anthropic API requirement) — do not merge them
- Fallback defaults are intentionally conservative (`no_trade`, `conviction=3`) — do not make them optimistic

## Pipeline Phase Order

1. Data fetch (yfinance -> Parquet cache)
2. Technical scoring (indicators -> percentile normalize -> geometric mean)
3. Catalyst detection (earnings dates -> exponential decay)
4. Options analysis (top 50 chains -> Greeks -> liquidity filter -> recommendation)
5. AI debate (top 10 -> Bull -> Bear -> Risk -> structured thesis)
6. Persist (SQLite)
7. Display (FastAPI dashboard)
