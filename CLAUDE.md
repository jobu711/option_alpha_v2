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
  data/             # fetcher, universe, universe_service, cache
  scoring/          # indicators, normalizer, composite (weighted geometric mean)
  catalysts/        # earnings proximity with exponential decay
  options/          # chains, greeks, recommender
  ai/               # clients, agents, context, debate
  pipeline/         # orchestrator, progress (WebSocket)
  persistence/      # database, repository
  web/              # app factory, routes, universe_routes, websocket, templates/, static/
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
- **AI debate**: On-demand via `POST /debate`, not part of scan pipeline. Sequential Bull -> Bear -> Risk (each needs prior context)
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

## Universe Management System

The ticker universe is database-driven (not hardcoded). The hardcoded lists in `data/universe.py` serve only as seed data.

### Database schema (`persistence/migrations/002_universe.sql`)
- **`universe_tickers`** — symbol (PK), name, sector, source, is_active, created_at, last_scanned_at
- **`universe_tags`** — id (PK), name, slug, is_preset, is_active, created_at
- **`ticker_tags`** — many-to-many join table (symbol FK, tag_id FK)

### Service layer (`data/universe_service.py`)
- All functions take `conn: sqlite3.Connection` as first parameter (same pattern as `repository.py`)
- **Queries:** `get_active_universe()`, `get_full_universe()` (backward-compat), `get_tickers_by_tag()`, `get_all_tags()`
- **Mutations:** `add_tickers()`, `remove_tickers()`, `toggle_ticker()`, `toggle_tag()`, `create_tag()`, `delete_tag()`, `tag_tickers()`, `untag_tickers()`
- **Seeding:** `seed_universe()` auto-populates from hardcoded lists on first run (idempotent)
- **Empty universe prevention:** `toggle_ticker()` and `toggle_tag()` use SAVEPOINTs to reject deactivations that would leave 0 active tickers

### Pipeline integration
- Orchestrator calls `get_active_universe(conn)` instead of `get_full_universe()`
- Universe resolves once at scan start (natural snapshot)

### Web routes (`web/universe_routes.py`)
- `GET /universe` — dashboard page with tag sidebar + ticker table
- `GET/POST/PATCH/DELETE /api/universe/tickers` — ticker CRUD
- `GET/POST/PATCH/DELETE /api/universe/tags` — tag CRUD
- `POST /api/universe/tickers/bulk` — bulk activate/deactivate/tag/remove
- `GET /api/universe/search?q=` — typeahead search
- All endpoints return HTMX partials for `HX-Request`, JSON otherwise

### Dashboard UI (`web/templates/universe/`)
- HTMX + Alpine.js powered (no JS build step)
- Tag sidebar with toggle, sortable/paginated ticker table, search modal

## On-Demand Debate System

AI debates are **user-initiated**, not part of the scan pipeline. Users select tickers via checkboxes on the dashboard and click "Debate Selected" to trigger debates.

### Trigger flow
1. User checks tickers in candidates table (checkboxes with `class="debate-check"`)
2. "Debate Selected (N)" button collects checked symbols
3. `fetch('/debate', {method: 'POST', body: JSON.stringify({symbols: [...]})})` sends request
4. `POST /debate` endpoint runs `DebateManager.run_debate()` per ticker sequentially
5. Results persisted to `ai_theses` table (DELETE-then-INSERT for "always fresh" semantics)
6. `_debate_results.html` partial returned and swapped into `#debate-results` container

### Guard conditions (`web/routes.py`)
- `_scan_running` → 409 (can't debate while scanning)
- `_debate_running` → 409 (can't run concurrent debates)
- Empty/invalid symbols → 400
- Symbols not in latest scan → 400

### Templates
- **`_candidates_table.html`** — checkbox column (first col) with select-all toggle; `event.stopPropagation()` prevents row navigation on checkbox click
- **`dashboard.html`** — "Debate Selected" button with count indicator, `#debate-results` container, vanilla JS for checkbox state + fetch
- **`_debate_results.html`** — debate cards with conviction badges (green/yellow/red), direction badges, fallback styling for conservative defaults

## Pipeline Phase Order

1. Data fetch (yfinance -> Parquet cache)
2. Technical scoring (indicators -> percentile normalize -> geometric mean)
3. Catalyst detection (earnings dates -> exponential decay)
4. Options analysis (top 50 chains -> Greeks -> liquidity filter -> recommendation)
5. Persist (SQLite)
6. Display (FastAPI dashboard)
