# CLAUDE.md

> Think carefully and implement the most concise solution that changes as little code as possible.

## Project Overview

Option Alpha is an AI-powered options scanner with multi-agent debate. Python 3.11+ application using FastAPI, yfinance, and Pydantic.

**Architecture**: 6-phase pipeline — Data Fetch → Scoring → Catalysts → Options → Persist (checkpoint) → AI Debate

## Quick Reference

```bash
# Install
pip install -e ".[all,dev]"

# Run the app (starts FastAPI dashboard at http://127.0.0.1:8000)
python -m option_alpha

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=option_alpha
```

## Project Structure

```
src/option_alpha/          # Main package
├── __main__.py            # Entry point (python -m option_alpha)
├── config.py              # Pydantic Settings + config.json persistence
├── models.py              # Shared Pydantic models
├── ai/                    # Multi-agent LLM system (Bull/Bear/Risk agents)
├── backtest/              # Backtesting engine
├── catalysts/             # Earnings date detection
├── data/                  # yfinance fetcher, parquet cache, ticker universe
├── options/               # Option chains, Greeks, contract recommendations
├── persistence/           # SQLite database + repository pattern
├── pipeline/              # 6-phase scan orchestrator + progress tracking
├── scoring/               # Technical indicators + weighted geometric mean
└── web/                   # FastAPI app, routes, WebSocket, Jinja2 templates
tests/                     # 23 test files, 925+ tests (pytest)
```

## Dependencies

All dependencies declared in `pyproject.toml` (single source of truth). Key groups:
- **Core**: FastAPI, uvicorn, yfinance, pandas, numpy, pydantic, pydantic-settings, httpx, pyarrow, apscheduler
- **scoring**: pandas_ta (requires Python <3.14 due to numba)
- **options**: py_vollib (Black-Scholes Greeks)
- **dev**: pytest, pytest-cov

## Testing

Always run tests before committing:

```bash
pytest tests/
```

- Test framework: **pytest** with `pythonpath = ["src"]` and `testpaths = ["tests"]`
- 23 test files covering all modules: config, data layer, scoring, options, AI, pipeline, web, persistence, universe, integration
- Tests use mocking extensively — no live API calls in tests
- One env-dependent test (`test_check_ollama_when_ollama_not_running`) may fail if Ollama is running locally — this is expected

## Code Style

- Follow existing patterns in the codebase
- No linter/formatter configured — maintain consistency with surrounding code
- Use type hints throughout
- Use Pydantic models for data structures and validation
- Configuration via `Settings` class in `config.py` (env vars with `OPTION_ALPHA_` prefix)

## Key Patterns

- **Factory pattern**: `create_app(config)` in `web/app.py`
- **Repository pattern**: `persistence/repository.py` for database access
- **Pipeline pattern**: Sequential phases in `pipeline/orchestrator.py` with checkpoint persist before AI debate
- **Checkpoint persist**: Scores saved with `PARTIAL` status before debate; `update_scan_run()` finalizes to `COMPLETED` after debate
- **Health check gating**: LLM `health_check()` called before debate phase; debates skipped if unhealthy
- **Agent retry**: Shared `_run_agent_with_retry()` in `ai/agents.py` with exponential backoff (`ai_retry_delays`); parse errors retry immediately, network errors sleep
- **Concurrent debates**: `asyncio.Semaphore`-gated concurrency in `ai/debate.py` (`ai_debate_concurrency`, default 3)
- **Caching**: Parquet files in `data/cache/` with 18-hour freshness
- **AI backends**: Ollama (local) or Claude (API), configured via `config.json`
- **AI settings**: `ai_retry_delays`, `ai_request_timeout`, `ai_debate_phase_timeout`, `ai_debate_concurrency` in `config.py`
- **Universe refresh**: `universe_refresh.py` fetches tickers from SEC EDGAR, validates OI via yfinance, writes `universe_data.json` atomically (`.tmp` → `.bak` → rename). ETFs exempt from OI gate. Two thresholds: `min_universe_oi` (universe inclusion, default 100) vs `min_open_interest` (options filtering)
- **Scheduled tasks**: APScheduler `BackgroundScheduler` in FastAPI lifespan runs weekly universe refresh (`universe_refresh_interval_days`)
