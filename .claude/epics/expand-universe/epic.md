---
name: expand-universe
status: backlog
created: 2026-02-13T21:12:37Z
progress: 0%
prd: .claude/prds/expand-universe.md
updated: 2026-02-13T21:24:02Z
github: https://github.com/jobu711/option_alpha_v2/issues/84
---

# Epic: expand-universe

## Overview

Replace the hardcoded 702-ticker Python lists in `data/universe.py` with a database-driven universe using a tags model (many-to-many), a service layer, REST API, and an HTMX dashboard at `/universe`. This epic covers **Phase 1 (MVP)** only. Phase 2 (auto-discovery, custom watchlists, expanded presets) is deferred to a follow-up epic.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | SQLite (existing DB) | No new dependencies; WAL mode already configured; 5K tickers trivial for SQLite |
| Schema | 3 tables (tickers, tags, join) | Clean many-to-many; tags unify presets and future custom watchlists |
| Migration | New `002_universe.sql` | Follows existing migration pattern in `persistence/migrations/` |
| Seeding | Import from existing hardcoded lists | `SP500_CORE`, `POPULAR_OPTIONS`, `OPTIONABLE_ETFS` become preset tags; exact parity with current behavior |
| Service layer | New `data/universe_service.py` | Single module for all universe CRUD; keeps `data/universe.py` intact as seed data source |
| Backward compat | `get_full_universe()` delegates to DB | Existing code that calls `get_full_universe()` continues to work unchanged |
| UI framework | HTMX + Jinja2 (existing stack) | Consistent with dashboard; no JS build step |
| Pipeline change | Swap `get_full_universe()` → `get_active_universe()` | One-line change in orchestrator; universe snapshot is natural (local variable) |

## Technical Approach

### Database Layer
- New migration `002_universe.sql`: creates `universe_tickers`, `universe_tags`, `ticker_tags` tables with indexes
- Seeding logic in `universe_service.py`: on first call, if `universe_tickers` is empty, auto-seed from hardcoded lists with 3 preset tags
- All mutations wrapped in transactions; prevent empty universe via validation

### Service Layer (`data/universe_service.py`)
- **Queries:** `get_active_universe()`, `get_full_universe()` (backward-compat), `get_tickers_by_tag()`, `get_all_tags()`
- **Mutations:** `add_tickers()`, `remove_tickers()`, `toggle_ticker()`, `toggle_tag()`, `create_tag()`, `tag_tickers()`, `untag_tickers()`
- Takes `db_path` from settings; uses same `sqlite3` patterns as existing `repository.py`

### Pipeline Integration
- `orchestrator.py` line ~207: replace `get_full_universe()` with `get_active_universe(db_path)`
- Log active ticker count at scan start
- No other pipeline changes needed

### API Routes (`web/routes.py` or new `web/universe_routes.py`)
- REST endpoints for ticker CRUD, tag CRUD, bulk operations, and typeahead search
- Returns HTMX partials for dashboard interactions, JSON for API consumers
- Follows existing route patterns (FastAPI router, Jinja2 templates)

### Dashboard UI (`web/templates/universe/`)
- Single-page layout: sidebar (tag filters + counts) | main area (toolbar + ticker table)
- Sortable, paginated ticker table with inline active/inactive toggle
- Search modal with typeahead for adding tickers
- Bulk select + bulk actions (activate, deactivate, tag, remove)
- Add "Universe" link to `base.html` navigation

## Implementation Strategy

- **Sequential dependency chain:** schema → service → pipeline → API → UI
- **Risk mitigation:** Seed from hardcoded lists ensures exact parity; backward-compat `get_full_universe()` prevents breakage
- **Testing:** Each task includes unit tests; pipeline integration test validates end-to-end

## Task Breakdown Preview

- [ ] Task 1: DB schema migration + auto-seeding from hardcoded lists
- [ ] Task 2: Universe service layer (CRUD operations + backward-compat wrapper)
- [ ] Task 3: Pipeline integration (orchestrator swap + logging)
- [ ] Task 4: REST API routes (ticker/tag CRUD + bulk + search endpoints)
- [ ] Task 5: Dashboard UI (templates, HTMX interactions, nav update)

## Dependencies

### Internal
- `persistence/database.py` — migration runner (add new migration file)
- `persistence/migrations/` — new `002_universe.sql`
- `data/universe.py` — seed data source (hardcoded lists remain as-is)
- `pipeline/orchestrator.py` — one-line universe call swap
- `web/app.py` — register new router
- `web/templates/base.html` — add nav link
- `config.py` — `db_path` setting (already exists)

### External
- None for MVP. Phase 2 will need CBOE data + yfinance for auto-discovery.

## Success Criteria (Technical)

| Metric | Target |
|--------|--------|
| Seeded universe size | 702 tickers (exact parity with hardcoded lists) |
| Preset tags created | 3 ("S&P 500", "Popular Options", "Optionable ETFs") |
| `get_active_universe()` query time | < 50ms |
| Dashboard load time | < 500ms for 700+ tickers |
| Existing tests | All pass without modification |
| Backward compatibility | `get_full_universe()` returns identical sorted list |

## Estimated Effort

| Task | Size | Est. Hours |
|------|------|------------|
| 1. DB schema + seeding | S | 2-3 |
| 2. Service layer | M | 4-5 |
| 3. Pipeline integration | S | 1-2 |
| 4. API routes | M | 3-4 |
| 5. Dashboard UI | L | 5-7 |
| **Total** | | **15-21 hours** |

**Critical path:** Tasks 1 → 2 → 3 (sequential). Tasks 4 and 5 depend on Task 2 but are independent of Task 3.

## Tasks Created

- [ ] #85 - DB schema, migration, and seeding (parallel: false)
- [ ] #86 - Universe service layer (parallel: false)
- [ ] #87 - Pipeline integration (parallel: true)
- [ ] #88 - REST API routes for universe CRUD (parallel: true)
- [ ] #89 - Universe dashboard UI (parallel: false)

Total tasks: 5
Parallel tasks: 2 (#87, #88 — can run concurrently after #86 completes)
Sequential tasks: 3 (#85 → #86 → #89)
Estimated total effort: 15-21 hours
