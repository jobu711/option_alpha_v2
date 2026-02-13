---
name: expand-universe
description: Replace hardcoded ticker lists with a dynamic, database-driven universe featuring auto-discovery, tag-based organization, and a full CRUD management dashboard
status: backlog
created: 2026-02-13T20:50:16Z
---

# PRD: expand-universe

## Executive Summary

Replace the current hardcoded 702-ticker Python lists with a dynamic, database-driven universe management system. The system uses a flexible **tags model** where both preset categories and user-created watchlists are tags applied to tickers. A **hybrid auto-discovery engine** (CBOE optionable lists + yfinance screening) auto-includes newly found tickers with opt-out. A **full CRUD dashboard** at `/universe` (top-level nav, HTMX-powered) provides a single-page filtered table view with a search modal for adding tickers. Changes take effect on the next pipeline run.

**Phasing:** MVP delivers DB migration, service layer, pipeline integration, and a basic dashboard (ticker table + tag filters + add/remove). Phase 2 adds custom watchlists, auto-discovery, and expanded preset data.

## Problem Statement

The current universe is 702 tickers hardcoded across three Python lists (`SP500_CORE`, `POPULAR_OPTIONS`, `OPTIONABLE_ETFS`) in `data/universe.py`. This creates several problems:

1. **Adding or removing tickers requires code changes** — users can't customize what gets scanned without editing source files.
2. **No small-cap, international ADR, or thematic ETF coverage** — the lists are manually curated and stale.
3. **No auto-discovery** — new high-activity options stocks (e.g., IPOs, newly optionable tickers) are missed until someone manually adds them.
4. **No per-user customization** — every scan processes the same fixed list regardless of user interest.
5. **Config hints at dynamic features (`universe_presets`, `universe_sectors`) that were never implemented**, creating dead configuration paths.

This matters now because the pipeline is mature (scoring, catalysts, options, AI debate all work) and the universe is the primary bottleneck for scan relevance.

## User Stories

### US-1: Tag-Based Universe Organization
**As a** trader using Option Alpha,
**I want to** organize tickers using tags (both preset categories and my own custom tags),
**So that** I can filter, toggle, and manage groups of tickers flexibly.

**Acceptance Criteria:**
- Preset tags seeded on first run: "S&P 500", "Popular Options", "Optionable ETFs" (matching current hardcoded lists)
- Tags are many-to-many: a ticker can have multiple tags
- Tags can be toggled active/inactive — toggling a tag deactivates/activates all its tickers for scanning
- Both system preset tags and user-created tags use the same mechanism
- At least one active ticker must remain (prevent empty universe)

### US-2: Auto-Discovery of Optionable Stocks
**As a** trader,
**I want the system to** periodically discover new optionable stocks and auto-include them in my universe,
**So that** I don't miss newly optionable or high-activity tickers.

**Acceptance Criteria:**
- Hybrid approach: CBOE optionable stock list for discovery, yfinance for volume/price filtering
- Auto-discovery runs on a configurable schedule (default: weekly, Saturday)
- Screens by: has listed options, stock price > `min_price`, avg daily volume > `min_avg_volume`
- **Auto-include with opt-out**: discovered tickers are added as active with an "Auto-Discovered" tag
- User can deactivate or remove unwanted discoveries from the dashboard
- Deduplicates against existing universe tickers

### US-3: Custom Watchlists (Phase 2)
**As a** trader,
**I want to** create named watchlists (which are just user-created tags),
**So that** I can organize and scan specific groups (e.g., "earnings plays", "high IV", "my positions").

**Acceptance Criteria:**
- Users can create, rename, and delete custom tags/watchlists
- Tickers can be tagged via the dashboard or search modal
- A ticker can have multiple tags
- Custom tags can be toggled active/inactive for scans (same as preset tags)
- Bulk operations: tag multiple tickers, remove tag from all, import tickers from CSV/text

### US-4: Universe CRUD Dashboard
**As a** trader,
**I want a** dedicated UI page to search, add, remove, tag, and manage tickers,
**So that** I have full control over what gets scanned.

**Acceptance Criteria:**
- Top-level navigation item: "Universe"
- **Single-page layout** with sidebar tag filters and main ticker table
- Table columns: symbol, name, tags, status (active/inactive), last scanned date
- Inline toggle for active/inactive per ticker
- Sidebar: tag list with counts, click to filter, toggle active/inactive per tag
- Bulk select + bulk actions (activate, deactivate, tag, remove)
- Filter/sort by: tag, status, alphabetical, last scanned
- Shows total active ticker count
- **Search modal**: click "+" button → modal with typeahead search by symbol/company name → select ticker, assign tags, confirm
- HTMX-powered, Jinja2 templates, no JS build step

## Requirements

### Functional Requirements

#### FR-1: Database Schema (Tags Model)

Three core tables:

```
universe_tickers
├── symbol TEXT PRIMARY KEY
├── name TEXT
├── sector TEXT (nullable)
├── source TEXT ('preset' | 'discovered' | 'manual')
├── is_active INTEGER DEFAULT 1
├── created_at TEXT (ISO 8601)
└── last_scanned_at TEXT (nullable)

universe_tags
├── id INTEGER PRIMARY KEY AUTOINCREMENT
├── name TEXT UNIQUE
├── slug TEXT UNIQUE
├── is_preset INTEGER DEFAULT 0
├── is_active INTEGER DEFAULT 1
└── created_at TEXT (ISO 8601)

ticker_tags (join table)
├── symbol TEXT REFERENCES universe_tickers(symbol)
├── tag_id INTEGER REFERENCES universe_tags(tag_id)
└── PRIMARY KEY (symbol, tag_id)
```

**Indexes:**
- `universe_tickers(is_active)` — fast active universe query
- `universe_tags(slug)` — fast tag lookup
- `universe_tags(is_preset)` — fast preset vs custom filter
- `ticker_tags(tag_id)` — fast "all tickers with tag X" query

**Migration:** Seed from current hardcoded lists on first run. Map `SP500_CORE` → tag "S&P 500", `POPULAR_OPTIONS` → tag "Popular Options", `OPTIONABLE_ETFS` → tag "Optionable ETFs". All seeded tickers marked active.

#### FR-2: Universe Service Layer
- New module `data/universe_service.py` replacing direct use of `data/universe.py`
- **Core queries:**
  - `get_active_universe() -> list[str]`: all active tickers (union of tickers with any active tag + individually active tickers)
  - `get_tickers_by_tag(tag_slug) -> list[str]`
  - `get_all_tags() -> list[TagInfo]`
- **Mutations:**
  - `add_tickers(symbols: list[str], tags: list[str], source: str)`
  - `remove_tickers(symbols: list[str])`
  - `toggle_ticker(symbol: str, active: bool)`
  - `toggle_tag(tag_slug: str, active: bool)`
  - `create_tag(name: str) -> TagInfo`
  - `tag_tickers(symbols: list[str], tag_slug: str)`
  - `untag_tickers(symbols: list[str], tag_slug: str)`
- **Backward-compatible:** `get_full_universe()` reads from DB, returns same sorted list

#### FR-3: Auto-Discovery Engine (Phase 2)
- New module `data/discovery.py`
- **Step 1**: Download CBOE optionable equity list (authoritative, fast)
- **Step 2**: Filter through yfinance for price > `min_price`, avg volume > `min_avg_volume`
- **Step 3**: Deduplicate against existing `universe_tickers`
- **Step 4**: Insert new tickers with `source='discovered'`, `is_active=True`, tagged "Auto-Discovered"
- Configurable schedule via `universe_refresh_schedule` setting (default: Saturday)
- Config settings: `discovery_min_options_volume`, `discovery_min_price`, `discovery_min_avg_volume`

#### FR-4: Preset Tags & Seed Data
- **MVP presets** (migrated from current hardcoded lists):
  - "S&P 500" — 489 tickers from `SP500_CORE`
  - "Popular Options" — 155 tickers from `POPULAR_OPTIONS`
  - "Optionable ETFs" — 66 tickers from `OPTIONABLE_ETFS`
- **Phase 2 expanded presets:**
  - "Russell 2000" — top ~200 by options volume
  - "International ADRs" — ~50 high-liquidity ADRs (TSM, BABA, NVO, ASML, SAP, etc.)
  - "Thematic ETFs" — ~30 ETFs (ARKK, BITQ, BOTZ, HACK, LIT, TAN, etc.)

#### FR-5: Web UI — Universe Dashboard
- New route: `/universe`
- Top-level nav item in base template
- **Layout:** sidebar (tag filters + counts) | main area (toolbar + ticker table)
- **Toolbar:** search/filter input, "+" add button, bulk action dropdown
- **Table:** sortable columns, row checkbox for bulk select, inline active toggle
- **Search modal:** triggered by "+" button, typeahead by symbol/name, tag assignment, confirm
- **HTMX patterns:**
  - Tag toggle → `hx-post` to toggle endpoint → swap tag count + table
  - Inline active toggle → `hx-patch` → swap row
  - Bulk actions → `hx-post` with selected IDs → swap table
  - Search typeahead → `hx-get` with debounce → swap results dropdown
- Jinja2 templates in `web/templates/universe/`

#### FR-6: Pipeline Integration
- Orchestrator calls `get_active_universe()` instead of `get_full_universe()`
- Universe list is resolved once at scan start (natural snapshot — list doesn't change mid-scan since it's a local variable)
- Log active ticker count at scan start

#### FR-7: API Routes
- `GET /universe` — dashboard page
- `GET /api/universe/tickers` — list tickers (filterable by tag, status, search query)
- `POST /api/universe/tickers` — add ticker(s)
- `PATCH /api/universe/tickers/{symbol}` — toggle active, update tags
- `DELETE /api/universe/tickers/{symbol}` — remove ticker
- `GET /api/universe/tags` — list all tags with counts
- `POST /api/universe/tags` — create tag
- `PATCH /api/universe/tags/{slug}` — toggle tag active, rename
- `DELETE /api/universe/tags/{slug}` — delete tag (untags tickers, doesn't delete them)
- `POST /api/universe/tickers/bulk` — bulk activate/deactivate/tag/remove
- `GET /api/universe/search?q=` — typeahead search (searches known symbols + yfinance for unknown)

### Non-Functional Requirements

#### NFR-1: Performance
- `get_active_universe()` must return in < 50ms (indexed DB query)
- Dashboard page load < 500ms for 2,000+ tickers
- Typeahead search responds within 200ms for local results
- Auto-discovery completes within 10 minutes (Phase 2)

#### NFR-2: Data Integrity
- All universe mutations wrapped in SQLite transactions
- Prevent empty universe: `toggle_ticker` and `toggle_tag` reject if result would be 0 active tickers
- Deleting a tag untags its tickers but doesn't delete or deactivate them

#### NFR-3: Scalability
- Support up to 5,000 tickers without performance degradation
- Paginated table view (50 tickers per page default, configurable)
- Batch insert/update for bulk operations (no N+1 queries)

#### NFR-4: Backward Compatibility
- Existing config settings (`min_price`, `min_avg_volume`) continue to work as pipeline filters
- `get_full_universe()` returns same data (now from DB) for any code that calls it
- No breaking changes to pipeline phases downstream of universe
- Existing tests pass without modification

## Success Criteria

| Metric | Target |
|--------|--------|
| MVP universe size | 702 tickers (seeded from current lists, parity) |
| Phase 2 universe size | 1,500+ tickers (with expanded presets + discovery) |
| Preset tags available (MVP) | 3 toggleable tags |
| Preset tags available (Phase 2) | 6+ tags + auto-discovered |
| Dashboard load time | < 500ms for 2,000 tickers |
| User can add a ticker | < 3 clicks from dashboard |
| Pipeline compatibility | All existing tests pass with DB-backed universe |
| Service layer query time | < 50ms for `get_active_universe()` |

## Phasing

### Phase 1 — MVP
DB migration + service layer + pipeline integration + basic dashboard.

| Issue | Scope | Depends On |
|-------|-------|------------|
| **1. DB schema, migration, seeding** | Create tables, indexes, seed script from hardcoded lists. Seed runs automatically if tables don't exist. | — |
| **2. Universe service layer** | `universe_service.py` with all CRUD operations, `get_active_universe()`, backward-compat `get_full_universe()`. Unit tests. | #1 |
| **3. Pipeline integration** | Orchestrator uses `get_active_universe()`. Existing tests updated. | #2 |
| **4. API routes** | REST endpoints for tickers and tags (CRUD + bulk + search). | #2 |
| **5. Dashboard UI** | `/universe` page with tag sidebar, ticker table, search modal, inline toggles, bulk actions. HTMX + Jinja2. | #4 |

### Phase 2 — Expansion
Custom watchlists, auto-discovery, expanded presets.

| Issue | Scope | Depends On |
|-------|-------|------------|
| **6. Custom tag/watchlist management** | Create/rename/delete custom tags from UI, bulk tag operations, CSV import. | Phase 1 |
| **7. Auto-discovery engine** | CBOE list download, yfinance filtering, scheduled runs, auto-include with "Auto-Discovered" tag. | Phase 1 |
| **8. Expanded preset data** | Curate and seed Russell 2000 top options, international ADRs, thematic ETFs as new preset tags. | Phase 1 |

## Constraints & Assumptions

### Constraints
- **SQLite only** — no external database servers; must work with existing WAL-mode SQLite
- **No JS build step** — dashboard must use HTMX + Jinja2 (consistent with existing stack)
- **yfinance rate limits** — auto-discovery must batch requests and respect API throttling
- **Local-first** — all data stored locally; no cloud sync or multi-user scenarios

### Assumptions
- yfinance (or CBOE public data) provides sufficient data to identify optionable stocks
- The existing Parquet caching layer handles increased ticker count without modification
- SQLite performance is adequate for up to 5,000 tickers with proper indexing
- Current hardcoded lists in `universe.py` remain as seed data source (not deleted)

## Out of Scope

- **Multi-user support** — single-user local application; no user accounts or permissions
- **Real-time universe updates mid-scan** — changes apply on next scan only
- **International exchanges** — only US-listed securities (including ADRs)
- **Crypto spot trading** — only crypto ETFs, not direct crypto assets
- **Mobile-responsive dashboard** — desktop-first; mobile is not a priority
- **Universe sharing/import from external services** — no integration with brokerage watchlists
- **Scan snapshots** — deferred; universe is naturally snapshot as a local variable at scan start, no formal snapshot table needed for now

## Dependencies

### Internal
- **`persistence/` module** — new tables and repository methods
- **`pipeline/orchestrator.py`** — switch from `get_full_universe()` to `get_active_universe()`
- **`web/` module** — new routes, templates, and static assets for dashboard
- **`web/templates/base.html`** — add "Universe" to top-level navigation
- **`data/cache.py`** — must handle larger ticker counts (already batch-capable)
- **`config.py`** — new settings for discovery schedule and screening thresholds (Phase 2)

### External
- **yfinance** — used by auto-discovery for volume/price screening (Phase 2)
- **CBOE optionable lists** — primary discovery data source (Phase 2)
- **SQLite** — already in use; no new external dependency
- **HTMX** — already in use; no new external dependency
