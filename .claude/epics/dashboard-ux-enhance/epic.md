---
name: dashboard-ux-enhance
status: completed
created: 2026-02-14T13:13:18Z
completed: 2026-02-14T14:30:00Z
progress: 100%
prd: .claude/prds/dashboard-ux-enhance.md
github: https://github.com/jobu711/option_alpha_v2/issues/97
---

# Epic: dashboard-ux-enhance

## Overview

Improve the Option Alpha dashboard across four areas: universe management (tag bug fix, ticker counts, advanced filtering), scan workflow (subset scanning, ETA estimation), and visual polish (empty states, table hierarchy). All changes stay within the existing Jinja2 + HTMX + Alpine.js stack with no new dependencies.

## Architecture Decisions

- **No new JS frameworks or build tools** — all UI changes use existing HTMX, Alpine.js, and vanilla JS patterns
- **Filter logic in SQL, not Python** — universe filtering uses parameterized SQL WHERE clauses built in `universe_routes.py` (extending the existing pattern at lines 148-171), not new service-layer functions
- **Subset scan via optional parameter** — add `ticker_subset: list[str] | None` to `run_scan()` rather than creating a separate scan pathway; keeps the pipeline unified
- **ETA from elapsed extrapolation** — calculate ETA from `(elapsed / percentage) * (100 - percentage)` using live scan progress, avoiding the need for a new `scan_history` table. The existing `scan_runs` table already stores `duration_seconds` and `ticker_count` which can be used as a secondary reference
- **Tag sidebar fix via targeted HTMX swap** — change `hx-target` from `#tag-sidebar` (full sidebar replace) to `closest .tag-item` with `outerHTML` swap, so only the toggled tag re-renders
- **Reusable empty state partial** — single `_empty_state.html` Jinja2 macro with parameters (icon, title, message, cta_text, cta_url) used everywhere

## Technical Approach

### Frontend Components

**Templates to modify:**
- `universe/_tag_sidebar.html` — fix `●`/`○` HTML entity rendering, change HTMX swap target
- `universe/universe.html` — add ticker count header, filter bar, "Scan This Tag" / "Scan Selected" buttons
- `_candidates_table.html` — score bar, direction icons, top-N highlight, debated badge, better empty state
- `_progress.html` — add ETA display below progress bar
- `dashboard.html` — active ticker count near "Run Scan", scan dropdown (All / By Tag), empty state for no-scan/no-debates
- New partial: `_empty_state.html` — reusable empty state component

**CSS additions (`style.css`):**
- `.score-bar` — colored fill bar (red→yellow→green gradient based on score 0-100)
- `.direction-icon` — arrow icons with semantic colors
- `.top-candidate` — subtle gold left-border highlight
- `.debated-badge` — small pill indicating debate exists
- `.empty-state` variants — centered layout with icon, message, CTA button
- `.filter-bar` — horizontal filter row with dropdowns
- `.ticker-count` — prominent counter component
- `.tag-dot` — CSS-only status dot replacing HTML entities

### Backend Services

**Routes (`web/routes.py`):**
- `POST /scan` — accept optional JSON body `{symbols: [...], tag: "slug"}` to trigger subset scan; resolve tag to symbols server-side via `get_tickers_by_tag()`
- Dashboard context — add `active_ticker_count` and `total_ticker_count` to template context

**Routes (`web/universe_routes.py`):**
- `GET /api/universe/tickers` — add `sector` and `last_scanned` query params to existing filter logic (extend WHERE clause at lines 148-171)
- `PATCH /api/universe/tags/{slug}` — return single `.tag-item` div instead of full sidebar HTML
- New: `GET /api/universe/sectors` — return distinct sector list for filter dropdown (simple SQL query)

**Pipeline (`pipeline/orchestrator.py`):**
- `run_scan(ticker_subset=None, on_progress=None)` — when `ticker_subset` provided, use it instead of `get_active_universe()` at line 198
- Pass subset through to `_phase_data_fetch()`

**Progress (`pipeline/progress.py`):**
- Add `eta_seconds: float | None = None` to `ScanProgress` model
- Add `ticker_count: int = 0` to `ScanProgress` for "Scanning N tickers" display
- Calculate ETA in orchestrator after first phase completes: `remaining = (elapsed / pct) * (100 - pct)`

### Infrastructure

- **No new database tables** — the existing `scan_runs` table already has `duration_seconds` and `ticker_count`; ETA calculation uses live elapsed time extrapolation
- **No new migrations needed for filtering** — `universe_tickers` already has `sector` and `last_scanned_at` columns
- **Optional migration** (`003_scan_subset.sql`) — add `scan_type TEXT DEFAULT 'full'` and `ticker_subset_json TEXT` columns to `scan_runs` if we want to distinguish subset scans in history

## Implementation Strategy

### Phase 1: Universe Page Fixes (lowest risk, highest impact)
- Fix tag sidebar `●` bug (CSS dots instead of HTML entities)
- Add ticker count header component
- Add advanced filter bar (sector, tag, status, last-scanned)

### Phase 2: Visual Polish (frontend-only, no backend changes)
- Reusable empty state partial
- Contextual empty states across dashboard/universe/debates
- Candidates table enhancements (score bar, direction icons, top-N, debated badge)

### Phase 3: Subset Scanning (backend + frontend)
- Orchestrator accepts ticker subset
- Route accepts symbols/tag parameters
- Dashboard dropdown + universe page buttons

### Phase 4: Scan ETA (independent, can ship alone)
- Progress model ETA field
- Calculation logic in orchestrator
- Template display

### Testing Approach
- Unit tests for new filter SQL logic (mock SQLite)
- Unit test for ETA calculation (given elapsed + percentage, verify ETA)
- Unit test for subset scan parameter passthrough
- Existing scan pipeline tests should pass unchanged (backwards-compatible)

## Task Breakdown Preview

- [ ] **Task 1: Fix tag sidebar display and HTMX swap** — Replace `●`/`○` HTML entities with CSS-styled dots; change tag toggle `hx-target` to swap single `.tag-item` instead of full sidebar; update backend to return single tag item partial
- [ ] **Task 2: Add active/total ticker count display** — Add count header to universe page and near "Run Scan" on dashboard; query counts via existing `universe_service` functions; update on HTMX interactions via OOB swap
- [ ] **Task 3: Implement advanced universe filtering** — Add sector, last-scanned filter params to `GET /api/universe/tickers`; add filter bar UI with dropdowns; add `GET /api/universe/sectors` endpoint; preserve filters during pagination
- [ ] **Task 4: Add contextual empty states** — Create reusable `_empty_state.html` partial; replace generic "No results" in candidates table, debate results, and universe table with contextual messages and CTAs
- [ ] **Task 5: Improve candidates table visual hierarchy** — Add colored score bar, direction icons (▲▼●), top-5 gold highlight, debated badge; improve sort indicators
- [ ] **Task 6: Add subset scanning** — Accept `symbols`/`tag` in `POST /scan`; pass `ticker_subset` through to orchestrator; add "Scan by Tag" dropdown on dashboard and "Scan Selected"/"Scan This Tag" buttons on universe page; show subset scope in progress
- [ ] **Task 7: Add scan ETA estimation** — Add `eta_seconds` and `ticker_count` to `ScanProgress` model; calculate ETA from elapsed/percentage after first phase; display in `_progress.html`; add migration for subset scan metadata

## Dependencies

**Internal (by task):**
- Task 1: `universe/_tag_sidebar.html`, `universe_routes.py` (PATCH handler), `style.css`
- Task 2: `universe/universe.html`, `dashboard.html`, `routes.py`, `universe_routes.py`
- Task 3: `universe_routes.py`, `universe/universe.html`, `style.css`
- Task 4: New `_empty_state.html`, `_candidates_table.html`, `dashboard.html`, `_debate_results.html`
- Task 5: `_candidates_table.html`, `style.css`, `routes.py` (pass debate status to template)
- Task 6: `orchestrator.py`, `routes.py`, `dashboard.html`, `universe/universe.html`
- Task 7: `progress.py`, `orchestrator.py`, `_progress.html`, optional `003_scan_subset.sql`

**Task dependencies:**
- Tasks 1-5 are independent of each other (can be done in any order)
- Task 6 depends on Task 2 (ticker count display shows subset scope)
- Task 7 is fully independent

**External:** None — all changes are internal

## Success Criteria (Technical)

| Criteria | Measurement |
|----------|-------------|
| Tag sidebar renders CSS dots, no `●` characters | Visual inspection |
| Tag toggle swaps single item, no sidebar flash | No scroll jump on toggle |
| Ticker count visible on universe page and dashboard | Accurate, updates on every HTMX interaction |
| Filter bar combines sector + tag + status + last-scanned | AND logic, < 200ms response |
| All empty states have contextual messages + CTAs | Zero generic "No results" text |
| Score bar + direction icons render in candidates table | Visual inspection |
| `POST /scan` with `{symbols: [...]}` scans only those tickers | Pipeline receives subset, results correct |
| ETA shown after first phase, within 30% accuracy | Compare ETA to actual remaining time |
| All existing tests pass | `pytest tests/` green |

## Estimated Effort

- **Task 1** (Tag fix): Small — template + CSS + minor route change
- **Task 2** (Ticker count): Small — query + template additions
- **Task 3** (Filtering): Medium — SQL filter logic + filter bar UI + pagination state
- **Task 4** (Empty states): Small — new partial + replace 4 empty states
- **Task 5** (Table polish): Medium — CSS + template + route context changes
- **Task 6** (Subset scan): Medium — backend parameter threading + frontend UI
- **Task 7** (Scan ETA): Small-Medium — model change + calculation + template

**Total: 7 tasks**, estimated 3-4 focused sessions. Critical path: Tasks 1-3 (universe fixes) should ship first as they address the most visible pain points.

## Tasks Created
- [ ] #100 - Fix tag sidebar display and HTMX swap (parallel: true)
- [ ] #102 - Add active/total ticker count display (parallel: true)
- [ ] #103 - Implement advanced universe filtering (parallel: true)
- [ ] #104 - Add contextual empty states (parallel: true)
- [ ] #98 - Improve candidates table visual hierarchy (parallel: true, conflicts_with: #104)
- [ ] #99 - Add subset scanning support (parallel: false, depends_on: #102)
- [ ] #101 - Add scan ETA estimation and display (parallel: true)

Total tasks: 7
Parallel tasks: 6
Sequential tasks: 1 (#99 depends on #102)
Estimated total effort: 3-4 focused sessions
