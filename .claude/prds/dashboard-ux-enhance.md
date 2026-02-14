---
name: dashboard-ux-enhance
description: Improve dashboard workflow efficiency, universe management, scan feedback, and visual polish
status: backlog
created: 2026-02-14T13:03:56Z
---

# PRD: dashboard-ux-enhance

## Executive Summary

Enhance the Option Alpha dashboard UX across four key areas: (1) overhaul the universe management page with proper ticker counts, advanced filtering, and fixed tag display, (2) add quick-scan-subset capability so users can scan specific tag groups or selected tickers, (3) improve scan progress feedback with estimated time remaining, and (4) polish empty states, confirmation dialogs, and table visual hierarchy across the app.

These changes reduce friction in the core scan-and-analyze workflow, fix known bugs (tag display showing raw `●` characters), and make the data-dense dashboard easier to scan visually.

## Problem Statement

The dashboard is functional but has UX gaps that slow down the core workflow:

- **Universe page is opaque**: Users cannot tell at a glance how many tickers are in their active scan universe. Filtering is limited to basic search and tag clicks. The tag sidebar has a display bug showing raw `●` characters instead of styled status indicators.
- **Scanning is all-or-nothing**: Users must scan the entire active universe even when they only want to re-evaluate a specific sector or handful of tickers. This wastes time and API calls.
- **Scan progress is a black box**: The progress bar shows phase completion but no estimated time remaining, leaving users uncertain about when results will be available.
- **Tables are hard to scan visually**: The candidates table lacks sufficient visual hierarchy to quickly identify the most actionable setups. Empty states show generic messages without guidance.

These issues compound: a user who can't easily manage their universe, can't scan a subset, and can't tell when results are coming will lose confidence in the tool.

## User Stories

### US-1: Universe Ticker Count Visibility
**As a** trader managing my scan universe,
**I want to** see a clear count of active tickers vs total tickers at the top of the universe page,
**So that** I know exactly how many tickers will be included in my next scan.

**Acceptance Criteria:**
- Prominent counter displays "X active of Y total tickers" at the top of the universe page
- Counter updates in real-time when tickers are activated/deactivated (via HTMX)
- Counter is also visible on the dashboard page (e.g., near the "Run Scan" button)
- If 0 active tickers, show a warning state with guidance

### US-2: Advanced Universe Filtering
**As a** trader with a large ticker universe,
**I want to** filter the ticker table by sector, tag, active status, and last-scanned date,
**So that** I can quickly find and manage specific groups of tickers.

**Acceptance Criteria:**
- Filter bar above the ticker table with dropdowns for: Sector, Tag, Status (Active/Inactive/All)
- "Last Scanned" filter with options: Today, This Week, This Month, Never, All
- Filters combine (AND logic) and update the table via HTMX without page reload
- Active filter count shown as a badge (e.g., "Filters (2)")
- "Clear all filters" button resets to defaults
- Filter state preserved during pagination

### US-3: Tag Display Fix and Improvement
**As a** user viewing tickers and tags,
**I want** tags to display correctly with proper styled indicators,
**So that** I can visually distinguish active vs inactive tags.

**Acceptance Criteria:**
- Fix the `●` raw character display bug in the tag sidebar
- Replace raw characters with CSS-styled status dots (green for active, gray for inactive)
- Tag chips on ticker rows use colored pill badges consistent with the tag sidebar
- Tag count displays next to each tag name are accurate and update on changes

### US-4: Quick Scan Subset
**As a** trader who wants to re-evaluate specific tickers,
**I want to** scan only tickers matching a specific tag or my current selection,
**So that** I get faster, targeted results without scanning my entire universe.

**Acceptance Criteria:**
- On the universe page: "Scan This Tag" button appears when a tag filter is active
- On the universe page: "Scan Selected (N)" button appears when tickers are checked
- On the dashboard: dropdown on "Run Scan" button with options: "Scan All", "Scan by Tag..." (shows tag list)
- Subset scan uses the same pipeline but with a filtered ticker list
- Progress display shows "Scanning N of M tickers" to clarify scope
- Results merge into the existing candidates table (update existing, add new)

### US-5: Scan Estimated Time Remaining
**As a** user waiting for a scan to complete,
**I want to** see an estimated time remaining,
**So that** I can decide whether to wait or come back later.

**Acceptance Criteria:**
- After the first phase completes, display "~X min remaining" below the progress bar
- Estimate based on: (a) average per-ticker duration from current scan phases already completed, extrapolated to remaining tickers, or (b) historical scan duration from previous scans if available
- Estimate updates as each phase completes (refines over time)
- If no historical data exists, show "Estimating..." until enough data is collected
- Store scan duration metadata (start time, end time, ticker count) in the database for future estimates

### US-6: Improved Empty States
**As a** new user or a user with no scan results,
**I want to** see helpful, contextual empty states,
**So that** I know what to do next.

**Acceptance Criteria:**
- **No scan results**: Show message with CTA to run first scan, include brief explanation of what scanning does
- **No universe tickers**: Show message with CTA to visit universe page and add tickers
- **No debate results**: Show message explaining how to select tickers and trigger debates
- **Empty filtered table**: Show "No tickers match your filters" with "Clear filters" button
- All empty states use consistent styling (centered, muted text, icon or illustration optional)

### US-7: Table Visual Hierarchy Improvements
**As a** trader reviewing scan candidates,
**I want** the candidates table to have better visual hierarchy,
**So that** I can quickly identify the most actionable setups.

**Acceptance Criteria:**
- Top 5 candidates have a subtle highlight (e.g., gold left border or background tint)
- Composite score displayed as a colored bar/pill in addition to the number (gradient from red to green)
- Direction column uses icon + text (▲ Bullish, ▼ Bearish, ● Neutral) with color
- Sort indicator arrows are clearly visible and indicate current sort direction
- Debated tickers show a small badge/icon indicating debate results are available
- Hover state shows a brief preview tooltip of the ticker's key metrics

## Requirements

### Functional Requirements

**FR-1: Universe Management Overhaul**
- Add active/total ticker counter component to universe page header
- Add active ticker count next to "Run Scan" on dashboard
- Implement multi-filter bar (sector, tag, status, last-scanned) with HTMX
- Fix tag sidebar `●` display bug — use CSS-styled dots
- Improve tag chip rendering on ticker rows

**FR-2: Subset Scanning**
- Add `POST /scan` parameter: `symbols: list[str] | None` (None = scan all active)
- Add `POST /scan` parameter: `tag: str | None` (resolve to symbols server-side)
- Update orchestrator to accept optional ticker override list
- Add "Scan by Tag" dropdown to dashboard scan controls
- Add "Scan Selected" and "Scan This Tag" buttons to universe page
- Update progress messaging to show subset scope

**FR-3: Scan Time Estimation**
- Add `scan_history` table: id, started_at, completed_at, ticker_count, phases (JSON)
- Record scan metadata on completion
- Calculate ETA from: current scan progress + historical averages
- Send ETA updates via existing WebSocket progress channel
- Display ETA below progress bar in `_progress.html` partial

**FR-4: Empty States**
- Create `_empty_state.html` partial component (icon, message, CTA)
- Replace all generic "No results" messages with contextual empty states
- Add empty state variants: no-scan, no-universe, no-debates, no-filter-results

**FR-5: Table Polish**
- Add visual score indicator (colored bar) to composite score column
- Add direction icons with color to direction column
- Add "debated" badge to tickers with existing debate results
- Enhance top-N highlight (configurable N, default 5)
- Improve sort indicators

### Non-Functional Requirements

**NFR-1: Performance**
- Filter operations on universe page must respond in < 200ms (HTMX partial)
- ETA calculation must not add latency to the scan pipeline
- No additional API calls for subset scan setup (resolve tags server-side)

**NFR-2: Consistency**
- All new UI components follow existing dark theme and CSS variable system
- HTMX patterns match existing conventions (hx-get, hx-target, hx-swap)
- No new JS frameworks — use existing Alpine.js + vanilla JS patterns only

**NFR-3: Backwards Compatibility**
- `POST /scan` without new parameters behaves identically to current behavior
- Existing API consumers (if any) are unaffected by new parameters
- Database migrations are additive (no destructive schema changes)

## Success Criteria

| Metric | Target |
|--------|--------|
| Universe page shows accurate active ticker count | Always accurate, updates on every change |
| Tag display bug resolved | No raw `●` characters visible |
| Subset scan available from dashboard + universe page | Both entry points functional |
| ETA displayed during scan | Shown after first phase, within 20% accuracy after 3+ scans |
| All empty states have contextual messages + CTAs | Zero generic "No results" messages |
| Candidates table has visual score indicators | Score bar + direction icons visible |

## Constraints & Assumptions

**Constraints:**
- No new JS build tooling — everything stays as CDN HTMX + Alpine.js + vanilla JS
- Must work with existing SQLite WAL mode database
- Templates must remain server-rendered Jinja2 (no client-side rendering)
- AI debate system architecture must not be modified (per CLAUDE.md)

**Assumptions:**
- Users have a modern browser (CSS custom properties, flexbox, grid support)
- Typical universe size is 50–500 tickers (filter performance tuned for this range)
- Scan history is available after 1-2 completed scans for ETA estimation
- The `●` tag bug is a template encoding issue, not a data issue

## Out of Scope

- **Score history charts** — will be a separate PRD
- **Mobile-first responsive redesign** — current responsive CSS is adequate
- **Keyboard shortcuts / accessibility overhaul** — worthy but separate effort
- **Debate flow improvements** (one-click debate, inline debate) — separate PRD
- **WebSocket reconnection resilience** — separate infrastructure concern
- **Loading skeletons / shimmer states** — nice-to-have, not in this scope
- **Export/report page improvements** — separate feature
- **Inline ticker editing** — separate feature

## Dependencies

**Internal:**
- `persistence/database.py` — new migration for `scan_history` table
- `pipeline/orchestrator.py` — accept optional ticker subset parameter
- `pipeline/progress.py` — add ETA field to WebSocket progress messages
- `web/routes.py` — update scan endpoint, add ETA logic
- `web/universe_routes.py` — add filter endpoints, fix tag rendering
- `data/universe_service.py` — add filter queries (by sector, tag combo, last-scanned)

**External:**
- None — all changes are internal to the existing stack

## Implementation Notes

**Suggested phasing (for epic decomposition):**

1. **Universe page fixes** (tag bug, ticker count, filter bar) — highest impact, lowest risk
2. **Empty states + table polish** — visual improvements, no backend changes
3. **Subset scanning** — backend + frontend, moderate complexity
4. **Scan ETA** — new table + estimation logic, can ship independently
