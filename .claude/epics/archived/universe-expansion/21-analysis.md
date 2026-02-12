# Analysis: Issue #21 - Universe data store and baseline generation

## Scope

Single-stream task (no parallelization needed) - sequential file modifications.

## Stream A: Full Implementation

### Files to modify:
- `src/option_alpha/models.py` — Add `UniverseTicker` Pydantic model
- `src/option_alpha/data/universe.py` — Major refactor: remove hardcoded lists, add file-backed loading

### Files to create:
- `src/option_alpha/data/universe_data.json` — Baseline data (~3k tickers)
- `scripts/generate_universe.py` — One-time generation script

### Implementation order:
1. Add `UniverseTicker` model to `models.py`
2. Create `scripts/generate_universe.py` (SEC EDGAR + yfinance)
3. Generate `universe_data.json` baseline (or create a representative sample for development)
4. Refactor `universe.py`: remove hardcoded lists, add `load_universe_data()`, keep `get_full_universe()` backward-compatible
5. Update `tests/test_universe.py` to work with new file-based structure

### Key constraints:
- `get_full_universe()` MUST remain backward compatible (returns sorted list of symbols)
- `filter_universe()` and `get_filtered_universe()` must continue to work
- The old list constants `SP500_CORE`, `POPULAR_OPTIONS`, `OPTIONABLE_ETFS` are removed
- Tests that import those constants need updating
- The generation script should be runnable offline (for CI) with a `--dry-run` or `--sample` mode

### Risk:
- Generating real 3k-ticker baseline requires live SEC EDGAR + yfinance calls
- For development, a representative sample JSON can be committed; full generation is a manual step
