"""Integration tests for the universe expansion feature.

Tests cross-module interactions between universe, presets,
watchlists, refresh, and the pipeline orchestrator.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from option_alpha.config import Settings
from option_alpha.data.universe import (
    GICS_SECTORS,
    PRESET_FILTERS,
    _clear_cache,
    get_full_universe,
    get_scan_universe,
    load_universe_data,
)
from option_alpha.data.watchlists import (
    create_watchlist,
    get_active_watchlist,
    set_active_watchlist,
    set_watchlist_path,
)
from option_alpha.models import (
    AgentResponse,
    DebateResult,
    Direction,
    OptionsRecommendation,
    ScanResult,
    TickerData,
    TickerScore,
    TradeThesis,
)
from option_alpha.pipeline.orchestrator import ScanOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ticker_data(symbol: str, n_days: int = 30) -> TickerData:
    """Create synthetic TickerData for testing."""
    base = 100.0
    dates = pd.bdate_range(end=datetime.now(UTC), periods=n_days).tolist()
    return TickerData(
        symbol=symbol,
        dates=dates,
        open=[base + i * 0.1 for i in range(n_days)],
        high=[base + i * 0.1 + 1.0 for i in range(n_days)],
        low=[base + i * 0.1 - 1.0 for i in range(n_days)],
        close=[base + i * 0.2 for i in range(n_days)],
        volume=[1_000_000 + i * 1000 for i in range(n_days)],
        last_price=base + (n_days - 1) * 0.2,
        avg_volume=1_000_000.0,
    )


def _make_ticker_score(symbol: str, score: float = 75.0) -> TickerScore:
    return TickerScore(
        symbol=symbol,
        composite_score=score,
        direction=Direction.BULLISH,
        last_price=150.0,
        avg_volume=2_000_000.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_watchlist(tmp_path):
    """Provide an isolated watchlist file for testing."""
    path = tmp_path / "watchlists.json"
    set_watchlist_path(path)
    yield path
    set_watchlist_path(None)


@pytest.fixture
def settings(tmp_path):
    """Settings with temp paths and small top_n values for fast tests."""
    return Settings(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        top_n_options=3,
        top_n_ai_debate=2,
        ai_backend="ollama",
    )


@pytest.fixture
def orchestrator(settings):
    return ScanOrchestrator(settings=settings)


# ===========================================================================
# 1. Watchlist + Scan Universe integration
# ===========================================================================


class TestWatchlistScanUniverseIntegration:
    """Watchlist tickers should be merged into scan universe via extra_tickers."""

    def test_active_watchlist_tickers_included_in_scan_universe(
        self, isolated_watchlist
    ):
        """get_scan_universe(extra_tickers=get_active_watchlist()) includes
        the active watchlist tickers alongside preset-filtered universe."""
        create_watchlist("custom", ["ZZZZZ", "YYYYY"])
        set_active_watchlist("custom")

        active = get_active_watchlist()
        assert active == ["YYYYY", "ZZZZZ"]

        result = get_scan_universe(presets=["etfs"], extra_tickers=active)
        # Custom tickers should be present even though they're not ETFs
        assert "YYYYY" in result
        assert "ZZZZZ" in result
        # ETFs should also be present
        data = load_universe_data()
        etf_symbols = {t["symbol"] for t in data if t["asset_type"] == "etf"}
        for sym in etf_symbols:
            assert sym in result

    def test_watchlist_tickers_already_in_universe_no_duplicates(
        self, isolated_watchlist
    ):
        """If a watchlist ticker is already in the universe, results are
        still deduplicated."""
        create_watchlist("overlap", ["AAPL", "SPY"])
        set_active_watchlist("overlap")

        active = get_active_watchlist()
        result = get_scan_universe(extra_tickers=active)
        # No duplicates
        assert len(result) == len(set(result))
        assert "AAPL" in result
        assert "SPY" in result

    def test_empty_active_watchlist_changes_nothing(self, isolated_watchlist):
        """With no active watchlist, get_active_watchlist returns [] and
        get_scan_universe behaves normally."""
        active = get_active_watchlist()
        assert active == []

        with_extra = get_scan_universe(presets=["sp500"], extra_tickers=active)
        without_extra = get_scan_universe(presets=["sp500"])
        assert with_extra == without_extra


# ===========================================================================
# 2. Preset + Sector combination tests
# ===========================================================================


class TestPresetSectorCombo:
    """Combined preset and sector filtering produces correct subsets."""

    def test_sp500_with_technology_is_proper_subset(self):
        """sp500 + Technology should be a subset of sp500 alone AND
        a subset of full + Technology."""
        sp500 = get_scan_universe(presets=["sp500"])
        tech = get_scan_universe(presets=["full"], sectors=["Technology"])
        sp500_tech = get_scan_universe(presets=["sp500"], sectors=["Technology"])

        # Must be proper subset of sp500
        assert set(sp500_tech) < set(sp500), "sp500+Tech should be smaller than sp500"
        # Must be proper subset of full+Tech
        assert set(sp500_tech) <= set(tech), "sp500+Tech should be within Tech"
        # Must not be empty
        assert len(sp500_tech) > 0

    def test_etfs_with_sector_returns_empty(self):
        """ETFs don't have sectors, so filtering etfs preset by a sector
        should return no results (ETFs have empty sector string)."""
        result = get_scan_universe(presets=["etfs"], sectors=["Technology"])
        # ETFs have sector="" which doesn't match "Technology"
        assert len(result) == 0

    def test_multiple_presets_with_sector_filter(self):
        """Union of presets followed by sector intersection."""
        sp500 = get_scan_universe(presets=["sp500"], sectors=["Financials"])
        midcap = get_scan_universe(presets=["midcap"], sectors=["Financials"])
        both = get_scan_universe(
            presets=["sp500", "midcap"], sectors=["Financials"]
        )
        assert both == sorted(set(sp500) | set(midcap))

    def test_all_gics_sectors_filter_correctly(self):
        """Each GICS sector should produce a non-empty result with full preset
        (assuming universe data has at least one ticker per sector)."""
        data = load_universe_data()
        sectors_present = {t["sector"] for t in data if t.get("sector")}
        for sector in GICS_SECTORS:
            if sector in sectors_present:
                result = get_scan_universe(presets=["full"], sectors=[sector])
                assert len(result) > 0, f"Sector {sector} returned empty"


# ===========================================================================
# 3. Full pipeline mock test with dynamic presets
# ===========================================================================


class TestPipelineWithDynamicPresets:
    """Orchestrator should respect Settings presets/sectors when calling
    get_scan_universe and merge active watchlist tickers."""

    @pytest.mark.asyncio
    async def test_pipeline_passes_settings_presets_to_universe(
        self, tmp_path
    ):
        """When Settings has custom presets, the orchestrator passes them
        to get_scan_universe correctly."""
        custom_settings = Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "data" / "test.db",
            top_n_options=2,
            top_n_ai_debate=1,
            ai_backend="ollama",
            universe_presets=["etfs"],
            universe_sectors=["Technology"],  # won't match ETFs, but that's fine
        )
        orch = ScanOrchestrator(settings=custom_settings)

        captured_kwargs = {}

        def mock_scan_universe(**kw):
            captured_kwargs.update(kw)
            return ["SPY", "QQQ"]

        with (
            patch(
                "option_alpha.pipeline.orchestrator.get_scan_universe",
                side_effect=mock_scan_universe,
            ),
            patch(
                "option_alpha.pipeline.orchestrator.get_active_watchlist",
                return_value=[],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.load_batch", return_value={}
            ),
            patch(
                "option_alpha.pipeline.orchestrator.fetch_batch",
                return_value={
                    "SPY": _make_ticker_data("SPY"),
                    "QQQ": _make_ticker_data("QQQ"),
                },
            ),
            patch(
                "option_alpha.pipeline.orchestrator.save_batch", return_value=2
            ),
            patch(
                "option_alpha.scoring.composite.score_universe",
                return_value=[
                    _make_ticker_score("SPY"),
                    _make_ticker_score("QQQ"),
                ],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.batch_earnings_info",
                return_value={},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                side_effect=lambda ts, ei, **kw: ts,
            ),
            patch(
                "option_alpha.pipeline.orchestrator.fetch_chains_for_tickers",
                return_value={},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.recommend_for_scored_tickers",
                return_value=[],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.get_client",
                return_value=MagicMock(),
            ),
            patch(
                "option_alpha.pipeline.orchestrator.DebateManager"
            ) as mock_dm_cls,
            patch(
                "option_alpha.pipeline.orchestrator.initialize_db"
            ) as mock_db,
            patch(
                "option_alpha.pipeline.orchestrator.save_scan_run",
                return_value=1,
            ),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orch.run_scan()

        # Verify get_scan_universe was called with correct preset/sector args
        assert captured_kwargs["presets"] == ["etfs"]
        assert captured_kwargs["sectors"] == ["Technology"]
        assert captured_kwargs["extra_tickers"] == []
        assert captured_kwargs["settings"] is custom_settings
        assert isinstance(result, ScanResult)

    @pytest.mark.asyncio
    async def test_pipeline_merges_watchlist_tickers(
        self, tmp_path, isolated_watchlist
    ):
        """Pipeline should call get_active_watchlist and pass results as
        extra_tickers to get_scan_universe."""
        create_watchlist("my-picks", ["TSLA", "NVDA"])
        set_active_watchlist("my-picks")

        custom_settings = Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "data" / "test.db",
            top_n_options=2,
            top_n_ai_debate=1,
            ai_backend="ollama",
            universe_presets=["sp500"],
        )
        orch = ScanOrchestrator(settings=custom_settings)

        captured_kwargs = {}

        def mock_scan_universe(**kw):
            captured_kwargs.update(kw)
            return ["AAPL", "NVDA", "TSLA"]

        with (
            patch(
                "option_alpha.pipeline.orchestrator.get_scan_universe",
                side_effect=mock_scan_universe,
            ),
            patch(
                "option_alpha.pipeline.orchestrator.load_batch", return_value={}
            ),
            patch(
                "option_alpha.pipeline.orchestrator.fetch_batch",
                return_value={
                    s: _make_ticker_data(s) for s in ["AAPL", "NVDA", "TSLA"]
                },
            ),
            patch(
                "option_alpha.pipeline.orchestrator.save_batch", return_value=3
            ),
            patch(
                "option_alpha.scoring.composite.score_universe",
                return_value=[_make_ticker_score("AAPL")],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.batch_earnings_info",
                return_value={},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                side_effect=lambda ts, ei, **kw: ts,
            ),
            patch(
                "option_alpha.pipeline.orchestrator.fetch_chains_for_tickers",
                return_value={},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.recommend_for_scored_tickers",
                return_value=[],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.get_client",
                return_value=MagicMock(),
            ),
            patch(
                "option_alpha.pipeline.orchestrator.DebateManager"
            ) as mock_dm_cls,
            patch(
                "option_alpha.pipeline.orchestrator.initialize_db"
            ) as mock_db,
            patch(
                "option_alpha.pipeline.orchestrator.save_scan_run",
                return_value=1,
            ),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orch.run_scan()

        # Active watchlist tickers should have been passed as extra_tickers
        assert set(captured_kwargs["extra_tickers"]) == {"NVDA", "TSLA"}
        assert result.total_tickers_scanned == 3


# ===========================================================================
# 4. Universe data consistency
# ===========================================================================


class TestUniverseDataConsistency:
    """Verify internal consistency of universe data structures."""

    def test_full_universe_matches_loaded_data(self):
        """Every symbol from get_full_universe should exist in
        load_universe_data results."""
        data = load_universe_data()
        data_symbols = {t["symbol"] for t in data}
        full = get_full_universe()
        for sym in full:
            assert sym in data_symbols, f"{sym} in full_universe but not in data"

    def test_loaded_data_covers_full_universe(self):
        """Every symbol in load_universe_data should appear in
        get_full_universe."""
        data = load_universe_data()
        full = set(get_full_universe())
        for entry in data:
            assert (
                entry["symbol"] in full
            ), f"{entry['symbol']} in data but not in full_universe"

    def test_preset_subsets_are_consistent(self):
        """Union of all non-overlapping presets should cover all stocks."""
        data = load_universe_data()
        all_stocks = {t["symbol"] for t in data if t["asset_type"] == "stock"}
        all_etfs = {t["symbol"] for t in data if t["asset_type"] == "etf"}

        sp500 = set(get_scan_universe(presets=["sp500"]))
        midcap = set(get_scan_universe(presets=["midcap"]))
        smallcap = set(get_scan_universe(presets=["smallcap"]))
        etfs = set(get_scan_universe(presets=["etfs"]))

        # ETFs preset should match all ETFs
        assert etfs == all_etfs

        # Stock presets should cover all stocks with a market_cap_tier
        stocks_with_tier = {
            t["symbol"]
            for t in data
            if t["asset_type"] == "stock" and t["market_cap_tier"]
        }
        assert stocks_with_tier == sp500 | midcap | smallcap

    def test_scan_universe_full_matches_full_universe(self):
        """get_scan_universe(presets=['full']) should match get_full_universe."""
        full = get_full_universe()
        scan_full = get_scan_universe(presets=["full"])
        assert full == scan_full

    def test_cache_clear_produces_same_data(self):
        """After clearing cache, reloaded data should be identical."""
        data_before = load_universe_data()
        _clear_cache()
        data_after = load_universe_data()
        assert data_before == data_after


# ===========================================================================
# 5. Config roundtrip with universe settings
# ===========================================================================


class TestConfigRoundtrip:
    """Settings with custom presets/sectors should survive save/load cycle."""

    def test_save_load_preserves_presets(self, tmp_path):
        """Custom universe_presets persist through JSON roundtrip."""
        config_path = tmp_path / "config.json"
        original = Settings(
            universe_presets=["sp500", "etfs"],
            universe_sectors=["Technology", "Healthcare"],
            universe_refresh_interval_days=14,
        )
        original.save(config_path)

        reloaded = Settings.load(config_path)
        assert reloaded.universe_presets == ["sp500", "etfs"]
        assert reloaded.universe_sectors == ["Technology", "Healthcare"]
        assert reloaded.universe_refresh_interval_days == 14

    def test_default_presets_are_full(self):
        """Default Settings should use ['full'] preset and no sector filter."""
        s = Settings()
        assert s.universe_presets == ["full"]
        assert s.universe_sectors == []
        assert s.universe_refresh_interval_days == 7

    def test_settings_used_by_get_scan_universe(self):
        """When passed to get_scan_universe, Settings presets take effect."""
        s = Settings(universe_presets=["etfs"], universe_sectors=[])
        result = get_scan_universe(settings=s)
        etfs_direct = get_scan_universe(presets=["etfs"])
        assert result == etfs_direct


# ===========================================================================
# 6. Refresh + universe interaction
# ===========================================================================


class TestRefreshUniverseInteraction:
    """After a refresh updates the JSON file, load_universe_data picks up
    changes after cache clear."""

    def test_cache_clear_picks_up_new_data(self, tmp_path, monkeypatch):
        """Simulates a refresh writing new data, then verifying
        load_universe_data returns the updated data after _clear_cache."""
        # Create a temporary universe file with initial data
        initial_data = [
            {
                "symbol": "AAPL",
                "name": "Apple",
                "sector": "Technology",
                "market_cap_tier": "large",
                "asset_type": "stock",
            }
        ]
        universe_file = tmp_path / "universe_data.json"
        universe_file.write_text(json.dumps(initial_data))

        # Monkeypatch the data path to point to our temp file
        monkeypatch.setattr(
            "option_alpha.data.universe._UNIVERSE_DATA_PATH", universe_file
        )
        _clear_cache()

        data = load_universe_data()
        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"

        # Simulate a refresh writing new data
        updated_data = [
            {
                "symbol": "AAPL",
                "name": "Apple",
                "sector": "Technology",
                "market_cap_tier": "large",
                "asset_type": "stock",
            },
            {
                "symbol": "NVDA",
                "name": "NVIDIA",
                "sector": "Technology",
                "market_cap_tier": "large",
                "asset_type": "stock",
            },
        ]
        universe_file.write_text(json.dumps(updated_data))

        # Without cache clear, still get old data
        cached = load_universe_data()
        assert len(cached) == 1  # still cached

        # After cache clear, get new data
        _clear_cache()
        refreshed = load_universe_data()
        assert len(refreshed) == 2
        symbols = {t["symbol"] for t in refreshed}
        assert symbols == {"AAPL", "NVDA"}

    def test_refresh_removed_ticker_disappears_from_full_universe(
        self, tmp_path, monkeypatch
    ):
        """If refresh removes a ticker, get_full_universe no longer returns it."""
        initial_data = [
            {
                "symbol": "AAPL",
                "name": "Apple",
                "sector": "Technology",
                "market_cap_tier": "large",
                "asset_type": "stock",
            },
            {
                "symbol": "OLD",
                "name": "Old Co",
                "sector": "Energy",
                "market_cap_tier": "mid",
                "asset_type": "stock",
            },
        ]
        universe_file = tmp_path / "universe_data.json"
        universe_file.write_text(json.dumps(initial_data))
        monkeypatch.setattr(
            "option_alpha.data.universe._UNIVERSE_DATA_PATH", universe_file
        )
        _clear_cache()

        assert "OLD" in get_full_universe()

        # Simulate refresh removing OLD
        updated_data = [initial_data[0]]  # Only AAPL
        universe_file.write_text(json.dumps(updated_data))
        _clear_cache()

        assert "OLD" not in get_full_universe()
        assert "AAPL" in get_full_universe()


# ===========================================================================
# 7. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases for the universe expansion feature."""

    def test_empty_presets_returns_full_universe(self):
        """get_scan_universe(presets=[]) should return full universe
        (empty is treated as 'full')."""
        result = get_scan_universe(presets=[])
        full = get_full_universe()
        assert result == full

    def test_only_nonexistent_preset_returns_empty(self):
        """get_scan_universe(presets=['nonexistent']) should return empty
        since no valid preset matched."""
        result = get_scan_universe(presets=["nonexistent"])
        # No valid preset filter matched, so filtered is empty
        assert result == []

    def test_nonexistent_preset_mixed_with_valid(self):
        """Unknown presets are silently skipped; valid ones work."""
        result = get_scan_universe(presets=["sp500", "bogus"])
        sp500_only = get_scan_universe(presets=["sp500"])
        assert result == sp500_only

    def test_large_extra_tickers_list(self):
        """A large extra_tickers list should all be included."""
        many_tickers = [f"FAKE{i:04d}" for i in range(500)]
        result = get_scan_universe(presets=["etfs"], extra_tickers=many_tickers)
        for t in many_tickers:
            assert t in result

    def test_extra_tickers_bypass_sector_filter(self):
        """Extra tickers should appear even when sector filter would exclude them."""
        result = get_scan_universe(
            presets=["sp500"],
            sectors=["Technology"],
            extra_tickers=["CUSTOMXYZ"],
        )
        assert "CUSTOMXYZ" in result

    def test_results_always_sorted(self):
        """No matter what combination, results should always be sorted."""
        combos = [
            {"presets": ["sp500"]},
            {"presets": ["etfs"], "extra_tickers": ["ZZZ", "AAA"]},
            {"presets": ["sp500", "midcap"], "sectors": ["Technology"]},
            {"presets": ["full"], "sectors": ["Healthcare", "Financials"]},
        ]
        for kwargs in combos:
            result = get_scan_universe(**kwargs)
            assert result == sorted(result), f"Not sorted with kwargs {kwargs}"

    def test_results_always_deduplicated(self):
        """No matter what combination, results should have no duplicates."""
        combos = [
            {"presets": ["sp500", "sp500"]},
            {"presets": ["sp500"], "extra_tickers": ["AAPL"]},  # AAPL likely in sp500
            {"presets": ["full", "etfs"]},
        ]
        for kwargs in combos:
            result = get_scan_universe(**kwargs)
            assert len(result) == len(set(result)), (
                f"Duplicates with kwargs {kwargs}"
            )

    def test_nonexistent_sector_returns_empty(self):
        """A sector that doesn't exist in data returns no results."""
        result = get_scan_universe(
            presets=["full"], sectors=["Nonexistent Sector"]
        )
        assert result == []

    def test_empty_sectors_list_means_all(self):
        """Explicit empty sectors list should not filter by sector."""
        full = get_scan_universe(presets=["sp500"], sectors=[])
        default = get_scan_universe(presets=["sp500"])
        assert full == default


# ===========================================================================
# 8. Pipeline universe_override vs dynamic universe
# ===========================================================================


class TestPipelineUniverseOverrideVsDynamic:
    """Verify the pipeline correctly chooses between override and dynamic universe."""

    @pytest.mark.asyncio
    async def test_override_skips_all_dynamic_logic(self, orchestrator):
        """With universe_override, neither get_scan_universe nor
        get_active_watchlist should be called."""
        with (
            patch(
                "option_alpha.pipeline.orchestrator.get_scan_universe"
            ) as mock_univ,
            patch(
                "option_alpha.pipeline.orchestrator.get_active_watchlist"
            ) as mock_wl,
            patch(
                "option_alpha.pipeline.orchestrator.load_batch", return_value={}
            ),
            patch(
                "option_alpha.pipeline.orchestrator.fetch_batch",
                return_value={"TSLA": _make_ticker_data("TSLA")},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.save_batch", return_value=1
            ),
            patch(
                "option_alpha.scoring.composite.score_universe",
                return_value=[_make_ticker_score("TSLA")],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.batch_earnings_info",
                return_value={},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                side_effect=lambda ts, ei, **kw: ts,
            ),
            patch(
                "option_alpha.pipeline.orchestrator.fetch_chains_for_tickers",
                return_value={},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.recommend_for_scored_tickers",
                return_value=[],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.get_client",
                return_value=MagicMock(),
            ),
            patch(
                "option_alpha.pipeline.orchestrator.DebateManager"
            ) as mock_dm_cls,
            patch(
                "option_alpha.pipeline.orchestrator.initialize_db"
            ) as mock_db,
            patch(
                "option_alpha.pipeline.orchestrator.save_scan_run",
                return_value=1,
            ),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan(universe_override=["TSLA"])

        mock_univ.assert_not_called()
        mock_wl.assert_not_called()
        assert result.total_tickers_scanned == 1

    @pytest.mark.asyncio
    async def test_dynamic_universe_calls_both(self, orchestrator):
        """Without override, both get_scan_universe and get_active_watchlist
        should be called."""
        with (
            patch(
                "option_alpha.pipeline.orchestrator.get_scan_universe",
                return_value=["AAPL"],
            ) as mock_univ,
            patch(
                "option_alpha.pipeline.orchestrator.get_active_watchlist",
                return_value=[],
            ) as mock_wl,
            patch(
                "option_alpha.pipeline.orchestrator.load_batch", return_value={}
            ),
            patch(
                "option_alpha.pipeline.orchestrator.fetch_batch",
                return_value={"AAPL": _make_ticker_data("AAPL")},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.save_batch", return_value=1
            ),
            patch(
                "option_alpha.scoring.composite.score_universe",
                return_value=[_make_ticker_score("AAPL")],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.batch_earnings_info",
                return_value={},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                side_effect=lambda ts, ei, **kw: ts,
            ),
            patch(
                "option_alpha.pipeline.orchestrator.fetch_chains_for_tickers",
                return_value={},
            ),
            patch(
                "option_alpha.pipeline.orchestrator.recommend_for_scored_tickers",
                return_value=[],
            ),
            patch(
                "option_alpha.pipeline.orchestrator.get_client",
                return_value=MagicMock(),
            ),
            patch(
                "option_alpha.pipeline.orchestrator.DebateManager"
            ) as mock_dm_cls,
            patch(
                "option_alpha.pipeline.orchestrator.initialize_db"
            ) as mock_db,
            patch(
                "option_alpha.pipeline.orchestrator.save_scan_run",
                return_value=1,
            ),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orchestrator.run_scan()

        mock_univ.assert_called_once()
        mock_wl.assert_called_once()
