"""Tests for ticker universe management."""

from option_alpha.config import Settings
from option_alpha.data.universe import (
    GICS_SECTORS,
    PRESET_FILTERS,
    _clear_cache,
    filter_universe,
    get_full_universe,
    get_scan_universe,
    load_universe_data,
)


class TestLoadUniverseData:
    def test_returns_list_of_dicts(self):
        _clear_cache()
        data = load_universe_data()
        assert isinstance(data, list)
        assert len(data) > 0
        assert isinstance(data[0], dict)

    def test_entries_have_expected_keys(self):
        data = load_universe_data()
        expected_keys = {"symbol", "name", "sector", "market_cap_tier", "asset_type"}
        for entry in data:
            assert set(entry.keys()) == expected_keys, f"Bad keys for {entry}"

    def test_all_symbols_are_strings(self):
        data = load_universe_data()
        for entry in data:
            assert isinstance(entry["symbol"], str)
            assert len(entry["symbol"]) > 0

    def test_asset_types_valid(self):
        data = load_universe_data()
        valid_types = {"stock", "etf"}
        for entry in data:
            assert entry["asset_type"] in valid_types, (
                f"Invalid asset_type for {entry['symbol']}: {entry['asset_type']}"
            )

    def test_market_cap_tiers_valid(self):
        data = load_universe_data()
        valid_tiers = {"large", "mid", "small", "micro", ""}
        for entry in data:
            assert entry["market_cap_tier"] in valid_tiers, (
                f"Invalid tier for {entry['symbol']}: {entry['market_cap_tier']}"
            )

    def test_clear_cache_works(self):
        # Load to populate cache
        load_universe_data()
        # Clear and reload
        _clear_cache()
        data = load_universe_data()
        assert len(data) > 0


class TestGetFullUniverse:
    def test_returns_sorted_list(self):
        universe = get_full_universe()
        assert universe == sorted(universe)

    def test_no_duplicates(self):
        universe = get_full_universe()
        assert len(universe) == len(set(universe))

    def test_includes_major_tickers(self):
        universe = get_full_universe()
        for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ"]:
            assert ticker in universe, f"{ticker} missing from universe"

    def test_reasonable_size(self):
        universe = get_full_universe()
        # Should have at least a few hundred tickers
        assert len(universe) >= 500
        # Should be bounded (we curate ~800-1000 across all lists)
        assert len(universe) <= 2000


class TestFilterUniverse:
    def test_filter_with_preloaded_data(self):
        tickers = ["AAPL", "MSFT", "PENNY", "LOWVOL"]
        price_data = {
            "AAPL": {"last_price": 175.0, "avg_volume": 50_000_000},
            "MSFT": {"last_price": 380.0, "avg_volume": 25_000_000},
            "PENNY": {"last_price": 2.0, "avg_volume": 1_000_000},  # below min price
            "LOWVOL": {"last_price": 50.0, "avg_volume": 100_000},  # below min volume
        }
        settings = Settings()

        result = filter_universe(tickers, settings=settings, price_data=price_data)

        assert "AAPL" in result
        assert "MSFT" in result
        assert "PENNY" not in result  # price < $5
        assert "LOWVOL" not in result  # volume < 500k

    def test_filter_with_custom_thresholds(self):
        tickers = ["A", "B", "C"]
        price_data = {
            "A": {"last_price": 15.0, "avg_volume": 1_000_000},
            "B": {"last_price": 8.0, "avg_volume": 200_000},
            "C": {"last_price": 25.0, "avg_volume": 2_000_000},
        }
        settings = Settings(min_price=10.0, min_avg_volume=500_000)

        result = filter_universe(tickers, settings=settings, price_data=price_data)

        assert "A" in result  # passes both
        assert "B" not in result  # fails volume
        assert "C" in result  # passes both

    def test_filter_missing_data(self):
        tickers = ["AAPL", "UNKNOWN"]
        price_data = {
            "AAPL": {"last_price": 175.0, "avg_volume": 50_000_000},
            # UNKNOWN has no data
        }
        settings = Settings()

        result = filter_universe(tickers, settings=settings, price_data=price_data)
        assert "AAPL" in result
        assert "UNKNOWN" not in result

    def test_filter_empty_list(self):
        result = filter_universe([], price_data={})
        assert result == []

    def test_filter_returns_sorted(self):
        tickers = ["MSFT", "AAPL", "GOOGL"]
        price_data = {
            sym: {"last_price": 100.0, "avg_volume": 1_000_000}
            for sym in tickers
        }
        result = filter_universe(tickers, price_data=price_data)
        assert result == sorted(result)


class TestGetScanUniverse:
    """Tests for preset-based and sector-based universe filtering."""

    def test_default_returns_full_universe(self):
        result = get_scan_universe()
        full = get_full_universe()
        assert result == full

    def test_sp500_returns_large_cap_stocks(self):
        result = get_scan_universe(presets=["sp500"])
        data = load_universe_data()
        expected = sorted(
            t["symbol"]
            for t in data
            if t["market_cap_tier"] == "large" and t["asset_type"] == "stock"
        )
        assert result == expected
        assert len(result) > 0

    def test_midcap_returns_mid_cap_stocks(self):
        result = get_scan_universe(presets=["midcap"])
        data = load_universe_data()
        expected = sorted(
            t["symbol"]
            for t in data
            if t["market_cap_tier"] == "mid" and t["asset_type"] == "stock"
        )
        assert result == expected
        assert len(result) > 0

    def test_smallcap_returns_small_and_micro_stocks(self):
        result = get_scan_universe(presets=["smallcap"])
        data = load_universe_data()
        expected = sorted(
            t["symbol"]
            for t in data
            if t["market_cap_tier"] in ("small", "micro")
            and t["asset_type"] == "stock"
        )
        assert result == expected
        assert len(result) > 0

    def test_etfs_returns_only_etfs(self):
        result = get_scan_universe(presets=["etfs"])
        data = load_universe_data()
        expected = sorted(t["symbol"] for t in data if t["asset_type"] == "etf")
        assert result == expected
        assert len(result) > 0

    def test_union_of_sp500_and_etfs(self):
        sp500 = get_scan_universe(presets=["sp500"])
        etfs = get_scan_universe(presets=["etfs"])
        union = get_scan_universe(presets=["sp500", "etfs"])
        assert union == sorted(set(sp500) | set(etfs))

    def test_full_preset_with_sector_filter(self):
        result = get_scan_universe(presets=["full"], sectors=["Technology"])
        data = load_universe_data()
        expected = sorted(
            t["symbol"] for t in data if t.get("sector") == "Technology"
        )
        assert result == expected
        assert len(result) > 0

    def test_sp500_with_sector_intersection(self):
        result = get_scan_universe(presets=["sp500"], sectors=["Technology"])
        data = load_universe_data()
        expected = sorted(
            t["symbol"]
            for t in data
            if t["market_cap_tier"] == "large"
            and t["asset_type"] == "stock"
            and t.get("sector") == "Technology"
        )
        assert result == expected
        assert len(result) > 0
        # Must be a subset of sp500
        sp500 = get_scan_universe(presets=["sp500"])
        assert all(s in sp500 for s in result)

    def test_empty_sectors_means_all(self):
        with_empty = get_scan_universe(presets=["sp500"], sectors=[])
        without = get_scan_universe(presets=["sp500"])
        assert with_empty == without

    def test_extra_tickers_always_included(self):
        result = get_scan_universe(presets=["etfs"], extra_tickers=["CUSTOM1", "CUSTOM2"])
        assert "CUSTOM1" in result
        assert "CUSTOM2" in result

    def test_extra_tickers_with_sector_filter(self):
        """Extra tickers bypass sector filtering."""
        result = get_scan_universe(
            presets=["sp500"],
            sectors=["Technology"],
            extra_tickers=["NOTREAL"],
        )
        assert "NOTREAL" in result

    def test_results_are_sorted(self):
        result = get_scan_universe(presets=["sp500", "etfs"])
        assert result == sorted(result)

    def test_results_are_deduplicated(self):
        result = get_scan_universe(presets=["sp500"])
        assert len(result) == len(set(result))

    def test_multiple_sectors(self):
        tech = get_scan_universe(presets=["full"], sectors=["Technology"])
        health = get_scan_universe(presets=["full"], sectors=["Healthcare"])
        both = get_scan_universe(presets=["full"], sectors=["Technology", "Healthcare"])
        assert both == sorted(set(tech) | set(health))

    def test_uses_settings_defaults(self):
        settings = Settings(universe_presets=["etfs"], universe_sectors=[])
        result = get_scan_universe(settings=settings)
        etfs_direct = get_scan_universe(presets=["etfs"])
        assert result == etfs_direct

    def test_explicit_args_override_settings(self):
        settings = Settings(universe_presets=["etfs"], universe_sectors=["Energy"])
        result = get_scan_universe(presets=["sp500"], sectors=[], settings=settings)
        sp500 = get_scan_universe(presets=["sp500"])
        assert result == sp500

    def test_unknown_preset_ignored(self):
        """Unknown preset names are silently skipped."""
        result = get_scan_universe(presets=["sp500", "nonexistent"])
        sp500 = get_scan_universe(presets=["sp500"])
        assert result == sp500

    def test_gics_sectors_constant(self):
        assert len(GICS_SECTORS) == 11
        assert "Technology" in GICS_SECTORS
        assert "Healthcare" in GICS_SECTORS
        assert "Financials" in GICS_SECTORS

    def test_preset_filters_constant(self):
        assert set(PRESET_FILTERS.keys()) == {
            "sp500", "midcap", "smallcap", "etfs", "full"
        }
