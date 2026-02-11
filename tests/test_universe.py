"""Tests for ticker universe management."""

from option_alpha.config import Settings
from option_alpha.data.universe import (
    _clear_cache,
    filter_universe,
    get_full_universe,
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
