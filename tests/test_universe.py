"""Tests for ticker universe management."""

from option_alpha.config import Settings
from option_alpha.data.universe import (
    OPTIONABLE_ETFS,
    POPULAR_OPTIONS,
    SP500_CORE,
    filter_universe,
    get_full_universe,
)


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


class TestTickerLists:
    def test_sp500_not_empty(self):
        assert len(SP500_CORE) > 400

    def test_popular_options_not_empty(self):
        assert len(POPULAR_OPTIONS) > 50

    def test_etfs_not_empty(self):
        assert len(OPTIONABLE_ETFS) > 30

    def test_no_duplicates_within_lists(self):
        # Within each list, no duplicates
        assert len(SP500_CORE) == len(set(SP500_CORE))
        assert len(OPTIONABLE_ETFS) == len(set(OPTIONABLE_ETFS))
