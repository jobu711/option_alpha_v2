"""Tests for Parquet/JSON caching layer."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from option_alpha.config import Settings
from option_alpha.data.cache import (
    clear_cache,
    is_fresh,
    load_batch,
    load_json,
    load_ticker_parquet,
    save_batch,
    save_json,
    save_ticker_parquet,
)
from option_alpha.models import TickerData


@pytest.fixture
def settings(tmp_path):
    """Create settings with a temporary data directory."""
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def sample_ticker_data():
    """Create sample TickerData for testing."""
    return TickerData(
        symbol="AAPL",
        dates=[datetime(2025, 1, i + 1) for i in range(5)],
        open=[150.0, 151.0, 152.0, 153.0, 154.0],
        high=[155.0, 156.0, 157.0, 158.0, 159.0],
        low=[148.0, 149.0, 150.0, 151.0, 152.0],
        close=[153.0, 154.0, 155.0, 156.0, 157.0],
        volume=[1000000, 1100000, 1200000, 1300000, 1400000],
        last_price=157.0,
        avg_volume=1200000.0,
    )


class TestIsFresh:
    def test_nonexistent_file(self, tmp_path):
        assert is_fresh(tmp_path / "nonexistent.parquet") is False

    def test_fresh_file(self, tmp_path):
        f = tmp_path / "fresh.parquet"
        f.write_text("data")
        assert is_fresh(f, max_age_hours=1) is True

    def test_stale_file(self, tmp_path):
        f = tmp_path / "stale.parquet"
        f.write_text("data")
        # Backdate the file
        import os
        old_time = time.time() - 3700  # 1 hour + 100 seconds ago
        os.utime(f, (old_time, old_time))
        assert is_fresh(f, max_age_hours=1) is False


class TestParquetCache:
    def test_save_and_load(self, settings, sample_ticker_data):
        path = save_ticker_parquet(sample_ticker_data, settings=settings)
        assert path.exists()
        assert path.suffix == ".parquet"

        loaded = load_ticker_parquet(
            "AAPL", max_age_hours=1, settings=settings
        )
        assert loaded is not None
        assert loaded.symbol == "AAPL"
        assert len(loaded.close) == 5
        assert loaded.close == sample_ticker_data.close
        assert loaded.volume == sample_ticker_data.volume
        assert loaded.last_price == sample_ticker_data.last_price

    def test_load_nonexistent(self, settings):
        result = load_ticker_parquet("NONEXIST", settings=settings)
        assert result is None

    def test_load_stale(self, settings, sample_ticker_data):
        save_ticker_parquet(sample_ticker_data, settings=settings)
        # Load with 0-hour max age = always stale
        result = load_ticker_parquet(
            "AAPL", max_age_hours=0, settings=settings
        )
        assert result is None


class TestJsonCache:
    def test_save_and_load(self, settings):
        data = {"scores": [1, 2, 3], "meta": "test"}
        save_json("test_key", data, settings=settings)

        loaded = load_json("test_key", max_age_hours=1, settings=settings)
        assert loaded is not None
        assert loaded["scores"] == [1, 2, 3]
        assert loaded["meta"] == "test"

    def test_load_nonexistent(self, settings):
        result = load_json("nonexistent_key", settings=settings)
        assert result is None

    def test_save_list(self, settings):
        data = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
        save_json("list_key", data, settings=settings)

        loaded = load_json("list_key", max_age_hours=1, settings=settings)
        assert loaded is not None
        assert len(loaded) == 2


class TestBatchOperations:
    def test_save_and_load_batch(self, settings):
        tickers = {}
        for sym in ["AAPL", "MSFT", "GOOGL"]:
            tickers[sym] = TickerData(
                symbol=sym,
                dates=[datetime(2025, 1, 1)],
                open=[100.0],
                high=[105.0],
                low=[98.0],
                close=[103.0],
                volume=[1000000],
                last_price=103.0,
                avg_volume=1000000.0,
            )

        saved = save_batch(tickers, settings=settings)
        assert saved == 3

        loaded = load_batch(
            ["AAPL", "MSFT", "GOOGL"], max_age_hours=1, settings=settings
        )
        assert len(loaded) == 3
        assert all(sym in loaded for sym in ["AAPL", "MSFT", "GOOGL"])

    def test_load_batch_partial(self, settings):
        td = TickerData(
            symbol="AAPL",
            dates=[datetime(2025, 1, 1)],
            open=[100.0],
            high=[105.0],
            low=[98.0],
            close=[103.0],
            volume=[1000000],
            last_price=103.0,
        )
        save_ticker_parquet(td, settings=settings)

        loaded = load_batch(
            ["AAPL", "MISSING"], max_age_hours=1, settings=settings
        )
        assert len(loaded) == 1
        assert "AAPL" in loaded


class TestClearCache:
    def test_clear(self, settings):
        td = TickerData(
            symbol="AAPL",
            dates=[datetime(2025, 1, 1)],
            open=[100.0],
            high=[105.0],
            low=[98.0],
            close=[103.0],
            volume=[1000000],
            last_price=103.0,
        )
        save_ticker_parquet(td, settings=settings)
        save_json("test", {"a": 1}, settings=settings)

        removed = clear_cache(settings=settings)
        assert removed == 2

        # Verify nothing loads
        assert load_ticker_parquet("AAPL", settings=settings) is None
        assert load_json("test", settings=settings) is None
