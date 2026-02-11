"""Tests for Parquet/JSON caching layer."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from option_alpha.config import Settings
from option_alpha.data.cache import (
    clear_cache,
    evict_stale_cache,
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


class TestEvictStaleCache:
    """Tests for evict_stale_cache() function."""

    def _create_parquet(self, cache_dir, symbol, age_days=0):
        """Helper to create a fake parquet file with a specific age."""
        import os
        fname = f"{symbol}_2025-01-01.parquet"
        f = cache_dir / fname
        f.write_bytes(b"fake parquet data")
        if age_days > 0:
            old_time = time.time() - (age_days * 86400)
            os.utime(f, (old_time, old_time))
        return f

    def test_evicts_old_non_universe_files(self, settings):
        """Files for tickers NOT in universe and older than max_age_days are removed."""
        cache_dir = settings.data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # AAPL is in universe, OLDTK is not
        self._create_parquet(cache_dir, "AAPL", age_days=60)
        self._create_parquet(cache_dir, "OLDTK", age_days=60)

        removed = evict_stale_cache({"AAPL"}, max_age_days=30, settings=settings)
        assert removed == 1

        remaining = [f.name for f in cache_dir.glob("*.parquet")]
        assert any("AAPL" in name for name in remaining)
        assert not any("OLDTK" in name for name in remaining)

    def test_keeps_recent_non_universe_files(self, settings):
        """Files for tickers NOT in universe but newer than max_age_days are kept."""
        cache_dir = settings.data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Recent file for ticker not in universe - should be kept
        self._create_parquet(cache_dir, "RECENT", age_days=5)

        removed = evict_stale_cache(set(), max_age_days=30, settings=settings)
        assert removed == 0

        remaining = [f.name for f in cache_dir.glob("*.parquet")]
        assert len(remaining) == 1

    def test_empty_universe_removes_all_old_files(self, settings):
        """With empty universe, all old files are removed."""
        cache_dir = settings.data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._create_parquet(cache_dir, "AAPL", age_days=60)
        self._create_parquet(cache_dir, "MSFT", age_days=60)
        self._create_parquet(cache_dir, "GOOGL", age_days=60)

        removed = evict_stale_cache(set(), max_age_days=30, settings=settings)
        assert removed == 3

        remaining = list(cache_dir.glob("*.parquet"))
        assert len(remaining) == 0

    def test_ignores_non_parquet_files(self, settings):
        """Non-parquet files are not touched by eviction."""
        cache_dir = settings.data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        import os
        json_file = cache_dir / "test_2025-01-01.json"
        json_file.write_text("{}")
        old_time = time.time() - (60 * 86400)
        os.utime(json_file, (old_time, old_time))

        removed = evict_stale_cache(set(), max_age_days=30, settings=settings)
        assert removed == 0
        assert json_file.exists()

    def test_ignores_malformed_filenames(self, settings):
        """Files without the SYMBOL_DATE.parquet format are skipped."""
        cache_dir = settings.data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        import os
        malformed = cache_dir / "nodatesuffix.parquet"
        malformed.write_bytes(b"data")
        old_time = time.time() - (60 * 86400)
        os.utime(malformed, (old_time, old_time))

        removed = evict_stale_cache(set(), max_age_days=30, settings=settings)
        assert removed == 0
        assert malformed.exists()
