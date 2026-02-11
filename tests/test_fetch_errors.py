"""Tests for failure cache, error classification, and fetcher enhancements.

Covers:
- load_failure_cache: empty/missing, TTL eviction, valid entries
- record_failures: new entries, existing entry updates
- clear_failure_cache: file deleted, no error on missing file
- get_failure_cache_stats: counts and type breakdown
- Corruption handling: corrupt JSON returns {} or is overwritten
- classify_fetch_error: FetchErrorType classification
- fetch_batch: configurable params, failure_tracker population
- Orchestrator: filtering known-failed tickers from fetch
- Settings: new fetch config fields round-trip via save/load
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from option_alpha.config import Settings
from option_alpha.data.cache import (
    clear_failure_cache,
    get_failure_cache_stats,
    load_failure_cache,
    record_failures,
)
from option_alpha.data.fetcher import classify_fetch_error, fetch_batch
from option_alpha.models import FetchErrorType, TickerData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    """Settings with a temporary data directory for isolated cache tests."""
    return Settings(data_dir=tmp_path / "data")


def _failure_cache_path(settings: Settings):
    """Return the failure cache file path for the given settings."""
    return settings.data_dir / "cache" / "_failures.json"


def _write_failure_cache(settings: Settings, data: dict) -> None:
    """Helper: write raw JSON to the failure cache file."""
    path = _failure_cache_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _make_failure_entry(
    error_type: str = "network",
    hours_ago: float = 1.0,
    message: str = "Fetch failed",
) -> dict:
    """Create a failure cache entry with a timestamp *hours_ago* in the past."""
    ts = datetime.now(UTC) - timedelta(hours=hours_ago)
    return {
        "error_type": error_type,
        "timestamp": ts.isoformat(),
        "message": message,
    }


# ---------------------------------------------------------------------------
# load_failure_cache tests
# ---------------------------------------------------------------------------

class TestLoadFailureCache:
    def test_empty_when_file_missing(self, settings):
        """Returns {} when no failure cache file exists."""
        result = load_failure_cache(ttl_hours=24, settings=settings)
        assert result == {}

    def test_empty_when_file_is_empty(self, settings):
        """Returns {} when the file exists but is empty."""
        _write_failure_cache(settings, {})
        # Overwrite with truly empty content
        path = _failure_cache_path(settings)
        path.write_text("")
        result = load_failure_cache(ttl_hours=24, settings=settings)
        assert result == {}

    def test_valid_entries_returned(self, settings):
        """Fresh entries (within TTL) are returned."""
        data = {
            "AAPL": _make_failure_entry("delisted", hours_ago=2),
            "MSFT": _make_failure_entry("network", hours_ago=5),
        }
        _write_failure_cache(settings, data)

        result = load_failure_cache(ttl_hours=24, settings=settings)
        assert "AAPL" in result
        assert "MSFT" in result
        assert result["AAPL"]["error_type"] == "delisted"

    def test_expired_entries_evicted(self, settings):
        """Entries older than ttl_hours are excluded."""
        data = {
            "AAPL": _make_failure_entry("delisted", hours_ago=2),
            "DEAD": _make_failure_entry("delisted", hours_ago=48),
        }
        _write_failure_cache(settings, data)

        result = load_failure_cache(ttl_hours=24, settings=settings)
        assert "AAPL" in result
        assert "DEAD" not in result

    def test_corrupt_json_returns_empty(self, settings):
        """Corrupt JSON is handled gracefully, returns {}."""
        path = _failure_cache_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{{{not valid json!!!")

        result = load_failure_cache(ttl_hours=24, settings=settings)
        assert result == {}


# ---------------------------------------------------------------------------
# record_failures tests
# ---------------------------------------------------------------------------

class TestRecordFailures:
    def test_new_entries_written(self, settings):
        """New failure entries are written to a fresh cache file."""
        failures = {
            "AAPL": {
                "error_type": FetchErrorType.NETWORK,
                "timestamp": datetime.now(UTC).isoformat(),
                "message": "Connection timeout",
            }
        }
        record_failures(failures, settings=settings)

        # Verify the file was written
        loaded = load_failure_cache(ttl_hours=24, settings=settings)
        assert "AAPL" in loaded
        assert loaded["AAPL"]["error_type"] == "network"

    def test_existing_entries_updated(self, settings):
        """Existing entries are preserved; new entries are merged in."""
        initial = {"MSFT": _make_failure_entry("rate_limited", hours_ago=1)}
        _write_failure_cache(settings, initial)

        new_failures = {
            "TSLA": {
                "error_type": FetchErrorType.DELISTED,
                "timestamp": datetime.now(UTC).isoformat(),
                "message": "Ticker delisted",
            }
        }
        record_failures(new_failures, settings=settings)

        loaded = load_failure_cache(ttl_hours=24, settings=settings)
        assert "MSFT" in loaded
        assert "TSLA" in loaded

    def test_empty_failures_noop(self, settings):
        """Passing empty dict does nothing; no file created."""
        record_failures({}, settings=settings)
        assert not _failure_cache_path(settings).exists()

    def test_overwrite_on_corrupt_existing(self, settings):
        """If existing cache is corrupt, record_failures overwrites it."""
        path = _failure_cache_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NOT JSON")

        failures = {
            "AAPL": {
                "error_type": FetchErrorType.UNKNOWN,
                "timestamp": datetime.now(UTC).isoformat(),
                "message": "Unknown error",
            }
        }
        record_failures(failures, settings=settings)

        loaded = load_failure_cache(ttl_hours=24, settings=settings)
        assert "AAPL" in loaded


# ---------------------------------------------------------------------------
# clear_failure_cache tests
# ---------------------------------------------------------------------------

class TestClearFailureCache:
    def test_file_deleted(self, settings):
        """Clears the failure cache file if it exists."""
        data = {"AAPL": _make_failure_entry("network")}
        _write_failure_cache(settings, data)
        assert _failure_cache_path(settings).exists()

        clear_failure_cache(settings=settings)
        assert not _failure_cache_path(settings).exists()

    def test_no_error_if_missing(self, settings):
        """No error raised when clearing a non-existent cache."""
        clear_failure_cache(settings=settings)  # should not raise


# ---------------------------------------------------------------------------
# get_failure_cache_stats tests
# ---------------------------------------------------------------------------

class TestGetFailureCacheStats:
    def test_empty_stats(self, settings):
        """Returns zero counts when no failure cache exists."""
        stats = get_failure_cache_stats(ttl_hours=24, settings=settings)
        assert stats["count"] == 0
        assert stats["by_type"] == {}
        assert stats["oldest"] is None

    def test_correct_counts_and_breakdown(self, settings):
        """Returns correct total count and per-type breakdown."""
        data = {
            "AAPL": _make_failure_entry("network", hours_ago=2),
            "MSFT": _make_failure_entry("network", hours_ago=3),
            "TSLA": _make_failure_entry("delisted", hours_ago=1),
        }
        _write_failure_cache(settings, data)

        stats = get_failure_cache_stats(ttl_hours=24, settings=settings)
        assert stats["count"] == 3
        assert stats["by_type"]["network"] == 2
        assert stats["by_type"]["delisted"] == 1
        assert stats["oldest"] is not None


# ---------------------------------------------------------------------------
# classify_fetch_error tests
# ---------------------------------------------------------------------------

class TestClassifyFetchError:
    def test_delisted(self):
        assert classify_fetch_error(Exception("Ticker DELISTED")) == FetchErrorType.DELISTED

    def test_rate_limited_401(self):
        assert classify_fetch_error(Exception("HTTP 401 unauthorized")) == FetchErrorType.RATE_LIMITED

    def test_rate_limited_keyword(self):
        assert classify_fetch_error(Exception("rate limit exceeded")) == FetchErrorType.RATE_LIMITED

    def test_network_connection(self):
        assert classify_fetch_error(Exception("Connection refused")) == FetchErrorType.NETWORK

    def test_network_timeout(self):
        assert classify_fetch_error(Exception("Request timeout")) == FetchErrorType.NETWORK

    def test_insufficient_data(self):
        assert classify_fetch_error(Exception("Insufficient data")) == FetchErrorType.INSUFFICIENT_DATA

    def test_unknown_fallback(self):
        assert classify_fetch_error(Exception("something weird")) == FetchErrorType.UNKNOWN


# ---------------------------------------------------------------------------
# fetch_batch configurable params + failure_tracker tests
# ---------------------------------------------------------------------------

class TestFetchBatchConfigurable:
    @patch("option_alpha.data.fetcher._download_batch")
    def test_custom_batch_size_and_workers(self, mock_download):
        """fetch_batch respects custom batch_size and max_workers."""
        symbols = [f"SYM{i}" for i in range(6)]
        # Return a valid DataFrame for each batch call
        dates = pd.date_range("2025-01-01", periods=20, freq="B")
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [100.0] * 20,
                "High": [102.0] * 20,
                "Low": [99.0] * 20,
                "Close": [101.0] * 20,
                "Volume": [1000000] * 20,
            },
            index=dates,
        )

        results = fetch_batch(
            symbols, batch_size=3, max_workers=1, max_retries=0, retry_delays=[0],
        )
        # 6 symbols / batch_size 3 = 2 batches
        assert mock_download.call_count == 2

    @patch("option_alpha.data.fetcher._download_batch")
    def test_failure_tracker_populated(self, mock_download):
        """failure_tracker dict is populated when batch download fails."""
        mock_download.side_effect = Exception("Connection refused")
        tracker: dict[str, FetchErrorType] = {}

        results = fetch_batch(
            ["AAPL", "MSFT"],
            batch_size=50,
            max_retries=0,
            retry_delays=[0],
            failure_tracker=tracker,
        )
        assert results == {}
        assert "AAPL" in tracker
        assert "MSFT" in tracker
        assert tracker["AAPL"] == FetchErrorType.NETWORK


# ---------------------------------------------------------------------------
# Orchestrator: failure cache filtering test
# ---------------------------------------------------------------------------

class TestOrchestratorFailureCacheFilter:
    @pytest.mark.asyncio
    async def test_known_failed_tickers_skipped(self, settings):
        """Orchestrator filters out tickers present in the failure cache."""
        from option_alpha.pipeline.orchestrator import ScanOrchestrator

        # Pre-populate failure cache with DEAD ticker
        failures = {
            "DEAD": {
                "error_type": "delisted",
                "timestamp": datetime.now(UTC).isoformat(),
                "message": "Ticker delisted",
            }
        }
        _write_failure_cache(settings, failures)

        settings.top_n_options = 3
        settings.top_n_ai_debate = 1
        orch = ScanOrchestrator(settings=settings)

        def _make_td(sym):
            return TickerData(
                symbol=sym,
                dates=[datetime(2025, 1, i + 1) for i in range(20)],
                open=[100.0] * 20,
                high=[102.0] * 20,
                low=[99.0] * 20,
                close=[101.0] * 20,
                volume=[1000000] * 20,
                last_price=101.0,
                avg_volume=1000000.0,
            )

        captured_fetch_symbols: list[str] = []

        def mock_fetch(symbols, **kw):
            captured_fetch_symbols.extend(symbols)
            return {s: _make_td(s) for s in symbols}

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe",
                  return_value=["AAPL", "DEAD", "MSFT"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", side_effect=mock_fetch),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=2),
            patch("option_alpha.scoring.composite.score_universe", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=MagicMock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
        ):
            from unittest.mock import AsyncMock
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orch.run_scan()

        # DEAD should have been filtered out by the failure cache
        assert "DEAD" not in captured_fetch_symbols
        assert "AAPL" in captured_fetch_symbols
        assert "MSFT" in captured_fetch_symbols


# ---------------------------------------------------------------------------
# Settings round-trip test for new fetch config fields
# ---------------------------------------------------------------------------

class TestSettingsFetchFields:
    def test_new_fields_defaults(self):
        """New fetch settings have correct default values."""
        s = Settings()
        assert s.fetch_max_retries == 3
        assert s.fetch_retry_delays == [1.0, 2.0, 4.0]
        assert s.fetch_batch_size == 20
        assert s.fetch_max_workers == 2
        assert s.failure_cache_ttl_hours == 24

    def test_save_load_roundtrip(self, tmp_path):
        """Fetch settings survive a save/load round-trip."""
        config_path = tmp_path / "config.json"
        s = Settings(
            fetch_max_retries=5,
            fetch_retry_delays=[0.5, 1.0, 2.0, 4.0],
            fetch_batch_size=10,
            fetch_max_workers=4,
            failure_cache_ttl_hours=48,
        )
        s.save(path=config_path)

        loaded = Settings.load(path=config_path)
        assert loaded.fetch_max_retries == 5
        assert loaded.fetch_retry_delays == [0.5, 1.0, 2.0, 4.0]
        assert loaded.fetch_batch_size == 10
        assert loaded.fetch_max_workers == 4
        assert loaded.failure_cache_ttl_hours == 48
