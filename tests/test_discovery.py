"""Tests for the discovery engine (CBOE fetch, validation, stale detection, run history)."""

import sqlite3
from datetime import datetime, timedelta, UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from option_alpha.config import Settings
from option_alpha.data.discovery import (
    DiscoveryResult,
    _cache_failures,
    _detect_stale_tickers,
    _fetch_cboe_optionable,
    _get_cached_failures,
    _validate_via_yfinance,
    run_discovery,
    should_run_discovery,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "option_alpha"
    / "persistence"
    / "migrations"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _test_settings() -> Settings:
    return Settings(
        min_price=5.0,
        min_avg_volume=500_000,
        discovery_batch_size=100,
        stale_ticker_threshold_days=90,
        failure_cache_ttl_hours=24,
        universe_refresh_interval_days=7,
    )


def _mock_cboe_response(symbols: list[str]) -> MagicMock:
    """Build a mock httpx Response with CBOE-style CSV."""
    csv_lines = ["Company Name, Stock Symbol, DPM Name, Post/Station"]
    for s in symbols:
        csv_lines.append(f'"Test Company {s}","{s}","Market Maker","1/1"')
    mock_resp = MagicMock()
    mock_resp.text = "\n".join(csv_lines)
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _mock_yf_download(
    symbols_that_pass: list[str],
    symbols_that_fail: list[str],
    price: float = 50.0,
    volume: int = 1_000_000,
) -> pd.DataFrame:
    """Build a mock DataFrame matching yfinance multi-ticker format."""
    all_symbols = symbols_that_pass + symbols_that_fail
    dates = pd.date_range("2026-02-10", periods=5)

    if len(all_symbols) == 1:
        sym = all_symbols[0]
        data = {
            "Close": [price] * 5,
            "Volume": [volume if sym in symbols_that_pass else 0] * 5,
        }
        return pd.DataFrame(data, index=dates)

    close_data = {}
    volume_data = {}
    for s in all_symbols:
        if s in symbols_that_pass:
            close_data[s] = [price] * 5
            volume_data[s] = [volume] * 5
        else:
            close_data[s] = [1.0] * 5
            volume_data[s] = [100] * 5

    close_df = pd.DataFrame(close_data, index=dates)
    volume_df = pd.DataFrame(volume_data, index=dates)
    return pd.concat({"Close": close_df, "Volume": volume_df}, axis=1)


def _seed_safe_tickers(conn: sqlite3.Connection) -> None:
    """Insert two 'safe' tickers with an active tag so toggle_ticker won't fail.

    Stale-detection tests deactivate specific tickers; we need at least one
    active ticker in an active tag to remain so the SAVEPOINT guard passes.
    """
    conn.execute(
        "INSERT OR IGNORE INTO universe_tags (name, slug, is_preset, is_active) "
        "VALUES ('Safe', 'safe', 1, 1)"
    )
    for sym in ("SAFE1", "SAFE2"):
        conn.execute(
            "INSERT OR IGNORE INTO universe_tickers (symbol, source, is_active) "
            "VALUES (?, 'preset', 1)",
            (sym,),
        )
    tag_id = conn.execute(
        "SELECT id FROM universe_tags WHERE slug = 'safe'"
    ).fetchone()[0]
    for sym in ("SAFE1", "SAFE2"):
        conn.execute(
            "INSERT OR IGNORE INTO ticker_tags (symbol, tag_id) VALUES (?, ?)",
            (sym, tag_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with all migrations applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        conn.executescript(sql_file.read_text())
    conn.commit()

    yield conn
    conn.close()


# ===========================================================================
# _fetch_cboe_optionable
# ===========================================================================


class TestFetchCboe:
    """Tests for CBOE CSV parsing and symbol filtering."""

    @patch("option_alpha.data.discovery.httpx.get")
    def test_fetch_cboe_parses_symbols(self, mock_get: MagicMock) -> None:
        """Valid symbols should be extracted from CSV."""
        mock_get.return_value = _mock_cboe_response(["AAPL", "MSFT", "TSLA"])
        result = _fetch_cboe_optionable("https://fake.url")

        assert "AAPL" in result
        assert "MSFT" in result
        assert "TSLA" in result
        assert len(result) == 3

    @patch("option_alpha.data.discovery.httpx.get")
    def test_fetch_cboe_filters_non_equity(self, mock_get: MagicMock) -> None:
        """Index options ($SPX), warrants (SPAK+), dots (A.B), numeric (123),
        and symbols > 5 chars should all be filtered out."""
        mock_get.return_value = _mock_cboe_response(
            ["$SPX", "SPAK+", "A.B", "123", "TOOLONG", "OK"]
        )
        result = _fetch_cboe_optionable("https://fake.url")

        # TOOLONG = 7 chars > 5 limit, filtered. OK = 2 chars, purely alpha, passes.
        assert result == ["OK"]

    @patch("option_alpha.data.discovery.httpx.get")
    def test_fetch_cboe_deduplicates_and_sorts(self, mock_get: MagicMock) -> None:
        """Duplicate symbols should be deduplicated, result sorted."""
        mock_get.return_value = _mock_cboe_response(["ZZZ", "AAA", "ZZZ", "BBB"])
        result = _fetch_cboe_optionable("https://fake.url")
        assert result == ["AAA", "BBB", "ZZZ"]


# ===========================================================================
# _validate_via_yfinance
# ===========================================================================


class TestValidateViaYfinance:
    """Tests for yfinance-based price/volume validation."""

    @patch("option_alpha.data.discovery.yf.download")
    def test_validate_filters_by_price_volume(self, mock_dl: MagicMock) -> None:
        """Tickers meeting price/volume thresholds pass; others fail."""
        settings = _test_settings()
        mock_dl.return_value = _mock_yf_download(
            symbols_that_pass=["GOOD1", "GOOD2"],
            symbols_that_fail=["BAD1"],
        )

        passed, failed = _validate_via_yfinance(
            ["GOOD1", "GOOD2", "BAD1"], settings
        )

        assert set(passed) == {"GOOD1", "GOOD2"}
        assert set(failed) == {"BAD1"}

    @patch("option_alpha.data.discovery.yf.download")
    def test_validate_empty_candidates(self, mock_dl: MagicMock) -> None:
        """Empty candidate list should return empty results."""
        settings = _test_settings()
        passed, failed = _validate_via_yfinance([], settings)
        assert passed == []
        assert failed == []
        mock_dl.assert_not_called()


# ===========================================================================
# _get_cached_failures / _cache_failures
# ===========================================================================


class TestFailureCache:
    """Tests for the discovery failure cache."""

    def test_failure_cache_within_ttl(self, db: sqlite3.Connection) -> None:
        """A recently cached failure should be returned within TTL."""
        _cache_failures(db, ["FAIL1"], reason="test")
        cached = _get_cached_failures(db, ttl_hours=24)
        assert "FAIL1" in cached

    def test_failure_cache_expired(self, db: sqlite3.Connection) -> None:
        """A failure older than TTL should not be returned."""
        # Insert with an old timestamp
        old_time = (datetime.now(UTC) - timedelta(hours=48)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        db.execute(
            "INSERT INTO discovery_failures (symbol, reason, failed_at) "
            "VALUES (?, ?, ?)",
            ("OLDFAIL", "test", old_time),
        )
        db.commit()

        cached = _get_cached_failures(db, ttl_hours=24)
        assert "OLDFAIL" not in cached


# ===========================================================================
# _detect_stale_tickers
# ===========================================================================


class TestDetectStaleTickers:
    """Tests for stale ticker detection and deactivation."""

    def test_detect_stale_null_scanned(self, db: sqlite3.Connection) -> None:
        """Ticker with NULL last_scanned_at and old created_at is stale."""
        _seed_safe_tickers(db)

        old_date = (datetime.now(UTC) - timedelta(days=120)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        db.execute(
            "INSERT INTO universe_tickers (symbol, source, is_active, created_at) "
            "VALUES (?, 'preset', 1, ?)",
            ("STALE1", old_date),
        )
        # Must be in an active tag for toggle_ticker's SAVEPOINT check
        tag_id = db.execute(
            "SELECT id FROM universe_tags WHERE slug = 'safe'"
        ).fetchone()[0]
        db.execute(
            "INSERT INTO ticker_tags (symbol, tag_id) VALUES (?, ?)",
            ("STALE1", tag_id),
        )
        db.commit()

        count = _detect_stale_tickers(db, threshold_days=90)
        assert count == 1

        row = db.execute(
            "SELECT is_active FROM universe_tickers WHERE symbol = 'STALE1'"
        ).fetchone()
        assert row["is_active"] == 0

    def test_detect_stale_old_scanned(self, db: sqlite3.Connection) -> None:
        """Ticker with old last_scanned_at is stale."""
        _seed_safe_tickers(db)

        old_date = (datetime.now(UTC) - timedelta(days=120)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        db.execute(
            "INSERT INTO universe_tickers "
            "(symbol, source, is_active, created_at, last_scanned_at) "
            "VALUES (?, 'preset', 1, ?, ?)",
            ("STALE2", old_date, old_date),
        )
        tag_id = db.execute(
            "SELECT id FROM universe_tags WHERE slug = 'safe'"
        ).fetchone()[0]
        db.execute(
            "INSERT INTO ticker_tags (symbol, tag_id) VALUES (?, ?)",
            ("STALE2", tag_id),
        )
        db.commit()

        count = _detect_stale_tickers(db, threshold_days=90)
        assert count == 1

        row = db.execute(
            "SELECT is_active FROM universe_tickers WHERE symbol = 'STALE2'"
        ).fetchone()
        assert row["is_active"] == 0

    def test_detect_stale_grace_period(self, db: sqlite3.Connection) -> None:
        """Ticker with NULL last_scanned_at but recent created_at is NOT stale."""
        _seed_safe_tickers(db)

        recent_date = (datetime.now(UTC) - timedelta(days=5)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        db.execute(
            "INSERT INTO universe_tickers (symbol, source, is_active, created_at) "
            "VALUES (?, 'preset', 1, ?)",
            ("NEWISH", recent_date),
        )
        tag_id = db.execute(
            "SELECT id FROM universe_tags WHERE slug = 'safe'"
        ).fetchone()[0]
        db.execute(
            "INSERT INTO ticker_tags (symbol, tag_id) VALUES (?, ?)",
            ("NEWISH", tag_id),
        )
        db.commit()

        count = _detect_stale_tickers(db, threshold_days=90)
        assert count == 0

        row = db.execute(
            "SELECT is_active FROM universe_tickers WHERE symbol = 'NEWISH'"
        ).fetchone()
        assert row["is_active"] == 1


# ===========================================================================
# should_run_discovery
# ===========================================================================


class TestShouldRunDiscovery:
    """Tests for the should_run_discovery scheduling check."""

    def test_should_run_never_run(self, db: sqlite3.Connection) -> None:
        """Empty discovery_runs table means discovery should run."""
        settings = _test_settings()
        assert should_run_discovery(db, settings) is True

    def test_should_run_overdue(self, db: sqlite3.Connection) -> None:
        """A completed run older than the refresh interval -> should run."""
        settings = _test_settings()
        old_time = (datetime.now(UTC) - timedelta(days=14)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        db.execute(
            "INSERT INTO discovery_runs (status, completed_at) "
            "VALUES ('completed', ?)",
            (old_time,),
        )
        db.commit()

        assert should_run_discovery(db, settings) is True

    def test_should_run_recent(self, db: sqlite3.Connection) -> None:
        """A completed run within the refresh interval -> should NOT run."""
        settings = _test_settings()
        recent_time = (datetime.now(UTC) - timedelta(days=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        db.execute(
            "INSERT INTO discovery_runs (status, completed_at) "
            "VALUES ('completed', ?)",
            (recent_time,),
        )
        db.commit()

        assert should_run_discovery(db, settings) is False


# ===========================================================================
# run_discovery (end-to-end, async)
# ===========================================================================


class TestRunDiscovery:
    """End-to-end tests for the full discovery pipeline."""

    @pytest.mark.asyncio
    @patch("option_alpha.data.discovery.yf.download")
    @patch("option_alpha.data.discovery.httpx.get")
    async def test_run_discovery_end_to_end(
        self,
        mock_httpx: MagicMock,
        mock_yf: MagicMock,
        db: sqlite3.Connection,
    ) -> None:
        """Full pipeline: fetch, validate, persist, detect stale."""
        settings = _test_settings()

        mock_httpx.return_value = _mock_cboe_response(["NEWA", "NEWB", "FAILX"])
        mock_yf.return_value = _mock_yf_download(
            symbols_that_pass=["NEWA", "NEWB"],
            symbols_that_fail=["FAILX"],
        )

        result = await run_discovery(db, settings=settings)

        assert isinstance(result, DiscoveryResult)
        assert result.cboe_fetched == 3
        assert result.new_added == 2
        assert result.failures_cached == 1

        # Verify tickers in DB
        row = db.execute(
            "SELECT COUNT(*) FROM universe_tickers WHERE symbol IN ('NEWA', 'NEWB')"
        ).fetchone()
        assert row[0] == 2

        # Verify failure cached
        cached = _get_cached_failures(db, ttl_hours=24)
        assert "FAILX" in cached

        # Verify discovery run recorded
        run_row = db.execute(
            "SELECT status FROM discovery_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run_row["status"] == "completed"

    @pytest.mark.asyncio
    @patch("option_alpha.data.discovery.yf.download")
    @patch("option_alpha.data.discovery.httpx.get")
    async def test_run_discovery_deduplicates(
        self,
        mock_httpx: MagicMock,
        mock_yf: MagicMock,
        db: sqlite3.Connection,
    ) -> None:
        """Existing tickers should not be re-added or re-validated."""
        settings = _test_settings()

        # Pre-populate with DUPX (all alpha, <=5 chars)
        db.execute(
            "INSERT INTO universe_tickers (symbol, source) VALUES ('DUPX', 'preset')"
        )
        db.commit()

        # CBOE returns both DUPX (already in DB) and BRAND (new)
        mock_httpx.return_value = _mock_cboe_response(["DUPX", "BRAND"])
        mock_yf.return_value = _mock_yf_download(
            symbols_that_pass=["BRAND"],
            symbols_that_fail=[],
        )

        result = await run_discovery(db, settings=settings)

        # Only BRAND is a candidate (DUPX deduplicated against existing universe)
        assert result.candidates_checked == 1
        assert result.new_added == 1

        # DUPX should still be there exactly once
        count = db.execute(
            "SELECT COUNT(*) FROM universe_tickers WHERE symbol = 'DUPX'"
        ).fetchone()[0]
        assert count == 1

    @pytest.mark.asyncio
    @patch("option_alpha.data.discovery.yf.download")
    @patch("option_alpha.data.discovery.httpx.get")
    async def test_run_discovery_tags_auto_discovered(
        self,
        mock_httpx: MagicMock,
        mock_yf: MagicMock,
        db: sqlite3.Connection,
    ) -> None:
        """Newly discovered tickers should have the 'auto-discovered' tag."""
        settings = _test_settings()

        mock_httpx.return_value = _mock_cboe_response(["DISCA", "DISCB"])
        mock_yf.return_value = _mock_yf_download(
            symbols_that_pass=["DISCA", "DISCB"],
            symbols_that_fail=[],
        )

        await run_discovery(db, settings=settings)

        # Verify auto-discovered tag association
        rows = db.execute(
            """
            SELECT tt.symbol
            FROM ticker_tags tt
            JOIN universe_tags tg ON tt.tag_id = tg.id
            WHERE tg.slug = 'auto-discovered'
              AND tt.symbol IN ('DISCA', 'DISCB')
            """
        ).fetchall()
        tagged_symbols = {row[0] for row in rows}
        assert tagged_symbols == {"DISCA", "DISCB"}


# ===========================================================================
# last_scanned_at update (orchestrator integration pattern)
# ===========================================================================


class TestLastScannedAtUpdate:
    """Test that the orchestrator's persist phase updates last_scanned_at."""

    def test_last_scanned_at_updated(self, db: sqlite3.Connection) -> None:
        """Simulates the orchestrator updating last_scanned_at after scoring."""
        # Insert tickers
        db.execute(
            "INSERT INTO universe_tickers (symbol, source, is_active) "
            "VALUES ('TST1', 'preset', 1)"
        )
        db.execute(
            "INSERT INTO universe_tickers (symbol, source, is_active) "
            "VALUES ('TST2', 'preset', 1)"
        )
        db.commit()

        # Verify NULL initially
        row = db.execute(
            "SELECT last_scanned_at FROM universe_tickers WHERE symbol = 'TST1'"
        ).fetchone()
        assert row["last_scanned_at"] is None

        # Simulate orchestrator persist phase
        scored_symbols = ["TST1", "TST2"]
        placeholders = ",".join("?" for _ in scored_symbols)
        db.execute(
            f"UPDATE universe_tickers SET last_scanned_at = datetime('now') "
            f"WHERE symbol IN ({placeholders})",
            scored_symbols,
        )
        db.commit()

        # Verify updated
        for sym in scored_symbols:
            row = db.execute(
                "SELECT last_scanned_at FROM universe_tickers WHERE symbol = ?",
                (sym,),
            ).fetchone()
            assert row["last_scanned_at"] is not None
