"""Tests for the SQLite persistence layer."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from option_alpha.models import (
    AgentResponse,
    DebateResult,
    Direction,
    ScanRun,
    ScanStatus,
    ScoreBreakdown,
    TickerScore,
    TradeThesis,
)
from option_alpha.persistence.database import (
    get_connection,
    initialize_db,
    run_migrations,
)
from option_alpha.persistence.repository import (
    get_all_scans,
    get_latest_scan,
    get_scan_by_id,
    get_scores_for_scan,
    get_ticker_history,
    save_ai_theses,
    save_scan_run,
    save_ticker_scores,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """Provide a fresh in-memory database with migrations applied."""
    connection = initialize_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def raw_conn():
    """Provide a raw connection without migrations for testing setup."""
    connection = get_connection(":memory:")
    yield connection
    connection.close()


def _make_scan_run(
    run_id: str = "scan-001",
    status: ScanStatus = ScanStatus.COMPLETED,
    ticker_count: int = 50,
    duration_seconds: float = 12.5,
    timestamp: datetime | None = None,
) -> ScanRun:
    """Helper to create a ScanRun with sensible defaults."""
    return ScanRun(
        run_id=run_id,
        timestamp=timestamp or datetime.now(UTC),
        ticker_count=ticker_count,
        duration_seconds=duration_seconds,
        status=status,
        scores_computed=10,
        debates_completed=5,
        options_analyzed=3,
    )


def _make_ticker_score(
    symbol: str = "AAPL",
    composite_score: float = 78.5,
    direction: Direction = Direction.BULLISH,
    timestamp: datetime | None = None,
) -> TickerScore:
    """Helper to create a TickerScore with breakdown."""
    return TickerScore(
        symbol=symbol,
        composite_score=composite_score,
        direction=direction,
        last_price=185.50,
        avg_volume=50_000_000.0,
        breakdown=[
            ScoreBreakdown(
                name="bb_width",
                raw_value=0.045,
                normalized=72.0,
                weight=0.20,
                contribution=14.4,
            ),
            ScoreBreakdown(
                name="rsi",
                raw_value=55.0,
                normalized=60.0,
                weight=0.10,
                contribution=6.0,
            ),
        ],
        timestamp=timestamp or datetime.now(UTC),
    )


def _make_debate_result(symbol: str = "AAPL") -> DebateResult:
    """Helper to create a DebateResult."""
    return DebateResult(
        symbol=symbol,
        bull=AgentResponse(
            role="bull",
            analysis="Strong momentum with rising volume.",
            key_points=["RSI improving", "Volume surge"],
            conviction=8,
        ),
        bear=AgentResponse(
            role="bear",
            analysis="Overvalued relative to sector peers.",
            key_points=["P/E ratio elevated", "Macro headwinds"],
            conviction=5,
        ),
        risk=AgentResponse(
            role="risk",
            analysis="Moderate risk profile with manageable downside.",
            key_points=["Defined risk with spreads", "Earnings in 14 days"],
            conviction=7,
        ),
        final_thesis=TradeThesis(
            symbol=symbol,
            direction=Direction.BULLISH,
            conviction=7,
            entry_rationale="Technical breakout with volume confirmation.",
            risk_factors=["Earnings risk", "Market correlation"],
            recommended_action="Buy AAPL 190C 45DTE",
        ),
    )


# ===========================================================================
# Database / Migration Tests
# ===========================================================================

class TestDatabase:
    """Test database connection and setup."""

    def test_wal_mode_enabled(self, tmp_path):
        """WAL journal mode should be set on file-based connections."""
        db_file = tmp_path / "test_wal.db"
        file_conn = initialize_db(db_file)
        try:
            result = file_conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0] == "wal"
        finally:
            file_conn.close()

    def test_wal_mode_memory_fallback(self, conn: sqlite3.Connection):
        """In-memory databases report 'memory' journal mode (WAL not applicable)."""
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "memory"

    def test_foreign_keys_enabled(self, conn: sqlite3.Connection):
        """Foreign keys should be enforced."""
        result = conn.execute("PRAGMA foreign_keys").fetchone()
        assert result[0] == 1

    def test_row_factory_set(self, conn: sqlite3.Connection):
        """Row factory should be sqlite3.Row for dict-like access."""
        assert conn.row_factory is sqlite3.Row

    def test_tables_created(self, conn: sqlite3.Connection):
        """All expected tables should exist after migration."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "scan_runs" in tables
        assert "ticker_scores" in tables
        assert "ai_theses" in tables
        assert "schema_version" in tables

    def test_indexes_created(self, conn: sqlite3.Connection):
        """Key indexes should exist for performance."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_ticker_scores_ticker_run" in indexes
        assert "idx_ai_theses_ticker_run" in indexes
        assert "idx_scan_runs_timestamp" in indexes

    def test_schema_version_tracked(self, conn: sqlite3.Connection):
        """Applied migrations should be recorded in schema_version."""
        cursor = conn.execute("SELECT version, filename FROM schema_version")
        versions = cursor.fetchall()
        assert len(versions) >= 1
        assert versions[0]["version"] == 1
        assert versions[0]["filename"] == "001_initial.sql"


class TestMigrations:
    """Test the migration system itself."""

    def test_migrations_idempotent(self, conn: sqlite3.Connection):
        """Running migrations twice should not fail or duplicate records."""
        # Migrations already ran in the fixture; run again.
        applied = run_migrations(conn)
        assert applied == []  # nothing new applied

        cursor = conn.execute("SELECT COUNT(*) FROM schema_version")
        assert cursor.fetchone()[0] == 1  # still just one version

    def test_run_migrations_on_fresh_db(self, raw_conn: sqlite3.Connection):
        """Migrations should work on a completely fresh database."""
        applied = run_migrations(raw_conn)
        assert 1 in applied

    def test_initialize_db_returns_usable_connection(self):
        """initialize_db should return a connection ready for queries."""
        conn = initialize_db(":memory:")
        try:
            # Should be able to insert immediately
            conn.execute(
                "INSERT INTO scan_runs (run_id, timestamp, status) VALUES (?, ?, ?)",
                ("test-1", datetime.now(UTC).isoformat(), "pending"),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM scan_runs").fetchone()
            assert row["run_id"] == "test-1"
        finally:
            conn.close()


# ===========================================================================
# Repository: Scan Runs
# ===========================================================================

class TestSaveScanRun:
    """Test saving scan runs."""

    def test_save_and_retrieve(self, conn: sqlite3.Connection):
        """Should insert a scan run and return a valid ID."""
        scan = _make_scan_run()
        scan_id = save_scan_run(conn, scan)
        assert isinstance(scan_id, int)
        assert scan_id >= 1

        row = conn.execute(
            "SELECT * FROM scan_runs WHERE id = ?", (scan_id,)
        ).fetchone()
        assert row["run_id"] == "scan-001"
        assert row["ticker_count"] == 50
        assert row["status"] == "completed"
        assert row["duration_seconds"] == 12.5

    def test_save_pending_scan(self, conn: sqlite3.Connection):
        """Should correctly store pending status."""
        scan = _make_scan_run(run_id="pending-1", status=ScanStatus.PENDING)
        scan_id = save_scan_run(conn, scan)
        row = conn.execute(
            "SELECT status FROM scan_runs WHERE id = ?", (scan_id,)
        ).fetchone()
        assert row["status"] == "pending"

    def test_save_failed_scan_with_error(self, conn: sqlite3.Connection):
        """Should store error message for failed scans."""
        scan = ScanRun(
            run_id="fail-1",
            status=ScanStatus.FAILED,
            error_message="API rate limit exceeded",
        )
        scan_id = save_scan_run(conn, scan)
        row = conn.execute(
            "SELECT error_message FROM scan_runs WHERE id = ?", (scan_id,)
        ).fetchone()
        assert row["error_message"] == "API rate limit exceeded"

    def test_duplicate_run_id_rejected(self, conn: sqlite3.Connection):
        """run_id has UNIQUE constraint, duplicates should fail."""
        scan1 = _make_scan_run(run_id="dup-1")
        scan2 = _make_scan_run(run_id="dup-1")
        save_scan_run(conn, scan1)
        with pytest.raises(sqlite3.IntegrityError):
            save_scan_run(conn, scan2)

    def test_multiple_scan_runs(self, conn: sqlite3.Connection):
        """Should be able to store multiple scan runs."""
        ids = []
        for i in range(5):
            scan = _make_scan_run(run_id=f"scan-{i:03d}")
            ids.append(save_scan_run(conn, scan))
        assert len(set(ids)) == 5  # all unique IDs


# ===========================================================================
# Repository: Ticker Scores
# ===========================================================================

class TestSaveTickerScores:
    """Test batch inserting ticker scores."""

    def test_batch_insert(self, conn: sqlite3.Connection):
        """Should insert multiple scores in a single call."""
        scan_id = save_scan_run(conn, _make_scan_run())
        scores = [
            _make_ticker_score("AAPL", 85.0),
            _make_ticker_score("MSFT", 72.0),
            _make_ticker_score("TSLA", 60.0, Direction.BEARISH),
        ]
        save_ticker_scores(conn, scan_id, scores)

        cursor = conn.execute(
            "SELECT COUNT(*) FROM ticker_scores WHERE scan_run_id = ?",
            (scan_id,),
        )
        assert cursor.fetchone()[0] == 3

    def test_score_breakdown_serialized(self, conn: sqlite3.Connection):
        """Score breakdown should be stored as valid JSON."""
        scan_id = save_scan_run(conn, _make_scan_run())
        score = _make_ticker_score("AAPL")
        save_ticker_scores(conn, scan_id, [score])

        row = conn.execute(
            "SELECT score_breakdown_json FROM ticker_scores WHERE ticker = 'AAPL'"
        ).fetchone()
        data = json.loads(row["score_breakdown_json"])
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "bb_width"
        assert data[1]["name"] == "rsi"

    def test_direction_stored(self, conn: sqlite3.Connection):
        """Direction enum should be stored as string."""
        scan_id = save_scan_run(conn, _make_scan_run())
        save_ticker_scores(conn, scan_id, [
            _make_ticker_score("SPY", 50.0, Direction.NEUTRAL),
        ])

        row = conn.execute(
            "SELECT direction FROM ticker_scores WHERE ticker = 'SPY'"
        ).fetchone()
        assert row["direction"] == "neutral"

    def test_empty_scores_list(self, conn: sqlite3.Connection):
        """Inserting an empty list should not fail."""
        scan_id = save_scan_run(conn, _make_scan_run())
        save_ticker_scores(conn, scan_id, [])

        cursor = conn.execute(
            "SELECT COUNT(*) FROM ticker_scores WHERE scan_run_id = ?",
            (scan_id,),
        )
        assert cursor.fetchone()[0] == 0

    def test_foreign_key_enforced(self, conn: sqlite3.Connection):
        """Inserting with invalid scan_run_id should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            save_ticker_scores(conn, 9999, [_make_ticker_score()])


# ===========================================================================
# Repository: AI Theses
# ===========================================================================

class TestSaveAiTheses:
    """Test batch inserting AI debate results."""

    def test_batch_insert(self, conn: sqlite3.Connection):
        """Should insert multiple debate results."""
        scan_id = save_scan_run(conn, _make_scan_run())
        theses = [_make_debate_result("AAPL"), _make_debate_result("MSFT")]
        save_ai_theses(conn, scan_id, theses)

        cursor = conn.execute(
            "SELECT COUNT(*) FROM ai_theses WHERE scan_run_id = ?",
            (scan_id,),
        )
        assert cursor.fetchone()[0] == 2

    def test_thesis_fields_stored(self, conn: sqlite3.Connection):
        """All thesis fields should be correctly stored."""
        scan_id = save_scan_run(conn, _make_scan_run())
        save_ai_theses(conn, scan_id, [_make_debate_result("AAPL")])

        row = conn.execute(
            "SELECT * FROM ai_theses WHERE ticker = 'AAPL'"
        ).fetchone()
        assert "momentum" in row["bull_thesis"].lower()
        assert "overvalued" in row["bear_thesis"].lower()
        assert "moderate" in row["risk_synthesis"].lower()
        assert row["conviction"] == 7
        assert "190C" in row["recommendation"]
        assert row["direction"] == "bullish"

    def test_empty_theses_list(self, conn: sqlite3.Connection):
        """Inserting an empty list should not fail."""
        scan_id = save_scan_run(conn, _make_scan_run())
        save_ai_theses(conn, scan_id, [])

        cursor = conn.execute(
            "SELECT COUNT(*) FROM ai_theses WHERE scan_run_id = ?",
            (scan_id,),
        )
        assert cursor.fetchone()[0] == 0

    def test_foreign_key_enforced(self, conn: sqlite3.Connection):
        """Inserting with invalid scan_run_id should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            save_ai_theses(conn, 9999, [_make_debate_result()])


# ===========================================================================
# Repository: Query Operations
# ===========================================================================

class TestGetLatestScan:
    """Test retrieving the most recent scan."""

    def test_returns_latest(self, conn: sqlite3.Connection):
        """Should return the scan with the most recent timestamp."""
        t1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        t2 = datetime(2025, 1, 2, 12, 0, 0, tzinfo=UTC)
        save_scan_run(conn, _make_scan_run("old", timestamp=t1))
        save_scan_run(conn, _make_scan_run("new", timestamp=t2))

        latest = get_latest_scan(conn)
        assert latest is not None
        assert latest.run_id == "new"

    def test_returns_none_when_empty(self, conn: sqlite3.Connection):
        """Should return None when no scans exist."""
        assert get_latest_scan(conn) is None

    def test_model_fields_populated(self, conn: sqlite3.Connection):
        """Returned ScanRun should have all fields correctly populated."""
        scan = _make_scan_run(status=ScanStatus.COMPLETED)
        save_scan_run(conn, scan)

        result = get_latest_scan(conn)
        assert result is not None
        assert result.status == ScanStatus.COMPLETED
        assert result.ticker_count == 50
        assert result.duration_seconds == 12.5
        assert result.scores_computed == 10
        assert result.debates_completed == 5


class TestGetScanById:
    """Test retrieving a scan by database ID."""

    def test_found(self, conn: sqlite3.Connection):
        """Should return the matching scan."""
        scan_id = save_scan_run(conn, _make_scan_run("by-id-test"))
        result = get_scan_by_id(conn, scan_id)
        assert result is not None
        assert result.run_id == "by-id-test"

    def test_not_found(self, conn: sqlite3.Connection):
        """Should return None for non-existent ID."""
        assert get_scan_by_id(conn, 9999) is None


class TestGetScoresForScan:
    """Test retrieving scores for a specific scan."""

    def test_returns_scores_ordered(self, conn: sqlite3.Connection):
        """Scores should be ordered by composite_score descending."""
        scan_id = save_scan_run(conn, _make_scan_run())
        scores = [
            _make_ticker_score("LOW", 40.0),
            _make_ticker_score("HIGH", 90.0),
            _make_ticker_score("MID", 65.0),
        ]
        save_ticker_scores(conn, scan_id, scores)

        result = get_scores_for_scan(conn, scan_id)
        assert len(result) == 3
        assert result[0].symbol == "HIGH"
        assert result[1].symbol == "MID"
        assert result[2].symbol == "LOW"

    def test_breakdown_deserialized(self, conn: sqlite3.Connection):
        """Score breakdowns should be properly deserialized from JSON."""
        scan_id = save_scan_run(conn, _make_scan_run())
        save_ticker_scores(conn, scan_id, [_make_ticker_score()])

        result = get_scores_for_scan(conn, scan_id)
        assert len(result) == 1
        assert len(result[0].breakdown) == 2
        assert isinstance(result[0].breakdown[0], ScoreBreakdown)
        assert result[0].breakdown[0].name == "bb_width"
        assert result[0].breakdown[0].normalized == 72.0

    def test_empty_for_nonexistent_scan(self, conn: sqlite3.Connection):
        """Should return empty list for non-existent scan_run_id."""
        assert get_scores_for_scan(conn, 9999) == []


class TestGetTickerHistory:
    """Test historical ticker score queries."""

    def test_returns_history_within_window(self, conn: sqlite3.Connection):
        """Should return scores within the specified day window."""
        now = datetime.now(UTC)
        # Create 3 scans over different days
        for i in range(3):
            ts = now - timedelta(days=i)
            scan = _make_scan_run(f"hist-{i}", timestamp=ts)
            scan_id = save_scan_run(conn, scan)
            save_ticker_scores(
                conn, scan_id, [_make_ticker_score("AAPL", 70.0 + i, timestamp=ts)]
            )

        history = get_ticker_history(conn, "AAPL", days=30)
        assert len(history) == 3

    def test_excludes_old_data(self, conn: sqlite3.Connection):
        """Scores older than the window should be excluded."""
        now = datetime.now(UTC)
        old_ts = now - timedelta(days=60)
        recent_ts = now - timedelta(days=1)

        scan1_id = save_scan_run(conn, _make_scan_run("old", timestamp=old_ts))
        save_ticker_scores(
            conn, scan1_id, [_make_ticker_score("AAPL", 50.0, timestamp=old_ts)]
        )

        scan2_id = save_scan_run(conn, _make_scan_run("recent", timestamp=recent_ts))
        save_ticker_scores(
            conn, scan2_id, [_make_ticker_score("AAPL", 80.0, timestamp=recent_ts)]
        )

        history = get_ticker_history(conn, "AAPL", days=30)
        assert len(history) == 1
        assert history[0].composite_score == 80.0

    def test_filters_by_ticker(self, conn: sqlite3.Connection):
        """Should only return scores for the requested ticker."""
        scan_id = save_scan_run(conn, _make_scan_run())
        save_ticker_scores(conn, scan_id, [
            _make_ticker_score("AAPL", 80.0),
            _make_ticker_score("MSFT", 75.0),
        ])

        history = get_ticker_history(conn, "AAPL")
        assert len(history) == 1
        assert history[0].symbol == "AAPL"

    def test_ordered_ascending(self, conn: sqlite3.Connection):
        """History should be ordered oldest-first for trend plotting."""
        now = datetime.now(UTC)
        for i in range(3):
            ts = now - timedelta(days=2 - i)  # day -2, -1, 0
            scan_id = save_scan_run(conn, _make_scan_run(f"ord-{i}", timestamp=ts))
            save_ticker_scores(
                conn, scan_id, [_make_ticker_score("SPY", 50.0 + i, timestamp=ts)]
            )

        history = get_ticker_history(conn, "SPY", days=30)
        assert len(history) == 3
        # Should go from lowest to highest score (oldest to newest)
        assert history[0].composite_score == 50.0
        assert history[2].composite_score == 52.0

    def test_empty_for_unknown_ticker(self, conn: sqlite3.Connection):
        """Should return empty list for ticker with no data."""
        assert get_ticker_history(conn, "ZZZZZ") == []


class TestGetAllScans:
    """Test listing recent scans."""

    def test_returns_limited_results(self, conn: sqlite3.Connection):
        """Should respect the limit parameter."""
        for i in range(15):
            save_scan_run(conn, _make_scan_run(f"all-{i:03d}"))

        scans = get_all_scans(conn, limit=5)
        assert len(scans) == 5

    def test_default_limit(self, conn: sqlite3.Connection):
        """Default limit should be 10."""
        for i in range(15):
            save_scan_run(conn, _make_scan_run(f"def-{i:03d}"))

        scans = get_all_scans(conn)
        assert len(scans) == 10

    def test_ordered_newest_first(self, conn: sqlite3.Connection):
        """Scans should be ordered newest-first."""
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 15, tzinfo=UTC)
        t3 = datetime(2025, 12, 31, tzinfo=UTC)

        save_scan_run(conn, _make_scan_run("first", timestamp=t1))
        save_scan_run(conn, _make_scan_run("second", timestamp=t2))
        save_scan_run(conn, _make_scan_run("third", timestamp=t3))

        scans = get_all_scans(conn)
        assert scans[0].run_id == "third"
        assert scans[1].run_id == "second"
        assert scans[2].run_id == "first"

    def test_empty_database(self, conn: sqlite3.Connection):
        """Should return empty list when no scans exist."""
        assert get_all_scans(conn) == []


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_score_with_no_breakdown(self, conn: sqlite3.Connection):
        """Score with empty breakdown should store/retrieve cleanly."""
        scan_id = save_scan_run(conn, _make_scan_run())
        score = TickerScore(
            symbol="BARE",
            composite_score=50.0,
            breakdown=[],
        )
        save_ticker_scores(conn, scan_id, [score])

        result = get_scores_for_scan(conn, scan_id)
        assert len(result) == 1
        assert result[0].breakdown == []

    def test_score_roundtrip_preserves_values(self, conn: sqlite3.Connection):
        """All numeric values should survive the DB roundtrip."""
        scan_id = save_scan_run(conn, _make_scan_run())
        original = _make_ticker_score("ROUND", 99.99)
        save_ticker_scores(conn, scan_id, [original])

        restored = get_scores_for_scan(conn, scan_id)[0]
        assert restored.symbol == original.symbol
        assert restored.composite_score == original.composite_score
        assert restored.direction == original.direction
        assert restored.last_price == original.last_price
        assert restored.avg_volume == original.avg_volume

    def test_scan_run_roundtrip(self, conn: sqlite3.Connection):
        """ScanRun fields should survive the DB roundtrip."""
        original = _make_scan_run("roundtrip-test")
        scan_id = save_scan_run(conn, original)
        restored = get_scan_by_id(conn, scan_id)

        assert restored is not None
        assert restored.run_id == original.run_id
        assert restored.ticker_count == original.ticker_count
        assert restored.duration_seconds == original.duration_seconds
        assert restored.status == original.status
        assert restored.scores_computed == original.scores_computed

    def test_special_characters_in_error_message(self, conn: sqlite3.Connection):
        """Error messages with special chars should be stored correctly."""
        scan = ScanRun(
            run_id="special-chars",
            status=ScanStatus.FAILED,
            error_message="Error: Can't connect to \"server\" (code=500) & retry\nfailed",
        )
        scan_id = save_scan_run(conn, scan)
        restored = get_scan_by_id(conn, scan_id)
        assert restored is not None
        assert restored.error_message == scan.error_message

    def test_cascade_delete(self, conn: sqlite3.Connection):
        """Deleting a scan_run should cascade to scores and theses."""
        scan_id = save_scan_run(conn, _make_scan_run("cascade-test"))
        save_ticker_scores(conn, scan_id, [_make_ticker_score()])
        save_ai_theses(conn, scan_id, [_make_debate_result()])

        # Verify data exists
        assert conn.execute(
            "SELECT COUNT(*) FROM ticker_scores WHERE scan_run_id = ?", (scan_id,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM ai_theses WHERE scan_run_id = ?", (scan_id,)
        ).fetchone()[0] == 1

        # Delete the scan run
        conn.execute("DELETE FROM scan_runs WHERE id = ?", (scan_id,))
        conn.commit()

        # Children should be gone
        assert conn.execute(
            "SELECT COUNT(*) FROM ticker_scores WHERE scan_run_id = ?", (scan_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM ai_theses WHERE scan_run_id = ?", (scan_id,)
        ).fetchone()[0] == 0

    def test_large_batch_insert(self, conn: sqlite3.Connection):
        """Should handle inserting many scores at once."""
        scan_id = save_scan_run(conn, _make_scan_run())
        scores = [
            _make_ticker_score(f"T{i:04d}", float(i % 100))
            for i in range(200)
        ]
        save_ticker_scores(conn, scan_id, scores)

        count = conn.execute(
            "SELECT COUNT(*) FROM ticker_scores WHERE scan_run_id = ?", (scan_id,)
        ).fetchone()[0]
        assert count == 200
