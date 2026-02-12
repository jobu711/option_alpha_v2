"""CRUD operations for scan results, ticker scores, and AI theses."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from option_alpha.models import (
    DebateResult,
    Direction,
    ScanRun,
    ScanStatus,
    ScoreBreakdown,
    TickerScore,
)


def save_scan_run(conn: sqlite3.Connection, scan_run: ScanRun) -> int:
    """Insert a scan run and return its database ID.

    Args:
        conn: Active database connection.
        scan_run: ScanRun model to persist.

    Returns:
        The auto-generated integer primary key (id) of the inserted row.
    """
    cursor = conn.execute(
        """
        INSERT INTO scan_runs
            (run_id, timestamp, ticker_count, duration_seconds, status,
             error_message, scores_computed, debates_completed, options_analyzed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scan_run.run_id,
            scan_run.timestamp.isoformat(),
            scan_run.ticker_count,
            scan_run.duration_seconds,
            scan_run.status.value,
            scan_run.error_message,
            scan_run.scores_computed,
            scan_run.debates_completed,
            scan_run.options_analyzed,
        ),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def save_ticker_scores(
    conn: sqlite3.Connection,
    scan_run_id: int,
    scores: list[TickerScore],
) -> None:
    """Batch insert ticker scores for a scan run.

    Args:
        conn: Active database connection.
        scan_run_id: The database ID of the parent scan_run.
        scores: List of TickerScore models to persist.
    """
    rows = [
        (
            scan_run_id,
            score.symbol,
            score.composite_score,
            score.direction.value,
            score.last_price,
            score.avg_volume,
            json.dumps([b.model_dump() for b in score.breakdown]),
            None,  # options_recommendation_json - populated later if needed
            score.timestamp.isoformat(),
        )
        for score in scores
    ]
    conn.executemany(
        """
        INSERT INTO ticker_scores
            (scan_run_id, ticker, composite_score, direction, last_price,
             avg_volume, score_breakdown_json, options_recommendation_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def save_ai_theses(
    conn: sqlite3.Connection,
    scan_run_id: int,
    theses: list[DebateResult],
) -> None:
    """Batch insert AI debate results for a scan run.

    Args:
        conn: Active database connection.
        scan_run_id: The database ID of the parent scan_run.
        theses: List of DebateResult models to persist.
    """
    rows = [
        (
            scan_run_id,
            thesis.symbol,
            thesis.bull.analysis,
            thesis.bear.analysis,
            thesis.risk.analysis,
            thesis.final_thesis.conviction,
            thesis.final_thesis.recommended_action,
            thesis.final_thesis.direction.value,
        )
        for thesis in theses
    ]
    conn.executemany(
        """
        INSERT INTO ai_theses
            (scan_run_id, ticker, bull_thesis, bear_thesis, risk_synthesis,
             conviction, recommendation, direction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def update_scan_run(
    conn: sqlite3.Connection,
    scan_db_id: int,
    *,
    status: ScanStatus | None = None,
    debates_completed: int | None = None,
    duration_seconds: float | None = None,
    error_message: str | None = None,
) -> None:
    """Update fields on an existing scan run row."""
    updates: list[str] = []
    params: list = []
    if status is not None:
        updates.append("status = ?")
        params.append(status.value)
    if debates_completed is not None:
        updates.append("debates_completed = ?")
        params.append(debates_completed)
    if duration_seconds is not None:
        updates.append("duration_seconds = ?")
        params.append(duration_seconds)
    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)
    if not updates:
        return
    params.append(scan_db_id)
    conn.execute(
        f"UPDATE scan_runs SET {', '.join(updates)} WHERE id = ?",
        params,
    )
    conn.commit()


def _row_to_scan_run(row: sqlite3.Row) -> ScanRun:
    """Convert a database row to a ScanRun model."""
    return ScanRun(
        run_id=row["run_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        ticker_count=row["ticker_count"],
        duration_seconds=row["duration_seconds"],
        status=ScanStatus(row["status"]),
        error_message=row["error_message"],
        scores_computed=row["scores_computed"],
        debates_completed=row["debates_completed"],
        options_analyzed=row["options_analyzed"],
    )


def _row_to_ticker_score(row: sqlite3.Row) -> TickerScore:
    """Convert a database row to a TickerScore model."""
    breakdown: list[ScoreBreakdown] = []
    if row["score_breakdown_json"]:
        raw = json.loads(row["score_breakdown_json"])
        breakdown = [ScoreBreakdown(**item) for item in raw]

    return TickerScore(
        symbol=row["ticker"],
        composite_score=row["composite_score"],
        direction=Direction(row["direction"]),
        last_price=row["last_price"],
        avg_volume=row["avg_volume"],
        breakdown=breakdown,
        timestamp=datetime.fromisoformat(row["timestamp"]),
    )


def get_latest_scan(conn: sqlite3.Connection) -> ScanRun | None:
    """Get the most recent scan run.

    Returns:
        The latest ScanRun, or None if no scans exist.
    """
    row = conn.execute(
        "SELECT * FROM scan_runs ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return _row_to_scan_run(row)


def get_scan_by_id(conn: sqlite3.Connection, scan_id: int) -> ScanRun | None:
    """Get a specific scan run by its database ID.

    Args:
        conn: Active database connection.
        scan_id: The integer primary key of the scan_run.

    Returns:
        The matching ScanRun, or None if not found.
    """
    row = conn.execute(
        "SELECT * FROM scan_runs WHERE id = ?", (scan_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_scan_run(row)


def get_scores_for_scan(
    conn: sqlite3.Connection, scan_run_id: int
) -> list[TickerScore]:
    """Get all ticker scores for a given scan run.

    Args:
        conn: Active database connection.
        scan_run_id: The database ID of the scan run.

    Returns:
        List of TickerScore models for the scan, ordered by composite_score DESC.
    """
    rows = conn.execute(
        """
        SELECT * FROM ticker_scores
        WHERE scan_run_id = ?
        ORDER BY composite_score DESC
        """,
        (scan_run_id,),
    ).fetchall()
    return [_row_to_ticker_score(row) for row in rows]


def get_ticker_history(
    conn: sqlite3.Connection,
    ticker: str,
    days: int = 30,
) -> list[TickerScore]:
    """Get score trend for a specific ticker over time.

    Args:
        conn: Active database connection.
        ticker: The ticker symbol to query.
        days: Number of days to look back (default 30).

    Returns:
        List of TickerScore models ordered by timestamp ascending (oldest first).
    """
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT * FROM ticker_scores
        WHERE ticker = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (ticker, cutoff),
    ).fetchall()
    return [_row_to_ticker_score(row) for row in rows]


def get_all_scans(
    conn: sqlite3.Connection,
    limit: int = 10,
) -> list[ScanRun]:
    """List recent scan runs, newest first.

    Args:
        conn: Active database connection.
        limit: Maximum number of scan runs to return (default 10).

    Returns:
        List of ScanRun models ordered by timestamp descending.
    """
    rows = conn.execute(
        "SELECT * FROM scan_runs ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_scan_run(row) for row in rows]
