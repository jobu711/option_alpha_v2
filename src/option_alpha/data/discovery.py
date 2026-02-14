"""Discovery engine — fetches CBOE optionable securities, validates via yfinance,
manages failure cache, detects stale tickers, and records run history."""

import asyncio
import csv
import io
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Callable, Optional

import httpx
import yfinance as yf
from pydantic import BaseModel

from option_alpha.config import Settings, get_settings
from option_alpha.data.universe_service import (
    add_tickers,
    get_full_universe,
    toggle_ticker,
)

logger = logging.getLogger(__name__)


class DiscoveryResult(BaseModel):
    cboe_fetched: int = 0
    candidates_checked: int = 0
    new_added: int = 0
    stale_deactivated: int = 0
    failures_cached: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_discovery(
    conn: sqlite3.Connection,
    settings: Optional[Settings] = None,
    on_progress: Optional[Callable] = None,
) -> DiscoveryResult:
    """Run the full discovery pipeline.

    Steps:
        1. Record start in discovery_runs.
        2. Fetch CBOE optionable symbols.
        3. Dedup against existing universe + cached failures.
        4. Validate candidates via yfinance batch download.
        5. Cache failures.
        6. Persist new tickers with 'auto-discovered' tag.
        7. Detect and deactivate stale tickers.
        8. Record completion.

    Args:
        conn: Active database connection.
        settings: Configuration settings (uses defaults if None).
        on_progress: Optional async callback ``async def(message: str) -> None``.

    Returns:
        DiscoveryResult with stats from the run.
    """
    if settings is None:
        settings = get_settings()

    result = DiscoveryResult()

    # 1. Record start
    cursor = conn.execute(
        "INSERT INTO discovery_runs (status) VALUES ('running')"
    )
    run_id = cursor.lastrowid
    conn.commit()

    async def _progress(msg: str) -> None:
        logger.info(msg)
        if on_progress:
            await on_progress(msg)

    try:
        # 2. Fetch CBOE optionable symbols
        await _progress("Fetching CBOE optionable securities list...")
        cboe_symbols = await asyncio.to_thread(
            _fetch_cboe_optionable, settings.cboe_optionable_url
        )
        result.cboe_fetched = len(cboe_symbols)
        await _progress(f"Fetched {result.cboe_fetched} symbols from CBOE")

        # 3. Dedup against existing universe + cached failures
        existing = set(get_full_universe(conn))
        cached_failures = _get_cached_failures(conn, settings.failure_cache_ttl_hours)
        candidates = [
            s for s in cboe_symbols
            if s not in existing and s not in cached_failures
        ]
        result.candidates_checked = len(candidates)
        await _progress(
            f"{result.candidates_checked} new candidates after dedup "
            f"(excluded {len(existing)} existing, {len(cached_failures)} cached failures)"
        )

        # 4. Validate candidates via yfinance
        if candidates:
            await _progress(f"Validating {len(candidates)} candidates via yfinance...")
            passed, failed = await asyncio.to_thread(
                _validate_via_yfinance, candidates, settings
            )
            await _progress(
                f"Validation complete: {len(passed)} passed, {len(failed)} failed"
            )

            # 5. Cache failures
            _cache_failures(conn, failed, reason="validation_failed")
            result.failures_cached = len(failed)

            # 6. Persist new tickers
            if passed:
                added = add_tickers(
                    conn, passed, tags=["auto-discovered"], source="discovered"
                )
                result.new_added = added
                await _progress(f"Added {added} new tickers to universe")

        # 7. Detect and deactivate stale tickers
        await _progress("Checking for stale tickers...")
        stale_count = _detect_stale_tickers(
            conn, settings.stale_ticker_threshold_days
        )
        result.stale_deactivated = stale_count
        if stale_count:
            await _progress(f"Deactivated {stale_count} stale tickers")

        # 8. Record completion
        conn.execute(
            """UPDATE discovery_runs
               SET completed_at = datetime('now'),
                   cboe_symbols_fetched = ?,
                   new_tickers_added = ?,
                   stale_tickers_deactivated = ?,
                   status = 'completed'
             WHERE id = ?""",
            (
                result.cboe_fetched,
                result.new_added,
                result.stale_deactivated,
                run_id,
            ),
        )
        conn.commit()
        await _progress("Discovery run completed successfully")

    except Exception as exc:
        logger.exception("Discovery run failed")
        conn.execute(
            """UPDATE discovery_runs
               SET completed_at = datetime('now'),
                   status = 'error',
                   error_message = ?
             WHERE id = ?""",
            (str(exc), run_id),
        )
        conn.commit()
        raise

    return result


def should_run_discovery(
    conn: sqlite3.Connection, settings: Optional[Settings] = None
) -> bool:
    """Check if discovery should run based on last completed run time."""
    if settings is None:
        settings = get_settings()
    row = conn.execute(
        "SELECT completed_at FROM discovery_runs "
        "WHERE status = 'completed' ORDER BY completed_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return True  # Never run before
    last_run = datetime.fromisoformat(row[0])
    if last_run.tzinfo is None:
        last_run = last_run.replace(tzinfo=UTC)
    return datetime.now(UTC) - last_run > timedelta(
        days=settings.universe_refresh_interval_days
    )


def get_last_discovery_run(conn: sqlite3.Connection) -> dict | None:
    """Return most recent discovery run as dict, or None."""
    # Ensure row_factory is set so dict() works
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM discovery_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.row_factory = old_factory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_cboe_optionable(url: str) -> list[str]:
    """Fetch CBOE optionable symbols CSV and return filtered symbol list.

    Filters: symbol must be purely alphabetic and 1-5 chars long (skips
    index options like $SPX, warrants like SPAK+, units with '.', numeric junk).
    """
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))

    symbols = []
    for row in reader:
        # CBOE CSV uses " Stock Symbol" (with leading space) as the column header
        symbol = (
            row.get("Stock Symbol")
            or row.get(" Stock Symbol")
            or row.get("Symbol")
            or row.get(" Symbol")
            or ""
        ).strip()
        if symbol and symbol.isalpha() and 1 <= len(symbol) <= 5:
            symbols.append(symbol.upper())

    return sorted(set(symbols))


def _get_cached_failures(conn: sqlite3.Connection, ttl_hours: int) -> set[str]:
    """Return set of symbols that failed within the TTL window."""
    rows = conn.execute(
        "SELECT symbol FROM discovery_failures WHERE failed_at > datetime('now', ?)",
        (f"-{ttl_hours} hours",),
    ).fetchall()
    return {row[0] for row in rows}


def _cache_failures(
    conn: sqlite3.Connection, symbols: list[str], reason: str = "validation_failed"
) -> None:
    """Cache failed symbols in discovery_failures table."""
    if not symbols:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO discovery_failures (symbol, reason, failed_at) "
        "VALUES (?, ?, datetime('now'))",
        [(s, reason) for s in symbols],
    )
    conn.commit()


def _validate_via_yfinance(
    candidates: list[str], settings: Settings
) -> tuple[list[str], list[str]]:
    """Validate candidates via yfinance batch download.

    Returns:
        Tuple of (passed, failed) symbol lists.
    """
    if not candidates:
        return [], []

    passed: list[str] = []
    failed: list[str] = []
    batch_size = max(1, settings.discovery_batch_size)

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        symbols_str = " ".join(batch)

        try:
            df = yf.download(
                symbols_str, period="5d", progress=False, threads=False
            )

            if df.empty:
                failed.extend(batch)
                continue

            for symbol in batch:
                try:
                    if len(batch) == 1:
                        close_series = df["Close"]
                        vol_series = df["Volume"]
                    else:
                        if symbol not in df["Close"].columns:
                            failed.append(symbol)
                            continue
                        close_series = df["Close"][symbol].dropna()
                        vol_series = df["Volume"][symbol].dropna()

                    if close_series.empty or vol_series.empty:
                        failed.append(symbol)
                        continue

                    last_price = float(close_series.iloc[-1])
                    avg_volume = float(vol_series.mean())

                    if (
                        last_price >= settings.min_price
                        and avg_volume >= settings.min_avg_volume
                    ):
                        passed.append(symbol)
                    else:
                        failed.append(symbol)
                except (KeyError, IndexError):
                    failed.append(symbol)

        except Exception as e:
            logger.warning("yfinance batch %d failed: %s", i, e)
            failed.extend(batch)

    return passed, failed


def _detect_stale_tickers(
    conn: sqlite3.Connection, threshold_days: int
) -> int:
    """Find and deactivate tickers that haven't been scanned recently.

    A ticker is stale if:
      - last_scanned_at IS NULL and created_at is older than threshold, OR
      - last_scanned_at is older than threshold.

    Returns:
        Number of tickers deactivated.
    """
    rows = conn.execute(
        """
        SELECT symbol FROM universe_tickers
        WHERE is_active = 1
          AND (
            (last_scanned_at IS NULL
             AND created_at < datetime('now', ?))
            OR
            (last_scanned_at IS NOT NULL
             AND last_scanned_at < datetime('now', ?))
          )
        """,
        (f"-{threshold_days} days", f"-{threshold_days} days"),
    ).fetchall()

    deactivated = 0
    for row in rows:
        symbol = row[0]
        try:
            toggle_ticker(conn, symbol, active=False)
            deactivated += 1
        except ValueError:
            # SAVEPOINT safety: can't deactivate if it would leave 0 active
            logger.debug(
                "Cannot deactivate stale ticker %s (would leave 0 active)",
                symbol,
            )

    return deactivated
