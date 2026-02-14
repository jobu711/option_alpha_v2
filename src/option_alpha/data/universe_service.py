"""Universe service — database-driven ticker universe management."""

import sqlite3
import logging
import re

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Convert tag name to URL-friendly slug."""
    slug = name.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s]+', '-', slug).strip('-')
    return slug


def seed_universe(conn: sqlite3.Connection) -> None:
    """Seed universe tables from hardcoded lists. Idempotent — skips if data exists."""
    count = conn.execute("SELECT COUNT(*) FROM universe_tickers").fetchone()[0]
    if count > 0:
        logger.debug("Universe already seeded (%d tickers), skipping", count)
        return

    from option_alpha.data.universe import SP500_CORE, POPULAR_OPTIONS, OPTIONABLE_ETFS

    # Define preset tags
    tag_map = {
        "S&P 500": SP500_CORE,
        "Popular Options": POPULAR_OPTIONS,
        "Optionable ETFs": OPTIONABLE_ETFS,
    }

    # Collect all unique tickers
    all_tickers = set()
    for tickers in tag_map.values():
        all_tickers.update(tickers)

    # Insert tickers
    conn.executemany(
        "INSERT OR IGNORE INTO universe_tickers (symbol, source) VALUES (?, 'preset')",
        [(t,) for t in sorted(all_tickers)],
    )

    # Insert preset tags
    for name in tag_map:
        slug = _slugify(name)
        conn.execute(
            "INSERT INTO universe_tags (name, slug, is_preset) VALUES (?, ?, 1)",
            (name, slug),
        )

    # Insert tag associations
    for name, tickers in tag_map.items():
        slug = _slugify(name)
        tag_id = conn.execute(
            "SELECT id FROM universe_tags WHERE slug = ?", (slug,)
        ).fetchone()[0]
        conn.executemany(
            "INSERT OR IGNORE INTO ticker_tags (symbol, tag_id) VALUES (?, ?)",
            [(t, tag_id) for t in tickers],
        )

    conn.commit()
    logger.info("Seeded universe: %d tickers, %d tags", len(all_tickers), len(tag_map))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _count_active_tickers(conn: sqlite3.Connection) -> int:
    """Return the number of distinct active tickers belonging to active tags."""
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT ut.symbol)
        FROM universe_tickers ut
        JOIN ticker_tags tt ON ut.symbol = tt.symbol
        JOIN universe_tags tg ON tt.tag_id = tg.id
        WHERE ut.is_active = 1 AND tg.is_active = 1
        """
    ).fetchone()
    return row[0]


def get_active_universe(conn: sqlite3.Connection) -> list[str]:
    """Return all active ticker symbols (active tickers in at least one active tag).

    Returns:
        Sorted list of distinct ticker symbols.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT ut.symbol
        FROM universe_tickers ut
        JOIN ticker_tags tt ON ut.symbol = tt.symbol
        JOIN universe_tags tg ON tt.tag_id = tg.id
        WHERE ut.is_active = 1 AND tg.is_active = 1
        ORDER BY ut.symbol
        """
    ).fetchall()
    return [row[0] for row in rows]


def get_full_universe(conn: sqlite3.Connection) -> list[str]:
    """Return ALL tickers regardless of active status, sorted and deduplicated.

    Backward-compatible wrapper — returns the same 702-ticker sorted list
    as the old ``universe.get_full_universe()`` when all tickers are active.

    Returns:
        Sorted list of all ticker symbols in the database.
    """
    rows = conn.execute(
        "SELECT symbol FROM universe_tickers ORDER BY symbol"
    ).fetchall()
    return [row[0] for row in rows]


def get_tickers_by_tag(conn: sqlite3.Connection, tag_slug: str) -> list[str]:
    """Return ticker symbols associated with a specific tag.

    Args:
        conn: Active database connection.
        tag_slug: URL-friendly slug identifying the tag.

    Returns:
        Sorted list of ticker symbols for the tag, or empty list if not found.
    """
    rows = conn.execute(
        """
        SELECT tt.symbol
        FROM ticker_tags tt
        JOIN universe_tags tg ON tt.tag_id = tg.id
        WHERE tg.slug = ?
        ORDER BY tt.symbol
        """,
        (tag_slug,),
    ).fetchall()
    return [row[0] for row in rows]


def get_all_tags(conn: sqlite3.Connection) -> list[dict]:
    """Return all tags with ticker counts.

    Returns:
        List of dicts with keys: id, name, slug, is_preset, is_active, ticker_count.
    """
    rows = conn.execute(
        """
        SELECT tg.id, tg.name, tg.slug, tg.is_preset, tg.is_active,
               COUNT(tt.symbol) AS ticker_count
        FROM universe_tags tg
        LEFT JOIN ticker_tags tt ON tg.id = tt.tag_id
        GROUP BY tg.id
        ORDER BY tg.name
        """
    ).fetchall()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "slug": row["slug"],
            "is_preset": row["is_preset"],
            "is_active": row["is_active"],
            "ticker_count": row["ticker_count"],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def add_tickers(
    conn: sqlite3.Connection,
    symbols: list[str],
    tags: list[str] | None = None,
    source: str = "manual",
) -> int:
    """Insert new tickers, optionally associating them with tags.

    Args:
        conn: Active database connection.
        symbols: Ticker symbols to add.
        tags: Optional list of tag slugs to associate with.
        source: Source label for the tickers (default ``"manual"``).

    Returns:
        Number of tickers actually inserted (duplicates are ignored).
    """
    if not symbols:
        return 0

    before = conn.execute("SELECT COUNT(*) FROM universe_tickers").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO universe_tickers (symbol, source) VALUES (?, ?)",
        [(s.upper(), source) for s in symbols],
    )

    if tags:
        for tag_slug in tags:
            tag_row = conn.execute(
                "SELECT id FROM universe_tags WHERE slug = ?", (tag_slug,)
            ).fetchone()
            if tag_row is None:
                continue
            tag_id = tag_row[0]
            conn.executemany(
                "INSERT OR IGNORE INTO ticker_tags (symbol, tag_id) VALUES (?, ?)",
                [(s.upper(), tag_id) for s in symbols],
            )

    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM universe_tickers").fetchone()[0]
    return after - before


def remove_tickers(conn: sqlite3.Connection, symbols: list[str]) -> int:
    """Delete tickers and their tag associations (CASCADE handles this).

    Args:
        conn: Active database connection.
        symbols: Ticker symbols to remove.

    Returns:
        Number of tickers actually removed.
    """
    if not symbols:
        return 0

    removed = 0
    for s in symbols:
        cursor = conn.execute(
            "DELETE FROM universe_tickers WHERE symbol = ?", (s.upper(),)
        )
        removed += cursor.rowcount
    conn.commit()
    return removed


def toggle_ticker(conn: sqlite3.Connection, symbol: str, active: bool) -> None:
    """Toggle a ticker's active status.

    Args:
        conn: Active database connection.
        symbol: Ticker symbol to toggle.
        active: New active state.

    Raises:
        ValueError: If deactivating would result in 0 active tickers.
    """
    symbol = symbol.upper()
    if not active:
        # Use savepoint to test the change safely
        conn.execute("SAVEPOINT toggle_ticker")
        conn.execute(
            "UPDATE universe_tickers SET is_active = 0 WHERE symbol = ?", (symbol,)
        )
        remaining = _count_active_tickers(conn)
        if remaining == 0:
            conn.execute("ROLLBACK TO toggle_ticker")
            conn.execute("RELEASE toggle_ticker")
            raise ValueError(
                "Cannot deactivate ticker — would result in 0 active tickers"
            )
        conn.execute("RELEASE toggle_ticker")
    else:
        conn.execute(
            "UPDATE universe_tickers SET is_active = 1 WHERE symbol = ?", (symbol,)
        )
    conn.commit()


def toggle_tag(conn: sqlite3.Connection, tag_slug: str, active: bool) -> None:
    """Toggle a tag's active status.

    Args:
        conn: Active database connection.
        tag_slug: Slug of the tag to toggle.
        active: New active state.

    Raises:
        ValueError: If deactivating would result in 0 active tickers.
    """
    if not active:
        conn.execute("SAVEPOINT toggle_tag")
        conn.execute(
            "UPDATE universe_tags SET is_active = 0 WHERE slug = ?", (tag_slug,)
        )
        remaining = _count_active_tickers(conn)
        if remaining == 0:
            conn.execute("ROLLBACK TO toggle_tag")
            conn.execute("RELEASE toggle_tag")
            raise ValueError(
                "Cannot deactivate tag — would result in 0 active tickers"
            )
        conn.execute("RELEASE toggle_tag")
    else:
        conn.execute(
            "UPDATE universe_tags SET is_active = 1 WHERE slug = ?", (tag_slug,)
        )
    conn.commit()


def create_tag(conn: sqlite3.Connection, name: str) -> dict:
    """Create a new user tag.

    Args:
        conn: Active database connection.
        name: Display name for the tag.

    Returns:
        Dict with keys: id, name, slug, is_preset, is_active.

    Raises:
        ValueError: If a tag with this name (slug) already exists.
    """
    slug = _slugify(name)
    if not slug:
        raise ValueError("Tag name must contain at least one alphanumeric character")
    existing = conn.execute(
        "SELECT id FROM universe_tags WHERE slug = ?", (slug,)
    ).fetchone()
    if existing is not None:
        raise ValueError(f"Tag '{name}' already exists (slug: {slug})")

    cursor = conn.execute(
        "INSERT INTO universe_tags (name, slug, is_preset, is_active) VALUES (?, ?, 0, 1)",
        (name, slug),
    )
    conn.commit()
    return {
        "id": cursor.lastrowid,
        "name": name,
        "slug": slug,
        "is_preset": 0,
        "is_active": 1,
    }


def delete_tag(conn: sqlite3.Connection, tag_slug: str) -> None:
    """Delete a tag. Untags tickers but does NOT delete them.

    CASCADE on ``ticker_tags`` handles unlinking automatically.

    Args:
        conn: Active database connection.
        tag_slug: Slug of the tag to delete.

    Raises:
        ValueError: If the tag is not found.
    """
    cursor = conn.execute(
        "DELETE FROM universe_tags WHERE slug = ?", (tag_slug,)
    )
    if cursor.rowcount == 0:
        raise ValueError(f"Tag not found: {tag_slug}")
    conn.commit()


def tag_tickers(conn: sqlite3.Connection, symbols: list[str], tag_slug: str) -> int:
    """Associate tickers with a tag.

    Args:
        conn: Active database connection.
        symbols: Ticker symbols to tag.
        tag_slug: Slug of the tag to associate with.

    Returns:
        Number of associations created (duplicates ignored).
    """
    tag_row = conn.execute(
        "SELECT id FROM universe_tags WHERE slug = ?", (tag_slug,)
    ).fetchone()
    if tag_row is None:
        return 0

    tag_id = tag_row[0]
    before = conn.execute(
        "SELECT COUNT(*) FROM ticker_tags WHERE tag_id = ?", (tag_id,)
    ).fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO ticker_tags (symbol, tag_id) VALUES (?, ?)",
        [(s.upper(), tag_id) for s in symbols],
    )
    conn.commit()
    after = conn.execute(
        "SELECT COUNT(*) FROM ticker_tags WHERE tag_id = ?", (tag_id,)
    ).fetchone()[0]
    return after - before


def untag_tickers(conn: sqlite3.Connection, symbols: list[str], tag_slug: str) -> int:
    """Remove ticker-tag associations.

    Args:
        conn: Active database connection.
        symbols: Ticker symbols to untag.
        tag_slug: Slug of the tag to disassociate from.

    Returns:
        Number of associations removed.
    """
    tag_row = conn.execute(
        "SELECT id FROM universe_tags WHERE slug = ?", (tag_slug,)
    ).fetchone()
    if tag_row is None:
        return 0

    tag_id = tag_row[0]
    removed = 0
    for s in symbols:
        cursor = conn.execute(
            "DELETE FROM ticker_tags WHERE symbol = ? AND tag_id = ?",
            (s.upper(), tag_id),
        )
        removed += cursor.rowcount
    conn.commit()
    return removed
