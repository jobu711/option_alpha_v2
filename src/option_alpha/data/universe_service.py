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
