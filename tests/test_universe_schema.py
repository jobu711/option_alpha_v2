"""Tests for universe DB schema, migration, and seeding."""

import sqlite3

import pytest

from option_alpha.data.universe import SP500_CORE, POPULAR_OPTIONS, OPTIONABLE_ETFS
from option_alpha.data.universe_service import _slugify, seed_universe
from option_alpha.persistence.database import initialize_db

EXPECTED_COUNT = len(set(SP500_CORE + POPULAR_OPTIONS + OPTIONABLE_ETFS))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """Provide a fresh in-memory database with all migrations applied."""
    connection = initialize_db(":memory:")
    yield connection
    connection.close()


# ===========================================================================
# Schema / Migration Tests
# ===========================================================================

class TestUniverseSchema:
    """Test that 002_universe migration creates correct tables and indexes."""

    def test_universe_tables_created(self, conn: sqlite3.Connection):
        """All three universe tables should exist."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "universe_tickers" in tables
        assert "universe_tags" in tables
        assert "ticker_tags" in tables

    def test_universe_tickers_columns(self, conn: sqlite3.Connection):
        """universe_tickers should have the expected columns."""
        cursor = conn.execute("PRAGMA table_info(universe_tickers)")
        columns = {row["name"] for row in cursor.fetchall()}
        expected = {"symbol", "name", "sector", "source", "is_active", "created_at", "last_scanned_at"}
        assert expected == columns

    def test_universe_tags_columns(self, conn: sqlite3.Connection):
        """universe_tags should have the expected columns."""
        cursor = conn.execute("PRAGMA table_info(universe_tags)")
        columns = {row["name"] for row in cursor.fetchall()}
        expected = {"id", "name", "slug", "is_preset", "is_active", "created_at"}
        assert expected == columns

    def test_ticker_tags_columns(self, conn: sqlite3.Connection):
        """ticker_tags should have the expected columns."""
        cursor = conn.execute("PRAGMA table_info(ticker_tags)")
        columns = {row["name"] for row in cursor.fetchall()}
        expected = {"symbol", "tag_id"}
        assert expected == columns

    def test_indexes_created(self, conn: sqlite3.Connection):
        """Universe indexes should exist."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_universe_tickers_active" in indexes
        assert "idx_universe_tags_slug" in indexes
        assert "idx_universe_tags_preset" in indexes
        assert "idx_ticker_tags_tag_id" in indexes

    def test_migration_version_recorded(self, conn: sqlite3.Connection):
        """Migration 002 should be recorded in schema_version."""
        cursor = conn.execute(
            "SELECT version, filename FROM schema_version WHERE version = 2"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["filename"] == "002_universe.sql"


# ===========================================================================
# Seeding Tests
# ===========================================================================

class TestUniverseSeeding:
    """Test that seed_universe() populates tables correctly."""

    def test_ticker_count(self, conn: sqlite3.Connection):
        """Seeding should insert the expected number of unique tickers."""
        count = conn.execute("SELECT COUNT(*) FROM universe_tickers").fetchone()[0]
        assert count == EXPECTED_COUNT

    def test_all_tickers_source_preset(self, conn: sqlite3.Connection):
        """All seeded tickers should have source='preset'."""
        count = conn.execute(
            "SELECT COUNT(*) FROM universe_tickers WHERE source != 'preset'"
        ).fetchone()[0]
        assert count == 0

    def test_all_tickers_active(self, conn: sqlite3.Connection):
        """All seeded tickers should be active by default."""
        count = conn.execute(
            "SELECT COUNT(*) FROM universe_tickers WHERE is_active != 1"
        ).fetchone()[0]
        assert count == 0

    def test_preset_tags_created(self, conn: sqlite3.Connection):
        """Seeding should create exactly 3 preset tags."""
        count = conn.execute(
            "SELECT COUNT(*) FROM universe_tags WHERE is_preset = 1"
        ).fetchone()[0]
        assert count == 3

    def test_tag_slugs(self, conn: sqlite3.Connection):
        """Tags should have correct URL-friendly slugs."""
        cursor = conn.execute(
            "SELECT slug FROM universe_tags ORDER BY slug"
        )
        slugs = [row["slug"] for row in cursor.fetchall()]
        assert "optionable-etfs" in slugs
        assert "popular-options" in slugs
        assert "sp-500" in slugs

    def test_tag_names(self, conn: sqlite3.Connection):
        """Tags should have correct display names."""
        cursor = conn.execute(
            "SELECT name FROM universe_tags ORDER BY name"
        )
        names = [row["name"] for row in cursor.fetchall()]
        assert "Optionable ETFs" in names
        assert "Popular Options" in names
        assert "S&P 500" in names

    def test_sp500_tag_associations(self, conn: sqlite3.Connection):
        """S&P 500 tag should be linked to all SP500_CORE tickers."""
        tag_id = conn.execute(
            "SELECT id FROM universe_tags WHERE slug = 'sp-500'"
        ).fetchone()[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE tag_id = ?", (tag_id,)
        ).fetchone()[0]
        assert count == len(SP500_CORE)

    def test_popular_options_tag_associations(self, conn: sqlite3.Connection):
        """Popular Options tag should be linked to all POPULAR_OPTIONS tickers."""
        tag_id = conn.execute(
            "SELECT id FROM universe_tags WHERE slug = 'popular-options'"
        ).fetchone()[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE tag_id = ?", (tag_id,)
        ).fetchone()[0]
        # POPULAR_OPTIONS has a duplicate (RIVN), so unique count is used via INSERT OR IGNORE
        assert count == len(set(POPULAR_OPTIONS))

    def test_optionable_etfs_tag_associations(self, conn: sqlite3.Connection):
        """Optionable ETFs tag should be linked to all OPTIONABLE_ETFS tickers."""
        tag_id = conn.execute(
            "SELECT id FROM universe_tags WHERE slug = 'optionable-etfs'"
        ).fetchone()[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE tag_id = ?", (tag_id,)
        ).fetchone()[0]
        assert count == len(OPTIONABLE_ETFS)

    def test_multi_tag_tickers(self, conn: sqlite3.Connection):
        """Tickers appearing in multiple lists should have multiple tag associations."""
        # ENPH appears in both SP500_CORE and POPULAR_OPTIONS
        count = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE symbol = 'ENPH'"
        ).fetchone()[0]
        assert count == 2

        # GNRC also appears in both SP500_CORE and POPULAR_OPTIONS
        count = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE symbol = 'GNRC'"
        ).fetchone()[0]
        assert count == 2

    def test_etf_only_single_tag(self, conn: sqlite3.Connection):
        """ETFs only in OPTIONABLE_ETFS should have exactly 1 tag."""
        # XBI is only in OPTIONABLE_ETFS (no longer in POPULAR_OPTIONS)
        count = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE symbol = 'XBI'"
        ).fetchone()[0]
        assert count == 1

    def test_seeding_idempotent(self, conn: sqlite3.Connection):
        """Calling seed_universe() a second time should not duplicate data."""
        # Seeding already happened in initialize_db; call again.
        seed_universe(conn)

        ticker_count = conn.execute("SELECT COUNT(*) FROM universe_tickers").fetchone()[0]
        assert ticker_count == EXPECTED_COUNT

        tag_count = conn.execute("SELECT COUNT(*) FROM universe_tags").fetchone()[0]
        assert tag_count == 3


# ===========================================================================
# Slugify Tests
# ===========================================================================

class TestSlugify:
    """Test the _slugify helper function."""

    def test_basic(self):
        assert _slugify("S&P 500") == "sp-500"

    def test_spaces(self):
        assert _slugify("Popular Options") == "popular-options"

    def test_already_slug(self):
        assert _slugify("optionable-etfs") == "optionable-etfs"

    def test_mixed_case(self):
        assert _slugify("Optionable ETFs") == "optionable-etfs"


# ===========================================================================
# Foreign Key / Cascade Tests
# ===========================================================================

class TestUniverseForeignKeys:
    """Test foreign key constraints on universe tables."""

    def test_ticker_tag_cascade_on_ticker_delete(self, conn: sqlite3.Connection):
        """Deleting a ticker should cascade-delete its tag associations."""
        # Pick a ticker we know exists
        before = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE symbol = 'AAPL'"
        ).fetchone()[0]
        assert before >= 1

        conn.execute("DELETE FROM universe_tickers WHERE symbol = 'AAPL'")
        conn.commit()

        after = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE symbol = 'AAPL'"
        ).fetchone()[0]
        assert after == 0

    def test_ticker_tag_cascade_on_tag_delete(self, conn: sqlite3.Connection):
        """Deleting a tag should cascade-delete its ticker associations."""
        tag_id = conn.execute(
            "SELECT id FROM universe_tags WHERE slug = 'sp-500'"
        ).fetchone()[0]

        before = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE tag_id = ?", (tag_id,)
        ).fetchone()[0]
        assert before > 0

        conn.execute("DELETE FROM universe_tags WHERE id = ?", (tag_id,))
        conn.commit()

        after = conn.execute(
            "SELECT COUNT(*) FROM ticker_tags WHERE tag_id = ?", (tag_id,)
        ).fetchone()[0]
        assert after == 0
