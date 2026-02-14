"""Tests for the universe service layer (CRUD operations)."""

import sqlite3

import pytest

from option_alpha.data.universe import (
    SP500_CORE,
    POPULAR_OPTIONS,
    OPTIONABLE_ETFS,
    get_full_universe as legacy_get_full_universe,
)
from option_alpha.data.universe_service import (
    add_tickers,
    create_tag,
    delete_tag,
    get_active_universe,
    get_all_tags,
    get_full_universe,
    get_tickers_by_tag,
    remove_tickers,
    tag_tickers,
    toggle_tag,
    toggle_ticker,
    untag_tickers,
)
from option_alpha.persistence.database import initialize_db

EXPECTED_COUNT = len(set(SP500_CORE + POPULAR_OPTIONS + OPTIONABLE_ETFS))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """Provide a fresh DB with migrations and seeding."""
    connection = initialize_db(":memory:")
    yield connection
    connection.close()


# ===========================================================================
# get_active_universe
# ===========================================================================

class TestGetActiveUniverse:
    """Test get_active_universe returns only active tickers with active tags."""

    def test_returns_all_when_everything_active(self, conn: sqlite3.Connection):
        """With default seeding (all active), should return all tickers."""
        result = get_active_universe(conn)
        assert len(result) == EXPECTED_COUNT

    def test_sorted_alphabetically(self, conn: sqlite3.Connection):
        """Result should be sorted."""
        result = get_active_universe(conn)
        assert result == sorted(result)

    def test_deactivated_ticker_excluded(self, conn: sqlite3.Connection):
        """A deactivated ticker should not appear in the active universe."""
        assert "AAPL" in get_active_universe(conn)
        conn.execute(
            "UPDATE universe_tickers SET is_active = 0 WHERE symbol = 'AAPL'"
        )
        conn.commit()
        result = get_active_universe(conn)
        assert "AAPL" not in result
        assert len(result) == EXPECTED_COUNT - 1

    def test_deactivated_tag_excludes_its_tickers(self, conn: sqlite3.Connection):
        """When a tag is deactivated, tickers ONLY in that tag disappear."""
        # Find a ticker only in the sp-500 tag (not in other tags)
        sp500_only = set(SP500_CORE) - set(POPULAR_OPTIONS) - set(OPTIONABLE_ETFS)
        sample = sorted(sp500_only)[0]  # deterministic pick

        assert sample in get_active_universe(conn)
        conn.execute(
            "UPDATE universe_tags SET is_active = 0 WHERE slug = 'sp-500'"
        )
        conn.commit()
        result = get_active_universe(conn)
        assert sample not in result

    def test_ticker_in_multiple_tags_survives_single_tag_deactivation(
        self, conn: sqlite3.Connection
    ):
        """Ticker in multiple active tags should remain if one tag is deactivated."""
        # ENPH is in both SP500_CORE and POPULAR_OPTIONS
        assert "ENPH" in get_active_universe(conn)
        conn.execute(
            "UPDATE universe_tags SET is_active = 0 WHERE slug = 'sp-500'"
        )
        conn.commit()
        assert "ENPH" in get_active_universe(conn)

    def test_no_duplicates(self, conn: sqlite3.Connection):
        """Result should have no duplicates even if ticker is in multiple tags."""
        result = get_active_universe(conn)
        assert len(result) == len(set(result))


# ===========================================================================
# get_full_universe
# ===========================================================================

class TestGetFullUniverse:
    """Test get_full_universe backward compatibility."""

    def test_returns_expected_tickers(self, conn: sqlite3.Connection):
        """Should return the expected number of tickers."""
        result = get_full_universe(conn)
        assert len(result) == EXPECTED_COUNT

    def test_matches_legacy_function(self, conn: sqlite3.Connection):
        """Should match the old hardcoded get_full_universe() output exactly."""
        legacy = legacy_get_full_universe()
        db_result = get_full_universe(conn)
        assert db_result == legacy

    def test_sorted(self, conn: sqlite3.Connection):
        """Should be sorted alphabetically."""
        result = get_full_universe(conn)
        assert result == sorted(result)

    def test_includes_inactive_tickers(self, conn: sqlite3.Connection):
        """Should return tickers regardless of active status."""
        conn.execute(
            "UPDATE universe_tickers SET is_active = 0 WHERE symbol = 'AAPL'"
        )
        conn.commit()
        result = get_full_universe(conn)
        assert "AAPL" in result
        assert len(result) == EXPECTED_COUNT


# ===========================================================================
# get_tickers_by_tag
# ===========================================================================

class TestGetTickersByTag:
    """Test get_tickers_by_tag returns correct tickers for each tag."""

    def test_sp500_tag(self, conn: sqlite3.Connection):
        """sp-500 tag should return all SP500_CORE tickers."""
        result = get_tickers_by_tag(conn, "sp-500")
        assert set(result) == set(SP500_CORE)

    def test_popular_options_tag(self, conn: sqlite3.Connection):
        """popular-options tag should return all unique POPULAR_OPTIONS tickers."""
        result = get_tickers_by_tag(conn, "popular-options")
        assert set(result) == set(POPULAR_OPTIONS)

    def test_optionable_etfs_tag(self, conn: sqlite3.Connection):
        """optionable-etfs tag should return all OPTIONABLE_ETFS tickers."""
        result = get_tickers_by_tag(conn, "optionable-etfs")
        assert set(result) == set(OPTIONABLE_ETFS)

    def test_sorted_result(self, conn: sqlite3.Connection):
        """Result should be sorted."""
        result = get_tickers_by_tag(conn, "sp-500")
        assert result == sorted(result)

    def test_unknown_slug_returns_empty(self, conn: sqlite3.Connection):
        """Unknown tag slug should return empty list."""
        assert get_tickers_by_tag(conn, "nonexistent-tag") == []


# ===========================================================================
# get_all_tags
# ===========================================================================

class TestGetAllTags:
    """Test get_all_tags returns correct tag metadata and counts."""

    def test_returns_three_preset_tags(self, conn: sqlite3.Connection):
        """Should return 3 preset tags after seeding."""
        tags = get_all_tags(conn)
        assert len(tags) == 3

    def test_tag_dict_keys(self, conn: sqlite3.Connection):
        """Each tag dict should have the expected keys."""
        tags = get_all_tags(conn)
        expected_keys = {"id", "name", "slug", "is_preset", "is_active", "ticker_count"}
        for tag in tags:
            assert set(tag.keys()) == expected_keys

    def test_all_preset(self, conn: sqlite3.Connection):
        """All seeded tags should be preset."""
        tags = get_all_tags(conn)
        assert all(tag["is_preset"] == 1 for tag in tags)

    def test_all_active(self, conn: sqlite3.Connection):
        """All seeded tags should be active."""
        tags = get_all_tags(conn)
        assert all(tag["is_active"] == 1 for tag in tags)

    def test_ticker_counts(self, conn: sqlite3.Connection):
        """Each tag should have the correct ticker count."""
        tags = get_all_tags(conn)
        counts = {tag["slug"]: tag["ticker_count"] for tag in tags}
        assert counts["sp-500"] == len(SP500_CORE)
        assert counts["popular-options"] == len(set(POPULAR_OPTIONS))
        assert counts["optionable-etfs"] == len(OPTIONABLE_ETFS)

    def test_includes_user_tags(self, conn: sqlite3.Connection):
        """User-created tags should also appear."""
        create_tag(conn, "My Custom Tag")
        tags = get_all_tags(conn)
        assert len(tags) == 4
        custom = [t for t in tags if t["slug"] == "my-custom-tag"]
        assert len(custom) == 1
        assert custom[0]["ticker_count"] == 0
        assert custom[0]["is_preset"] == 0


# ===========================================================================
# add_tickers
# ===========================================================================

class TestAddTickers:
    """Test add_tickers inserts new tickers correctly."""

    def test_add_new_tickers(self, conn: sqlite3.Connection):
        """Should add new tickers and return the count."""
        count = add_tickers(conn, ["NEWT1", "NEWT2", "NEWT3"])
        assert count == 3
        result = get_full_universe(conn)
        assert "NEWT1" in result
        assert "NEWT2" in result
        assert "NEWT3" in result

    def test_duplicate_tickers_ignored(self, conn: sqlite3.Connection):
        """Existing tickers should be ignored gracefully."""
        count = add_tickers(conn, ["AAPL", "MSFT", "NEWX"])
        assert count == 1  # only NEWX is new
        assert "NEWX" in get_full_universe(conn)

    def test_returns_zero_for_all_duplicates(self, conn: sqlite3.Connection):
        """Should return 0 when all tickers already exist."""
        count = add_tickers(conn, ["AAPL", "MSFT"])
        assert count == 0

    def test_with_tag_association(self, conn: sqlite3.Connection):
        """New tickers should be associated with specified tags."""
        add_tickers(conn, ["NEWTAG1"], tags=["sp-500"])
        result = get_tickers_by_tag(conn, "sp-500")
        assert "NEWTAG1" in result

    def test_with_multiple_tags(self, conn: sqlite3.Connection):
        """Ticker should be associated with all specified tags."""
        add_tickers(conn, ["MULTI1"], tags=["sp-500", "popular-options"])
        assert "MULTI1" in get_tickers_by_tag(conn, "sp-500")
        assert "MULTI1" in get_tickers_by_tag(conn, "popular-options")

    def test_unknown_tag_ignored(self, conn: sqlite3.Connection):
        """Unknown tag slugs should be silently ignored."""
        count = add_tickers(conn, ["SAFE1"], tags=["nonexistent"])
        assert count == 1
        assert "SAFE1" in get_full_universe(conn)

    def test_source_parameter(self, conn: sqlite3.Connection):
        """Source should be stored correctly."""
        add_tickers(conn, ["SRCT1"], source="api")
        row = conn.execute(
            "SELECT source FROM universe_tickers WHERE symbol = 'SRCT1'"
        ).fetchone()
        assert row["source"] == "api"

    def test_empty_list(self, conn: sqlite3.Connection):
        """Empty list should return 0."""
        assert add_tickers(conn, []) == 0

    def test_uppercase_normalization(self, conn: sqlite3.Connection):
        """Symbols should be uppercased."""
        add_tickers(conn, ["lowcase"])
        assert "LOWCASE" in get_full_universe(conn)


# ===========================================================================
# remove_tickers
# ===========================================================================

class TestRemoveTickers:
    """Test remove_tickers deletes tickers and cascades."""

    def test_removes_existing_tickers(self, conn: sqlite3.Connection):
        """Should remove tickers and return the count."""
        count = remove_tickers(conn, ["AAPL", "MSFT"])
        assert count == 2
        result = get_full_universe(conn)
        assert "AAPL" not in result
        assert "MSFT" not in result

    def test_cascade_removes_tag_associations(self, conn: sqlite3.Connection):
        """Tag associations should be removed by CASCADE."""
        # AAPL is in sp-500
        assert "AAPL" in get_tickers_by_tag(conn, "sp-500")
        remove_tickers(conn, ["AAPL"])
        assert "AAPL" not in get_tickers_by_tag(conn, "sp-500")

    def test_nonexistent_ticker_ignored(self, conn: sqlite3.Connection):
        """Removing a non-existent ticker should not fail."""
        count = remove_tickers(conn, ["ZZZZZ"])
        assert count == 0

    def test_mixed_existing_and_nonexistent(self, conn: sqlite3.Connection):
        """Should remove existing tickers and skip non-existent ones."""
        count = remove_tickers(conn, ["AAPL", "ZZZZZ"])
        assert count == 1

    def test_returns_correct_count(self, conn: sqlite3.Connection):
        """Return value should match actual deletions."""
        before = len(get_full_universe(conn))
        removed = remove_tickers(conn, ["AAPL", "MSFT", "GOOGL"])
        assert removed == 3
        assert len(get_full_universe(conn)) == before - 3

    def test_empty_list(self, conn: sqlite3.Connection):
        """Empty list should return 0."""
        assert remove_tickers(conn, []) == 0


# ===========================================================================
# toggle_ticker
# ===========================================================================

class TestToggleTicker:
    """Test toggle_ticker activates/deactivates tickers."""

    def test_deactivate_ticker(self, conn: sqlite3.Connection):
        """Should deactivate a ticker."""
        toggle_ticker(conn, "AAPL", active=False)
        row = conn.execute(
            "SELECT is_active FROM universe_tickers WHERE symbol = 'AAPL'"
        ).fetchone()
        assert row["is_active"] == 0
        assert "AAPL" not in get_active_universe(conn)

    def test_reactivate_ticker(self, conn: sqlite3.Connection):
        """Should reactivate a previously deactivated ticker."""
        toggle_ticker(conn, "AAPL", active=False)
        assert "AAPL" not in get_active_universe(conn)
        toggle_ticker(conn, "AAPL", active=True)
        assert "AAPL" in get_active_universe(conn)

    def test_prevents_empty_universe(self, conn: sqlite3.Connection):
        """Should raise ValueError if deactivation would leave 0 active tickers."""
        # Deactivate all tags except one, then try to deactivate all tickers in it
        conn.execute("UPDATE universe_tags SET is_active = 0 WHERE slug != 'sp-500'")
        conn.commit()

        # Deactivate all SP500 tickers except one
        sp500_tickers = get_tickers_by_tag(conn, "sp-500")
        # Only tickers exclusively in sp-500 matter (not in other active tags)
        # Since other tags are inactive, only sp-500 tickers are active
        for ticker in sp500_tickers[:-1]:
            toggle_ticker(conn, ticker, active=False)

        # The last one should fail
        with pytest.raises(ValueError, match="0 active tickers"):
            toggle_ticker(conn, sp500_tickers[-1], active=False)

        # Verify the ticker is still active (rollback worked)
        assert sp500_tickers[-1] in get_active_universe(conn)

    def test_activate_is_always_safe(self, conn: sqlite3.Connection):
        """Activating should never raise, even if already active."""
        toggle_ticker(conn, "AAPL", active=True)
        assert "AAPL" in get_active_universe(conn)


# ===========================================================================
# toggle_tag
# ===========================================================================

class TestToggleTag:
    """Test toggle_tag activates/deactivates tags."""

    def test_deactivate_tag(self, conn: sqlite3.Connection):
        """Should deactivate a tag."""
        toggle_tag(conn, "optionable-etfs", active=False)
        tags = get_all_tags(conn)
        etf_tag = [t for t in tags if t["slug"] == "optionable-etfs"][0]
        assert etf_tag["is_active"] == 0

    def test_reactivate_tag(self, conn: sqlite3.Connection):
        """Should reactivate a tag."""
        toggle_tag(conn, "optionable-etfs", active=False)
        toggle_tag(conn, "optionable-etfs", active=True)
        tags = get_all_tags(conn)
        etf_tag = [t for t in tags if t["slug"] == "optionable-etfs"][0]
        assert etf_tag["is_active"] == 1

    def test_deactivating_tag_reduces_active_universe(self, conn: sqlite3.Connection):
        """Deactivating a tag should remove tickers unique to that tag."""
        before = len(get_active_universe(conn))
        toggle_tag(conn, "optionable-etfs", active=False)
        after = len(get_active_universe(conn))
        # Some ETFs might also be in other tags, so just check it decreased
        assert after < before

    def test_prevents_empty_universe(self, conn: sqlite3.Connection):
        """Should raise ValueError if deactivation would leave 0 active tickers."""
        # Deactivate two tags first
        toggle_tag(conn, "popular-options", active=False)
        toggle_tag(conn, "optionable-etfs", active=False)

        # Now deactivate all tickers NOT in sp-500 (they're already excluded by tag)
        # Then try to deactivate sp-500 tag
        with pytest.raises(ValueError, match="0 active tickers"):
            toggle_tag(conn, "sp-500", active=False)

        # Verify tag is still active (rollback worked)
        tags = get_all_tags(conn)
        sp500 = [t for t in tags if t["slug"] == "sp-500"][0]
        assert sp500["is_active"] == 1

    def test_activate_is_always_safe(self, conn: sqlite3.Connection):
        """Activating should never raise."""
        toggle_tag(conn, "sp-500", active=True)
        tags = get_all_tags(conn)
        sp500 = [t for t in tags if t["slug"] == "sp-500"][0]
        assert sp500["is_active"] == 1


# ===========================================================================
# create_tag
# ===========================================================================

class TestCreateTag:
    """Test create_tag creates tags with correct metadata."""

    def test_creates_tag(self, conn: sqlite3.Connection):
        """Should create a new tag and return its metadata."""
        result = create_tag(conn, "My Watchlist")
        assert result["name"] == "My Watchlist"
        assert result["slug"] == "my-watchlist"
        assert result["is_preset"] == 0
        assert result["is_active"] == 1
        assert isinstance(result["id"], int)

    def test_tag_persisted(self, conn: sqlite3.Connection):
        """Created tag should be retrievable from the database."""
        create_tag(conn, "Tech Stocks")
        tags = get_all_tags(conn)
        slugs = [t["slug"] for t in tags]
        assert "tech-stocks" in slugs

    def test_prevents_duplicate_name(self, conn: sqlite3.Connection):
        """Should raise ValueError if a tag with the same slug already exists."""
        create_tag(conn, "Unique Tag")
        with pytest.raises(ValueError, match="already exists"):
            create_tag(conn, "Unique Tag")

    def test_prevents_duplicate_slug_collision(self, conn: sqlite3.Connection):
        """Tags that slugify to the same value should be rejected."""
        create_tag(conn, "My Tag")
        with pytest.raises(ValueError, match="already exists"):
            create_tag(conn, "My  Tag")  # extra space, same slug

    def test_slug_generation(self, conn: sqlite3.Connection):
        """Slug should be auto-generated correctly."""
        result = create_tag(conn, "High Beta & Growth")
        assert result["slug"] == "high-beta-growth"


# ===========================================================================
# delete_tag
# ===========================================================================

class TestDeleteTag:
    """Test delete_tag removes tags without deleting tickers."""

    def test_deletes_tag(self, conn: sqlite3.Connection):
        """Should remove the tag from the database."""
        create_tag(conn, "Temp Tag")
        delete_tag(conn, "temp-tag")
        tags = get_all_tags(conn)
        slugs = [t["slug"] for t in tags]
        assert "temp-tag" not in slugs

    def test_untags_tickers_without_deleting_them(self, conn: sqlite3.Connection):
        """Deleting a tag should unlink tickers but not delete them."""
        # Create tag and associate a ticker
        create_tag(conn, "Deletable")
        add_tickers(conn, ["KEEP1"], tags=["deletable"])
        assert "KEEP1" in get_tickers_by_tag(conn, "deletable")

        delete_tag(conn, "deletable")

        # Ticker should still exist
        assert "KEEP1" in get_full_universe(conn)
        # But no longer tagged
        assert get_tickers_by_tag(conn, "deletable") == []

    def test_error_on_not_found(self, conn: sqlite3.Connection):
        """Should raise ValueError for non-existent tag."""
        with pytest.raises(ValueError, match="Tag not found"):
            delete_tag(conn, "nonexistent-slug")

    def test_delete_preset_tag(self, conn: sqlite3.Connection):
        """Should be able to delete a preset tag."""
        delete_tag(conn, "optionable-etfs")
        tags = get_all_tags(conn)
        slugs = [t["slug"] for t in tags]
        assert "optionable-etfs" not in slugs

    def test_tickers_still_active_after_tag_delete(self, conn: sqlite3.Connection):
        """Tickers that were in the deleted tag should remain active."""
        # Get a ticker unique to optionable-etfs
        etf_only = set(OPTIONABLE_ETFS) - set(SP500_CORE) - set(POPULAR_OPTIONS)
        sample = sorted(etf_only)[0]

        before = conn.execute(
            "SELECT is_active FROM universe_tickers WHERE symbol = ?", (sample,)
        ).fetchone()["is_active"]
        assert before == 1

        delete_tag(conn, "optionable-etfs")

        after = conn.execute(
            "SELECT is_active FROM universe_tickers WHERE symbol = ?", (sample,)
        ).fetchone()["is_active"]
        assert after == 1  # still active, just untagged


# ===========================================================================
# tag_tickers / untag_tickers
# ===========================================================================

class TestTagTickers:
    """Test tag_tickers associates tickers with tags."""

    def test_tag_new_tickers(self, conn: sqlite3.Connection):
        """Should create associations and return count."""
        add_tickers(conn, ["TAG1", "TAG2"])
        count = tag_tickers(conn, ["TAG1", "TAG2"], "sp-500")
        assert count == 2
        result = get_tickers_by_tag(conn, "sp-500")
        assert "TAG1" in result
        assert "TAG2" in result

    def test_duplicate_association_ignored(self, conn: sqlite3.Connection):
        """Tagging an already-tagged ticker should not fail."""
        # AAPL is already in sp-500
        count = tag_tickers(conn, ["AAPL"], "sp-500")
        assert count == 0

    def test_unknown_tag_returns_zero(self, conn: sqlite3.Connection):
        """Unknown tag slug should return 0."""
        count = tag_tickers(conn, ["AAPL"], "nonexistent")
        assert count == 0

    def test_mixed_new_and_existing(self, conn: sqlite3.Connection):
        """Should only count new associations."""
        add_tickers(conn, ["NEWTG"])
        count = tag_tickers(conn, ["AAPL", "NEWTG"], "sp-500")
        assert count == 1  # only NEWTG is new


class TestUntagTickers:
    """Test untag_tickers removes ticker-tag associations."""

    def test_untag_tickers(self, conn: sqlite3.Connection):
        """Should remove associations and return count."""
        assert "AAPL" in get_tickers_by_tag(conn, "sp-500")
        count = untag_tickers(conn, ["AAPL"], "sp-500")
        assert count == 1
        assert "AAPL" not in get_tickers_by_tag(conn, "sp-500")

    def test_ticker_not_deleted(self, conn: sqlite3.Connection):
        """Untagging should not delete the ticker itself."""
        untag_tickers(conn, ["AAPL"], "sp-500")
        assert "AAPL" in get_full_universe(conn)

    def test_untag_nonexistent_association(self, conn: sqlite3.Connection):
        """Untagging a ticker not in the tag should return 0."""
        add_tickers(conn, ["ORPHAN"])
        count = untag_tickers(conn, ["ORPHAN"], "sp-500")
        assert count == 0

    def test_unknown_tag_returns_zero(self, conn: sqlite3.Connection):
        """Unknown tag slug should return 0."""
        count = untag_tickers(conn, ["AAPL"], "nonexistent")
        assert count == 0

    def test_untag_multiple(self, conn: sqlite3.Connection):
        """Should untag multiple tickers at once."""
        count = untag_tickers(conn, ["AAPL", "MSFT", "GOOGL"], "sp-500")
        assert count == 3
        result = get_tickers_by_tag(conn, "sp-500")
        assert "AAPL" not in result
        assert "MSFT" not in result
        assert "GOOGL" not in result
