"""Tests for custom watchlist management."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from option_alpha.data.watchlists import (
    MAX_TICKERS_PER_WATCHLIST,
    MAX_WATCHLISTS,
    _load_watchlists,
    _save_watchlists,
    _validate_name,
    create_watchlist,
    delete_watchlist,
    get_active_watchlist,
    get_watchlist,
    list_watchlists,
    set_active_watchlist,
    set_watchlist_path,
    update_watchlist,
    validate_tickers,
)


@pytest.fixture(autouse=True)
def isolated_watchlist_file(tmp_path):
    """Use a temp file for every test to avoid cross-test pollution."""
    path = tmp_path / "watchlists.json"
    set_watchlist_path(path)
    yield path
    set_watchlist_path(None)


# ── Create ────────────────────────────────────────────────────────────


class TestCreateWatchlist:
    def test_create_writes_file(self, isolated_watchlist_file):
        create_watchlist("my-list", ["AAPL", "MSFT"])
        assert isolated_watchlist_file.exists()
        data = json.loads(isolated_watchlist_file.read_text())
        assert "my-list" in data["watchlists"]

    def test_create_normalizes_tickers(self):
        create_watchlist("tech", ["aapl", "  msft ", "GOOG", "aapl"])
        tickers = get_watchlist("tech")
        assert tickers == ["AAPL", "GOOG", "MSFT"]

    def test_create_sorts_tickers(self):
        create_watchlist("sorted", ["TSLA", "AAPL", "MSFT"])
        assert get_watchlist("sorted") == ["AAPL", "MSFT", "TSLA"]

    def test_create_deduplicates_tickers(self):
        create_watchlist("dedup", ["AAPL", "aapl", "AAPL"])
        assert get_watchlist("dedup") == ["AAPL"]

    def test_create_strips_empty_tickers(self):
        create_watchlist("clean", ["AAPL", "", "  ", "MSFT"])
        assert get_watchlist("clean") == ["AAPL", "MSFT"]

    def test_create_duplicate_name_raises(self):
        create_watchlist("exists", ["AAPL"])
        with pytest.raises(ValueError, match="already exists"):
            create_watchlist("exists", ["MSFT"])

    def test_create_max_watchlists_raises(self):
        for i in range(MAX_WATCHLISTS):
            create_watchlist(f"list-{i:02d}", ["AAPL"])
        with pytest.raises(ValueError, match=f"Maximum {MAX_WATCHLISTS}"):
            create_watchlist("one-too-many", ["AAPL"])

    def test_create_max_tickers_raises(self):
        too_many = [f"SYM{i}" for i in range(MAX_TICKERS_PER_WATCHLIST + 1)]
        with pytest.raises(ValueError, match=f"Maximum {MAX_TICKERS_PER_WATCHLIST}"):
            create_watchlist("big", too_many)


# ── Get ───────────────────────────────────────────────────────────────


class TestGetWatchlist:
    def test_get_returns_tickers(self):
        create_watchlist("tech", ["AAPL", "MSFT"])
        assert get_watchlist("tech") == ["AAPL", "MSFT"]

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_watchlist("nonexistent")


# ── Update ────────────────────────────────────────────────────────────


class TestUpdateWatchlist:
    def test_update_replaces_tickers(self):
        create_watchlist("tech", ["AAPL"])
        update_watchlist("tech", ["GOOG", "MSFT"])
        assert get_watchlist("tech") == ["GOOG", "MSFT"]

    def test_update_normalizes_tickers(self):
        create_watchlist("tech", ["AAPL"])
        update_watchlist("tech", ["goog", "  msft ", "goog"])
        assert get_watchlist("tech") == ["GOOG", "MSFT"]

    def test_update_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            update_watchlist("nonexistent", ["AAPL"])

    def test_update_max_tickers_raises(self):
        create_watchlist("tech", ["AAPL"])
        too_many = [f"SYM{i}" for i in range(MAX_TICKERS_PER_WATCHLIST + 1)]
        with pytest.raises(ValueError, match=f"Maximum {MAX_TICKERS_PER_WATCHLIST}"):
            update_watchlist("tech", too_many)


# ── Delete ────────────────────────────────────────────────────────────


class TestDeleteWatchlist:
    def test_delete_removes_watchlist(self):
        create_watchlist("tech", ["AAPL"])
        delete_watchlist("tech")
        assert "tech" not in list_watchlists()

    def test_delete_clears_active_if_was_active(self):
        create_watchlist("tech", ["AAPL"])
        set_active_watchlist("tech")
        delete_watchlist("tech")
        assert get_active_watchlist() == []

    def test_delete_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            delete_watchlist("nonexistent")

    def test_delete_preserves_other_watchlists(self):
        create_watchlist("tech", ["AAPL"])
        create_watchlist("energy", ["XOM"])
        delete_watchlist("tech")
        assert get_watchlist("energy") == ["XOM"]


# ── List ──────────────────────────────────────────────────────────────


class TestListWatchlists:
    def test_list_empty(self):
        assert list_watchlists() == {}

    def test_list_returns_all(self):
        create_watchlist("tech", ["AAPL"])
        create_watchlist("energy", ["XOM"])
        result = list_watchlists()
        assert "tech" in result
        assert "energy" in result
        assert result["tech"] == ["AAPL"]
        assert result["energy"] == ["XOM"]


# ── Active watchlist ──────────────────────────────────────────────────


class TestActiveWatchlist:
    def test_get_active_returns_empty_when_none(self):
        assert get_active_watchlist() == []

    def test_set_and_get_active(self):
        create_watchlist("tech", ["AAPL", "MSFT"])
        set_active_watchlist("tech")
        assert get_active_watchlist() == ["AAPL", "MSFT"]

    def test_set_active_none_deactivates(self):
        create_watchlist("tech", ["AAPL"])
        set_active_watchlist("tech")
        set_active_watchlist(None)
        assert get_active_watchlist() == []

    def test_set_active_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            set_active_watchlist("nonexistent")

    def test_active_returns_empty_if_watchlist_deleted_externally(
        self, isolated_watchlist_file
    ):
        """If the active watchlist name points to a deleted list, return []."""
        create_watchlist("tech", ["AAPL"])
        set_active_watchlist("tech")
        # Manually corrupt: remove the watchlist but leave active_watchlist set
        data = json.loads(isolated_watchlist_file.read_text())
        del data["watchlists"]["tech"]
        isolated_watchlist_file.write_text(json.dumps(data))
        assert get_active_watchlist() == []


# ── Name validation ──────────────────────────────────────────────────


class TestNameValidation:
    @pytest.mark.parametrize(
        "name",
        [
            "a",
            "tech",
            "my-list",
            "my-long-watchlist-name",
            "list-1",
            "0",
            "1a",
            "a1",
        ],
    )
    def test_valid_names(self, name):
        _validate_name(name)  # Should not raise

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "-starts-with-dash",
            "ends-with-dash-",
            "UPPERCASE",
            "has spaces",
            "has_underscore",
            "has.dot",
            "a" * 51,
        ],
    )
    def test_invalid_names(self, name):
        with pytest.raises(ValueError):
            _validate_name(name)

    def test_invalid_name_on_create(self):
        with pytest.raises(ValueError):
            create_watchlist("INVALID", ["AAPL"])


# ── File persistence ─────────────────────────────────────────────────


class TestPersistence:
    def test_data_persists_across_loads(self, isolated_watchlist_file):
        create_watchlist("tech", ["AAPL", "MSFT"])
        set_active_watchlist("tech")

        # Directly reload from file (simulating app restart)
        data = json.loads(isolated_watchlist_file.read_text())
        assert data["watchlists"]["tech"] == ["AAPL", "MSFT"]
        assert data["active_watchlist"] == "tech"

    def test_corrupt_file_returns_defaults(self, isolated_watchlist_file):
        isolated_watchlist_file.write_text("not valid json {{{")
        result = _load_watchlists()
        assert result == {"watchlists": {}, "active_watchlist": None}

    def test_missing_file_returns_defaults(self):
        result = _load_watchlists()
        assert result == {"watchlists": {}, "active_watchlist": None}

    def test_save_and_load_roundtrip(self, isolated_watchlist_file):
        original = {
            "watchlists": {"tech": ["AAPL", "GOOG"]},
            "active_watchlist": "tech",
        }
        _save_watchlists(original)
        loaded = _load_watchlists()
        assert loaded == original


# ── Validate tickers ──────────────────────────────────────────────────


class TestValidateTickers:
    @patch("yfinance.Ticker")
    def test_valid_tickers(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 150.0}
        mock_ticker_cls.return_value = mock_ticker

        valid, invalid = validate_tickers(["AAPL", "MSFT"])
        assert valid == ["AAPL", "MSFT"]
        assert invalid == []

    @patch("yfinance.Ticker")
    def test_invalid_tickers(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        valid, invalid = validate_tickers(["FAKESYM"])
        assert valid == []
        assert invalid == ["FAKESYM"]

    @patch("yfinance.Ticker")
    def test_mixed_valid_invalid(self, mock_ticker_cls):
        def make_ticker(sym):
            t = MagicMock()
            if sym == "AAPL":
                t.info = {"regularMarketPrice": 150.0}
            else:
                t.info = {}
            return t

        mock_ticker_cls.side_effect = make_ticker

        valid, invalid = validate_tickers(["AAPL", "FAKE"])
        assert valid == ["AAPL"]
        assert invalid == ["FAKE"]

    @patch("yfinance.Ticker")
    def test_exception_marks_invalid(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("network error")

        valid, invalid = validate_tickers(["AAPL"])
        assert valid == []
        assert invalid == ["AAPL"]

    @patch("yfinance.Ticker")
    def test_normalizes_input(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 150.0}
        mock_ticker_cls.return_value = mock_ticker

        valid, invalid = validate_tickers(["  aapl  "])
        assert valid == ["AAPL"]
