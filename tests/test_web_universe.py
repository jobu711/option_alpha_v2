"""Tests for universe and watchlist web API routes.

Tests cover:
- GET /api/universe/stats returns correct structure
- POST /api/scan/config updates presets and sectors
- GET /api/watchlists returns watchlist listing
- POST /api/watchlists creates and updates watchlists
- DELETE /api/watchlists/{name} removes watchlists
- POST /api/watchlists/{name}/activate sets active watchlist
- POST /api/watchlists/deactivate clears active watchlist
- POST /api/universe/refresh triggers refresh
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from option_alpha.config import Settings
from option_alpha.persistence.database import initialize_db
from option_alpha.web.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    """Settings with temp db path and config path."""
    s = Settings(db_path=tmp_path / "test.db")
    # Set config file path so save() writes to temp dir
    import option_alpha.config as config_mod
    original = config_mod.CONFIG_FILE
    config_mod.CONFIG_FILE = tmp_path / "config.json"
    yield s
    config_mod.CONFIG_FILE = original


@pytest.fixture
def app(settings):
    """Create FastAPI app with test settings."""
    return create_app(config=settings)


@pytest.fixture
def client(app, settings):
    """Starlette test client for the app."""
    # Ensure DB exists.
    conn = initialize_db(settings.db_path)
    conn.close()
    return TestClient(app)


@pytest.fixture
def watchlist_path(tmp_path):
    """Set watchlist file to temp directory."""
    from option_alpha.data import watchlists as wl_mod
    original = wl_mod._WATCHLIST_FILE
    path = tmp_path / "watchlists.json"
    wl_mod._WATCHLIST_FILE = path
    yield path
    wl_mod._WATCHLIST_FILE = original


# ---------------------------------------------------------------------------
# Universe Stats Tests
# ---------------------------------------------------------------------------

class TestUniverseStats:
    """Tests for GET /api/universe/stats."""

    @patch("option_alpha.data.universe.load_universe_data")
    @patch("option_alpha.data.universe.get_scan_universe")
    def test_stats_returns_correct_structure(self, mock_scan, mock_load, client):
        """Stats endpoint returns expected JSON keys."""
        mock_load.return_value = [
            {"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "GOOG"},
        ]
        mock_scan.return_value = ["AAPL", "MSFT"]

        resp = client.get("/api/universe/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tickers"] == 3
        assert data["filtered_tickers"] == 2
        assert isinstance(data["presets"], list)
        assert isinstance(data["sectors"], list)
        assert isinstance(data["available_sectors"], list)

    @patch("option_alpha.data.universe.load_universe_data")
    @patch("option_alpha.data.universe.get_scan_universe")
    def test_stats_default_presets(self, mock_scan, mock_load, client):
        """Stats endpoint returns default presets from settings."""
        mock_load.return_value = []
        mock_scan.return_value = []

        resp = client.get("/api/universe/stats")
        data = resp.json()
        assert "full" in data["presets"]

    @patch("option_alpha.data.universe.load_universe_data")
    @patch("option_alpha.data.universe.get_scan_universe")
    def test_stats_includes_gics_sectors(self, mock_scan, mock_load, client):
        """Stats endpoint lists all 11 GICS sectors."""
        mock_load.return_value = []
        mock_scan.return_value = []

        resp = client.get("/api/universe/stats")
        data = resp.json()
        assert len(data["available_sectors"]) == 11
        assert "Technology" in data["available_sectors"]
        assert "Healthcare" in data["available_sectors"]


# ---------------------------------------------------------------------------
# Scan Config Tests
# ---------------------------------------------------------------------------

class TestScanConfig:
    """Tests for POST /api/scan/config."""

    @patch("option_alpha.data.universe.get_scan_universe")
    def test_update_presets(self, mock_scan, client, settings):
        """Updating presets persists to settings."""
        mock_scan.return_value = ["AAPL", "MSFT"]

        resp = client.post(
            "/api/scan/config",
            json={"presets": ["sp500", "etfs"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["presets"] == ["sp500", "etfs"]
        assert data["filtered_tickers"] == 2
        assert settings.universe_presets == ["sp500", "etfs"]

    @patch("option_alpha.data.universe.get_scan_universe")
    def test_update_sectors(self, mock_scan, client, settings):
        """Updating sectors persists to settings."""
        mock_scan.return_value = ["AAPL"]

        resp = client.post(
            "/api/scan/config",
            json={"sectors": ["Technology", "Healthcare"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sectors"] == ["Technology", "Healthcare"]
        assert settings.universe_sectors == ["Technology", "Healthcare"]

    @patch("option_alpha.data.universe.get_scan_universe")
    def test_update_both(self, mock_scan, client, settings):
        """Updating both presets and sectors works."""
        mock_scan.return_value = []

        resp = client.post(
            "/api/scan/config",
            json={"presets": ["midcap"], "sectors": ["Energy"]},
        )
        assert resp.status_code == 200
        assert settings.universe_presets == ["midcap"]
        assert settings.universe_sectors == ["Energy"]

    @patch("option_alpha.data.universe.get_scan_universe")
    def test_partial_update_preserves_other(self, mock_scan, client, settings):
        """Updating only presets preserves existing sectors."""
        settings.universe_sectors = ["Financials"]
        mock_scan.return_value = []

        resp = client.post(
            "/api/scan/config",
            json={"presets": ["sp500"]},
        )
        assert resp.status_code == 200
        assert settings.universe_presets == ["sp500"]
        assert settings.universe_sectors == ["Financials"]


# ---------------------------------------------------------------------------
# Watchlist Routes Tests
# ---------------------------------------------------------------------------

class TestWatchlistRoutes:
    """Tests for watchlist CRUD API routes."""

    def test_list_watchlists_empty(self, client, watchlist_path):
        """GET /api/watchlists returns empty when no watchlists exist."""
        resp = client.get("/api/watchlists")
        assert resp.status_code == 200
        data = resp.json()
        assert data["watchlists"] == {}
        assert data["active_watchlist"] is None

    def test_create_watchlist(self, client, watchlist_path):
        """POST /api/watchlists creates a new watchlist."""
        resp = client.post(
            "/api/watchlists",
            json={"name": "tech-picks", "tickers": ["AAPL", "MSFT", "GOOG"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["name"] == "tech-picks"

        # Verify it appears in listing.
        resp2 = client.get("/api/watchlists")
        data2 = resp2.json()
        assert "tech-picks" in data2["watchlists"]
        assert "AAPL" in data2["watchlists"]["tech-picks"]

    def test_update_watchlist(self, client, watchlist_path):
        """POST /api/watchlists updates an existing watchlist."""
        # Create first.
        client.post(
            "/api/watchlists",
            json={"name": "my-list", "tickers": ["AAPL"]},
        )
        # Update.
        resp = client.post(
            "/api/watchlists",
            json={"name": "my-list", "tickers": ["MSFT", "GOOG"]},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Verify updated.
        data = client.get("/api/watchlists").json()
        assert "MSFT" in data["watchlists"]["my-list"]
        assert "AAPL" not in data["watchlists"]["my-list"]

    def test_create_watchlist_invalid_name(self, client, watchlist_path):
        """POST /api/watchlists rejects invalid names."""
        resp = client.post(
            "/api/watchlists",
            json={"name": "Invalid Name!", "tickers": ["AAPL"]},
        )
        assert resp.status_code == 400
        assert resp.json()["success"] is False

    def test_delete_watchlist(self, client, watchlist_path):
        """DELETE /api/watchlists/{name} removes a watchlist."""
        client.post(
            "/api/watchlists",
            json={"name": "to-delete", "tickers": ["AAPL"]},
        )

        resp = client.delete("/api/watchlists/to-delete")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Verify gone.
        data = client.get("/api/watchlists").json()
        assert "to-delete" not in data["watchlists"]

    def test_delete_watchlist_not_found(self, client, watchlist_path):
        """DELETE /api/watchlists/{name} returns 404 for missing watchlist."""
        resp = client.delete("/api/watchlists/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["success"] is False

    def test_activate_watchlist(self, client, watchlist_path):
        """POST /api/watchlists/{name}/activate sets active watchlist."""
        client.post(
            "/api/watchlists",
            json={"name": "active-test", "tickers": ["TSLA"]},
        )

        resp = client.post("/api/watchlists/active-test/activate")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Verify active.
        data = client.get("/api/watchlists").json()
        assert data["active_watchlist"] == "active-test"

    def test_activate_nonexistent_watchlist(self, client, watchlist_path):
        """POST /api/watchlists/{name}/activate returns 404 for missing watchlist."""
        resp = client.post("/api/watchlists/missing/activate")
        assert resp.status_code == 404

    def test_deactivate_watchlist(self, client, watchlist_path):
        """POST /api/watchlists/deactivate clears active watchlist."""
        client.post(
            "/api/watchlists",
            json={"name": "temp", "tickers": ["SPY"]},
        )
        client.post("/api/watchlists/temp/activate")

        resp = client.post("/api/watchlists/deactivate")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        data = client.get("/api/watchlists").json()
        assert data["active_watchlist"] is None

    def test_watchlist_tickers_normalized(self, client, watchlist_path):
        """Watchlist tickers are uppercased and deduplicated."""
        resp = client.post(
            "/api/watchlists",
            json={"name": "normalized", "tickers": ["aapl", "msft", "AAPL"]},
        )
        assert resp.status_code == 200

        data = client.get("/api/watchlists").json()
        tickers = data["watchlists"]["normalized"]
        assert tickers == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Universe Refresh Tests
# ---------------------------------------------------------------------------

class TestUniverseRefresh:
    """Tests for POST /api/universe/refresh."""

    @patch("option_alpha.data.universe_refresh.refresh_universe")
    def test_refresh_success(self, mock_refresh, client):
        """Refresh endpoint returns success result."""
        mock_refresh.return_value = {
            "success": True,
            "ticker_count": 2500,
            "added": 10,
            "removed": 5,
            "last_refresh": "2026-02-11T12:00:00+00:00",
        }

        resp = client.post("/api/universe/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["ticker_count"] == 2500

    @patch("option_alpha.data.universe_refresh.refresh_universe")
    def test_refresh_failure(self, mock_refresh, client):
        """Refresh endpoint returns failure result."""
        mock_refresh.return_value = {
            "success": False,
            "error": "Connection timeout",
            "ticker_count": 0,
            "added": 0,
            "removed": 0,
        }

        resp = client.post("/api/universe/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "timeout" in data["error"].lower()


# ---------------------------------------------------------------------------
# Dashboard Template Tests
# ---------------------------------------------------------------------------

class TestDashboardUniverse:
    """Tests that dashboard template includes universe controls."""

    @patch("option_alpha.web.routes._get_market_regime")
    def test_dashboard_has_universe_section(self, mock_market, client, settings):
        """Dashboard includes universe controls section."""
        mock_market.return_value = {"vix": None, "spy_trend": None, "spy_price": None}
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/")
        assert resp.status_code == 200
        assert "universe-controls" in resp.text
        assert "universeControls" in resp.text

    @patch("option_alpha.web.routes._get_market_regime")
    def test_dashboard_has_preset_chips(self, mock_market, client, settings):
        """Dashboard includes preset chip labels."""
        mock_market.return_value = {"vix": None, "spy_trend": None, "spy_price": None}
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/")
        assert resp.status_code == 200
        assert "S&amp;P 500" in resp.text or "S&P 500" in resp.text
        assert "Mid-Cap" in resp.text
        assert "Small-Cap" in resp.text
        assert "ETFs" in resp.text


# ---------------------------------------------------------------------------
# Settings Template Tests
# ---------------------------------------------------------------------------

class TestSettingsWatchlist:
    """Tests that settings template includes watchlist management."""

    def test_settings_has_watchlist_section(self, client):
        """Settings page includes watchlist management section."""
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "Watchlists" in resp.text
        assert "watchlistManager" in resp.text

    def test_settings_has_universe_refresh(self, client):
        """Settings page includes universe refresh button."""
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "Universe Data" in resp.text
        assert "Refresh Universe" in resp.text
        assert "refreshUniverse" in resp.text
