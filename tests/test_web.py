"""Tests for the Option Alpha web dashboard.

Tests cover:
- App factory creates valid FastAPI app
- Route handlers return correct status codes
- Dashboard loads with no scan data (empty state)
- Candidates table sorting
- Ticker detail with mock data
- Scan trigger endpoint
- WebSocket module functions
- History JSON endpoint
- Settings page (load, save, reset, validation, API key masking)
- HTML report export (self-contained, content-disposition)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from option_alpha.config import DEFAULT_SCORING_WEIGHTS, Settings
from option_alpha.models import (
    Direction,
    ScanRun,
    ScanStatus,
    ScoreBreakdown,
    TickerScore,
)
from option_alpha.persistence.database import initialize_db
from option_alpha.persistence.repository import save_scan_run, save_ticker_scores
from option_alpha.web.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    """Settings with temp db path."""
    return Settings(db_path=tmp_path / "test.db")


@pytest.fixture
def app(settings):
    """Create FastAPI app with test settings."""
    return create_app(config=settings)


@pytest.fixture
def client(app):
    """Starlette test client for the app."""
    return TestClient(app)


@pytest.fixture
def db_conn(settings):
    """Provide a migrated database connection."""
    conn = initialize_db(settings.db_path)
    yield conn
    conn.close()


def _make_scan_run(
    run_id: str = "test-run-001",
    status: ScanStatus = ScanStatus.COMPLETED,
) -> ScanRun:
    """Create a test scan run."""
    return ScanRun(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        ticker_count=5,
        duration_seconds=10.0,
        status=status,
        scores_computed=5,
        debates_completed=2,
        options_analyzed=3,
    )


def _make_ticker_scores(count: int = 5) -> list[TickerScore]:
    """Create test ticker scores."""
    symbols = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
    directions = [
        Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL,
        Direction.BULLISH, Direction.BEARISH,
    ]
    return [
        TickerScore(
            symbol=symbols[i],
            composite_score=90.0 - i * 10,
            direction=directions[i],
            last_price=150.0 + i * 20,
            avg_volume=1_000_000.0,
            breakdown=[
                ScoreBreakdown(
                    name="bb_width",
                    raw_value=0.05,
                    normalized=75.0,
                    weight=0.20,
                    contribution=15.0,
                ),
                ScoreBreakdown(
                    name="rsi",
                    raw_value=55.0,
                    normalized=60.0,
                    weight=0.10,
                    contribution=6.0,
                ),
            ],
            timestamp=datetime.now(UTC),
        )
        for i in range(min(count, len(symbols)))
    ]


def _seed_db(conn, scores=None):
    """Insert a scan run and optionally scores into the database."""
    scan_run = _make_scan_run()
    scan_db_id = save_scan_run(conn, scan_run)
    if scores:
        save_ticker_scores(conn, scan_db_id, scores)
    return scan_db_id


# ---------------------------------------------------------------------------
# App Factory Tests
# ---------------------------------------------------------------------------

class TestAppFactory:
    """Tests for create_app function."""

    def test_creates_fastapi_app(self, app):
        """App factory returns a FastAPI instance."""
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)

    def test_app_has_settings(self, app):
        """App has settings in state."""
        assert hasattr(app.state, "settings")
        assert isinstance(app.state.settings, Settings)

    def test_app_has_routes(self, app):
        """App includes expected routes."""
        paths = [route.path for route in app.routes]
        assert "/" in paths
        assert "/scan" in paths
        assert "/candidates" in paths

    def test_app_title(self, app):
        """App has correct title."""
        assert app.title == "Option Alpha Dashboard"


# ---------------------------------------------------------------------------
# Dashboard Route Tests
# ---------------------------------------------------------------------------

class TestDashboardRoute:
    """Tests for GET / route."""

    @patch("option_alpha.web.routes._get_market_regime")
    def test_dashboard_empty_state(self, mock_market, client, settings):
        """Dashboard loads with no scan data and shows empty state."""
        mock_market.return_value = {"vix": None, "spy_trend": None, "spy_price": None}
        # Ensure DB is initialized.
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Option Alpha" in resp.text
        assert "No scan data yet" in resp.text

    @patch("option_alpha.web.routes._get_market_regime")
    def test_dashboard_with_scan_data(self, mock_market, client, settings):
        """Dashboard loads and shows scan info when data exists."""
        mock_market.return_value = {"vix": 18.5, "spy_trend": "up", "spy_price": 450.0}
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Last scan:" in resp.text

    @patch("option_alpha.web.routes._get_market_regime")
    def test_dashboard_market_regime_display(self, mock_market, client, settings):
        """Dashboard shows market regime indicators."""
        mock_market.return_value = {"vix": 22.5, "spy_trend": "down", "spy_price": 445.0}
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/")
        assert resp.status_code == 200
        assert "22.50" in resp.text
        assert "$445.00" in resp.text


# ---------------------------------------------------------------------------
# Candidates Table Tests
# ---------------------------------------------------------------------------

class TestCandidatesRoute:
    """Tests for GET /candidates route."""

    def test_candidates_empty(self, client, settings):
        """Candidates table shows empty state when no data."""
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/candidates")
        assert resp.status_code == 200
        assert "No candidates available" in resp.text

    def test_candidates_with_data(self, client, settings):
        """Candidates table shows scored tickers."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(5)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/candidates")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "MSFT" in resp.text

    def test_candidates_sort_by_symbol(self, client, settings):
        """Candidates table sorts by symbol."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/candidates?sort_by=symbol&order=asc")
        assert resp.status_code == 200
        text = resp.text
        # AAPL should appear before MSFT in ascending order.
        assert text.index("AAPL") < text.index("MSFT")

    def test_candidates_sort_by_score_desc(self, client, settings):
        """Candidates table sorts by composite score descending."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/candidates?sort_by=composite_score&order=desc")
        assert resp.status_code == 200
        text = resp.text
        # AAPL (90) should appear before GOOG (70) in desc order.
        assert text.index("AAPL") < text.index("GOOG")

    def test_candidates_sort_by_score_asc(self, client, settings):
        """Candidates table sorts by composite score ascending."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/candidates?sort_by=composite_score&order=asc")
        assert resp.status_code == 200
        text = resp.text
        # GOOG (70) should appear before AAPL (90) in asc order.
        assert text.index("GOOG") < text.index("AAPL")

    def test_candidates_invalid_sort_field(self, client, settings):
        """Invalid sort field defaults to composite_score."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/candidates?sort_by=invalid_field&order=desc")
        assert resp.status_code == 200

    def test_candidates_direction_badges(self, client, settings):
        """Candidates table shows direction badges."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/candidates")
        assert resp.status_code == 200
        assert "BULLISH" in resp.text
        assert "BEARISH" in resp.text


# ---------------------------------------------------------------------------
# Ticker Detail Tests
# ---------------------------------------------------------------------------

class TestTickerDetailRoute:
    """Tests for GET /ticker/{symbol} route."""

    def test_ticker_detail_with_data(self, client, settings):
        """Ticker detail page shows score breakdown."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/ticker/AAPL")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "Score Breakdown" in resp.text
        assert "bb_width" in resp.text.lower() or "Bb Width" in resp.text

    def test_ticker_detail_no_data(self, client, settings):
        """Ticker detail shows empty state for unknown ticker."""
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/ticker/UNKNOWN")
        assert resp.status_code == 200
        assert "No data available" in resp.text

    def test_ticker_detail_with_score_values(self, client, settings):
        """Ticker detail shows score values."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(1)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/ticker/AAPL")
        assert resp.status_code == 200
        assert "90.0" in resp.text  # composite score
        assert "BULLISH" in resp.text


# ---------------------------------------------------------------------------
# Scan Trigger Tests
# ---------------------------------------------------------------------------

class TestScanTriggerRoute:
    """Tests for POST /scan route."""

    @patch("option_alpha.web.routes._run_scan_task")
    def test_scan_trigger_returns_progress(self, mock_scan, client, settings):
        """POST /scan returns progress HTMX partial."""
        conn = initialize_db(settings.db_path)
        conn.close()

        import option_alpha.web.routes as routes_mod
        routes_mod._scan_running = False

        resp = client.post("/scan")
        assert resp.status_code == 200
        assert "Scan started" in resp.text or "scan" in resp.text.lower()

    @patch("option_alpha.web.routes._run_scan_task")
    def test_scan_trigger_shows_phases(self, mock_scan, client, settings):
        """POST /scan progress shows phase names."""
        conn = initialize_db(settings.db_path)
        conn.close()

        import option_alpha.web.routes as routes_mod
        routes_mod._scan_running = False

        resp = client.post("/scan")
        assert resp.status_code == 200
        assert "Data Fetch" in resp.text or "data_fetch" in resp.text.lower()

    @patch("option_alpha.web.routes._run_scan_task")
    def test_scan_already_running(self, mock_scan, client, settings):
        """POST /scan when scan is running returns running message."""
        conn = initialize_db(settings.db_path)
        conn.close()

        import option_alpha.web.routes as routes_mod
        routes_mod._scan_running = True

        resp = client.post("/scan")
        assert resp.status_code == 200
        assert "already running" in resp.text

        # Reset for other tests.
        routes_mod._scan_running = False


# ---------------------------------------------------------------------------
# History JSON Endpoint Tests
# ---------------------------------------------------------------------------

class TestHistoryRoute:
    """Tests for GET /history/{symbol} route."""

    def test_history_returns_json(self, client, settings):
        """History endpoint returns JSON with symbol and history array."""
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/history/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert isinstance(data["history"], list)

    def test_history_with_data(self, client, settings):
        """History endpoint returns score history when data exists."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/history/AAPL?days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert len(data["history"]) >= 1


# ---------------------------------------------------------------------------
# WebSocket Module Tests
# ---------------------------------------------------------------------------

class TestWebSocketModule:
    """Tests for websocket module functions."""

    def test_get_connected_count_initial(self):
        """Initial connected count is 0."""
        from option_alpha.web.websocket import get_connected_count
        # Note: count might not be 0 if other tests connected, but should be int.
        count = get_connected_count()
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.asyncio
    async def test_broadcast_progress_no_clients(self):
        """broadcast_progress works with no connected clients."""
        from option_alpha.pipeline.progress import ScanProgress
        from option_alpha.web.websocket import broadcast_progress

        progress = ScanProgress(
            overall_percentage=50.0,
            current_phase="scoring",
        )
        # Should not raise.
        await broadcast_progress(progress)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests across routes and data."""

    @patch("option_alpha.web.routes._run_scan_task")
    @patch("option_alpha.web.routes._get_market_regime")
    def test_full_flow_empty_db(self, mock_market, mock_scan, client, settings):
        """Full flow: dashboard -> candidates -> trigger scan."""
        mock_market.return_value = {"vix": None, "spy_trend": None, "spy_price": None}
        conn = initialize_db(settings.db_path)
        conn.close()

        # Load dashboard.
        resp = client.get("/")
        assert resp.status_code == 200

        # Load candidates (empty).
        resp = client.get("/candidates")
        assert resp.status_code == 200
        assert "No candidates available" in resp.text

        # Trigger scan.
        import option_alpha.web.routes as routes_mod
        routes_mod._scan_running = False
        resp = client.post("/scan")
        assert resp.status_code == 200

    @patch("option_alpha.web.routes._get_market_regime")
    def test_full_flow_with_data(self, mock_market, client, settings):
        """Full flow: seed data -> dashboard -> candidates -> detail."""
        mock_market.return_value = {"vix": 20.0, "spy_trend": "up", "spy_price": 450.0}
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(5)
        _seed_db(conn, scores)
        conn.close()

        # Dashboard shows scan info.
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Last scan:" in resp.text

        # Candidates table shows tickers.
        resp = client.get("/candidates")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "TSLA" in resp.text

        # Ticker detail shows breakdown.
        resp = client.get("/ticker/AAPL")
        assert resp.status_code == 200
        assert "Score Breakdown" in resp.text

        # History endpoint returns data.
        resp = client.get("/history/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["history"]) >= 1

    def test_static_files_served(self, client):
        """Static CSS and JS files are served."""
        resp = client.get("/static/style.css")
        assert resp.status_code == 200
        assert "bg-primary" in resp.text

        resp = client.get("/static/app.js")
        assert resp.status_code == 200
        assert "WebSocket" in resp.text


# ---------------------------------------------------------------------------
# Settings Page Tests
# ---------------------------------------------------------------------------

class TestSettingsPage:
    """Tests for GET /settings route."""

    def test_settings_page_loads(self, client, settings):
        """Settings page returns 200 and shows form sections."""
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "Scoring Weights" in resp.text
        assert "Filters" in resp.text
        assert "Options Parameters" in resp.text
        assert "AI Configuration" in resp.text
        assert "FRED API" in resp.text

    def test_settings_shows_current_values(self, client, settings):
        """Settings page pre-populates with current config values."""
        resp = client.get("/settings")
        assert resp.status_code == 200
        # Default scoring weight for bb_width is 0.12
        assert "0.12" in resp.text
        # Default min_composite_score is 50.0
        assert "50.0" in resp.text
        # Default DTE range
        assert 'value="30"' in resp.text  # dte_min
        assert 'value="60"' in resp.text  # dte_max

    def test_settings_masks_claude_api_key(self, client, settings):
        """Settings page masks Claude API key, showing only last 4 chars."""
        settings.claude_api_key = "sk-ant-abcdefghij1234"
        resp = client.get("/settings")
        assert resp.status_code == 200
        # Full key should NOT appear in the HTML
        assert "sk-ant-abcdefghij1234" not in resp.text
        # Last 4 chars should appear as placeholder
        assert "1234" in resp.text

    def test_settings_masks_fred_api_key(self, client, settings):
        """Settings page masks FRED API key."""
        settings.fred_api_key = "my-secret-fred-key"
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "my-secret-fred-key" not in resp.text
        assert "-key" in resp.text  # last 4 chars

    def test_settings_no_api_key_shows_not_set(self, client, settings):
        """Settings page shows 'Not set' when no API key configured."""
        settings.claude_api_key = None
        settings.fred_api_key = None
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "Not set" in resp.text

    def test_settings_has_routes_registered(self, app):
        """App includes settings routes."""
        paths = [route.path for route in app.routes]
        assert "/settings" in paths
        assert "/settings/reset" in paths
        assert "/export" in paths


# ---------------------------------------------------------------------------
# Settings Save Tests
# ---------------------------------------------------------------------------

class TestSettingsSave:
    """Tests for POST /settings route."""

    def test_save_settings_updates_config(self, client, settings, tmp_path):
        """POST /settings updates config values and persists to file."""
        form_data = {
            "weight_bb_width": "0.30",
            "weight_atr_percentile": "0.15",
            "weight_rsi": "0.10",
            "weight_obv_trend": "0.10",
            "weight_sma_alignment": "0.10",
            "weight_relative_volume": "0.10",
            "weight_catalyst_proximity": "0.15",
            "min_composite_score": "60.0",
            "min_price": "10.0",
            "min_avg_volume": "1000000",
            "dte_min": "20",
            "dte_max": "45",
            "min_open_interest": "200",
            "max_bid_ask_spread_pct": "0.05",
            "min_option_volume": "10",
            "ai_backend": "claude",
            "ollama_model": "llama3.2:3b",
            "claude_api_key": "",
            "fred_api_key": "",
        }
        resp = client.post("/settings", data=form_data)
        assert resp.status_code == 200
        assert "saved successfully" in resp.text

        # Verify settings were updated in memory.
        assert settings.scoring_weights["bb_width"] == 0.30
        assert settings.min_composite_score == 60.0
        assert settings.dte_min == 20
        assert settings.dte_max == 45
        assert settings.ai_backend == "claude"
        assert settings.ollama_model == "llama3.2:3b"

    def test_save_settings_preserves_existing_api_key(self, client, settings):
        """POST /settings with blank API key preserves existing key."""
        settings.claude_api_key = "sk-existing-key-1234"
        form_data = {
            "weight_bb_width": "0.20",
            "weight_atr_percentile": "0.15",
            "weight_rsi": "0.10",
            "weight_obv_trend": "0.10",
            "weight_sma_alignment": "0.10",
            "weight_relative_volume": "0.10",
            "weight_catalyst_proximity": "0.25",
            "min_composite_score": "50.0",
            "min_price": "5.0",
            "min_avg_volume": "500000",
            "dte_min": "30",
            "dte_max": "60",
            "min_open_interest": "100",
            "max_bid_ask_spread_pct": "0.10",
            "min_option_volume": "1",
            "ai_backend": "ollama",
            "ollama_model": "llama3.1:8b",
            "claude_api_key": "",
            "fred_api_key": "",
        }
        resp = client.post("/settings", data=form_data)
        assert resp.status_code == 200
        assert "saved successfully" in resp.text
        assert settings.claude_api_key == "sk-existing-key-1234"

    def test_save_settings_updates_api_key(self, client, settings):
        """POST /settings with new API key updates the key."""
        form_data = {
            "weight_bb_width": "0.20",
            "weight_atr_percentile": "0.15",
            "weight_rsi": "0.10",
            "weight_obv_trend": "0.10",
            "weight_sma_alignment": "0.10",
            "weight_relative_volume": "0.10",
            "weight_catalyst_proximity": "0.25",
            "min_composite_score": "50.0",
            "min_price": "5.0",
            "min_avg_volume": "500000",
            "dte_min": "30",
            "dte_max": "60",
            "min_open_interest": "100",
            "max_bid_ask_spread_pct": "0.10",
            "min_option_volume": "1",
            "ai_backend": "claude",
            "ollama_model": "llama3.1:8b",
            "claude_api_key": "sk-new-key-5678",
            "fred_api_key": "",
        }
        resp = client.post("/settings", data=form_data)
        assert resp.status_code == 200
        assert settings.claude_api_key == "sk-new-key-5678"

    def test_save_settings_validates_negative_weight(self, client, settings):
        """POST /settings rejects negative scoring weights."""
        form_data = {
            "weight_bb_width": "-0.10",
            "weight_atr_percentile": "0.15",
            "weight_rsi": "0.10",
            "weight_obv_trend": "0.10",
            "weight_sma_alignment": "0.10",
            "weight_relative_volume": "0.10",
            "weight_catalyst_proximity": "0.25",
            "min_composite_score": "50.0",
            "min_price": "5.0",
            "min_avg_volume": "500000",
            "dte_min": "30",
            "dte_max": "60",
            "min_open_interest": "100",
            "max_bid_ask_spread_pct": "0.10",
            "min_option_volume": "1",
            "ai_backend": "ollama",
            "ollama_model": "llama3.1:8b",
            "claude_api_key": "",
            "fred_api_key": "",
        }
        resp = client.post("/settings", data=form_data)
        assert resp.status_code == 200
        assert "error" in resp.text.lower() or "non-negative" in resp.text

    def test_save_settings_validates_dte_range(self, client, settings):
        """POST /settings rejects DTE min > DTE max."""
        form_data = {
            "weight_bb_width": "0.20",
            "weight_atr_percentile": "0.15",
            "weight_rsi": "0.10",
            "weight_obv_trend": "0.10",
            "weight_sma_alignment": "0.10",
            "weight_relative_volume": "0.10",
            "weight_catalyst_proximity": "0.25",
            "min_composite_score": "50.0",
            "min_price": "5.0",
            "min_avg_volume": "500000",
            "dte_min": "90",
            "dte_max": "30",
            "min_open_interest": "100",
            "max_bid_ask_spread_pct": "0.10",
            "min_option_volume": "1",
            "ai_backend": "ollama",
            "ollama_model": "llama3.1:8b",
            "claude_api_key": "",
            "fred_api_key": "",
        }
        resp = client.post("/settings", data=form_data)
        assert resp.status_code == 200
        assert "DTE Min" in resp.text or "error" in resp.text.lower()

    def test_save_settings_validates_invalid_numeric(self, client, settings):
        """POST /settings rejects non-numeric input for numeric fields."""
        form_data = {
            "weight_bb_width": "not_a_number",
            "weight_atr_percentile": "0.15",
            "weight_rsi": "0.10",
            "weight_obv_trend": "0.10",
            "weight_sma_alignment": "0.10",
            "weight_relative_volume": "0.10",
            "weight_catalyst_proximity": "0.25",
            "min_composite_score": "50.0",
            "min_price": "5.0",
            "min_avg_volume": "500000",
            "dte_min": "30",
            "dte_max": "60",
            "min_open_interest": "100",
            "max_bid_ask_spread_pct": "0.10",
            "min_option_volume": "1",
            "ai_backend": "ollama",
            "ollama_model": "llama3.1:8b",
            "claude_api_key": "",
            "fred_api_key": "",
        }
        resp = client.post("/settings", data=form_data)
        assert resp.status_code == 200
        assert "error" in resp.text.lower() or "Invalid" in resp.text


# ---------------------------------------------------------------------------
# Settings Reset Tests
# ---------------------------------------------------------------------------

class TestSettingsReset:
    """Tests for POST /settings/reset route."""

    def test_reset_restores_defaults(self, client, settings):
        """POST /settings/reset restores default settings."""
        # Modify settings from defaults first.
        settings.min_composite_score = 99.0
        settings.dte_min = 5

        resp = client.post("/settings/reset")
        assert resp.status_code == 200

        # App state should now have defaults.
        new_settings = client.app.state.settings
        assert new_settings.min_composite_score == 50.0
        assert new_settings.dte_min == 30

    def test_reset_returns_redirect_header(self, client, settings):
        """POST /settings/reset returns HX-Redirect header to /settings."""
        resp = client.post("/settings/reset")
        assert resp.status_code == 200
        assert resp.headers.get("HX-Redirect") == "/settings"


# ---------------------------------------------------------------------------
# API Key Masking Unit Tests
# ---------------------------------------------------------------------------

class TestApiKeyMasking:
    """Tests for the _mask_key helper function."""

    def test_mask_key_none(self):
        """None key returns 'Not set'."""
        from option_alpha.web.routes import _mask_key
        assert _mask_key(None) == "Not set"

    def test_mask_key_empty(self):
        """Empty key returns 'Not set'."""
        from option_alpha.web.routes import _mask_key
        assert _mask_key("") == "Not set"

    def test_mask_key_short(self):
        """Key with 4 or fewer chars returns all asterisks."""
        from option_alpha.web.routes import _mask_key
        assert _mask_key("1234") == "****"
        assert _mask_key("abc") == "****"

    def test_mask_key_normal(self):
        """Normal key shows only last 4 chars."""
        from option_alpha.web.routes import _mask_key
        result = _mask_key("sk-ant-abcdefghij1234")
        assert result.endswith("1234")
        assert "sk-ant" not in result
        assert "*" in result

    def test_mask_key_preserves_length(self):
        """Masked key has same length as original."""
        from option_alpha.web.routes import _mask_key
        key = "sk-ant-abcdefghij1234"
        result = _mask_key(key)
        assert len(result) == len(key)


# ---------------------------------------------------------------------------
# Export Report Tests
# ---------------------------------------------------------------------------

class TestExportReport:
    """Tests for GET /export route."""

    def test_export_empty_db(self, client, settings):
        """Export report works with empty database."""
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/export")
        assert resp.status_code == 200
        assert "Option Alpha Scan Report" in resp.text

    def test_export_with_data(self, client, settings):
        """Export report includes scored candidates."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(5)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/export")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "MSFT" in resp.text

    def test_export_content_disposition(self, client, settings):
        """Export report has correct Content-Disposition header for download."""
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/export")
        assert resp.status_code == 200
        content_disp = resp.headers.get("content-disposition", "")
        assert "attachment" in content_disp
        assert "report_" in content_disp
        assert ".html" in content_disp

    def test_export_self_contained(self, client, settings):
        """Export report is self-contained with no external URLs for CSS/JS."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/export")
        assert resp.status_code == 200
        html = resp.text
        # Should have inline styles.
        assert "<style>" in html
        # Should NOT reference external stylesheets or scripts.
        assert 'href="/static' not in html
        assert 'src="/static' not in html
        assert "unpkg.com" not in html

    def test_export_valid_html(self, client, settings):
        """Export report produces valid HTML structure."""
        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/export")
        assert resp.status_code == 200
        html = resp.text
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</body>" in html

    def test_export_includes_score_breakdown(self, client, settings):
        """Export report includes score values in the table."""
        conn = initialize_db(settings.db_path)
        scores = _make_ticker_scores(3)
        _seed_db(conn, scores)
        conn.close()

        resp = client.get("/export")
        assert resp.status_code == 200
        # Check AAPL's composite score of 90.0
        assert "90.0" in resp.text
        assert "BULLISH" in resp.text
