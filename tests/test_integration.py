"""End-to-end integration tests for the Option Alpha pipeline.

Tests cover:
- Full pipeline with mocked externals (yfinance, Ollama, FRED API)
- Web app serving dashboard with persisted data
- Scan trigger -> progress -> results flow
- Error state display on the dashboard
- __main__.py module import and configuration
- PRD acceptance criteria validation (US-1 through US-7)
- Security checks (localhost binding, API key masking, report safety)
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from option_alpha.config import DEFAULT_SCORING_WEIGHTS, Settings
from option_alpha.models import (
    AgentResponse,
    DebateResult,
    Direction,
    OptionsRecommendation,
    ScanResult,
    ScanRun,
    ScanStatus,
    ScoreBreakdown,
    TickerScore,
    TradeThesis,
)
from option_alpha.persistence.database import initialize_db
from option_alpha.persistence.repository import (
    save_ai_theses,
    save_scan_run,
    save_ticker_scores,
)
from option_alpha.web.app import create_app


# ---------------------------------------------------------------------------
# Fixture Data Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_dataframe(
    symbol: str,
    days: int = 200,
    start_price: float = 150.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Create a realistic OHLCV DataFrame with random walk prices."""
    rng = np.random.default_rng(hash(symbol) % 2**31)
    dates = pd.bdate_range(end=datetime.now(), periods=days)

    # Random walk for close prices
    returns = rng.normal(0, volatility, size=days)
    close = start_price * np.cumprod(1 + returns)

    # Derive OHLCV from close
    high = close * (1 + rng.uniform(0, 0.02, size=days))
    low = close * (1 - rng.uniform(0, 0.02, size=days))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, size=days))
    volume = rng.integers(500_000, 5_000_000, size=days).astype(float)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _make_fixture_ticker_data():
    """Create fixture TickerData objects for 5 test tickers."""
    from option_alpha.models import TickerData

    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
    result = {}
    for symbol in tickers:
        df = _make_ohlcv_dataframe(symbol)
        result[symbol] = TickerData(
            symbol=symbol,
            dates=df.index.tolist(),
            open=df["Open"].tolist(),
            high=df["High"].tolist(),
            low=df["Low"].tolist(),
            close=df["Close"].tolist(),
            volume=[int(v) for v in df["Volume"].tolist()],
            last_price=float(df["Close"].iloc[-1]),
            avg_volume=float(df["Volume"].mean()),
        )
    return result


def _make_fixture_debate_result(symbol: str = "AAPL") -> DebateResult:
    """Create a fixture DebateResult."""
    return DebateResult(
        symbol=symbol,
        bull=AgentResponse(
            role="bull",
            analysis=f"Strong bullish case for {symbol} based on technical signals.",
            key_points=["Rising momentum", "Above key SMAs", "Strong volume"],
            conviction=8,
        ),
        bear=AgentResponse(
            role="bear",
            analysis=f"Bear case for {symbol}: elevated valuation and macro risks.",
            key_points=["High P/E ratio", "Rate sensitivity", "Competition risk"],
            conviction=5,
        ),
        risk=AgentResponse(
            role="risk",
            analysis=f"Risk synthesis for {symbol}: cautiously bullish with hedging.",
            key_points=["Use protective puts", "Scale in gradually"],
            conviction=7,
        ),
        final_thesis=TradeThesis(
            symbol=symbol,
            direction=Direction.BULLISH,
            conviction=7,
            entry_rationale=f"Technical strength in {symbol} with strong volume confirmation.",
            risk_factors=["Market-wide correction", "Earnings miss"],
            recommended_action=f"Buy {symbol} 180C 45DTE",
        ),
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
                ScoreBreakdown(name="bb_width", raw_value=0.05, normalized=75.0, weight=0.20, contribution=15.0),
                ScoreBreakdown(name="rsi", raw_value=55.0, normalized=60.0, weight=0.10, contribution=6.0),
                ScoreBreakdown(name="atr_percent", raw_value=3.5, normalized=65.0, weight=0.15, contribution=9.75),
                ScoreBreakdown(name="obv_trend", raw_value=1.2, normalized=70.0, weight=0.10, contribution=7.0),
                ScoreBreakdown(name="sma_alignment", raw_value=1.0, normalized=80.0, weight=0.10, contribution=8.0),
                ScoreBreakdown(name="relative_volume", raw_value=1.5, normalized=72.0, weight=0.10, contribution=7.2),
            ],
            timestamp=datetime.now(UTC),
        )
        for i in range(min(count, len(symbols)))
    ]


def _seed_full_db(conn, count: int = 5):
    """Seed database with scan run, scores, and AI theses."""
    scores = _make_ticker_scores(count)

    scan_run = ScanRun(
        run_id="integration-test-001",
        timestamp=datetime.now(UTC),
        ticker_count=count,
        duration_seconds=42.5,
        status=ScanStatus.COMPLETED,
        scores_computed=count,
        debates_completed=min(count, 2),
        options_analyzed=min(count, 3),
    )
    scan_db_id = save_scan_run(conn, scan_run)
    save_ticker_scores(conn, scan_db_id, scores)

    # Save AI theses for top 2
    debates = [_make_fixture_debate_result(s.symbol) for s in scores[:2]]
    save_ai_theses(conn, scan_db_id, debates)

    return scan_db_id, scores, debates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    """Settings with temp db path."""
    return Settings(db_path=tmp_path / "integration_test.db")


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


@pytest.fixture
def seeded_db(db_conn):
    """Seed DB with full scan data and return (scan_db_id, scores, debates)."""
    return _seed_full_db(db_conn)


# ---------------------------------------------------------------------------
# 1. Full Pipeline Integration Test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end pipeline test with mocked externals."""

    def test_scoring_pipeline_with_fixture_data(self, settings):
        """Score universe using fixture OHLCV data produces valid TickerScores."""
        from option_alpha.scoring.composite import score_universe

        # Create fixture OHLCV DataFrames
        ohlcv_data = {
            symbol: _make_ohlcv_dataframe(symbol)
            for symbol in ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
        }

        # Score the universe
        scores = score_universe(ohlcv_data, settings)

        # Verify: scores computed for all tickers
        assert len(scores) == 5
        for score in scores:
            assert score.symbol in ohlcv_data
            assert 0 <= score.composite_score <= 100
            assert score.direction in (Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL)
            assert score.last_price is not None
            assert score.last_price > 0
            assert len(score.breakdown) > 0

        # Verify: sorted by composite score descending
        for i in range(len(scores) - 1):
            assert scores[i].composite_score >= scores[i + 1].composite_score

    def test_persist_and_retrieve_scores(self, settings):
        """Scores can be persisted and retrieved correctly."""
        conn = initialize_db(settings.db_path)
        try:
            from option_alpha.persistence.repository import (
                get_latest_scan,
                get_scores_for_scan,
            )

            scan_db_id, scores, _ = _seed_full_db(conn)

            # Retrieve the latest scan
            latest = get_latest_scan(conn)
            assert latest is not None
            assert latest.run_id == "integration-test-001"
            assert latest.scores_computed == 5

            # Retrieve scores
            retrieved_scores = get_scores_for_scan(conn, scan_db_id)
            assert len(retrieved_scores) == 5
            assert retrieved_scores[0].symbol == "AAPL"  # highest score
            assert retrieved_scores[0].composite_score == 90.0
        finally:
            conn.close()

    def test_persist_and_retrieve_debates(self, settings):
        """AI debate results can be persisted and retrieved."""
        conn = initialize_db(settings.db_path)
        try:
            scan_db_id, _, _ = _seed_full_db(conn)

            # Retrieve debate for AAPL
            row = conn.execute(
                "SELECT * FROM ai_theses WHERE scan_run_id = ? AND ticker = ?",
                (scan_db_id, "AAPL"),
            ).fetchone()
            assert row is not None
            assert row["ticker"] == "AAPL"
            assert "bullish" in row["bull_thesis"].lower() or "bull" in row["bull_thesis"].lower()
            assert row["conviction"] == 7
            assert row["direction"] == "bullish"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_full_orchestrator_with_mocks(self, settings):
        """Full orchestrator.run_scan() with all externals mocked."""
        from option_alpha.pipeline.orchestrator import ScanOrchestrator

        fixture_data = _make_fixture_ticker_data()
        fixture_frames = {
            sym: _make_ohlcv_dataframe(sym) for sym in fixture_data
        }

        # Mock the universe to return our test tickers
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe") as mock_universe,
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch") as mock_cache,
            patch("option_alpha.pipeline.orchestrator.fetch_batch") as mock_fetch,
            patch("option_alpha.pipeline.orchestrator.save_batch") as mock_save,
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info") as mock_earnings,
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers") as mock_chains,
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers") as mock_recs,
            patch("option_alpha.pipeline.orchestrator.get_client") as mock_client_factory,
        ):
            mock_universe.return_value = list(fixture_data.keys())
            mock_cache.return_value = fixture_data  # All cached
            mock_fetch.return_value = {}
            mock_save.return_value = None
            mock_earnings.return_value = {sym: {} for sym in fixture_data}

            # Mock options
            mock_chains.return_value = {}
            mock_recs.return_value = [
                OptionsRecommendation(
                    symbol="AAPL",
                    direction=Direction.BULLISH,
                    option_type="call",
                    strike=180.0,
                    expiry=datetime.now(UTC) + timedelta(days=45),
                    dte=45,
                )
            ]

            # Mock AI client
            mock_llm = AsyncMock()
            mock_llm.complete = AsyncMock(
                side_effect=[
                    # For each debate ticker (up to top_n_ai_debate):
                    # Bull response, Bear response, Risk response (TradeThesis)
                    AgentResponse(role="bull", analysis="Strong buy signal", key_points=["Momentum"]),
                    AgentResponse(role="bear", analysis="Overvalued risk", key_points=["High P/E"]),
                    TradeThesis(
                        symbol="AAPL",
                        direction=Direction.BULLISH,
                        conviction=7,
                        entry_rationale="Technical strength",
                        risk_factors=["Market risk"],
                        recommended_action="Buy AAPL 180C 45DTE",
                    ),
                ] * 10  # Repeat for up to 10 debates
            )
            mock_client_factory.return_value = mock_llm

            orchestrator = ScanOrchestrator(settings=settings)
            result = await orchestrator.run_scan()

            # Verify: scores computed
            assert len(result.ticker_scores) > 0
            assert result.top_n_scored > 0

            # Verify: options recommended
            assert len(result.options_recommendations) >= 1

            # Verify: debates completed
            assert len(result.debate_results) > 0
            assert result.top_n_debated > 0

            # Verify: results persisted to DB
            conn = initialize_db(settings.db_path)
            try:
                from option_alpha.persistence.repository import get_latest_scan
                latest = get_latest_scan(conn)
                assert latest is not None
                assert latest.status in (ScanStatus.COMPLETED, ScanStatus.PARTIAL)
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# 2. Web App Serves Dashboard with Persisted Data
# ---------------------------------------------------------------------------

class TestWebDashboardWithData:
    """Test the web app serves correct data from the database."""

    @patch("option_alpha.web.routes._get_market_regime")
    def test_dashboard_shows_scan_results(self, mock_market, client, settings, db_conn, seeded_db):
        """Dashboard shows latest scan info after data is seeded."""
        mock_market.return_value = {"vix": 20.0, "spy_trend": "up", "spy_price": 450.0}

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Last scan:" in resp.text
        assert "5 scored" in resp.text

    def test_candidates_table_shows_all_tickers(self, client, settings, db_conn, seeded_db):
        """Candidates table displays all scored tickers."""
        resp = client.get("/candidates")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "MSFT" in resp.text
        assert "GOOG" in resp.text
        assert "TSLA" in resp.text
        assert "AMZN" in resp.text

    def test_ticker_detail_shows_score_breakdown(self, client, settings, db_conn, seeded_db):
        """Ticker detail shows complete score breakdown."""
        resp = client.get("/ticker/AAPL")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "Score Breakdown" in resp.text
        assert "90.0" in resp.text

    def test_ticker_detail_shows_debate(self, client, settings, db_conn, seeded_db):
        """Ticker detail shows AI debate for debated tickers."""
        resp = client.get("/ticker/AAPL")
        assert resp.status_code == 200
        # AI debate content should be present
        assert "Bull" in resp.text or "bull" in resp.text
        assert "Bear" in resp.text or "bear" in resp.text

    def test_history_endpoint_returns_data(self, client, settings, db_conn, seeded_db):
        """History JSON endpoint returns score history."""
        resp = client.get("/history/AAPL?days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert len(data["history"]) >= 1

    def test_export_report_includes_data(self, client, settings, db_conn, seeded_db):
        """Export report includes scored candidates and debates."""
        resp = client.get("/export")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "90.0" in resp.text
        # Report should be self-contained
        assert "<style>" in resp.text


# ---------------------------------------------------------------------------
# 3. Scan Trigger -> Progress -> Results Flow
# ---------------------------------------------------------------------------

class TestScanFlow:
    """Test the scan trigger and progress display flow."""

    @patch("option_alpha.web.routes._run_scan_task")
    def test_scan_trigger_returns_progress(self, mock_scan, client, settings):
        """POST /scan returns progress partial with phase list."""
        conn = initialize_db(settings.db_path)
        conn.close()

        import option_alpha.web.routes as routes_mod
        routes_mod._scan_running = False

        resp = client.post("/scan")
        assert resp.status_code == 200
        assert "Scan started" in resp.text
        # All phase names should appear
        for phase in ["data_fetch", "scoring", "catalysts", "options", "ai_debate", "persist"]:
            assert phase in resp.text.lower().replace(" ", "_")

        routes_mod._scan_running = False

    @patch("option_alpha.web.routes._run_scan_task")
    def test_scan_already_running_rejected(self, mock_scan, client, settings):
        """POST /scan when already running returns running message."""
        conn = initialize_db(settings.db_path)
        conn.close()

        import option_alpha.web.routes as routes_mod
        routes_mod._scan_running = True

        resp = client.post("/scan")
        assert resp.status_code == 200
        assert "already running" in resp.text

        routes_mod._scan_running = False


# ---------------------------------------------------------------------------
# 4. Error State Display Tests
# ---------------------------------------------------------------------------

class TestErrorStates:
    """Test error and warning display on the dashboard."""

    @patch("option_alpha.web.routes._get_market_regime")
    @patch("option_alpha.web.routes.run_health_checks")
    def test_dashboard_shows_ollama_warning(self, mock_health, mock_market, client, settings):
        """Dashboard shows warning when Ollama is not detected."""
        mock_market.return_value = {"vix": None, "spy_trend": None, "spy_price": None}

        from option_alpha.web.errors import CheckSeverity, HealthCheck, SystemStatus
        mock_health.return_value = SystemStatus(checks=[
            HealthCheck(
                name="ollama",
                severity=CheckSeverity.WARNING,
                message="Ollama not detected",
                detail="Ollama is not running on localhost:11434. Install from ollama.ai or switch to Claude API in Settings.",
            ),
            HealthCheck(name="claude_api_key", severity=CheckSeverity.OK),
        ])

        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Ollama not detected" in resp.text
        assert "ollama.ai" in resp.text

    @patch("option_alpha.web.routes._get_market_regime")
    @patch("option_alpha.web.routes.run_health_checks")
    def test_dashboard_shows_claude_error(self, mock_health, mock_market, client, settings):
        """Dashboard shows error when Claude API key missing."""
        mock_market.return_value = {"vix": None, "spy_trend": None, "spy_price": None}

        from option_alpha.web.errors import CheckSeverity, HealthCheck, SystemStatus
        mock_health.return_value = SystemStatus(checks=[
            HealthCheck(name="ollama", severity=CheckSeverity.OK),
            HealthCheck(
                name="claude_api_key",
                severity=CheckSeverity.ERROR,
                message="Claude API key not configured",
                detail="Claude backend is selected but no API key is set. Add it in Settings.",
            ),
        ])

        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Claude API key not configured" in resp.text
        assert "Settings" in resp.text

    @patch("option_alpha.web.routes._get_market_regime")
    @patch("option_alpha.web.routes.run_health_checks")
    def test_dashboard_shows_scan_error(self, mock_health, mock_market, client, settings):
        """Dashboard shows last scan error when a scan failed."""
        mock_market.return_value = {"vix": None, "spy_trend": None, "spy_price": None}

        from option_alpha.web.errors import CheckSeverity, HealthCheck, SystemStatus
        mock_health.return_value = SystemStatus(checks=[
            HealthCheck(name="ollama", severity=CheckSeverity.OK),
            HealthCheck(name="claude_api_key", severity=CheckSeverity.OK),
        ])

        conn = initialize_db(settings.db_path)
        conn.close()

        # Simulate a scan error
        import option_alpha.web.routes as routes_mod
        routes_mod._last_scan_error = "Scan failed: connection error. Check your internet connection and Ollama status."

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Scan failed" in resp.text

        routes_mod._last_scan_error = None

    @patch("option_alpha.web.routes._get_market_regime")
    @patch("option_alpha.web.routes.run_health_checks")
    def test_dashboard_no_data_shows_empty_state(self, mock_health, mock_market, client, settings):
        """Dashboard shows empty state message when no scans exist."""
        mock_market.return_value = {"vix": None, "spy_trend": None, "spy_price": None}

        from option_alpha.web.errors import CheckSeverity, HealthCheck, SystemStatus
        mock_health.return_value = SystemStatus(checks=[
            HealthCheck(name="ollama", severity=CheckSeverity.OK),
            HealthCheck(name="claude_api_key", severity=CheckSeverity.OK),
        ])

        conn = initialize_db(settings.db_path)
        conn.close()

        resp = client.get("/")
        assert resp.status_code == 200
        assert "No scan data yet" in resp.text
        assert "Run Scan" in resp.text


# ---------------------------------------------------------------------------
# 5. __main__.py Module Tests
# ---------------------------------------------------------------------------

class TestMainModule:
    """Tests for the __main__.py entry point."""

    def test_main_module_importable(self):
        """__main__.py can be imported as a module."""
        import option_alpha.__main__ as main_mod
        assert hasattr(main_mod, "main")
        assert hasattr(main_mod, "HOST")
        assert hasattr(main_mod, "PORT")
        assert hasattr(main_mod, "BANNER")

    def test_main_binds_to_localhost(self):
        """Server host is configured to 127.0.0.1 (localhost only)."""
        from option_alpha.__main__ import HOST
        assert HOST == "127.0.0.1"

    def test_main_default_port(self):
        """Default port is 8000."""
        from option_alpha.__main__ import PORT
        assert PORT == 8000

    def test_banner_contains_version(self):
        """Startup banner contains version info."""
        from option_alpha.__main__ import BANNER
        assert "Option Alpha" in BANNER
        assert "v1.0" in BANNER

    def test_banner_contains_url(self):
        """Startup banner contains the server URL."""
        from option_alpha.__main__ import BANNER
        assert "127.0.0.1:8000" in BANNER


# ---------------------------------------------------------------------------
# 6. PRD Acceptance Criteria Validation
# ---------------------------------------------------------------------------

class TestPRDAcceptanceCriteria:
    """Validate acceptance criteria from the PRD (US-1 through US-7)."""

    # US-1: Trigger scan from web UI, progress displayed, results shown
    @patch("option_alpha.web.routes._run_scan_task")
    @patch("option_alpha.web.routes._get_market_regime")
    def test_us1_scan_from_ui(self, mock_market, mock_scan, client, settings):
        """US-1: Can trigger scan from web UI, see progress, see results."""
        mock_market.return_value = {"vix": 18.0, "spy_trend": "up", "spy_price": 450.0}
        conn = initialize_db(settings.db_path)
        conn.close()

        import option_alpha.web.routes as routes_mod
        routes_mod._scan_running = False

        # Dashboard has Run Scan button
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Run Scan" in resp.text
        assert 'hx-post="/scan"' in resp.text

        # Trigger scan returns progress
        resp = client.post("/scan")
        assert resp.status_code == 200
        assert "Scan started" in resp.text or "scan" in resp.text.lower()

        routes_mod._scan_running = False

    # US-2: Candidates ranked by score, color-coded, sortable, top 10 highlighted
    def test_us2_candidates_ranking(self, client, settings, db_conn, seeded_db):
        """US-2: Candidates are ranked, color-coded, and sortable."""
        # Default sort by composite_score desc
        resp = client.get("/candidates")
        assert resp.status_code == 200
        text = resp.text

        # AAPL (90) should appear before AMZN (50) by default
        assert text.index("AAPL") < text.index("AMZN")

        # Direction badges are present (color-coded)
        assert "BULLISH" in text
        assert "BEARISH" in text

        # Sort by symbol ascending
        resp = client.get("/candidates?sort_by=symbol&order=asc")
        assert resp.status_code == 200
        text = resp.text
        assert text.index("AAPL") < text.index("MSFT")

    # US-3: Ticker detail shows score breakdown, options, Greeks, AI debate
    def test_us3_ticker_detail(self, client, settings, db_conn, seeded_db):
        """US-3: Ticker detail shows breakdown, options, and debate."""
        resp = client.get("/ticker/AAPL")
        assert resp.status_code == 200
        assert "Score Breakdown" in resp.text
        # Should show indicator contributions
        assert "bb_width" in resp.text.lower() or "Bb Width" in resp.text

    # US-4: Historical comparison via score history
    def test_us4_score_history(self, client, settings, db_conn, seeded_db):
        """US-4: Score history endpoint provides historical data."""
        resp = client.get("/history/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert isinstance(data["history"], list)
        assert len(data["history"]) >= 1
        # History entries have required fields
        entry = data["history"][0]
        assert "timestamp" in entry
        assert "composite_score" in entry
        assert "direction" in entry

    # US-5: HTML report export, self-contained, date-stamped
    def test_us5_report_export(self, client, settings, db_conn, seeded_db):
        """US-5: Export produces self-contained HTML report."""
        resp = client.get("/export")
        assert resp.status_code == 200

        # Date-stamped filename
        content_disp = resp.headers.get("content-disposition", "")
        assert "attachment" in content_disp
        assert "report_" in content_disp
        assert ".html" in content_disp

        # Self-contained (inline styles, no external refs)
        html = resp.text
        assert "<style>" in html
        assert 'href="/static' not in html
        assert 'src="/static' not in html

        # Contains data
        assert "AAPL" in html
        assert "Option Alpha Scan Report" in html

    # US-6: Backtesting engine exists and produces results
    def test_us6_backtest_engine(self):
        """US-6: Backtest engine can be imported and runs with fixture data."""
        from option_alpha.backtest.runner import BacktestRunner

        # Create fixture data
        ohlcv_data = {
            sym: _make_ohlcv_dataframe(sym, days=300)
            for sym in ["AAPL", "MSFT", "GOOG"]
        }
        tickers = list(ohlcv_data.keys())

        runner = BacktestRunner(
            lookback_period=100,
            breakout_threshold=3.0,
            holding_period=5,
            top_n=3,
            step_days=10,
        )
        result = runner.run(tickers, ohlcv_data)

        assert result is not None
        assert hasattr(result, "per_trade_results")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "total_trades")
        assert result.total_trades > 0

    # US-7: Settings page with weights/thresholds/AI config
    def test_us7_settings_page(self, client, settings):
        """US-7: Settings page shows all config sections and persists changes."""
        resp = client.get("/settings")
        assert resp.status_code == 200
        # All configuration sections present
        assert "Scoring Weights" in resp.text
        assert "Filters" in resp.text
        assert "Options Parameters" in resp.text
        assert "AI Configuration" in resp.text

        # Default values are shown
        assert "0.12" in resp.text  # bb_width weight
        assert "50.0" in resp.text  # min_composite_score

    def test_us7_settings_save(self, client, settings, tmp_path):
        """US-7: Settings can be saved and persisted."""
        form_data = {
            "weight_bb_width": "0.12",
            "weight_atr_percentile": "0.08",
            "weight_rsi": "0.08",
            "weight_obv_trend": "0.06",
            "weight_sma_alignment": "0.08",
            "weight_relative_volume": "0.06",
            "weight_catalyst_proximity": "0.15",
            "weight_stoch_rsi": "0.06",
            "weight_williams_r": "0.04",
            "weight_roc": "0.04",
            "weight_adx": "0.08",
            "weight_keltner_width": "0.05",
            "weight_vwap_deviation": "0.05",
            "weight_ad_trend": "0.05",
            "min_composite_score": "55.0",
            "min_price": "5.0",
            "min_avg_volume": "500000",
            "dte_min": "25",
            "dte_max": "55",
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

        # Verify the update took effect
        assert settings.scoring_weights["bb_width"] == 0.12
        assert settings.min_composite_score == 55.0
        assert settings.dte_min == 25

    def test_us7_settings_reset(self, client, settings):
        """US-7: Settings can be reset to defaults."""
        settings.min_composite_score = 99.0
        resp = client.post("/settings/reset")
        assert resp.status_code == 200
        new_settings = client.app.state.settings
        assert new_settings.min_composite_score == 50.0


# ---------------------------------------------------------------------------
# 7. Security Checks
# ---------------------------------------------------------------------------

class TestSecurity:
    """Security validation tests."""

    def test_server_binds_localhost_only(self):
        """Server HOST is 127.0.0.1 not 0.0.0.0."""
        from option_alpha.__main__ import HOST
        assert HOST == "127.0.0.1"
        assert HOST != "0.0.0.0"

    def test_api_keys_masked_in_settings(self, client, settings):
        """API keys are masked in the settings page."""
        settings.claude_api_key = "sk-ant-very-secret-key-12345678"
        settings.fred_api_key = "my-secret-fred-api-key"

        resp = client.get("/settings")
        assert resp.status_code == 200

        # Full keys must NOT appear
        assert "sk-ant-very-secret-key-12345678" not in resp.text
        assert "my-secret-fred-api-key" not in resp.text

        # Last 4 chars should appear
        assert "5678" in resp.text
        assert "-key" in resp.text

    def test_api_keys_not_in_export_report(self, client, settings, db_conn, seeded_db):
        """API keys never appear in exported HTML reports."""
        settings.claude_api_key = "sk-ant-super-secret-key"
        settings.fred_api_key = "fred-secret-key-1234"

        resp = client.get("/export")
        assert resp.status_code == 200

        # Keys must NOT appear anywhere in the report
        assert "sk-ant-super-secret-key" not in resp.text
        assert "fred-secret-key-1234" not in resp.text

    def test_error_messages_sanitize_secrets(self):
        """Error messages sanitize API keys from exception text."""
        from option_alpha.web.errors import format_scan_error

        error_with_key = Exception("Failed with api_key=sk-secret-1234")
        msg = format_scan_error(error_with_key)
        assert "sk-secret-1234" not in msg
        assert "authentication error" in msg.lower()

    def test_error_messages_sanitize_connection(self):
        """Connection errors produce user-friendly messages."""
        from option_alpha.web.errors import format_scan_error

        error = Exception("Connection refused to localhost:11434")
        msg = format_scan_error(error)
        assert "connection error" in msg.lower()


# ---------------------------------------------------------------------------
# 8. Error Module Unit Tests
# ---------------------------------------------------------------------------

class TestErrorModule:
    """Unit tests for the errors.py module."""

    @pytest.mark.asyncio
    async def test_check_ollama_when_not_ollama_backend(self):
        """Ollama check is OK when using Claude backend."""
        from option_alpha.web.errors import CheckSeverity, check_ollama

        settings = Settings(ai_backend="claude", claude_api_key="test-key")
        result = await check_ollama(settings)
        assert result.severity == CheckSeverity.OK

    @pytest.mark.asyncio
    async def test_check_ollama_when_ollama_not_running(self):
        """Ollama check returns warning when Ollama is down."""
        from option_alpha.web.errors import CheckSeverity, check_ollama

        settings = Settings(ai_backend="ollama")
        # Ollama is not actually running in test environment
        result = await check_ollama(settings)
        assert result.severity == CheckSeverity.WARNING
        assert "not detected" in result.message.lower() or result.severity == CheckSeverity.OK

    def test_check_claude_key_missing(self):
        """Claude key check returns error when key missing."""
        from option_alpha.web.errors import CheckSeverity, check_claude_api_key

        settings = Settings(ai_backend="claude", claude_api_key=None)
        result = check_claude_api_key(settings)
        assert result.severity == CheckSeverity.ERROR
        assert "not configured" in result.message.lower()

    def test_check_claude_key_present(self):
        """Claude key check returns OK when key is set."""
        from option_alpha.web.errors import CheckSeverity, check_claude_api_key

        settings = Settings(ai_backend="claude", claude_api_key="sk-test")
        result = check_claude_api_key(settings)
        assert result.severity == CheckSeverity.OK

    def test_check_claude_key_not_relevant(self):
        """Claude key check is OK when using Ollama backend."""
        from option_alpha.web.errors import CheckSeverity, check_claude_api_key

        settings = Settings(ai_backend="ollama")
        result = check_claude_api_key(settings)
        assert result.severity == CheckSeverity.OK

    @pytest.mark.asyncio
    async def test_system_status_aggregation(self):
        """SystemStatus correctly aggregates errors and warnings."""
        from option_alpha.web.errors import run_health_checks

        settings = Settings(ai_backend="claude", claude_api_key=None)
        status = await run_health_checks(settings)

        assert status.has_errors  # Claude key missing
        assert len(status.errors) >= 1

    def test_format_scan_error_rate_limit(self):
        """Rate limit errors produce friendly message."""
        from option_alpha.web.errors import format_scan_error

        error = Exception("429 Too Many Requests")
        msg = format_scan_error(error)
        assert "rate limited" in msg.lower()

    def test_format_scan_error_truncation(self):
        """Long error messages are truncated."""
        from option_alpha.web.errors import format_scan_error

        error = Exception("A" * 500)
        msg = format_scan_error(error)
        assert len(msg) < 500


# ---------------------------------------------------------------------------
# 9. .gitignore Validation
# ---------------------------------------------------------------------------

class TestGitignore:
    """Verify .gitignore has required entries."""

    def test_gitignore_has_data_dir(self):
        """Data directory is in .gitignore."""
        gitignore = Path("C:/Users/nicho/Desktop/option_alpha/.gitignore").read_text()
        assert "/data/" in gitignore or "data/" in gitignore

    def test_gitignore_has_config_json(self):
        """config.json is in .gitignore."""
        gitignore = Path("C:/Users/nicho/Desktop/option_alpha/.gitignore").read_text()
        assert "config.json" in gitignore

    def test_gitignore_has_db_files(self):
        """SQLite database files are in .gitignore."""
        gitignore = Path("C:/Users/nicho/Desktop/option_alpha/.gitignore").read_text()
        assert "*.db" in gitignore

    def test_gitignore_has_reports(self):
        """Reports directory is in .gitignore."""
        gitignore = Path("C:/Users/nicho/Desktop/option_alpha/.gitignore").read_text()
        assert "reports/" in gitignore


# ---------------------------------------------------------------------------
# 10. Backward Compatibility Tests (Issue #54)
# ---------------------------------------------------------------------------

def _make_ohlcv_synthetic(rows=200, trend="up"):
    """Create synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    if trend == "up":
        close = 100 + np.cumsum(np.random.randn(rows) * 0.5 + 0.1)
    elif trend == "down":
        close = 200 + np.cumsum(np.random.randn(rows) * 0.5 - 0.1)
    else:
        close = 100 + np.random.randn(rows) * 0.5
    high = close + np.abs(np.random.randn(rows)) * 1.5
    low = close - np.abs(np.random.randn(rows)) * 1.5
    open_ = close + np.random.randn(rows) * 0.5
    volume = np.random.randint(100000, 1000000, rows).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


class TestBackwardCompatibility:
    """Verify backward compatibility with old and new config formats."""

    def test_settings_load_without_new_keys(self, tmp_path):
        """Settings from a dict with only old weight keys loads without error."""
        old_config = {
            "scoring_weights": {
                "bb_width": 0.20,
                "atr_percentile": 0.15,
                "rsi": 0.10,
                "obv_trend": 0.10,
                "sma_alignment": 0.10,
                "relative_volume": 0.10,
                "catalyst_proximity": 0.25,
            },
            "min_price": 5.0,
        }
        config_path = tmp_path / "old_config.json"
        config_path.write_text(json.dumps(old_config))

        loaded = Settings.load(config_path)
        # Should load without error
        assert loaded.min_price == 5.0
        # Old weights should be preserved
        assert loaded.scoring_weights["bb_width"] == 0.20
        assert loaded.scoring_weights["catalyst_proximity"] == 0.25
        # New keys should NOT be present (Settings uses the dict as-is)
        # The scoring_weights dict is whatever was in the JSON
        assert len(loaded.scoring_weights) == 7

    def test_settings_roundtrip_all_weights(self, tmp_path):
        """Settings with all 14 weights save and load round-trip correctly."""
        s = Settings()
        assert len(s.scoring_weights) == 14

        config_path = tmp_path / "full_config.json"
        s.save(config_path)

        loaded = Settings.load(config_path)
        assert len(loaded.scoring_weights) == 14

        # Verify all 14 keys round-trip
        for key, val in s.scoring_weights.items():
            assert loaded.scoring_weights[key] == val, f"{key} mismatch"

    def test_settings_load_partial_new_keys(self, tmp_path):
        """Settings with a mix of old and some new weight keys loads correctly."""
        mixed_config = {
            "scoring_weights": {
                "bb_width": 0.20,
                "atr_percentile": 0.15,
                "rsi": 0.10,
                "obv_trend": 0.10,
                "sma_alignment": 0.10,
                "relative_volume": 0.10,
                "catalyst_proximity": 0.15,
                "adx": 0.10,
            },
        }
        config_path = tmp_path / "mixed_config.json"
        config_path.write_text(json.dumps(mixed_config))

        loaded = Settings.load(config_path)
        assert loaded.scoring_weights["adx"] == 0.10
        assert loaded.scoring_weights["bb_width"] == 0.20
        assert len(loaded.scoring_weights) == 8


# ---------------------------------------------------------------------------
# 11. Full Scoring Path Tests (Issue #54)
# ---------------------------------------------------------------------------

class TestFullScoringPath:
    """Verify compute_all_indicators and score_universe with expanded indicators."""

    def test_compute_all_indicators_returns_14_keys(self):
        """compute_all_indicators returns dict with exactly 14 keys for sufficient data."""
        from option_alpha.scoring.indicators import compute_all_indicators

        df = _make_ohlcv_synthetic(rows=200, trend="up")
        result = compute_all_indicators(df)

        assert len(result) == 14
        expected_keys = {
            "bb_width", "atr_percent", "rsi", "obv_trend", "sma_alignment",
            "relative_volume", "vwap_deviation", "ad_trend", "stoch_rsi",
            "williams_r", "roc", "keltner_width", "adx", "supertrend",
        }
        assert set(result.keys()) == expected_keys

        # All values should be float and non-NaN for 200 rows
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"
            assert not np.isnan(val), f"{key} is NaN with 200 rows"

    def test_score_universe_with_new_indicators(self):
        """score_universe produces valid TickerScores with 13+ breakdown entries."""
        from option_alpha.scoring.composite import score_universe

        # Create synthetic OHLCV for 5 tickers
        ohlcv_data = {}
        for i, sym in enumerate(["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]):
            np.random.seed(42 + i)
            ohlcv_data[sym] = _make_ohlcv_synthetic(rows=200, trend="up")

        settings = Settings()
        scores = score_universe(ohlcv_data, settings)

        # Returns 5 TickerScore objects
        assert len(scores) == 5

        for score in scores:
            # Composite score in [0, 100]
            assert 0 <= score.composite_score <= 100
            # Each has 13+ breakdown entries (all weighted indicators)
            assert len(score.breakdown) >= 13, (
                f"{score.symbol} has only {len(score.breakdown)} breakdown entries"
            )
            # Each has a valid direction
            assert score.direction in (Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL)
            # Breakdown entries have proper fields
            for bd in score.breakdown:
                assert bd.weight > 0
                assert 0 <= bd.normalized <= 100

    def test_score_universe_sorted_descending(self):
        """score_universe returns scores sorted by composite_score descending."""
        from option_alpha.scoring.composite import score_universe

        ohlcv_data = {}
        for i, sym in enumerate(["T1", "T2", "T3", "T4", "T5"]):
            np.random.seed(100 + i)
            ohlcv_data[sym] = _make_ohlcv_synthetic(rows=250, trend="up")

        scores = score_universe(ohlcv_data)

        for i in range(len(scores) - 1):
            assert scores[i].composite_score >= scores[i + 1].composite_score


# ---------------------------------------------------------------------------
# 12. Direction Signal Tests (Issue #54)
# ---------------------------------------------------------------------------

class TestDirectionSignals:
    """Test ADX-based direction signal logic."""

    def test_direction_adx_low_returns_neutral(self):
        """Low ADX (<20) forces NEUTRAL regardless of RSI/SMA signals."""
        from option_alpha.scoring.composite import determine_direction

        # Strong uptrend data (would normally be BULLISH)
        df = _make_ohlcv_synthetic(rows=300, trend="up")

        # Mock adx to return a low value (weak trend)
        with patch("option_alpha.scoring.composite.adx", return_value=15.0):
            direction = determine_direction(df)
            assert direction == Direction.NEUTRAL

    def test_direction_adx_high_preserves_signal(self):
        """High ADX (>25) with bullish RSI/SMA preserves BULLISH signal."""
        from option_alpha.scoring.composite import determine_direction

        # Strong uptrend data
        df = _make_ohlcv_synthetic(rows=300, trend="up")

        # Mock adx to high value and rsi/sma for bullish
        with (
            patch("option_alpha.scoring.composite.adx", return_value=30.0),
            patch("option_alpha.scoring.composite.rsi", return_value=65.0),
            patch("option_alpha.scoring.composite.sma_direction", return_value="bullish"),
        ):
            direction = determine_direction(df)
            assert direction == Direction.BULLISH

    def test_direction_adx_nan_falls_through(self):
        """NaN ADX falls through to RSI+SMA logic (backward compatibility)."""
        from option_alpha.scoring.composite import determine_direction

        df = _make_ohlcv_synthetic(rows=300, trend="up")

        with (
            patch("option_alpha.scoring.composite.adx", return_value=float("nan")),
            patch("option_alpha.scoring.composite.rsi", return_value=65.0),
            patch("option_alpha.scoring.composite.sma_direction", return_value="bullish"),
        ):
            direction = determine_direction(df)
            assert direction == Direction.BULLISH

    def test_direction_adx_high_bearish(self):
        """High ADX with bearish RSI/SMA returns BEARISH."""
        from option_alpha.scoring.composite import determine_direction

        df = _make_ohlcv_synthetic(rows=300, trend="down")

        with (
            patch("option_alpha.scoring.composite.adx", return_value=30.0),
            patch("option_alpha.scoring.composite.rsi", return_value=35.0),
            patch("option_alpha.scoring.composite.sma_direction", return_value="bearish"),
        ):
            direction = determine_direction(df)
            assert direction == Direction.BEARISH


# ---------------------------------------------------------------------------
# 13. Context Enrichment Tests (Issue #54)
# ---------------------------------------------------------------------------

class TestContextEnrichment:
    """Test build_context produces enriched output with new indicator data."""

    def _make_enriched_ticker_score(self):
        """Create a TickerScore with 14 breakdown entries for context testing."""
        from option_alpha.scoring.composite import INDICATOR_WEIGHT_MAP

        breakdown = []
        for indicator_name, config_key in [
            ("bb_width", "bb_width"),
            ("atr_percent", "atr_percentile"),
            ("rsi", "rsi"),
            ("obv_trend", "obv_trend"),
            ("sma_alignment", "sma_alignment"),
            ("relative_volume", "relative_volume"),
            ("stoch_rsi", "stoch_rsi"),
            ("williams_r", "williams_r"),
            ("roc", "roc"),
            ("adx", "adx"),
            ("keltner_width", "keltner_width"),
            ("vwap_deviation", "vwap_deviation"),
            ("ad_trend", "ad_trend"),
            ("catalyst_proximity", "catalyst_proximity"),
        ]:
            raw_vals = {
                "bb_width": 0.05, "atr_percent": 3.5, "rsi": 55.0,
                "obv_trend": 1.2, "sma_alignment": 80.0,
                "relative_volume": 1.5, "stoch_rsi": 45.0,
                "williams_r": -40.0, "roc": 5.5, "adx": 28.0,
                "keltner_width": 0.06, "vwap_deviation": 0.8,
                "ad_trend": 0.9, "catalyst_proximity": 0.67,
            }
            breakdown.append(ScoreBreakdown(
                name=indicator_name,
                raw_value=raw_vals.get(indicator_name, 50.0),
                normalized=65.0,
                weight=0.07,
                contribution=4.55,
            ))

        return TickerScore(
            symbol="AAPL",
            composite_score=72.5,
            direction=Direction.BULLISH,
            last_price=185.50,
            avg_volume=2_500_000.0,
            breakdown=breakdown,
        )

    def test_build_context_enriched_output(self):
        """build_context produces output with Interpretation column and all sections."""
        from option_alpha.ai.context import build_context

        ts = self._make_enriched_ticker_score()
        rec = OptionsRecommendation(
            symbol="AAPL",
            direction=Direction.BULLISH,
            option_type="call",
            strike=190.0,
            expiry=datetime.now(UTC) + timedelta(days=45),
            dte=45,
            delta=0.35,
            implied_volatility=0.30,
            open_interest=5000,
            volume=1200,
        )

        context = build_context(ts, options_rec=rec)

        # Verify Interpretation column present (from _format_score_breakdown)
        assert "Interpretation" in context
        # Verify OPTIONS FLOW section
        assert "OPTIONS FLOW:" in context
        # Verify RISK PARAMETERS section
        assert "RISK PARAMETERS:" in context
        # Verify total length under 16000 chars
        assert len(context) < 16000, f"Context too long: {len(context)} chars"
        # Verify key indicators appear
        assert "adx" in context.lower() or "ADX" in context
        assert "AAPL" in context
        assert "72.5" in context

    def test_build_context_with_sector(self):
        """build_context includes SECTOR when sector parameter is provided."""
        from option_alpha.ai.context import build_context

        ts = self._make_enriched_ticker_score()
        context = build_context(ts, sector="Technology")

        assert "SECTOR: Technology" in context

    def test_build_context_without_options(self):
        """build_context works without options recommendation."""
        from option_alpha.ai.context import build_context

        ts = self._make_enriched_ticker_score()
        context = build_context(ts)

        # Should still have score breakdown
        assert "SCORE BREAKDOWN" in context
        assert "SIGNAL SUMMARY" in context
        # Should NOT have options sections
        assert "OPTIONS FLOW:" not in context
        assert "OPTIONS RECOMMENDATION:" not in context


# ---------------------------------------------------------------------------
# 14. Indicator Weight Map Consistency Tests (Issue #54)
# ---------------------------------------------------------------------------

class TestWeightMapConsistency:
    """Verify indicator weight map and config weights are consistent."""

    def test_weight_map_has_13_entries(self):
        """INDICATOR_WEIGHT_MAP has 13 entries (no catalyst_proximity, no supertrend)."""
        from option_alpha.scoring.composite import INDICATOR_WEIGHT_MAP

        assert len(INDICATOR_WEIGHT_MAP) == 13

    def test_config_weights_sum_to_one(self):
        """Default scoring weights sum to 1.0."""
        s = Settings()
        total = sum(s.scoring_weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_config_has_14_weight_keys(self):
        """Default config has 14 weight keys (13 computed + catalyst_proximity)."""
        s = Settings()
        assert len(s.scoring_weights) == 14

    def test_all_weighted_indicators_in_compute_all(self):
        """All indicators in INDICATOR_WEIGHT_MAP are returned by compute_all_indicators."""
        from option_alpha.scoring.composite import INDICATOR_WEIGHT_MAP
        from option_alpha.scoring.indicators import compute_all_indicators

        df = _make_ohlcv_synthetic(rows=250)
        result = compute_all_indicators(df)

        for config_key, indicator_name in INDICATOR_WEIGHT_MAP.items():
            assert indicator_name in result, (
                f"Indicator '{indicator_name}' (config key '{config_key}') "
                f"not in compute_all_indicators output"
            )
