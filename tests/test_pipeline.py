"""Comprehensive tests for pipeline orchestrator.

Tests cover:
- Orchestration order (phases run in correct sequence)
- Progress callback invocation at start/completion of each phase
- Partial failure handling (some tickers fail, pipeline continues)
- Phase timing is recorded
- Filter chain (all -> top_n_options -> top_n_ai_debate)
- ScanResult assembly with correct fields
- Error handling for each phase
- TickerData-to-DataFrame conversion
- Progress model validation
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from option_alpha.config import Settings
from option_alpha.models import (
    AgentResponse,
    DebateResult,
    Direction,
    OptionsRecommendation,
    ScanResult,
    TickerData,
    TickerScore,
    TradeThesis,
)
from option_alpha.pipeline.orchestrator import (
    PHASE_NAMES,
    ScanOrchestrator,
    _ticker_data_to_dataframe,
)
from option_alpha.pipeline.progress import (
    PhaseProgress,
    PhaseStatus,
    ScanProgress,
)


def _make_async_client_mock():
    """Create an AsyncMock LLM client with health_check returning True."""
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ticker_data(symbol: str, n_days: int = 30) -> TickerData:
    """Create synthetic TickerData for testing."""
    base = 100.0
    dates = pd.bdate_range(end=datetime.now(UTC), periods=n_days).tolist()
    return TickerData(
        symbol=symbol,
        dates=dates,
        open=[base + i * 0.1 for i in range(n_days)],
        high=[base + i * 0.1 + 1.0 for i in range(n_days)],
        low=[base + i * 0.1 - 1.0 for i in range(n_days)],
        close=[base + i * 0.2 for i in range(n_days)],
        volume=[1_000_000 + i * 1000 for i in range(n_days)],
        last_price=base + (n_days - 1) * 0.2,
        avg_volume=1_000_000.0,
    )


def _make_ticker_score(symbol: str, score: float = 75.0) -> TickerScore:
    """Create a TickerScore for testing."""
    return TickerScore(
        symbol=symbol,
        composite_score=score,
        direction=Direction.BULLISH,
        last_price=150.0,
        avg_volume=2_000_000.0,
    )


def _make_options_rec(symbol: str) -> OptionsRecommendation:
    """Create a minimal OptionsRecommendation for testing."""
    return OptionsRecommendation(
        symbol=symbol,
        direction=Direction.BULLISH,
        option_type="call",
        strike=155.0,
        expiry=datetime.now(UTC),
        dte=45,
        delta=0.35,
    )


def _make_debate_result(symbol: str) -> DebateResult:
    """Create a minimal DebateResult for testing."""
    return DebateResult(
        symbol=symbol,
        bull=AgentResponse(role="bull", analysis="Bullish outlook"),
        bear=AgentResponse(role="bear", analysis="Bearish outlook"),
        risk=AgentResponse(role="risk", analysis="Moderate risk"),
        final_thesis=TradeThesis(
            symbol=symbol,
            direction=Direction.BULLISH,
            conviction=7,
            entry_rationale="Strong momentum",
            recommended_action=f"Buy {symbol} 155C 45DTE",
        ),
    )


@pytest.fixture
def settings(tmp_path):
    """Settings with temp paths and small top_n values for fast tests."""
    return Settings(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        top_n_options=3,
        top_n_ai_debate=2,
        ai_backend="ollama",
    )


@pytest.fixture
def orchestrator(settings):
    """ScanOrchestrator configured with test settings."""
    return ScanOrchestrator(settings=settings)


# ---------------------------------------------------------------------------
# Progress model tests
# ---------------------------------------------------------------------------

class TestPhaseProgress:
    """Tests for PhaseProgress model."""

    def test_defaults(self):
        p = PhaseProgress(phase_name="test")
        assert p.status == PhaseStatus.PENDING
        assert p.percentage == 0.0
        assert p.ticker_count == 0
        assert p.elapsed_seconds == 0.0
        assert p.message == ""

    def test_all_fields(self):
        p = PhaseProgress(
            phase_name="scoring",
            status=PhaseStatus.COMPLETED,
            percentage=100.0,
            ticker_count=500,
            elapsed_seconds=12.5,
            message="Scored 500 tickers",
        )
        assert p.phase_name == "scoring"
        assert p.status == PhaseStatus.COMPLETED
        assert p.ticker_count == 500


class TestScanProgress:
    """Tests for ScanProgress model."""

    def test_defaults(self):
        sp = ScanProgress()
        assert sp.phases == []
        assert sp.overall_percentage == 0.0
        assert sp.current_phase is None
        assert sp.elapsed_total == 0.0

    def test_with_phases(self):
        phases = [PhaseProgress(phase_name=name) for name in PHASE_NAMES]
        sp = ScanProgress(phases=phases)
        assert len(sp.phases) == 6
        assert sp.phases[0].phase_name == "data_fetch"


class TestPhaseStatus:
    """Tests for PhaseStatus enum."""

    def test_values(self):
        assert PhaseStatus.PENDING == "pending"
        assert PhaseStatus.RUNNING == "running"
        assert PhaseStatus.COMPLETED == "completed"
        assert PhaseStatus.FAILED == "failed"


# ---------------------------------------------------------------------------
# TickerData-to-DataFrame conversion tests
# ---------------------------------------------------------------------------

class TestTickerDataToDataFrame:
    """Tests for _ticker_data_to_dataframe helper."""

    def test_basic_conversion(self):
        td = _make_ticker_data("AAPL", n_days=10)
        df = _ticker_data_to_dataframe(td)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 10

    def test_values_match(self):
        td = _make_ticker_data("TSLA", n_days=5)
        df = _ticker_data_to_dataframe(td)

        assert df["Close"].iloc[0] == td.close[0]
        assert df["Volume"].iloc[-1] == td.volume[-1]

    def test_empty_ticker_data(self):
        td = TickerData(symbol="EMPTY")
        df = _ticker_data_to_dataframe(td)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Orchestrator tests (fully mocked phases)
# ---------------------------------------------------------------------------

class TestScanOrchestratorPhaseOrder:
    """Verify that phases execute in the correct order."""

    @pytest.mark.asyncio
    async def test_phases_run_in_order(self, orchestrator, settings):
        """All 6 phases should run in order: data_fetch -> ... -> persist."""
        call_order = []

        tickers = ["AAPL", "MSFT", "GOOGL"]
        ticker_data = {s: _make_ticker_data(s) for s in tickers}
        scores = [_make_ticker_score(s, 90 - i * 5) for i, s in enumerate(tickers)]
        recs = [_make_options_rec(s) for s in tickers[:2]]
        debates = [_make_debate_result(s) for s in tickers[:1]]

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe") as mock_univ,
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch") as mock_cache_load,
            patch("option_alpha.pipeline.orchestrator.fetch_batch") as mock_fetch,
            patch("option_alpha.pipeline.orchestrator.save_batch") as mock_save_cache,
            patch("option_alpha.scoring.composite.score_universe") as mock_score,
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info") as mock_earn,
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores") as mock_merge,
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers") as mock_chains,
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers") as mock_recs,
            patch("option_alpha.pipeline.orchestrator.get_client") as mock_get_client,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_init_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run") as mock_save_run,
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores") as mock_save_scores,
            patch("option_alpha.pipeline.orchestrator.save_ai_theses") as mock_save_theses,
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            # Phase 1 mocks
            mock_univ.side_effect = lambda **kw: (call_order.append("get_universe") or tickers)
            mock_cache_load.return_value = {}
            mock_fetch.side_effect = lambda syms, **kw: (call_order.append("fetch_batch") or ticker_data)
            mock_save_cache.return_value = 3

            # Phase 2 mock
            mock_score.side_effect = lambda frames, **kw: (call_order.append("score_universe") or scores)

            # Phase 3 mocks
            mock_earn.side_effect = lambda syms, **kw: (call_order.append("batch_earnings") or {})
            mock_merge.side_effect = lambda ts, ei, **kw: (call_order.append("merge_catalysts") or ts)

            # Phase 4 mocks
            mock_chains.side_effect = lambda syms, **kw: (call_order.append("fetch_chains") or {})
            mock_recs.side_effect = lambda ts, ch, **kw: (call_order.append("recommend") or recs)

            # Phase 5 mocks
            mock_client = AsyncMock()
            mock_client.health_check = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_client
            mock_debate_mgr = AsyncMock()
            mock_debate_mgr.run_debates = AsyncMock(return_value=debates)
            with patch("option_alpha.pipeline.orchestrator.DebateManager", return_value=mock_debate_mgr):
                # Phase 6 mocks
                mock_conn = MagicMock()
                mock_init_db.return_value = mock_conn
                mock_save_run.side_effect = lambda conn, sr: (call_order.append("save_scan_run") or 1)
                mock_save_scores.side_effect = lambda *a: call_order.append("save_scores")
                mock_save_theses.side_effect = lambda *a: call_order.append("save_theses")

                result = await orchestrator.run_scan()

            # Verify order
            assert call_order.index("get_universe") < call_order.index("fetch_batch")
            assert call_order.index("fetch_batch") < call_order.index("score_universe")
            assert call_order.index("score_universe") < call_order.index("batch_earnings")
            assert call_order.index("merge_catalysts") < call_order.index("fetch_chains")
            assert call_order.index("recommend") < call_order.index("save_scan_run")

            # Verify result type
            assert isinstance(result, ScanResult)


class TestProgressCallback:
    """Verify progress callback is called at each phase start/completion."""

    @pytest.mark.asyncio
    async def test_callback_called_for_each_phase(self, orchestrator):
        """Progress callback should be called at start and end of each phase."""
        progress_updates: list[ScanProgress] = []

        async def track_progress(p: ScanProgress):
            # Deep copy to avoid mutation.
            import copy
            progress_updates.append(copy.deepcopy(p))

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=[_make_ticker_score("AAPL")]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orchestrator.run_scan(on_progress=track_progress)

        # 6 phases x 2 (start + complete) + 1 final = 13 updates
        assert len(progress_updates) >= 13

        # First update should be phase 0 starting (data_fetch).
        assert progress_updates[0].current_phase == "data_fetch"

        # Final update should have 100% overall.
        assert progress_updates[-1].overall_percentage == 100.0

    @pytest.mark.asyncio
    async def test_no_callback_no_error(self, orchestrator):
        """Pipeline should work fine without a progress callback."""
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=0),
            patch("option_alpha.scoring.composite.score_universe", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan(on_progress=None)
            assert isinstance(result, ScanResult)


class TestPartialFailureHandling:
    """Test that the pipeline handles per-ticker and per-phase failures gracefully."""

    @pytest.mark.asyncio
    async def test_data_fetch_partial_failure(self, orchestrator):
        """If some tickers fail to fetch, pipeline continues with the rest."""
        # Only AAPL succeeds, MSFT and GOOGL missing from fetch result.
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL", "MSFT", "GOOGL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=[_make_ticker_score("AAPL")]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        # Pipeline should complete with just AAPL.
        assert result.total_tickers_scanned == 1
        assert len(result.ticker_scores) == 1
        assert result.ticker_scores[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_scoring_phase_exception(self, orchestrator):
        """If scoring raises an exception, pipeline continues with empty scores."""
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", side_effect=RuntimeError("Scoring exploded")),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        # Scoring failed, so no scores.
        assert len(result.ticker_scores) == 0
        # Pipeline still completed.
        assert isinstance(result, ScanResult)

    @pytest.mark.asyncio
    async def test_catalysts_phase_exception(self, orchestrator):
        """If catalysts phase raises, pipeline continues with original scores."""
        scores = [_make_ticker_score("AAPL")]
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", side_effect=RuntimeError("Earnings API down")),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        # Original scores preserved despite catalyst failure.
        assert len(result.ticker_scores) == 1

    @pytest.mark.asyncio
    async def test_options_phase_exception(self, orchestrator):
        """If options phase raises, pipeline continues with empty recs."""
        scores = [_make_ticker_score("AAPL")]
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", side_effect=RuntimeError("Chains API down")),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        assert len(result.options_recommendations) == 0
        # Scores still exist.
        assert len(result.ticker_scores) == 1

    @pytest.mark.asyncio
    async def test_ai_debate_phase_exception(self, orchestrator):
        """If AI debate phase raises, pipeline continues with empty debates."""
        scores = [_make_ticker_score("AAPL")]
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", side_effect=RuntimeError("No LLM")),
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
        ):
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        assert len(result.debate_results) == 0

    @pytest.mark.asyncio
    async def test_persist_phase_exception(self, orchestrator):
        """If persist phase raises, pipeline still returns result."""
        scores = [_make_ticker_score("AAPL")]
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db", side_effect=RuntimeError("DB init failed")),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm

            result = await orchestrator.run_scan()

        # Result still returned with scores intact.
        assert len(result.ticker_scores) == 1


class TestPhaseTiming:
    """Verify that phase timing is recorded."""

    @pytest.mark.asyncio
    async def test_timings_recorded(self, orchestrator):
        """All 6 phase timings should be recorded after scan."""
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=[_make_ticker_score("AAPL")]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orchestrator.run_scan()

        timings = orchestrator.phase_timings
        assert set(timings.keys()) == set(PHASE_NAMES)
        # All timings should be non-negative.
        for phase, duration in timings.items():
            assert duration >= 0, f"{phase} timing is negative: {duration}"


class TestFilterChain:
    """Test the filter chain: all tickers -> top_n_options -> top_n_ai_debate."""

    @pytest.mark.asyncio
    async def test_top_n_options_filter(self, settings):
        """Only top_n_options tickers should be passed to options phase."""
        settings.top_n_options = 2
        settings.top_n_ai_debate = 1
        orch = ScanOrchestrator(settings=settings)

        # 5 tickers scored, only top 2 should go to options.
        all_scores = [_make_ticker_score(f"T{i}", 100 - i * 10) for i in range(5)]

        captured_options_symbols = []

        def mock_fetch_chains(symbols, **kw):
            captured_options_symbols.extend(symbols)
            return {}

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe",
                  return_value=[f"T{i}" for i in range(5)]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={f"T{i}": _make_ticker_data(f"T{i}") for i in range(5)}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=5),
            patch("option_alpha.scoring.composite.score_universe", return_value=all_scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers",
                  side_effect=mock_fetch_chains),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orch.run_scan()

        # Only top 2 (by score) should be sent to options phase.
        assert len(captured_options_symbols) == 2
        assert captured_options_symbols[0] == "T0"  # score=100
        assert captured_options_symbols[1] == "T1"  # score=90

    @pytest.mark.asyncio
    async def test_top_n_ai_debate_filter(self, settings):
        """Only top_n_ai_debate tickers should be passed to debate phase."""
        settings.top_n_options = 5
        settings.top_n_ai_debate = 2
        orch = ScanOrchestrator(settings=settings)

        all_scores = [_make_ticker_score(f"T{i}", 100 - i * 10) for i in range(5)]

        captured_debate_top_n = []

        class FakeDebateManager:
            def __init__(self, client):
                pass

            async def run_debates(self, scores, options_recs=None, top_n=10, **kw):
                captured_debate_top_n.append(top_n)
                return []

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe",
                  return_value=[f"T{i}" for i in range(5)]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={f"T{i}": _make_ticker_data(f"T{i}") for i in range(5)}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=5),
            patch("option_alpha.scoring.composite.score_universe", return_value=all_scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager", FakeDebateManager),
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_db.return_value = MagicMock()
            await orch.run_scan()

        assert captured_debate_top_n == [2]


class TestScanResultAssembly:
    """Test that ScanResult is properly assembled from all phase outputs."""

    @pytest.mark.asyncio
    async def test_result_contains_all_data(self, orchestrator):
        """ScanResult should contain scores, recs, and debates."""
        tickers = ["AAPL", "MSFT"]
        scores = [_make_ticker_score(s, 90 - i * 10) for i, s in enumerate(tickers)]
        recs = [_make_options_rec("AAPL")]
        debates = [_make_debate_result("AAPL")]

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=tickers),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={s: _make_ticker_data(s) for s in tickers}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=2),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=recs),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=debates)
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        assert result.total_tickers_scanned == 2
        assert result.top_n_scored == 2
        assert result.top_n_debated == 1
        assert len(result.ticker_scores) == 2
        assert len(result.options_recommendations) == 1
        assert len(result.debate_results) == 1
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_empty_universe_scan(self, orchestrator):
        """Scan with empty universe should return empty result without errors."""
        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=0),
            patch("option_alpha.scoring.composite.score_universe", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        assert result.total_tickers_scanned == 0
        assert result.top_n_scored == 0
        assert result.top_n_debated == 0


class TestCacheIntegration:
    """Test cache load/save interactions."""

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_fetch(self, orchestrator):
        """Cached tickers should not be re-fetched."""
        cached_data = {"AAPL": _make_ticker_data("AAPL")}

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value=cached_data),
            patch("option_alpha.pipeline.orchestrator.fetch_batch") as mock_fetch,
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=0),
            patch("option_alpha.scoring.composite.score_universe", return_value=[_make_ticker_score("AAPL")]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        # fetch_batch should NOT be called since all tickers are cached.
        mock_fetch.assert_not_called()
        assert result.total_tickers_scanned == 1


class TestPersistPhase:
    """Test persistence phase saves all data correctly."""

    @pytest.mark.asyncio
    async def test_persist_saves_scores_and_theses(self, orchestrator):
        """Persist phase should call save_scan_run, save_ticker_scores, save_ai_theses."""
        scores = [_make_ticker_score("AAPL")]
        debates = [_make_debate_result("AAPL")]

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client") as mock_get_client,
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=42) as mock_save_run,
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores") as mock_save_scores,
            patch("option_alpha.pipeline.orchestrator.save_ai_theses") as mock_save_theses,
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_client = AsyncMock()
            mock_client.health_check = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_client
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=debates)
            mock_dm_cls.return_value = mock_dm
            mock_conn = MagicMock()
            mock_db.return_value = mock_conn

            await orchestrator.run_scan()

        mock_save_run.assert_called_once()
        mock_save_scores.assert_called_once()
        # Verify scan_run_id is passed correctly.
        assert mock_save_scores.call_args[0][1] == 42
        mock_save_theses.assert_called_once()
        assert mock_save_theses.call_args[0][1] == 42

    @pytest.mark.asyncio
    async def test_persist_skips_empty_theses(self, orchestrator):
        """Persist should not call save_ai_theses when no debates exist."""
        scores = [_make_ticker_score("AAPL")]

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1) as mock_save_run,
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores") as mock_save_scores,
            patch("option_alpha.pipeline.orchestrator.save_ai_theses") as mock_save_theses,
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orchestrator.run_scan()

        mock_save_run.assert_called_once()
        mock_save_scores.assert_called_once()
        # No debates -> should not save theses.
        mock_save_theses.assert_not_called()


class TestOrchestratorConfig:
    """Test orchestrator configuration."""

    def test_default_settings(self):
        """Orchestrator should work with default settings."""
        orch = ScanOrchestrator()
        assert orch.settings.top_n_options == 50
        assert orch.settings.top_n_ai_debate == 10

    def test_custom_settings(self, settings):
        """Orchestrator should use provided settings."""
        orch = ScanOrchestrator(settings=settings)
        assert orch.settings.top_n_options == 3
        assert orch.settings.top_n_ai_debate == 2

    def test_phase_timings_empty_before_scan(self, orchestrator):
        """Phase timings should be empty before running a scan."""
        assert orchestrator.phase_timings == {}


class TestProgressPercentage:
    """Test that progress percentage calculations are correct."""

    @pytest.mark.asyncio
    async def test_progress_percentage_increments(self, orchestrator):
        """Overall percentage should increment after each phase completion."""
        percentages = []

        async def capture_pct(p: ScanProgress):
            percentages.append(p.overall_percentage)

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=0),
            patch("option_alpha.scoring.composite.score_universe", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orchestrator.run_scan(on_progress=capture_pct)

        # Percentages should be monotonically non-decreasing.
        for i in range(1, len(percentages)):
            assert percentages[i] >= percentages[i - 1], (
                f"Percentage decreased at index {i}: {percentages[i-1]} -> {percentages[i]}"
            )

        # Final should be 100%.
        assert percentages[-1] == 100.0


class TestDynamicUniverse:
    """Test dynamic universe integration with presets, watchlists, and override."""

    @pytest.mark.asyncio
    async def test_universe_override_bypasses_dynamic_universe(self, orchestrator):
        """When universe_override is provided, get_scan_universe is not called."""
        override = ["TSLA", "NVDA"]

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe") as mock_scan_univ,
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist") as mock_watchlist,
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={s: _make_ticker_data(s) for s in override}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=2),
            patch("option_alpha.scoring.composite.score_universe",
                  return_value=[_make_ticker_score(s) for s in override]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan(universe_override=override)

        # get_scan_universe and get_active_watchlist should NOT be called.
        mock_scan_univ.assert_not_called()
        mock_watchlist.assert_not_called()
        # Override tickers should be scanned.
        assert result.total_tickers_scanned == 2

    @pytest.mark.asyncio
    async def test_watchlist_tickers_merged_into_universe(self, orchestrator):
        """Active watchlist tickers should be passed as extra_tickers to get_scan_universe."""
        captured_kwargs = {}

        def mock_scan_univ(**kw):
            captured_kwargs.update(kw)
            return ["AAPL", "MSFT", "CUSTOM1"]

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe",
                  side_effect=mock_scan_univ),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist",
                  return_value=["CUSTOM1"]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={"AAPL": _make_ticker_data("AAPL"),
                                "MSFT": _make_ticker_data("MSFT"),
                                "CUSTOM1": _make_ticker_data("CUSTOM1")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=3),
            patch("option_alpha.scoring.composite.score_universe",
                  return_value=[_make_ticker_score("AAPL")]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        # Watchlist tickers should be passed to get_scan_universe.
        assert captured_kwargs["extra_tickers"] == ["CUSTOM1"]
        assert result.total_tickers_scanned == 3


class TestHealthCheckGating:
    """Test LLM health check gating in AI debate phase."""

    @pytest.mark.asyncio
    async def test_health_check_failure_skips_debates(self, orchestrator):
        """When health check fails, debates should be skipped."""
        scores = [_make_ticker_score("AAPL")]

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client") as mock_get_client,
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            # Health check returns False
            mock_client = AsyncMock()
            mock_client.health_check = AsyncMock(return_value=False)
            mock_get_client.return_value = mock_client
            mock_db.return_value = MagicMock()

            result = await orchestrator.run_scan()

        # Debates should be skipped
        assert result.top_n_debated == 0
        assert len(result.debate_results) == 0
        # DebateManager should not have been instantiated
        mock_dm_cls.assert_not_called()
        # Scores should still be present
        assert len(result.ticker_scores) == 1


class TestCheckpointPersist:
    """Test that persist phase saves with PARTIAL status."""

    @pytest.mark.asyncio
    async def test_persist_saves_with_partial_status(self, orchestrator):
        """Persist phase should save scan run with PARTIAL status."""
        scores = [_make_ticker_score("AAPL")]

        captured_scan_runs = []

        def mock_save_run(conn, scan_run):
            captured_scan_runs.append(scan_run)
            return 1

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch",
                  return_value={"AAPL": _make_ticker_data("AAPL")}),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=scores),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores",
                  side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client") as mock_get_client,
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", side_effect=mock_save_run),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_client = AsyncMock()
            mock_client.health_check = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_client
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orchestrator.run_scan()

        # Persist phase should save with PARTIAL status
        assert len(captured_scan_runs) == 1
        from option_alpha.models import ScanStatus
        assert captured_scan_runs[0].status == ScanStatus.PARTIAL


class TestDataFetchPeriodPassthrough:
    """Test that data_fetch_period setting is passed to fetch_batch."""

    @pytest.mark.asyncio
    async def test_data_fetch_period_passed_to_fetch_batch(self, tmp_path):
        """fetch_batch should receive period from settings.data_fetch_period."""
        settings = Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "data" / "test.db",
            top_n_options=1,
            top_n_ai_debate=1,
            ai_backend="ollama",
            data_fetch_period="2y",  # Non-default period
        )
        orch = ScanOrchestrator(settings=settings)

        captured_kwargs = {}

        def mock_fetch(symbols, **kw):
            captured_kwargs.update(kw)
            return {"AAPL": _make_ticker_data("AAPL")}

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", side_effect=mock_fetch),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=[_make_ticker_score("AAPL")]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orch.run_scan()

        assert captured_kwargs["period"] == "2y"

    @pytest.mark.asyncio
    async def test_default_data_fetch_period_is_1y(self, tmp_path):
        """Default settings should pass period='1y' to fetch_batch."""
        settings = Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "data" / "test.db",
            top_n_options=1,
            top_n_ai_debate=1,
            ai_backend="ollama",
        )
        orch = ScanOrchestrator(settings=settings)

        captured_kwargs = {}

        def mock_fetch(symbols, **kw):
            captured_kwargs.update(kw)
            return {"AAPL": _make_ticker_data("AAPL")}

        with (
            patch("option_alpha.pipeline.orchestrator.get_scan_universe", return_value=["AAPL"]),
            patch("option_alpha.pipeline.orchestrator.get_active_watchlist", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.load_batch", return_value={}),
            patch("option_alpha.pipeline.orchestrator.fetch_batch", side_effect=mock_fetch),
            patch("option_alpha.pipeline.orchestrator.save_batch", return_value=1),
            patch("option_alpha.scoring.composite.score_universe", return_value=[_make_ticker_score("AAPL")]),
            patch("option_alpha.pipeline.orchestrator.batch_earnings_info", return_value={}),
            patch("option_alpha.pipeline.orchestrator.merge_catalyst_scores", side_effect=lambda ts, ei, **kw: ts),
            patch("option_alpha.pipeline.orchestrator.fetch_chains_for_tickers", return_value={}),
            patch("option_alpha.pipeline.orchestrator.recommend_for_scored_tickers", return_value=[]),
            patch("option_alpha.pipeline.orchestrator.get_client", return_value=_make_async_client_mock()),
            patch("option_alpha.pipeline.orchestrator.DebateManager") as mock_dm_cls,
            patch("option_alpha.pipeline.orchestrator.initialize_db") as mock_db,
            patch("option_alpha.pipeline.orchestrator.save_scan_run", return_value=1),
            patch("option_alpha.pipeline.orchestrator.save_ticker_scores"),
            patch("option_alpha.pipeline.orchestrator.save_ai_theses"),
            patch("option_alpha.pipeline.orchestrator.update_scan_run"),
        ):
            mock_dm = AsyncMock()
            mock_dm.run_debates = AsyncMock(return_value=[])
            mock_dm_cls.return_value = mock_dm
            mock_db.return_value = MagicMock()

            await orch.run_scan()

        assert captured_kwargs["period"] == "1y"


class TestAgentRetryHelper:
    """Test the shared agent retry helper."""

    @pytest.mark.asyncio
    async def test_retry_sleeps_on_network_error(self):
        """Retry helper should sleep with configured delays on network errors."""
        import httpx
        from option_alpha.ai.agents import _run_agent_with_retry
        from option_alpha.models import AgentResponse

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )

        with patch("option_alpha.ai.agents.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(httpx.TimeoutException):
                await _run_agent_with_retry(
                    mock_client,
                    [{"role": "user", "content": "test"}],
                    AgentResponse,
                    "test",
                    retry_delays=[1.0, 2.0, 4.0],
                )

        # Should have slept twice (not after last attempt)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @pytest.mark.asyncio
    async def test_no_sleep_on_parse_error(self):
        """Retry helper should NOT sleep on parse errors (immediate retry)."""
        import json
        from option_alpha.ai.agents import _run_agent_with_retry
        from option_alpha.models import AgentResponse

        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=json.JSONDecodeError("bad json", "", 0)
        )

        with patch("option_alpha.ai.agents.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(json.JSONDecodeError):
                await _run_agent_with_retry(
                    mock_client,
                    [{"role": "user", "content": "test"}],
                    AgentResponse,
                    "test",
                    retry_delays=[1.0, 2.0, 4.0],
                )

        # Should NOT have slept (parse errors retry immediately)
        mock_sleep.assert_not_called()
