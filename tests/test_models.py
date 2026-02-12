"""Tests for Pydantic models."""

from datetime import datetime

from option_alpha.models import (
    AgentResponse,
    DebateResult,
    Direction,
    OptionsRecommendation,
    ScanResult,
    ScanRun,
    ScanStatus,
    ScoreBreakdown,
    TickerData,
    TickerScore,
    TradeThesis,
)


class TestTickerData:
    def test_create_minimal(self):
        td = TickerData(symbol="AAPL")
        assert td.symbol == "AAPL"
        assert td.dates == []
        assert td.close == []

    def test_create_with_data(self):
        td = TickerData(
            symbol="AAPL",
            close=[150.0, 151.0, 152.0],
            volume=[1000000, 1100000, 1200000],
            last_price=152.0,
            avg_volume=1100000.0,
        )
        assert td.last_price == 152.0
        assert len(td.close) == 3

    def test_serialization_roundtrip(self):
        td = TickerData(
            symbol="MSFT",
            close=[300.0],
            volume=[500000],
            last_price=300.0,
        )
        data = td.model_dump()
        td2 = TickerData(**data)
        assert td2.symbol == td.symbol
        assert td2.last_price == td.last_price


class TestScoreBreakdown:
    def test_valid_breakdown(self):
        sb = ScoreBreakdown(
            name="bb_width",
            raw_value=0.05,
            normalized=75.0,
            weight=0.2,
            contribution=15.0,
        )
        assert sb.contribution == 15.0

    def test_normalized_bounds(self):
        """normalized must be 0-100."""
        import pytest

        with pytest.raises(Exception):
            ScoreBreakdown(
                name="test",
                raw_value=1.0,
                normalized=150.0,  # out of range
                weight=0.1,
                contribution=15.0,
            )


class TestTickerScore:
    def test_create_with_breakdown(self):
        breakdown = ScoreBreakdown(
            name="bb_width",
            raw_value=0.05,
            normalized=75.0,
            weight=0.2,
            contribution=15.0,
        )
        ts = TickerScore(
            symbol="AAPL",
            composite_score=75.0,
            breakdown=[breakdown],
            direction=Direction.BULLISH,
        )
        assert ts.composite_score == 75.0
        assert ts.direction == Direction.BULLISH
        assert len(ts.breakdown) == 1


class TestTradeThesis:
    def test_create(self):
        tt = TradeThesis(
            symbol="TSLA",
            direction=Direction.BULLISH,
            conviction=8,
            entry_rationale="Strong momentum with squeeze breakout",
            risk_factors=["High IV", "Earnings in 3 days"],
            recommended_action="Buy TSLA 250C 45DTE",
        )
        assert tt.conviction == 8
        assert len(tt.risk_factors) == 2


class TestAgentResponse:
    def test_create(self):
        ar = AgentResponse(
            role="bull",
            analysis="Strong uptrend with volume confirmation",
            key_points=["Breakout above resistance", "Rising OBV"],
            conviction=7,
        )
        assert ar.role == "bull"
        assert len(ar.key_points) == 2


class TestDebateResult:
    def test_create(self):
        bull = AgentResponse(role="bull", analysis="Bullish case", key_points=[])
        bear = AgentResponse(role="bear", analysis="Bearish case", key_points=[])
        risk = AgentResponse(role="risk", analysis="Risk assessment", key_points=[])
        thesis = TradeThesis(
            symbol="AAPL",
            direction=Direction.BULLISH,
            conviction=7,
            entry_rationale="Test",
            recommended_action="Buy",
        )
        dr = DebateResult(
            symbol="AAPL", bull=bull, bear=bear, risk=risk, final_thesis=thesis
        )
        assert dr.symbol == "AAPL"


class TestOptionsRecommendation:
    def test_create(self):
        rec = OptionsRecommendation(
            symbol="AAPL",
            direction=Direction.BULLISH,
            option_type="call",
            strike=180.0,
            expiry=datetime(2025, 3, 21),
            dte=45,
            delta=0.35,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
        )
        assert rec.strike == 180.0
        assert rec.option_type == "call"


class TestScanResult:
    def test_empty_scan(self):
        sr = ScanResult()
        assert sr.ticker_scores == []
        assert sr.total_tickers_scanned == 0


class TestScanRun:
    def test_create(self):
        run = ScanRun(
            run_id="run-001",
            ticker_count=500,
            status=ScanStatus.RUNNING,
        )
        assert run.status == ScanStatus.RUNNING
        assert run.ticker_count == 500

    def test_completed_run(self):
        run = ScanRun(
            run_id="run-002",
            ticker_count=2800,
            duration_seconds=120.5,
            status=ScanStatus.COMPLETED,
            scores_computed=2500,
            debates_completed=10,
            options_analyzed=50,
        )
        assert run.status == ScanStatus.COMPLETED
        assert run.duration_seconds == 120.5
