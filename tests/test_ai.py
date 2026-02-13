"""Comprehensive tests for AI multi-agent debate system.

Tests cover:
- LLM client construction and interface (mocked)
- Structured output parsing
- Bull, Bear, Risk agent implementations
- Context builder output
- Debate pipeline orchestration
- Retry/fallback behavior
- Conservative defaults on failure
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from option_alpha.ai.clients import (
    ClaudeClient,
    LLMClient,
    OllamaClient,
    _build_example_hint,
    _extract_json_from_text,
    _parse_structured_output,
    get_client,
)
from option_alpha.ai.agents import (
    BEAR_SYSTEM_PROMPT,
    BULL_SYSTEM_PROMPT,
    MAX_RETRIES,
    RISK_SYSTEM_PROMPT,
    _fallback_agent_response,
    _fallback_thesis,
    _run_agent_with_retry,
    run_bear_agent,
    run_bull_agent,
    run_risk_agent,
)
from option_alpha.ai.context import _interpret_indicator, build_context
from option_alpha.ai.debate import (
    DebateManager,
    _build_risk_response,
    _fallback_debate_result,
)
from option_alpha.config import Settings
from option_alpha.models import (
    AgentError,
    AgentResponse,
    DebateResult,
    Direction,
    ErrorCategory,
    OptionsRecommendation,
    ScoreBreakdown,
    TickerScore,
    TradeThesis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ticker_score() -> TickerScore:
    """Create a sample TickerScore with full breakdown."""
    return TickerScore(
        symbol="AAPL",
        composite_score=78.5,
        direction=Direction.BULLISH,
        last_price=185.50,
        avg_volume=55_000_000,
        breakdown=[
            ScoreBreakdown(
                name="bb_width",
                raw_value=0.045,
                normalized=72.0,
                weight=0.20,
                contribution=14.4,
            ),
            ScoreBreakdown(
                name="atr_percentile",
                raw_value=0.65,
                normalized=65.0,
                weight=0.15,
                contribution=9.75,
            ),
            ScoreBreakdown(
                name="rsi",
                raw_value=58.3,
                normalized=55.0,
                weight=0.10,
                contribution=5.5,
            ),
            ScoreBreakdown(
                name="obv_trend",
                raw_value=1.2,
                normalized=80.0,
                weight=0.10,
                contribution=8.0,
            ),
            ScoreBreakdown(
                name="sma_alignment",
                raw_value=1.0,
                normalized=90.0,
                weight=0.10,
                contribution=9.0,
            ),
            ScoreBreakdown(
                name="relative_volume",
                raw_value=1.35,
                normalized=70.0,
                weight=0.10,
                contribution=7.0,
            ),
            ScoreBreakdown(
                name="catalyst_proximity",
                raw_value=0.85,
                normalized=85.0,
                weight=0.25,
                contribution=21.25,
            ),
        ],
    )


@pytest.fixture
def sample_options_rec() -> OptionsRecommendation:
    """Create a sample OptionsRecommendation."""
    return OptionsRecommendation(
        symbol="AAPL",
        contract_symbol="AAPL240315C00185000",
        direction=Direction.BULLISH,
        option_type="call",
        strike=185.0,
        expiry=datetime(2024, 3, 15, tzinfo=UTC),
        dte=45,
        delta=0.3512,
        gamma=0.0234,
        theta=-0.0856,
        vega=0.2345,
        implied_volatility=0.2850,
        bid=5.20,
        ask=5.40,
        mid_price=5.30,
        open_interest=12500,
        volume=3400,
        underlying_price=185.50,
    )


@pytest.fixture
def mock_bull_response() -> AgentResponse:
    """Bull agent response for testing."""
    return AgentResponse(
        role="bull",
        analysis=(
            "AAPL shows strong bullish signals with SMA alignment at 90th "
            "percentile and OBV trend confirming buying pressure."
        ),
        key_points=[
            "SMA alignment at 90th percentile signals strong uptrend",
            "OBV trend confirms institutional accumulation",
            "Earnings catalyst approaching with high proximity score",
        ],
        conviction=7,
    )


@pytest.fixture
def mock_bear_response() -> AgentResponse:
    """Bear agent response for testing."""
    return AgentResponse(
        role="bear",
        analysis=(
            "Despite bullish technical signals, RSI is neutral at 55th "
            "percentile and Bollinger Band width suggests compressed volatility."
        ),
        key_points=[
            "RSI neutral suggesting limited upside momentum",
            "BB width at 72nd percentile — potential volatility contraction",
            "Elevated option premium due to upcoming earnings",
        ],
        conviction=5,
    )


@pytest.fixture
def mock_thesis() -> TradeThesis:
    """Risk agent thesis for testing."""
    return TradeThesis(
        symbol="AAPL",
        direction=Direction.BULLISH,
        conviction=6,
        entry_rationale=(
            "Moderate bullish bias supported by strong trend alignment "
            "and volume confirmation, tempered by neutral momentum."
        ),
        risk_factors=[
            "Earnings volatility could move against position",
            "RSI showing neutral momentum",
        ],
        recommended_action="Buy AAPL 185C 45DTE",
    )


class MockLLMClient(LLMClient):
    """Mock LLM client that returns pre-configured responses."""

    def __init__(self, responses: list[Any] | None = None):
        self._responses = responses or []
        self._call_count = 0

    async def complete(self, messages, response_model=None):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, Exception):
                raise resp
            return resp
        raise RuntimeError("No more mock responses")

    async def health_check(self) -> bool:
        return True


class FailingLLMClient(LLMClient):
    """Mock LLM client that always raises errors."""

    def __init__(self, error: Exception | None = None):
        self._error = error or RuntimeError("LLM unavailable")
        self.call_count = 0

    async def complete(self, messages, response_model=None):
        self.call_count += 1
        raise self._error

    async def health_check(self) -> bool:
        return False


# ===========================================================================
# Test: JSON extraction and structured output parsing
# ===========================================================================


class TestJsonExtraction:
    """Tests for _extract_json_from_text."""

    def test_plain_json(self):
        text = '{"role": "bull", "analysis": "good", "key_points": []}'
        result = _extract_json_from_text(text)
        assert json.loads(result)["role"] == "bull"

    def test_json_in_code_fence(self):
        text = 'Here is the result:\n```json\n{"role": "bear"}\n```\nDone.'
        result = _extract_json_from_text(text)
        assert json.loads(result)["role"] == "bear"

    def test_json_in_plain_fence(self):
        text = 'Result:\n```\n{"role": "risk"}\n```'
        result = _extract_json_from_text(text)
        assert json.loads(result)["role"] == "risk"

    def test_json_with_surrounding_text(self):
        text = 'The analysis shows: {"role": "bull", "analysis": "up", "key_points": []} end.'
        result = _extract_json_from_text(text)
        parsed = json.loads(result)
        assert parsed["role"] == "bull"

    def test_nested_json(self):
        text = '{"outer": {"inner": "value"}, "key": "val"}'
        result = _extract_json_from_text(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"


class TestStructuredOutputParsing:
    """Tests for _parse_structured_output."""

    def test_parse_agent_response(self):
        text = json.dumps({
            "role": "bull",
            "analysis": "Looks good",
            "key_points": ["point1", "point2"],
        })
        result = _parse_structured_output(text, AgentResponse)
        assert isinstance(result, AgentResponse)
        assert result.role == "bull"
        assert len(result.key_points) == 2

    def test_parse_trade_thesis(self):
        text = json.dumps({
            "symbol": "AAPL",
            "direction": "bullish",
            "conviction": 7,
            "entry_rationale": "Strong signals",
            "risk_factors": ["earnings risk"],
            "recommended_action": "Buy call",
        })
        result = _parse_structured_output(text, TradeThesis)
        assert isinstance(result, TradeThesis)
        assert result.conviction == 7
        assert result.direction == Direction.BULLISH

    def test_parse_with_code_fence(self):
        text = '```json\n{"role": "bear", "analysis": "Risky", "key_points": []}\n```'
        result = _parse_structured_output(text, AgentResponse)
        assert result.role == "bear"

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_structured_output("not json at all", AgentResponse)


# ===========================================================================
# Test: Client construction
# ===========================================================================


class TestOllamaClient:
    """Tests for OllamaClient construction and configuration."""

    def test_default_construction(self):
        client = OllamaClient()
        assert client.model == "llama3.1:8b"
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 120.0

    def test_custom_construction(self):
        client = OllamaClient(
            model="mistral:7b",
            base_url="http://gpu-server:11434",
            timeout=60.0,
        )
        assert client.model == "mistral:7b"
        assert client.base_url == "http://gpu-server:11434"
        assert client.timeout == 60.0

    def test_trailing_slash_stripped(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"


class TestClaudeClient:
    """Tests for ClaudeClient construction and configuration."""

    def test_construction_with_key(self):
        client = ClaudeClient(api_key="sk-ant-test-key")
        assert client.api_key == "sk-ant-test-key"
        assert client.model == ClaudeClient.DEFAULT_MODEL

    def test_custom_model(self):
        client = ClaudeClient(api_key="sk-test", model="claude-opus-4-20250514")
        assert client.model == "claude-opus-4-20250514"

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="API key is required"):
            ClaudeClient(api_key="")

    def test_none_key_raises(self):
        with pytest.raises(ValueError, match="API key is required"):
            ClaudeClient(api_key=None)


class TestGetClient:
    """Tests for the get_client factory function."""

    def test_default_ollama(self):
        config = Settings()
        client = get_client(config)
        assert isinstance(client, OllamaClient)
        assert client.model == "llama3.1:8b"

    def test_ollama_custom_model(self):
        config = Settings(ollama_model="mistral:7b")
        client = get_client(config)
        assert isinstance(client, OllamaClient)
        assert client.model == "mistral:7b"

    def test_claude_with_key(self):
        config = Settings(ai_backend="claude", claude_api_key="sk-ant-test")
        client = get_client(config)
        assert isinstance(client, ClaudeClient)

    def test_claude_without_key_raises(self):
        config = Settings(ai_backend="claude", claude_api_key=None)
        with pytest.raises(ValueError, match="no API key"):
            get_client(config)

    def test_ollama_passes_health_check_timeout(self):
        config = Settings(ai_health_check_timeout=25)
        client = get_client(config)
        assert isinstance(client, OllamaClient)
        assert client._health_timeout == 25

    def test_claude_passes_health_check_timeout(self):
        config = Settings(ai_backend="claude", claude_api_key="sk-test", ai_health_check_timeout=30)
        client = get_client(config)
        assert isinstance(client, ClaudeClient)
        assert client._health_timeout == 30


# ===========================================================================
# Test: Context builder
# ===========================================================================


class TestContextBuilder:
    """Tests for build_context."""

    def test_basic_context_output(self, sample_ticker_score):
        ctx = build_context(sample_ticker_score)
        assert "AAPL" in ctx
        assert "78.5" in ctx
        assert "BULLISH" in ctx

    def test_includes_score_breakdown(self, sample_ticker_score):
        ctx = build_context(sample_ticker_score)
        assert "bb_width" in ctx
        assert "atr_percentile" in ctx
        assert "rsi" in ctx
        assert "SCORE BREAKDOWN" in ctx

    def test_includes_price_and_volume(self, sample_ticker_score):
        ctx = build_context(sample_ticker_score)
        assert "185.50" in ctx
        assert "55,000,000" in ctx

    def test_includes_signal_summary(self, sample_ticker_score):
        ctx = build_context(sample_ticker_score)
        assert "SIGNAL SUMMARY" in ctx
        assert "bullish" in ctx.lower()

    def test_includes_catalyst_info(self, sample_ticker_score):
        ctx = build_context(sample_ticker_score)
        assert "CATALYST" in ctx
        assert "catalyst_proximity" in ctx or "proximity" in ctx.lower()

    def test_includes_options_recommendation(
        self, sample_ticker_score, sample_options_rec
    ):
        ctx = build_context(sample_ticker_score, sample_options_rec)
        assert "OPTIONS RECOMMENDATION" in ctx
        assert "CALL" in ctx
        assert "185.00" in ctx
        assert "Delta=" in ctx
        assert "Gamma=" in ctx
        assert "Theta=" in ctx
        assert "Vega=" in ctx

    def test_no_options_rec(self, sample_ticker_score):
        ctx = build_context(sample_ticker_score, None)
        assert "OPTIONS RECOMMENDATION" not in ctx

    def test_context_reasonable_length(
        self, sample_ticker_score, sample_options_rec
    ):
        """Context should be roughly 2500-3000 tokens, under 4000 max."""
        ctx = build_context(sample_ticker_score, sample_options_rec)
        # Character-based check: under 16000 chars
        assert len(ctx) < 16000, f"Context too long: {len(ctx)} chars"
        word_count = len(ctx.split())
        # Rough token estimation: ~1.3 tokens per word
        estimated_tokens = word_count * 1.3
        assert estimated_tokens < 4000, f"Context too long: ~{estimated_tokens:.0f} tokens"
        assert estimated_tokens > 150, f"Context too short: ~{estimated_tokens:.0f} tokens"

    def test_empty_breakdown(self):
        score = TickerScore(
            symbol="XYZ",
            composite_score=50.0,
            direction=Direction.NEUTRAL,
            breakdown=[],
        )
        ctx = build_context(score)
        assert "XYZ" in ctx
        assert "50.0" in ctx
        assert "No detailed breakdown" in ctx

    def test_options_without_greeks(self, sample_ticker_score):
        rec = OptionsRecommendation(
            symbol="AAPL",
            direction=Direction.BULLISH,
            option_type="call",
            strike=185.0,
            expiry=datetime(2024, 3, 15, tzinfo=UTC),
            dte=45,
        )
        ctx = build_context(sample_ticker_score, rec)
        assert "OPTIONS RECOMMENDATION" in ctx
        assert "CALL" in ctx

    def test_includes_interpretation_column(self, sample_ticker_score):
        """Score breakdown table should include Interpretation column header."""
        ctx = build_context(sample_ticker_score)
        assert "Interpretation" in ctx

    def test_includes_options_flow_section(
        self, sample_ticker_score, sample_options_rec
    ):
        """Context with options_rec should include OPTIONS FLOW section."""
        ctx = build_context(sample_ticker_score, sample_options_rec)
        assert "OPTIONS FLOW:" in ctx
        assert "Direction: CALL" in ctx
        assert "bullish alignment" in ctx

    def test_no_options_flow_without_rec(self, sample_ticker_score):
        """Context without options_rec should not include OPTIONS FLOW."""
        ctx = build_context(sample_ticker_score)
        assert "OPTIONS FLOW:" not in ctx

    def test_includes_risk_parameters_with_atr(self):
        """Context should include RISK PARAMETERS when atr_percent is in breakdown."""
        score = TickerScore(
            symbol="TEST",
            composite_score=65.0,
            direction=Direction.BULLISH,
            last_price=180.0,
            breakdown=[
                ScoreBreakdown(
                    name="atr_percent",
                    raw_value=3.2,
                    normalized=60.0,
                    weight=0.10,
                    contribution=6.0,
                ),
                ScoreBreakdown(
                    name="rsi",
                    raw_value=55.0,
                    normalized=55.0,
                    weight=0.10,
                    contribution=5.5,
                ),
            ],
        )
        ctx = build_context(score)
        assert "RISK PARAMETERS:" in ctx
        assert "ATR-based stop distance: 3.2%" in ctx
        assert "Suggested stop (long): $174.24" in ctx
        assert "Suggested stop (short): $185.76" in ctx

    def test_no_risk_parameters_without_atr(self, sample_ticker_score):
        """Context should not include RISK PARAMETERS when atr_percent is absent."""
        ctx = build_context(sample_ticker_score)
        assert "RISK PARAMETERS:" not in ctx

    def test_sector_included_when_provided(self, sample_ticker_score):
        """Context should include SECTOR line when sector parameter is given."""
        ctx = build_context(sample_ticker_score, sector="Technology")
        assert "SECTOR: Technology" in ctx

    def test_sector_not_included_when_none(self, sample_ticker_score):
        """Context should NOT include SECTOR line when sector is None."""
        ctx = build_context(sample_ticker_score)
        assert "SECTOR:" not in ctx

    def test_context_length_under_16000_chars(self, sample_ticker_score, sample_options_rec):
        """Full context with all sections should stay under 16000 characters."""
        # Create a score with atr_percent for risk params section
        score = TickerScore(
            symbol="AAPL",
            composite_score=78.5,
            direction=Direction.BULLISH,
            last_price=185.50,
            avg_volume=55_000_000,
            breakdown=[
                ScoreBreakdown(name="bb_width", raw_value=0.045, normalized=72.0, weight=0.20, contribution=14.4),
                ScoreBreakdown(name="atr_percent", raw_value=2.8, normalized=65.0, weight=0.15, contribution=9.75),
                ScoreBreakdown(name="rsi", raw_value=58.3, normalized=55.0, weight=0.10, contribution=5.5),
                ScoreBreakdown(name="obv_trend", raw_value=1.2, normalized=80.0, weight=0.10, contribution=8.0),
                ScoreBreakdown(name="sma_alignment", raw_value=85.0, normalized=90.0, weight=0.10, contribution=9.0),
                ScoreBreakdown(name="relative_volume", raw_value=1.35, normalized=70.0, weight=0.10, contribution=7.0),
                ScoreBreakdown(name="catalyst_proximity", raw_value=0.85, normalized=85.0, weight=0.25, contribution=21.25),
                ScoreBreakdown(name="adx", raw_value=32.4, normalized=72.0, weight=0.08, contribution=5.76),
                ScoreBreakdown(name="stoch_rsi", raw_value=45.0, normalized=50.0, weight=0.05, contribution=2.5),
                ScoreBreakdown(name="williams_r", raw_value=-55.0, normalized=50.0, weight=0.05, contribution=2.5),
                ScoreBreakdown(name="roc", raw_value=2.1, normalized=60.0, weight=0.05, contribution=3.0),
                ScoreBreakdown(name="supertrend", raw_value=1.0, normalized=80.0, weight=0.05, contribution=4.0),
            ],
        )
        ctx = build_context(score, sample_options_rec, sector="Technology")
        assert len(ctx) < 16000, f"Context too long: {len(ctx)} chars"


# ===========================================================================
# Test: Interpret indicator
# ===========================================================================


class TestInterpretIndicator:
    """Tests for _interpret_indicator helper."""

    def test_adx_weak(self):
        assert _interpret_indicator("adx", 15.0) == "weak trend"

    def test_adx_developing(self):
        assert _interpret_indicator("adx", 22.0) == "developing"

    def test_adx_moderate(self):
        assert _interpret_indicator("adx", 35.0) == "moderate trend"

    def test_adx_strong(self):
        assert _interpret_indicator("adx", 60.0) == "strong trend"

    def test_adx_extreme(self):
        assert _interpret_indicator("adx", 80.0) == "extreme"

    def test_rsi_oversold(self):
        assert _interpret_indicator("rsi", 25.0) == "oversold"

    def test_rsi_bearish_momentum(self):
        assert _interpret_indicator("rsi", 40.0) == "bearish momentum"

    def test_rsi_bullish_momentum(self):
        assert _interpret_indicator("rsi", 58.0) == "bullish momentum"

    def test_rsi_overbought(self):
        assert _interpret_indicator("rsi", 75.0) == "overbought"

    def test_stoch_rsi_oversold(self):
        assert _interpret_indicator("stoch_rsi", 10.0) == "oversold"

    def test_stoch_rsi_overbought(self):
        assert _interpret_indicator("stoch_rsi", 85.0) == "overbought"

    def test_stoch_rsi_neutral(self):
        assert _interpret_indicator("stoch_rsi", 50.0) == "neutral"

    def test_williams_r_oversold(self):
        assert _interpret_indicator("williams_r", -90.0) == "oversold"

    def test_williams_r_overbought(self):
        assert _interpret_indicator("williams_r", -10.0) == "overbought"

    def test_williams_r_neutral(self):
        assert _interpret_indicator("williams_r", -50.0) == "neutral"

    def test_roc_positive(self):
        assert _interpret_indicator("roc", 2.5) == "positive momentum"

    def test_roc_negative(self):
        assert _interpret_indicator("roc", -1.5) == "negative momentum"

    def test_roc_flat(self):
        assert _interpret_indicator("roc", 0.0) == "flat"

    def test_relative_volume_very_quiet(self):
        assert _interpret_indicator("relative_volume", 0.3) == "very quiet"

    def test_relative_volume_below_average(self):
        assert _interpret_indicator("relative_volume", 0.7) == "below average"

    def test_relative_volume_normal(self):
        assert _interpret_indicator("relative_volume", 1.0) == "normal"

    def test_relative_volume_elevated(self):
        assert _interpret_indicator("relative_volume", 1.5) == "elevated"

    def test_relative_volume_surge(self):
        assert _interpret_indicator("relative_volume", 3.0) == "volume surge"

    def test_bb_width_tight_squeeze(self):
        assert _interpret_indicator("bb_width", 0.03) == "tight squeeze"

    def test_bb_width_moderate(self):
        assert _interpret_indicator("bb_width", 0.07) == "moderate"

    def test_bb_width_wide(self):
        assert _interpret_indicator("bb_width", 0.15) == "wide"

    def test_keltner_width_tight_squeeze(self):
        assert _interpret_indicator("keltner_width", 0.03) == "tight squeeze"

    def test_sma_alignment_tight(self):
        assert _interpret_indicator("sma_alignment", 85.0) == "tight alignment"

    def test_sma_alignment_moderate(self):
        assert _interpret_indicator("sma_alignment", 65.0) == "moderate spread"

    def test_sma_alignment_wide(self):
        assert _interpret_indicator("sma_alignment", 30.0) == "wide spread"

    def test_supertrend_bullish(self):
        assert _interpret_indicator("supertrend", 1.0) == "bullish"

    def test_supertrend_bearish(self):
        assert _interpret_indicator("supertrend", -1.0) == "bearish"

    def test_vwap_above(self):
        assert _interpret_indicator("vwap_deviation", 2.0) == "above VWAP"

    def test_vwap_below(self):
        assert _interpret_indicator("vwap_deviation", -2.0) == "below VWAP"

    def test_vwap_near(self):
        assert _interpret_indicator("vwap_deviation", 0.5) == "near VWAP"

    def test_obv_trend_accumulating(self):
        assert _interpret_indicator("obv_trend", 1.5) == "accumulating"

    def test_obv_trend_distributing(self):
        assert _interpret_indicator("obv_trend", -0.8) == "distributing"

    def test_obv_trend_neutral(self):
        assert _interpret_indicator("obv_trend", 0.0) == "neutral"

    def test_ad_trend_accumulating(self):
        assert _interpret_indicator("ad_trend", 1.0) == "accumulating"

    def test_ad_trend_distributing(self):
        assert _interpret_indicator("ad_trend", -1.0) == "distributing"

    def test_atr_percent_format(self):
        assert _interpret_indicator("atr_percent", 3.2) == "3.2% daily range"

    def test_nan_returns_na(self):
        assert _interpret_indicator("rsi", float("nan")) == "N/A"

    def test_unknown_indicator_returns_formatted(self):
        assert _interpret_indicator("unknown_thing", 42.5) == "42.50"


# ===========================================================================
# Test: Bull Agent
# ===========================================================================


class TestBullAgent:
    """Tests for run_bull_agent."""

    @pytest.mark.asyncio
    async def test_successful_bull_analysis(self, mock_bull_response):
        client = MockLLMClient([mock_bull_response])
        result = await run_bull_agent("test context", client)
        assert result.role == "bull"
        assert len(result.key_points) > 0
        assert "SMA" in result.analysis or len(result.analysis) > 0

    @pytest.mark.asyncio
    async def test_bull_role_forced(self):
        """Even if LLM returns wrong role, it gets corrected."""
        wrong_role = AgentResponse(
            role="bear",  # Wrong role
            analysis="Analysis text",
            key_points=["point"],
        )
        client = MockLLMClient([wrong_role])
        result = await run_bull_agent("context", client)
        assert result.role == "bull"

    @pytest.mark.asyncio
    async def test_bull_retries_on_failure(self):
        """Should retry and eventually succeed."""
        good_response = AgentResponse(
            role="bull", analysis="Good analysis", key_points=["point"]
        )
        client = MockLLMClient([
            RuntimeError("timeout"),
            RuntimeError("timeout"),
            good_response,
        ])
        result = await run_bull_agent("context", client)
        assert result.role == "bull"
        assert result.analysis == "Good analysis"

    @pytest.mark.asyncio
    async def test_bull_fallback_on_total_failure(self):
        """Should return conservative fallback after all retries fail."""
        client = FailingLLMClient()
        result = await run_bull_agent("context", client)
        assert result.role == "bull"
        assert "unavailable" in result.analysis.lower() or "failed" in result.analysis.lower()
        assert client.call_count == MAX_RETRIES


# ===========================================================================
# Test: Bear Agent
# ===========================================================================


class TestBearAgent:
    """Tests for run_bear_agent."""

    @pytest.mark.asyncio
    async def test_successful_bear_analysis(
        self, mock_bull_response, mock_bear_response
    ):
        client = MockLLMClient([mock_bear_response])
        result = await run_bear_agent("test context", mock_bull_response, client)
        assert result.role == "bear"
        assert len(result.key_points) > 0

    @pytest.mark.asyncio
    async def test_bear_role_forced(self, mock_bull_response):
        wrong_role = AgentResponse(
            role="bull", analysis="Counter args", key_points=["contra"]
        )
        client = MockLLMClient([wrong_role])
        result = await run_bear_agent("context", mock_bull_response, client)
        assert result.role == "bear"

    @pytest.mark.asyncio
    async def test_bear_fallback_on_total_failure(self, mock_bull_response):
        client = FailingLLMClient()
        result = await run_bear_agent("context", mock_bull_response, client)
        assert result.role == "bear"
        assert "unavailable" in result.analysis.lower() or "failed" in result.analysis.lower()
        assert client.call_count == MAX_RETRIES


# ===========================================================================
# Test: Risk Agent
# ===========================================================================


class TestRiskAgent:
    """Tests for run_risk_agent."""

    @pytest.mark.asyncio
    async def test_successful_risk_synthesis(
        self, mock_bull_response, mock_bear_response, mock_thesis
    ):
        client = MockLLMClient([mock_thesis])
        result = await run_risk_agent(
            "context", mock_bull_response, mock_bear_response, "AAPL", client
        )
        assert isinstance(result, TradeThesis)
        assert result.symbol == "AAPL"
        assert 1 <= result.conviction <= 10

    @pytest.mark.asyncio
    async def test_risk_symbol_forced(
        self, mock_bull_response, mock_bear_response
    ):
        """Symbol should be set even if LLM returns wrong one."""
        wrong_symbol = TradeThesis(
            symbol="WRONG",
            direction=Direction.BULLISH,
            conviction=5,
            entry_rationale="test",
            risk_factors=[],
            recommended_action="Buy",
        )
        client = MockLLMClient([wrong_symbol])
        result = await run_risk_agent(
            "context", mock_bull_response, mock_bear_response, "AAPL", client
        )
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_risk_fallback_on_total_failure(
        self, mock_bull_response, mock_bear_response
    ):
        client = FailingLLMClient()
        result = await run_risk_agent(
            "context", mock_bull_response, mock_bear_response, "AAPL", client
        )
        assert isinstance(result, TradeThesis)
        assert result.symbol == "AAPL"
        assert result.direction == Direction.NEUTRAL
        assert result.conviction == 3
        assert result.recommended_action == "No trade"
        assert client.call_count == MAX_RETRIES

    @pytest.mark.asyncio
    async def test_risk_retries_then_succeeds(
        self, mock_bull_response, mock_bear_response, mock_thesis
    ):
        client = MockLLMClient([
            ValueError("bad json"),
            mock_thesis,
        ])
        result = await run_risk_agent(
            "context", mock_bull_response, mock_bear_response, "AAPL", client
        )
        assert isinstance(result, TradeThesis)
        assert result.conviction == 6


# ===========================================================================
# Test: Fallback defaults
# ===========================================================================


class TestFallbackDefaults:
    """Tests for conservative fallback behavior."""

    def test_fallback_agent_response(self):
        resp = _fallback_agent_response("bull")
        assert resp.role == "bull"
        assert "failed" in resp.analysis.lower() or "unavailable" in resp.analysis.lower()
        assert resp.conviction == 3

    def test_fallback_thesis(self):
        thesis = _fallback_thesis("TSLA")
        assert thesis.symbol == "TSLA"
        assert thesis.direction == Direction.NEUTRAL
        assert thesis.conviction == 3
        assert thesis.recommended_action == "No trade"

    def test_fallback_debate_result(self, sample_ticker_score):
        result = _fallback_debate_result(sample_ticker_score)
        assert result.symbol == "AAPL"
        assert result.bull.role == "bull"
        assert result.bear.role == "bear"
        assert result.risk.role == "risk"
        # Context-aware fallback derives direction/conviction from ticker_score
        assert result.final_thesis.direction == sample_ticker_score.direction
        expected_conviction = max(2, min(8, round(sample_ticker_score.composite_score / 12.5)))
        assert result.final_thesis.conviction == expected_conviction

    def test_fallback_agent_response_has_fallback_prefix(self):
        """Fallback agent responses should start with [FALLBACK]."""
        for role in ("bull", "bear", "risk"):
            resp = _fallback_agent_response(role)
            assert resp.analysis.startswith("[FALLBACK]"), (
                f"Fallback for {role} does not start with [FALLBACK]: {resp.analysis}"
            )

    def test_fallback_thesis_has_fallback_prefix(self):
        """Fallback thesis entry_rationale should start with [FALLBACK]."""
        thesis = _fallback_thesis("AAPL")
        assert thesis.entry_rationale.startswith("[FALLBACK]"), (
            f"Fallback thesis does not start with [FALLBACK]: {thesis.entry_rationale}"
        )


# ===========================================================================
# Test: Risk System Prompt
# ===========================================================================


class TestRiskSystemPrompt:
    """Tests for the rebalanced RISK_SYSTEM_PROMPT."""

    def test_no_be_conservative(self):
        """RISK_SYSTEM_PROMPT should NOT contain 'Be conservative'."""
        assert "Be conservative" not in RISK_SYSTEM_PROMPT

    def test_contains_conviction_rubric(self):
        """RISK_SYSTEM_PROMPT should contain 'Conviction rubric'."""
        assert "Conviction rubric" in RISK_SYSTEM_PROMPT

    def test_contains_direction_signal_guidance(self):
        """RISK_SYSTEM_PROMPT should reference the pre-computed direction signal."""
        assert "DIRECTION SIGNAL" in RISK_SYSTEM_PROMPT

    def test_contains_conviction_scale(self):
        """RISK_SYSTEM_PROMPT should describe the 1-10 conviction scale."""
        assert "1-10" in RISK_SYSTEM_PROMPT
        assert "conviction" in RISK_SYSTEM_PROMPT.lower()


# ===========================================================================
# Test: Bull and Bear System Prompts
# ===========================================================================


class TestBullSystemPrompt:
    """Tests for the updated BULL_SYSTEM_PROMPT."""

    def test_contains_price_target(self):
        """BULL_SYSTEM_PROMPT should instruct citing a price target."""
        assert "price target" in BULL_SYSTEM_PROMPT.lower()

    def test_contains_3_data_points(self):
        """BULL_SYSTEM_PROMPT should instruct citing at least 3 indicators."""
        assert "3" in BULL_SYSTEM_PROMPT

    def test_contains_data_driven(self):
        """BULL_SYSTEM_PROMPT should emphasize data-driven analysis."""
        assert "data-driven" in BULL_SYSTEM_PROMPT.lower()

    def test_contains_options_reference(self):
        """BULL_SYSTEM_PROMPT should reference options data."""
        assert "options" in BULL_SYSTEM_PROMPT.lower()


class TestBearSystemPrompt:
    """Tests for the updated BEAR_SYSTEM_PROMPT."""

    def test_contains_downside(self):
        """BEAR_SYSTEM_PROMPT should instruct quantifying downside."""
        assert "downside" in BEAR_SYSTEM_PROMPT.lower()

    def test_contains_weakest(self):
        """BEAR_SYSTEM_PROMPT should instruct identifying the weakest indicator."""
        assert "weakest" in BEAR_SYSTEM_PROMPT.lower()

    def test_contains_risk_scenario(self):
        """BEAR_SYSTEM_PROMPT should instruct citing risk scenarios."""
        assert "risk scenario" in BEAR_SYSTEM_PROMPT.lower()

    def test_contains_dollar_and_percentage(self):
        """BEAR_SYSTEM_PROMPT should instruct quantifying in dollar and percentage."""
        assert "dollar" in BEAR_SYSTEM_PROMPT.lower() or "$" in BEAR_SYSTEM_PROMPT
        assert "percentage" in BEAR_SYSTEM_PROMPT.lower() or "%" in BEAR_SYSTEM_PROMPT


class TestRiskSystemPromptEnhanced:
    """Tests for the enhanced RISK_SYSTEM_PROMPT output requirements."""

    def test_contains_stop_loss(self):
        """RISK_SYSTEM_PROMPT should require stop-loss output."""
        assert "stop-loss" in RISK_SYSTEM_PROMPT.lower() or "Stop-loss" in RISK_SYSTEM_PROMPT

    def test_contains_risk_reward(self):
        """RISK_SYSTEM_PROMPT should require risk/reward ratio."""
        assert "risk/reward" in RISK_SYSTEM_PROMPT.lower() or "Risk/reward" in RISK_SYSTEM_PROMPT

    def test_contains_position_sizing(self):
        """RISK_SYSTEM_PROMPT should require position sizing."""
        assert "position sizing" in RISK_SYSTEM_PROMPT.lower() or "Position sizing" in RISK_SYSTEM_PROMPT

    def test_contains_entry_price(self):
        """RISK_SYSTEM_PROMPT should require entry price."""
        assert "entry price" in RISK_SYSTEM_PROMPT.lower() or "Entry price" in RISK_SYSTEM_PROMPT

    def test_contains_iv_assessment(self):
        """RISK_SYSTEM_PROMPT should require IV assessment."""
        assert "IV assessment" in RISK_SYSTEM_PROMPT or "iv assessment" in RISK_SYSTEM_PROMPT.lower()

    def test_contains_profit_target(self):
        """RISK_SYSTEM_PROMPT should require profit target."""
        assert "profit target" in RISK_SYSTEM_PROMPT.lower() or "Profit target" in RISK_SYSTEM_PROMPT


# ===========================================================================
# Test: Debate Manager
# ===========================================================================


class TestDebateManager:
    """Tests for DebateManager orchestration."""

    @pytest.mark.asyncio
    async def test_run_single_debate(
        self,
        sample_ticker_score,
        sample_options_rec,
        mock_bull_response,
        mock_bear_response,
        mock_thesis,
    ):
        client = MockLLMClient([
            mock_bull_response,
            mock_bear_response,
            mock_thesis,
        ])
        manager = DebateManager(client)
        result = await manager.run_debate(sample_ticker_score, sample_options_rec)

        assert isinstance(result, DebateResult)
        assert result.symbol == "AAPL"
        assert result.bull.role == "bull"
        assert result.bear.role == "bear"
        assert result.risk.role == "risk"
        assert result.final_thesis.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_run_debates_top_n(
        self,
        mock_bull_response,
        mock_bear_response,
        mock_thesis,
    ):
        """Test that run_debates respects top_n and processes correctly."""
        scores = [
            TickerScore(
                symbol=f"T{i}",
                composite_score=90 - i * 10,
                direction=Direction.BULLISH,
            )
            for i in range(5)
        ]

        # 3 responses per debate * 3 top_n = 9 responses
        responses = [mock_bull_response, mock_bear_response, mock_thesis] * 3

        client = MockLLMClient(responses)
        manager = DebateManager(client)
        results = await manager.run_debates(scores, top_n=3)

        assert len(results) == 3
        # Should be in score order (highest first)
        assert results[0].symbol == "T0"
        assert results[1].symbol == "T1"
        assert results[2].symbol == "T2"

    @pytest.mark.asyncio
    async def test_run_debates_with_progress_callback(
        self,
        mock_bull_response,
        mock_bear_response,
        mock_thesis,
    ):
        scores = [
            TickerScore(
                symbol="AAPL",
                composite_score=80,
                direction=Direction.BULLISH,
            ),
        ]

        client = MockLLMClient([mock_bull_response, mock_bear_response, mock_thesis])
        manager = DebateManager(client)

        progress_calls = []

        def on_progress(completed, total, symbol):
            progress_calls.append((completed, total, symbol))

        await manager.run_debates(scores, top_n=1, progress_callback=on_progress)
        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1, "AAPL")

    @pytest.mark.asyncio
    async def test_debate_handles_individual_failure(self):
        """If one debate fails entirely, others should still complete."""
        scores = [
            TickerScore(
                symbol="AAPL", composite_score=90, direction=Direction.BULLISH
            ),
            TickerScore(
                symbol="FAIL", composite_score=80, direction=Direction.BULLISH
            ),
        ]

        good_bull = AgentResponse(
            role="bull", analysis="Good", key_points=["p1"]
        )
        good_bear = AgentResponse(
            role="bear", analysis="Bad", key_points=["p2"]
        )
        good_thesis = TradeThesis(
            symbol="AAPL",
            direction=Direction.BULLISH,
            conviction=7,
            entry_rationale="Test",
            risk_factors=[],
            recommended_action="Buy",
        )

        call_count = 0

        class SelectiveClient(LLMClient):
            async def complete(self_inner, messages, response_model=None):
                nonlocal call_count
                call_count += 1
                # First 3 calls (AAPL debate): succeed
                if call_count <= 3:
                    if call_count == 1:
                        return good_bull
                    elif call_count == 2:
                        return good_bear
                    else:
                        return good_thesis
                # All subsequent calls (FAIL debate): error
                raise RuntimeError("Connection refused")

            async def health_check(self_inner):
                return True

        manager = DebateManager(SelectiveClient())
        results = await manager.run_debates(scores, top_n=2)

        assert len(results) == 2
        # First should be real
        assert results[0].symbol == "AAPL"
        assert results[0].final_thesis.conviction == 7
        # Second should be fallback — context-aware from ticker_score
        assert results[1].symbol == "FAIL"
        assert results[1].final_thesis.direction == Direction.BULLISH
        expected_conviction = max(2, min(8, round(80 / 12.5)))
        assert results[1].final_thesis.conviction == expected_conviction

    @pytest.mark.asyncio
    async def test_run_debates_with_options_recs(
        self,
        sample_options_rec,
        mock_bull_response,
        mock_bear_response,
        mock_thesis,
    ):
        scores = [
            TickerScore(
                symbol="AAPL",
                composite_score=80,
                direction=Direction.BULLISH,
            ),
        ]
        recs = {"AAPL": sample_options_rec}

        client = MockLLMClient([mock_bull_response, mock_bear_response, mock_thesis])
        manager = DebateManager(client)
        results = await manager.run_debates(scores, options_recs=recs, top_n=1)

        assert len(results) == 1
        assert results[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_run_debates_empty_scores(self):
        client = MockLLMClient([])
        manager = DebateManager(client)
        results = await manager.run_debates([], top_n=10)
        assert results == []


# ===========================================================================
# Test: Build risk response helper
# ===========================================================================


class TestBuildRiskResponse:
    """Tests for _build_risk_response."""

    def test_converts_thesis_to_agent_response(self, mock_thesis):
        resp = _build_risk_response(mock_thesis)
        assert resp.role == "risk"
        assert resp.conviction == 6
        assert resp.analysis == mock_thesis.entry_rationale
        assert resp.key_points == mock_thesis.risk_factors


# ===========================================================================
# Test: Ollama client request format (mocked httpx)
# ===========================================================================


class TestOllamaClientRequest:
    """Tests for OllamaClient request formatting."""

    @pytest.mark.asyncio
    async def test_complete_plain_text(self):
        client = OllamaClient(model="test-model")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Hello world"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.complete([
                {"role": "user", "content": "Say hello"}
            ])

            assert result == "Hello world"
            mock_ctx.post.assert_called_once()
            call_args = mock_ctx.post.call_args
            payload = call_args[1]["json"]
            assert payload["model"] == "test-model"
            assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_complete_structured_output(self):
        client = OllamaClient()

        response_data = {
            "role": "bull",
            "analysis": "Test analysis",
            "key_points": ["p1"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": json.dumps(response_data)}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.complete(
                [{"role": "user", "content": "analyze"}],
                response_model=AgentResponse,
            )

            assert isinstance(result, AgentResponse)
            assert result.role == "bull"

            # Verify the prompt uses a concrete example hint, not a schema dump
            call_args = mock_ctx.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            last_msg = payload["messages"][-1]["content"]
            # Must NOT contain schema metadata keys
            assert '"description"' not in last_msg
            assert '"properties"' not in last_msg
            assert '"type": "object"' not in last_msg
            # Must contain concrete example with model field names
            assert "example" in last_msg.lower()
            assert '"role"' in last_msg
            assert '"analysis"' in last_msg

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Health check succeeds when both /api/tags and /api/generate respond OK."""
        client = OllamaClient()

        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.raise_for_status = MagicMock()

        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(return_value=mock_get_response)
            mock_ctx.post = AsyncMock(return_value=mock_post_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is True
            # Verify both steps were called
            mock_ctx.get.assert_called_once()
            mock_ctx.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure_connection(self):
        """Health check fails on connection error."""
        client = OllamaClient()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_failure_timeout(self):
        """Health check fails on timeout."""
        client = OllamaClient()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_model_not_loaded(self):
        """Health check fails when /api/tags OK but model generate fails."""
        client = OllamaClient()

        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.raise_for_status = MagicMock()

        mock_post_response = MagicMock()
        mock_post_response.status_code = 404
        mock_post_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_post_response
            )
        )

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(return_value=mock_get_response)
            mock_ctx.post = AsyncMock(return_value=mock_post_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_uses_configured_timeout(self):
        """Health check uses the configured health_check_timeout."""
        client = OllamaClient(health_check_timeout=42.0)
        assert client._health_timeout == 42.0

    @pytest.mark.asyncio
    async def test_health_check_sends_model_generate(self):
        """Health check sends the correct model name in the generate request."""
        client = OllamaClient(model="my-model:7b")

        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.raise_for_status = MagicMock()

        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(return_value=mock_get_response)
            mock_ctx.post = AsyncMock(return_value=mock_post_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is True

            # Verify the generate request used the correct model
            post_call = mock_ctx.post.call_args
            payload = post_call.kwargs.get("json") or post_call[1].get("json")
            assert payload["model"] == "my-model:7b"
            assert payload["stream"] is False


# ===========================================================================
# Test: ClaudeClient health check
# ===========================================================================


class TestClaudeClientHealthCheck:
    """Tests for ClaudeClient.health_check() with real API validation."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Health check succeeds when API returns 200."""
        client = ClaudeClient(api_key="sk-valid-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is True
            mock_ctx.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_invalid_api_key(self):
        """Health check fails with 401 for invalid API key."""
        client = ClaudeClient(api_key="sk-invalid")

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self):
        """Health check fails on connection error."""
        client = ClaudeClient(api_key="sk-test")

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Health check fails on timeout."""
        client = ClaudeClient(api_key="sk-test")

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_http_error(self):
        """Health check fails on non-401 HTTP error (e.g. 500)."""
        client = ClaudeClient(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response
            )
        )

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_uses_configured_timeout(self):
        """Health check uses the configured health_check_timeout."""
        client = ClaudeClient(api_key="sk-test", health_check_timeout=30.0)
        assert client._health_timeout == 30.0

    @pytest.mark.asyncio
    async def test_health_check_sends_minimal_request(self):
        """Health check sends max_tokens=1 to minimize cost."""
        client = ClaudeClient(api_key="sk-test-key", model="claude-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.health_check()

            post_call = mock_ctx.post.call_args
            payload = post_call.kwargs.get("json") or post_call[1].get("json")
            assert payload["max_tokens"] == 1
            assert payload["model"] == "claude-test"
            headers = post_call.kwargs.get("headers") or post_call[1].get("headers")
            assert headers["x-api-key"] == "sk-test-key"


# ===========================================================================
# Test: _build_example_hint helper
# ===========================================================================


class TestBuildExampleHint:
    """Tests for the example-based prompt hint builder."""

    def test_agent_response_hint(self):
        hint = _build_example_hint(AgentResponse)
        parsed = json.loads(hint.split("example:\n", 1)[1])
        assert "role" in parsed
        assert "analysis" in parsed
        assert "key_points" in parsed
        # Must NOT contain schema metadata
        assert '"description"' not in hint
        assert '"properties"' not in hint
        assert '"type": "object"' not in hint

    def test_trade_thesis_hint(self):
        hint = _build_example_hint(TradeThesis)
        parsed = json.loads(hint.split("example:\n", 1)[1])
        assert "symbol" in parsed
        assert "direction" in parsed
        assert "conviction" in parsed
        # Must NOT contain schema metadata
        assert '"description"' not in hint
        assert '"properties"' not in hint

    def test_hint_produces_valid_json(self):
        """The example in the hint must be valid JSON."""
        hint = _build_example_hint(AgentResponse)
        json_part = hint.split("example:\n", 1)[1]
        data = json.loads(json_part)
        assert isinstance(data, dict)


# ===========================================================================
# Test: Claude client request format (mocked httpx)
# ===========================================================================


class TestClaudeClientRequest:
    """Tests for ClaudeClient request formatting."""

    @pytest.mark.asyncio
    async def test_complete_plain_text(self):
        client = ClaudeClient(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello from Claude"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.complete([
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ])

            assert result == "Hello from Claude"
            call_args = mock_ctx.post.call_args
            payload = call_args[1]["json"]
            assert payload["system"] == "Be helpful"
            # System message should NOT be in messages list
            assert all(m["role"] != "system" for m in payload["messages"])

    @pytest.mark.asyncio
    async def test_complete_structured_output(self):
        client = ClaudeClient(api_key="sk-test")

        response_data = {
            "role": "bear",
            "analysis": "Bearish outlook",
            "key_points": ["risk1"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": json.dumps(response_data)}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.complete(
                [{"role": "user", "content": "analyze"}],
                response_model=AgentResponse,
            )

            assert isinstance(result, AgentResponse)
            assert result.role == "bear"

    @pytest.mark.asyncio
    async def test_headers_include_api_key(self):
        client = ClaudeClient(api_key="sk-ant-secret")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "ok"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.complete([{"role": "user", "content": "test"}])

            call_args = mock_ctx.post.call_args
            headers = call_args[1]["headers"]
            assert headers["x-api-key"] == "sk-ant-secret"
            assert headers["anthropic-version"] == "2023-06-01"


# ===========================================================================
# Test: Edge cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_context_with_no_price_no_volume(self):
        score = TickerScore(
            symbol="XYZ",
            composite_score=50.0,
            direction=Direction.NEUTRAL,
            last_price=None,
            avg_volume=None,
            breakdown=[],
        )
        ctx = build_context(score)
        assert "XYZ" in ctx
        assert "LAST PRICE" not in ctx
        assert "AVG VOLUME" not in ctx

    def test_context_with_bearish_direction(self):
        score = TickerScore(
            symbol="BEAR",
            composite_score=35.0,
            direction=Direction.BEARISH,
            breakdown=[
                ScoreBreakdown(
                    name="rsi",
                    raw_value=25.0,
                    normalized=20.0,
                    weight=0.10,
                    contribution=2.0,
                ),
            ],
        )
        ctx = build_context(score)
        assert "BEARISH" in ctx
        assert "35.0" in ctx

    @pytest.mark.asyncio
    async def test_debate_manager_sorts_by_score(
        self,
        mock_bull_response,
        mock_bear_response,
        mock_thesis,
    ):
        """Verify candidates are sorted by composite_score descending."""
        scores = [
            TickerScore(
                symbol="LOW", composite_score=30, direction=Direction.NEUTRAL
            ),
            TickerScore(
                symbol="HIGH", composite_score=90, direction=Direction.BULLISH
            ),
            TickerScore(
                symbol="MID", composite_score=60, direction=Direction.BULLISH
            ),
        ]

        responses = [mock_bull_response, mock_bear_response, mock_thesis] * 2

        client = MockLLMClient(responses)
        manager = DebateManager(client)
        results = await manager.run_debates(scores, top_n=2)

        assert len(results) == 2
        assert results[0].symbol == "HIGH"
        assert results[1].symbol == "MID"

    def test_options_rec_minimal_fields(self, sample_ticker_score):
        """Options rec with only required fields should still work."""
        rec = OptionsRecommendation(
            symbol="AAPL",
            direction=Direction.BULLISH,
            option_type="put",
            strike=180.0,
            expiry=datetime(2024, 6, 21, tzinfo=UTC),
            dte=90,
        )
        ctx = build_context(sample_ticker_score, rec)
        assert "PUT" in ctx
        assert "180.00" in ctx


# ===========================================================================
# Test: Structured Error Models (Issue #63)
# ===========================================================================


class TestStructuredErrors:
    """Tests for AgentError model and ErrorCategory enum."""

    def test_error_categories_exist(self):
        """All expected error categories exist."""
        assert ErrorCategory.NETWORK == "NETWORK"
        assert ErrorCategory.PARSE == "PARSE"
        assert ErrorCategory.VALIDATION == "VALIDATION"
        assert ErrorCategory.TIMEOUT == "TIMEOUT"
        assert ErrorCategory.UNKNOWN == "UNKNOWN"

    def test_agent_error_model_construction(self):
        """AgentError can be constructed with all fields."""
        err = AgentError(
            category=ErrorCategory.NETWORK,
            message="connection refused",
            agent_role="bull",
            attempt=2,
        )
        assert err.category == ErrorCategory.NETWORK
        assert err.message == "connection refused"
        assert err.agent_role == "bull"
        assert err.attempt == 2

    def test_debate_result_error_field(self, sample_ticker_score):
        """DebateResult can carry an error."""
        result = _fallback_debate_result(sample_ticker_score)
        result.error = AgentError(
            category=ErrorCategory.TIMEOUT,
            message="timed out",
            agent_role="debate",
            attempt=0,
        )
        assert result.error.category == ErrorCategory.TIMEOUT

    def test_debate_result_composite_score_field(self, sample_ticker_score):
        """DebateResult can carry a composite_score."""
        result = _fallback_debate_result(sample_ticker_score)
        result.composite_score = 78.5
        assert result.composite_score == 78.5

    def test_debate_result_defaults_none(self, sample_ticker_score):
        """DebateResult error and composite_score default to None."""
        result = _fallback_debate_result(sample_ticker_score)
        assert result.error is None
        assert result.composite_score is None


# ===========================================================================
# Test: Per-Ticker Timeout (Issue #69)
# ===========================================================================


class TestPerTickerTimeout:
    """Tests for per-ticker timeout in debate runner."""

    @pytest.mark.asyncio
    async def test_timeout_returns_fallback_with_error(self):
        """When a debate exceeds ai_per_ticker_timeout, a fallback with TIMEOUT error is returned."""
        score = TickerScore(
            symbol="SLOW",
            composite_score=50.0,
            direction=Direction.NEUTRAL,
            breakdown=[],
        )

        async def slow_debate(ts, opts=None):
            await asyncio.sleep(100)  # way longer than timeout

        client = MagicMock(spec=LLMClient)
        manager = DebateManager(client)
        manager.run_debate = slow_debate

        settings = Settings()
        settings.ai_per_ticker_timeout = 0.1  # very short
        settings.ai_debate_phase_timeout = 5
        settings.ai_debate_concurrency = 1

        results = await manager.run_debates(
            [score], top_n=1, settings=settings,
        )

        assert len(results) == 1
        assert results[0].symbol == "SLOW"
        assert results[0].error is not None
        assert results[0].error.category == ErrorCategory.TIMEOUT

    @pytest.mark.asyncio
    async def test_successful_debate_no_error(
        self, mock_bull_response, mock_bear_response, mock_thesis,
    ):
        """Successful debates should have no error."""
        score = TickerScore(
            symbol="FAST",
            composite_score=80.0,
            direction=Direction.BULLISH,
            breakdown=[],
        )

        responses = [mock_bull_response, mock_bear_response, mock_thesis]
        client = MockLLMClient(responses)
        manager = DebateManager(client)

        settings = Settings()
        settings.ai_per_ticker_timeout = 60
        settings.ai_debate_phase_timeout = 120
        settings.ai_debate_concurrency = 1

        results = await manager.run_debates(
            [score], top_n=1, settings=settings,
        )

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].composite_score == 80.0


# ===========================================================================
# Test: Deterministic Ordering (Issue #69)
# ===========================================================================


class TestDeterministicOrdering:
    """Tests for deterministic result ordering by composite_score."""

    @pytest.mark.asyncio
    async def test_results_sorted_by_composite_score_descending(
        self, mock_bull_response, mock_bear_response, mock_thesis,
    ):
        """Results should be sorted by composite_score descending."""
        scores = [
            TickerScore(symbol="LOW", composite_score=20, direction=Direction.BEARISH, breakdown=[]),
            TickerScore(symbol="HIGH", composite_score=90, direction=Direction.BULLISH, breakdown=[]),
            TickerScore(symbol="MID", composite_score=55, direction=Direction.NEUTRAL, breakdown=[]),
        ]

        responses = [mock_bull_response, mock_bear_response, mock_thesis] * 3
        client = MockLLMClient(responses)
        manager = DebateManager(client)

        settings = Settings()
        settings.ai_debate_concurrency = 3

        results = await manager.run_debates(
            scores, top_n=3, settings=settings,
        )

        assert len(results) == 3
        assert results[0].composite_score >= results[1].composite_score
        assert results[1].composite_score >= results[2].composite_score


# ===========================================================================
# Test: Retry Jitter (Issue #67)
# ===========================================================================


class TestRetryJitter:
    """Tests for retry jitter in _run_agent_with_retry."""

    @pytest.mark.asyncio
    async def test_jitter_applied_to_network_error_sleep(self):
        """Sleep delays for network errors should include random jitter."""
        client = MagicMock(spec=LLMClient)
        client.complete = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )

        with patch("option_alpha.ai.agents.random.uniform", return_value=0.5) as mock_rand, \
             patch("option_alpha.ai.agents.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(httpx.TimeoutException):
                await _run_agent_with_retry(
                    client,
                    [{"role": "user", "content": "test"}],
                    AgentResponse,
                    "bull",
                    retry_delays=[2.0, 4.0],
                    ticker="AAPL",
                )

            # Should have slept with delay + jitter for first attempt
            assert mock_sleep.call_count >= 1
            first_sleep = mock_sleep.call_args_list[0][0][0]
            assert first_sleep == 2.0 + 0.5  # delay + jitter

    @pytest.mark.asyncio
    async def test_validation_error_appends_hint(self):
        """On ValidationError, a corrective hint should be appended."""
        from pydantic import ValidationError as PydanticValidationError

        client = MagicMock(spec=LLMClient)
        # First call raises validation error, second succeeds
        good_response = AgentResponse(
            role="bull", analysis="Good", key_points=["ok"], conviction=5
        )
        client.complete = AsyncMock(
            side_effect=[
                PydanticValidationError.from_exception_data(
                    "AgentResponse",
                    [{"type": "missing", "loc": ("analysis",), "msg": "Field required", "input": {}}],
                ),
                good_response,
            ]
        )

        messages = [{"role": "user", "content": "test"}]
        result = await _run_agent_with_retry(
            client, messages, AgentResponse, "bull",
            retry_delays=[1.0, 2.0], ticker="AAPL",
        )
        assert result == good_response
        # A corrective hint should have been appended to messages
        assert any("validation issues" in m.get("content", "") for m in messages)


# ===========================================================================
# Test: Context-Aware Fallbacks (Issue #64)
# ===========================================================================


class TestContextAwareFallbacks:
    """Tests for context-aware fallback responses with ticker_score."""

    def test_high_score_bullish_fallback(self):
        """High composite score should produce higher conviction."""
        ts = TickerScore(
            symbol="AAPL", composite_score=90.0,
            direction=Direction.BULLISH, breakdown=[],
        )
        resp = _fallback_agent_response("bull", ticker_score=ts)
        expected_conviction = max(2, min(8, round(90.0 / 12.5)))
        assert resp.conviction == expected_conviction
        assert "90.0" in resp.analysis
        assert "FALLBACK" in resp.analysis

    def test_low_score_bearish_fallback(self):
        """Low composite score should produce lower conviction."""
        ts = TickerScore(
            symbol="BAD", composite_score=15.0,
            direction=Direction.BEARISH, breakdown=[],
        )
        resp = _fallback_agent_response("bear", ticker_score=ts)
        expected_conviction = max(2, min(8, round(15.0 / 12.5)))
        assert resp.conviction == expected_conviction

    def test_none_ticker_score_backward_compat(self):
        """When ticker_score=None, behavior is unchanged."""
        resp = _fallback_agent_response("bull", ticker_score=None)
        assert resp.conviction == 3
        assert "FALLBACK" in resp.analysis

    def test_fallback_thesis_uses_direction(self):
        """Fallback thesis should use ticker_score.direction."""
        ts = TickerScore(
            symbol="UP", composite_score=75.0,
            direction=Direction.BULLISH, breakdown=[],
        )
        thesis = _fallback_thesis("UP", ticker_score=ts)
        assert thesis.direction == Direction.BULLISH
        assert thesis.conviction == max(2, min(8, round(75.0 / 12.5)))

    def test_fallback_thesis_none_is_neutral(self):
        """Fallback thesis with no ticker_score should be NEUTRAL."""
        thesis = _fallback_thesis("XYZ", ticker_score=None)
        assert thesis.direction == Direction.NEUTRAL
        assert thesis.conviction == 3


# ===========================================================================
# Test: Data-Driven Thresholds (Issue #66)
# ===========================================================================


class TestDataDrivenThresholds:
    """Tests for data-driven indicator interpretation."""

    def test_unknown_indicator_returns_raw_value(self):
        """Unknown indicators should return formatted raw value."""
        result = _interpret_indicator("unknown_indicator", 42.123)
        assert result == "42.12"

    def test_nan_returns_na(self):
        """NaN values should return N/A."""
        assert _interpret_indicator("rsi", float("nan")) == "N/A"

    def test_none_returns_na(self):
        """None values should return N/A."""
        assert _interpret_indicator("rsi", None) == "N/A"

    def test_supertrend_bullish(self):
        assert _interpret_indicator("supertrend", 1.0) == "bullish"

    def test_supertrend_bearish(self):
        assert _interpret_indicator("supertrend", -1.0) == "bearish"

    def test_rsi_overbought(self):
        assert _interpret_indicator("rsi", 75.0) == "overbought"

    def test_rsi_oversold(self):
        assert _interpret_indicator("rsi", 25.0) == "oversold"

    def test_adx_strong_trend(self):
        assert _interpret_indicator("adx", 55.0) == "strong trend"

    def test_atr_percent_formatting(self):
        result = _interpret_indicator("atr_percent", 2.5)
        assert "2.5%" in result


# ===========================================================================
# Test: Compressed Prompts (Issue #68)
# ===========================================================================


class TestCompressedPrompts:
    """Tests verifying compressed prompts are concise but complete."""

    def test_bull_prompt_concise(self):
        """Bull prompt should be significantly shorter than 100 words."""
        assert len(BULL_SYSTEM_PROMPT.split()) < 100

    def test_bear_prompt_concise(self):
        """Bear prompt should be significantly shorter than 100 words."""
        assert len(BEAR_SYSTEM_PROMPT.split()) < 100

    def test_risk_prompt_concise(self):
        """Risk prompt should be significantly shorter than 150 words."""
        assert len(RISK_SYSTEM_PROMPT.split()) < 150

    def test_combined_prompts_under_target(self):
        """Combined prompts should be under 850 whitespace-delimited tokens."""
        total = (
            len(BULL_SYSTEM_PROMPT.split())
            + len(BEAR_SYSTEM_PROMPT.split())
            + len(RISK_SYSTEM_PROMPT.split())
        )
        assert total < 850


# ===========================================================================
# Test: Enhanced Health Checks (Issue #65)
# ===========================================================================


class TestEnhancedHealthChecks:
    """Tests for enhanced health check behavior."""

    @pytest.mark.asyncio
    async def test_ollama_health_sends_generate_request(self):
        """Ollama health check should send /api/generate after /api/tags."""
        client = OllamaClient(model="test-model", base_url="http://localhost:11434")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(return_value=mock_response)
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()

            assert result is True
            # Should have called GET /api/tags and POST /api/generate
            mock_ctx.get.assert_called_once()
            mock_ctx.post.assert_called_once()
            post_args = mock_ctx.post.call_args
            assert "/api/generate" in post_args[0][0]

    @pytest.mark.asyncio
    async def test_ollama_health_fails_on_connection_error(self):
        """Ollama health check should return False on connection error."""
        client = OllamaClient()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_claude_health_returns_false_on_401(self):
        """Claude health check should return False for invalid API key."""
        client = ClaudeClient(api_key="sk-invalid")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockAsyncClient:
            mock_ctx = AsyncMock()
            mock_ctx.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.health_check()
            assert result is False

    def test_health_check_timeout_configurable(self):
        """Health check timeout should be configurable."""
        client = OllamaClient(health_check_timeout=30.0)
        assert client._health_timeout == 30.0

        claude = ClaudeClient(api_key="sk-test", health_check_timeout=20.0)
        assert claude._health_timeout == 20.0


# ===========================================================================
# Test: Phase Summary Logging (Issue #69)
# ===========================================================================


class TestPhaseSummaryLogging:
    """Tests for debate phase summary logging."""

    @pytest.mark.asyncio
    async def test_phase_summary_logged(
        self, mock_bull_response, mock_bear_response, mock_thesis, caplog,
    ):
        """Phase summary should be logged after all debates complete."""
        score = TickerScore(
            symbol="LOG", composite_score=70.0,
            direction=Direction.BULLISH, breakdown=[],
        )

        responses = [mock_bull_response, mock_bear_response, mock_thesis]
        client = MockLLMClient(responses)
        manager = DebateManager(client)

        settings = Settings()
        settings.ai_debate_concurrency = 1

        with caplog.at_level(logging.INFO, logger="option_alpha.ai.debate"):
            await manager.run_debates([score], top_n=1, settings=settings)

        # Check that phase summary was logged
        assert any("Completed" in record.message and "debates" in record.message
                    for record in caplog.records)
