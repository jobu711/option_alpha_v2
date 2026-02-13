"""Comprehensive tests for AI multi-agent debate system.

Tests cover:
- LLM client construction and interface (mocked)
- SDK-based completions and health checks
- Bull, Bear, Risk agent implementations
- Simplified retry logic (1+1)
- Context builder output
- Debate pipeline orchestration
- Conservative defaults on failure
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from option_alpha.ai.clients import (
    ClaudeClient,
    LLMClient,
    OllamaClient,
    _build_example_hint,
    get_client,
)
from option_alpha.ai.agents import (
    _call_with_retry,
    _fallback_agent_response,
    _fallback_thesis,
    run_bear_agent,
    run_bull_agent,
    run_risk_agent,
)
from option_alpha.ai.context import build_context
from option_alpha.ai.debate import (
    DebateManager,
    _build_risk_response,
    _fallback_debate_result,
)
from option_alpha.config import Settings
from option_alpha.models import (
    AgentResponse,
    DebateResult,
    Direction,
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
# Test: _build_example_hint
# ===========================================================================


class TestBuildExampleHint:
    """Tests for _build_example_hint."""

    def test_includes_schema(self):
        hint = _build_example_hint(AgentResponse)
        assert "JSON" in hint
        assert "role" in hint
        assert "analysis" in hint

    def test_returns_string(self):
        hint = _build_example_hint(TradeThesis)
        assert isinstance(hint, str)
        assert len(hint) > 50


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

    def test_has_sdk_client(self):
        client = OllamaClient()
        assert client._client is not None


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

    def test_has_sdk_client(self):
        client = ClaudeClient(api_key="sk-test")
        assert client._client is not None


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


# ===========================================================================
# Test: OllamaClient SDK calls (mocked)
# ===========================================================================


class TestOllamaClientRequest:
    """Tests for OllamaClient SDK-based operations."""

    @pytest.mark.asyncio
    async def test_complete_plain_text(self):
        client = OllamaClient(model="test-model")

        mock_resp = MagicMock()
        mock_resp.message.content = "Hello world"

        with patch.object(client._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.complete([
                {"role": "user", "content": "Say hello"}
            ])

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_complete_structured_output(self):
        client = OllamaClient()

        response_data = {
            "role": "bull",
            "analysis": "Test analysis",
            "key_points": ["p1"],
        }
        mock_resp = MagicMock()
        mock_resp.message.content = json.dumps(response_data)

        with patch.object(client._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.complete(
                [{"role": "user", "content": "analyze"}],
                response_model=AgentResponse,
            )

        assert isinstance(result, AgentResponse)
        assert result.role == "bull"

    @pytest.mark.asyncio
    async def test_complete_passes_format_json(self):
        client = OllamaClient()

        mock_resp = MagicMock()
        mock_resp.message.content = json.dumps({
            "role": "bull", "analysis": "x", "key_points": [],
        })

        with patch.object(client._client, "chat", new_callable=AsyncMock, return_value=mock_resp) as mock_chat:
            await client.complete(
                [{"role": "user", "content": "test"}],
                response_model=AgentResponse,
            )
            _, kwargs = mock_chat.call_args
            assert isinstance(kwargs["format"], dict)
            assert kwargs["format"]["title"] == "AgentResponse"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = OllamaClient(model="llama3.1:8b")

        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b"
        mock_resp = MagicMock()
        mock_resp.models = [mock_model]

        with patch.object(client._client, "list", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_model_not_found(self):
        client = OllamaClient(model="missing-model")

        mock_model = MagicMock()
        mock_model.model = "other-model"
        mock_resp = MagicMock()
        mock_resp.models = [mock_model]

        with patch.object(client._client, "list", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        client = OllamaClient()

        with patch.object(client._client, "list", new_callable=AsyncMock, side_effect=ConnectionError("refused")):
            result = await client.health_check()
        assert result is False


# ===========================================================================
# Test: ClaudeClient SDK calls (mocked)
# ===========================================================================


class TestClaudeClientRequest:
    """Tests for ClaudeClient SDK-based operations."""

    @pytest.mark.asyncio
    async def test_complete_plain_text(self):
        client = ClaudeClient(api_key="sk-test")

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hello from Claude"
        mock_resp = MagicMock()
        mock_resp.content = [mock_block]

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.complete([
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ])

        assert result == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_complete_structured_output(self):
        client = ClaudeClient(api_key="sk-test")

        response_data = {
            "role": "bear",
            "analysis": "Bearish outlook",
            "key_points": ["risk1"],
        }
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = json.dumps(response_data)
        mock_resp = MagicMock()
        mock_resp.content = [mock_block]

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.complete(
                [{"role": "user", "content": "analyze"}],
                response_model=AgentResponse,
            )

        assert isinstance(result, AgentResponse)
        assert result.role == "bear"

    @pytest.mark.asyncio
    async def test_system_message_separated(self):
        client = ClaudeClient(api_key="sk-test")

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "ok"
        mock_resp = MagicMock()
        mock_resp.content = [mock_block]

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_resp) as mock_create:
            await client.complete([
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "test"},
            ])

            _, kwargs = mock_create.call_args
            assert kwargs["system"] == "Be helpful"
            assert all(m["role"] != "system" for m in kwargs["messages"])

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = ClaudeClient(api_key="sk-test")

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "hi"
        mock_resp = MagicMock()
        mock_resp.content = [mock_block]

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        client = ClaudeClient(api_key="sk-test")

        import anthropic
        with patch.object(
            client._client.messages, "create",
            new_callable=AsyncMock,
            side_effect=anthropic.APIConnectionError(request=MagicMock()),
        ):
            result = await client.health_check()
        assert result is False


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
        """Context should be roughly 1500-2000 tokens (~500-800 words)."""
        ctx = build_context(sample_ticker_score, sample_options_rec)
        word_count = len(ctx.split())
        # Rough token estimation: ~1.3 tokens per word
        estimated_tokens = word_count * 1.3
        assert estimated_tokens < 3000, f"Context too long: ~{estimated_tokens:.0f} tokens"
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


# ===========================================================================
# Test: _call_with_retry
# ===========================================================================


class TestCallWithRetry:
    """Tests for the simplified 1+1 retry wrapper."""

    @pytest.mark.asyncio
    async def test_success_first_try(self, mock_bull_response):
        client = MockLLMClient([mock_bull_response])
        result = await _call_with_retry(client, [], AgentResponse, "bull")
        assert isinstance(result, AgentResponse)
        assert client._call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_parse_error(self, mock_bull_response):
        client = MockLLMClient([
            json.JSONDecodeError("bad", "", 0),
            mock_bull_response,
        ])
        result = await _call_with_retry(client, [{"role": "user", "content": "test"}], AgentResponse, "bull")
        assert isinstance(result, AgentResponse)
        assert client._call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self, mock_bull_response):
        client = MockLLMClient([
            ConnectionError("timeout"),
            mock_bull_response,
        ])
        result = await _call_with_retry(client, [], AgentResponse, "bull")
        assert isinstance(result, AgentResponse)
        assert client._call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_both_fail(self):
        client = MockLLMClient([
            RuntimeError("first"),
            RuntimeError("second"),
        ])
        with pytest.raises(RuntimeError, match="second"):
            await _call_with_retry(client, [], AgentResponse, "bull")


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
        """Should retry once and succeed on second attempt."""
        good_response = AgentResponse(
            role="bull", analysis="Good analysis", key_points=["point"]
        )
        client = MockLLMClient([
            RuntimeError("timeout"),
            good_response,
        ])
        result = await run_bull_agent("context", client)
        assert result.role == "bull"
        assert result.analysis == "Good analysis"

    @pytest.mark.asyncio
    async def test_bull_fallback_on_total_failure(self):
        """Should return conservative fallback after retry fails."""
        client = FailingLLMClient()
        result = await run_bull_agent("context", client)
        assert result.role == "bull"
        assert "unavailable" in result.analysis.lower() or "failed" in result.analysis.lower()
        # 1+1 retry = 2 calls total
        assert client.call_count == 2


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
        assert client.call_count == 2


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
        assert client.call_count == 2

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
        assert result.final_thesis.direction == Direction.NEUTRAL
        assert result.final_thesis.conviction == 3


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
        # Second should be fallback
        assert results[1].symbol == "FAIL"
        assert results[1].final_thesis.direction == Direction.NEUTRAL
        assert results[1].final_thesis.conviction == 3

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

    @pytest.mark.asyncio
    async def test_debate_timeout_returns_fallback(self, sample_ticker_score):
        """When per_ticker_timeout is very small, should fallback gracefully."""
        async def slow_complete(messages, response_model=None):
            await asyncio.sleep(10)
            return AgentResponse(role="bull", analysis="x", key_points=[])

        class SlowClient(LLMClient):
            async def complete(self_inner, messages, response_model=None):
                return await slow_complete(messages, response_model)

            async def health_check(self_inner):
                return True

        manager = DebateManager(SlowClient())
        result = await manager.run_debate(
            sample_ticker_score, per_ticker_timeout=0.01,
        )
        assert result.final_thesis.direction == Direction.NEUTRAL
        assert result.final_thesis.conviction == 3
