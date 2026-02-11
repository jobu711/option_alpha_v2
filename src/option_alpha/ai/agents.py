"""Bull, Bear, and Risk agent implementations for multi-agent debate.

Each agent receives ticker context and produces structured analysis.
Agents include retry logic with conservative fallback defaults.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import TYPE_CHECKING, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from option_alpha.models import (
    AgentResponse,
    Direction,
    TradeThesis,
)

if TYPE_CHECKING:
    from option_alpha.ai.clients import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 3  # Kept for backward compatibility; retry count now driven by len(retry_delays)

# --- System Prompts ---

BULL_SYSTEM_PROMPT = (
    "You are a bullish stock analyst. Your job is to find the strongest "
    "reasons to be optimistic about this stock based on the provided data. "
    "Focus on positive technical signals, favorable options positioning, "
    "and upcoming catalysts. Be specific and data-driven in your analysis."
)

BEAR_SYSTEM_PROMPT = (
    "You are a bearish stock analyst. Counter the following bull thesis "
    "with specific data-driven arguments. Identify risks, overvaluation "
    "signals, negative technical patterns, and potential downsides. "
    "Be specific about what could go wrong."
)

RISK_SYSTEM_PROMPT = (
    "You are a risk analyst. Synthesize the bull and bear cases to produce "
    "a final trade thesis. Weigh both sides objectively. Your output must "
    "include: direction (bullish/bearish/neutral), conviction (1-10 where "
    "10 is highest), entry rationale, risk factors, and a recommended "
    "action (a specific trade like 'Buy AAPL 180C 30DTE' or 'No trade'). "
    "Be conservative: when in doubt, recommend no trade with low conviction."
)


def _fallback_agent_response(role: str) -> AgentResponse:
    """Conservative fallback when agent fails completely."""
    return AgentResponse(
        role=role,
        analysis=f"Analysis unavailable ({role} agent failed after retries).",
        key_points=["Analysis could not be completed"],
        conviction=3,
    )


def _fallback_thesis(symbol: str) -> TradeThesis:
    """Conservative fallback thesis when risk agent fails."""
    return TradeThesis(
        symbol=symbol,
        direction=Direction.NEUTRAL,
        conviction=3,
        entry_rationale="Insufficient analysis due to agent failure.",
        risk_factors=["AI analysis incomplete"],
        recommended_action="No trade",
    )


async def _run_agent_with_retry(
    client: LLMClient,
    messages: list[dict[str, str]],
    response_model: type[T],
    role: str,
    retry_delays: list[float] | None = None,
) -> T:
    """Run an LLM completion with retry logic and error differentiation.

    Parse errors (JSONDecodeError, ValidationError) retry immediately.
    Network errors (TimeoutException, HTTPStatusError) sleep with backoff.
    """
    if retry_delays is None:
        retry_delays = [2.0, 4.0, 8.0]
    max_retries = len(retry_delays)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await client.complete(messages, response_model=response_model)
        except (_json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.debug(
                "%s agent parse error (attempt %d/%d): %s",
                role, attempt + 1, max_retries, e,
            )
            # Parse errors are fast failures — retry immediately, no sleep
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_error = e
            logger.warning(
                "%s agent network error (attempt %d/%d): %s",
                role, attempt + 1, max_retries, e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])
        except Exception as e:
            last_error = e
            logger.warning(
                "%s agent unexpected error (attempt %d/%d): %s",
                role, attempt + 1, max_retries, e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])

    raise last_error or RuntimeError(f"{role} agent failed after {max_retries} retries")


async def run_bull_agent(
    context: str,
    client: LLMClient,
    retry_delays: list[float] | None = None,
) -> AgentResponse:
    """Run the bull agent to produce a bullish analysis.

    Args:
        context: Curated ticker context string.
        client: LLM client to use for completion.
        retry_delays: Optional list of delay seconds between retries.

    Returns:
        AgentResponse with role='bull'.
    """
    messages = [
        {"role": "system", "content": BULL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Analyze the following stock data and make the bullish case.\n\n"
                f"{context}"
            ),
        },
    ]

    try:
        result = await _run_agent_with_retry(
            client, messages, AgentResponse, "bull", retry_delays=retry_delays,
        )
        if isinstance(result, AgentResponse):
            result.role = "bull"
        return result
    except Exception as e:
        logger.error("Bull agent failed after retries: %s", e)
        return _fallback_agent_response("bull")


async def run_bear_agent(
    context: str,
    bull_analysis: AgentResponse,
    client: LLMClient,
    retry_delays: list[float] | None = None,
) -> AgentResponse:
    """Run the bear agent to counter the bull thesis.

    Args:
        context: Curated ticker context string.
        bull_analysis: The bull agent's response to counter.
        client: LLM client to use for completion.
        retry_delays: Optional list of delay seconds between retries.

    Returns:
        AgentResponse with role='bear'.
    """
    bull_summary = (
        f"BULL THESIS:\n{bull_analysis.analysis}\n\n"
        f"KEY BULL POINTS:\n"
        + "\n".join(f"- {p}" for p in bull_analysis.key_points)
    )

    messages = [
        {"role": "system", "content": BEAR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Counter this bull thesis with bearish arguments based on "
                f"the data.\n\n{context}\n\n{bull_summary}"
            ),
        },
    ]

    try:
        result = await _run_agent_with_retry(
            client, messages, AgentResponse, "bear", retry_delays=retry_delays,
        )
        if isinstance(result, AgentResponse):
            result.role = "bear"
        return result
    except Exception as e:
        logger.error("Bear agent failed after retries: %s", e)
        return _fallback_agent_response("bear")


async def run_risk_agent(
    context: str,
    bull_analysis: AgentResponse,
    bear_analysis: AgentResponse,
    symbol: str,
    client: LLMClient,
    retry_delays: list[float] | None = None,
) -> TradeThesis:
    """Run the risk agent to synthesize bull and bear into a final thesis.

    Args:
        context: Curated ticker context string.
        bull_analysis: The bull agent's response.
        bear_analysis: The bear agent's response.
        symbol: Ticker symbol for the thesis.
        client: LLM client to use for completion.
        retry_delays: Optional list of delay seconds between retries.

    Returns:
        TradeThesis with final direction, conviction, and recommendation.
    """
    debate_summary = (
        f"BULL CASE:\n{bull_analysis.analysis}\n"
        f"Key points: {', '.join(bull_analysis.key_points)}\n\n"
        f"BEAR CASE:\n{bear_analysis.analysis}\n"
        f"Key points: {', '.join(bear_analysis.key_points)}"
    )

    messages = [
        {"role": "system", "content": RISK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Synthesize these bull and bear analyses into a final trade "
                f"thesis for {symbol}.\n\n{context}\n\n{debate_summary}"
            ),
        },
    ]

    try:
        result = await _run_agent_with_retry(
            client, messages, TradeThesis, "risk", retry_delays=retry_delays,
        )
        if isinstance(result, TradeThesis):
            result.symbol = symbol
        return result
    except Exception as e:
        logger.error("Risk agent failed after retries: %s", e)
        return _fallback_thesis(symbol)
