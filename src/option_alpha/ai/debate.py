"""Debate pipeline orchestration for multi-agent AI analysis.

Runs sequential Bull -> Bear -> Risk debates on top N scored tickers,
assembling DebateResult objects with full retry/fallback handling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from option_alpha.config import Settings, get_settings
from option_alpha.ai.agents import (
    _fallback_agent_response,
    _fallback_thesis,
    run_bear_agent,
    run_bull_agent,
    run_risk_agent,
)
from option_alpha.ai.clients import LLMClient
from option_alpha.ai.context import build_context
from option_alpha.models import (
    DebateResult,
    OptionsRecommendation,
    TickerScore,
)

logger = logging.getLogger(__name__)

# Type alias for progress callback: (completed, total, symbol)
ProgressCallback = Callable[[int, int, str], None]


class DebateManager:
    """Orchestrates multi-agent debates for scored tickers."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def run_debate(
        self,
        ticker_score: TickerScore,
        options_rec: Optional[OptionsRecommendation] = None,
    ) -> DebateResult:
        """Run a full Bull -> Bear -> Risk debate for a single ticker.

        Args:
            ticker_score: Scored ticker with breakdown.
            options_rec: Optional options recommendation.

        Returns:
            Complete DebateResult with all three agent responses and thesis.
        """
        symbol = ticker_score.symbol
        logger.info("Starting debate for %s (score: %.1f)", symbol, ticker_score.composite_score)

        # Build context once for all agents
        context = build_context(ticker_score, options_rec)

        # Step 1: Bull analysis
        logger.debug("Running bull agent for %s", symbol)
        bull = await run_bull_agent(context, self.client)

        # Step 2: Bear analysis (receives bull thesis)
        logger.debug("Running bear agent for %s", symbol)
        bear = await run_bear_agent(context, bull, self.client)

        # Step 3: Risk synthesis (receives both)
        logger.debug("Running risk agent for %s", symbol)
        thesis = await run_risk_agent(context, bull, bear, symbol, self.client)

        # Build risk agent response from thesis for the DebateResult
        risk_response = _build_risk_response(thesis)

        result = DebateResult(
            symbol=symbol,
            bull=bull,
            bear=bear,
            risk=risk_response,
            final_thesis=thesis,
        )

        logger.info(
            "Debate complete for %s: direction=%s, conviction=%d, action=%s",
            symbol,
            thesis.direction.value,
            thesis.conviction,
            thesis.recommended_action,
        )
        return result

    async def run_debates(
        self,
        scores: list[TickerScore],
        options_recs: Optional[dict[str, OptionsRecommendation]] = None,
        top_n: int = 10,
        progress_callback: Optional[ProgressCallback] = None,
        settings: Optional[Settings] = None,
    ) -> list[DebateResult]:
        """Run debates for top N scored tickers.

        Processes sequentially to respect LLM rate limits. Each debate
        failure is handled gracefully with conservative defaults.

        Args:
            scores: List of scored tickers (should be pre-sorted by score).
            options_recs: Dict mapping symbol -> OptionsRecommendation.
            top_n: Maximum number of tickers to debate.
            progress_callback: Optional callback(completed, total, symbol).

        Returns:
            List of DebateResult objects.
        """
        if options_recs is None:
            options_recs = {}

        # Take top N by composite score
        candidates = sorted(
            scores, key=lambda s: s.composite_score, reverse=True
        )[:top_n]

        total = len(candidates)
        results: list[DebateResult] = []

        if settings is None:
            settings = get_settings()

        sem = asyncio.Semaphore(settings.ai_debate_concurrency)
        completed = 0

        logger.info("Starting debates for top %d candidates (concurrency=%d)",
                     total, settings.ai_debate_concurrency)

        async def _debate_one(ticker_score: TickerScore) -> DebateResult | None:
            nonlocal completed
            symbol = ticker_score.symbol
            async with sem:
                try:
                    result = await self.run_debate(
                        ticker_score,
                        options_recs.get(symbol),
                    )
                    results.append(result)
                except Exception as e:
                    logger.error("Debate failed for %s: %s", symbol, e)
                    results.append(_fallback_debate_result(ticker_score))

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, symbol)
            return None

        tasks = [_debate_one(ts) for ts in candidates]
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=settings.ai_debate_phase_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Debate phase timed out after %ds, returning %d/%d results",
                settings.ai_debate_phase_timeout, len(results), total,
            )

        logger.info(
            "Debates complete: %d/%d tickers processed", len(results), total
        )
        return results


def _build_risk_response(thesis: TradeThesis) -> AgentResponse:
    """Convert a TradeThesis into an AgentResponse for the risk slot."""
    from option_alpha.models import AgentResponse

    return AgentResponse(
        role="risk",
        analysis=thesis.entry_rationale,
        key_points=thesis.risk_factors or ["No specific risk factors identified"],
        conviction=thesis.conviction,
    )


def _fallback_debate_result(ticker_score: TickerScore) -> DebateResult:
    """Create a conservative fallback DebateResult when debate fails entirely."""
    symbol = ticker_score.symbol
    return DebateResult(
        symbol=symbol,
        bull=_fallback_agent_response("bull"),
        bear=_fallback_agent_response("bear"),
        risk=_fallback_agent_response("risk"),
        final_thesis=_fallback_thesis(symbol),
    )
