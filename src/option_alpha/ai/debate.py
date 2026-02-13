"""Debate pipeline orchestration for multi-agent AI analysis.

Runs sequential Bull -> Bear -> Risk debates on top N scored tickers,
assembling DebateResult objects with full retry/fallback handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

from option_alpha.config import Settings, get_settings, get_effective_ai_settings
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
    AgentError,
    DebateResult,
    ErrorCategory,
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
        per_ticker_timeout: Optional[int] = None,
    ) -> DebateResult:
        """Run a full Bull -> Bear -> Risk debate for a single ticker.

        Args:
            ticker_score: Scored ticker with breakdown.
            options_rec: Optional options recommendation.
            per_ticker_timeout: Optional time budget in seconds for all three agents.

        Returns:
            Complete DebateResult with all three agent responses and thesis.
        """
        symbol = ticker_score.symbol
        logger.info("Starting debate for %s (score: %.1f)", symbol, ticker_score.composite_score)

        context = build_context(ticker_score, options_rec)

        budget_total = per_ticker_timeout  # None means no budget enforcement
        budget_start = time.monotonic() if budget_total else None

        # Step 1: Bull analysis
        remaining = (budget_total - (time.monotonic() - budget_start)) if budget_start else None
        logger.debug("Running bull agent for %s%s", symbol, f" ({remaining:.0f}s budget)" if remaining else "")
        if remaining is not None and remaining < 10:
            logger.warning("Insufficient budget for bull agent on %s (%.0fs remaining)", symbol, remaining)
            return _fallback_debate_result(ticker_score)
        try:
            if remaining is not None:
                bull = await asyncio.wait_for(
                    run_bull_agent(context, self.client, ticker_score=ticker_score),
                    timeout=remaining * 0.4,
                )
            else:
                bull = await run_bull_agent(context, self.client, ticker_score=ticker_score)
        except asyncio.TimeoutError:
            logger.warning("Bull agent timed out for %s", symbol)
            return _fallback_debate_result(ticker_score)

        # Step 2: Bear analysis
        remaining = (budget_total - (time.monotonic() - budget_start)) if budget_start else None
        logger.debug("Running bear agent for %s%s", symbol, f" ({remaining:.0f}s budget)" if remaining else "")
        if remaining is not None and remaining < 10:
            logger.warning("Insufficient budget for bear agent on %s (%.0fs remaining)", symbol, remaining)
            return _fallback_debate_result(ticker_score)
        try:
            if remaining is not None:
                bear = await asyncio.wait_for(
                    run_bear_agent(context, bull, self.client, ticker_score=ticker_score),
                    timeout=remaining * 0.5,
                )
            else:
                bear = await run_bear_agent(context, bull, self.client, ticker_score=ticker_score)
        except asyncio.TimeoutError:
            logger.warning("Bear agent timed out for %s", symbol)
            return _fallback_debate_result(ticker_score)

        # Step 3: Risk synthesis
        remaining = (budget_total - (time.monotonic() - budget_start)) if budget_start else None
        logger.debug("Running risk agent for %s%s", symbol, f" ({remaining:.0f}s budget)" if remaining else "")
        if remaining is not None and remaining < 10:
            logger.warning("Insufficient budget for risk agent on %s (%.0fs remaining)", symbol, remaining)
            return _fallback_debate_result(ticker_score)
        try:
            if remaining is not None:
                thesis = await asyncio.wait_for(
                    run_risk_agent(context, bull, bear, symbol, self.client, ticker_score=ticker_score),
                    timeout=remaining,
                )
            else:
                thesis = await run_risk_agent(context, bull, bear, symbol, self.client, ticker_score=ticker_score)
        except asyncio.TimeoutError:
            logger.warning("Risk agent timed out for %s", symbol)
            return _fallback_debate_result(ticker_score)

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

        # Apply backend-aware defaults
        effective = get_effective_ai_settings(settings)
        concurrency = effective["ai_debate_concurrency"]
        per_ticker_timeout = effective["ai_per_ticker_timeout"]

        sem = asyncio.Semaphore(concurrency)
        completed = 0

        logger.info("Starting debates for top %d candidates (concurrency=%d)",
                     total, concurrency)

        async def _debate_one(ticker_score: TickerScore) -> DebateResult | None:
            nonlocal completed
            symbol = ticker_score.symbol
            async with sem:
                start = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        self.run_debate(
                            ticker_score,
                            options_recs.get(symbol),
                            per_ticker_timeout=per_ticker_timeout,
                        ),
                        timeout=per_ticker_timeout,
                    )
                    result.composite_score = ticker_score.composite_score
                except asyncio.TimeoutError:
                    logger.warning(
                        "Debate timed out for %s after %ds",
                        symbol, per_ticker_timeout,
                    )
                    result = _fallback_debate_result(ticker_score)
                    result.composite_score = ticker_score.composite_score
                    result.error = AgentError(
                        category=ErrorCategory.TIMEOUT,
                        message=f"Debate timed out after {per_ticker_timeout}s",
                        agent_role="debate",
                        attempt=0,
                    )
                except Exception as e:
                    logger.error("Debate failed for %s: %s", symbol, e)
                    result = _fallback_debate_result(ticker_score)
                    result.composite_score = ticker_score.composite_score
                    result.error = AgentError(
                        category=ErrorCategory.UNKNOWN,
                        message=str(e),
                        agent_role="debate",
                        attempt=0,
                    )

                elapsed = time.monotonic() - start
                logger.debug("Debate for %s completed in %.1fs", symbol, elapsed)
                results.append(result)

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, symbol)
            return None

        phase_start = time.monotonic()
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

        # Sort by composite_score descending for deterministic ordering
        results.sort(key=lambda r: r.composite_score or 0, reverse=True)

        # Phase summary
        phase_elapsed = time.monotonic() - phase_start
        failures = sum(1 for r in results if r.error is not None)
        logger.info(
            "Completed %d/%d debates (%d failures) in %.1fs",
            len(results), total, failures, phase_elapsed,
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
    """Create a conservative fallback DebateResult when debate fails entirely.

    Passes *ticker_score* through to individual fallback helpers so that
    conviction and direction are derived from pre-computed scoring data.
    """
    symbol = ticker_score.symbol
    return DebateResult(
        symbol=symbol,
        bull=_fallback_agent_response("bull", ticker_score=ticker_score),
        bear=_fallback_agent_response("bear", ticker_score=ticker_score),
        risk=_fallback_agent_response("risk", ticker_score=ticker_score),
        final_thesis=_fallback_thesis(symbol, ticker_score=ticker_score),
    )
