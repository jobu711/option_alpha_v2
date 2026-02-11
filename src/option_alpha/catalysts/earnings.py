"""Earnings date fetching and proximity scoring.

Fetches next earnings dates via yfinance and computes an exponential decay
proximity score. Tickers with earnings 0-3 days out score highest; beyond
14 days the score is minimal. An IV crush warning flag is set when earnings
fall within 7 days.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import yfinance as yf

from option_alpha.config import Settings, get_settings
from option_alpha.models import TickerScore

logger = logging.getLogger(__name__)

# Decay constant tuned so that:
#   exp(-3 / 5)  ~ 0.55  (3-day window  = high)
#   exp(-7 / 5)  ~ 0.25  (7-day window  = medium)
#   exp(-14 / 5) ~ 0.06  (14-day window = low)
DEFAULT_DECAY_CONSTANT = 5.0


@dataclass
class EarningsInfo:
    """Earnings catalyst information for a single ticker."""

    symbol: str
    earnings_date: Optional[date] = None
    days_until: Optional[int] = None
    proximity_score: float = 0.0
    iv_crush_warning: bool = False


def fetch_earnings_date(symbol: str) -> Optional[date]:
    """Fetch the next earnings date for a ticker via yfinance.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL').

    Returns:
        The next earnings date, or None if unavailable.
    """
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        if calendar is None:
            return None

        # yfinance returns calendar as a dict or DataFrame depending on version.
        # Handle both formats.
        if isinstance(calendar, dict):
            # Newer yfinance returns dict with 'Earnings Date' key
            earnings_dates = calendar.get("Earnings Date")
            if earnings_dates is None:
                # Try alternate key
                earnings_dates = calendar.get("Earnings Dates")
            if earnings_dates is None:
                return None
            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                dt = earnings_dates[0]
            else:
                dt = earnings_dates
        else:
            # Older yfinance returns DataFrame
            if hasattr(calendar, "columns") and "Earnings Date" in calendar.columns:
                dt = calendar["Earnings Date"].iloc[0]
            elif hasattr(calendar, "loc"):
                try:
                    dt = calendar.loc["Earnings Date"].iloc[0]
                except (KeyError, IndexError):
                    return None
            else:
                return None

        # Convert to date object
        if isinstance(dt, datetime):
            return dt.date()
        elif isinstance(dt, date):
            return dt
        elif hasattr(dt, "date"):
            return dt.date()
        else:
            return None

    except Exception as e:
        logger.debug(f"Could not fetch earnings date for {symbol}: {e}")
        return None


def compute_proximity_score(
    days_until: int,
    decay_constant: float = DEFAULT_DECAY_CONSTANT,
) -> float:
    """Compute exponential decay proximity score.

    Args:
        days_until: Number of days until earnings.
        decay_constant: Controls how fast the score decays (default 5.0).

    Returns:
        Score between 0.0 and 1.0 (1.0 = earnings today).
    """
    if days_until < 0:
        return 0.0
    return math.exp(-days_until / decay_constant)


def get_earnings_info(
    symbol: str,
    reference_date: Optional[date] = None,
    decay_constant: float = DEFAULT_DECAY_CONSTANT,
) -> EarningsInfo:
    """Get full earnings catalyst info for a single ticker.

    Args:
        symbol: Ticker symbol.
        reference_date: Date to compute days-until from (default: today).
        decay_constant: Exponential decay constant.

    Returns:
        EarningsInfo with proximity score and IV crush warning.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc).date()

    earnings_date = fetch_earnings_date(symbol)
    if earnings_date is None:
        return EarningsInfo(symbol=symbol)

    days_until = (earnings_date - reference_date).days
    if days_until < 0:
        # Earnings already passed
        return EarningsInfo(
            symbol=symbol,
            earnings_date=earnings_date,
            days_until=days_until,
            proximity_score=0.0,
        )

    score = compute_proximity_score(days_until, decay_constant)
    iv_crush = days_until <= 7

    return EarningsInfo(
        symbol=symbol,
        earnings_date=earnings_date,
        days_until=days_until,
        proximity_score=score,
        iv_crush_warning=iv_crush,
    )


def batch_earnings_info(
    symbols: list[str],
    reference_date: Optional[date] = None,
    decay_constant: float = DEFAULT_DECAY_CONSTANT,
) -> dict[str, EarningsInfo]:
    """Get earnings info for multiple tickers.

    Args:
        symbols: List of ticker symbols.
        reference_date: Date to compute days-until from.
        decay_constant: Exponential decay constant.

    Returns:
        Dict mapping symbol -> EarningsInfo.
    """
    results: dict[str, EarningsInfo] = {}
    for symbol in symbols:
        results[symbol] = get_earnings_info(symbol, reference_date, decay_constant)
    return results


def merge_catalyst_scores(
    ticker_scores: list[TickerScore],
    earnings_info: dict[str, EarningsInfo],
    settings: Optional[Settings] = None,
) -> list[TickerScore]:
    """Merge catalyst proximity scores into existing composite scores.

    The catalyst_proximity weight from settings (default 0.25) determines
    how much the catalyst score influences the final composite. The merged
    score is a weighted average: new = (1 - w) * old + w * catalyst * 100.

    Args:
        ticker_scores: Existing scored tickers.
        earnings_info: Earnings info per symbol from batch_earnings_info.
        settings: Optional settings override.

    Returns:
        Updated list of TickerScore with catalyst scores merged in.
        List is re-sorted by composite_score descending.
    """
    if settings is None:
        settings = get_settings()

    catalyst_weight = settings.scoring_weights.get("catalyst_proximity", 0.25)

    updated: list[TickerScore] = []
    for ts in ticker_scores:
        info = earnings_info.get(ts.symbol)
        if info is None or info.proximity_score == 0.0:
            updated.append(ts)
            continue

        # Blend: (1 - w) * composite + w * (proximity * 100)
        catalyst_score_100 = info.proximity_score * 100.0
        new_composite = (
            (1 - catalyst_weight) * ts.composite_score
            + catalyst_weight * catalyst_score_100
        )
        new_composite = min(100.0, max(0.0, new_composite))

        updated.append(
            ts.model_copy(update={"composite_score": round(new_composite, 2)})
        )

    # Re-sort descending
    updated.sort(key=lambda s: s.composite_score, reverse=True)
    return updated
