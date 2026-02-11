"""Options chain fetching and filtering.

Fetches option chains via yfinance and filters by DTE range, selecting
the expiration date closest to the target midpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from option_alpha.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ChainData:
    """Structured options chain data for a single ticker and expiration."""

    symbol: str
    expiration: date
    dte: int
    underlying_price: float
    calls: pd.DataFrame = field(default_factory=pd.DataFrame)
    puts: pd.DataFrame = field(default_factory=pd.DataFrame)


def get_available_expirations(symbol: str) -> list[date]:
    """Fetch available option expiration dates for a ticker.

    Args:
        symbol: Ticker symbol.

    Returns:
        List of expiration dates sorted chronologically.
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options  # tuple of date strings
        if not expirations:
            return []
        return sorted(
            datetime.strptime(exp, "%Y-%m-%d").date() for exp in expirations
        )
    except Exception as e:
        logger.debug(f"Could not fetch expirations for {symbol}: {e}")
        return []


def select_expiration(
    expirations: list[date],
    dte_min: int = 30,
    dte_max: int = 60,
    reference_date: Optional[date] = None,
) -> Optional[date]:
    """Select the best expiration date within the DTE range.

    Picks the expiration closest to the midpoint of the DTE range.

    Args:
        expirations: Available expiration dates.
        dte_min: Minimum days to expiration.
        dte_max: Maximum days to expiration.
        reference_date: Date to calculate DTE from (default: today).

    Returns:
        Best expiration date, or None if no expirations in range.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc).date()

    target_dte = (dte_min + dte_max) / 2

    candidates = []
    for exp in expirations:
        dte = (exp - reference_date).days
        if dte_min <= dte <= dte_max:
            candidates.append((exp, dte))

    if not candidates:
        return None

    # Pick closest to midpoint
    candidates.sort(key=lambda x: abs(x[1] - target_dte))
    return candidates[0][0]


def fetch_chain(
    symbol: str,
    expiration: date,
) -> Optional[ChainData]:
    """Fetch the options chain for a specific ticker and expiration.

    Args:
        symbol: Ticker symbol.
        expiration: Expiration date.

    Returns:
        ChainData with calls and puts DataFrames, or None on failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        exp_str = expiration.strftime("%Y-%m-%d")
        chain = ticker.option_chain(exp_str)

        # Get underlying price
        info = ticker.info
        underlying_price = info.get("regularMarketPrice") or info.get(
            "currentPrice", 0.0
        )
        if underlying_price == 0:
            # Fallback: use last close from history
            hist = ticker.history(period="1d")
            if not hist.empty:
                underlying_price = float(hist["Close"].iloc[-1])

        reference = datetime.now(timezone.utc).date()
        dte = (expiration - reference).days

        return ChainData(
            symbol=symbol,
            expiration=expiration,
            dte=dte,
            underlying_price=float(underlying_price),
            calls=chain.calls,
            puts=chain.puts,
        )

    except Exception as e:
        logger.warning(f"Failed to fetch chain for {symbol} exp={expiration}: {e}")
        return None


def fetch_chains_for_tickers(
    symbols: list[str],
    settings: Optional[Settings] = None,
    reference_date: Optional[date] = None,
) -> dict[str, ChainData]:
    """Fetch option chains for multiple tickers.

    For each ticker, selects the best expiration within the DTE range
    and fetches the corresponding chain.

    Args:
        symbols: List of ticker symbols.
        settings: Optional settings override.
        reference_date: Date to calculate DTE from.

    Returns:
        Dict mapping symbol -> ChainData for successfully fetched chains.
    """
    if settings is None:
        settings = get_settings()

    results: dict[str, ChainData] = {}

    for symbol in symbols:
        expirations = get_available_expirations(symbol)
        if not expirations:
            logger.debug(f"No option expirations for {symbol}")
            continue

        best_exp = select_expiration(
            expirations,
            dte_min=settings.dte_min,
            dte_max=settings.dte_max,
            reference_date=reference_date,
        )
        if best_exp is None:
            logger.debug(
                f"No expiration in DTE range [{settings.dte_min}-{settings.dte_max}] "
                f"for {symbol}"
            )
            continue

        chain = fetch_chain(symbol, best_exp)
        if chain is not None:
            results[symbol] = chain
            logger.info(
                f"Fetched chain for {symbol}: exp={best_exp}, "
                f"dte={chain.dte}, calls={len(chain.calls)}, puts={len(chain.puts)}"
            )

    logger.info(f"Chains fetched: {len(results)}/{len(symbols)} tickers")
    return results
