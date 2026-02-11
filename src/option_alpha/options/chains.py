"""Options chain fetching and filtering.

Fetches option chains via Yahoo Finance API directly (with proper
cookie+crumb authentication) and filters by DTE range, selecting
the expiration date closest to the target midpoint.

Uses finance.yahoo.com for cookie acquisition instead of fc.yahoo.com
to avoid DNS/crumb issues that break yfinance's built-in Ticker API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests

from option_alpha.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Module-level session for cookie/crumb reuse across calls.
_session: Optional[cffi_requests.Session] = None
_crumb: Optional[str] = None


def _get_session_and_crumb() -> tuple[cffi_requests.Session, str]:
    """Get an authenticated session with a valid crumb.

    Fetches cookies from finance.yahoo.com and a crumb from the
    getcrumb endpoint. Caches both at module level for reuse.

    Returns:
        Tuple of (session, crumb string).

    Raises:
        RuntimeError: If crumb cannot be obtained.
    """
    global _session, _crumb

    if _session is not None and _crumb is not None:
        return _session, _crumb

    s = cffi_requests.Session(impersonate="chrome")
    s.get("https://finance.yahoo.com/quote/AAPL/", timeout=10)
    r = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10)
    if r.status_code != 200 or not r.text or "<html>" in r.text:
        raise RuntimeError(f"Failed to get Yahoo crumb: status={r.status_code}")

    _session = s
    _crumb = r.text
    return _session, _crumb


def _invalidate_session() -> None:
    """Clear cached session so next call re-authenticates."""
    global _session, _crumb
    _session = None
    _crumb = None


def _fetch_options_json(symbol: str, exp_epoch: Optional[int] = None) -> dict:
    """Fetch raw options JSON from Yahoo Finance v7 API.

    Args:
        symbol: Ticker symbol.
        exp_epoch: Optional expiration date as unix epoch. If None,
            returns the first expiration with all available dates.

    Returns:
        The first result dict from the optionChain response.

    Raises:
        RuntimeError: If the API call fails.
    """
    session, crumb = _get_session_and_crumb()
    url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
    params = {"crumb": crumb}
    if exp_epoch is not None:
        params["date"] = str(exp_epoch)

    r = session.get(url, params=params, timeout=15)

    if r.status_code == 401:
        # Crumb expired — retry once with fresh session.
        _invalidate_session()
        session, crumb = _get_session_and_crumb()
        params["crumb"] = crumb
        r = session.get(url, params=params, timeout=15)

    if r.status_code != 200:
        raise RuntimeError(
            f"Yahoo options API error for {symbol}: {r.status_code} {r.text[:200]}"
        )

    data = r.json()
    results = data.get("optionChain", {}).get("result", [])
    if not results:
        return {}
    return results[0]


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
        result = _fetch_options_json(symbol)
        epochs = result.get("expirationDates", [])
        if not epochs:
            return []
        return sorted(
            datetime.fromtimestamp(ep, tz=timezone.utc).date() for ep in epochs
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
        # Convert date to epoch for Yahoo API.
        exp_epoch = int(
            datetime.combine(expiration, datetime.min.time())
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        result = _fetch_options_json(symbol, exp_epoch=exp_epoch)

        # Get underlying price from the quote embedded in the response.
        quote = result.get("quote", {})
        underlying_price = quote.get("regularMarketPrice", 0.0)
        if not underlying_price:
            # Fallback: use fast_info or history.
            ticker = yf.Ticker(symbol)
            try:
                underlying_price = ticker.fast_info["last_price"]
            except Exception:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    underlying_price = float(hist["Close"].iloc[-1])

        # Parse option chains from response.
        options = result.get("options", [{}])
        opt_data = options[0] if options else {}

        calls_raw = opt_data.get("calls", [])
        puts_raw = opt_data.get("puts", [])
        calls = pd.DataFrame(calls_raw) if calls_raw else pd.DataFrame()
        puts = pd.DataFrame(puts_raw) if puts_raw else pd.DataFrame()

        reference = datetime.now(timezone.utc).date()
        dte = (expiration - reference).days

        return ChainData(
            symbol=symbol,
            expiration=expiration,
            dte=dte,
            underlying_price=float(underlying_price),
            calls=calls,
            puts=puts,
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
