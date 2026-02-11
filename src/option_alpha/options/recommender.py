"""Contract recommendation engine.

Selects optimal options contracts based on delta targeting, liquidity
filtering, and direction from scoring. Integrates with FRED API for
risk-free rate.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Optional

import httpx
import pandas as pd

from option_alpha.config import Settings, get_settings
from option_alpha.models import Direction, OptionsRecommendation, TickerScore
from option_alpha.options.chains import ChainData
from option_alpha.options.greeks import calculate_greeks

logger = logging.getLogger(__name__)

# FRED API endpoint for 10-Year Treasury Constant Maturity Rate
FRED_SERIES_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)
FRED_SERIES_ID = "DGS10"

# Target delta range for contract selection
TARGET_DELTA_MIN = 0.30
TARGET_DELTA_MAX = 0.40


def fetch_risk_free_rate(
    api_key: Optional[str] = None,
    fallback: float = 0.05,
    timeout: float = 10.0,
) -> float:
    """Fetch the current risk-free rate from FRED API.

    Uses the 10-Year Treasury Constant Maturity Rate (DGS10).

    Args:
        api_key: FRED API key. If None, returns fallback.
        fallback: Fallback rate if API call fails (default 5%).
        timeout: HTTP request timeout in seconds.

    Returns:
        Risk-free rate as a decimal (e.g. 0.045 for 4.5%).
    """
    if not api_key:
        logger.debug("No FRED API key provided; using fallback rate %.2f%%", fallback * 100)
        return fallback

    try:
        params = {
            "series_id": FRED_SERIES_ID,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }

        with httpx.Client(timeout=timeout) as client:
            response = client.get(FRED_SERIES_URL, params=params)
            response.raise_for_status()

        data = response.json()
        observations = data.get("observations", [])

        for obs in observations:
            value = obs.get("value", ".")
            if value != ".":
                rate = float(value) / 100.0  # Convert percentage to decimal
                logger.info(f"FRED risk-free rate: {rate:.4f} ({obs.get('date', 'N/A')})")
                return rate

        logger.warning("No valid FRED observations; using fallback rate")
        return fallback

    except Exception as e:
        logger.warning(f"FRED API call failed: {e}; using fallback rate")
        return fallback


def filter_by_liquidity(
    chain_df: pd.DataFrame,
    min_open_interest: int = 100,
    max_bid_ask_spread_pct: float = 0.10,
    min_volume: int = 1,
) -> pd.DataFrame:
    """Filter options chain by liquidity criteria.

    Args:
        chain_df: DataFrame with option chain data (from yfinance).
        min_open_interest: Minimum open interest threshold.
        max_bid_ask_spread_pct: Maximum bid-ask spread as percentage of mid price.
        min_volume: Minimum volume threshold.

    Returns:
        Filtered DataFrame meeting all liquidity criteria.
    """
    if chain_df.empty:
        return chain_df

    df = chain_df.copy()

    # Filter by open interest
    if "openInterest" in df.columns:
        df = df[df["openInterest"].fillna(0) >= min_open_interest]

    # Filter by volume
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= min_volume]

    # Filter by bid-ask spread
    if "bid" in df.columns and "ask" in df.columns:
        df = df.copy()
        bid = df["bid"].fillna(0)
        ask = df["ask"].fillna(0)
        mid = (bid + ask) / 2
        spread = ask - bid

        # Avoid division by zero
        valid_mid = mid > 0
        spread_pct = pd.Series(float("inf"), index=df.index)
        spread_pct[valid_mid] = spread[valid_mid] / mid[valid_mid]

        df = df[spread_pct <= max_bid_ask_spread_pct]

    return df


def select_contract(
    chain_df: pd.DataFrame,
    underlying_price: float,
    dte: int,
    risk_free_rate: float,
    option_type: str,
    target_delta_min: float = TARGET_DELTA_MIN,
    target_delta_max: float = TARGET_DELTA_MAX,
) -> Optional[pd.Series]:
    """Select the best contract from a filtered chain based on delta targeting.

    Args:
        chain_df: Filtered options chain DataFrame.
        underlying_price: Current underlying price.
        dte: Days to expiration.
        risk_free_rate: Risk-free rate (decimal).
        option_type: 'call' or 'put'.
        target_delta_min: Minimum absolute delta target.
        target_delta_max: Maximum absolute delta target.

    Returns:
        Best matching contract row as pd.Series, or None if no match.
    """
    if chain_df.empty or underlying_price <= 0 or dte <= 0:
        return None

    T = dte / 365.0
    target_mid = (target_delta_min + target_delta_max) / 2

    best_row = None
    best_delta_diff = float("inf")

    for idx, row in chain_df.iterrows():
        strike = row.get("strike", 0)
        if strike <= 0:
            continue

        # Use implied volatility from chain if available, else estimate from price
        iv = row.get("impliedVolatility")
        if iv is None or pd.isna(iv) or iv <= 0:
            iv = 0.30  # Default estimate

        try:
            greeks = calculate_greeks(
                S=underlying_price,
                K=strike,
                T=T,
                r=risk_free_rate,
                sigma=iv,
                option_type=option_type,
                use_vollib=False,
            )
        except Exception:
            continue

        abs_delta = abs(greeks.delta)
        delta_diff = abs(abs_delta - target_mid)

        if target_delta_min <= abs_delta <= target_delta_max:
            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                best_row = row

    # If no contract in range, find closest to target
    if best_row is None:
        for idx, row in chain_df.iterrows():
            strike = row.get("strike", 0)
            if strike <= 0:
                continue

            iv = row.get("impliedVolatility")
            if iv is None or pd.isna(iv) or iv <= 0:
                iv = 0.30

            try:
                greeks = calculate_greeks(
                    S=underlying_price,
                    K=strike,
                    T=T,
                    r=risk_free_rate,
                    sigma=iv,
                    option_type=option_type,
                    use_vollib=False,
                )
            except Exception:
                continue

            abs_delta = abs(greeks.delta)
            delta_diff = abs(abs_delta - target_mid)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                best_row = row

    return best_row


def recommend_contract(
    ticker_score: TickerScore,
    chain_data: ChainData,
    risk_free_rate: float,
    settings: Optional[Settings] = None,
) -> Optional[OptionsRecommendation]:
    """Generate an options recommendation for a scored ticker.

    Selects calls for BULLISH direction, puts for BEARISH. NEUTRAL tickers
    are skipped (returns None).

    Args:
        ticker_score: Scored ticker with direction.
        chain_data: Options chain data for the ticker.
        risk_free_rate: Current risk-free rate.
        settings: Optional settings override.

    Returns:
        OptionsRecommendation or None if no suitable contract found.
    """
    if settings is None:
        settings = get_settings()

    if ticker_score.direction == Direction.NEUTRAL:
        logger.debug(f"Skipping {ticker_score.symbol}: NEUTRAL direction")
        return None

    option_type = "call" if ticker_score.direction == Direction.BULLISH else "put"
    chain_df = chain_data.calls if option_type == "call" else chain_data.puts

    # Apply liquidity filter
    filtered = filter_by_liquidity(
        chain_df,
        min_open_interest=settings.min_open_interest,
        max_bid_ask_spread_pct=settings.max_bid_ask_spread_pct,
        min_volume=settings.min_option_volume,
    )

    if filtered.empty:
        logger.debug(
            f"No liquid contracts for {ticker_score.symbol} ({option_type}s)"
        )
        return None

    # Select best contract
    best = select_contract(
        filtered,
        chain_data.underlying_price,
        chain_data.dte,
        risk_free_rate,
        option_type,
    )

    if best is None:
        logger.debug(f"No contract matching delta target for {ticker_score.symbol}")
        return None

    # Calculate Greeks for selected contract
    strike = float(best["strike"])
    iv = best.get("impliedVolatility")
    if iv is None or pd.isna(iv) or iv <= 0:
        iv = 0.30

    T = chain_data.dte / 365.0
    greeks = calculate_greeks(
        S=chain_data.underlying_price,
        K=strike,
        T=T,
        r=risk_free_rate,
        sigma=float(iv),
        option_type=option_type,
        use_vollib=False,
    )

    bid = best.get("bid")
    ask = best.get("ask")
    mid_price = None
    if bid is not None and ask is not None and not pd.isna(bid) and not pd.isna(ask):
        mid_price = round((float(bid) + float(ask)) / 2, 2)

    volume = best.get("volume")
    if volume is not None and not pd.isna(volume):
        volume = int(volume)
    else:
        volume = None

    oi = best.get("openInterest")
    if oi is not None and not pd.isna(oi):
        oi = int(oi)
    else:
        oi = None

    return OptionsRecommendation(
        symbol=ticker_score.symbol,
        contract_symbol=best.get("contractSymbol"),
        direction=ticker_score.direction,
        option_type=option_type,
        strike=strike,
        expiry=datetime(
            chain_data.expiration.year,
            chain_data.expiration.month,
            chain_data.expiration.day,
            tzinfo=UTC,
        ),
        dte=chain_data.dte,
        delta=round(greeks.delta, 4),
        gamma=round(greeks.gamma, 6),
        theta=round(greeks.theta, 4),
        vega=round(greeks.vega, 4),
        implied_volatility=round(float(iv), 4),
        bid=round(float(bid), 2) if bid is not None and not pd.isna(bid) else None,
        ask=round(float(ask), 2) if ask is not None and not pd.isna(ask) else None,
        mid_price=mid_price,
        open_interest=oi,
        volume=volume,
        underlying_price=chain_data.underlying_price,
    )


def recommend_for_scored_tickers(
    ticker_scores: list[TickerScore],
    chains: dict[str, ChainData],
    settings: Optional[Settings] = None,
) -> list[OptionsRecommendation]:
    """Generate options recommendations for a list of scored tickers.

    Fetches the risk-free rate once and applies it to all recommendations.

    Args:
        ticker_scores: List of scored tickers (already sorted/filtered).
        chains: Dict mapping symbol -> ChainData.
        settings: Optional settings override.

    Returns:
        List of OptionsRecommendation for tickers with viable contracts.
    """
    if settings is None:
        settings = get_settings()

    # Fetch risk-free rate
    risk_free_rate = fetch_risk_free_rate(
        api_key=settings.fred_api_key,
        fallback=settings.risk_free_rate_fallback,
    )

    recommendations: list[OptionsRecommendation] = []
    for ts in ticker_scores:
        chain = chains.get(ts.symbol)
        if chain is None:
            continue

        rec = recommend_contract(ts, chain, risk_free_rate, settings)
        if rec is not None:
            recommendations.append(rec)
            logger.info(
                f"Recommended {rec.option_type} for {rec.symbol}: "
                f"strike={rec.strike}, delta={rec.delta}, dte={rec.dte}"
            )

    logger.info(
        f"Recommendations: {len(recommendations)}/{len(ticker_scores)} tickers"
    )
    return recommendations
