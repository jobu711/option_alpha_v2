"""Weighted geometric mean composite scoring.

Produces TickerScore with full ScoreBreakdown for each ticker in the universe.
The geometric mean ensures that one strong indicator cannot mask weak ones -
a ticker must score well across multiple dimensions to rank highly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from option_alpha.config import Settings, get_settings
from option_alpha.models import Direction, ScoreBreakdown, TickerScore
from option_alpha.scoring.indicators import compute_all_indicators, rsi, sma_direction
from option_alpha.scoring.normalizer import normalize_universe, penalize_insufficient_data

# Mapping from config weight keys to indicator names used in compute_all_indicators.
# catalyst_proximity is not computed here (it's a future feature), so we skip it.
INDICATOR_WEIGHT_MAP = {
    "bb_width": "bb_width",
    "atr_percentile": "atr_percent",
    "rsi": "rsi",
    "obv_trend": "obv_trend",
    "sma_alignment": "sma_alignment",
    "relative_volume": "relative_volume",
}


def weighted_geometric_mean(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute the weighted geometric mean: (prod(s_i^w_i))^(1/sum(w_i)).

    Args:
        scores: {indicator_name: percentile_score (0-100)}
        weights: {indicator_name: weight (0-1)}

    Returns:
        Composite score 0-100. Returns 0 if any input is invalid.
    """
    if not scores or not weights:
        return 0.0

    log_sum = 0.0
    weight_sum = 0.0

    for name, weight in weights.items():
        if weight <= 0 or name not in scores:
            continue
        score = scores[name]
        if np.isnan(score) or score <= 0:
            # Geometric mean is 0 if any factor is 0; use small epsilon
            score = 1.0  # Floor at 1 to avoid log(0)
        log_sum += weight * np.log(score)
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    return float(np.exp(log_sum / weight_sum))


def determine_direction(
    df: pd.DataFrame,
    settings: Settings | None = None,
) -> Direction:
    """Determine trade direction based on RSI and SMA alignment.

    Bullish: RSI > 50 AND SMA bullish, OR RSI > strong_bullish threshold when SMA neutral
    Bearish: RSI < 50 AND SMA bearish, OR RSI < strong_bearish threshold when SMA neutral
    Neutral: otherwise
    """
    if settings is None:
        settings = get_settings()

    rsi_val = rsi(df)
    sma_dir = sma_direction(df)

    if np.isnan(rsi_val):
        return Direction.NEUTRAL

    if rsi_val > 50 and sma_dir == "bullish":
        return Direction.BULLISH
    elif rsi_val < 50 and sma_dir == "bearish":
        return Direction.BEARISH

    # RSI-only signal when SMA is neutral but RSI is decisive
    if sma_dir == "neutral":
        if rsi_val > settings.direction_rsi_strong_bullish:
            return Direction.BULLISH
        elif rsi_val < settings.direction_rsi_strong_bearish:
            return Direction.BEARISH

    return Direction.NEUTRAL


def score_universe(
    ohlcv_data: dict[str, pd.DataFrame],
    settings: Settings | None = None,
) -> list[TickerScore]:
    """Score all tickers in the universe and return sorted TickerScores.

    This is the main entry point for the scoring engine.

    Args:
        ohlcv_data: {ticker_symbol: DataFrame with OHLCV columns}
        settings: Optional Settings override. Uses default if None.

    Returns:
        List of TickerScore objects sorted by composite_score descending.
    """
    if settings is None:
        settings = get_settings()

    if not ohlcv_data:
        return []

    # Step 1: Compute raw indicator values for each ticker
    raw_scores: dict[str, dict[str, float]] = {}
    for ticker, df in ohlcv_data.items():
        raw_scores[ticker] = compute_all_indicators(df)

    # Step 2: Normalize across the universe via percentiles
    normalized = normalize_universe(raw_scores)
    normalized = penalize_insufficient_data(normalized)

    # Step 3: Build active weights (map config keys to indicator names, skip missing)
    active_weights: dict[str, float] = {}
    for config_key, indicator_name in INDICATOR_WEIGHT_MAP.items():
        w = settings.scoring_weights.get(config_key, 0.0)
        if w > 0:
            active_weights[indicator_name] = w

    # Step 4: Compute composite scores and build TickerScore objects
    results: list[TickerScore] = []
    timestamp = datetime.now(UTC)

    for ticker, df in ohlcv_data.items():
        ticker_normalized = normalized.get(ticker, {})
        ticker_raw = raw_scores.get(ticker, {})

        # Compute weighted geometric mean
        composite = weighted_geometric_mean(ticker_normalized, active_weights)
        composite = min(100.0, max(0.0, composite))  # Clamp to [0, 100]

        # Build breakdown
        breakdown: list[ScoreBreakdown] = []
        for indicator_name, weight in active_weights.items():
            raw_val = ticker_raw.get(indicator_name, float("nan"))
            norm_val = ticker_normalized.get(indicator_name, 0.0)
            if np.isnan(raw_val):
                raw_val = 0.0
            contribution = norm_val * weight
            breakdown.append(
                ScoreBreakdown(
                    name=indicator_name,
                    raw_value=round(raw_val, 6),
                    normalized=round(min(100.0, max(0.0, norm_val)), 2),
                    weight=round(weight, 4),
                    contribution=round(contribution, 4),
                )
            )

        # Determine direction
        direction = determine_direction(df, settings=settings)

        # Get price/volume info
        last_price = float(df["Close"].iloc[-1]) if len(df) > 0 else None
        avg_vol = float(df["Volume"].astype(float).iloc[-20:].mean()) if len(df) >= 20 else None

        results.append(
            TickerScore(
                symbol=ticker,
                composite_score=round(composite, 2),
                breakdown=breakdown,
                direction=direction,
                last_price=last_price,
                avg_volume=avg_vol,
                timestamp=timestamp,
            )
        )

    # Sort descending by composite score
    results.sort(key=lambda s: s.composite_score, reverse=True)
    return results
