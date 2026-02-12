"""Percentile normalization across the ticker universe.

Each indicator's raw values are ranked across all tickers to produce
a 0-100 percentile score, making scores comparable across indicators.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def percentile_rank(value: float, universe_values: list[float]) -> float:
    """Compute the percentile rank of a value within a distribution.

    Args:
        value: The value to rank.
        universe_values: All values in the universe for this indicator.

    Returns:
        Percentile rank from 0 to 100.
        Returns NaN if value is NaN or universe is empty.
    """
    if np.isnan(value) or len(universe_values) == 0:
        return float("nan")

    # Filter out NaN from universe
    valid = [v for v in universe_values if not np.isnan(v)]
    if len(valid) == 0:
        return float("nan")

    return float(stats.percentileofscore(valid, value, kind="rank"))


def normalize_universe(
    raw_scores: dict[str, dict[str, float]],
    invert: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Normalize all indicator raw values to percentiles across the universe.

    Args:
        raw_scores: {ticker: {indicator_name: raw_value, ...}, ...}
        invert: Set of indicator names where LOWER raw values should get
                HIGHER percentile scores (e.g., bb_width, atr_percent,
                relative_volume - tighter/lower = better for squeeze detection).

    Returns:
        {ticker: {indicator_name: percentile_score (0-100), ...}, ...}
    """
    if invert is None:
        # For squeeze/consolidation, lower BB width, lower ATR%, and lower
        # relative volume indicate tighter consolidation = higher score.
        invert = {"bb_width", "atr_percent", "relative_volume", "keltner_width"}

    if not raw_scores:
        return {}

    # Gather all indicator names from any ticker
    all_indicators: set[str] = set()
    for indicators in raw_scores.values():
        all_indicators.update(indicators.keys())

    # For each indicator, collect all universe values
    indicator_values: dict[str, list[float]] = {ind: [] for ind in all_indicators}
    for indicators in raw_scores.values():
        for ind_name, val in indicators.items():
            indicator_values[ind_name].append(val)

    # Compute percentile for each ticker/indicator
    normalized: dict[str, dict[str, float]] = {}
    for ticker, indicators in raw_scores.items():
        normalized[ticker] = {}
        for ind_name, raw_val in indicators.items():
            pct = percentile_rank(raw_val, indicator_values[ind_name])

            if ind_name in invert and not np.isnan(pct):
                # Invert: lower raw value -> higher percentile
                pct = 100.0 - pct

            normalized[ticker][ind_name] = pct

    return normalized


def penalize_insufficient_data(
    normalized_scores: dict[str, dict[str, float]],
    penalty_factor: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Apply penalty to tickers that have NaN indicators.

    Tickers with NaN values get those indicators set to a penalized default
    (penalty_factor * 50, i.e., below-median) instead of being excluded.

    Args:
        normalized_scores: Output from normalize_universe.
        penalty_factor: Multiplier for the default score (0-1).

    Returns:
        Same structure with NaN values replaced by penalized defaults.
    """
    default_score = penalty_factor * 50.0  # Below-median score

    result: dict[str, dict[str, float]] = {}
    for ticker, indicators in normalized_scores.items():
        result[ticker] = {}
        for ind_name, score in indicators.items():
            if np.isnan(score):
                result[ticker][ind_name] = default_score
            else:
                result[ticker][ind_name] = score
    return result
