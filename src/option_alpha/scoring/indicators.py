"""Technical indicators implemented with pure pandas/numpy.

All functions accept a pandas DataFrame with OHLCV columns
(Open, High, Low, Close, Volume) and return indicator values.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum number of rows required for each indicator to be meaningful.
MIN_ROWS_BB = 20
MIN_ROWS_ATR = 15  # 14-period ATR needs 15 rows (14 TR values)
MIN_ROWS_RSI = 15
MIN_ROWS_OBV = 10
MIN_ROWS_SMA = 50
MIN_ROWS_RELVOL = 20


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise ValueError if DataFrame is missing required OHLCV columns."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def bollinger_band_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> float:
    """Calculate Bollinger Band width: (upper - lower) / middle.

    A lower value indicates tighter consolidation (squeeze).

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < period:
        return float("nan")

    close = df["Close"]
    middle = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std

    # Use the most recent completed value
    bb_width = (upper - lower) / middle
    latest = bb_width.iloc[-1]
    if pd.isna(latest) or middle.iloc[-1] == 0:
        return float("nan")
    return float(latest)


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range (ATR) using Wilder's smoothing.

    Lower ATR relative to price indicates consolidation.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < period + 1:
        return float("nan")

    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing: first value is SMA, then EMA with alpha=1/period
    atr_series = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    latest = atr_series.iloc[-1]
    if pd.isna(latest):
        return float("nan")
    return float(latest)


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as a percentage of current price. Normalizes ATR across stocks."""
    _validate_ohlcv(df)
    atr_val = atr(df, period)
    if np.isnan(atr_val):
        return float("nan")
    last_close = df["Close"].iloc[-1]
    if last_close == 0:
        return float("nan")
    return float(atr_val / last_close * 100)


def rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate RSI using Wilder's smoothing method.

    Returns value between 0-100. NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < period + 1:
        return float("nan")

    delta = df["Close"].diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # Wilder's smoothing (EMA with alpha=1/period)
    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))

    latest = rsi_val.iloc[-1]
    if pd.isna(latest):
        return float("nan")
    return float(latest)


def obv_trend(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate On-Balance Volume trend (slope over the period).

    Returns the normalized OBV slope (OBV change / mean absolute OBV over period).
    A value near zero indicates volume indecision / consolidation.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < period:
        return float("nan")

    close = df["Close"]
    volume = df["Volume"].astype(float)

    # OBV: cumulative sum of volume * sign(price change)
    price_change = close.diff()
    sign = np.sign(price_change)
    sign.iloc[0] = 0  # first row has no change
    obv = (volume * sign).cumsum()

    # Linear regression slope over the last `period` bars
    obv_window = obv.iloc[-period:].values
    if np.all(obv_window == 0):
        return 0.0

    x = np.arange(period, dtype=float)
    # Use polyfit for slope
    slope, _ = np.polyfit(x, obv_window, 1)

    # Normalize by mean absolute OBV to make comparable across stocks
    mean_abs_obv = np.mean(np.abs(obv_window))
    if mean_abs_obv == 0:
        return 0.0

    return float(slope / mean_abs_obv)


def sma_alignment(df: pd.DataFrame, periods: tuple[int, ...] = (20, 50, 200)) -> float:
    """Measure how tightly aligned the SMAs are.

    Returns a score from 0 to 100 where:
    - 100 = all SMAs are at the same price (perfect consolidation)
    - 0 = wide spread between SMAs

    The metric is: 1 - (max_spread / price) mapped to 0-100.
    max_spread = max(SMAs) - min(SMAs), normalized by current price.

    Returns NaN if insufficient data for the longest SMA period.
    """
    _validate_ohlcv(df)
    max_period = max(periods)
    if len(df) < max_period:
        return float("nan")

    close = df["Close"]
    sma_values = []
    for p in periods:
        sma_val = close.rolling(window=p).mean().iloc[-1]
        if pd.isna(sma_val):
            return float("nan")
        sma_values.append(sma_val)

    current_price = close.iloc[-1]
    if current_price == 0:
        return float("nan")

    spread = max(sma_values) - min(sma_values)
    normalized_spread = spread / current_price

    # Clamp to reasonable range. Spread of 20%+ of price is very wide.
    # Map 0% spread -> 100, 20%+ spread -> 0
    score = max(0.0, min(100.0, (1 - normalized_spread / 0.20) * 100))
    return float(score)


def sma_direction(df: pd.DataFrame) -> str:
    """Determine SMA alignment direction: bullish, bearish, or neutral.

    Uses tiered SMA comparison based on available data:
    - >= 200 bars: SMA20/50/200 (full analysis)
    - >= 50 bars: SMA20/50 fallback
    - < 50 bars: neutral (genuinely insufficient data)
    """
    _validate_ohlcv(df)
    if len(df) < 50:
        logger.debug("sma_direction: %d bars, using insufficient tier", len(df))
        return "neutral"

    close = df["Close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]

    if len(df) >= 200:
        sma200 = close.rolling(200).mean().iloc[-1]
        if any(pd.isna(v) for v in (sma20, sma50, sma200)):
            result = "neutral"
        elif sma20 > sma50 > sma200:
            result = "bullish"
        elif sma20 < sma50 < sma200:
            result = "bearish"
        else:
            result = "neutral"
        logger.debug("sma_direction: %d bars, using SMA20/50/200 tier", len(df))
        return result
    else:
        # Fallback: SMA20 vs SMA50
        if any(pd.isna(v) for v in (sma20, sma50)):
            result = "neutral"
        elif sma20 > sma50:
            result = "bullish"
        elif sma20 < sma50:
            result = "bearish"
        else:
            result = "neutral"
        logger.debug("sma_direction: %d bars, using SMA20/50 tier", len(df))
        return result


def relative_volume(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate relative volume: current volume / N-day average volume.

    Values near 1.0 indicate normal volume; < 1 is quiet, > 1 is elevated.
    For squeeze detection, lower relative volume is preferred.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < period:
        return float("nan")

    volume = df["Volume"].astype(float)
    avg_vol = volume.iloc[-period:].mean()
    if avg_vol == 0:
        return float("nan")

    current_vol = volume.iloc[-1]
    return float(current_vol / avg_vol)


def compute_all_indicators(df: pd.DataFrame) -> dict[str, float]:
    """Compute all indicators for a single ticker.

    Returns a dict mapping indicator names to raw values.
    Missing/invalid indicators are set to NaN.
    """
    return {
        "bb_width": bollinger_band_width(df),
        "atr_percent": atr_percent(df),
        "rsi": rsi(df),
        "obv_trend": obv_trend(df),
        "sma_alignment": sma_alignment(df),
        "relative_volume": relative_volume(df),
    }
