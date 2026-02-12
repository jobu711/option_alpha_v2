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
MIN_ROWS_VWAP = 20
MIN_ROWS_AD = 20
MIN_ROWS_STOCH_RSI = 28
MIN_ROWS_WILLIAMS = 14
MIN_ROWS_ROC = 13
MIN_ROWS_KELTNER = 20
MIN_ROWS_ADX = 28
MIN_ROWS_SUPERTREND = 11


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


def vwap_deviation(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate % distance from VWAP over the last `period` bars.

    Daily approximation: typical_price = (H+L+C)/3,
    VWAP = cumsum(TP*V)/cumsum(V) over last `period` bars.
    Returns (close - vwap) / vwap * 100.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < MIN_ROWS_VWAP:
        return float("nan")

    tail = df.iloc[-period:]
    typical_price = (tail["High"] + tail["Low"] + tail["Close"]) / 3.0
    volume = tail["Volume"].astype(float)

    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()

    if cum_vol.iloc[-1] == 0:
        return float("nan")

    vwap = cum_tp_vol.iloc[-1] / cum_vol.iloc[-1]
    if vwap == 0:
        return float("nan")

    close = df["Close"].iloc[-1]
    return float((close - vwap) / vwap * 100)


def ad_trend(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate A/D line slope normalized (same pattern as obv_trend).

    CLV = ((C-L)-(H-C))/(H-L), handle H==L.
    A/D = cumsum(CLV * Volume).
    Slope via polyfit over last `period`, normalize by mean abs A/D.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < MIN_ROWS_AD:
        return float("nan")

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"].astype(float)

    hl_range = high - low
    clv = pd.Series(np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0), index=df.index)
    ad_line = (clv * volume).cumsum()

    ad_window = ad_line.iloc[-period:].values
    if np.all(ad_window == 0):
        return 0.0

    x = np.arange(period, dtype=float)
    slope, _ = np.polyfit(x, ad_window, 1)

    mean_abs_ad = np.mean(np.abs(ad_window))
    if mean_abs_ad == 0:
        return 0.0

    return float(slope / mean_abs_ad)


def stoch_rsi(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14) -> float:
    """Calculate Stochastic RSI.

    Compute RSI series, then apply stochastic formula:
    (RSI - min(RSI, n)) / (max(RSI, n) - min(RSI, n)) * 100
    over `stoch_period` window. Returns latest value (0-100).

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < MIN_ROWS_STOCH_RSI:
        return float("nan")

    # Compute full RSI series
    delta = df["Close"].diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = gains.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))

    # Apply stochastic to RSI
    rsi_min = rsi_series.rolling(window=stoch_period).min()
    rsi_max = rsi_series.rolling(window=stoch_period).max()
    denom = rsi_max - rsi_min

    stoch = pd.Series(
        np.where(denom != 0, (rsi_series - rsi_min) / denom * 100, 50.0),
        index=df.index,
    )

    latest = stoch.iloc[-1]
    if pd.isna(latest):
        return float("nan")
    return float(latest)


def williams_r(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Williams %R.

    (highest_high - close) / (highest_high - lowest_low) * -100
    Range: -100 to 0.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < MIN_ROWS_WILLIAMS:
        return float("nan")

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()

    denom = highest - lowest
    wr = pd.Series(
        np.where(denom != 0, (highest - close) / denom * -100, -50.0),
        index=df.index,
    )

    latest = wr.iloc[-1]
    if pd.isna(latest):
        return float("nan")
    return float(latest)


def roc(df: pd.DataFrame, period: int = 12) -> float:
    """Calculate Rate of Change.

    (close - close[period_ago]) / close[period_ago] * 100.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < MIN_ROWS_ROC:
        return float("nan")

    close = df["Close"]
    prev_close = close.iloc[-period - 1]
    if prev_close == 0:
        return float("nan")

    current_close = close.iloc[-1]
    return float((current_close - prev_close) / prev_close * 100)


def keltner_width(df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5) -> float:
    """Calculate Keltner Channel width.

    middle = EMA(close, period)
    upper = middle + atr_mult * ATR(period)
    lower = middle - atr_mult * ATR(period)
    width = (upper - lower) / middle

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < MIN_ROWS_KELTNER:
        return float("nan")

    atr_val = atr(df, period)
    if np.isnan(atr_val):
        return float("nan")

    close = df["Close"]
    middle = close.ewm(span=period, adjust=False).mean().iloc[-1]
    if pd.isna(middle) or middle == 0:
        return float("nan")

    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    return float((upper - lower) / middle)


def adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average Directional Index (ADX).

    Full calculation with +DM/-DM, Wilder's smoothing, +DI/-DI, DX, and ADX.
    Returns 0-100.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < MIN_ROWS_ADX:
        return float("nan")

    high = df["High"]
    low = df["Low"]

    high_diff = high.diff()
    low_diff = -low.diff()  # positive when low decreases

    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=df.index,
    )

    # True Range
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing (EMA with alpha=1/period)
    alpha = 1.0 / period
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_tr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # +DI and -DI
    plus_di = pd.Series(
        np.where(smoothed_tr != 0, smoothed_plus_dm / smoothed_tr * 100, 0.0),
        index=df.index,
    )
    minus_di = pd.Series(
        np.where(smoothed_tr != 0, smoothed_minus_dm / smoothed_tr * 100, 0.0),
        index=df.index,
    )

    # DX
    di_sum = plus_di + minus_di
    dx = pd.Series(
        np.where(di_sum != 0, (plus_di - minus_di).abs() / di_sum * 100, 0.0),
        index=df.index,
    )

    # ADX = Wilder's smooth of DX
    adx_series = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    latest = adx_series.iloc[-1]
    if pd.isna(latest):
        return float("nan")
    return float(latest)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> float:
    """Calculate Supertrend indicator.

    Returns 1.0 (bullish) or -1.0 (bearish) for the latest bar.

    Returns NaN if insufficient data.
    """
    _validate_ohlcv(df)
    if len(df) < MIN_ROWS_SUPERTREND:
        return float("nan")

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    # ATR via Wilder's smoothing
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )

    # Simple Wilder's smoothing for ATR
    atr_arr = np.full_like(close, np.nan)
    atr_arr[period] = np.nanmean(tr[1 : period + 1])
    alpha = 1.0 / period
    for i in range(period + 1, len(close)):
        atr_arr[i] = alpha * tr[i] + (1 - alpha) * atr_arr[i - 1]

    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr_arr
    lower_band = hl2 - multiplier * atr_arr

    # Find first valid index (where ATR is available)
    start = period
    if np.isnan(atr_arr[start]):
        return float("nan")

    trend = np.ones(len(close))  # 1 = bullish
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(start + 1, len(close)):
        # Adjust bands based on previous bands
        if final_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            pass  # keep current upper
        else:
            final_upper[i] = final_upper[i - 1]

        if final_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            pass  # keep current lower
        else:
            final_lower[i] = final_lower[i - 1]

        # Determine trend
        if trend[i - 1] == 1.0:
            if close[i] < final_lower[i]:
                trend[i] = -1.0
            else:
                trend[i] = 1.0
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1.0
            else:
                trend[i] = -1.0

    return float(trend[-1])


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
        "vwap_deviation": vwap_deviation(df),
        "ad_trend": ad_trend(df),
        "stoch_rsi": stoch_rsi(df),
        "williams_r": williams_r(df),
        "roc": roc(df),
        "keltner_width": keltner_width(df),
        "adx": adx(df),
        "supertrend": supertrend(df),
    }
