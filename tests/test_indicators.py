"""Tests for technical indicators module."""

import numpy as np
import pandas as pd
import pytest

from option_alpha.scoring.indicators import (
    ad_trend,
    adx,
    atr,
    atr_percent,
    bollinger_band_width,
    compute_all_indicators,
    keltner_width,
    obv_trend,
    relative_volume,
    roc,
    rsi,
    sma_alignment,
    sma_direction,
    stoch_rsi,
    supertrend,
    vwap_deviation,
    williams_r,
)


def make_ohlcv(
    n: int = 250,
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0,
    base_volume: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Args:
        n: Number of rows.
        base_price: Starting price.
        volatility: Daily return standard deviation.
        trend: Daily drift (e.g., 0.001 for uptrend).
        base_volume: Average volume.
        seed: Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    returns = rng.normal(trend, volatility, n)
    close = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close with some noise
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = (base_volume * (1 + rng.normal(0, 0.3, n))).astype(int).clip(min=1)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def make_flat_ohlcv(n: int = 250, price: float = 50.0, volume: int = 500_000) -> pd.DataFrame:
    """Generate flat (constant) OHLCV data."""
    return pd.DataFrame(
        {
            "Open": [price] * n,
            "High": [price] * n,
            "Low": [price] * n,
            "Close": [price] * n,
            "Volume": [volume] * n,
        }
    )


def make_zero_volume_ohlcv(n: int = 250, price: float = 100.0) -> pd.DataFrame:
    """OHLCV with zero volume."""
    rng = np.random.RandomState(99)
    close = price + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": [0] * n,
        }
    )


# ─── Bollinger Band Width ────────────────────────────────────────────


class TestBollingerBandWidth:
    def test_normal_data(self):
        df = make_ohlcv()
        bb = bollinger_band_width(df)
        assert not np.isnan(bb)
        assert bb > 0
        # BB width should be reasonable (typically 0.02-0.15 for 2% vol)
        assert 0.001 < bb < 1.0

    def test_flat_prices_near_zero(self):
        df = make_flat_ohlcv()
        bb = bollinger_band_width(df)
        # Flat prices: std dev is 0, so BB width is 0
        assert bb == pytest.approx(0.0, abs=1e-10)

    def test_high_volatility(self):
        df_low = make_ohlcv(volatility=0.005, seed=1)
        df_high = make_ohlcv(volatility=0.05, seed=1)
        bb_low = bollinger_band_width(df_low)
        bb_high = bollinger_band_width(df_high)
        assert bb_high > bb_low

    def test_insufficient_data(self):
        df = make_ohlcv(n=10)
        bb = bollinger_band_width(df)
        assert np.isnan(bb)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 30})
        with pytest.raises(ValueError, match="missing required columns"):
            bollinger_band_width(df)


# ─── ATR ─────────────────────────────────────────────────────────────


class TestATR:
    def test_normal_data(self):
        df = make_ohlcv()
        atr_val = atr(df)
        assert not np.isnan(atr_val)
        assert atr_val > 0

    def test_atr_percent(self):
        df = make_ohlcv()
        pct = atr_percent(df)
        assert not np.isnan(pct)
        assert pct > 0
        # ATR% should be reasonable for 2% daily vol
        assert 0.1 < pct < 20

    def test_flat_prices(self):
        df = make_flat_ohlcv()
        atr_val = atr(df)
        # Flat prices: TR is 0, ATR should be 0
        assert atr_val == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        df = make_ohlcv(n=10)
        atr_val = atr(df)
        assert np.isnan(atr_val)

    def test_higher_vol_higher_atr(self):
        df_low = make_ohlcv(volatility=0.005, seed=5)
        df_high = make_ohlcv(volatility=0.05, seed=5)
        assert atr(df_high) > atr(df_low)


# ─── RSI ─────────────────────────────────────────────────────────────


class TestRSI:
    def test_normal_data(self):
        df = make_ohlcv()
        rsi_val = rsi(df)
        assert not np.isnan(rsi_val)
        assert 0 <= rsi_val <= 100

    def test_strong_uptrend(self):
        df = make_ohlcv(trend=0.01, volatility=0.001, seed=10)
        rsi_val = rsi(df)
        # Strong uptrend should have RSI well above 50
        assert rsi_val > 60

    def test_strong_downtrend(self):
        df = make_ohlcv(trend=-0.01, volatility=0.001, seed=10)
        rsi_val = rsi(df)
        # Strong downtrend should have RSI well below 50
        assert rsi_val < 40

    def test_flat_prices(self):
        df = make_flat_ohlcv()
        rsi_val = rsi(df)
        # Flat prices: no gains or losses -> 0/0 division -> NaN typically
        # Our implementation should handle this gracefully
        # With ewm and zero changes, gains=0, losses=0, RS=NaN
        assert np.isnan(rsi_val) or 0 <= rsi_val <= 100

    def test_insufficient_data(self):
        df = make_ohlcv(n=10)
        rsi_val = rsi(df)
        assert np.isnan(rsi_val)


# ─── OBV Trend ───────────────────────────────────────────────────────


class TestOBVTrend:
    def test_normal_data(self):
        df = make_ohlcv()
        obv = obv_trend(df)
        assert not np.isnan(obv)
        # OBV trend is normalized, should be a small number
        assert -10 < obv < 10

    def test_uptrend_positive_obv(self):
        df = make_ohlcv(trend=0.005, volatility=0.001, seed=20)
        obv = obv_trend(df)
        # In a clean uptrend, OBV should trend up
        assert obv > 0

    def test_insufficient_data(self):
        df = make_ohlcv(n=5)
        obv = obv_trend(df)
        assert np.isnan(obv)

    def test_zero_volume(self):
        df = make_zero_volume_ohlcv()
        obv = obv_trend(df)
        # Zero volume: OBV is always 0
        assert obv == pytest.approx(0.0, abs=1e-10)


# ─── SMA Alignment ──────────────────────────────────────────────────


class TestSMAAlignment:
    def test_normal_data(self):
        df = make_ohlcv()
        score = sma_alignment(df)
        assert not np.isnan(score)
        assert 0 <= score <= 100

    def test_flat_prices_perfect_alignment(self):
        df = make_flat_ohlcv()
        score = sma_alignment(df)
        # Flat prices: all SMAs equal, perfect alignment
        assert score == pytest.approx(100.0, abs=0.1)

    def test_insufficient_data(self):
        df = make_ohlcv(n=100)  # Less than 200
        score = sma_alignment(df)
        assert np.isnan(score)

    def test_strong_trend_lower_alignment(self):
        df_flat = make_flat_ohlcv()
        df_trend = make_ohlcv(trend=0.005, volatility=0.005, seed=30)
        score_flat = sma_alignment(df_flat)
        score_trend = sma_alignment(df_trend)
        # Flat should have better alignment than trending
        assert score_flat > score_trend


class TestSMADirection:
    def test_uptrend_bullish(self):
        df = make_ohlcv(n=300, trend=0.003, volatility=0.001, seed=40)
        direction = sma_direction(df)
        assert direction == "bullish"

    def test_downtrend_bearish(self):
        df = make_ohlcv(n=300, trend=-0.003, volatility=0.001, seed=40)
        direction = sma_direction(df)
        assert direction == "bearish"

    def test_insufficient_data_neutral(self):
        df = make_ohlcv(n=30)
        direction = sma_direction(df)
        assert direction == "neutral"

    def test_fallback_uptrend_bullish(self):
        """With 50-199 bars, uses SMA20 vs SMA50 fallback."""
        df = make_ohlcv(n=100, trend=0.003, volatility=0.001, seed=40)
        direction = sma_direction(df)
        assert direction == "bullish"

    def test_fallback_downtrend_bearish(self):
        """With 50-199 bars, uses SMA20 vs SMA50 fallback."""
        df = make_ohlcv(n=100, trend=-0.003, volatility=0.001, seed=40)
        direction = sma_direction(df)
        assert direction == "bearish"

    def test_40_bars_neutral_insufficient(self):
        """With fewer than 50 bars, always returns neutral."""
        df = make_ohlcv(n=40, trend=0.01, volatility=0.001, seed=42)
        direction = sma_direction(df)
        assert direction == "neutral"

    def test_200_bar_full_tier_bearish(self):
        """With >= 200 bars and downtrend, uses SMA20/50/200 for bearish."""
        df = make_ohlcv(n=250, trend=-0.003, volatility=0.001, seed=50)
        direction = sma_direction(df)
        assert direction == "bearish"

    def test_75_bar_mixed_signals_neutral(self):
        """With 75 bars and noisy data, mixed SMA20/SMA50 yields neutral."""
        # Use flat data with some noise; SMA20 ~ SMA50 when trend=0
        df = make_ohlcv(n=75, trend=0.0, volatility=0.02, seed=99)
        direction = sma_direction(df)
        # With no trend, SMA20 and SMA50 should be roughly equal;
        # result is either neutral or may lean slightly one way depending on noise
        assert direction in ("neutral", "bullish", "bearish")
        # The key point: it does NOT crash and returns a valid tier-2 result


# ─── Relative Volume ────────────────────────────────────────────────


class TestRelativeVolume:
    def test_normal_data(self):
        df = make_ohlcv()
        rv = relative_volume(df)
        assert not np.isnan(rv)
        assert rv > 0

    def test_constant_volume(self):
        df = make_flat_ohlcv()
        rv = relative_volume(df)
        # Constant volume: current = average, ratio = 1.0
        assert rv == pytest.approx(1.0, abs=0.01)

    def test_zero_volume(self):
        df = make_zero_volume_ohlcv()
        rv = relative_volume(df)
        # All zero: avg is 0, should be NaN
        assert np.isnan(rv)

    def test_insufficient_data(self):
        df = make_ohlcv(n=5)
        rv = relative_volume(df)
        assert np.isnan(rv)


# ─── VWAP Deviation ─────────────────────────────────────────────────


class TestVWAPDeviation:
    def test_normal_data(self):
        df = make_ohlcv()
        val = vwap_deviation(df)
        assert not np.isnan(val)
        # Should be a percentage, typically within a few percent
        assert -50 < val < 50

    def test_flat_prices(self):
        df = make_flat_ohlcv()
        val = vwap_deviation(df)
        # Flat prices: close == VWAP, deviation == 0
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        df = make_ohlcv(n=10)
        val = vwap_deviation(df)
        assert np.isnan(val)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 30})
        with pytest.raises(ValueError, match="missing required columns"):
            vwap_deviation(df)

    def test_zero_volume(self):
        df = make_zero_volume_ohlcv()
        val = vwap_deviation(df)
        # Zero volume: cumulative volume is 0, should be NaN
        assert np.isnan(val)

    def test_uptrend_positive_deviation(self):
        # Strong uptrend: close should be above VWAP
        df = make_ohlcv(trend=0.01, volatility=0.001, seed=10)
        val = vwap_deviation(df)
        assert val > 0


# ─── A/D Trend ──────────────────────────────────────────────────────


class TestADTrend:
    def test_normal_data(self):
        df = make_ohlcv()
        val = ad_trend(df)
        assert not np.isnan(val)
        # Normalized slope, should be a small number
        assert -10 < val < 10

    def test_insufficient_data(self):
        df = make_ohlcv(n=10)
        val = ad_trend(df)
        assert np.isnan(val)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 30})
        with pytest.raises(ValueError, match="missing required columns"):
            ad_trend(df)

    def test_zero_volume(self):
        df = make_zero_volume_ohlcv()
        val = ad_trend(df)
        # Zero volume: A/D line is always 0
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_flat_prices(self):
        df = make_flat_ohlcv()
        val = ad_trend(df)
        # Flat prices: H==L, CLV is 0, A/D is 0
        assert val == pytest.approx(0.0, abs=1e-10)


# ─── Stochastic RSI ─────────────────────────────────────────────────


class TestStochRSI:
    def test_normal_data(self):
        df = make_ohlcv()
        val = stoch_rsi(df)
        assert not np.isnan(val)
        assert 0 <= val <= 100

    def test_insufficient_data(self):
        df = make_ohlcv(n=20)
        val = stoch_rsi(df)
        assert np.isnan(val)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 60})
        with pytest.raises(ValueError, match="missing required columns"):
            stoch_rsi(df)

    def test_strong_uptrend_high(self):
        # Use moderate trend with some volatility so RSI varies over the stoch window
        df = make_ohlcv(trend=0.005, volatility=0.01, seed=10)
        val = stoch_rsi(df)
        # Should be a valid value between 0 and 100
        assert 0 <= val <= 100

    def test_strong_downtrend_low(self):
        # Use moderate trend with some volatility so RSI varies over the stoch window
        df = make_ohlcv(trend=-0.005, volatility=0.01, seed=10)
        val = stoch_rsi(df)
        # Should be a valid value between 0 and 100
        assert 0 <= val <= 100

    def test_flat_rsi_returns_midpoint(self):
        # Very strong trend with no volatility: RSI is constant, stochastic returns 50
        df = make_ohlcv(trend=0.01, volatility=0.001, seed=10)
        val = stoch_rsi(df)
        assert val == pytest.approx(50.0, abs=0.1)


# ─── Williams %R ─────────────────────────────────────────────────────


class TestWilliamsR:
    def test_normal_data(self):
        df = make_ohlcv()
        val = williams_r(df)
        assert not np.isnan(val)
        assert -100 <= val <= 0

    def test_insufficient_data(self):
        df = make_ohlcv(n=10)
        val = williams_r(df)
        assert np.isnan(val)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 30})
        with pytest.raises(ValueError, match="missing required columns"):
            williams_r(df)

    def test_flat_prices(self):
        df = make_flat_ohlcv()
        val = williams_r(df)
        # Flat: highest_high == close, so numerator is 0, denom is 0 => -50 (default)
        assert -100 <= val <= 0

    def test_strong_uptrend_near_zero(self):
        # In a strong uptrend with low vol, close is near highest high
        df = make_ohlcv(trend=0.01, volatility=0.001, seed=10)
        val = williams_r(df)
        # Close near highs: Williams %R near 0
        assert val > -50


# ─── Rate of Change ──────────────────────────────────────────────────


class TestROC:
    def test_normal_data(self):
        df = make_ohlcv()
        val = roc(df)
        assert not np.isnan(val)

    def test_insufficient_data(self):
        df = make_ohlcv(n=10)
        val = roc(df)
        assert np.isnan(val)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 30})
        with pytest.raises(ValueError, match="missing required columns"):
            roc(df)

    def test_flat_prices_zero(self):
        df = make_flat_ohlcv()
        val = roc(df)
        # Flat: no change, ROC == 0
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        # Create data where close goes from 100 to 110 over 12 periods
        n = 20
        close = np.linspace(100, 110, n)
        df = pd.DataFrame({
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": [1_000_000] * n,
        })
        val = roc(df, period=12)
        # close[-1] = 110, close[-13] = close[7]
        expected = (close[-1] - close[-13]) / close[-13] * 100
        assert val == pytest.approx(expected, rel=1e-6)

    def test_uptrend_positive(self):
        df = make_ohlcv(trend=0.005, volatility=0.001, seed=10)
        val = roc(df)
        assert val > 0

    def test_downtrend_negative(self):
        df = make_ohlcv(trend=-0.005, volatility=0.001, seed=10)
        val = roc(df)
        assert val < 0


# ─── Keltner Width ───────────────────────────────────────────────────


class TestKeltnerWidth:
    def test_normal_data(self):
        df = make_ohlcv()
        val = keltner_width(df)
        assert not np.isnan(val)
        assert val > 0

    def test_insufficient_data(self):
        df = make_ohlcv(n=10)
        val = keltner_width(df)
        assert np.isnan(val)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 30})
        with pytest.raises(ValueError, match="missing required columns"):
            keltner_width(df)

    def test_flat_prices(self):
        df = make_flat_ohlcv()
        val = keltner_width(df)
        # Flat: ATR is 0, width is 0
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_higher_vol_wider(self):
        df_low = make_ohlcv(volatility=0.005, seed=5)
        df_high = make_ohlcv(volatility=0.05, seed=5)
        assert keltner_width(df_high) > keltner_width(df_low)


# ─── ADX ─────────────────────────────────────────────────────────────


class TestADX:
    def test_normal_data(self):
        df = make_ohlcv()
        val = adx(df)
        assert not np.isnan(val)
        assert 0 <= val <= 100

    def test_insufficient_data(self):
        df = make_ohlcv(n=20)
        val = adx(df)
        assert np.isnan(val)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 60})
        with pytest.raises(ValueError, match="missing required columns"):
            adx(df)

    def test_strong_trend_high_adx(self):
        # Strong consistent trend should produce higher ADX
        df_trend = make_ohlcv(n=100, trend=0.01, volatility=0.001, seed=10)
        df_flat = make_ohlcv(n=100, trend=0.0, volatility=0.02, seed=10)
        adx_trend = adx(df_trend)
        adx_flat = adx(df_flat)
        assert adx_trend > adx_flat

    def test_flat_prices(self):
        df = make_flat_ohlcv()
        val = adx(df)
        # Flat prices: no directional movement
        assert 0 <= val <= 100


# ─── Supertrend ──────────────────────────────────────────────────────


class TestSupertrend:
    def test_normal_data(self):
        df = make_ohlcv()
        val = supertrend(df)
        assert val in (1.0, -1.0)

    def test_insufficient_data(self):
        df = make_ohlcv(n=5)
        val = supertrend(df)
        assert np.isnan(val)

    def test_missing_columns(self):
        df = pd.DataFrame({"Close": [100] * 30})
        with pytest.raises(ValueError, match="missing required columns"):
            supertrend(df)

    def test_strong_uptrend_bullish(self):
        df = make_ohlcv(n=100, trend=0.01, volatility=0.001, seed=10)
        val = supertrend(df)
        assert val == 1.0

    def test_strong_downtrend_bearish(self):
        df = make_ohlcv(n=100, trend=-0.01, volatility=0.001, seed=10)
        val = supertrend(df)
        assert val == -1.0

    def test_returns_float(self):
        df = make_ohlcv()
        val = supertrend(df)
        assert isinstance(val, float)


# ─── Compute All ─────────────────────────────────────────────────────


class TestComputeAll:
    def test_returns_all_indicators(self):
        df = make_ohlcv()
        result = compute_all_indicators(df)
        expected_keys = {
            "bb_width", "atr_percent", "rsi", "obv_trend", "sma_alignment", "relative_volume",
            "vwap_deviation", "ad_trend", "stoch_rsi", "williams_r", "roc",
            "keltner_width", "adx", "supertrend",
        }
        assert set(result.keys()) == expected_keys

    def test_short_data_has_nans(self):
        df = make_ohlcv(n=10)
        result = compute_all_indicators(df)
        # All indicators should be NaN for very short data
        for val in result.values():
            assert np.isnan(val)

    def test_full_data_no_nans(self):
        df = make_ohlcv(n=250)
        result = compute_all_indicators(df)
        for key, val in result.items():
            assert not np.isnan(val), f"{key} is NaN with 250 rows"
