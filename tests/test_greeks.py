"""Tests for Black-Scholes Greeks calculations."""

import math

import pytest
from scipy.stats import norm

from option_alpha.options.greeks import (
    GreeksResult,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta,
    bs_vega,
    calculate_greeks,
    implied_volatility,
)

# Reference values for Black-Scholes validation:
# S=100, K=100, T=1.0 (1 year), r=0.05, sigma=0.20
# These are well-known textbook values.
REF_S = 100.0
REF_K = 100.0
REF_T = 1.0
REF_R = 0.05
REF_SIGMA = 0.20


# ─── BS Price ────────────────────────────────────────────────────────


class TestBsPrice:
    def test_atm_call(self):
        """ATM call with known parameters."""
        price = bs_price(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "call")
        # Known value ~10.45 for these parameters
        assert price == pytest.approx(10.45, abs=0.1)

    def test_atm_put(self):
        """ATM put with known parameters."""
        price = bs_price(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "put")
        # Put-call parity: P = C - S + K*exp(-rT)
        call_price = bs_price(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "call")
        expected = call_price - REF_S + REF_K * math.exp(-REF_R * REF_T)
        assert price == pytest.approx(expected, abs=0.01)

    def test_deep_itm_call(self):
        """Deep in-the-money call should be near intrinsic."""
        price = bs_price(150.0, 100.0, 0.1, 0.05, 0.20, "call")
        assert price >= 49.0  # At least intrinsic value

    def test_deep_otm_call(self):
        """Deep out-of-the-money call should be near zero."""
        price = bs_price(50.0, 100.0, 0.1, 0.05, 0.20, "call")
        assert price < 1.0

    def test_deep_itm_put(self):
        """Deep in-the-money put."""
        price = bs_price(50.0, 100.0, 0.1, 0.05, 0.20, "put")
        assert price >= 49.0

    def test_deep_otm_put(self):
        """Deep out-of-the-money put."""
        price = bs_price(150.0, 100.0, 0.1, 0.05, 0.20, "put")
        assert price < 1.0

    def test_zero_time(self):
        """At expiration, return intrinsic value."""
        assert bs_price(110.0, 100.0, 0.0, 0.05, 0.20, "call") == pytest.approx(10.0)
        assert bs_price(90.0, 100.0, 0.0, 0.05, 0.20, "call") == pytest.approx(0.0)
        assert bs_price(90.0, 100.0, 0.0, 0.05, 0.20, "put") == pytest.approx(10.0)

    def test_zero_vol(self):
        """Zero volatility returns intrinsic."""
        assert bs_price(110.0, 100.0, 1.0, 0.05, 0.0, "call") == pytest.approx(10.0)

    def test_put_call_parity(self):
        """Verify put-call parity: C - P = S - K*exp(-rT)."""
        for S in [80, 100, 120]:
            for K in [90, 100, 110]:
                C = bs_price(S, K, REF_T, REF_R, REF_SIGMA, "call")
                P = bs_price(S, K, REF_T, REF_R, REF_SIGMA, "put")
                parity = S - K * math.exp(-REF_R * REF_T)
                assert C - P == pytest.approx(parity, abs=0.01)

    def test_positive_price(self):
        """Option price must always be non-negative."""
        for ot in ["call", "put"]:
            price = bs_price(100, 100, 0.5, 0.05, 0.30, ot)
            assert price >= 0


# ─── BS Delta ────────────────────────────────────────────────────────


class TestBsDelta:
    def test_atm_call_delta(self):
        """ATM call delta should be slightly above 0.5."""
        delta = bs_delta(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "call")
        assert 0.5 < delta < 0.7

    def test_atm_put_delta(self):
        """ATM put delta should be slightly below -0.5."""
        delta = bs_delta(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "put")
        assert -0.7 < delta < -0.3

    def test_call_delta_range(self):
        """Call delta must be between 0 and 1."""
        delta = bs_delta(100, 100, 0.5, 0.05, 0.30, "call")
        assert 0 <= delta <= 1

    def test_put_delta_range(self):
        """Put delta must be between -1 and 0."""
        delta = bs_delta(100, 100, 0.5, 0.05, 0.30, "put")
        assert -1 <= delta <= 0

    def test_deep_itm_call_delta(self):
        """Deep ITM call delta approaches 1."""
        delta = bs_delta(200.0, 100.0, 0.5, 0.05, 0.20, "call")
        assert delta > 0.95

    def test_deep_otm_call_delta(self):
        """Deep OTM call delta approaches 0."""
        delta = bs_delta(50.0, 100.0, 0.5, 0.05, 0.20, "call")
        assert delta < 0.05

    def test_call_put_delta_relationship(self):
        """Call delta - Put delta = 1 (approximately, for European options)."""
        call_delta = bs_delta(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "call")
        put_delta = bs_delta(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "put")
        # For European options: Delta_call - Delta_put = exp(-qT) ≈ 1 (no dividends)
        assert call_delta - put_delta == pytest.approx(1.0, abs=0.01)

    def test_zero_time_delta(self):
        """At expiration, delta is 0 or 1 for calls."""
        assert bs_delta(110.0, 100.0, 0.0, 0.05, 0.20, "call") == 1.0
        assert bs_delta(90.0, 100.0, 0.0, 0.05, 0.20, "call") == 0.0
        assert bs_delta(90.0, 100.0, 0.0, 0.05, 0.20, "put") == -1.0


# ─── BS Gamma ────────────────────────────────────────────────────────


class TestBsGamma:
    def test_atm_gamma_positive(self):
        """Gamma is always positive."""
        gamma = bs_gamma(REF_S, REF_K, REF_T, REF_R, REF_SIGMA)
        assert gamma > 0

    def test_atm_gamma_highest(self):
        """ATM gamma should be higher than OTM or ITM gamma."""
        atm = bs_gamma(100, 100, 0.5, 0.05, 0.20)
        otm = bs_gamma(100, 130, 0.5, 0.05, 0.20)
        itm = bs_gamma(100, 70, 0.5, 0.05, 0.20)
        assert atm > otm
        assert atm > itm

    def test_gamma_increases_near_expiry(self):
        """ATM gamma increases as expiration approaches."""
        gamma_far = bs_gamma(100, 100, 1.0, 0.05, 0.20)
        gamma_near = bs_gamma(100, 100, 0.05, 0.05, 0.20)
        assert gamma_near > gamma_far

    def test_zero_time_gamma(self):
        """At expiration, gamma is 0."""
        assert bs_gamma(100, 100, 0.0, 0.05, 0.20) == 0.0


# ─── BS Theta ────────────────────────────────────────────────────────


class TestBsTheta:
    def test_call_theta_negative(self):
        """Theta should be negative for long options (time decay)."""
        theta = bs_theta(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "call")
        assert theta < 0

    def test_put_theta_negative(self):
        """Put theta is usually negative for ATM options."""
        theta = bs_theta(REF_S, REF_K, REF_T, REF_R, REF_SIGMA, "put")
        assert theta < 0

    def test_atm_theta_largest(self):
        """ATM options have the largest absolute theta."""
        atm = abs(bs_theta(100, 100, 0.5, 0.05, 0.30, "call"))
        otm = abs(bs_theta(100, 130, 0.5, 0.05, 0.30, "call"))
        assert atm > otm

    def test_theta_per_day(self):
        """Theta should be in per-day units (small magnitude)."""
        theta = bs_theta(100, 100, 0.5, 0.05, 0.30, "call")
        # Annualized theta ~-9 to -15, per day ~-0.03 to -0.04
        assert -1.0 < theta < 0

    def test_zero_time_theta(self):
        """At expiration, theta is 0."""
        assert bs_theta(100, 100, 0.0, 0.05, 0.20, "call") == 0.0


# ─── BS Vega ─────────────────────────────────────────────────────────


class TestBsVega:
    def test_vega_positive(self):
        """Vega is always positive."""
        vega = bs_vega(REF_S, REF_K, REF_T, REF_R, REF_SIGMA)
        assert vega > 0

    def test_atm_vega_highest(self):
        """ATM options have the highest vega."""
        atm = bs_vega(100, 100, 0.5, 0.05, 0.20)
        otm = bs_vega(100, 130, 0.5, 0.05, 0.20)
        assert atm > otm

    def test_vega_per_1pct(self):
        """Vega is per 1% vol change (S * N'(d1) * sqrt(T) / 100)."""
        vega = bs_vega(100, 100, 1.0, 0.05, 0.20)
        # Should be on the order of ~0.4 per 1% vol change
        assert 0.1 < vega < 1.0

    def test_zero_time_vega(self):
        """At expiration, vega is 0."""
        assert bs_vega(100, 100, 0.0, 0.05, 0.20) == 0.0


# ─── Implied Volatility ─────────────────────────────────────────────


class TestImpliedVolatility:
    def test_round_trip(self):
        """Price -> IV -> Price should round-trip correctly."""
        target_sigma = 0.25
        price = bs_price(100, 100, 0.5, 0.05, target_sigma, "call")
        iv = implied_volatility(price, 100, 100, 0.5, 0.05, "call")
        assert iv is not None
        assert iv == pytest.approx(target_sigma, abs=0.001)

    def test_round_trip_put(self):
        """Round-trip for put options."""
        target_sigma = 0.30
        price = bs_price(100, 105, 0.25, 0.05, target_sigma, "put")
        iv = implied_volatility(price, 100, 105, 0.25, 0.05, "put")
        assert iv is not None
        assert iv == pytest.approx(target_sigma, abs=0.001)

    def test_itm_call(self):
        """IV for in-the-money call."""
        target_sigma = 0.20
        price = bs_price(110, 100, 0.5, 0.05, target_sigma, "call")
        iv = implied_volatility(price, 110, 100, 0.5, 0.05, "call")
        assert iv is not None
        assert iv == pytest.approx(target_sigma, abs=0.005)

    def test_otm_call(self):
        """IV for out-of-the-money call."""
        target_sigma = 0.35
        price = bs_price(90, 100, 0.5, 0.05, target_sigma, "call")
        iv = implied_volatility(price, 90, 100, 0.5, 0.05, "call")
        assert iv is not None
        assert iv == pytest.approx(target_sigma, abs=0.005)

    def test_zero_price(self):
        """Zero market price returns None."""
        assert implied_volatility(0.0, 100, 100, 0.5, 0.05) is None

    def test_zero_time(self):
        """Zero time to expiry returns None."""
        assert implied_volatility(5.0, 100, 100, 0.0, 0.05) is None

    def test_negative_price(self):
        """Negative market price returns None."""
        assert implied_volatility(-1.0, 100, 100, 0.5, 0.05) is None

    def test_high_volatility(self):
        """High IV should be recoverable."""
        target_sigma = 1.50
        price = bs_price(100, 100, 0.5, 0.05, target_sigma, "call")
        iv = implied_volatility(price, 100, 100, 0.5, 0.05, "call")
        assert iv is not None
        assert iv == pytest.approx(target_sigma, abs=0.01)

    def test_low_volatility(self):
        """Low IV should be recoverable."""
        target_sigma = 0.05
        price = bs_price(100, 100, 1.0, 0.05, target_sigma, "call")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, "call")
        assert iv is not None
        assert iv == pytest.approx(target_sigma, abs=0.005)


# ─── Calculate Greeks (high-level) ───────────────────────────────────


class TestCalculateGreeks:
    def test_basic_call(self):
        result = calculate_greeks(100, 100, 0.5, 0.05, 0.25, "call", use_vollib=False)
        assert isinstance(result, GreeksResult)
        assert 0 < result.delta < 1
        assert result.gamma > 0
        assert result.theta < 0
        assert result.vega > 0
        assert result.price > 0

    def test_basic_put(self):
        result = calculate_greeks(100, 100, 0.5, 0.05, 0.25, "put", use_vollib=False)
        assert -1 < result.delta < 0
        assert result.gamma > 0
        assert result.theta < 0
        assert result.vega > 0
        assert result.price > 0

    def test_with_market_price_calculates_iv(self):
        """When market_price is provided, IV should be calculated."""
        market_price = bs_price(100, 100, 0.5, 0.05, 0.30, "call")
        result = calculate_greeks(
            100, 100, 0.5, 0.05, 0.20, "call",
            market_price=market_price, use_vollib=False,
        )
        assert result.implied_volatility is not None
        assert result.implied_volatility == pytest.approx(0.30, abs=0.01)

    def test_without_market_price_no_iv(self):
        """Without market_price, IV should be None."""
        result = calculate_greeks(100, 100, 0.5, 0.05, 0.25, "call", use_vollib=False)
        assert result.implied_volatility is None

    def test_edge_case_near_expiry(self):
        """Near expiry should not crash."""
        result = calculate_greeks(
            100, 100, 1 / 365, 0.05, 0.30, "call", use_vollib=False,
        )
        assert result.price >= 0
        assert 0 <= result.delta <= 1

    def test_edge_case_deep_itm(self):
        """Deep ITM should give delta close to 1."""
        result = calculate_greeks(
            200, 100, 0.5, 0.05, 0.20, "call", use_vollib=False,
        )
        assert result.delta > 0.99

    def test_edge_case_deep_otm(self):
        """Deep OTM should give delta close to 0."""
        result = calculate_greeks(
            50, 100, 0.5, 0.05, 0.20, "call", use_vollib=False,
        )
        assert result.delta < 0.01
