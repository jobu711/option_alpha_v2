"""Black-Scholes-Merton Greeks calculation.

Primary implementation uses numpy/scipy for full Python 3.14 compatibility.
Optional py_vollib backend is used if available (installed via `pip install
option-alpha[options]`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Try optional py_vollib backend
_HAS_VOLLIB = False
try:
    from py_vollib.black_scholes import black_scholes as _vollib_bs
    from py_vollib.black_scholes.greeks.analytical import (
        delta as _vollib_delta,
        gamma as _vollib_gamma,
        theta as _vollib_theta,
        vega as _vollib_vega,
    )
    from py_vollib.black_scholes.implied_volatility import (
        implied_volatility as _vollib_iv,
    )

    _HAS_VOLLIB = True
    logger.debug("py_vollib available; using as enhanced backend")
except ImportError:
    logger.debug("py_vollib not available; using built-in BSM implementation")

OptionType = Literal["call", "put"]


@dataclass
class GreeksResult:
    """Container for calculated Greeks."""

    delta: float
    gamma: float
    theta: float
    vega: float
    price: float
    implied_volatility: float | None = None


# ─── Core BSM Functions ─────────────────────────────────────────────


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 in the Black-Scholes formula.

    d1 = (ln(S/K) + (r + sigma^2/2) * T) / (sigma * sqrt(T))
    """
    return (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 = d1 - sigma * sqrt(T)."""
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Calculate Black-Scholes option price.

    Args:
        S: Underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized, e.g. 0.05 for 5%).
        sigma: Volatility (annualized, e.g. 0.20 for 20%).
        option_type: 'call' or 'put'.

    Returns:
        Theoretical option price.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # At expiration or invalid inputs, return intrinsic value
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Calculate Black-Scholes delta.

    Call delta: N(d1)
    Put delta: N(d1) - 1
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def bs_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Calculate Black-Scholes gamma (same for calls and puts).

    Gamma = N'(d1) / (S * sigma * sqrt(T))
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def bs_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Calculate Black-Scholes theta (per day).

    Returns theta in dollars per calendar day (divided by 365).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    sqrt_T = math.sqrt(T)

    # First term (same for calls and puts)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)

    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)

    # Return per-day theta (annualized theta / 365)
    return float((term1 + term2) / 365.0)


def bs_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Calculate Black-Scholes vega (same for calls and puts).

    Returns vega per 1% change in volatility (divided by 100).
    Vega = S * N'(d1) * sqrt(T) / 100
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * math.sqrt(T) / 100.0)


# ─── Implied Volatility ─────────────────────────────────────────────


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float | None:
    """Calculate implied volatility using Newton-Raphson method.

    Falls back to bisection if Newton's method doesn't converge.

    Args:
        market_price: Observed market price of the option.
        S: Underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate.
        option_type: 'call' or 'put'.
        max_iterations: Maximum iterations for solver.
        tolerance: Convergence tolerance.

    Returns:
        Implied volatility (annualized) or None if not solvable.
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return None

    # Check if market price is below intrinsic value
    if option_type == "call":
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * math.exp(-r * T) - S, 0.0)

    if market_price < intrinsic - tolerance:
        return None

    # Try Newton-Raphson first
    sigma = _newton_iv(market_price, S, K, T, r, option_type, max_iterations, tolerance)
    if sigma is not None:
        return sigma

    # Fall back to bisection
    return _bisection_iv(market_price, S, K, T, r, option_type, max_iterations, tolerance)


def _newton_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    max_iterations: int,
    tolerance: float,
) -> float | None:
    """Newton-Raphson implied volatility solver."""
    sigma = 0.3  # Initial guess

    for _ in range(max_iterations):
        try:
            price = bs_price(S, K, T, r, sigma, option_type)
            diff = price - market_price

            if abs(diff) < tolerance:
                return sigma

            # Vega (unscaled, for Newton step)
            d1 = _d1(S, K, T, r, sigma)
            vega_raw = S * norm.pdf(d1) * math.sqrt(T)

            if abs(vega_raw) < 1e-12:
                return None  # Vega too small, can't converge

            sigma -= diff / vega_raw

            if sigma <= 0.001:
                sigma = 0.001
            elif sigma > 10.0:
                return None  # Unreasonable IV

        except (ValueError, ZeroDivisionError, OverflowError):
            return None

    return None


def _bisection_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    max_iterations: int,
    tolerance: float,
) -> float | None:
    """Bisection implied volatility solver."""
    lo, hi = 0.001, 5.0

    try:
        price_lo = bs_price(S, K, T, r, lo, option_type)
        price_hi = bs_price(S, K, T, r, hi, option_type)
    except (ValueError, OverflowError):
        return None

    if market_price < price_lo or market_price > price_hi:
        return None

    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        try:
            price_mid = bs_price(S, K, T, r, mid, option_type)
        except (ValueError, OverflowError):
            return None

        if abs(price_mid - market_price) < tolerance:
            return mid

        if price_mid < market_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


# ─── High-Level Greeks Calculator ───────────────────────────────────


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    market_price: float | None = None,
    use_vollib: bool = True,
) -> GreeksResult:
    """Calculate all Greeks for an option.

    Uses py_vollib backend if available and use_vollib=True.
    Otherwise falls back to built-in BSM implementation.

    Args:
        S: Underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate (annualized).
        sigma: Volatility (annualized).
        option_type: 'call' or 'put'.
        market_price: If provided, calculates implied volatility.
        use_vollib: Whether to try py_vollib backend.

    Returns:
        GreeksResult with all calculated values.
    """
    iv = None
    if market_price is not None and market_price > 0:
        iv = implied_volatility(market_price, S, K, T, r, option_type)
        if iv is not None:
            sigma = iv

    # Use py_vollib if available
    if use_vollib and _HAS_VOLLIB and T > 0 and sigma > 0:
        try:
            flag = "c" if option_type == "call" else "p"
            return GreeksResult(
                delta=float(_vollib_delta(flag, S, K, T, r, sigma)),
                gamma=float(_vollib_gamma(flag, S, K, T, r, sigma)),
                theta=float(_vollib_theta(flag, S, K, T, r, sigma)) / 365.0,
                vega=float(_vollib_vega(flag, S, K, T, r, sigma)) / 100.0,
                price=float(_vollib_bs(flag, S, K, T, r, sigma)),
                implied_volatility=iv,
            )
        except Exception as e:
            logger.debug(f"py_vollib failed, using built-in BSM: {e}")

    # Built-in implementation
    return GreeksResult(
        delta=bs_delta(S, K, T, r, sigma, option_type),
        gamma=bs_gamma(S, K, T, r, sigma),
        theta=bs_theta(S, K, T, r, sigma, option_type),
        vega=bs_vega(S, K, T, r, sigma),
        price=bs_price(S, K, T, r, sigma, option_type),
        implied_volatility=iv,
    )
