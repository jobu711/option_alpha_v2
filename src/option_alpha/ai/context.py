"""Context builder for AI debate agents.

Curates ~2500-3000 token prompts from scoring data, options recommendations,
catalyst information, and risk parameters into clean structured text for LLM
consumption.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from option_alpha.models import OptionsRecommendation, TickerScore

# ---------------------------------------------------------------------------
# Data-driven indicator interpretation tables
# ---------------------------------------------------------------------------

# INDICATOR_THRESHOLDS maps indicator names to a list of (predicate, label)
# tuples evaluated in order.  The first predicate that returns True wins.
# Using callable predicates preserves exact boundary behavior (mix of <, <=,
# >, >= in the original logic) without fragile epsilon tricks.

INDICATOR_THRESHOLDS: dict[str, list[tuple[Callable[[float], bool], str]]] = {
    "adx": [
        (lambda v: v > 75, "extreme"),
        (lambda v: v > 50, "strong trend"),
        (lambda v: v > 25, "moderate trend"),
        (lambda v: v >= 20, "developing"),
        (lambda _: True, "weak trend"),
    ],
    "rsi": [
        (lambda v: v > 70, "overbought"),
        (lambda v: v >= 50, "bullish momentum"),
        (lambda v: v >= 30, "bearish momentum"),
        (lambda _: True, "oversold"),
    ],
    "stoch_rsi": [
        (lambda v: v > 80, "overbought"),
        (lambda v: v >= 20, "neutral"),
        (lambda _: True, "oversold"),
    ],
    "williams_r": [
        (lambda v: v > -20, "overbought"),
        (lambda v: v >= -80, "neutral"),
        (lambda _: True, "oversold"),
    ],
    "relative_volume": [
        (lambda v: v > 2.0, "volume surge"),
        (lambda v: v > 1.2, "elevated"),
        (lambda v: v >= 0.8, "normal"),
        (lambda v: v >= 0.5, "below average"),
        (lambda _: True, "very quiet"),
    ],
    "bb_width": [
        (lambda v: v > 0.1, "wide"),
        (lambda v: v >= 0.05, "moderate"),
        (lambda _: True, "tight squeeze"),
    ],
    "keltner_width": [
        (lambda v: v > 0.1, "wide"),
        (lambda v: v >= 0.05, "moderate"),
        (lambda _: True, "tight squeeze"),
    ],
    "sma_alignment": [
        (lambda v: v > 80, "tight alignment"),
        (lambda v: v >= 50, "moderate spread"),
        (lambda _: True, "wide spread"),
    ],
}

# INDICATOR_FORMATTERS handles indicators that need custom formatting rather
# than simple threshold-to-label mapping (binary signals, three-way sign
# checks, percentage formatting, etc.).

INDICATOR_FORMATTERS: dict[str, Callable[[float], str]] = {
    "supertrend": lambda v: "bullish" if v >= 0 else "bearish",
    "vwap_deviation": lambda v: (
        "above VWAP" if v > 1 else ("below VWAP" if v < -1 else "near VWAP")
    ),
    "obv_trend": lambda v: (
        "accumulating" if v > 0 else ("distributing" if v < 0 else "neutral")
    ),
    "ad_trend": lambda v: (
        "accumulating" if v > 0 else ("distributing" if v < 0 else "neutral")
    ),
    "atr_percent": lambda v: f"{v:.1f}% daily range",
    "roc": lambda v: (
        "positive momentum" if v > 0
        else ("negative momentum" if v < 0 else "flat")
    ),
}


def _interpret_indicator(name: str, raw_value: float) -> str:
    """Return a concise human-readable interpretation of an indicator value.

    Uses data-driven lookup tables (INDICATOR_THRESHOLDS and
    INDICATOR_FORMATTERS) so adding a new indicator requires only a dict
    entry, not a code change.

    Args:
        name: Indicator name (e.g. 'rsi', 'adx', 'bb_width').
        raw_value: The raw numeric value of the indicator.

    Returns:
        Short interpretive label string, or 'N/A' for NaN values.
    """
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return "N/A"
    if name in INDICATOR_FORMATTERS:
        return INDICATOR_FORMATTERS[name](raw_value)
    if name in INDICATOR_THRESHOLDS:
        for predicate, label in INDICATOR_THRESHOLDS[name]:
            if predicate(raw_value):
                return label
    return f"{raw_value:.2f}"


def _format_score_breakdown(score: TickerScore) -> str:
    """Format the per-indicator score breakdown table with interpretations."""
    if not score.breakdown:
        return "  No detailed breakdown available."

    lines = [
        "  Indicator         | Raw     | Pctl | Weight | Interpretation",
        "  ------------------|---------|------|--------|---------------",
    ]
    for b in score.breakdown:
        name = b.name[:18].ljust(18)
        raw = f"{b.raw_value:>7.2f}"
        norm = f"{b.normalized:>4.0f}"
        weight = f"{b.weight:>6.2f}"
        interp = _interpret_indicator(b.name, b.raw_value)
        lines.append(f"  {name}| {raw} | {norm} | {weight} | {interp}")
    return "\n".join(lines)


def _format_options_section(rec: OptionsRecommendation) -> str:
    """Format options recommendation details."""
    lines = [
        "OPTIONS RECOMMENDATION:",
        f"  Type: {rec.option_type.upper()}",
        f"  Strike: ${rec.strike:.2f}",
        f"  Expiry: {rec.expiry.strftime('%Y-%m-%d')}",
        f"  DTE: {rec.dte}",
    ]

    # Greeks
    greeks_parts = []
    if rec.delta is not None:
        greeks_parts.append(f"Delta={rec.delta:.4f}")
    if rec.gamma is not None:
        greeks_parts.append(f"Gamma={rec.gamma:.6f}")
    if rec.theta is not None:
        greeks_parts.append(f"Theta={rec.theta:.4f}")
    if rec.vega is not None:
        greeks_parts.append(f"Vega={rec.vega:.4f}")
    if greeks_parts:
        lines.append(f"  Greeks: {', '.join(greeks_parts)}")

    if rec.implied_volatility is not None:
        lines.append(f"  IV: {rec.implied_volatility:.2%}")

    # Pricing
    pricing_parts = []
    if rec.bid is not None:
        pricing_parts.append(f"Bid=${rec.bid:.2f}")
    if rec.ask is not None:
        pricing_parts.append(f"Ask=${rec.ask:.2f}")
    if rec.mid_price is not None:
        pricing_parts.append(f"Mid=${rec.mid_price:.2f}")
    if pricing_parts:
        lines.append(f"  Price: {', '.join(pricing_parts)}")

    # Liquidity
    liquidity_parts = []
    if rec.open_interest is not None:
        liquidity_parts.append(f"OI={rec.open_interest:,}")
    if rec.volume is not None:
        liquidity_parts.append(f"Vol={rec.volume:,}")
    if liquidity_parts:
        lines.append(f"  Liquidity: {', '.join(liquidity_parts)}")

    if rec.underlying_price is not None:
        lines.append(f"  Underlying: ${rec.underlying_price:.2f}")

    return "\n".join(lines)


def _format_options_flow(rec: OptionsRecommendation) -> str:
    """Format options flow context section.

    Shows direction alignment, implied volatility interpretation,
    and volume/open-interest ratio with activity flags.

    Args:
        rec: Options recommendation with contract details.

    Returns:
        Formatted options flow section string.
    """
    lines = ["OPTIONS FLOW:"]

    # Direction
    direction_label = rec.option_type.upper()
    alignment = "bullish" if rec.direction.value == "bullish" else "bearish"
    lines.append(f"  Direction: {direction_label} ({alignment} alignment)")

    # IV interpretation
    if rec.implied_volatility is not None:
        iv = rec.implied_volatility
        if iv > 0.5:
            iv_label = "high/expensive"
        elif iv >= 0.2:
            iv_label = "moderate"
        else:
            iv_label = "low/cheap"
        lines.append(f"  IV: {iv:.1%} ({iv_label})")

    # Volume/OI ratio
    if rec.volume is not None and rec.open_interest is not None and rec.open_interest > 0:
        ratio = rec.volume / rec.open_interest
        activity = "unusual activity" if ratio > 0.5 else "normal"
        lines.append(
            f"  Volume/OI: {rec.volume:,}/{rec.open_interest:,} "
            f"({ratio:.2f} ratio - {activity})"
        )

    return "\n".join(lines)


def _format_risk_params(
    ticker_score: TickerScore,
    options_rec: Optional[OptionsRecommendation] = None,
) -> Optional[str]:
    """Format ATR-based risk parameters section.

    Computes stop-loss distances from entry price using ATR percent.

    Args:
        ticker_score: Scored ticker with breakdown containing atr_percent.
        options_rec: Optional options recommendation (unused currently).

    Returns:
        Formatted risk parameters string, or None if data unavailable.
    """
    if ticker_score.last_price is None:
        return None

    # Find atr_percent in breakdown
    atr_pct = None
    for b in ticker_score.breakdown:
        if b.name == "atr_percent":
            atr_pct = b.raw_value
            break

    if atr_pct is None or (isinstance(atr_pct, float) and math.isnan(atr_pct)):
        return None

    price = ticker_score.last_price
    stop_long = price * (1 - atr_pct / 100)
    stop_short = price * (1 + atr_pct / 100)

    lines = [
        "RISK PARAMETERS:",
        f"  ATR-based stop distance: {atr_pct:.1f}% from entry",
        f"  Suggested stop (long): ${stop_long:.2f}",
        f"  Suggested stop (short): ${stop_short:.2f}",
    ]
    return "\n".join(lines)


def build_context(
    ticker_score: TickerScore,
    options_rec: Optional[OptionsRecommendation] = None,
    sector: Optional[str] = None,
) -> str:
    """Build a curated ~2500-3000 token context prompt for AI debate agents.

    Combines scoring data, indicator breakdown with interpretive labels,
    direction signals, options flow summary, risk parameters, options
    recommendation details, and catalyst information into a clean
    structured text format.

    Args:
        ticker_score: Scored ticker with full breakdown.
        options_rec: Optional options recommendation with Greeks.
        sector: Optional sector name for the ticker.

    Returns:
        Structured text context string (~2500-3000 tokens).
    """
    sections: list[str] = []

    # Header
    sections.append(f"TICKER: {ticker_score.symbol}")
    sections.append(f"COMPOSITE SCORE: {ticker_score.composite_score:.1f}/100")
    sections.append(f"DIRECTION SIGNAL: {ticker_score.direction.value.upper()}")

    if ticker_score.last_price is not None:
        sections.append(f"LAST PRICE: ${ticker_score.last_price:.2f}")
    if ticker_score.avg_volume is not None:
        sections.append(f"AVG VOLUME: {ticker_score.avg_volume:,.0f}")
    if sector is not None:
        sections.append(f"SECTOR: {sector}")

    # Score breakdown with signal summary (merged into single section)
    bullish_count = 0
    bearish_count = 0
    for b in ticker_score.breakdown:
        if b.normalized >= 60:
            bullish_count += 1
        elif b.normalized <= 40:
            bearish_count += 1

    sections.append("")
    sections.append(
        f"SCORE BREAKDOWN / SIGNAL SUMMARY: {bullish_count} bullish, "
        f"{bearish_count} bearish, "
        f"{len(ticker_score.breakdown) - bullish_count - bearish_count} neutral"
    )
    sections.append(_format_score_breakdown(ticker_score))

    # Catalyst information (from breakdown if present)
    catalyst_info = None
    for b in ticker_score.breakdown:
        if "catalyst" in b.name.lower() or "earning" in b.name.lower():
            catalyst_info = b
            break

    if catalyst_info is not None:
        sections.append("")
        sections.append("CATALYST:")
        sections.append(
            f"  Earnings proximity score: {catalyst_info.raw_value:.2f} "
            f"(normalized: {catalyst_info.normalized:.0f}/100)"
        )
        if catalyst_info.normalized >= 70:
            sections.append("  Note: Earnings event approaching - elevated catalyst score")
        elif catalyst_info.normalized <= 30:
            sections.append("  Note: No near-term earnings catalyst")

    # Options flow (new section)
    if options_rec is not None:
        sections.append("")
        sections.append(_format_options_flow(options_rec))

    # Options recommendation
    if options_rec is not None:
        sections.append("")
        sections.append(_format_options_section(options_rec))

    # Risk parameters (new section)
    risk_section = _format_risk_params(ticker_score, options_rec)
    if risk_section is not None:
        sections.append("")
        sections.append(risk_section)

    # Summary prompt
    sections.append("")
    sections.append(
        "Based on the above data, provide your analysis of this stock's "
        "near-term outlook."
    )

    return "\n".join(sections)
