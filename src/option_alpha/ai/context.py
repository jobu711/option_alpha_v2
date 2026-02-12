"""Context builder for AI debate agents.

Curates ~2000 token prompts from scoring data, options recommendations,
and catalyst information into clean structured text for LLM consumption.
"""

from __future__ import annotations

from typing import Optional

from option_alpha.models import OptionsRecommendation, TickerScore


def _format_score_breakdown(score: TickerScore) -> str:
    """Format the per-indicator score breakdown table."""
    if not score.breakdown:
        return "  No detailed breakdown available."

    lines = [
        "  Indicator         | Raw     | Pctl | Weight | Contrib",
        "  ------------------|---------|------|--------|--------",
    ]
    for b in score.breakdown:
        name = b.name[:18].ljust(18)
        raw = f"{b.raw_value:>7.2f}"
        norm = f"{b.normalized:>4.0f}"
        weight = f"{b.weight:>6.2f}"
        contrib = f"{b.contribution:>7.2f}"
        lines.append(f"  {name}| {raw} | {norm} | {weight} | {contrib}")
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


def build_context(
    ticker_score: TickerScore,
    options_rec: Optional[OptionsRecommendation] = None,
) -> str:
    """Build a curated ~2000 token context prompt for AI debate agents.

    Combines scoring data, indicator breakdown, direction signals,
    options recommendation details, and catalyst information into a
    clean structured text format.

    Args:
        ticker_score: Scored ticker with full breakdown.
        options_rec: Optional options recommendation with Greeks.

    Returns:
        Structured text context string (~1500-2000 tokens).
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

    # Score breakdown
    sections.append("")
    sections.append("SCORE BREAKDOWN:")
    sections.append(_format_score_breakdown(ticker_score))

    # Direction analysis
    sections.append("")
    bullish_count = 0
    bearish_count = 0
    for b in ticker_score.breakdown:
        if b.normalized >= 60:
            bullish_count += 1
        elif b.normalized <= 40:
            bearish_count += 1

    sections.append(
        f"SIGNAL SUMMARY: {bullish_count} bullish, {bearish_count} bearish, "
        f"{len(ticker_score.breakdown) - bullish_count - bearish_count} neutral "
        f"indicators"
    )

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

    # Options recommendation
    if options_rec is not None:
        sections.append("")
        sections.append(_format_options_section(options_rec))

    # Summary prompt
    sections.append("")
    sections.append(
        "Based on the above data, provide your analysis of this stock's "
        "near-term outlook."
    )

    return "\n".join(sections)
