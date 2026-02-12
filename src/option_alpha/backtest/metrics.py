"""Backtest metrics: trade results, summary statistics, and performance calculations.

Provides Pydantic models for individual trade results and aggregate backtest
statistics, plus pure-function calculators for win rate, max drawdown, and
a Sharpe-like metric.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from option_alpha.models import Direction


class TradeResult(BaseModel):
    """Result of a single simulated trade from the backtester."""

    ticker: str
    signal_date: date
    direction: Direction
    composite_score: float = Field(ge=0, le=100)
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    return_pct: float = Field(description="Signed return percentage")
    holding_days: int = Field(ge=0)
    was_winner: bool


class BacktestResult(BaseModel):
    """Aggregate statistics from a backtest run."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = Field(default=0.0, description="Percentage 0-100")
    avg_return: float = Field(default=0.0, description="Mean return % across trades")
    max_drawdown: float = Field(
        default=0.0, description="Largest peak-to-trough decline in cumulative returns %"
    )
    sharpe_like_metric: float = Field(
        default=0.0,
        description="mean_return / std_return, annualized approximation",
    )
    per_trade_results: list[TradeResult] = Field(default_factory=list)

    # Configuration that produced this result
    lookback_period: Optional[int] = None
    breakout_threshold: Optional[float] = None
    holding_period: Optional[int] = None
    top_n: Optional[int] = None


def calculate_metrics(
    trades: list[TradeResult],
    *,
    annualization_factor: float = 252.0,
) -> BacktestResult:
    """Compute all summary statistics from a list of trade results.

    Args:
        trades: List of individual trade results.
        annualization_factor: Trading days per year for Sharpe annualization.

    Returns:
        BacktestResult with all aggregate statistics populated.
    """
    if not trades:
        return BacktestResult(per_trade_results=[])

    total = len(trades)
    winners = [t for t in trades if t.was_winner]
    losers = [t for t in trades if not t.was_winner]

    win_rate = (len(winners) / total) * 100.0 if total > 0 else 0.0

    returns = [t.return_pct for t in trades]
    avg_return = float(np.mean(returns))

    max_dd = _max_drawdown(returns)

    sharpe = _sharpe_like(returns, annualization_factor)

    return BacktestResult(
        total_trades=total,
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=round(win_rate, 2),
        avg_return=round(avg_return, 4),
        max_drawdown=round(max_dd, 4),
        sharpe_like_metric=round(sharpe, 4),
        per_trade_results=trades,
    )


def _max_drawdown(returns: list[float]) -> float:
    """Calculate max drawdown from a sequence of trade returns.

    Computes the cumulative equity curve (starting at 100) and finds the
    largest peak-to-trough decline as a percentage.
    """
    if not returns:
        return 0.0

    # Build cumulative equity curve
    equity = [100.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r / 100.0))

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)

    # Drawdown at each point
    drawdown = (peak - equity_arr) / peak * 100.0

    return float(np.max(drawdown))


def _sharpe_like(returns: list[float], annualization_factor: float = 252.0) -> float:
    """Compute a Sharpe-like metric: mean_return / std_return * sqrt(annualization).

    This is a simplified Sharpe ratio (no risk-free rate subtraction) that
    provides a risk-adjusted return measure.

    Returns 0.0 if standard deviation is zero or insufficient data.
    """
    if len(returns) < 2:
        return 0.0

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))

    if std_r == 0:
        return 0.0

    # Annualize: multiply by sqrt(trading_days / avg_holding_period)
    # Since each "return" is one trade, approximate annualization
    return float(mean_r / std_r * np.sqrt(annualization_factor))
