"""Historical backtesting engine that replays scoring against past data.

Walks through each trading day in the configured date range, scores the
universe using only data available up to that date (no look-ahead bias),
selects top-N candidates, and measures whether price moved in the predicted
direction by at least the breakout threshold within the holding period.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd

from option_alpha.config import Settings
from option_alpha.models import Direction
from option_alpha.scoring.composite import score_universe

from .metrics import BacktestResult, TradeResult, calculate_metrics


class BacktestRunner:
    """Configurable backtesting engine for the scoring system.

    Args:
        lookback_period: Days of history required for indicator calculation.
        breakout_threshold: Minimum % price move to count as a winning trade.
        holding_period: Trading days to hold after a signal.
        date_range: (start_date, end_date) inclusive range for simulation.
        top_n: Number of top-scoring candidates to track per signal day.
        scoring_weights: Optional dict to override default config weights.
        step_days: Number of trading days between each scoring evaluation.
            Default 5 (weekly) to avoid overlapping trades.
    """

    def __init__(
        self,
        *,
        lookback_period: int = 200,
        breakout_threshold: float = 5.0,
        holding_period: int = 10,
        date_range: tuple[date, date] | None = None,
        top_n: int = 10,
        scoring_weights: dict[str, float] | None = None,
        step_days: int = 5,
    ) -> None:
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.holding_period = holding_period
        self.date_range = date_range
        self.top_n = top_n
        self.scoring_weights = scoring_weights
        self.step_days = max(1, step_days)

    def run(
        self,
        tickers: list[str],
        ohlcv_data: dict[str, pd.DataFrame],
        on_progress: Optional[Callable[[date, float], None]] = None,
    ) -> BacktestResult:
        """Run the backtest over the configured date range.

        Args:
            tickers: List of ticker symbols to evaluate.
            ohlcv_data: {ticker: DataFrame} with OHLCV columns and
                DatetimeIndex or a 'Date' column.
            on_progress: Optional callback(signal_date, pct_complete).

        Returns:
            BacktestResult with all trades and aggregate statistics.
        """
        # Normalize DataFrames: ensure DatetimeIndex
        normalized = self._normalize_data(tickers, ohlcv_data)
        if not normalized:
            return calculate_metrics([])

        # Build trading calendar from available data
        trading_dates = self._build_trading_calendar(normalized)
        if not trading_dates:
            return calculate_metrics([])

        # Build settings with optional weight overrides
        settings = self._build_settings()

        # Generate signal dates (every step_days trading days)
        signal_dates = trading_dates[:: self.step_days]
        total_signals = len(signal_dates)

        all_trades: list[TradeResult] = []

        for idx, signal_date in enumerate(signal_dates):
            if on_progress is not None:
                pct = ((idx + 1) / total_signals) * 100.0
                on_progress(signal_date, pct)

            trades = self._evaluate_signal_date(
                signal_date, normalized, trading_dates, settings
            )
            all_trades.extend(trades)

        result = calculate_metrics(all_trades)
        result.lookback_period = self.lookback_period
        result.breakout_threshold = self.breakout_threshold
        result.holding_period = self.holding_period
        result.top_n = self.top_n

        return result

    def _normalize_data(
        self,
        tickers: list[str],
        ohlcv_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Ensure each DataFrame has a DatetimeIndex sorted by date."""
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            if ticker not in ohlcv_data:
                continue
            df = ohlcv_data[ticker].copy()

            # Convert Date column to index if needed
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            df = df.sort_index()

            # Require OHLCV columns
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(set(df.columns)):
                continue

            result[ticker] = df

        return result

    def _build_trading_calendar(
        self, normalized: dict[str, pd.DataFrame]
    ) -> list[date]:
        """Build sorted list of trading dates within the configured range.

        Uses the intersection of dates across all tickers, filtered to
        the configured date_range.
        """
        # Get union of all dates (any date where at least one ticker traded)
        all_dates: set[date] = set()
        for df in normalized.values():
            all_dates.update(d.date() for d in df.index)

        sorted_dates = sorted(all_dates)

        if not sorted_dates:
            return []

        # Apply date range filter
        if self.date_range is not None:
            start, end = self.date_range
            sorted_dates = [d for d in sorted_dates if start <= d <= end]

        # Require enough history for lookback + holding period
        if len(sorted_dates) <= self.lookback_period + self.holding_period:
            return []

        # Skip first lookback_period dates to ensure indicator history
        return sorted_dates[self.lookback_period :]

        return sorted_dates

    def _build_settings(self) -> Settings:
        """Create Settings instance with optional weight overrides."""
        kwargs: dict = {}
        if self.scoring_weights is not None:
            kwargs["scoring_weights"] = self.scoring_weights
        return Settings(**kwargs)

    def _evaluate_signal_date(
        self,
        signal_date: date,
        normalized: dict[str, pd.DataFrame],
        trading_dates: list[date],
        settings: Settings,
    ) -> list[TradeResult]:
        """Score universe on signal_date and measure outcomes.

        Only uses data up to signal_date (no look-ahead bias).
        """
        # Build sliced OHLCV data: only data up to and including signal_date
        sliced_data: dict[str, pd.DataFrame] = {}
        for ticker, df in normalized.items():
            # Slice: everything up to signal_date
            mask = df.index.date <= signal_date
            sliced = df.loc[mask]

            # Need enough history for indicators
            if len(sliced) < self.lookback_period:
                continue

            sliced_data[ticker] = sliced

        if not sliced_data:
            return []

        # Score the universe with only historical data
        scores = score_universe(sliced_data, settings)

        # Take top N candidates
        top_candidates = scores[: self.top_n]

        # Determine exit date
        signal_idx = _find_date_index(trading_dates, signal_date)
        if signal_idx is None:
            return []

        exit_idx = signal_idx + self.holding_period
        if exit_idx >= len(trading_dates):
            # Not enough future data for this signal
            return []

        exit_date = trading_dates[exit_idx]

        # Build trade results
        trades: list[TradeResult] = []
        for ticker_score in top_candidates:
            ticker = ticker_score.symbol
            if ticker not in normalized:
                continue

            df = normalized[ticker]
            entry_price = self._get_close_on_date(df, signal_date)
            exit_price = self._get_close_on_date(df, exit_date)

            if entry_price is None or exit_price is None:
                continue
            if entry_price <= 0:
                continue

            direction = ticker_score.direction
            return_pct = self._calculate_return(
                entry_price, exit_price, direction
            )

            was_winner = return_pct >= self.breakout_threshold

            trades.append(
                TradeResult(
                    ticker=ticker,
                    signal_date=signal_date,
                    direction=direction,
                    composite_score=ticker_score.composite_score,
                    entry_price=round(entry_price, 4),
                    exit_price=round(exit_price, 4),
                    return_pct=round(return_pct, 4),
                    holding_days=self.holding_period,
                    was_winner=was_winner,
                )
            )

        return trades

    @staticmethod
    def _get_close_on_date(df: pd.DataFrame, target_date: date) -> Optional[float]:
        """Get closing price on a specific date.

        Falls back to the nearest prior trading day if target_date is missing.
        """
        mask = df.index.date == target_date
        matching = df.loc[mask]
        if not matching.empty:
            return float(matching["Close"].iloc[-1])

        # Fallback: closest prior date
        prior = df.loc[df.index.date <= target_date]
        if prior.empty:
            return None
        return float(prior["Close"].iloc[-1])

    @staticmethod
    def _calculate_return(
        entry_price: float, exit_price: float, direction: Direction
    ) -> float:
        """Calculate return percentage, adjusted for trade direction.

        For BULLISH: positive return if price went up.
        For BEARISH: positive return if price went down.
        For NEUTRAL: use absolute return (long bias).
        """
        raw_return = (exit_price - entry_price) / entry_price * 100.0

        if direction == Direction.BEARISH:
            return -raw_return  # Profit from price decline
        # BULLISH or NEUTRAL: profit from price increase
        return raw_return


def _find_date_index(dates: list[date], target: date) -> Optional[int]:
    """Find the index of target in a sorted date list, or None if not found."""
    # Binary search for efficiency
    lo, hi = 0, len(dates) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if dates[mid] == target:
            return mid
        elif dates[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return None
