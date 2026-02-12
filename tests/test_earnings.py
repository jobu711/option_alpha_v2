"""Tests for catalyst earnings module."""

import math
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from option_alpha.catalysts.earnings import (
    DEFAULT_DECAY_CONSTANT,
    EarningsInfo,
    batch_earnings_info,
    compute_proximity_score,
    fetch_earnings_date,
    get_earnings_info,
    merge_catalyst_scores,
)
from option_alpha.config import Settings
from option_alpha.models import Direction, TickerScore


# ─── Proximity Score ─────────────────────────────────────────────────


class TestComputeProximityScore:
    def test_zero_days(self):
        """Earnings today should score 1.0."""
        assert compute_proximity_score(0) == pytest.approx(1.0)

    def test_three_days(self):
        """3 days out ~ high score."""
        score = compute_proximity_score(3)
        expected = math.exp(-3 / DEFAULT_DECAY_CONSTANT)
        assert score == pytest.approx(expected)
        assert score > 0.5

    def test_seven_days(self):
        """7 days out ~ medium score."""
        score = compute_proximity_score(7)
        expected = math.exp(-7 / DEFAULT_DECAY_CONSTANT)
        assert score == pytest.approx(expected)
        assert 0.1 < score < 0.5

    def test_fourteen_days(self):
        """14 days out ~ low score."""
        score = compute_proximity_score(14)
        expected = math.exp(-14 / DEFAULT_DECAY_CONSTANT)
        assert score == pytest.approx(expected)
        assert score < 0.1

    def test_negative_days(self):
        """Past earnings should score 0."""
        assert compute_proximity_score(-1) == 0.0
        assert compute_proximity_score(-30) == 0.0

    def test_large_days(self):
        """Very distant earnings should approach 0."""
        score = compute_proximity_score(100)
        assert score < 0.001

    def test_custom_decay_constant(self):
        """Custom decay constant changes the curve."""
        score_fast = compute_proximity_score(5, decay_constant=2.0)
        score_slow = compute_proximity_score(5, decay_constant=10.0)
        assert score_slow > score_fast

    def test_monotonically_decreasing(self):
        """Score must decrease as days increase."""
        prev = 1.0
        for d in range(1, 30):
            score = compute_proximity_score(d)
            assert score < prev
            prev = score


# ─── Fetch Earnings Date ─────────────────────────────────────────────


class TestFetchEarningsDate:
    @patch("option_alpha.catalysts.earnings.yf.Ticker")
    def test_future_earnings_returned(self, mock_ticker_cls):
        """get_earnings_dates returns future date correctly."""
        import pandas as pd

        mock_ticker = MagicMock()
        # Simulate get_earnings_dates returning a DataFrame with future dates.
        future_dt = datetime(2099, 7, 15, 17, 0, tzinfo=timezone.utc)
        past_dt = datetime(2020, 1, 15, 17, 0, tzinfo=timezone.utc)
        mock_ticker.get_earnings_dates.return_value = pd.DataFrame(
            {"EPS Estimate": [1.5, 1.2]},
            index=pd.DatetimeIndex([future_dt, past_dt]),
        )
        mock_ticker_cls.return_value = mock_ticker

        result = fetch_earnings_date("AAPL")
        assert result == date(2099, 7, 15)

    @patch("option_alpha.catalysts.earnings.yf.Ticker")
    def test_only_past_earnings_returns_none(self, mock_ticker_cls):
        """All earnings dates in the past returns None."""
        import pandas as pd

        mock_ticker = MagicMock()
        past_dt = datetime(2020, 1, 15, 17, 0, tzinfo=timezone.utc)
        mock_ticker.get_earnings_dates.return_value = pd.DataFrame(
            {"EPS Estimate": [1.2]},
            index=pd.DatetimeIndex([past_dt]),
        )
        mock_ticker_cls.return_value = mock_ticker

        result = fetch_earnings_date("OLD")
        assert result is None

    @patch("option_alpha.catalysts.earnings.yf.Ticker")
    def test_none_dataframe(self, mock_ticker_cls):
        """Ticker with no earnings data."""
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_dates.return_value = None
        mock_ticker_cls.return_value = mock_ticker

        result = fetch_earnings_date("UNKNOWN")
        assert result is None

    @patch("option_alpha.catalysts.earnings.yf.Ticker")
    def test_empty_dataframe(self, mock_ticker_cls):
        """get_earnings_dates returns empty DataFrame."""
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.get_earnings_dates.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = fetch_earnings_date("XYZ")
        assert result is None

    @patch("option_alpha.catalysts.earnings.yf.Ticker")
    def test_exception_handling(self, mock_ticker_cls):
        """Exception during fetch returns None."""
        mock_ticker_cls.side_effect = Exception("Network error")

        result = fetch_earnings_date("ERR")
        assert result is None

    @patch("option_alpha.catalysts.earnings.yf.Ticker")
    def test_picks_earliest_future_date(self, mock_ticker_cls):
        """When multiple future dates exist, picks the earliest."""
        import pandas as pd

        mock_ticker = MagicMock()
        dt1 = datetime(2099, 4, 15, 17, 0, tzinfo=timezone.utc)
        dt2 = datetime(2099, 7, 15, 17, 0, tzinfo=timezone.utc)
        mock_ticker.get_earnings_dates.return_value = pd.DataFrame(
            {"EPS Estimate": [1.5, 1.2]},
            index=pd.DatetimeIndex([dt1, dt2]),
        )
        mock_ticker_cls.return_value = mock_ticker

        result = fetch_earnings_date("ALT")
        assert result == date(2099, 4, 15)


# ─── Get Earnings Info ───────────────────────────────────────────────


class TestGetEarningsInfo:
    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_upcoming_earnings_3_days(self, mock_fetch):
        ref = date(2025, 7, 1)
        mock_fetch.return_value = date(2025, 7, 4)

        info = get_earnings_info("AAPL", reference_date=ref)
        assert info.symbol == "AAPL"
        assert info.days_until == 3
        assert info.proximity_score == pytest.approx(math.exp(-3 / DEFAULT_DECAY_CONSTANT))
        assert info.iv_crush_warning is True

    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_upcoming_earnings_14_days(self, mock_fetch):
        ref = date(2025, 7, 1)
        mock_fetch.return_value = date(2025, 7, 15)

        info = get_earnings_info("MSFT", reference_date=ref)
        assert info.days_until == 14
        assert info.proximity_score < 0.1
        assert info.iv_crush_warning is False

    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_earnings_within_7_days_crush_warning(self, mock_fetch):
        ref = date(2025, 7, 1)
        mock_fetch.return_value = date(2025, 7, 8)  # 7 days

        info = get_earnings_info("GOOG", reference_date=ref)
        assert info.iv_crush_warning is True

    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_earnings_8_days_no_crush_warning(self, mock_fetch):
        ref = date(2025, 7, 1)
        mock_fetch.return_value = date(2025, 7, 9)  # 8 days

        info = get_earnings_info("GOOG", reference_date=ref)
        assert info.iv_crush_warning is False

    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_past_earnings(self, mock_fetch):
        ref = date(2025, 7, 10)
        mock_fetch.return_value = date(2025, 7, 5)

        info = get_earnings_info("OLD", reference_date=ref)
        assert info.days_until == -5
        assert info.proximity_score == 0.0
        assert info.iv_crush_warning is False

    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_no_earnings_data(self, mock_fetch):
        mock_fetch.return_value = None

        info = get_earnings_info("NONE", reference_date=date(2025, 7, 1))
        assert info.earnings_date is None
        assert info.days_until is None
        assert info.proximity_score == 0.0
        assert info.iv_crush_warning is False

    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_earnings_today(self, mock_fetch):
        ref = date(2025, 7, 1)
        mock_fetch.return_value = date(2025, 7, 1)

        info = get_earnings_info("TODAY", reference_date=ref)
        assert info.days_until == 0
        assert info.proximity_score == pytest.approx(1.0)
        assert info.iv_crush_warning is True


# ─── Batch Earnings Info ─────────────────────────────────────────────


class TestBatchEarningsInfo:
    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_batch_multiple(self, mock_fetch):
        ref = date(2025, 7, 1)
        mock_fetch.side_effect = [
            date(2025, 7, 3),
            None,
            date(2025, 7, 20),
        ]

        results = batch_earnings_info(["A", "B", "C"], reference_date=ref)
        assert len(results) == 3
        assert results["A"].days_until == 2
        assert results["B"].earnings_date is None
        assert results["C"].days_until == 19

    @patch("option_alpha.catalysts.earnings.fetch_earnings_date")
    def test_batch_empty(self, mock_fetch):
        results = batch_earnings_info([], reference_date=date(2025, 7, 1))
        assert results == {}


# ─── Merge Catalyst Scores ───────────────────────────────────────────


class TestMergeCatalystScores:
    def _make_ticker_score(self, symbol: str, score: float) -> TickerScore:
        return TickerScore(
            symbol=symbol,
            composite_score=score,
            direction=Direction.BULLISH,
        )

    def test_merge_with_catalyst(self):
        scores = [
            self._make_ticker_score("AAPL", 60.0),
            self._make_ticker_score("MSFT", 50.0),
        ]
        earnings = {
            "AAPL": EarningsInfo(
                symbol="AAPL",
                earnings_date=date(2025, 7, 3),
                days_until=2,
                proximity_score=0.67,
            ),
            "MSFT": EarningsInfo(symbol="MSFT"),
        }

        settings = Settings()
        merged = merge_catalyst_scores(scores, earnings, settings)

        aapl = next(s for s in merged if s.symbol == "AAPL")
        msft = next(s for s in merged if s.symbol == "MSFT")

        # AAPL should have blended score using the actual catalyst_proximity weight
        catalyst_weight = settings.scoring_weights.get("catalyst_proximity", 0.25)
        expected_aapl = (1 - catalyst_weight) * 60.0 + catalyst_weight * (0.67 * 100.0)
        assert aapl.composite_score == pytest.approx(expected_aapl, abs=0.01)

        # MSFT unchanged (no earnings data)
        assert msft.composite_score == 50.0

    def test_merge_sorted_descending(self):
        scores = [
            self._make_ticker_score("LOW", 30.0),
            self._make_ticker_score("HIGH", 40.0),
        ]
        earnings = {
            "LOW": EarningsInfo(
                symbol="LOW", proximity_score=0.9,
                earnings_date=date(2025, 7, 1), days_until=1,
            ),
            "HIGH": EarningsInfo(symbol="HIGH"),
        }

        merged = merge_catalyst_scores(scores, earnings)
        # LOW should now score higher due to catalyst boost
        assert merged[0].composite_score >= merged[1].composite_score

    def test_merge_no_earnings_unchanged(self):
        scores = [self._make_ticker_score("X", 75.0)]
        earnings: dict[str, EarningsInfo] = {}

        merged = merge_catalyst_scores(scores, earnings)
        assert merged[0].composite_score == 75.0

    def test_merge_clamps_to_100(self):
        scores = [self._make_ticker_score("MAX", 99.0)]
        earnings = {
            "MAX": EarningsInfo(
                symbol="MAX", proximity_score=1.0,
                earnings_date=date(2025, 7, 1), days_until=0,
            ),
        }

        merged = merge_catalyst_scores(scores, earnings)
        assert merged[0].composite_score <= 100.0
