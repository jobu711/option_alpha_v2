"""Tests for percentile normalization module."""

import numpy as np
import pytest

from option_alpha.scoring.normalizer import (
    normalize_universe,
    penalize_insufficient_data,
    percentile_rank,
)


class TestPercentileRank:
    def test_middle_value(self):
        values = [10, 20, 30, 40, 50]
        pct = percentile_rank(30, values)
        # 30 is the median (3rd of 5), should be around 50-60%
        assert 40 <= pct <= 70

    def test_lowest_value(self):
        values = [10, 20, 30, 40, 50]
        pct = percentile_rank(10, values)
        assert pct <= 25

    def test_highest_value(self):
        values = [10, 20, 30, 40, 50]
        pct = percentile_rank(50, values)
        assert pct >= 80

    def test_nan_value(self):
        pct = percentile_rank(float("nan"), [1, 2, 3])
        assert np.isnan(pct)

    def test_empty_universe(self):
        pct = percentile_rank(5.0, [])
        assert np.isnan(pct)

    def test_all_nan_universe(self):
        pct = percentile_rank(5.0, [float("nan"), float("nan")])
        assert np.isnan(pct)

    def test_single_value_universe(self):
        pct = percentile_rank(5.0, [5.0])
        # Only one value, it should be 100th percentile
        assert pct == pytest.approx(100.0)

    def test_uniform_distribution(self):
        values = list(range(1, 101))  # 1 to 100
        # 50 should be roughly 50th percentile
        pct = percentile_rank(50, values)
        assert 45 <= pct <= 55


class TestNormalizeUniverse:
    def test_basic_normalization(self):
        raw = {
            "AAPL": {"bb_width": 0.05, "rsi": 60.0},
            "MSFT": {"bb_width": 0.10, "rsi": 40.0},
            "GOOG": {"bb_width": 0.03, "rsi": 50.0},
        }
        normalized = normalize_universe(raw)

        # bb_width is inverted: lower = better, so GOOG (0.03) should score highest
        assert normalized["GOOG"]["bb_width"] > normalized["AAPL"]["bb_width"]
        assert normalized["AAPL"]["bb_width"] > normalized["MSFT"]["bb_width"]

        # rsi is NOT inverted by default: higher = higher percentile
        assert normalized["AAPL"]["rsi"] > normalized["GOOG"]["rsi"]
        assert normalized["GOOG"]["rsi"] > normalized["MSFT"]["rsi"]

    def test_all_scores_0_to_100(self):
        raw = {
            f"T{i}": {"bb_width": 0.01 * (i + 1), "rsi": 30 + i * 5}
            for i in range(20)
        }
        normalized = normalize_universe(raw)
        for ticker, indicators in normalized.items():
            for name, score in indicators.items():
                assert 0 <= score <= 100, f"{ticker}.{name} = {score}"

    def test_empty_input(self):
        assert normalize_universe({}) == {}

    def test_single_ticker(self):
        raw = {"AAPL": {"bb_width": 0.05, "rsi": 55.0}}
        normalized = normalize_universe(raw)
        assert "AAPL" in normalized
        # Single ticker: percentile should be 100 (only value)
        # But inverted bb_width: 100 - 100 = 0
        assert normalized["AAPL"]["bb_width"] == pytest.approx(0.0)
        assert normalized["AAPL"]["rsi"] == pytest.approx(100.0)

    def test_custom_invert(self):
        raw = {
            "A": {"x": 10, "y": 20},
            "B": {"x": 20, "y": 10},
        }
        # Invert only y: lower y = higher score
        normalized = normalize_universe(raw, invert={"y"})
        assert normalized["B"]["y"] > normalized["A"]["y"]
        # x not inverted: higher x = higher score
        assert normalized["B"]["x"] > normalized["A"]["x"]

    def test_keltner_width_inverted_by_default(self):
        """keltner_width should be in the default invert set (lower = tighter = better)."""
        raw = {
            "TIGHT": {"keltner_width": 0.03},
            "WIDE": {"keltner_width": 0.10},
        }
        normalized = normalize_universe(raw)
        # Lower keltner_width -> higher percentile (inverted)
        assert normalized["TIGHT"]["keltner_width"] > normalized["WIDE"]["keltner_width"]

    def test_nan_handling(self):
        raw = {
            "AAPL": {"bb_width": 0.05, "rsi": 60.0},
            "MSFT": {"bb_width": float("nan"), "rsi": 40.0},
        }
        normalized = normalize_universe(raw)
        assert np.isnan(normalized["MSFT"]["bb_width"])
        assert not np.isnan(normalized["AAPL"]["bb_width"])


class TestPenalizeInsufficientData:
    def test_nan_replaced(self):
        normalized = {
            "AAPL": {"bb_width": 80.0, "rsi": 60.0},
            "MSFT": {"bb_width": float("nan"), "rsi": 50.0},
        }
        result = penalize_insufficient_data(normalized)
        # NaN should be replaced with penalty * 50
        assert result["MSFT"]["bb_width"] == pytest.approx(25.0)
        assert result["AAPL"]["bb_width"] == pytest.approx(80.0)

    def test_no_nans_unchanged(self):
        normalized = {
            "AAPL": {"rsi": 70.0},
            "MSFT": {"rsi": 30.0},
        }
        result = penalize_insufficient_data(normalized)
        assert result["AAPL"]["rsi"] == pytest.approx(70.0)
        assert result["MSFT"]["rsi"] == pytest.approx(30.0)

    def test_custom_penalty(self):
        normalized = {"AAPL": {"rsi": float("nan")}}
        result = penalize_insufficient_data(normalized, penalty_factor=0.8)
        assert result["AAPL"]["rsi"] == pytest.approx(40.0)  # 0.8 * 50
