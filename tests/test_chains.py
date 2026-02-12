"""Tests for options chain fetching and filtering."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from option_alpha.config import Settings
from option_alpha.options.chains import (
    ChainData,
    fetch_chain,
    fetch_chains_for_tickers,
    get_available_expirations,
    select_expiration,
)


# ─── Select Expiration ───────────────────────────────────────────────


class TestSelectExpiration:
    def test_picks_closest_to_midpoint(self):
        ref = date(2025, 7, 1)
        expirations = [
            date(2025, 8, 1),   # 31 DTE
            date(2025, 8, 8),   # 38 DTE
            date(2025, 8, 15),  # 45 DTE  <- closest to midpoint (45)
            date(2025, 8, 22),  # 52 DTE
            date(2025, 8, 29),  # 59 DTE
        ]
        result = select_expiration(expirations, 30, 60, ref)
        assert result == date(2025, 8, 15)

    def test_filters_by_dte_range(self):
        ref = date(2025, 7, 1)
        expirations = [
            date(2025, 7, 10),  # 9 DTE  - too short
            date(2025, 7, 20),  # 19 DTE - too short
            date(2025, 8, 15),  # 45 DTE - in range
            date(2025, 10, 1),  # 92 DTE - too long
        ]
        result = select_expiration(expirations, 30, 60, ref)
        assert result == date(2025, 8, 15)

    def test_no_expirations_in_range(self):
        ref = date(2025, 7, 1)
        expirations = [
            date(2025, 7, 10),  # 9 DTE
            date(2025, 10, 1),  # 92 DTE
        ]
        result = select_expiration(expirations, 30, 60, ref)
        assert result is None

    def test_empty_expirations(self):
        result = select_expiration([], 30, 60, date(2025, 7, 1))
        assert result is None

    def test_single_valid_expiration(self):
        ref = date(2025, 7, 1)
        expirations = [date(2025, 8, 15)]
        result = select_expiration(expirations, 30, 60, ref)
        assert result == date(2025, 8, 15)

    def test_boundary_dte_min(self):
        ref = date(2025, 7, 1)
        expirations = [date(2025, 7, 31)]  # exactly 30 DTE
        result = select_expiration(expirations, 30, 60, ref)
        assert result == date(2025, 7, 31)

    def test_boundary_dte_max(self):
        ref = date(2025, 7, 1)
        expirations = [date(2025, 8, 30)]  # exactly 60 DTE
        result = select_expiration(expirations, 30, 60, ref)
        assert result == date(2025, 8, 30)

    def test_custom_dte_range(self):
        ref = date(2025, 7, 1)
        expirations = [
            date(2025, 7, 10),  # 9 DTE
            date(2025, 7, 15),  # 14 DTE
        ]
        result = select_expiration(expirations, 7, 21, ref)
        # Midpoint = 14, date(7,15) = 14 DTE is exact match
        assert result == date(2025, 7, 15)


# ─── Get Available Expirations ───────────────────────────────────────


class TestGetAvailableExpirations:
    @patch("option_alpha.options.chains._fetch_options_json")
    def test_returns_sorted_dates(self, mock_fetch):
        # Epoch timestamps for 2025-07-18, 2025-08-15, 2025-09-19
        mock_fetch.return_value = {
            "expirationDates": [1752796800, 1755216000, 1758240000]
        }

        result = get_available_expirations("AAPL")
        assert result == [
            date(2025, 7, 18),
            date(2025, 8, 15),
            date(2025, 9, 19),
        ]

    @patch("option_alpha.options.chains._fetch_options_json")
    def test_empty_options(self, mock_fetch):
        mock_fetch.return_value = {"expirationDates": []}

        result = get_available_expirations("NONE")
        assert result == []

    @patch("option_alpha.options.chains._fetch_options_json")
    def test_exception_returns_empty(self, mock_fetch):
        mock_fetch.side_effect = Exception("Network error")

        result = get_available_expirations("ERR")
        assert result == []


# ─── Fetch Chain ─────────────────────────────────────────────────────


class TestFetchChain:
    @patch("option_alpha.options.chains._fetch_options_json")
    def test_successful_fetch(self, mock_fetch):
        mock_fetch.return_value = {
            "quote": {"regularMarketPrice": 100.0},
            "options": [{
                "calls": [
                    {"strike": 95.0, "lastPrice": 8.0, "bid": 7.5, "ask": 8.5,
                     "volume": 100, "openInterest": 500},
                    {"strike": 100.0, "lastPrice": 5.0, "bid": 4.5, "ask": 5.5,
                     "volume": 200, "openInterest": 1000},
                    {"strike": 105.0, "lastPrice": 2.0, "bid": 1.5, "ask": 2.5,
                     "volume": 50, "openInterest": 300},
                ],
                "puts": [
                    {"strike": 95.0, "lastPrice": 2.0, "bid": 1.5, "ask": 2.5,
                     "volume": 50, "openInterest": 300},
                    {"strike": 100.0, "lastPrice": 5.0, "bid": 4.5, "ask": 5.5,
                     "volume": 200, "openInterest": 1000},
                    {"strike": 105.0, "lastPrice": 8.0, "bid": 7.5, "ask": 8.5,
                     "volume": 100, "openInterest": 500},
                ],
            }],
        }

        result = fetch_chain("AAPL", date(2025, 8, 15))
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.expiration == date(2025, 8, 15)
        assert result.underlying_price == 100.0
        assert len(result.calls) == 3
        assert len(result.puts) == 3

    @patch("option_alpha.options.chains._fetch_options_json")
    def test_failed_fetch_returns_none(self, mock_fetch):
        mock_fetch.side_effect = Exception("API error")

        result = fetch_chain("ERR", date(2025, 8, 15))
        assert result is None

    @patch("option_alpha.options.chains.yf.Ticker")
    @patch("option_alpha.options.chains._fetch_options_json")
    def test_fallback_price_from_fast_info(self, mock_fetch, mock_ticker_cls):
        mock_fetch.return_value = {
            "quote": {},  # No price in quote
            "options": [{"calls": [], "puts": []}],
        }
        mock_ticker = MagicMock()
        mock_ticker.fast_info.__getitem__ = MagicMock(return_value=150.0)
        mock_ticker_cls.return_value = mock_ticker

        result = fetch_chain("FB", date(2025, 8, 15))
        assert result is not None
        assert result.underlying_price == 150.0


# ─── Fetch Chains For Tickers ────────────────────────────────────────


class TestFetchChainsForTickers:
    @patch("option_alpha.options.chains.fetch_chain")
    @patch("option_alpha.options.chains.get_available_expirations")
    def test_fetches_multiple(self, mock_expirations, mock_fetch):
        ref = date(2025, 7, 1)
        mock_expirations.return_value = [date(2025, 8, 15)]

        mock_fetch.return_value = ChainData(
            symbol="AAPL",
            expiration=date(2025, 8, 15),
            dte=45,
            underlying_price=100.0,
        )

        settings = Settings()
        results = fetch_chains_for_tickers(
            ["AAPL", "MSFT"], settings, reference_date=ref,
        )
        assert len(results) == 2

    @patch("option_alpha.options.chains.get_available_expirations")
    def test_no_expirations_skipped(self, mock_expirations):
        mock_expirations.return_value = []

        results = fetch_chains_for_tickers(
            ["NONE"], reference_date=date(2025, 7, 1),
        )
        assert len(results) == 0

    @patch("option_alpha.options.chains.fetch_chain")
    @patch("option_alpha.options.chains.get_available_expirations")
    def test_no_valid_dte_skipped(self, mock_expirations, mock_fetch):
        ref = date(2025, 7, 1)
        # Only expirations outside DTE range
        mock_expirations.return_value = [date(2025, 7, 5)]  # 4 DTE

        results = fetch_chains_for_tickers(
            ["SHORT"], reference_date=ref,
        )
        assert len(results) == 0
        mock_fetch.assert_not_called()

    @patch("option_alpha.options.chains.fetch_chain")
    @patch("option_alpha.options.chains.get_available_expirations")
    def test_failed_chain_fetch_skipped(self, mock_expirations, mock_fetch):
        ref = date(2025, 7, 1)
        mock_expirations.return_value = [date(2025, 8, 15)]
        mock_fetch.return_value = None

        results = fetch_chains_for_tickers(
            ["FAIL"], reference_date=ref,
        )
        assert len(results) == 0
