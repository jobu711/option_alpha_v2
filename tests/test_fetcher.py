"""Tests for yfinance batch data fetcher (mocked yfinance)."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from option_alpha.data.fetcher import (
    _parse_ticker_data,
    fetch_batch,
    fetch_single,
    retry_with_backoff,
)
from option_alpha.models import TickerData


def _make_ohlcv_df(symbols: list[str], rows: int = 20) -> pd.DataFrame:
    """Create a mock multi-ticker OHLCV DataFrame like yfinance returns."""
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")

    if len(symbols) == 1:
        # Single ticker: no MultiIndex
        return pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(rows)],
                "High": [102.0 + i for i in range(rows)],
                "Low": [99.0 + i for i in range(rows)],
                "Close": [101.0 + i for i in range(rows)],
                "Volume": [1000000 + i * 1000 for i in range(rows)],
            },
            index=dates,
        )

    # Multiple tickers: MultiIndex columns
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], symbols]
    )
    data = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        for sym in symbols:
            base = 100.0 if field != "Volume" else 1000000
            data[(field, sym)] = [base + i for i in range(rows)]

    return pd.DataFrame(data, index=dates, columns=columns)


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, delays=[0, 0, 0])
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeeds() == "ok"
        assert call_count == 1

    def test_retries_on_failure(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, delays=[0, 0, 0])
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        assert fails_twice() == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        @retry_with_backoff(max_retries=2, delays=[0, 0])
        def always_fails():
            raise ValueError("permanent fail")

        with pytest.raises(ValueError, match="permanent fail"):
            always_fails()


class TestParseTickerData:
    def test_parse_single_ticker(self):
        df = _make_ohlcv_df(["AAPL"], rows=10)
        result = _parse_ticker_data("AAPL", df, is_single=True)

        assert result is not None
        assert result.symbol == "AAPL"
        assert len(result.close) == 10
        assert result.last_price == result.close[-1]

    def test_parse_multi_ticker(self):
        df = _make_ohlcv_df(["AAPL", "MSFT"], rows=15)
        result = _parse_ticker_data("AAPL", df, is_single=False)

        assert result is not None
        assert result.symbol == "AAPL"
        assert len(result.close) == 15

    def test_parse_missing_symbol_returns_none(self):
        df = _make_ohlcv_df(["AAPL", "GOOGL"], rows=10)
        result = _parse_ticker_data("MSFT", df, is_single=False)
        assert result is None

    def test_parse_insufficient_data_returns_none(self):
        df = _make_ohlcv_df(["AAPL"], rows=3)  # less than 5 rows
        result = _parse_ticker_data("AAPL", df, is_single=True)
        assert result is None


class TestFetchBatch:
    @patch("option_alpha.data.fetcher._download_batch")
    def test_fetch_batch_single(self, mock_download):
        mock_download.return_value = _make_ohlcv_df(["AAPL"])
        results = fetch_batch(["AAPL"], batch_size=50)

        assert "AAPL" in results
        assert isinstance(results["AAPL"], TickerData)
        assert len(results["AAPL"].close) == 20

    @patch("option_alpha.data.fetcher._download_batch")
    def test_fetch_batch_multiple(self, mock_download):
        mock_download.return_value = _make_ohlcv_df(["AAPL", "MSFT", "GOOGL"])
        results = fetch_batch(["AAPL", "MSFT", "GOOGL"], batch_size=50)

        assert len(results) == 3
        for sym in ["AAPL", "MSFT", "GOOGL"]:
            assert sym in results

    @patch("option_alpha.data.fetcher._download_batch")
    def test_fetch_batch_splits_into_batches(self, mock_download):
        symbols = [f"SYM{i}" for i in range(10)]
        mock_download.return_value = _make_ohlcv_df(symbols[:5])

        results = fetch_batch(symbols, batch_size=5, max_workers=2)
        # Should have called download twice (10 symbols / 5 per batch)
        assert mock_download.call_count == 2

    @patch("option_alpha.data.fetcher._download_batch")
    def test_fetch_batch_empty(self, mock_download):
        results = fetch_batch([])
        assert results == {}
        mock_download.assert_not_called()

    @patch("option_alpha.data.fetcher._download_batch")
    def test_fetch_batch_handles_download_failure(self, mock_download):
        mock_download.side_effect = Exception("Network error")
        results = fetch_batch(["AAPL"], batch_size=50)
        assert results == {}


class TestFetchSingle:
    @patch("option_alpha.data.fetcher._download_batch")
    def test_fetch_single_success(self, mock_download):
        mock_download.return_value = _make_ohlcv_df(["TSLA"])
        result = fetch_single("TSLA")

        assert result is not None
        assert result.symbol == "TSLA"

    @patch("option_alpha.data.fetcher._download_batch")
    def test_fetch_single_failure(self, mock_download):
        mock_download.side_effect = Exception("Error")
        result = fetch_single("TSLA")
        assert result is None
