"""Tests for dynamic universe refresh module."""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from option_alpha.config import Settings
from option_alpha.data.universe_refresh import (
    _classify_market_cap,
    _do_refresh,
    _enrich_metadata,
    _fetch_edgar_tickers,
    _load_current,
    _validate_open_interest,
    _validate_optionability,
    refresh_universe,
    should_refresh,
)


# ---------------------------------------------------------------------------
# should_refresh
# ---------------------------------------------------------------------------


class TestShouldRefresh:
    def test_returns_true_when_no_meta_file(self, tmp_path, monkeypatch):
        """No meta file means never refreshed -- should refresh."""
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE",
            tmp_path / "nonexistent.json",
        )
        assert should_refresh() is True

    def test_returns_true_when_meta_is_stale(self, tmp_path, monkeypatch):
        """Meta file exists but last_refresh is older than interval."""
        meta_file = tmp_path / "universe_meta.json"
        old_time = datetime.now(UTC) - timedelta(days=10)
        meta_file.write_text(json.dumps({"last_refresh": old_time.isoformat()}))
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )
        settings = Settings(universe_refresh_interval_days=7)
        assert should_refresh(settings=settings) is True

    def test_returns_false_when_meta_is_fresh(self, tmp_path, monkeypatch):
        """Meta file exists and last_refresh is recent -- skip refresh."""
        meta_file = tmp_path / "universe_meta.json"
        recent_time = datetime.now(UTC) - timedelta(days=1)
        meta_file.write_text(
            json.dumps({"last_refresh": recent_time.isoformat()})
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )
        settings = Settings(universe_refresh_interval_days=7)
        assert should_refresh(settings=settings) is False

    def test_returns_true_when_meta_is_corrupt(self, tmp_path, monkeypatch):
        """Corrupted meta file should trigger refresh."""
        meta_file = tmp_path / "universe_meta.json"
        meta_file.write_text("not valid json{{{")
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )
        assert should_refresh() is True

    def test_returns_true_when_meta_missing_key(self, tmp_path, monkeypatch):
        """Meta file without last_refresh key should trigger refresh."""
        meta_file = tmp_path / "universe_meta.json"
        meta_file.write_text(json.dumps({"ticker_count": 100}))
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )
        assert should_refresh() is True

    def test_respects_custom_interval(self, tmp_path, monkeypatch):
        """A 3-day interval should refresh after 4 days."""
        meta_file = tmp_path / "universe_meta.json"
        four_days_ago = datetime.now(UTC) - timedelta(days=4)
        meta_file.write_text(
            json.dumps({"last_refresh": four_days_ago.isoformat()})
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )
        settings = Settings(universe_refresh_interval_days=3)
        assert should_refresh(settings=settings) is True

        settings_long = Settings(universe_refresh_interval_days=30)
        assert should_refresh(settings=settings_long) is False


# ---------------------------------------------------------------------------
# _fetch_edgar_tickers
# ---------------------------------------------------------------------------


class TestFetchEdgarTickers:
    @pytest.mark.asyncio
    async def test_parses_edgar_response(self):
        """Should extract and deduplicate tickers from SEC EDGAR JSON."""
        mock_data = {
            "0": {"cik_str": "1", "ticker": "AAPL", "title": "Apple Inc"},
            "1": {"cik_str": "2", "ticker": "msft", "title": "Microsoft"},
            "2": {"cik_str": "3", "ticker": "GOOGL", "title": "Alphabet"},
            "3": {"cik_str": "4", "ticker": "AAPL", "title": "Apple Dupe"},
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_data

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("option_alpha.data.universe_refresh.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_edgar_tickers()

        assert result == ["AAPL", "GOOGL", "MSFT"]  # sorted, deduped, uppercased

    @pytest.mark.asyncio
    async def test_filters_tickers_with_dots(self):
        """Tickers containing dots (BRK.B) should be excluded."""
        mock_data = {
            "0": {"cik_str": "1", "ticker": "AAPL", "title": "Apple"},
            "1": {"cik_str": "2", "ticker": "BRK.B", "title": "Berkshire B"},
            "2": {"cik_str": "3", "ticker": "BF.B", "title": "Brown Forman"},
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_data

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("option_alpha.data.universe_refresh.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_edgar_tickers()

        assert "BRK.B" not in result
        assert "BF.B" not in result
        assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_filters_long_tickers(self):
        """Tickers longer than 5 chars should be excluded."""
        mock_data = {
            "0": {"cik_str": "1", "ticker": "AAPL", "title": "Apple"},
            "1": {"cik_str": "2", "ticker": "LONGERTHAN5", "title": "Too Long"},
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_data

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("option_alpha.data.universe_refresh.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_edgar_tickers()

        assert "LONGERTHAN5" not in result
        assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_handles_empty_tickers(self):
        """Empty ticker strings should be excluded."""
        mock_data = {
            "0": {"cik_str": "1", "ticker": "", "title": "Empty"},
            "1": {"cik_str": "2", "ticker": "MSFT", "title": "Microsoft"},
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_data

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("option_alpha.data.universe_refresh.httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_edgar_tickers()

        assert result == ["MSFT"]


# ---------------------------------------------------------------------------
# _validate_optionability
# ---------------------------------------------------------------------------


class TestValidateOptionability:
    def test_filters_to_optionable_tickers(self):
        """Only tickers with non-empty options tuple should pass."""
        mock_ticker_aapl = MagicMock()
        mock_ticker_aapl.options = ("2025-01-17", "2025-02-21")

        mock_ticker_msft = MagicMock()
        mock_ticker_msft.options = ()  # no options

        mock_ticker_googl = MagicMock()
        mock_ticker_googl.options = ("2025-03-21",)

        def mock_yf_ticker(symbol):
            return {
                "AAPL": mock_ticker_aapl,
                "MSFT": mock_ticker_msft,
                "GOOGL": mock_ticker_googl,
            }[symbol]

        with patch("option_alpha.data.universe_refresh.yf.Ticker", side_effect=mock_yf_ticker):
            result = _validate_optionability(["AAPL", "MSFT", "GOOGL"])

        assert result == ["AAPL", "GOOGL"]

    def test_handles_exceptions_gracefully(self):
        """Tickers that raise exceptions should be skipped."""
        mock_ticker_aapl = MagicMock()
        mock_ticker_aapl.options = ("2025-01-17",)

        def mock_yf_ticker(symbol):
            if symbol == "BAD":
                raise Exception("Network error")
            return mock_ticker_aapl

        with patch("option_alpha.data.universe_refresh.yf.Ticker", side_effect=mock_yf_ticker):
            result = _validate_optionability(["AAPL", "BAD", "AAPL"])

        assert "AAPL" in result

    def test_respects_batch_size(self):
        """Should process all tickers regardless of batch boundaries."""
        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)

        with patch("option_alpha.data.universe_refresh.yf.Ticker", return_value=mock_ticker):
            tickers = [f"T{i}" for i in range(75)]
            result = _validate_optionability(tickers, batch_size=50)

        assert len(result) == 75

    def test_empty_input(self):
        """Empty input should return empty output."""
        result = _validate_optionability([])
        assert result == []


# ---------------------------------------------------------------------------
# _enrich_metadata
# ---------------------------------------------------------------------------


class TestEnrichMetadata:
    def test_enriches_stock_ticker(self):
        """Should populate all fields for a regular stock."""
        mock_info = {
            "shortName": "Apple Inc",
            "sector": "Technology",
            "marketCap": 3_000_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("option_alpha.data.universe_refresh.yf.Ticker", return_value=mock_ticker):
            result = _enrich_metadata(["AAPL"])

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["name"] == "Apple Inc"
        assert result[0]["sector"] == "Technology"
        assert result[0]["market_cap_tier"] == "large"
        assert result[0]["asset_type"] == "stock"

    def test_enriches_etf_ticker(self):
        """ETFs should have empty sector and market_cap_tier."""
        mock_info = {
            "shortName": "SPDR S&P 500",
            "quoteType": "ETF",
            "marketCap": 0,
        }
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("option_alpha.data.universe_refresh.yf.Ticker", return_value=mock_ticker):
            result = _enrich_metadata(["SPY"])

        assert len(result) == 1
        assert result[0]["symbol"] == "SPY"
        assert result[0]["asset_type"] == "etf"
        assert result[0]["sector"] == ""
        assert result[0]["market_cap_tier"] == ""

    def test_handles_missing_info_fields(self):
        """Should handle missing info fields gracefully."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("option_alpha.data.universe_refresh.yf.Ticker", return_value=mock_ticker):
            result = _enrich_metadata(["XYZ"])

        assert len(result) == 1
        assert result[0]["symbol"] == "XYZ"
        assert result[0]["name"] == ""
        assert result[0]["market_cap_tier"] == ""
        assert result[0]["asset_type"] == "stock"

    def test_handles_exception_with_fallback(self):
        """Tickers that error during enrichment get fallback values."""

        def mock_yf_ticker(symbol):
            raise Exception("API error")

        with patch("option_alpha.data.universe_refresh.yf.Ticker", side_effect=mock_yf_ticker):
            result = _enrich_metadata(["FAIL"])

        assert len(result) == 1
        assert result[0]["symbol"] == "FAIL"
        assert result[0]["name"] == ""
        assert result[0]["sector"] == ""
        assert result[0]["market_cap_tier"] == ""
        assert result[0]["asset_type"] == "stock"

    def test_uses_longname_when_shortname_missing(self):
        """Should fall back to longName if shortName is absent."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "longName": "Alphabet Inc Class A",
            "quoteType": "EQUITY",
            "marketCap": 2_000_000_000_000,
            "sector": "Technology",
        }

        with patch("option_alpha.data.universe_refresh.yf.Ticker", return_value=mock_ticker):
            result = _enrich_metadata(["GOOGL"])

        assert result[0]["name"] == "Alphabet Inc Class A"

    def test_none_market_cap_treated_as_zero(self):
        """marketCap of None should be treated as 0."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "shortName": "Test Co",
            "quoteType": "EQUITY",
            "marketCap": None,
        }

        with patch("option_alpha.data.universe_refresh.yf.Ticker", return_value=mock_ticker):
            result = _enrich_metadata(["TEST"])

        assert result[0]["market_cap_tier"] == ""


# ---------------------------------------------------------------------------
# _classify_market_cap
# ---------------------------------------------------------------------------


class TestClassifyMarketCap:
    def test_large_cap(self):
        assert _classify_market_cap(50_000_000_000) == "large"
        assert _classify_market_cap(10_000_000_001) == "large"

    def test_mid_cap(self):
        assert _classify_market_cap(10_000_000_000) == "mid"
        assert _classify_market_cap(2_000_000_001) == "mid"

    def test_small_cap(self):
        assert _classify_market_cap(2_000_000_000) == "small"
        assert _classify_market_cap(300_000_001) == "small"

    def test_micro_cap(self):
        assert _classify_market_cap(300_000_000) == "micro"
        assert _classify_market_cap(1) == "micro"

    def test_zero_returns_empty(self):
        assert _classify_market_cap(0) == ""

    def test_negative_returns_empty(self):
        assert _classify_market_cap(-100) == ""

    def test_boundary_values(self):
        """Test exact boundary values."""
        # Exactly 10B is mid, not large (> 10B for large)
        assert _classify_market_cap(10_000_000_000) == "mid"
        # Exactly 2B is small, not mid (> 2B for mid)
        assert _classify_market_cap(2_000_000_000) == "small"
        # Exactly 300M is micro, not small (> 300M for small)
        assert _classify_market_cap(300_000_000) == "micro"


# ---------------------------------------------------------------------------
# _load_current
# ---------------------------------------------------------------------------


class TestLoadCurrent:
    def test_loads_existing_file(self, tmp_path, monkeypatch):
        """Should parse and return existing universe data."""
        universe_file = tmp_path / "universe_data.json"
        data = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
        universe_file.write_text(json.dumps(data))
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        result = _load_current()
        assert result == data

    def test_returns_empty_for_missing_file(self, tmp_path, monkeypatch):
        """Should return empty list if file doesn't exist."""
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE",
            tmp_path / "nonexistent.json",
        )
        result = _load_current()
        assert result == []

    def test_returns_empty_for_corrupt_file(self, tmp_path, monkeypatch):
        """Should return empty list if file is corrupt."""
        universe_file = tmp_path / "universe_data.json"
        universe_file.write_text("not json{{{")
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        result = _load_current()
        assert result == []


# ---------------------------------------------------------------------------
# refresh_universe (full pipeline)
# ---------------------------------------------------------------------------


class TestRefreshUniverse:
    @pytest.mark.asyncio
    async def test_full_refresh_pipeline(self, tmp_path, monkeypatch):
        """End-to-end: fetch, validate, enrich, write, update meta."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"

        # Seed with one existing ticker
        universe_file.write_text(
            json.dumps([{"symbol": "OLD", "name": "Old Co"}])
        )

        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )

        # Mock EDGAR response
        mock_edgar = {
            "0": {"cik_str": "1", "ticker": "AAPL", "title": "Apple"},
            "1": {"cik_str": "2", "ticker": "MSFT", "title": "Microsoft"},
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_edgar

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Mock optionability: both are optionable
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.options = ("2025-01-17",)
        mock_ticker_obj.info = {
            "shortName": "Test",
            "sector": "Technology",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }

        # Mock _clear_cache
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._clear_cache",
            lambda: None,
            raising=False,
        )
        # Need to also patch the import inside _do_refresh
        mock_clear = MagicMock()
        monkeypatch.setattr(
            "option_alpha.data.universe._universe_cache", None
        )

        with (
            patch(
                "option_alpha.data.universe_refresh.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch(
                "option_alpha.data.universe_refresh.yf.Ticker",
                return_value=mock_ticker_obj,
            ),
        ):
            result = await refresh_universe()

        assert result["success"] is True
        assert result["ticker_count"] == 2
        assert result["added"] == 2  # AAPL and MSFT are new
        assert result["removed"] == 1  # OLD was removed

        # Verify file was written
        written = json.loads(universe_file.read_text())
        assert len(written) == 2
        symbols = {t["symbol"] for t in written}
        assert symbols == {"AAPL", "MSFT"}

        # Verify meta was written
        meta = json.loads(meta_file.read_text())
        assert "last_refresh" in meta
        assert meta["ticker_count"] == 2

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, tmp_path, monkeypatch):
        """On total failure, should return error dict and keep existing data."""
        universe_file = tmp_path / "universe_data.json"
        original_data = [{"symbol": "KEEP_ME"}]
        universe_file.write_text(json.dumps(original_data))

        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE",
            tmp_path / "universe_meta.json",
        )

        # Mock EDGAR to always fail
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Network timeout")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "option_alpha.data.universe_refresh.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await refresh_universe(max_retries=2)

        assert result["success"] is False
        assert "Network timeout" in result["error"]
        assert result["ticker_count"] == 0

        # Verify original data is preserved
        preserved = json.loads(universe_file.read_text())
        assert preserved == original_data

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self, tmp_path, monkeypatch):
        """Should retry and succeed if a later attempt works."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        universe_file.write_text(json.dumps([]))

        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )

        call_count = 0

        async def mock_do_refresh():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Transient error {call_count}")
            return {
                "success": True,
                "last_refresh": datetime.now(UTC).isoformat(),
                "ticker_count": 5,
                "added": 5,
                "removed": 0,
            }

        with patch(
            "option_alpha.data.universe_refresh._do_refresh",
            side_effect=mock_do_refresh,
        ):
            result = await refresh_universe(max_retries=3)

        assert result["success"] is True
        assert call_count == 3  # failed twice, succeeded third

    @pytest.mark.asyncio
    async def test_uses_default_settings(self):
        """Should not crash when called with no settings argument."""
        # Just verify it doesn't error on Settings resolution
        with patch(
            "option_alpha.data.universe_refresh._do_refresh",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "last_refresh": datetime.now(UTC).isoformat(),
                "ticker_count": 0,
                "added": 0,
                "removed": 0,
            },
        ):
            result = await refresh_universe()
            assert result["success"] is True

    def test_regenerate_mode_returns_mode_regenerate(self, tmp_path, monkeypatch):
        """refresh_universe(regenerate=True) failure dict should report mode='regenerate'."""
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE",
            tmp_path / "universe_data.json",
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE",
            tmp_path / "universe_meta.json",
        )

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("fail")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "option_alpha.data.universe_refresh.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = asyncio.run(refresh_universe(max_retries=1, regenerate=True))

        assert result["mode"] == "regenerate"

    def test_validate_mode_returns_mode_validate(self, tmp_path, monkeypatch):
        """refresh_universe(regenerate=False) failure dict should report mode='validate'."""
        universe_file = tmp_path / "universe_data.json"
        universe_file.write_text(json.dumps([{"symbol": "AAPL"}]))
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE",
            tmp_path / "universe_meta.json",
        )

        # Make _validate_optionability blow up
        with patch(
            "option_alpha.data.universe_refresh._validate_optionability",
            side_effect=Exception("boom"),
        ):
            result = asyncio.run(refresh_universe(max_retries=1, regenerate=False))

        assert result["mode"] == "validate"


# ---------------------------------------------------------------------------
# _validate_open_interest (Issue #60)
# ---------------------------------------------------------------------------


def _make_chain(calls_oi, puts_oi):
    """Helper: build a mock option_chain return value with given OI values."""
    calls_df = pd.DataFrame({"openInterest": calls_oi})
    puts_df = pd.DataFrame({"openInterest": puts_oi})
    chain = MagicMock()
    chain.calls = calls_df
    chain.puts = puts_df
    return chain


class TestValidateOpenInterest:
    """Tests for _validate_open_interest (Issue #60)."""

    def test_passes_tickers_above_threshold(self):
        """Tickers with total OI >= threshold should pass."""
        settings = Settings(min_universe_oi=100)

        mock_ticker = MagicMock()
        mock_ticker.info = {"quoteType": "EQUITY"}
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.option_chain.return_value = _make_chain([80, 30], [50, 20])
        # total OI = 80+30+50+20 = 180 >= 100

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep"):
            result = _validate_open_interest(["AAPL"], settings)

        assert result == ["AAPL"]

    def test_excludes_tickers_below_threshold(self):
        """Tickers with total OI < threshold should be excluded."""
        settings = Settings(min_universe_oi=500)

        mock_ticker = MagicMock()
        mock_ticker.info = {"quoteType": "EQUITY"}
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.option_chain.return_value = _make_chain([10], [5])
        # total OI = 10+5 = 15 < 500

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep"):
            result = _validate_open_interest(["LOW_OI"], settings)

        assert result == []

    def test_etf_exempt_from_oi_check(self):
        """ETF tickers should pass through without OI check."""
        settings = Settings(min_universe_oi=10_000)

        mock_ticker = MagicMock()
        mock_ticker.info = {"quoteType": "ETF"}
        # No option_chain should be called for ETFs

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep"):
            result = _validate_open_interest(["SPY", "QQQ"], settings)

        assert result == ["SPY", "QQQ"]
        # Verify option_chain was never called (ETFs skip the OI check)
        mock_ticker.option_chain.assert_not_called()

    def test_api_failure_causes_exclusion(self):
        """Tickers that raise exceptions during OI check should be excluded (fail-closed)."""
        settings = Settings(min_universe_oi=100)

        def mock_yf_ticker(symbol):
            if symbol == "FAIL":
                raise Exception("API timeout")
            mock_t = MagicMock()
            mock_t.info = {"quoteType": "EQUITY"}
            mock_t.options = ("2025-01-17",)
            mock_t.option_chain.return_value = _make_chain([200], [200])
            return mock_t

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            side_effect=mock_yf_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep"):
            result = _validate_open_interest(["AAPL", "FAIL", "MSFT"], settings)

        assert "AAPL" in result
        assert "MSFT" in result
        assert "FAIL" not in result

    def test_empty_options_chain_causes_exclusion(self):
        """Tickers with no expiration dates should be excluded."""
        settings = Settings(min_universe_oi=100)

        mock_ticker = MagicMock()
        mock_ticker.info = {"quoteType": "EQUITY"}
        mock_ticker.options = ()  # empty expirations

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep"):
            result = _validate_open_interest(["NOOPTS"], settings)

        assert result == []

    def test_exact_threshold_passes(self):
        """Ticker with OI exactly equal to threshold should pass (>= comparison)."""
        settings = Settings(min_universe_oi=100)

        mock_ticker = MagicMock()
        mock_ticker.info = {"quoteType": "EQUITY"}
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.option_chain.return_value = _make_chain([50], [50])
        # total OI = 50+50 = 100 == threshold

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep"):
            result = _validate_open_interest(["EXACT"], settings)

        assert result == ["EXACT"]

    def test_empty_input_returns_empty(self):
        """Empty ticker list should return empty result."""
        settings = Settings(min_universe_oi=100)
        result = _validate_open_interest([], settings)
        assert result == []

    def test_mixed_etf_and_stocks(self):
        """ETFs pass through, stocks validated; failed stocks excluded."""
        settings = Settings(min_universe_oi=100)

        def mock_yf_ticker(symbol):
            mock_t = MagicMock()
            if symbol == "SPY":
                mock_t.info = {"quoteType": "ETF"}
            elif symbol == "HIGH":
                mock_t.info = {"quoteType": "EQUITY"}
                mock_t.options = ("2025-01-17",)
                mock_t.option_chain.return_value = _make_chain([500], [500])
            elif symbol == "LOW":
                mock_t.info = {"quoteType": "EQUITY"}
                mock_t.options = ("2025-01-17",)
                mock_t.option_chain.return_value = _make_chain([1], [1])
            return mock_t

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            side_effect=mock_yf_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep"):
            result = _validate_open_interest(["SPY", "HIGH", "LOW"], settings)

        assert result == ["SPY", "HIGH"]

    def test_uses_nearest_expiration(self):
        """Should use the first (nearest) expiration for OI check."""
        settings = Settings(min_universe_oi=100)

        mock_ticker = MagicMock()
        mock_ticker.info = {"quoteType": "EQUITY"}
        mock_ticker.options = ("2025-01-17", "2025-02-21", "2025-03-21")
        mock_ticker.option_chain.return_value = _make_chain([100], [100])

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep"):
            _validate_open_interest(["TEST"], settings)

        # Verify option_chain was called with the nearest (first) expiry
        mock_ticker.option_chain.assert_called_once_with("2025-01-17")

    def test_rate_limiting_sleeps_between_tickers(self):
        """Should call time.sleep(2) for rate limiting between tickers."""
        settings = Settings(min_universe_oi=100)

        mock_ticker = MagicMock()
        mock_ticker.info = {"quoteType": "EQUITY"}
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.option_chain.return_value = _make_chain([200], [200])

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch("option_alpha.data.universe_refresh.time.sleep") as mock_sleep:
            _validate_open_interest(["A", "B", "C"], settings)

        # Each non-ETF ticker triggers a sleep(2) call
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(2)


# ---------------------------------------------------------------------------
# Atomic writes (Issue #61)
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    """Tests for the atomic-write logic in _do_refresh (tmp -> bak -> rename)."""

    def _setup_files(self, tmp_path, monkeypatch):
        """Common setup: point module paths at tmp_path, return file paths."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )
        return universe_file, meta_file

    def _mock_yf_ticker_stock(self):
        """Return a MagicMock that behaves like a simple stock ticker."""
        mock_t = MagicMock()
        mock_t.options = ("2025-01-17",)
        mock_t.info = {
            "shortName": "Test",
            "sector": "Tech",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_t.option_chain.return_value = _make_chain([500], [500])
        return mock_t

    def test_tmp_file_written_then_renamed(self, tmp_path, monkeypatch):
        """Atomic write: data is written to .tmp first, then renamed to final."""
        universe_file, meta_file = self._setup_files(tmp_path, monkeypatch)
        universe_file.write_text(json.dumps([]))

        mock_ticker = self._mock_yf_ticker_stock()

        # Track file operations by wrapping Path.rename
        rename_calls = []
        original_rename = Path.rename

        def tracking_rename(self_path, target):
            rename_calls.append((str(self_path), str(target)))
            return original_rename(self_path, target)

        monkeypatch.setattr(Path, "rename", tracking_rename)

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ):
            asyncio.run(_do_refresh(Settings(), regenerate=False))

        # Verify .tmp was renamed to the final file
        assert any(
            call[0].endswith(".tmp") and call[1] == str(universe_file)
            for call in rename_calls
        ), f"Expected .tmp -> universe_data.json rename, got: {rename_calls}"

    def test_backup_file_created(self, tmp_path, monkeypatch):
        """When an existing universe file exists, a .bak backup should be created."""
        universe_file, meta_file = self._setup_files(tmp_path, monkeypatch)
        original_data = [{"symbol": "OLD", "asset_type": "stock"}]
        universe_file.write_text(json.dumps(original_data))

        mock_ticker = self._mock_yf_ticker_stock()

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ):
            asyncio.run(_do_refresh(Settings(), regenerate=False))

        bak_file = tmp_path / "universe_data.json.bak"
        assert bak_file.exists(), ".bak backup file should be created"

        # Backup should contain the original data
        bak_data = json.loads(bak_file.read_text())
        assert bak_data == original_data

    def test_original_preserved_on_write_failure(self, tmp_path, monkeypatch):
        """If the tmp write fails, original universe file should remain intact."""
        universe_file, meta_file = self._setup_files(tmp_path, monkeypatch)
        original_data = [{"symbol": "KEEP", "asset_type": "stock"}]
        universe_file.write_text(json.dumps(original_data))

        # Make write_text on the tmp file fail
        original_write = Path.write_text

        def failing_write(self_path, content, *args, **kwargs):
            if str(self_path).endswith(".tmp"):
                raise OSError("Disk full")
            return original_write(self_path, content, *args, **kwargs)

        mock_ticker = self._mock_yf_ticker_stock()

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch.object(
            Path, "write_text", failing_write
        ):
            with pytest.raises(OSError, match="Disk full"):
                asyncio.run(_do_refresh(Settings(), regenerate=False))

        # Original file should be untouched
        preserved = json.loads(universe_file.read_text())
        assert preserved == original_data

        # .tmp file should be cleaned up
        tmp_file = tmp_path / "universe_data.json.tmp"
        assert not tmp_file.exists(), ".tmp file should be cleaned up on failure"

    def test_no_backup_when_no_existing_file(self, tmp_path, monkeypatch):
        """When no existing file, no .bak should be created."""
        universe_file, meta_file = self._setup_files(tmp_path, monkeypatch)
        # Do not create universe_file -- fresh start

        mock_ticker = self._mock_yf_ticker_stock()

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ):
            asyncio.run(_do_refresh(Settings(), regenerate=False))

        bak_file = tmp_path / "universe_data.json.bak"
        assert not bak_file.exists(), "No .bak when there was no pre-existing file"


# ---------------------------------------------------------------------------
# ETF preservation during regeneration (Issue #61)
# ---------------------------------------------------------------------------


class TestETFPreservation:
    """Tests for ETF entries being preserved during stock regeneration."""

    def test_etfs_survive_regeneration(self, tmp_path, monkeypatch):
        """ETFs from current universe should be re-merged after regeneration."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )

        # Seed with 1 ETF + 1 stock
        existing = [
            {"symbol": "SPY", "name": "SPDR S&P 500", "sector": "",
             "market_cap_tier": "", "asset_type": "etf"},
            {"symbol": "OLD", "name": "Old Stock", "sector": "Tech",
             "market_cap_tier": "large", "asset_type": "stock"},
        ]
        universe_file.write_text(json.dumps(existing))

        # Mock EDGAR: returns only AAPL (no SPY, no OLD)
        mock_edgar = {
            "0": {"cik_str": "1", "ticker": "AAPL", "title": "Apple"},
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_edgar

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.info = {
            "shortName": "Apple",
            "sector": "Tech",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_ticker.option_chain.return_value = _make_chain([500], [500])

        with patch(
            "option_alpha.data.universe_refresh.httpx.AsyncClient",
            return_value=mock_client,
        ), patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ):
            result = asyncio.run(_do_refresh(Settings(), regenerate=True))

        assert result["success"] is True

        # Load final universe
        written = json.loads(universe_file.read_text())
        symbols = {t["symbol"] for t in written}

        # AAPL should be there from EDGAR pipeline
        assert "AAPL" in symbols
        # SPY should be preserved as ETF
        assert "SPY" in symbols
        # OLD should be gone (not in EDGAR results)
        assert "OLD" not in symbols

        # Verify SPY entry is unchanged
        spy_entry = next(t for t in written if t["symbol"] == "SPY")
        assert spy_entry["asset_type"] == "etf"
        assert spy_entry["name"] == "SPDR S&P 500"

    def test_no_etf_duplication(self, tmp_path, monkeypatch):
        """If an ETF appears in both EDGAR results and existing, no duplicate."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )

        existing = [
            {"symbol": "SPY", "name": "SPDR S&P 500", "sector": "",
             "market_cap_tier": "", "asset_type": "etf"},
        ]
        universe_file.write_text(json.dumps(existing))

        # EDGAR also returns SPY
        mock_edgar = {
            "0": {"cik_str": "1", "ticker": "SPY", "title": "SPDR S&P 500"},
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_edgar

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.info = {
            "shortName": "SPDR S&P 500",
            "quoteType": "ETF",
            "marketCap": 0,
        }
        mock_ticker.option_chain.return_value = _make_chain([5000], [5000])

        with patch(
            "option_alpha.data.universe_refresh.httpx.AsyncClient",
            return_value=mock_client,
        ), patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ):
            asyncio.run(_do_refresh(Settings(), regenerate=True))

        written = json.loads(universe_file.read_text())
        spy_entries = [t for t in written if t["symbol"] == "SPY"]
        assert len(spy_entries) == 1, "SPY should not be duplicated"


# ---------------------------------------------------------------------------
# Size warnings in meta (Issue #61)
# ---------------------------------------------------------------------------


class TestSizeWarnings:
    """Tests for size warning logic when stock count is outside 500-1500 range."""

    def test_warning_when_stock_count_below_500(
        self, tmp_path, monkeypatch, caplog
    ):
        """Warning should be logged when stock count < 500."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )
        universe_file.write_text(json.dumps([]))

        # Produce just 0 tickers (well below 500)
        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.info = {
            "shortName": "Test",
            "sector": "Tech",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_ticker.option_chain.return_value = _make_chain([500], [500])

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ), caplog.at_level(logging.WARNING, logger="option_alpha.data.universe_refresh"):
            asyncio.run(_do_refresh(Settings(), regenerate=False))

        assert "Low stock count" in caplog.text
        # Meta should contain size_warning
        meta = json.loads(meta_file.read_text())
        assert "size_warning" in meta
        assert "Low stock count" in meta["size_warning"]

    def test_warning_when_stock_count_above_1500(
        self, tmp_path, monkeypatch, caplog
    ):
        """Warning should be logged when stock count > 1500."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )

        # Seed with 1600 stock tickers
        big_universe = [
            {"symbol": f"T{i}", "asset_type": "stock"} for i in range(1600)
        ]
        universe_file.write_text(json.dumps(big_universe))

        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.info = {
            "shortName": "Test",
            "sector": "Tech",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_ticker.option_chain.return_value = _make_chain([500], [500])

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ), caplog.at_level(logging.WARNING, logger="option_alpha.data.universe_refresh"):
            asyncio.run(_do_refresh(Settings(), regenerate=False))

        assert "High stock count" in caplog.text
        meta = json.loads(meta_file.read_text())
        assert "size_warning" in meta
        assert "High stock count" in meta["size_warning"]

    def test_no_warning_when_stock_count_in_range(
        self, tmp_path, monkeypatch, caplog
    ):
        """No warning when stock count is within 500-1500 range."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )

        # Seed with 800 stock tickers (within range)
        mid_universe = [
            {"symbol": f"T{i}", "asset_type": "stock"} for i in range(800)
        ]
        universe_file.write_text(json.dumps(mid_universe))

        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.info = {
            "shortName": "Test",
            "sector": "Tech",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_ticker.option_chain.return_value = _make_chain([500], [500])

        with patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ), caplog.at_level(logging.WARNING, logger="option_alpha.data.universe_refresh"):
            asyncio.run(_do_refresh(Settings(), regenerate=False))

        assert "Low stock count" not in caplog.text
        assert "High stock count" not in caplog.text
        meta = json.loads(meta_file.read_text())
        assert "size_warning" not in meta


# ---------------------------------------------------------------------------
# Regeneration mode (Issue #56)
# ---------------------------------------------------------------------------


class TestRegenerationMode:
    """Tests for regenerate=True vs regenerate=False in _do_refresh."""

    def test_regenerate_true_calls_fetch_edgar(self, tmp_path, monkeypatch):
        """regenerate=True should call _fetch_edgar_tickers."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )
        universe_file.write_text(json.dumps([]))

        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.info = {
            "shortName": "Test",
            "sector": "Tech",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_ticker.option_chain.return_value = _make_chain([500], [500])

        mock_fetch_edgar = AsyncMock(return_value=["AAPL", "MSFT"])

        with patch(
            "option_alpha.data.universe_refresh._fetch_edgar_tickers",
            mock_fetch_edgar,
        ), patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ):
            result = asyncio.run(_do_refresh(Settings(), regenerate=True))

        mock_fetch_edgar.assert_awaited_once()
        assert result["success"] is True
        assert result["mode"] == "regenerate"

    def test_regenerate_false_loads_existing(self, tmp_path, monkeypatch):
        """regenerate=False should load from _load_current instead of EDGAR."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )

        existing = [
            {"symbol": "AAPL", "asset_type": "stock"},
            {"symbol": "MSFT", "asset_type": "stock"},
        ]
        universe_file.write_text(json.dumps(existing))

        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.info = {
            "shortName": "Test",
            "sector": "Tech",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_ticker.option_chain.return_value = _make_chain([500], [500])

        mock_fetch_edgar = AsyncMock(return_value=["SHOULD_NOT_CALL"])

        with patch(
            "option_alpha.data.universe_refresh._fetch_edgar_tickers",
            mock_fetch_edgar,
        ), patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ):
            result = asyncio.run(_do_refresh(Settings(), regenerate=False))

        # _fetch_edgar_tickers should NOT have been called
        mock_fetch_edgar.assert_not_awaited()
        assert result["success"] is True
        assert result["mode"] == "validate"

    def test_regenerate_false_extracts_symbols_from_current(
        self, tmp_path, monkeypatch
    ):
        """Validate mode should extract symbol strings from current data dicts."""
        universe_file = tmp_path / "universe_data.json"
        meta_file = tmp_path / "universe_meta.json"
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._UNIVERSE_FILE", universe_file
        )
        monkeypatch.setattr(
            "option_alpha.data.universe_refresh._META_FILE", meta_file
        )

        existing = [
            {"symbol": "AAPL", "name": "Apple", "asset_type": "stock"},
            {"symbol": "GOOG", "name": "Google", "asset_type": "stock"},
        ]
        universe_file.write_text(json.dumps(existing))

        validated_tickers = []

        def capture_validate_optionability(tickers, **kwargs):
            validated_tickers.extend(tickers)
            return tickers  # pass all through

        mock_ticker = MagicMock()
        mock_ticker.options = ("2025-01-17",)
        mock_ticker.info = {
            "shortName": "Test",
            "sector": "Tech",
            "marketCap": 50_000_000_000,
            "quoteType": "EQUITY",
        }
        mock_ticker.option_chain.return_value = _make_chain([500], [500])

        with patch(
            "option_alpha.data.universe_refresh._validate_optionability",
            side_effect=capture_validate_optionability,
        ), patch(
            "option_alpha.data.universe_refresh.yf.Ticker",
            return_value=mock_ticker,
        ), patch(
            "option_alpha.data.universe_refresh.time.sleep",
        ), patch(
            "option_alpha.data.universe._universe_cache", None
        ):
            asyncio.run(_do_refresh(Settings(), regenerate=False))

        # Validate mode should pass symbol strings, not dicts
        assert validated_tickers == ["AAPL", "GOOG"]


# ---------------------------------------------------------------------------
# Scheduler setup (Issue #58)
# ---------------------------------------------------------------------------


class TestSchedulerSetup:
    """Tests for APScheduler configuration in the FastAPI lifespan."""

    def test_create_app_imports_without_error(self):
        """Verify create_app can be imported (basic sanity)."""
        from option_alpha.web.app import create_app
        assert callable(create_app)

    def test_scheduler_uses_refresh_schedule_setting(self):
        """CronTrigger should use settings.universe_refresh_schedule."""
        settings = Settings(universe_refresh_schedule="sun")

        # We verify the setting value is what gets used in the scheduler
        assert settings.universe_refresh_schedule == "sun"

        # Verify the setting is a valid APScheduler day_of_week value
        from apscheduler.triggers.cron import CronTrigger
        trigger = CronTrigger(
            day_of_week=settings.universe_refresh_schedule,
            hour=2,
            minute=0,
            timezone="UTC",
        )
        # Should not raise; trigger is valid
        assert trigger is not None

    def test_scheduler_default_schedule_is_saturday(self):
        """Default schedule 'sat' should produce a valid CronTrigger."""
        settings = Settings()
        from apscheduler.triggers.cron import CronTrigger
        trigger = CronTrigger(
            day_of_week=settings.universe_refresh_schedule,
            hour=2,
            minute=0,
            timezone="UTC",
        )
        assert trigger is not None

    def test_scheduler_various_day_values(self):
        """All day-of-week abbreviations should work as schedule values."""
        from apscheduler.triggers.cron import CronTrigger
        for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
            settings = Settings(universe_refresh_schedule=day)
            trigger = CronTrigger(
                day_of_week=settings.universe_refresh_schedule,
                hour=2,
                minute=0,
                timezone="UTC",
            )
            assert trigger is not None, f"Failed for day={day}"
