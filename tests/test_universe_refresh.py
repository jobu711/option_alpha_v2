"""Tests for dynamic universe refresh module."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from option_alpha.config import Settings
from option_alpha.data.universe_refresh import (
    _classify_market_cap,
    _enrich_metadata,
    _fetch_edgar_tickers,
    _load_current,
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
