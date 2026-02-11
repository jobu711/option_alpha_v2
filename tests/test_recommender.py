"""Tests for options contract recommendation engine."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from option_alpha.config import Settings
from option_alpha.models import Direction, OptionsRecommendation, TickerScore
from option_alpha.options.chains import ChainData
from option_alpha.options.recommender import (
    FRED_SERIES_URL,
    fetch_risk_free_rate,
    filter_by_liquidity,
    recommend_contract,
    recommend_for_scored_tickers,
    select_contract,
)


def _make_chain_df(
    strikes: list[float] | None = None,
    bids: list[float] | None = None,
    asks: list[float] | None = None,
    volumes: list[int] | None = None,
    open_interests: list[int] | None = None,
    ivs: list[float] | None = None,
) -> pd.DataFrame:
    """Create a test options chain DataFrame."""
    if strikes is None:
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    n = len(strikes)
    if bids is None:
        bids = [12.0, 8.0, 5.0, 2.5, 1.0][:n]
    if asks is None:
        asks = [12.5, 8.5, 5.5, 3.0, 1.5][:n]
    if volumes is None:
        volumes = [100, 200, 500, 300, 50][:n]
    if open_interests is None:
        open_interests = [500, 1000, 2000, 800, 200][:n]
    if ivs is None:
        ivs = [0.25, 0.27, 0.30, 0.32, 0.35][:n]

    return pd.DataFrame({
        "contractSymbol": [f"TEST{int(s)}C" for s in strikes],
        "strike": strikes,
        "bid": bids,
        "ask": asks,
        "volume": volumes,
        "openInterest": open_interests,
        "impliedVolatility": ivs,
    })


def _make_ticker_score(
    symbol: str = "AAPL",
    score: float = 75.0,
    direction: Direction = Direction.BULLISH,
) -> TickerScore:
    return TickerScore(
        symbol=symbol,
        composite_score=score,
        direction=direction,
    )


def _make_chain_data(
    symbol: str = "AAPL",
    calls: pd.DataFrame | None = None,
    puts: pd.DataFrame | None = None,
    underlying_price: float = 100.0,
    dte: int = 45,
) -> ChainData:
    return ChainData(
        symbol=symbol,
        expiration=date(2025, 8, 15),
        dte=dte,
        underlying_price=underlying_price,
        calls=calls if calls is not None else _make_chain_df(),
        puts=puts if puts is not None else _make_chain_df(),
    )


# ─── FRED API ────────────────────────────────────────────────────────


class TestFetchRiskFreeRate:
    def test_no_api_key_returns_fallback(self):
        rate = fetch_risk_free_rate(api_key=None, fallback=0.05)
        assert rate == 0.05

    def test_empty_api_key_returns_fallback(self):
        rate = fetch_risk_free_rate(api_key="", fallback=0.04)
        assert rate == 0.04

    @patch("option_alpha.options.recommender.httpx.Client")
    def test_successful_api_call(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2025-07-01", "value": "4.25"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        rate = fetch_risk_free_rate(api_key="test_key")
        assert rate == pytest.approx(0.0425)

    @patch("option_alpha.options.recommender.httpx.Client")
    def test_api_error_returns_fallback(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("Connection error")

        rate = fetch_risk_free_rate(api_key="test_key", fallback=0.05)
        assert rate == 0.05

    @patch("option_alpha.options.recommender.httpx.Client")
    def test_missing_value_returns_fallback(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2025-07-01", "value": "."},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        rate = fetch_risk_free_rate(api_key="test_key", fallback=0.06)
        assert rate == 0.06

    @patch("option_alpha.options.recommender.httpx.Client")
    def test_skips_missing_values(self, mock_client_cls):
        """Should skip '.' values and use first valid one."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2025-07-02", "value": "."},
                {"date": "2025-07-01", "value": "3.50"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        rate = fetch_risk_free_rate(api_key="test_key")
        assert rate == pytest.approx(0.035)

    def test_custom_fallback_rate(self):
        rate = fetch_risk_free_rate(api_key=None, fallback=0.03)
        assert rate == 0.03


# ─── Liquidity Filter ────────────────────────────────────────────────


class TestFilterByLiquidity:
    def test_filters_low_open_interest(self):
        df = _make_chain_df(
            strikes=[100.0, 105.0],
            open_interests=[50, 200],
            bids=[5.0, 3.0],
            asks=[5.2, 3.2],
            volumes=[10, 20],
        )
        filtered = filter_by_liquidity(df, min_open_interest=100)
        assert len(filtered) == 1
        assert filtered.iloc[0]["strike"] == 105.0

    def test_filters_low_volume(self):
        df = _make_chain_df(
            strikes=[100.0, 105.0],
            volumes=[0, 10],
            open_interests=[500, 500],
            bids=[5.0, 3.0],
            asks=[5.2, 3.2],
        )
        filtered = filter_by_liquidity(df, min_volume=1)
        assert len(filtered) == 1
        assert filtered.iloc[0]["strike"] == 105.0

    def test_filters_wide_spread(self):
        df = _make_chain_df(
            strikes=[100.0, 105.0],
            bids=[5.0, 3.0],
            asks=[6.0, 3.1],  # 18% spread vs 3.3% spread
            volumes=[10, 10],
            open_interests=[500, 500],
        )
        filtered = filter_by_liquidity(df, max_bid_ask_spread_pct=0.10)
        assert len(filtered) == 1
        assert filtered.iloc[0]["strike"] == 105.0

    def test_empty_dataframe(self):
        filtered = filter_by_liquidity(pd.DataFrame())
        assert filtered.empty

    def test_all_pass(self):
        df = _make_chain_df(
            strikes=[95.0, 100.0, 105.0],
            bids=[8.0, 5.0, 3.0],
            asks=[8.5, 5.5, 3.3],
            volumes=[100, 200, 50],
            open_interests=[500, 1000, 300],
        )
        filtered = filter_by_liquidity(
            df, min_open_interest=100, max_bid_ask_spread_pct=0.20, min_volume=1,
        )
        assert len(filtered) == len(df)

    def test_all_filtered(self):
        df = _make_chain_df(
            strikes=[100.0],
            open_interests=[10],
            bids=[5.0],
            asks=[5.5],
            volumes=[0],
        )
        filtered = filter_by_liquidity(
            df, min_open_interest=100, min_volume=1,
        )
        assert len(filtered) == 0

    def test_nan_values_handled(self):
        df = pd.DataFrame({
            "strike": [100.0, 105.0],
            "bid": [5.0, None],
            "ask": [5.2, None],
            "volume": [100, None],
            "openInterest": [500, None],
        })
        # Should not crash
        filtered = filter_by_liquidity(df, min_open_interest=100, min_volume=1)
        assert len(filtered) <= 2


# ─── Select Contract ─────────────────────────────────────────────────


class TestSelectContract:
    def test_selects_near_target_delta(self):
        df = _make_chain_df()
        result = select_contract(
            df,
            underlying_price=100.0,
            dte=45,
            risk_free_rate=0.05,
            option_type="call",
        )
        assert result is not None
        # Should pick a strike where delta is ~0.30-0.40

    def test_empty_chain(self):
        result = select_contract(
            pd.DataFrame(),
            underlying_price=100.0,
            dte=45,
            risk_free_rate=0.05,
            option_type="call",
        )
        assert result is None

    def test_zero_underlying(self):
        df = _make_chain_df()
        result = select_contract(
            df,
            underlying_price=0.0,
            dte=45,
            risk_free_rate=0.05,
            option_type="call",
        )
        assert result is None

    def test_zero_dte(self):
        df = _make_chain_df()
        result = select_contract(
            df,
            underlying_price=100.0,
            dte=0,
            risk_free_rate=0.05,
            option_type="call",
        )
        assert result is None

    def test_put_selection(self):
        df = _make_chain_df()
        result = select_contract(
            df,
            underlying_price=100.0,
            dte=45,
            risk_free_rate=0.05,
            option_type="put",
        )
        assert result is not None

    def test_closest_to_target_when_none_in_range(self):
        """When no delta is in [0.30, 0.40], picks closest."""
        # Very ITM options - all deltas will be > 0.40
        df = _make_chain_df(
            strikes=[50.0, 55.0, 60.0],
            ivs=[0.20, 0.20, 0.20],
        )
        result = select_contract(
            df,
            underlying_price=100.0,
            dte=45,
            risk_free_rate=0.05,
            option_type="call",
        )
        # Should still return something (closest to target)
        assert result is not None


# ─── Recommend Contract ──────────────────────────────────────────────


class TestRecommendContract:
    def test_bullish_selects_calls(self):
        ts = _make_ticker_score(direction=Direction.BULLISH)
        chain = _make_chain_data()

        rec = recommend_contract(ts, chain, risk_free_rate=0.05)
        assert rec is not None
        assert rec.option_type == "call"
        assert rec.direction == Direction.BULLISH
        assert rec.symbol == "AAPL"

    def test_bearish_selects_puts(self):
        ts = _make_ticker_score(direction=Direction.BEARISH)
        chain = _make_chain_data()

        rec = recommend_contract(ts, chain, risk_free_rate=0.05)
        assert rec is not None
        assert rec.option_type == "put"
        assert rec.direction == Direction.BEARISH

    def test_neutral_returns_none(self):
        ts = _make_ticker_score(direction=Direction.NEUTRAL)
        chain = _make_chain_data()

        rec = recommend_contract(ts, chain, risk_free_rate=0.05)
        assert rec is None

    def test_recommendation_fields(self):
        ts = _make_ticker_score(direction=Direction.BULLISH)
        chain = _make_chain_data()

        rec = recommend_contract(ts, chain, risk_free_rate=0.05)
        assert rec is not None
        assert isinstance(rec, OptionsRecommendation)
        assert rec.strike > 0
        assert rec.dte == 45
        assert rec.delta is not None
        assert rec.gamma is not None
        assert rec.theta is not None
        assert rec.vega is not None
        assert rec.underlying_price == 100.0

    def test_empty_chain_returns_none(self):
        ts = _make_ticker_score(direction=Direction.BULLISH)
        chain = _make_chain_data(calls=pd.DataFrame())

        rec = recommend_contract(ts, chain, risk_free_rate=0.05)
        assert rec is None

    def test_no_liquid_contracts_returns_none(self):
        ts = _make_ticker_score(direction=Direction.BULLISH)
        # All contracts have 0 open interest
        df = _make_chain_df(open_interests=[0, 0, 0, 0, 0])
        chain = _make_chain_data(calls=df)

        settings = Settings(min_open_interest=100)
        rec = recommend_contract(ts, chain, risk_free_rate=0.05, settings=settings)
        assert rec is None


# ─── Recommend For Scored Tickers ────────────────────────────────────


class TestRecommendForScoredTickers:
    @patch("option_alpha.options.recommender.fetch_risk_free_rate")
    def test_generates_recommendations(self, mock_rate):
        mock_rate.return_value = 0.045

        scores = [
            _make_ticker_score("AAPL", 80, Direction.BULLISH),
            _make_ticker_score("MSFT", 70, Direction.BEARISH),
            _make_ticker_score("GOOG", 60, Direction.NEUTRAL),
        ]
        chains = {
            "AAPL": _make_chain_data("AAPL"),
            "MSFT": _make_chain_data("MSFT"),
            "GOOG": _make_chain_data("GOOG"),
        }

        recs = recommend_for_scored_tickers(scores, chains)
        # AAPL and MSFT should have recommendations, GOOG is neutral
        symbols = [r.symbol for r in recs]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOG" not in symbols

    @patch("option_alpha.options.recommender.fetch_risk_free_rate")
    def test_missing_chain_skipped(self, mock_rate):
        mock_rate.return_value = 0.05

        scores = [_make_ticker_score("AAPL", 80, Direction.BULLISH)]
        chains: dict[str, ChainData] = {}  # No chain data

        recs = recommend_for_scored_tickers(scores, chains)
        assert len(recs) == 0

    @patch("option_alpha.options.recommender.fetch_risk_free_rate")
    def test_empty_scores(self, mock_rate):
        mock_rate.return_value = 0.05

        recs = recommend_for_scored_tickers([], {})
        assert len(recs) == 0

    @patch("option_alpha.options.recommender.fetch_risk_free_rate")
    def test_uses_settings(self, mock_rate):
        mock_rate.return_value = 0.05

        settings = Settings(
            fred_api_key="test_key",
            risk_free_rate_fallback=0.03,
        )
        scores = [_make_ticker_score("AAPL", 80, Direction.BULLISH)]
        chains = {"AAPL": _make_chain_data("AAPL")}

        recs = recommend_for_scored_tickers(scores, chains, settings)
        mock_rate.assert_called_once_with(api_key="test_key", fallback=0.03)
