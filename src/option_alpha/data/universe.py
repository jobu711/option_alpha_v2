"""Ticker universe management with pre-filtering.

Loads a curated universe of optionable tickers from a JSON data file
and provides filtering by minimum price and average volume.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import yfinance as yf

from option_alpha.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Path to the shipped universe data file
_UNIVERSE_DATA_PATH = Path(__file__).parent / "universe_data.json"

# Module-level cache for loaded universe data
_universe_cache: list[dict] | None = None


def load_universe_data() -> list[dict]:
    """Load universe metadata from the JSON data file.

    Returns a list of dicts, each with keys:
        symbol, name, sector, market_cap_tier, asset_type

    Results are cached after the first call. Use _clear_cache() to reset.
    """
    global _universe_cache
    if _universe_cache is not None:
        return _universe_cache

    with open(_UNIVERSE_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    _universe_cache = data
    logger.debug(f"Loaded {len(data)} tickers from {_UNIVERSE_DATA_PATH}")
    return data


def _clear_cache() -> None:
    """Clear the cached universe data. Useful for testing."""
    global _universe_cache
    _universe_cache = None


def get_full_universe() -> list[str]:
    """Return the complete curated ticker universe (deduplicated, sorted)."""
    data = load_universe_data()
    return sorted(set(t["symbol"] for t in data))


# ---------------------------------------------------------------------------
# Preset / sector filtering
# ---------------------------------------------------------------------------

PRESET_FILTERS = {
    "sp500": lambda t: t["market_cap_tier"] == "large" and t["asset_type"] == "stock",
    "midcap": lambda t: t["market_cap_tier"] == "mid" and t["asset_type"] == "stock",
    "smallcap": lambda t: t["market_cap_tier"] in ("small", "micro") and t["asset_type"] == "stock",
    "etfs": lambda t: t["asset_type"] == "etf",
    "full": lambda t: True,
}

GICS_SECTORS = [
    "Technology",
    "Healthcare",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Industrials",
    "Energy",
    "Materials",
    "Utilities",
    "Real Estate",
    "Communication Services",
]


def get_scan_universe(
    presets: list[str] | None = None,
    sectors: list[str] | None = None,
    extra_tickers: list[str] | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Get filtered universe based on presets, sectors, and extra tickers.

    Presets are union-combined (sp500 + etfs = both sets).
    Sectors are intersection-filtered (sp500 AND Technology = large-cap tech only).
    Extra tickers (from watchlists) are always included regardless of filters.
    """
    if settings is None:
        settings = get_settings()

    if presets is None:
        presets = settings.universe_presets
    if sectors is None:
        sectors = settings.universe_sectors

    data = load_universe_data()

    # Apply preset filters (union)
    if not presets or "full" in presets:
        filtered = data
    else:
        filtered = []
        seen: set[str] = set()
        for preset in presets:
            filt = PRESET_FILTERS.get(preset)
            if filt:
                for t in data:
                    if t["symbol"] not in seen and filt(t):
                        filtered.append(t)
                        seen.add(t["symbol"])

    # Apply sector filter (intersection)
    if sectors:
        sector_set = set(sectors)
        filtered = [t for t in filtered if t.get("sector", "") in sector_set]

    # Extract symbols
    symbols = {t["symbol"] for t in filtered}

    # Merge extra tickers
    if extra_tickers:
        symbols.update(extra_tickers)

    return sorted(symbols)


def filter_universe(
    tickers: list[str],
    settings: Optional[Settings] = None,
    price_data: Optional[dict[str, dict]] = None,
) -> list[str]:
    """Filter tickers by minimum price and average volume.

    Args:
        tickers: List of ticker symbols to filter.
        settings: Configuration settings (uses defaults if None).
        price_data: Optional pre-fetched price data dict.
            Keys are symbols, values are dicts with 'last_price' and 'avg_volume'.
            If None, data is fetched via yfinance.

    Returns:
        List of tickers that pass all filters.
    """
    if settings is None:
        settings = get_settings()

    if price_data is not None:
        return _filter_with_data(tickers, price_data, settings)

    return _filter_via_yfinance(tickers, settings)


def _filter_with_data(
    tickers: list[str],
    price_data: dict[str, dict],
    settings: Settings,
) -> list[str]:
    """Filter tickers using pre-fetched price data."""
    passed = []
    for symbol in tickers:
        data = price_data.get(symbol)
        if data is None:
            continue
        price = data.get("last_price", 0)
        volume = data.get("avg_volume", 0)
        if price >= settings.min_price and volume >= settings.min_avg_volume:
            passed.append(symbol)
    return sorted(passed)


def _filter_via_yfinance(
    tickers: list[str],
    settings: Settings,
) -> list[str]:
    """Filter tickers by fetching quick info from yfinance.

    Downloads recent price data in batch to check price/volume thresholds.
    """
    if not tickers:
        return []

    passed = []
    batch_size = 100

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        symbols_str = " ".join(batch)

        try:
            df = yf.download(
                symbols_str,
                period="5d",
                progress=False,
                threads=True,
            )

            if df.empty:
                continue

            for symbol in batch:
                try:
                    if len(batch) == 1:
                        close_series = df["Close"]
                        vol_series = df["Volume"]
                    else:
                        if symbol not in df["Close"].columns:
                            continue
                        close_series = df["Close"][symbol].dropna()
                        vol_series = df["Volume"][symbol].dropna()

                    if close_series.empty or vol_series.empty:
                        continue

                    last_price = float(close_series.iloc[-1])
                    avg_volume = float(vol_series.mean())

                    if (
                        last_price >= settings.min_price
                        and avg_volume >= settings.min_avg_volume
                    ):
                        passed.append(symbol)

                except (KeyError, IndexError) as e:
                    logger.debug(f"Skipping {symbol}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to download batch starting at index {i}: {e}")
            continue

    return sorted(passed)


def get_filtered_universe(settings: Optional[Settings] = None) -> list[str]:
    """Get the full universe, filtered by price and volume.

    This is the main entry point for getting the list of tickers to scan.
    """
    full = get_full_universe()
    logger.info(f"Full universe: {len(full)} tickers")
    filtered = filter_universe(full, settings=settings)
    logger.info(f"After filtering: {len(filtered)} tickers")
    return filtered
