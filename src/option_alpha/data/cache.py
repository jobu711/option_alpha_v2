"""Parquet/JSON caching layer with freshness checks.

Provides transparent caching for OHLCV data (Parquet) and
JSON-serializable data with date-based file naming.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from option_alpha.config import Settings, get_settings
from option_alpha.models import TickerData

logger = logging.getLogger(__name__)

# Default cache staleness threshold
DEFAULT_MAX_AGE_HOURS = 18  # Data fetched before 6pm is stale by next morning


def _get_cache_dir(settings: Optional[Settings] = None) -> Path:
    """Get the cache directory, creating it if necessary."""
    if settings is None:
        settings = get_settings()
    cache_dir = settings.data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_file_path(
    symbol: str,
    date: Optional[datetime] = None,
    ext: str = ".parquet",
    settings: Optional[Settings] = None,
) -> Path:
    """Build a cache file path with date-based naming.

    Format: {cache_dir}/{symbol}_{YYYY-MM-DD}{ext}
    """
    if date is None:
        date = datetime.utcnow()
    date_str = date.strftime("%Y-%m-%d")
    cache_dir = _get_cache_dir(settings)
    return cache_dir / f"{symbol}_{date_str}{ext}"


def is_fresh(
    filepath: Path,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> bool:
    """Check if a cache file is still fresh.

    Args:
        filepath: Path to the cache file.
        max_age_hours: Maximum age in hours before considering stale.

    Returns:
        True if file exists and is within max_age_hours of now.
    """
    if not filepath.exists():
        return False

    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
    age = datetime.now() - mtime
    fresh = age < timedelta(hours=max_age_hours)

    if not fresh:
        logger.debug(f"Cache stale: {filepath.name} (age: {age})")

    return fresh


def save_ticker_parquet(
    ticker_data: TickerData,
    settings: Optional[Settings] = None,
) -> Path:
    """Save TickerData OHLCV to a Parquet file.

    Args:
        ticker_data: TickerData with OHLCV arrays.
        settings: Configuration settings.

    Returns:
        Path to the saved Parquet file.
    """
    filepath = _cache_file_path(ticker_data.symbol, ext=".parquet", settings=settings)

    df = pd.DataFrame(
        {
            "open": ticker_data.open,
            "high": ticker_data.high,
            "low": ticker_data.low,
            "close": ticker_data.close,
            "volume": ticker_data.volume,
        },
        index=pd.DatetimeIndex(ticker_data.dates, name="date"),
    )

    df.to_parquet(filepath, engine="pyarrow")
    logger.debug(f"Saved {ticker_data.symbol} OHLCV to {filepath}")
    return filepath


def load_ticker_parquet(
    symbol: str,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    settings: Optional[Settings] = None,
) -> Optional[TickerData]:
    """Load TickerData from a cached Parquet file.

    Args:
        symbol: Ticker symbol.
        max_age_hours: Maximum cache age in hours.
        settings: Configuration settings.

    Returns:
        TickerData if fresh cache exists, None otherwise.
    """
    filepath = _cache_file_path(symbol, ext=".parquet", settings=settings)

    if not is_fresh(filepath, max_age_hours):
        return None

    try:
        df = pd.read_parquet(filepath, engine="pyarrow")
        return TickerData(
            symbol=symbol,
            dates=df.index.tolist(),
            open=df["open"].tolist(),
            high=df["high"].tolist(),
            low=df["low"].tolist(),
            close=df["close"].tolist(),
            volume=[int(v) for v in df["volume"].tolist()],
            last_price=float(df["close"].iloc[-1]),
            avg_volume=float(df["volume"].mean()),
        )
    except Exception as e:
        logger.warning(f"Failed to read cache for {symbol}: {e}")
        return None


def save_json(
    key: str,
    data: dict | list,
    settings: Optional[Settings] = None,
) -> Path:
    """Save JSON-serializable data to cache.

    Args:
        key: Cache key (used as filename).
        data: JSON-serializable data.
        settings: Configuration settings.

    Returns:
        Path to the saved JSON file.
    """
    filepath = _cache_file_path(key, ext=".json", settings=settings)
    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.debug(f"Saved JSON cache: {filepath}")
    return filepath


def load_json(
    key: str,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    settings: Optional[Settings] = None,
) -> Optional[dict | list]:
    """Load JSON data from cache.

    Args:
        key: Cache key (used as filename).
        max_age_hours: Maximum cache age in hours.
        settings: Configuration settings.

    Returns:
        Parsed JSON data if fresh cache exists, None otherwise.
    """
    filepath = _cache_file_path(key, ext=".json", settings=settings)

    if not is_fresh(filepath, max_age_hours):
        return None

    try:
        return json.loads(filepath.read_text())
    except Exception as e:
        logger.warning(f"Failed to read JSON cache for {key}: {e}")
        return None


def save_batch(
    ticker_data_map: dict[str, TickerData],
    settings: Optional[Settings] = None,
) -> int:
    """Save multiple TickerData objects to Parquet cache.

    Args:
        ticker_data_map: Dict mapping symbol -> TickerData.
        settings: Configuration settings.

    Returns:
        Number of tickers successfully cached.
    """
    saved = 0
    for symbol, data in ticker_data_map.items():
        try:
            save_ticker_parquet(data, settings=settings)
            saved += 1
        except Exception as e:
            logger.warning(f"Failed to cache {symbol}: {e}")
    logger.info(f"Cached {saved}/{len(ticker_data_map)} tickers to Parquet")
    return saved


def load_batch(
    symbols: list[str],
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    settings: Optional[Settings] = None,
) -> dict[str, TickerData]:
    """Load multiple tickers from Parquet cache.

    Args:
        symbols: List of ticker symbols.
        max_age_hours: Maximum cache age in hours.
        settings: Configuration settings.

    Returns:
        Dict mapping symbol -> TickerData for cache hits.
    """
    results = {}
    for symbol in symbols:
        data = load_ticker_parquet(symbol, max_age_hours=max_age_hours, settings=settings)
        if data is not None:
            results[symbol] = data
    logger.info(f"Cache hits: {len(results)}/{len(symbols)} tickers")
    return results


def clear_cache(settings: Optional[Settings] = None) -> int:
    """Remove all files from the cache directory.

    Returns:
        Number of files removed.
    """
    cache_dir = _get_cache_dir(settings)
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file():
            f.unlink()
            removed += 1
    logger.info(f"Cleared {removed} files from cache")
    return removed
