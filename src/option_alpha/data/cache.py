"""Parquet/JSON caching layer with freshness checks.

Provides transparent caching for OHLCV data (Parquet) and
JSON-serializable data with date-based file naming.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from option_alpha.config import Settings, get_settings
from option_alpha.models import FetchErrorType, TickerData

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
        date = datetime.now(UTC)
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


def evict_stale_cache(
    current_universe: set[str],
    max_age_days: int = 30,
    settings: Optional[Settings] = None,
) -> int:
    """Remove cached Parquet files for tickers no longer in the universe.

    Only removes files older than max_age_days to avoid removing
    recently-cached data for tickers temporarily removed from universe.

    Returns count of files removed.
    """
    cache_dir = _get_cache_dir(settings)
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    removed = 0

    for parquet_file in cache_dir.glob("*.parquet"):
        # Extract ticker from filename (format: SYMBOL_YYYY-MM-DD.parquet)
        stem = parquet_file.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        symbol = parts[0]

        if symbol not in current_universe:
            # Check file age
            mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime, tz=UTC)
            if mtime < cutoff:
                parquet_file.unlink()
                removed += 1
                logger.debug("Evicted stale cache: %s", parquet_file.name)

    if removed:
        logger.info("Evicted %d stale cache files", removed)
    return removed


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


# ---------------------------------------------------------------------------
# Failure cache — JSON file tracking fetch errors with TTL-based eviction
# ---------------------------------------------------------------------------

_FAILURE_CACHE_FILENAME = "_failures.json"


def _failure_cache_path(settings: Optional[Settings] = None) -> Path:
    """Return the path to the failure cache JSON file."""
    return _get_cache_dir(settings) / _FAILURE_CACHE_FILENAME


def load_failure_cache(
    ttl_hours: float = 24,
    settings: Optional[Settings] = None,
) -> dict[str, dict]:
    """Load the failure cache, evicting entries older than *ttl_hours*.

    Args:
        ttl_hours: Maximum age of a failure entry in hours.
        settings: Configuration settings.

    Returns:
        Dict mapping ticker symbol to failure entry.  Each entry contains
        ``error_type``, ``timestamp``, and ``message``.
        Returns an empty dict on any I/O or parse error.
    """
    filepath = _failure_cache_path(settings)
    if not filepath.exists():
        return {}

    try:
        raw = filepath.read_text()
        if not raw.strip():
            return {}
        data: dict = json.loads(raw)
    except Exception as e:
        logger.warning(f"Failure cache corrupt or unreadable, ignoring: {e}")
        return {}

    cutoff = datetime.now(UTC) - timedelta(hours=ttl_hours)
    active: dict[str, dict] = {}
    for ticker, entry in data.items():
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            # Ensure timezone-aware for comparison
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts >= cutoff:
                active[ticker] = entry
        except Exception:
            # Skip malformed entries silently
            continue

    return active


def record_failures(
    failures: dict[str, dict],
    settings: Optional[Settings] = None,
) -> None:
    """Append or update failure entries in the cache file.

    Args:
        failures: Dict mapping ticker symbol to a failure entry.  Each entry
            must include ``error_type`` (a :class:`FetchErrorType` value or
            its string equivalent), ``timestamp`` (ISO-8601 string), and
            ``message``.
        settings: Configuration settings.
    """
    if not failures:
        return

    filepath = _failure_cache_path(settings)

    # Load existing entries (no TTL eviction — we just merge)
    existing: dict[str, dict] = {}
    if filepath.exists():
        try:
            raw = filepath.read_text()
            if raw.strip():
                existing = json.loads(raw)
        except Exception as e:
            logger.warning(f"Failure cache unreadable, overwriting: {e}")

    # Normalise error_type to its string value (handles FetchErrorType enums)
    for ticker, entry in failures.items():
        et = entry.get("error_type", FetchErrorType.UNKNOWN)
        if isinstance(et, FetchErrorType):
            entry["error_type"] = et.value
        existing[ticker] = entry

    try:
        filepath.write_text(json.dumps(existing, indent=2, default=str))
        logger.debug(f"Recorded {len(failures)} failure(s) to {filepath}")
    except Exception as e:
        logger.warning(f"Failed to write failure cache: {e}")


def clear_failure_cache(settings: Optional[Settings] = None) -> None:
    """Delete the failure cache file.

    No-op if the file does not exist.
    """
    filepath = _failure_cache_path(settings)
    try:
        if filepath.exists():
            filepath.unlink()
            logger.info("Failure cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear failure cache: {e}")


def get_failure_cache_stats(
    ttl_hours: float = 24,
    settings: Optional[Settings] = None,
) -> dict:
    """Return summary statistics for the failure cache.

    Args:
        ttl_hours: TTL used when loading (expired entries are excluded).
        settings: Configuration settings.

    Returns:
        Dict with keys ``count``, ``by_type`` (error_type -> count), and
        ``oldest`` (ISO-8601 string of the oldest non-expired entry, or
        ``None``).
    """
    entries = load_failure_cache(ttl_hours=ttl_hours, settings=settings)
    stats: dict = {"count": len(entries), "by_type": {}, "oldest": None}

    if not entries:
        return stats

    oldest_ts: Optional[datetime] = None
    for entry in entries.values():
        error_type = entry.get("error_type", FetchErrorType.UNKNOWN.value)
        stats["by_type"][error_type] = stats["by_type"].get(error_type, 0) + 1

        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
        except Exception:
            continue

    if oldest_ts is not None:
        stats["oldest"] = oldest_ts.isoformat()

    return stats
