"""yfinance batch data fetcher with threading and retry/backoff.

Downloads OHLCV data for ticker groups using ThreadPoolExecutor
with exponential backoff retry logic and error classification.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Callable, Optional

import pandas as pd
import yfinance as yf

from option_alpha.config import Settings, get_settings
from option_alpha.models import FetchErrorType, TickerData

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAYS = [1.0, 2.0, 4.0]  # seconds: exponential backoff


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    delays: list[float] | None = None,
) -> Callable:
    """Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        delays: List of delay durations in seconds for each retry.
    """
    if delays is None:
        delays = list(DEFAULT_RETRY_DELAYS)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = delays[min(attempt, len(delays) - 1)]
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                            f"{func.__name__}: {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for "
                            f"{func.__name__}: {e}"
                        )
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def classify_fetch_error(error: Exception) -> FetchErrorType:
    """Classify a fetch exception into a FetchErrorType.

    Parses the exception message to determine the category of failure.

    Args:
        error: The exception raised during fetching.

    Returns:
        FetchErrorType classification.
    """
    msg = str(error).lower()
    if "delisted" in msg:
        return FetchErrorType.DELISTED
    if "401" in msg or "rate" in msg:
        return FetchErrorType.RATE_LIMITED
    if "connection" in msg or "timeout" in msg:
        return FetchErrorType.NETWORK
    if "insufficient" in msg:
        return FetchErrorType.INSUFFICIENT_DATA
    return FetchErrorType.UNKNOWN


def _download_batch(
    symbols: list[str],
    period: str = "6mo",
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delays: list[float] | None = None,
) -> pd.DataFrame:
    """Download OHLCV data for a batch of symbols with retry logic.

    Args:
        symbols: List of ticker symbols.
        period: yfinance period string (e.g., '6mo', '1y').
        max_retries: Maximum number of retry attempts.
        retry_delays: List of delay durations in seconds for each retry.

    Returns:
        DataFrame with OHLCV data (MultiIndex columns if multiple symbols).

    Raises:
        Exception: Re-raises the last exception after all retries exhausted.
    """
    if retry_delays is None:
        retry_delays = list(DEFAULT_RETRY_DELAYS)

    symbols_str = " ".join(symbols)
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Downloading {len(symbols)} tickers: {symbols_str[:80]}...")

            df = yf.download(
                symbols_str,
                period=period,
                progress=False,
                threads=True,
            )

            if df.empty:
                raise ValueError(f"Empty DataFrame returned for {symbols_str[:50]}...")

            return df
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                    f"_download_batch: {e}. Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for "
                    f"_download_batch: {e}"
                )

    raise last_exception  # type: ignore[misc]


def _parse_ticker_data(
    symbol: str,
    df: pd.DataFrame,
    is_single: bool = False,
) -> Optional[TickerData]:
    """Extract TickerData for a single symbol from a batch DataFrame.

    Args:
        symbol: Ticker symbol.
        df: DataFrame from yfinance download.
        is_single: True if df contains only one ticker (no MultiIndex).

    Returns:
        TickerData or None if data is insufficient.
    """
    try:
        # Detect if DataFrame has MultiIndex columns (multi-ticker)
        has_multi_index = isinstance(df.columns, pd.MultiIndex)

        if is_single or not has_multi_index:
            close = df["Close"].dropna()
            open_ = df["Open"].dropna()
            high = df["High"].dropna()
            low = df["Low"].dropna()
            volume = df["Volume"].dropna()
        else:
            if symbol not in df["Close"].columns:
                return None
            close = df["Close"][symbol].dropna()
            open_ = df["Open"][symbol].dropna()
            high = df["High"][symbol].dropna()
            low = df["Low"][symbol].dropna()
            volume = df["Volume"][symbol].dropna()

        if close.empty or len(close) < 5:
            logger.debug(f"Insufficient data for {symbol}: {len(close)} rows")
            return None

        return TickerData(
            symbol=symbol,
            dates=close.index.tolist(),
            open=open_.tolist(),
            high=high.tolist(),
            low=low.tolist(),
            close=close.tolist(),
            volume=[int(v) for v in volume.tolist()],
            last_price=float(close.iloc[-1]),
            avg_volume=float(volume.mean()),
        )

    except (KeyError, IndexError, TypeError) as e:
        logger.debug(f"Failed to parse data for {symbol}: {e}")
        return None


def fetch_batch(
    symbols: list[str],
    period: str = "6mo",
    batch_size: int = 20,
    max_workers: int = 2,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delays: list[float] | None = None,
    failure_tracker: dict[str, FetchErrorType] | None = None,
) -> dict[str, TickerData]:
    """Fetch OHLCV data for multiple symbols using parallel batch downloads.

    Splits symbols into groups and downloads each group in parallel using
    ThreadPoolExecutor. Each group download has retry/backoff logic.
    Inter-batch stagger delays reduce rate-limiting risk.

    Args:
        symbols: List of ticker symbols to fetch.
        period: yfinance period string (default '6mo').
        batch_size: Number of symbols per batch (default 20).
        max_workers: Maximum parallel threads (default 2).
        max_retries: Maximum retry attempts per batch (default 3).
        retry_delays: Delay durations in seconds for each retry.
        failure_tracker: Optional dict populated with symbol -> FetchErrorType
            for tickers that failed to fetch. Allows callers to inspect
            error classifications for downstream handling.

    Returns:
        Dict mapping symbol -> TickerData for successfully fetched tickers.
    """
    if retry_delays is None:
        retry_delays = list(DEFAULT_RETRY_DELAYS)

    if not symbols:
        return {}

    # Split into batches
    batches = [
        symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
    ]
    logger.info(
        f"Fetching {len(symbols)} tickers in {len(batches)} batches "
        f"(batch_size={batch_size}, workers={max_workers})"
    )

    results: dict[str, TickerData] = {}
    failures: dict[str, FetchErrorType] = {}

    def _process_batch(
        batch: list[str],
        batch_idx: int = 0,
    ) -> tuple[dict[str, TickerData], dict[str, FetchErrorType]]:
        """Download and parse a single batch with inter-batch stagger.

        Args:
            batch: List of ticker symbols in this batch.
            batch_idx: Zero-based index of this batch, used for stagger delay.

        Returns:
            Tuple of (successful results dict, failure classifications dict).
        """
        # Stagger: delay proportional to batch index to reduce rate-limiting
        if batch_idx > 0:
            stagger = batch_idx * 0.5
            logger.debug(f"Batch {batch_idx}: stagger delay {stagger}s")
            time.sleep(stagger)

        batch_results: dict[str, TickerData] = {}
        batch_failures: dict[str, FetchErrorType] = {}
        try:
            df = _download_batch(
                batch,
                period=period,
                max_retries=max_retries,
                retry_delays=retry_delays,
            )
            is_single = len(batch) == 1

            for symbol in batch:
                ticker_data = _parse_ticker_data(symbol, df, is_single=is_single)
                if ticker_data is not None:
                    batch_results[symbol] = ticker_data
                else:
                    batch_failures[symbol] = FetchErrorType.INSUFFICIENT_DATA
        except Exception as e:
            error_type = classify_fetch_error(e)
            logger.error(
                f"Batch download failed after retries: {e} "
                f"(classified as {error_type.value})"
            )
            for symbol in batch:
                if symbol not in batch_results:
                    batch_failures[symbol] = error_type
        return batch_results, batch_failures

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_batch, batch, idx): batch
            for idx, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch = futures[future]
            try:
                batch_results, batch_failures = future.result()
                results.update(batch_results)
                failures.update(batch_failures)
                logger.info(
                    f"Batch complete: {len(batch_results)}/{len(batch)} tickers"
                )
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    # Populate caller-provided failure tracker if given
    if failure_tracker is not None:
        failure_tracker.update(failures)

    if failures:
        logger.info(
            f"Fetch failures: {len(failures)} tickers - "
            + ", ".join(f"{s}:{t.value}" for s, t in failures.items())
        )

    logger.info(f"Fetch complete: {len(results)}/{len(symbols)} tickers successful")
    return results


def fetch_single(symbol: str, period: str = "6mo") -> Optional[TickerData]:
    """Fetch OHLCV data for a single ticker.

    Convenience wrapper around fetch_batch for single-ticker use.

    Args:
        symbol: Ticker symbol.
        period: yfinance period string.

    Returns:
        TickerData or None if fetch failed.
    """
    results = fetch_batch([symbol], period=period)
    return results.get(symbol)
