"""Dynamic universe refresh from SEC EDGAR + yfinance."""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import yfinance as yf

from option_alpha.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Paths
_DATA_DIR = Path(__file__).parent
_UNIVERSE_FILE = _DATA_DIR / "universe_data.json"
_META_FILE = _DATA_DIR / "universe_meta.json"

SEC_EDGAR_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_EDGAR_HEADERS = {"User-Agent": "OptionAlpha/1.0 contact@example.com"}


def should_refresh(settings: Optional[Settings] = None) -> bool:
    """Check if universe needs refreshing (last refresh > interval days ago)."""
    if settings is None:
        settings = get_settings()
    if not _META_FILE.exists():
        return True
    try:
        meta = json.loads(_META_FILE.read_text())
        last_refresh = datetime.fromisoformat(meta["last_refresh"])
        return (
            datetime.now(UTC) - last_refresh
            > timedelta(days=settings.universe_refresh_interval_days)
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return True


async def refresh_universe(
    settings: Optional[Settings] = None, max_retries: int = 3
) -> dict:
    """Refresh universe data from SEC EDGAR + yfinance.

    Returns dict with keys: success, ticker_count, added, removed, error
    """
    if settings is None:
        settings = get_settings()

    last_error = None
    for attempt in range(max_retries):
        try:
            result = await _do_refresh()
            return result
        except Exception as e:
            last_error = e
            logger.warning(
                f"Refresh attempt {attempt + 1}/{max_retries} failed: {e}"
            )

    logger.error(
        f"All {max_retries} refresh attempts failed. Keeping existing universe."
    )
    return {
        "success": False,
        "error": str(last_error),
        "ticker_count": 0,
        "added": 0,
        "removed": 0,
    }


async def _do_refresh() -> dict:
    """Execute the actual refresh logic."""
    # 1. Fetch SEC EDGAR tickers
    edgar_tickers = await _fetch_edgar_tickers()
    logger.info(f"Fetched {len(edgar_tickers)} tickers from SEC EDGAR")

    # 2. Batch-validate optionability
    optionable = _validate_optionability(edgar_tickers)
    logger.info(f"Validated {len(optionable)} optionable tickers")

    # 3. Enrich with metadata
    enriched = _enrich_metadata(optionable)
    logger.info(f"Enriched {len(enriched)} tickers with metadata")

    # 4. Diff against current
    current = _load_current()
    current_symbols = {t["symbol"] for t in current}
    new_symbols = {t["symbol"] for t in enriched}
    added = new_symbols - current_symbols
    removed = current_symbols - new_symbols

    if added:
        logger.info(
            f"New tickers: {sorted(added)[:20]}"
            f"{'...' if len(added) > 20 else ''}"
        )
    if removed:
        logger.info(
            f"Removed tickers: {sorted(removed)[:20]}"
            f"{'...' if len(removed) > 20 else ''}"
        )

    # 5. Write updated file
    _UNIVERSE_FILE.write_text(json.dumps(enriched, indent=2))

    # 6. Clear the in-memory cache so next load picks up new data
    from option_alpha.data.universe import _clear_cache

    _clear_cache()

    # 7. Update meta
    meta = {
        "last_refresh": datetime.now(UTC).isoformat(),
        "ticker_count": len(enriched),
        "added": len(added),
        "removed": len(removed),
    }
    _META_FILE.write_text(json.dumps(meta, indent=2))

    return {"success": True, **meta}


async def _fetch_edgar_tickers() -> list[str]:
    """Fetch US-listed tickers from SEC EDGAR."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            SEC_EDGAR_URL, headers=SEC_EDGAR_HEADERS, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

    # data is {index: {cik_str, ticker, title}}
    tickers = []
    for entry in data.values():
        ticker = entry.get("ticker", "").upper()
        if ticker and "." not in ticker and len(ticker) <= 5:
            tickers.append(ticker)
    return sorted(set(tickers))


def _validate_optionability(
    tickers: list[str], batch_size: int = 50
) -> list[str]:
    """Validate which tickers have options chains."""
    optionable = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        for symbol in batch:
            try:
                t = yf.Ticker(symbol)
                if t.options:  # non-empty tuple of expiration dates
                    optionable.append(symbol)
            except Exception:
                continue
    return optionable


def _enrich_metadata(tickers: list[str]) -> list[dict]:
    """Enrich tickers with sector, market-cap, and asset type."""
    enriched = []
    for symbol in tickers:
        try:
            info = yf.Ticker(symbol).info
            market_cap = info.get("marketCap", 0) or 0
            tier = _classify_market_cap(market_cap)
            quote_type = info.get("quoteType", "EQUITY")
            asset_type = "etf" if quote_type == "ETF" else "stock"
            enriched.append(
                {
                    "symbol": symbol,
                    "name": info.get("shortName", info.get("longName", "")),
                    "sector": info.get("sector", "")
                    if asset_type == "stock"
                    else "",
                    "market_cap_tier": tier if asset_type == "stock" else "",
                    "asset_type": asset_type,
                }
            )
        except Exception:
            enriched.append(
                {
                    "symbol": symbol,
                    "name": "",
                    "sector": "",
                    "market_cap_tier": "",
                    "asset_type": "stock",
                }
            )
    return enriched


def _classify_market_cap(market_cap: int) -> str:
    """Classify market cap into tiers."""
    if market_cap > 10_000_000_000:
        return "large"
    elif market_cap > 2_000_000_000:
        return "mid"
    elif market_cap > 300_000_000:
        return "small"
    elif market_cap > 0:
        return "micro"
    return ""


def _load_current() -> list[dict]:
    """Load current universe data for diffing."""
    if _UNIVERSE_FILE.exists():
        try:
            return json.loads(_UNIVERSE_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []
