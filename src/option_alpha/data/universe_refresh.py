"""Dynamic universe refresh from SEC EDGAR + yfinance."""

import json
import logging
import shutil
import time
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
    settings: Optional[Settings] = None,
    max_retries: int = 3,
    regenerate: bool = False,
) -> dict:
    """Refresh universe data from SEC EDGAR + yfinance.

    Args:
        settings: Application settings.
        max_retries: Number of retry attempts on failure.
        regenerate: If True, full SEC EDGAR rebuild. If False, validate/prune
                    existing tickers only.

    Returns dict with keys: success, ticker_count, added, removed, mode, error
    """
    if settings is None:
        settings = get_settings()

    last_error = None
    for attempt in range(max_retries):
        try:
            result = await _do_refresh(settings, regenerate=regenerate)
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
        "mode": "regenerate" if regenerate else "validate",
    }


async def _do_refresh(
    settings: Optional[Settings] = None, regenerate: bool = False
) -> dict:
    """Execute the actual refresh logic.

    Args:
        settings: Application settings.
        regenerate: If True, full SEC EDGAR rebuild. If False, validate/prune
                    existing tickers only.
    """
    if settings is None:
        settings = get_settings()

    mode = "regenerate" if regenerate else "validate"

    # 1a. Preserve ETF entries before regeneration
    preserved_etfs: list[dict] = []
    if regenerate:
        current_before = _load_current()
        preserved_etfs = [
            t for t in current_before if t.get("asset_type") == "etf"
        ]
        if preserved_etfs:
            logger.info(
                f"Preserved {len(preserved_etfs)} ETF entries for re-merge"
            )

    if regenerate:
        # Full pipeline: fetch from SEC EDGAR
        tickers = await _fetch_edgar_tickers()
        logger.info(f"Fetched {len(tickers)} tickers from SEC EDGAR")
    else:
        # Validate-only: load existing stock tickers
        current_data = _load_current()
        tickers = [t["symbol"] for t in current_data]
        logger.info(
            f"Loaded {len(tickers)} existing tickers for validation"
        )

    # 2. Batch-validate optionability
    optionable = _validate_optionability(tickers)
    logger.info(f"Validated {len(optionable)} optionable tickers")

    # 3. Validate open interest
    oi_validated = _validate_open_interest(optionable, settings)
    logger.info(f"OI-validated {len(oi_validated)} tickers (threshold={settings.min_universe_oi})")

    # 4. Enrich with metadata
    enriched = _enrich_metadata(oi_validated)
    logger.info(f"Enriched {len(enriched)} tickers with metadata")

    # 4a. Merge preserved ETFs back (regenerate mode only)
    if regenerate and preserved_etfs:
        enriched_symbols = {t["symbol"] for t in enriched}
        for etf in preserved_etfs:
            if etf["symbol"] not in enriched_symbols:
                enriched.append(etf)
        enriched.sort(key=lambda t: t["symbol"])
        logger.info(
            f"Merged {len(preserved_etfs)} preserved ETFs; "
            f"total universe now {len(enriched)}"
        )

    # 4b. Count stocks and ETFs, emit size warnings
    stock_count = sum(1 for t in enriched if t.get("asset_type") != "etf")
    etf_count = sum(1 for t in enriched if t.get("asset_type") == "etf")
    size_warning: Optional[str] = None
    if stock_count < 500:
        size_warning = f"Low stock count: {stock_count} (expected >= 500)"
        logger.warning(size_warning)
    elif stock_count > 1500:
        size_warning = f"High stock count: {stock_count} (expected <= 1500)"
        logger.warning(size_warning)

    # 5. Diff against current
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

    # 6. Atomic write: tmp -> backup -> rename
    tmp_file = _UNIVERSE_FILE.parent / (_UNIVERSE_FILE.name + ".tmp")
    try:
        tmp_file.write_text(json.dumps(enriched, indent=2))
        if _UNIVERSE_FILE.exists():
            bak_file = _UNIVERSE_FILE.parent / (_UNIVERSE_FILE.name + ".bak")
            shutil.copy2(str(_UNIVERSE_FILE), str(bak_file))
        tmp_file.replace(_UNIVERSE_FILE)
    except Exception:
        # On failure, clean up tmp file if it exists; previous file stays intact
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    # 7. Clear the in-memory cache so next load picks up new data
    from option_alpha.data.universe import _clear_cache

    _clear_cache()

    # 8. Update meta
    meta: dict = {
        "last_refresh": datetime.now(UTC).isoformat(),
        "ticker_count": len(enriched),
        "stock_count": stock_count,
        "etf_count": etf_count,
        "added": len(added),
        "removed": len(removed),
        "mode": mode,
    }
    if size_warning:
        meta["size_warning"] = size_warning
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
    total = len(tickers)
    optionable = []
    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        for symbol in batch:
            try:
                t = yf.Ticker(symbol)
                if t.options:  # non-empty tuple of expiration dates
                    optionable.append(symbol)
            except Exception:
                continue
        checked = min(i + batch_size, total)
        logger.info(
            f"Optionability progress: {checked}/{total} checked, "
            f"{len(optionable)} optionable so far"
        )
    return optionable


def _validate_open_interest(tickers: list[str], settings: Settings) -> list[str]:
    """Filter tickers by total open interest on nearest expiration.

    ETFs are exempt from the OI check and always pass through.
    Tickers that fail lookup are excluded (fail-closed).
    """
    threshold = settings.min_universe_oi
    passed: list[str] = []

    for idx, symbol in enumerate(tickers):
        if idx > 0 and idx % 50 == 0:
            logger.info(f"OI validation progress: {idx}/{len(tickers)} tickers checked")

        try:
            t = yf.Ticker(symbol)

            # ETFs are exempt from OI check
            info = t.info
            if info.get("quoteType") == "ETF":
                logger.debug(f"{symbol}: ETF — exempt from OI check")
                passed.append(symbol)
                continue

            # Get nearest expiration and sum OI across calls and puts
            expirations = t.options
            if not expirations:
                logger.debug(f"{symbol}: no expirations found — excluded")
                time.sleep(2)
                continue

            nearest_expiry = expirations[0]
            chain = t.option_chain(nearest_expiry)
            total_oi = int(chain.calls["openInterest"].sum()) + int(
                chain.puts["openInterest"].sum()
            )

            if total_oi >= threshold:
                logger.debug(f"{symbol}: OI={total_oi} >= {threshold} — passed")
                passed.append(symbol)
            else:
                logger.debug(f"{symbol}: OI={total_oi} < {threshold} — excluded")

        except Exception as e:
            logger.debug(f"{symbol}: OI check failed ({e}) — excluded")

        time.sleep(2)

    logger.info(
        f"OI validation complete: {len(passed)}/{len(tickers)} tickers passed "
        f"(threshold={threshold})"
    )
    return passed


def _enrich_metadata(tickers: list[str]) -> list[dict]:
    """Enrich tickers with sector, market-cap, and asset type."""
    enriched = []
    for idx, symbol in enumerate(tickers):
        if idx > 0 and idx % 50 == 0:
            logger.info(f"Enrichment progress: {idx}/{len(tickers)} tickers enriched")
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
