#!/usr/bin/env python3
"""Generate universe_data.json from SEC EDGAR + yfinance.

This is a one-time generation script. It fetches ~10k US-listed tickers from
SEC EDGAR, validates optionability via yfinance, enriches with sector/market-cap
metadata, and writes the result to src/option_alpha/data/universe_data.json.

Usage:
    python scripts/generate_universe.py            # Full generation (~2-4 hours)
    python scripts/generate_universe.py --sample   # Small representative sample (~5 min)

Requirements:
    - Internet access (SEC EDGAR + yfinance API calls)
    - yfinance installed (pip install yfinance)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance is required. Install with: pip install yfinance")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install with: pip install httpx")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).parent.parent / "src" / "option_alpha" / "data" / "universe_data.json"

SEC_EDGAR_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_EDGAR_HEADERS = {"User-Agent": "OptionAlpha/1.0 (contact@example.com)"}

# Market cap tier boundaries
LARGE_CAP = 10_000_000_000   # >$10B
MID_CAP = 2_000_000_000      # $2B-$10B
SMALL_CAP = 300_000_000      # $300M-$2B
# Below $300M = micro

# Representative sample tickers for --sample mode
SAMPLE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH",
    "JNJ", "XOM", "PG", "MA", "HD", "MRK", "ABBV", "PFE", "KO", "PEP",
    "COST", "AVGO", "TMO", "MCD", "WMT", "CSCO", "CRM", "ABT", "LIN", "DHR",
    # Mid-cap
    "CRWD", "DDOG", "NET", "SNAP", "PINS", "ROKU", "SOFI", "HOOD", "COIN", "RBLX",
    # Small-cap
    "FUBO", "LCID", "PLUG", "CHPT", "IONQ", "MARA", "RIOT", "BYND", "WISH", "OPEN",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "XLF", "XLE", "XLK", "TLT",
]


def fetch_sec_tickers() -> list[str]:
    """Fetch ticker symbols from SEC EDGAR company_tickers.json."""
    logger.info("Fetching tickers from SEC EDGAR...")
    resp = httpx.get(SEC_EDGAR_URL, headers=SEC_EDGAR_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    symbols = []
    for entry in data.values():
        ticker = entry.get("ticker", "")
        if ticker and "." not in ticker and "-" not in ticker and len(ticker) <= 5:
            symbols.append(ticker.upper())

    symbols = sorted(set(symbols))
    logger.info(f"Found {len(symbols)} candidate tickers from SEC EDGAR")
    return symbols


def classify_market_cap(market_cap: float | None) -> str:
    """Classify market cap into tier."""
    if market_cap is None or market_cap <= 0:
        return ""
    if market_cap >= LARGE_CAP:
        return "large"
    if market_cap >= MID_CAP:
        return "mid"
    if market_cap >= SMALL_CAP:
        return "small"
    return "micro"


def enrich_ticker(symbol: str) -> dict | None:
    """Fetch metadata for a single ticker via yfinance.

    Returns a dict with symbol, name, sector, market_cap_tier, asset_type,
    or None if the ticker is not optionable.
    """
    try:
        t = yf.Ticker(symbol)

        # Check optionability
        try:
            options = t.options
            if not options:
                return None
        except Exception:
            return None

        # Fetch info for enrichment
        info = t.info or {}

        quote_type = info.get("quoteType", "")
        asset_type = "etf" if quote_type == "ETF" else "stock"

        name = info.get("shortName", "") or info.get("longName", "")
        sector = info.get("sector", "") if asset_type == "stock" else ""
        market_cap = info.get("marketCap")
        market_cap_tier = classify_market_cap(market_cap) if asset_type == "stock" else ""

        return {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "market_cap_tier": market_cap_tier,
            "asset_type": asset_type,
        }
    except Exception as e:
        logger.debug(f"Error enriching {symbol}: {e}")
        return None


def generate_universe(symbols: list[str], batch_delay: float = 0.5) -> list[dict]:
    """Validate and enrich a list of ticker symbols.

    Args:
        symbols: List of ticker symbols to process.
        batch_delay: Seconds to wait between tickers (rate limiting).

    Returns:
        List of enriched ticker dicts.
    """
    universe = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        if i % 50 == 0 or i == 1:
            logger.info(f"Processing {i}/{total} ({len(universe)} optionable so far)")

        result = enrich_ticker(symbol)
        if result:
            universe.append(result)

        if batch_delay > 0 and i < total:
            time.sleep(batch_delay)

    logger.info(f"Completed: {len(universe)} optionable tickers out of {total} candidates")
    return universe


def write_universe(universe: list[dict], output_path: Path) -> None:
    """Write universe data to JSON file."""
    universe_sorted = sorted(universe, key=lambda t: t["symbol"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(universe_sorted, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(universe_sorted)} tickers to {output_path}")

    # Summary stats
    stocks = sum(1 for t in universe_sorted if t["asset_type"] == "stock")
    etfs = sum(1 for t in universe_sorted if t["asset_type"] == "etf")
    large = sum(1 for t in universe_sorted if t["market_cap_tier"] == "large")
    mid = sum(1 for t in universe_sorted if t["market_cap_tier"] == "mid")
    small = sum(1 for t in universe_sorted if t["market_cap_tier"] == "small")
    micro = sum(1 for t in universe_sorted if t["market_cap_tier"] == "micro")
    logger.info(f"  Stocks: {stocks}, ETFs: {etfs}")
    logger.info(f"  Large: {large}, Mid: {mid}, Small: {small}, Micro: {micro}")


def main():
    parser = argparse.ArgumentParser(description="Generate universe_data.json")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate a small representative sample (~50 tickers) for testing/CI",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output file path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    args = parser.parse_args()

    if args.sample:
        logger.info("Running in --sample mode (small representative set)")
        symbols = SAMPLE_TICKERS
    else:
        symbols = fetch_sec_tickers()

    universe = generate_universe(symbols, batch_delay=args.delay)

    if not universe:
        logger.error("No optionable tickers found! Check network connection.")
        sys.exit(1)

    write_universe(universe, args.output)
    logger.info("Done!")


if __name__ == "__main__":
    main()
