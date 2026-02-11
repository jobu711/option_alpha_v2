"""Ticker universe management with pre-filtering.

Maintains a curated list of ~3000 optionable stocks and provides
filtering by minimum price and average volume.
"""

import logging
from typing import Optional

import yfinance as yf

from option_alpha.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Curated list of major optionable tickers across US exchanges.
# This is a representative subset; a full production list would include ~3000+.
# Organized by sector/index for maintainability.

# S&P 500 representative tickers (large-cap core)
SP500_CORE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
    "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK",
    "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
    "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO",
    "ATVI", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX",
    "BBWI", "BBY", "BDX", "BEN", "BF.B", "BIIB", "BIO", "BK", "BKNG", "BKR",
    "BLK", "BMY", "BR", "BRK.B", "BRO", "BSX", "BWA", "BXP", "C", "CAG",
    "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDAY", "CDNS",
    "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF",
    "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP",
    "COF", "COO", "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO",
    "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR",
    "D", "DAL", "DD", "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS",
    "DISH", "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA",
    "DVN", "DXC", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL",
    "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS",
    "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F",
    "FANG", "FAST", "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS", "FISV",
    "FITB", "FLT", "FMC", "FOX", "FOXA", "FRC", "FRT", "FTNT", "FTV", "GD",
    "GE", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC",
    "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HOLX",
    "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUM", "HWM", "IBM",
    "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP",
    "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT",
    "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC",
    "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "L", "LDOS", "LEN",
    "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW", "LRCX",
    "LUMN", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS",
    "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK",
    "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC",
    "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH",
    "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE",
    "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWL",
    "NWS", "NWSA", "NXPI", "O", "ODFL", "OGN", "OKE", "OMC", "ON", "ORCL",
    "ORLY", "OTIS", "OXY", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEAK", "PEG",
    "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PKI", "PLD",
    "PM", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX",
    "PTC", "PVH", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG",
    "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST",
    "RSG", "RTX", "RVTY", "SBAC", "SBNY", "SBUX", "SCHW", "SEE", "SHW", "SIVB",
    "SJM", "SLB", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STT",
    "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG",
    "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TMO", "TMUS", "TPR",
    "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN",
    "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI",
    "USB", "V", "VFC", "VICI", "VLO", "VMC", "VRSK", "VRSN", "VRTX", "VTR",
    "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC",
    "WHR", "WM", "WMB", "WMT", "WRB", "WRK", "WST", "WTW", "WY", "WYNN",
    "XEL", "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
]

# Popular mid/small-cap optionable tickers
POPULAR_OPTIONS = [
    "ABNB", "ACHR", "AI", "AFRM", "AG", "AMC", "AMPS", "ANNA", "APLD", "APP",
    "ARKG", "ARKK", "ARM", "ARQQ", "BABA", "BITO", "BLDE", "BLDP", "BNGO",
    "BTBT", "BYND", "CELH", "CHPT", "CIFR", "CLSK", "COIN", "CRWD", "CVNA",
    "CZOO", "DASH", "DDOG", "DFLI", "DNA", "DNUT", "DOCS", "DOCU", "DKNG",
    "DTST", "DUOL", "DWAC", "EDR", "ENPH", "ENVX", "EQX", "ERX", "ETHE",
    "EVGO", "EXAI", "FCEL", "FLNC", "FSLY", "FUBO", "FUTU", "GBTC", "GENI",
    "GEVO", "GME", "GNRC", "GNUS", "GRAB", "GRPN", "HIMS", "HOOD", "HUT",
    "HYMC", "IONQ", "IQ", "IOVA", "JD", "JOBY", "KGC", "KVUE", "LAZR",
    "LCID", "LI", "LMND", "LOVE", "LQDA", "LTHM", "LYFT", "MARA", "MELI",
    "MGNI", "MNDY", "MSTR", "MQ", "MTTR", "MULN", "NET", "NIO", "NKLA",
    "NU", "NVAX", "OKTA", "OPEN", "OPK", "ORGN", "PANW", "PATH", "PAYO",
    "PCOR", "PINS", "PLTR", "PLUG", "PRCH", "QS", "QUBT", "RBLX", "RDDT",
    "RGTI", "RIOT", "RIVN", "RIVN", "ROKU", "RSKD", "RUM", "S", "SE",
    "SEDG", "SHOP", "SKLZ", "SNAP", "SNOW", "SOFI", "SOUN", "SPOT", "SQ",
    "STEM", "STNE", "TASK", "TGTX", "TLRY", "TNA", "TQQQ", "TRIP",
    "TTD", "TWLO", "U", "UBER", "UPST", "URBN", "VRM", "VUZI", "W",
    "WDAY", "WISH", "WKHS", "XBI", "XLE", "XLF", "XLK", "XPEV", "ZI",
    "ZM", "ZS",
]

# ETFs with high options volume
OPTIONABLE_ETFS = [
    "DIA", "EEM", "EFA", "EWJ", "EWZ", "FXI", "GDX", "GDXJ", "GLD", "HYG",
    "IBB", "IEF", "IGV", "IJR", "IVV", "IWD", "IWF", "IWM", "IWN", "IYR",
    "JNUG", "KRE", "LQD", "MDY", "MSOS", "OIH", "ONEQ", "QQQ", "QUAL", "RSP",
    "SCHD", "SLV", "SMH", "SOXX", "SOXL", "SOXS", "SPXL", "SPXS", "SPY",
    "SQQQ", "TLT", "TQQQ", "USO", "UVXY", "VGK", "VNQ", "VO", "VOO", "VTI",
    "VTV", "VWO", "VXX", "XBI", "XHB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLU", "XLV", "XLY", "XME", "XOP", "XRT",
]


def get_full_universe() -> list[str]:
    """Return the complete curated ticker universe (deduplicated, sorted)."""
    all_tickers = set(SP500_CORE + POPULAR_OPTIONS + OPTIONABLE_ETFS)
    return sorted(all_tickers)


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
