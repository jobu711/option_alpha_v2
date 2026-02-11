"""Custom watchlist management with file-based persistence."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Store watchlists alongside config.json in the current working directory
_WATCHLIST_FILE: Path | None = None

MAX_WATCHLISTS = 20
MAX_TICKERS_PER_WATCHLIST = 200
WATCHLIST_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,48}[a-z0-9]$|^[a-z0-9]$")


def _get_watchlist_path() -> Path:
    """Get the path to the watchlists.json file."""
    global _WATCHLIST_FILE
    if _WATCHLIST_FILE is not None:
        return _WATCHLIST_FILE
    return Path("watchlists.json")


def set_watchlist_path(path: Path) -> None:
    """Set custom watchlist file path (for testing)."""
    global _WATCHLIST_FILE
    _WATCHLIST_FILE = path


def _load_watchlists() -> dict:
    """Load watchlists from file."""
    path = _get_watchlist_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Invalid watchlist file at {path}, returning defaults")
    return {"watchlists": {}, "active_watchlist": None}


def _save_watchlists(data: dict) -> None:
    """Save watchlists to file."""
    path = _get_watchlist_path()
    path.write_text(json.dumps(data, indent=2))


def _validate_name(name: str) -> None:
    """Validate watchlist name is kebab-case, 1-50 chars."""
    if not name or len(name) > 50:
        raise ValueError(f"Watchlist name must be 1-50 characters, got {len(name) if name else 0}")
    if not WATCHLIST_NAME_PATTERN.match(name):
        raise ValueError(
            f"Watchlist name must be kebab-case (lowercase letters, numbers, hyphens): {name}"
        )


def list_watchlists() -> dict[str, list[str]]:
    """Return all watchlists as {name: [tickers]}."""
    data = _load_watchlists()
    return data.get("watchlists", {})


def get_watchlist(name: str) -> list[str]:
    """Get tickers from a named watchlist. Raises KeyError if not found."""
    data = _load_watchlists()
    watchlists = data.get("watchlists", {})
    if name not in watchlists:
        raise KeyError(f"Watchlist '{name}' not found")
    return watchlists[name]


def create_watchlist(name: str, tickers: list[str]) -> None:
    """Create a new watchlist. Raises ValueError if name invalid or already exists."""
    _validate_name(name)
    data = _load_watchlists()
    watchlists = data.get("watchlists", {})

    if name in watchlists:
        raise ValueError(f"Watchlist '{name}' already exists. Use update_watchlist() to modify.")
    if len(watchlists) >= MAX_WATCHLISTS:
        raise ValueError(f"Maximum {MAX_WATCHLISTS} watchlists allowed")
    if len(tickers) > MAX_TICKERS_PER_WATCHLIST:
        raise ValueError(f"Maximum {MAX_TICKERS_PER_WATCHLIST} tickers per watchlist")

    # Normalize tickers: uppercase, deduplicate, sort
    normalized = sorted(set(t.upper().strip() for t in tickers if t.strip()))
    data["watchlists"][name] = normalized
    _save_watchlists(data)


def update_watchlist(name: str, tickers: list[str]) -> None:
    """Update an existing watchlist. Raises KeyError if not found."""
    data = _load_watchlists()
    if name not in data.get("watchlists", {}):
        raise KeyError(f"Watchlist '{name}' not found")
    if len(tickers) > MAX_TICKERS_PER_WATCHLIST:
        raise ValueError(f"Maximum {MAX_TICKERS_PER_WATCHLIST} tickers per watchlist")

    normalized = sorted(set(t.upper().strip() for t in tickers if t.strip()))
    data["watchlists"][name] = normalized
    _save_watchlists(data)


def delete_watchlist(name: str) -> None:
    """Delete a watchlist. Raises KeyError if not found."""
    data = _load_watchlists()
    if name not in data.get("watchlists", {}):
        raise KeyError(f"Watchlist '{name}' not found")

    del data["watchlists"][name]
    if data.get("active_watchlist") == name:
        data["active_watchlist"] = None
    _save_watchlists(data)


def get_active_watchlist() -> list[str]:
    """Return tickers from the active watchlist, or empty list if none."""
    data = _load_watchlists()
    active = data.get("active_watchlist")
    if active and active in data.get("watchlists", {}):
        return data["watchlists"][active]
    return []


def set_active_watchlist(name: Optional[str]) -> None:
    """Set the active watchlist. Pass None to deactivate. Raises KeyError if not found."""
    data = _load_watchlists()
    if name is not None and name not in data.get("watchlists", {}):
        raise KeyError(f"Watchlist '{name}' not found")
    data["active_watchlist"] = name
    _save_watchlists(data)


def validate_tickers(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Validate ticker symbols via yfinance. Returns (valid, invalid) lists."""
    import yfinance as yf

    valid = []
    invalid = []
    for symbol in symbols:
        sym = symbol.upper().strip()
        try:
            t = yf.Ticker(sym)
            # Check if ticker has any data
            if t.info and t.info.get("regularMarketPrice"):
                valid.append(sym)
            else:
                invalid.append(sym)
        except Exception:
            invalid.append(sym)
    return valid, invalid
