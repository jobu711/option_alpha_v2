"""SQLite persistence layer for Option Alpha scan results."""

from option_alpha.persistence.database import get_connection, initialize_db
from option_alpha.persistence.repository import (
    get_all_scans,
    get_latest_scan,
    get_scan_by_id,
    get_ticker_history,
    save_ai_theses,
    save_scan_run,
    save_ticker_scores,
)

__all__ = [
    "get_connection",
    "initialize_db",
    "save_scan_run",
    "save_ticker_scores",
    "save_ai_theses",
    "get_latest_scan",
    "get_scan_by_id",
    "get_ticker_history",
    "get_all_scans",
]
