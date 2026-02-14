"""SQLite connection management, WAL mode, and migration system."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_connection(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and foreign keys enabled.

    Args:
        db_path: Path to the database file, or ":memory:" for in-memory.

    Returns:
        Configured sqlite3.Connection with WAL mode and foreign keys.
    """
    db_path = str(db_path)
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    """Create the schema_version tracking table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INTEGER PRIMARY KEY,
            filename    TEXT    NOT NULL,
            applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def _get_applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Return the set of already-applied migration version numbers."""
    cursor = conn.execute("SELECT version FROM schema_version ORDER BY version")
    return {row[0] for row in cursor.fetchall()}


def _discover_migrations() -> list[tuple[int, Path]]:
    """Find all SQL migration files and return sorted (version, path) pairs.

    Migration files must be named NNN_description.sql where NNN is an integer.
    """
    if not MIGRATIONS_DIR.is_dir():
        return []

    migrations: list[tuple[int, Path]] = []
    for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        # Extract version number from filename like "001_initial.sql"
        name = sql_file.stem  # "001_initial"
        version_str = name.split("_", 1)[0]  # "001"
        try:
            version = int(version_str)
        except ValueError:
            logger.warning("Skipping migration with non-numeric prefix: %s", sql_file)
            continue
        migrations.append((version, sql_file))

    return sorted(migrations, key=lambda m: m[0])


def run_migrations(conn: sqlite3.Connection) -> list[int]:
    """Apply any pending migrations in order.

    Returns:
        List of version numbers that were applied.
    """
    _ensure_schema_version_table(conn)
    applied = _get_applied_versions(conn)
    all_migrations = _discover_migrations()
    newly_applied: list[int] = []

    for version, sql_path in all_migrations:
        if version in applied:
            logger.debug("Migration %03d already applied, skipping", version)
            continue

        logger.info("Applying migration %03d: %s", version, sql_path.name)
        sql = sql_path.read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_version (version, filename) VALUES (?, ?)",
            (version, sql_path.name),
        )
        conn.commit()
        newly_applied.append(version)
        logger.info("Migration %03d applied successfully", version)

    return newly_applied


def initialize_db(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Create/open database, run pending migrations, seed universe if needed, return connection.

    This is the main entry point for getting a ready-to-use database.

    Args:
        db_path: Path to the database file, or ":memory:" for in-memory.

    Returns:
        Fully migrated sqlite3.Connection.
    """
    conn = get_connection(db_path)
    applied = run_migrations(conn)
    if applied:
        logger.info("Applied migrations: %s", applied)

    # Seed universe after migrations (idempotent)
    from option_alpha.data.universe_service import seed_universe
    try:
        seed_universe(conn)
    except Exception:
        logger.debug("Universe tables not yet available, skipping seed")

    return conn
