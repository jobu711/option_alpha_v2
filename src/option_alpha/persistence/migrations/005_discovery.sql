CREATE TABLE IF NOT EXISTS discovery_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    cboe_symbols_fetched INTEGER DEFAULT 0,
    new_tickers_added INTEGER DEFAULT 0,
    stale_tickers_deactivated INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS discovery_failures (
    symbol TEXT PRIMARY KEY,
    reason TEXT,
    failed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_discovery_failures_failed_at ON discovery_failures(failed_at);

INSERT OR IGNORE INTO universe_tags (name, slug, is_preset, is_active)
VALUES ('Auto-Discovered', 'auto-discovered', 1, 1);
