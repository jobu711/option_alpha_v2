-- 002_universe.sql: Universe management tables for ticker/tag system

CREATE TABLE IF NOT EXISTS universe_tickers (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    source TEXT NOT NULL DEFAULT 'preset',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_scanned_at TEXT
);

CREATE TABLE IF NOT EXISTS universe_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    slug TEXT NOT NULL UNIQUE,
    is_preset INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ticker_tags (
    symbol TEXT NOT NULL REFERENCES universe_tickers(symbol) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES universe_tags(id) ON DELETE CASCADE,
    PRIMARY KEY (symbol, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_universe_tickers_active ON universe_tickers(is_active);
CREATE INDEX IF NOT EXISTS idx_universe_tags_slug ON universe_tags(slug);
CREATE INDEX IF NOT EXISTS idx_universe_tags_preset ON universe_tags(is_preset);
CREATE INDEX IF NOT EXISTS idx_ticker_tags_tag_id ON ticker_tags(tag_id);
