"""Pydantic-based configuration with JSON file persistence."""

import json
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


DEFAULT_SCORING_WEIGHTS = {
    "bb_width": 0.12,
    "atr_percentile": 0.08,
    "rsi": 0.08,
    "obv_trend": 0.06,
    "sma_alignment": 0.08,
    "relative_volume": 0.06,
    "catalyst_proximity": 0.15,
    "stoch_rsi": 0.06,
    "williams_r": 0.04,
    "roc": 0.04,
    "adx": 0.08,
    "keltner_width": 0.05,
    "vwap_deviation": 0.05,
    "ad_trend": 0.05,
}

CONFIG_FILE = Path("config.json")


class Settings(BaseSettings):
    """Application settings with sensible defaults."""

    # --- Scoring ---
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_SCORING_WEIGHTS.copy()
    )
    min_composite_score: float = 50.0
    direction_rsi_strong_bullish: float = 60.0
    direction_rsi_strong_bearish: float = 40.0

    # --- Options filtering ---
    dte_min: int = 30
    dte_max: int = 60
    min_open_interest: int = 100
    max_bid_ask_spread_pct: float = 0.30
    min_option_volume: int = 1

    # --- AI backend ---
    ai_backend: str = "ollama"
    ollama_model: str = "llama3.1:8b"
    claude_api_key: Optional[str] = None
    ai_retry_delays: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0, 16.0])
    ai_request_timeout: int = 120
    ai_per_ticker_timeout: int = 60
    ai_health_check_timeout: int = 15
    ai_debate_phase_timeout: int = 600
    ai_debate_concurrency: int = 1

    # --- Pipeline ---
    top_n_options: int = 50
    top_n_ai_debate: int = 10

    # --- Data pre-filters ---
    min_price: float = 5.0
    min_avg_volume: int = 500_000

    # --- Paths ---
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/option_alpha.db")

    # --- Risk-free rate ---
    fred_api_key: Optional[str] = None
    risk_free_rate_fallback: float = 0.05

    # --- Universe filtering ---
    universe_presets: list[str] = Field(default_factory=lambda: ["full"])
    universe_sectors: list[str] = Field(default_factory=list)
    universe_refresh_interval_days: int = 7
    min_universe_oi: int = 100
    universe_refresh_schedule: str = "sat"

    # --- Fetch settings ---
    fetch_max_retries: int = 3
    fetch_retry_delays: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0])
    fetch_batch_size: int = 50
    fetch_max_workers: int = 4
    failure_cache_ttl_hours: int = 24
    data_fetch_period: str = "1y"

    model_config = {
        "env_prefix": "OPTION_ALPHA_",
        "env_file": ".env",
        "extra": "ignore",
    }

    def save(self, path: Optional[Path] = None) -> None:
        """Persist current settings to a JSON file."""
        target = path or CONFIG_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        # Convert Path objects to strings for JSON serialization
        for key in ("data_dir", "db_path"):
            if key in data and data[key] is not None:
                data[key] = str(data[key])
        target.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings from JSON file, falling back to defaults."""
        target = path or CONFIG_FILE
        if target.exists():
            raw = json.loads(target.read_text())
            return cls(**raw)
        return cls()


def get_settings(path: Optional[Path] = None) -> Settings:
    """Get settings, loading from config file if available."""
    return Settings.load(path)


def get_effective_ai_settings(settings: Settings) -> dict:
    """Return backend-appropriate AI settings, respecting user overrides.

    For each key, if the current value matches the Settings class field default,
    the backend-specific default is used instead.  If the user explicitly changed
    the value (so it differs from the class default), their override is kept.
    """
    OLLAMA_DEFAULTS = {
        "ai_debate_concurrency": 1,
        "ai_per_ticker_timeout": 180,
        "ai_request_timeout": 120,
        "ai_retry_delays": [2.0, 4.0],
    }
    CLAUDE_DEFAULTS = {
        "ai_debate_concurrency": 1,
        "ai_per_ticker_timeout": 60,
        "ai_request_timeout": 30,
        "ai_retry_delays": [1.0, 2.0, 4.0],
    }

    backend_defaults = OLLAMA_DEFAULTS if settings.ai_backend == "ollama" else CLAUDE_DEFAULTS

    result = {}
    for key, backend_val in backend_defaults.items():
        current = getattr(settings, key)
        field_info = Settings.model_fields[key]
        class_default = (
            field_info.default_factory() if field_info.default_factory else field_info.default
        )
        result[key] = backend_val if current == class_default else current
    return result
