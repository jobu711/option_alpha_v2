"""Pydantic-based configuration with JSON file persistence."""

import json
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


DEFAULT_SCORING_WEIGHTS = {
    "bb_width": 0.20,
    "atr_percentile": 0.15,
    "rsi": 0.10,
    "obv_trend": 0.10,
    "sma_alignment": 0.10,
    "relative_volume": 0.10,
    "catalyst_proximity": 0.25,
}

CONFIG_FILE = Path("config.json")


class Settings(BaseSettings):
    """Application settings with sensible defaults."""

    # --- Scoring ---
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_SCORING_WEIGHTS.copy()
    )
    min_composite_score: float = 50.0

    # --- Options filtering ---
    dte_min: int = 30
    dte_max: int = 60
    min_open_interest: int = 100
    max_bid_ask_spread_pct: float = 0.10
    min_option_volume: int = 1

    # --- AI backend ---
    ai_backend: str = "ollama"
    ollama_model: str = "llama3.1:8b"
    claude_api_key: Optional[str] = None

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

    # --- Fetch settings ---
    fetch_max_retries: int = 3
    fetch_retry_delays: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0])
    fetch_batch_size: int = 20
    fetch_max_workers: int = 2
    failure_cache_ttl_hours: int = 24

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
