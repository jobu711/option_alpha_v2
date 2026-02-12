"""Tests for configuration system."""

import json
from pathlib import Path

import pytest

from option_alpha.config import Settings, get_settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.min_price == 5.0
        assert s.min_avg_volume == 500_000
        assert s.dte_min == 30
        assert s.dte_max == 60
        assert s.min_open_interest == 100
        assert s.max_bid_ask_spread_pct == 0.30
        assert s.ai_backend == "ollama"
        assert s.ollama_model == "llama3.1:8b"
        assert s.top_n_options == 50
        assert s.top_n_ai_debate == 10
        assert s.risk_free_rate_fallback == 0.05

    def test_scoring_weights_sum(self):
        s = Settings()
        total = sum(s.scoring_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_scoring_weights_has_expected_keys(self):
        s = Settings()
        expected = {
            "bb_width", "atr_percentile", "rsi", "obv_trend",
            "sma_alignment", "relative_volume", "catalyst_proximity",
        }
        assert set(s.scoring_weights.keys()) == expected

    def test_custom_values(self):
        s = Settings(min_price=10.0, top_n_options=100)
        assert s.min_price == 10.0
        assert s.top_n_options == 100

    def test_save_and_load(self, tmp_path):
        config_path = tmp_path / "test_config.json"
        s = Settings(min_price=7.5, ai_backend="claude")
        s.save(config_path)

        assert config_path.exists()

        loaded = Settings.load(config_path)
        assert loaded.min_price == 7.5
        assert loaded.ai_backend == "claude"

    def test_load_nonexistent_returns_defaults(self, tmp_path):
        config_path = tmp_path / "nonexistent.json"
        loaded = Settings.load(config_path)
        assert loaded.min_price == 5.0

    def test_save_creates_valid_json(self, tmp_path):
        config_path = tmp_path / "config.json"
        s = Settings()
        s.save(config_path)

        data = json.loads(config_path.read_text())
        assert "min_price" in data
        assert "scoring_weights" in data

    def test_get_settings_default(self, tmp_path):
        config_path = tmp_path / "no_such_config.json"
        s = get_settings(config_path)
        assert isinstance(s, Settings)

    def test_paths_are_path_objects(self):
        s = Settings()
        assert isinstance(s.data_dir, Path)
        assert isinstance(s.db_path, Path)

    def test_fetch_defaults_for_3k_universe(self):
        """Verify fetch defaults are tuned for ~3,000 ticker universe."""
        s = Settings()
        assert s.fetch_batch_size == 50
        assert s.fetch_max_workers == 4

    def test_data_fetch_period_default(self):
        """Verify data_fetch_period defaults to '1y'."""
        s = Settings()
        assert s.data_fetch_period == "1y"

    def test_direction_rsi_thresholds_defaults(self):
        """Verify RSI direction thresholds have correct defaults."""
        s = Settings()
        assert s.direction_rsi_strong_bullish == 60.0
        assert s.direction_rsi_strong_bearish == 40.0

    def test_direction_rsi_thresholds_customizable(self):
        """Verify RSI direction thresholds can be overridden."""
        s = Settings(
            direction_rsi_strong_bullish=55.0,
            direction_rsi_strong_bearish=45.0,
        )
        assert s.direction_rsi_strong_bullish == 55.0
        assert s.direction_rsi_strong_bearish == 45.0
