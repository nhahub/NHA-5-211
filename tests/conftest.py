"""Shared pytest fixtures for SMART IDS v3 test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import AppConfig, DataConfig, PathsConfig, ProjectSettings, SyntheticFallbackConfig
from src.data.loader import generate_synthetic_cicids2017


@pytest.fixture(scope="session")
def small_synthetic_df() -> pd.DataFrame:
    """Small schema-faithful synthetic CICIDS2017 DataFrame with Inf, NaN, and duplicates."""
    return generate_synthetic_cicids2017(
        n_samples=600,
        seed=42,
        duplicate_fraction=0.02,
        inf_fraction=0.03,
        nan_fraction=0.02,
        save_path=None,
    )


@pytest.fixture()
def tmp_app_config(tmp_path: Path) -> AppConfig:
    """Isolated AppConfig pointing to pytest temporary directories."""
    for sub in (
        "data/raw",
        "data/processed",
        "data/synthetic",
        "models",
        "artifacts",
        "reports",
        "reports/figures",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return AppConfig(
        project=ProjectSettings(random_seed=42),
        paths=PathsConfig(
            raw_data_dir=str(tmp_path / "data/raw"),
            processed_data_dir=str(tmp_path / "data/processed"),
            synthetic_data_dir=str(tmp_path / "data/synthetic"),
            models_dir=str(tmp_path / "models"),
            artifacts_dir=str(tmp_path / "artifacts"),
            reports_dir=str(tmp_path / "reports"),
            figures_dir=str(tmp_path / "reports/figures"),
            response_log_file=str(tmp_path / "reports/ids_response_log.txt"),
            incident_history_file=str(tmp_path / "reports/incident_history.json"),
        ),
        data=DataConfig(
            synthetic_fallback=SyntheticFallbackConfig(enabled=True, n_samples=600),
            drop_ultra_rare_classes_below=2,
        ),
    )
