"""Centralized Pydantic-validated configuration loader for SMART IDS v3."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProjectSettings(BaseModel):
    """Top-level project metadata."""

    name: str = "SMART IDS v3"
    subtitle: str = (
        "An Intelligent, Multi-Modal Hybrid Ensemble Architecture for "
        "Network Intrusion Detection and Zero-Day Anomaly Tracking"
    )
    version: str = "3.0.0"
    random_seed: int = 42


class PathsConfig(BaseModel):
    """Filesystem path configuration relative to project root."""

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    synthetic_data_dir: str = "data/synthetic"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    response_log_file: str = "reports/ids_response_log.txt"
    incident_history_file: str = "reports/incident_history.json"

    def resolve(self, attr: str, root: Path = PROJECT_ROOT) -> Path:
        """Resolve a configured relative path against the project root."""
        raw = getattr(self, attr)
        path = Path(raw)
        return path if path.is_absolute() else (root / path).resolve()


class SyntheticFallbackConfig(BaseModel):
    """Schema-faithful synthetic demonstration configuration."""

    enabled: bool = True
    n_samples: int = Field(default=60000, ge=200)
    duplicate_fraction: float = Field(default=0.004, ge=0.0, le=0.2)
    inf_fraction: float = Field(default=0.008, ge=0.0, le=0.2)
    nan_fraction: float = Field(default=0.003, ge=0.0, le=0.2)
    class_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "BENIGN": 0.83,
            "DoS Hulk": 0.06,
            "PortScan": 0.04,
            "DDoS": 0.03,
            "DoS GoldenEye": 0.01,
            "FTP-Patator": 0.01,
            "SSH-Patator": 0.01,
            "Web Attack": 0.01,
        }
    )


class SplitConfig(BaseModel):
    """Stratified train/validation/test split ratios."""

    test_size: float = Field(default=0.15, gt=0.0, lt=0.5)
    val_size_of_remainder: float = Field(default=0.15, gt=0.0, lt=0.5)
    stratify: bool = True


class ImbalanceConfig(BaseModel):
    """Class imbalance remediation configuration."""

    strategy: str = "auto_compare"
    smote_sampling_strategy: str = "auto"
    smote_max_k_neighbors: int = Field(default=5, ge=1)
    smote_cap_per_class: int | None = None


class DataConfig(BaseModel):
    """Dataset ingestion and cleaning configuration."""

    dataset_name: str = "CICIDS2017"
    expected_raw_files: list[str] = Field(default_factory=list)
    max_rows_sample: int | None = 200000
    synthetic_fallback: SyntheticFallbackConfig = Field(default_factory=SyntheticFallbackConfig)
    label_column: str = "Label"
    canonical_classes: list[str] = Field(default_factory=list)
    merge_web_attacks: bool = True
    drop_ultra_rare_classes_below: int = Field(default=10, ge=1)
    nan_column_drop_threshold: float = Field(default=0.50, gt=0.0, le=1.0)
    outlier_clip_sigma: float = Field(default=5.0, gt=0.0)
    identifier_keywords: list[str] = Field(
        default_factory=lambda: ["ip", "timestamp", "flow id", "src", "dst", "source port"]
    )
    preserve_features: list[str] = Field(default_factory=lambda: ["Destination Port"])
    strict_leakage_check: bool = True
    split: SplitConfig = Field(default_factory=SplitConfig)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)


class EvaluationConfig(BaseModel):
    """Evaluation, benchmarking, and SHAP configuration."""

    latency_warmup_runs: int = Field(default=5, ge=1)
    latency_timed_runs: int = Field(default=100, ge=5)
    batch_benchmark_size: int = Field(default=1024, ge=16)
    shap_xgb_max_samples: int = Field(default=500, ge=10)
    shap_mlp_background_samples: int = Field(default=50, ge=5)
    shap_mlp_explain_samples: int = Field(default=20, ge=1)
    autoencoder_percentiles: list[float] = Field(default_factory=lambda: [90.0, 95.0, 97.0, 99.0])
    unseen_attack_holdout_class: str = "Web Attack"


class ResponseConfig(BaseModel):
    """Active response taxonomy and dry-run configuration."""

    dry_run: bool = True
    rate_limit_pps: int = Field(default=100, ge=1)
    threat_thresholds: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "Low": (0.0, 0.40),
            "Medium": (0.40, 0.70),
            "High": (0.70, 0.90),
            "Critical": (0.90, 1.01),
        }
    )
    threat_colors: dict[str, str] = Field(
        default_factory=lambda: {
            "Low": "#22c55e",
            "Medium": "#eab308",
            "High": "#f97316",
            "Critical": "#ef4444",
        }
    )
    threat_icons: dict[str, str] = Field(
        default_factory=lambda: {
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🟠",
            "Critical": "🔴",
        }
    )


class DashboardConfig(BaseModel):
    """Streamlit UI runtime settings."""

    page_title: str = "SMART IDS v3 — Hybrid SOC Analytics"
    page_icon: str = "🛡️"
    layout: str = "wide"
    default_model: str = "Hybrid Ensemble"
    default_confidence_threshold: float = 0.40


class AppConfig(BaseModel):
    """Root application configuration."""

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


class HardwareConfig(BaseModel):
    """Hardware and GPU acceleration preferences."""

    use_gpu: str | bool = "auto"
    mixed_precision: bool = False
    memory_growth: bool = True


class MLPConfig(BaseModel):
    """MLP model architecture and training hyperparameters."""

    hidden_units: list[int] = Field(default_factory=lambda: [512, 256, 128, 64])
    l2_reg: float = 1e-4
    leaky_relu_slope: float = 0.1
    dropout_rate: float = 0.4
    learning_rate: float = 1e-3
    clipnorm: float = 1.0
    batch_size: int = 512
    epochs: int = 50
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    checkpoint_name: str = "best_mlp.keras"


class CNNConfig(BaseModel):
    """1D-CNN model architecture and training hyperparameters."""

    block1_filters: int = 64
    block2_filters: int = 128
    kernel_size: int = 3
    pool_size: int = 2
    l2_reg: float = 1e-4
    dropout_block1: float = 0.3
    dropout_block2: float = 0.4
    dense_units: int = 128
    dropout_head: float = 0.3
    learning_rate: float = 1e-3
    clipnorm: float = 1.0
    batch_size: int = 512
    epochs: int = 50
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    checkpoint_name: str = "best_cnn.keras"


class AutoencoderConfig(BaseModel):
    """Unsupervised Autoencoder hyperparameters."""

    encoder_units: list[int] = Field(default_factory=lambda: [256, 128, 64])
    latent_dim: int = 32
    decoder_units: list[int] = Field(default_factory=lambda: [64, 128, 256])
    learning_rate: float = 5e-4
    batch_size: int = 512
    epochs: int = 50
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    threshold_percentile: float = 95.0
    checkpoint_name: str = "best_autoencoder.keras"
    threshold_file: str = "anomaly_threshold.json"


class XGBoostOptunaConfig(BaseModel):
    """Optuna tuning configuration for XGBoost."""

    enabled: bool = True
    n_trials: int = 12
    timeout_seconds: int = 180
    sample_cap_for_tuning: int = 100000


class XGBoostConfig(BaseModel):
    """XGBoost baseline and HPO configuration."""

    baseline: dict[str, Any] = Field(default_factory=dict)
    optuna: XGBoostOptunaConfig = Field(default_factory=XGBoostOptunaConfig)
    model_name: str = "xgb_model.json"
    baseline_model_name: str = "xgb_baseline.json"


class EnsembleConfig(BaseModel):
    """Hybrid ensemble decision strategy configuration."""

    strategy: str = "hybrid_anomaly_override"
    supervised_weights: dict[str, float] = Field(
        default_factory=lambda: {"XGBoost": 0.45, "MLP": 0.35, "CNN": 0.20}
    )
    anomaly_override_min_confidence: float = 0.65
    calibrate_probabilities: bool = True


class ModelConfig(BaseModel):
    """Root model and training configuration."""

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    mlp: MLPConfig = Field(default_factory=MLPConfig)
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def get_app_config(config_path: str | Path | None = None) -> AppConfig:
    """Load and validate application configuration."""
    target = Path(config_path) if config_path else (PROJECT_ROOT / "configs" / "config.yaml")
    raw = _read_yaml(target)
    cfg = AppConfig(**raw)

    # Allow environment variable overrides without mutating YAML files
    env_seed = os.getenv("SMART_IDS_SEED")
    if env_seed and env_seed.isdigit():
        cfg.project.random_seed = int(env_seed)
    env_dry_run = os.getenv("SMART_IDS_DRY_RUN")
    if env_dry_run is not None:
        cfg.response.dry_run = env_dry_run.strip().lower() in {"1", "true", "yes"}
    return cfg


@lru_cache(maxsize=1)
def get_model_config(config_path: str | Path | None = None) -> ModelConfig:
    """Load and validate model configuration."""
    target = Path(config_path) if config_path else (PROJECT_ROOT / "configs" / "model_config.yaml")
    raw = _read_yaml(target)
    return ModelConfig(**raw)
