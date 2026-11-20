"""Centralized Pydantic-validated configuration loader for SMART IDS v3."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MACHINE_PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/][^\s\"',;)\]}]+|/(?:home|Users)/[^\s\"',;)\]}]+)"
)
_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def to_repo_relative_path(
    path: Path | str | None,
    root: Path | PureWindowsPath | PurePosixPath | str = PROJECT_ROOT,
) -> str | None:
    """Convert an absolute filesystem path into a repo-relative POSIX path or basename."""
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None

    root_str = str(root).strip()
    norm_root = root_str.replace("\\", "/").rstrip("/")
    foreign_win_root = _WINDOWS_ABS_RE.match(root_str) and os.name != "nt"
    foreign_posix_root = norm_root.startswith("/") and os.name == "nt"
    if isinstance(root, PureWindowsPath) and os.name != "nt":
        foreign_win_root = True
    if foreign_win_root or foreign_posix_root:
        root_posix = norm_root
        root_resolved: Path | None = None
    else:
        root_resolved = Path(root_str).resolve()
        root_posix = root_resolved.as_posix().rstrip("/")

    norm_text = text.replace("\\", "/")
    if norm_text.lower() == root_posix.lower():
        return "."
    if norm_text.lower().startswith(root_posix.lower() + "/"):
        return norm_text[len(root_posix) + 1 :]

    if _WINDOWS_ABS_RE.match(text):
        return PureWindowsPath(text).name
    if norm_text.startswith(("/home/", "/Users/")):
        return PurePosixPath(norm_text).name

    candidate = Path(text)
    if candidate.is_absolute() and root_resolved is not None:
        try:
            return candidate.resolve().relative_to(root_resolved).as_posix()
        except ValueError:
            return candidate.name
    return norm_text[2:] if norm_text.startswith("./") else norm_text


def sanitize_machine_paths(
    obj: Any,
    root: Path | PureWindowsPath | PurePosixPath | str = PROJECT_ROOT,
) -> Any:
    """Recursively replace absolute Windows/home paths inside serialized artifacts."""
    if isinstance(obj, Path):
        return to_repo_relative_path(obj, root=root)
    if isinstance(obj, str):
        if _WINDOWS_ABS_RE.match(obj.strip()) or obj.strip().replace("\\", "/").startswith(
            ("/home/", "/Users/")
        ):
            return to_repo_relative_path(obj, root=root)
        if _MACHINE_PATH_TOKEN_RE.search(obj):
            return _MACHINE_PATH_TOKEN_RE.sub(
                lambda m: to_repo_relative_path(m.group(0), root=root) or m.group(0),
                obj,
            )
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_machine_paths(v, root=root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_machine_paths(v, root=root) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_machine_paths(v, root=root) for v in obj]
    return obj


class ProjectSettings(BaseModel):
    """Top-level project metadata."""

    name: str = "SMART IDS v3"
    subtitle: str = "Multi-Model Intrusion Detection with Zero-Day Anomaly Tracking"
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
    """Train/validation/test split ratios."""

    test_size: float = Field(default=0.15, gt=0.0, lt=0.5)
    val_size_of_remainder: float = Field(default=0.15, gt=0.0, lt=0.5)
    thursday_val_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    time_block_train_fraction: float = Field(default=0.60, gt=0.0, lt=1.0)
    time_block_val_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    stratify: bool = True

    @model_validator(mode="after")
    def _validate_time_block_fractions(self) -> SplitConfig:
        if self.time_block_train_fraction + self.time_block_val_fraction >= 1.0:
            raise ValueError(
                "time_block_train_fraction + time_block_val_fraction must be strictly less than 1.0 "
                "so chronological test blocks receive positive support."
            )
        return self


class ImbalanceConfig(BaseModel):
    """Class imbalance remediation configuration."""

    strategy: str = "auto_compare"
    imbalance_tie_tolerance: float = Field(default=0.002, ge=0.0, le=0.2)
    smote_sampling_strategy: str = "auto"
    smote_max_k_neighbors: int = Field(default=5, ge=1)
    smote_cap_per_class: int | None = 5000
    smote_max_multiplier: float | None = Field(default=5.0, gt=1.0)
    smote_min_real_samples: int = Field(default=50, ge=2)


class DataConfig(BaseModel):
    """Dataset ingestion and cleaning configuration."""

    dataset_name: str = "CICIDS2017"
    expected_raw_files: list[str] = Field(default_factory=list)
    sample_size: int | None = None
    use_full_dataset: bool = False
    max_rows_sample: int | None = 200000
    label_correction_file: str | None = None
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
    drop_features: list[str] = Field(default_factory=list)
    strict_leakage_check: bool = True
    split_strategy: Literal["random", "by_day", "time_block"] = "random"
    timestamp_source_dir: str | None = None
    split: SplitConfig = Field(default_factory=SplitConfig)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)

    @model_validator(mode="after")
    def _sync_sample_limits(self) -> DataConfig:
        if self.use_full_dataset:
            self.sample_size = None
            self.max_rows_sample = None
        elif self.sample_size is not None:
            if self.max_rows_sample == 200000 or self.max_rows_sample is None:
                self.max_rows_sample = self.sample_size
            elif self.sample_size == 200000 and self.max_rows_sample != 200000:
                self.sample_size = self.max_rows_sample
        else:
            self.sample_size = self.max_rows_sample
        return self

    @property
    def effective_sample_size(self) -> int | None:
        if self.use_full_dataset:
            return None
        if self.sample_size is not None and (self.max_rows_sample is None or self.sample_size != 200000):
            return int(self.sample_size)
        if self.max_rows_sample is not None:
            return int(self.max_rows_sample)
        return int(self.sample_size) if self.sample_size is not None else None


class EvaluationConfig(BaseModel):
    """Evaluation, benchmarking, and SHAP configuration."""

    latency_warmup_calls: int = Field(default=50, ge=1)
    latency_timed_runs: int = Field(default=100, ge=5)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_warmup_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "latency_warmup_calls" not in data and "latency_warmup_runs" in data:
            copied = dict(data)
            copied["latency_warmup_calls"] = copied.pop("latency_warmup_runs")
            return copied
        return data

    batch_benchmark_size: int = Field(default=1024, ge=16)
    shap_xgb_max_samples: int = Field(default=500, ge=10)
    shap_mlp_background_samples: int = Field(default=50, ge=5)
    shap_mlp_explain_samples: int = Field(default=20, ge=1)
    autoencoder_percentiles: list[float] = Field(default_factory=lambda: [90.0, 95.0, 97.0, 99.0])
    unseen_attack_holdout_class: str = "Web Attack"
    run_by_day_comparison: bool = True
    by_day_keras_epochs: int = Field(default=20, ge=1)
    run_time_block_comparison: bool = True
    time_block_keras_epochs: int = Field(default=20, ge=1)
    run_feature_ablation: bool = True
    ablation_drop_features: list[str] = Field(default_factory=lambda: ["Destination Port"])
    bootstrap_n_resamples: int = Field(default=1000, ge=100)
    low_support_threshold: int = Field(default=30, ge=1)
    reliable_support_threshold: int | None = Field(default=None, ge=1)
    run_mitigation_analysis: bool = True
    failing_class_recall_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    drift_top_k_features: int = Field(default=5, ge=1)
    mitigation_benign_percentiles: list[float] = Field(default_factory=lambda: [90.0, 95.0, 97.0, 99.0])
    run_sample_size_sensitivity: bool = True
    sensitivity_sample_sizes: list[int] = Field(default_factory=lambda: [200000, 500000])
    run_xgboost_cv: bool = True
    cv_n_splits: int = Field(default=5, ge=2)

    @property
    def effective_reliable_support_threshold(self) -> int:
        return int(
            self.reliable_support_threshold
            if self.reliable_support_threshold is not None
            else self.low_support_threshold
        )


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


class ImbalanceSelectionConfig(BaseModel):
    """Per-model imbalance strategy selection controls."""

    per_model: bool = True
    neural_epochs_cap: int = Field(default=10, ge=1)
    tie_tolerance: float | None = Field(default=None, ge=0.0, le=0.2)


class DeploymentConfig(BaseModel):
    """Production deployment profile controls for multi-profile artifacts."""

    xgboost_imbalance_profile: Literal["class_weight", "none"] = "class_weight"


class SecurityConfig(BaseModel):
    """Model artifact integrity verification controls."""

    verify_model_hashes: bool = True
    require_manifest: bool = False
    manifest_file: str = "MANIFEST.json"


class ModelsDownloadConfig(BaseModel):
    """LFS-free fallback asset download settings (GitHub Releases / HTTP mirror)."""

    release_base_url: str = ""
    release_repo: str = ""
    release_tag: str = "models-v3"
    required_assets: list[str] = Field(
        default_factory=lambda: [
            "xgb_model.json",
            "scaler.pkl",
            "imputer.pkl",
            "outlier_clipper.pkl",
            "label_encoder.pkl",
            "feature_names.json",
            "anomaly_threshold.json",
            "best_mlp.keras",
            "best_cnn.keras",
            "best_autoencoder.keras",
        ]
    )
    optional_assets: list[str] = Field(
        default_factory=lambda: [
            "MANIFEST.json",
            "model_metadata.json",
            "training_histories.json",
            "xgb_baseline.json",
            "xgb_model_none.json",
            "xgb_model_class_weight.json",
            "calibrators.pkl",
            "stacking_meta.pkl",
            "anomaly_baselines.pkl",
            "anomaly_fusion.pkl",
            "best_autoencoder_deep.keras",
            "best_vae.keras",
        ]
    )

    def resolved_base_url(self) -> str:
        """Return configured release base URL or construct one from owner/repo + tag."""
        if self.release_base_url.strip():
            return self.release_base_url.strip().rstrip("/")
        if self.release_repo.strip():
            repo = self.release_repo.strip().strip("/")
            tag = self.release_tag.strip() or "models-v3"
            return f"https://github.com/{repo}/releases/download/{tag}"
        return ""


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
    imbalance_selection: ImbalanceSelectionConfig = Field(default_factory=ImbalanceSelectionConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    models_download: ModelsDownloadConfig = Field(default_factory=ModelsDownloadConfig)
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
    """Unsupervised Autoencoder and enhanced benign-only anomaly variant hyperparameters."""

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
    try_variants: bool = True
    variant_epochs_cap: int = Field(default=20, ge=1)
    deep_reg_encoder_units: list[int] = Field(default_factory=lambda: [512, 256, 128, 64])
    deep_reg_latent_dim: int = 24
    deep_reg_l2: float = 1e-4
    deep_reg_dropout: float = 0.2
    deep_reg_noise_std: float = 0.05
    deep_reg_checkpoint_name: str = "best_autoencoder_deep.keras"
    vae_encoder_units: list[int] = Field(default_factory=lambda: [256, 128, 64])
    vae_latent_dim: int = 16
    vae_kl_weight: float = 0.01
    vae_l2: float = 1e-4
    vae_checkpoint_name: str = "best_vae.keras"
    fusion_enabled: bool = True
    fusion_file: str = "anomaly_fusion.pkl"
    fusion_grid_step: float = Field(default=0.2, gt=0.0, le=0.5)


class AnomalyBaselinesConfig(BaseModel):
    """Configuration for classical benign-only unsupervised anomaly baselines."""

    enabled: bool = True
    isolation_forest_n_estimators: int = Field(default=200, ge=10)
    isolation_forest_max_samples: int = Field(default=512, ge=16)
    ocsvm_nu: float = Field(default=0.05, gt=0.0, lt=1.0)
    ocsvm_gamma: str | float = "scale"
    ocsvm_max_train_samples: int = Field(default=15000, ge=100)
    lof_n_neighbors: int = Field(default=20, ge=2)
    lof_max_train_samples: int = Field(default=15000, ge=100)
    baselines_file: str = "anomaly_baselines.pkl"


class NeuralExperimentConfig(BaseModel):
    """Bounded neural-baseline improvement experiments (loss variants & extra time cap)."""

    try_loss_variants: bool = True
    max_training_minutes: float = Field(default=3.0, gt=0.0)
    label_smoothing: float = Field(default=0.05, ge=0.0, lt=0.5)
    focal_gamma: float = Field(default=2.0, gt=0.0)
    improvement_tolerance: float = Field(default=0.002, ge=0.0)
    epochs_cap: int = Field(default=15, ge=1)


class XGBoostOptunaConfig(BaseModel):
    """Optuna tuning configuration for XGBoost."""

    enabled: bool = True
    n_trials: int = Field(default=40, ge=0)
    timeout_minutes: float | None = Field(default=6.0, gt=0.0)
    timeout_seconds: int = Field(default=360, ge=1)
    sample_cap_for_tuning: int = 100000

    @model_validator(mode="after")
    def _sync_optuna_timeout(self) -> XGBoostOptunaConfig:
        if "timeout_seconds" in self.model_fields_set and "timeout_minutes" not in self.model_fields_set:
            self.timeout_minutes = round(float(self.timeout_seconds) / 60.0, 4)
        elif self.timeout_minutes is not None:
            self.timeout_seconds = max(1, int(round(float(self.timeout_minutes) * 60.0)))
        return self


class XGBoostConfig(BaseModel):
    """XGBoost baseline and HPO configuration."""

    baseline: dict[str, Any] = Field(default_factory=dict)
    optuna: XGBoostOptunaConfig = Field(default_factory=XGBoostOptunaConfig)
    model_name: str = "xgb_model.json"
    baseline_model_name: str = "xgb_baseline.json"


class EnsembleConfig(BaseModel):
    """Hybrid ensemble decision strategy configuration."""

    strategy: str = "hybrid_anomaly_override"
    base_supervised_strategy: str = "best_single"
    supervised_weights: dict[str, float] = Field(
        default_factory=lambda: {"XGBoost": 0.45, "MLP": 0.35, "CNN": 0.20}
    )
    anomaly_override_min_confidence: float = 0.65
    calibrate_probabilities: bool = True
    stacker_file: str = "stacking_meta.pkl"


class ModelConfig(BaseModel):
    """Root model and training configuration."""

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    mlp: MLPConfig = Field(default_factory=MLPConfig)
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    neural: NeuralExperimentConfig = Field(default_factory=NeuralExperimentConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    anomaly_baselines: AnomalyBaselinesConfig = Field(default_factory=AnomalyBaselinesConfig)

    @model_validator(mode="before")
    @classmethod
    def _merge_top_level_optuna(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("optuna"), dict):
            copied = dict(data)
            top_opt = dict(copied.pop("optuna"))
            xgb_raw = dict(copied.get("xgboost") or {})
            merged_opt = {**dict(xgb_raw.get("optuna") or {}), **top_opt}
            xgb_raw["optuna"] = merged_opt
            copied["xgboost"] = xgb_raw
            return copied
        return data

    @property
    def optuna(self) -> XGBoostOptunaConfig:
        return self.xgboost.optuna


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
