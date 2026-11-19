"""Model and preprocessing artifact persistence registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import joblib

from src.utils.logging import get_logger

logger = get_logger("smart_ids.registry")

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_git_lfs_pointer(path: Path) -> bool:
    """Return True when ``path`` is an unfetched Git LFS pointer stub instead of raw data."""
    try:
        if not path.is_file() or path.stat().st_size > 1024:
            return False
        with path.open("rb") as fh:
            head = fh.read(len(_LFS_POINTER_PREFIX))
        return head.startswith(_LFS_POINTER_PREFIX)
    except OSError:
        return False


@dataclass
class LoadedArtifacts:
    """Container holding all saved models and preprocessing transformers."""

    models: dict[str, Any]
    scaler: Any | None
    imputer: Any | None
    outlier_clipper: Any | None
    label_encoder: Any | None
    feature_names: list[str]
    anomaly_threshold: float
    anomaly_threshold_meta: dict[str, Any]
    manifest: dict[str, Any]
    calibrators: dict[str, Any] | None = None
    stacker: Any | None = None
    anomaly_baselines: dict[str, Any] | None = None
    anomaly_fusion: Any | None = None
    model_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_imbalance_strategies: dict[str, str] = field(default_factory=dict)
    lfs_pointers: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)


class ModelRegistry:
    """Save, load, and inspect SMART IDS v3 models and preprocessing artifacts."""

    def __init__(self, models_dir: Path, artifacts_dir: Path | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else self.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_preprocessing_artifacts(
        self,
        scaler: Any,
        label_encoder: Any,
        feature_names: list[str],
        imputer: Any | None = None,
        outlier_clipper: Any | None = None,
    ) -> None:
        """Persist fitted preprocessing transformers and schema metadata."""
        joblib.dump(scaler, self.models_dir / "scaler.pkl")
        joblib.dump(label_encoder, self.models_dir / "label_encoder.pkl")
        if imputer is not None:
            joblib.dump(imputer, self.models_dir / "imputer.pkl")
        if outlier_clipper is not None:
            joblib.dump(outlier_clipper, self.models_dir / "outlier_clipper.pkl")

        with (self.models_dir / "feature_names.json").open("w", encoding="utf-8") as fh:
            json.dump(list(feature_names), fh, indent=2)
        logger.info("Saved preprocessing artifacts (%d features) to %s", len(feature_names), self.models_dir)

    def save_anomaly_threshold(
        self,
        threshold: float,
        percentile: float,
        benign_errors_summary: dict[str, float] | None = None,
    ) -> Path:
        """Persist calibrated autoencoder reconstruction threshold configuration."""
        payload: dict[str, Any] = {
            "threshold": float(threshold),
            "percentile": float(percentile),
            "calibration_method": f"{percentile:.1f}th percentile of benign validation MSE",
            "saved_at_utc": datetime.now(UTC).isoformat(),
        }
        if benign_errors_summary:
            payload["benign_validation_mse_stats"] = benign_errors_summary
        out_path = self.models_dir / "anomaly_threshold.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Saved anomaly threshold (%.6f) to %s", threshold, out_path)
        return out_path

    def save_manifest(self, metadata: dict[str, Any]) -> Path:
        """Persist run manifest tracking dataset provenance, versions, and metrics."""
        payload = {
            "saved_at_utc": datetime.now(UTC).isoformat(),
            **metadata,
        }
        out_path = self.artifacts_dir / "run_manifest.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return out_path

    def save_model_metadata(self, model_metadata: dict[str, dict[str, Any]]) -> Path:
        """Persist per-model training metadata (e.g., imbalance strategy) alongside models."""
        out_path = self.models_dir / "model_metadata.json"
        payload = {
            "saved_at_utc": datetime.now(UTC).isoformat(),
            "models": model_metadata,
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Saved per-model metadata to %s", out_path)
        return out_path

    def load_all(self, load_keras: bool = True) -> LoadedArtifacts:
        """Load all available model and preprocessing artifacts defensively."""
        models: dict[str, Any] = {}
        lfs_pointers: list[str] = []
        missing_artifacts: list[str] = []

        keras_targets = {
            "MLP": self.models_dir / "best_mlp.keras",
            "CNN": self.models_dir / "best_cnn.keras",
            "Autoencoder": self.models_dir / "best_autoencoder.keras",
        }
        optional_keras_targets = {
            "Autoencoder (Deep Reg)": self.models_dir / "best_autoencoder_deep.keras",
            "VAE": self.models_dir / "best_vae.keras",
        }
        if load_keras:
            try:
                import tensorflow as tf

                import src.models.autoencoder  # noqa: F401 - register custom VAESampling layer

                for name, path in {**keras_targets, **optional_keras_targets}.items():
                    is_optional = name in optional_keras_targets
                    if not path.exists():
                        if not is_optional:
                            missing_artifacts.append(path.name)
                    elif is_git_lfs_pointer(path):
                        lfs_pointers.append(path.name)
                        logger.warning(
                            "%s is a Git LFS pointer stub; run 'git lfs pull' or train locally.",
                            path.name,
                        )
                    else:
                        try:
                            models[name] = tf.keras.models.load_model(path, compile=False)
                        except Exception as exc:
                            logger.warning("Failed loading Keras model '%s': %s", path.name, exc)
            except Exception as exc:
                logger.warning("Failed initializing TensorFlow for Keras model loading: %s", exc)

        xgb_path = self.models_dir / "xgb_model.json"
        if not xgb_path.exists():
            missing_artifacts.append(xgb_path.name)
        elif is_git_lfs_pointer(xgb_path):
            lfs_pointers.append(xgb_path.name)
            logger.warning("xgb_model.json is a Git LFS pointer stub; run 'git lfs pull' or train locally.")
        else:
            try:
                import xgboost as xgb

                clf = xgb.XGBClassifier()
                clf.load_model(str(xgb_path))
                models["XGBoost"] = clf
            except Exception as exc:
                logger.warning("Failed loading XGBoost model: %s", exc)

        def _maybe_joblib(fname: str, optional: bool = False) -> Any | None:
            p = self.models_dir / fname
            if not p.exists():
                if not optional:
                    missing_artifacts.append(fname)
                return None
            if is_git_lfs_pointer(p):
                lfs_pointers.append(fname)
                logger.warning("%s is a Git LFS pointer stub; run 'git lfs pull' to fetch binary.", fname)
                return None
            try:
                return joblib.load(p)
            except Exception as exc:
                logger.warning("Failed loading joblib artifact '%s': %s", fname, exc)
                return None

        scaler = _maybe_joblib("scaler.pkl")
        imputer = _maybe_joblib("imputer.pkl")
        outlier_clipper = _maybe_joblib("outlier_clipper.pkl")
        label_encoder = _maybe_joblib("label_encoder.pkl")
        calibrators = _maybe_joblib("calibrators.pkl")
        stacker = _maybe_joblib("stacking_meta.pkl")
        anomaly_baselines = _maybe_joblib("anomaly_baselines.pkl")
        anomaly_fusion = _maybe_joblib("anomaly_fusion.pkl", optional=True)

        feat_path = self.models_dir / "feature_names.json"
        feature_names: list[str] = []
        if feat_path.exists():
            with feat_path.open("r", encoding="utf-8") as fh:
                feature_names = json.load(fh)

        thresh_path = self.models_dir / "anomaly_threshold.json"
        anomaly_threshold = 1.0
        anomaly_threshold_meta: dict[str, Any] = {}
        if thresh_path.exists():
            with thresh_path.open("r", encoding="utf-8") as fh:
                anomaly_threshold_meta = json.load(fh)
                anomaly_threshold = float(anomaly_threshold_meta.get("threshold", 1.0))

        manifest_path = self.artifacts_dir / "run_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)

        model_metadata: dict[str, dict[str, Any]] = {}
        meta_path = self.models_dir / "model_metadata.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                raw_meta = json.load(fh)
            model_metadata = dict(raw_meta.get("models", {}) or {})
        elif isinstance(manifest.get("model_metadata"), dict):
            model_metadata = dict(manifest["model_metadata"])

        model_imbalance_strategies: dict[str, str] = {}
        if isinstance(manifest.get("imbalance_strategies_by_model"), dict):
            model_imbalance_strategies.update(
                {str(k): str(v) for k, v in manifest["imbalance_strategies_by_model"].items()}
            )
        for m_key, m_info in model_metadata.items():
            if isinstance(m_info, dict) and "imbalance_strategy" in m_info:
                model_imbalance_strategies.setdefault(str(m_key), str(m_info["imbalance_strategy"]))

        return LoadedArtifacts(
            models=models,
            scaler=scaler,
            imputer=imputer,
            outlier_clipper=outlier_clipper,
            label_encoder=label_encoder,
            feature_names=feature_names,
            anomaly_threshold=anomaly_threshold,
            anomaly_threshold_meta=anomaly_threshold_meta,
            manifest=manifest,
            calibrators=calibrators,
            stacker=stacker,
            anomaly_baselines=anomaly_baselines,
            anomaly_fusion=anomaly_fusion,
            model_metadata=model_metadata,
            model_imbalance_strategies=model_imbalance_strategies,
            lfs_pointers=sorted(set(lfs_pointers)),
            missing_artifacts=sorted(set(missing_artifacts)),
        )
