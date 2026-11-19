"""Model and preprocessing artifact persistence registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from src.utils.logging import get_logger

logger = get_logger("smart_ids.registry")

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_MANIFEST_TRACKED_SUFFIXES = {".pkl", ".keras", ".json", ".h5", ".onnx"}


class ModelIntegrityError(RuntimeError):
    """Raised when artifact SHA-256 verification fails or manifest checks refuse loading."""


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute lowercase hexadecimal SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
    manifest_sha256: dict[str, Any] = field(default_factory=dict)
    integrity_warnings: list[str] = field(default_factory=list)


class ModelRegistry:
    """Save, load, and inspect SMART IDS v3 models and preprocessing artifacts."""

    def __init__(
        self,
        models_dir: Path,
        artifacts_dir: Path | None = None,
        verify_model_hashes: bool | None = None,
        require_manifest: bool | None = None,
        manifest_filename: str | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else self.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        cfg_verify = True
        cfg_require_manifest = False
        cfg_manifest_name = "MANIFEST.json"
        if verify_model_hashes is None or require_manifest is None or manifest_filename is None:
            try:
                from src.config import get_app_config

                sec = get_app_config().security
                cfg_verify = bool(sec.verify_model_hashes)
                cfg_require_manifest = bool(sec.require_manifest)
                cfg_manifest_name = str(sec.manifest_file or "MANIFEST.json")
            except Exception:
                pass

        self.verify_model_hashes = cfg_verify if verify_model_hashes is None else bool(verify_model_hashes)
        self.require_manifest = cfg_require_manifest if require_manifest is None else bool(require_manifest)
        self.manifest_filename = cfg_manifest_name if manifest_filename is None else str(manifest_filename)

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

    def save_artifact_manifest(self) -> Path:
        """Generate ``models/MANIFEST.json`` with SHA-256 digests for all binary/config artifacts."""
        manifest_path = self.models_dir / self.manifest_filename
        artifacts: dict[str, dict[str, Any]] = {}
        for path in sorted(self.models_dir.iterdir(), key=lambda p: p.name):
            if not path.is_file() or path.name == self.manifest_filename:
                continue
            if path.suffix.lower() not in _MANIFEST_TRACKED_SUFFIXES:
                continue
            if is_git_lfs_pointer(path):
                logger.info("Skipping Git LFS pointer stub from MANIFEST.json: %s", path.name)
                continue
            artifacts[path.name] = {
                "sha256": compute_file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }

        payload: dict[str, Any] = {
            "algorithm": "sha256",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Saved SHA-256 MANIFEST (%d artifacts) to %s", len(artifacts), manifest_path)
        return manifest_path

    def _load_sha256_manifest(self) -> dict[str, Any] | None:
        manifest_path = self.models_dir / self.manifest_filename
        if not manifest_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise ModelIntegrityError(
                f"Failed parsing model integrity manifest '{manifest_path}': {exc}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), dict):
            raise ModelIntegrityError(f"Invalid model integrity manifest structure in '{manifest_path}'.")
        return payload

    def verify_artifact_file(
        self,
        path: Path,
        manifest_payload: dict[str, Any] | None = None,
        integrity_warnings: list[str] | None = None,
        require_entry_when_manifest_present: bool = True,
    ) -> None:
        """Verify SHA-256 digest of ``path`` against ``MANIFEST.json`` before loading."""
        if not self.verify_model_hashes or not path.exists():
            return
        if is_git_lfs_pointer(path):
            return

        payload = manifest_payload if manifest_payload is not None else self._load_sha256_manifest()
        manifest_path = self.models_dir / self.manifest_filename
        if payload is None:
            msg = f"Model integrity manifest '{manifest_path}' is missing while loading '{path.name}'."
            if self.require_manifest:
                logger.error(msg)
                raise ModelIntegrityError(msg)
            logger.warning("%s Proceeding unverified because security.require_manifest=false.", msg)
            if integrity_warnings is not None and msg not in integrity_warnings:
                integrity_warnings.append(msg)
            return

        artifacts = payload.get("artifacts", {})
        if path.name not in artifacts:
            if not require_entry_when_manifest_present:
                return
            msg = f"Artifact '{path.name}' is not registered in '{manifest_path}'; refusing unverified load."
            logger.error(msg)
            raise ModelIntegrityError(msg)

        expected = str(artifacts[path.name].get("sha256", "")).strip().lower()
        if not expected:
            msg = f"Manifest entry for '{path.name}' has no sha256 digest in '{manifest_path}'."
            logger.error(msg)
            raise ModelIntegrityError(msg)

        actual = compute_file_sha256(path).lower()
        if actual != expected:
            msg = (
                f"SHA-256 integrity mismatch for '{path.name}': expected {expected}, got {actual}. "
                "Refusing to deserialize artifact."
            )
            logger.error(msg)
            raise ModelIntegrityError(msg)

    def load_all(self, load_keras: bool = True) -> LoadedArtifacts:
        """Load all available model and preprocessing artifacts defensively."""
        models: dict[str, Any] = {}
        lfs_pointers: list[str] = []
        missing_artifacts: list[str] = []
        integrity_warnings: list[str] = []
        manifest_sha256 = self._load_sha256_manifest() or {}

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
                import src.models.losses  # noqa: F401 - register custom sparse losses

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
                        self.verify_artifact_file(
                            path,
                            manifest_payload=manifest_sha256 if manifest_sha256 else None,
                            integrity_warnings=integrity_warnings,
                        )
                        try:
                            models[name] = tf.keras.models.load_model(path, compile=False)
                        except ModelIntegrityError:
                            raise
                        except Exception as exc:
                            logger.warning("Failed loading Keras model '%s': %s", path.name, exc)
            except ModelIntegrityError:
                raise
            except Exception as exc:
                logger.warning("Failed initializing TensorFlow for Keras model loading: %s", exc)

        xgb_path = self.models_dir / "xgb_model.json"
        if not xgb_path.exists():
            missing_artifacts.append(xgb_path.name)
        elif is_git_lfs_pointer(xgb_path):
            lfs_pointers.append(xgb_path.name)
            logger.warning("xgb_model.json is a Git LFS pointer stub; run 'git lfs pull' or train locally.")
        else:
            self.verify_artifact_file(
                xgb_path,
                manifest_payload=manifest_sha256 if manifest_sha256 else None,
                integrity_warnings=integrity_warnings,
            )
            try:
                import xgboost as xgb

                clf = xgb.XGBClassifier()
                clf.load_model(str(xgb_path))
                models["XGBoost"] = clf
            except ModelIntegrityError:
                raise
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
            self.verify_artifact_file(
                p,
                manifest_payload=manifest_sha256 if manifest_sha256 else None,
                integrity_warnings=integrity_warnings,
            )
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
            manifest_sha256=manifest_sha256,
            integrity_warnings=integrity_warnings,
        )
