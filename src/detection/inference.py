"""Unified detection & inference pipeline combining supervised models, Autoencoder, SHAP, and response."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import EnsembleConfig
from src.data.feature_engineering import align_feature_schema
from src.detection.anomaly_detection import AnomalyDetector
from src.detection.threat_scoring import get_threat_level
from src.explainability.shap_explainer import explain_single_sample
from src.models.autoencoder import find_benign_class_index
from src.models.ensemble import HybridEnsembleEngine
from src.response.response_engine import ActiveResponseEngine
from src.utils.model_registry import LoadedArtifacts


@dataclass
class FlowPrediction:
    """Complete explainable prediction contract for a single network flow."""

    predicted_class: str
    confidence_score: float
    anomaly_score: float
    is_anomaly: bool
    threat_level: str
    selected_model: str
    important_contributing_features: list[dict[str, Any]]
    recommended_response: dict[str, Any]
    class_probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedDetectionEngine:
    """End-to-end inference service over persisted SMART IDS v3 artifacts."""

    def __init__(
        self,
        artifacts: LoadedArtifacts,
        ensemble_cfg: EnsembleConfig | None = None,
        response_engine: ActiveResponseEngine | None = None,
        calibrators: dict[str, Any] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.ensemble_cfg = ensemble_cfg or EnsembleConfig()
        self.response_engine = response_engine or ActiveResponseEngine(dry_run=True)

        if artifacts.label_encoder is not None:
            self.class_names = [str(c) for c in artifacts.label_encoder.classes_]
            self.benign_index, _ = find_benign_class_index(artifacts.label_encoder)
        else:
            self.class_names = [
                "BENIGN",
                "DDoS",
                "DoS GoldenEye",
                "DoS Hulk",
                "FTP-Patator",
                "PortScan",
                "SSH-Patator",
                "Web Attack",
            ]
            self.benign_index = 0

        self.anomaly_detector = AnomalyDetector(
            autoencoder_model=artifacts.models.get("Autoencoder"),
            threshold=artifacts.anomaly_threshold,
        )
        best_sup = str(
            artifacts.manifest.get("best_supervised_model")
            or ("XGBoost" if "XGBoost" in artifacts.models else "MLP")
        )
        self.ensemble = HybridEnsembleEngine(
            class_names=self.class_names,
            cfg=self.ensemble_cfg,
            best_model_name=best_sup,
            benign_index=self.benign_index,
            calibrators=calibrators or {},
            stacker=artifacts.stacker,
        )

    def preprocess_dataframe(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Align schema and apply fitted train-only imputer, clipper, and scaler."""
        if not self.artifacts.feature_names:
            raise RuntimeError("Missing feature_names.json in model registry.")
        aligned = align_feature_schema(df_raw, self.artifacts.feature_names)
        work = aligned
        if self.artifacts.imputer is not None:
            work = self.artifacts.imputer.transform(work)
        else:
            work = work.fillna(0.0)
        if self.artifacts.outlier_clipper is not None:
            work = self.artifacts.outlier_clipper.transform(work)
        if self.artifacts.scaler is not None:
            return np.asarray(self.artifacts.scaler.transform(work), dtype=np.float32)
        return work.to_numpy(dtype=np.float32)

    def _predict_supervised_proba(self, X_scaled: np.ndarray, model_name: str) -> np.ndarray:
        model = self.artifacts.models.get(model_name)
        if model is None:
            raise KeyError(f"Requested model '{model_name}' is not loaded.")
        if model_name in ("CNN", "MLP"):
            X_in = X_scaled.reshape(-1, X_scaled.shape[1], 1) if model_name == "CNN" else X_scaled
            if len(X_scaled) <= 32 and (
                "keras" in type(model).__module__ or "tensorflow" in type(model).__module__
            ):
                import tensorflow as tf

                cache = getattr(self, "_tf_callable_cache", None)
                if cache is None:
                    cache = {}
                    self._tf_callable_cache = cache
                fn = cache.get(model_name)
                if fn is None:
                    sig_shape = (
                        (None, X_scaled.shape[1], 1) if model_name == "CNN" else (None, X_scaled.shape[1])
                    )

                    @tf.function(
                        input_signature=[tf.TensorSpec(sig_shape, tf.float32)],
                        reduce_retracing=True,
                    )
                    def _direct_keras_fn(x: Any) -> Any:
                        return model(x, training=False)

                    fn = _direct_keras_fn
                    cache[model_name] = fn
                preds = fn(tf.constant(np.asarray(X_in, dtype=np.float32))).numpy()
                return np.asarray(preds, dtype=np.float64)
            return np.asarray(model.predict(X_in, batch_size=512, verbose=0), dtype=np.float64)

        if len(X_scaled) <= 32 and hasattr(model, "get_booster"):
            import json

            booster = model.get_booster()
            best_iter = None
            try:
                best_iter = model.best_iteration
            except AttributeError:
                best_iter = None
            iter_range = (0, int(best_iter) + 1) if best_iter is not None else (0, 0)
            orig_device: str | None = None
            try:
                dev = json.loads(booster.save_config())["learner"]["generic_param"].get("device", "cpu")
                if isinstance(dev, str) and "cuda" in dev.lower():
                    orig_device = dev
                    booster.set_param({"device": "cpu"})
            except Exception:
                orig_device = None
            try:
                preds = booster.inplace_predict(
                    np.asarray(X_scaled, dtype=np.float32),
                    iteration_range=iter_range,
                )
            finally:
                if orig_device is not None:
                    booster.set_param({"device": orig_device})
            return np.asarray(preds, dtype=np.float64)
        return np.asarray(model.predict_proba(X_scaled), dtype=np.float64)

    def predict_batch(
        self,
        df_raw: pd.DataFrame,
        model_choice: str = "Hybrid Ensemble",
        anomaly_threshold_override: float | None = None,
        min_actionable_confidence: float | None = None,
        trigger_response: bool = False,
        X_background: np.ndarray | None = None,
        explain_top_k: int = 0,
    ) -> tuple[pd.DataFrame, list[FlowPrediction]]:
        """Run unified supervised + unsupervised inference across a DataFrame of flows."""
        X_scaled = self.preprocess_dataframe(df_raw)
        anom_res = self.anomaly_detector.score(X_scaled, threshold_override=anomaly_threshold_override)

        supervised_available = [m for m in ("XGBoost", "MLP", "CNN") if m in self.artifacts.models]
        if not supervised_available and model_choice != "Autoencoder":
            raise RuntimeError("No supervised models are loaded in the model registry.")

        if model_choice == "Autoencoder":
            # Pure unsupervised anomaly view
            pred_labels = [
                "Zero-Day Anomaly (Novel)" if bool(flag) else self.class_names[self.benign_index]
                for flag in anom_res.is_anomaly
            ]
            # Confidence mapped smoothly via normalized reconstruction score
            confidences = np.where(
                anom_res.is_anomaly,
                np.clip(0.5 + 0.49 * (1.0 - np.exp(-(anom_res.severity_ratio - 1.0))), 0.5, 0.9999),
                np.clip(1.0 - 0.5 * anom_res.severity_ratio, 0.5, 0.9999),
            )
            probas = np.zeros((len(X_scaled), len(self.class_names)), dtype=np.float64)
            probas[:, self.benign_index] = np.where(anom_res.is_anomaly, 1.0 - confidences, confidences)
            used_label = "Autoencoder"
        elif model_choice in supervised_available:
            probas = self._predict_supervised_proba(X_scaled, model_choice)
            pred_idx = np.argmax(probas, axis=1)
            confidences = np.max(probas, axis=1)
            pred_labels = [self.class_names[int(i)] for i in pred_idx]
            used_label = model_choice
        else:
            # Hybrid Ensemble (or soft_voting / confidence_weighted / best_single)
            probas_dict = {m: self._predict_supervised_proba(X_scaled, m) for m in supervised_available}
            strategy = (
                model_choice
                if model_choice
                in {"best_single", "soft_voting", "confidence_weighted", "hybrid_anomaly_override"}
                else None
            )
            decision = self.ensemble.decide(
                model_probas=probas_dict,
                anomaly_scores=anom_res.scores,
                anomaly_threshold=anom_res.threshold,
                strategy=strategy,
            )
            probas = decision.probabilities
            confidences = decision.confidences
            pred_labels = decision.predicted_labels
            used_label = f"Hybrid Ensemble ({decision.strategy_used})"

        src_ips = (
            df_raw["Source IP"].astype(str).tolist()
            if "Source IP" in df_raw.columns
            else ["192.168.10.99"] * len(df_raw)
        )
        dst_ports = (
            pd.to_numeric(df_raw["Destination Port"], errors="coerce").fillna(0).astype(int).tolist()
            if "Destination Port" in df_raw.columns
            else [0] * len(df_raw)
        )
        timestamps = (
            df_raw["Timestamp"].astype(str).tolist()
            if "Timestamp" in df_raw.columns
            else [pd.Timestamp.utcnow().isoformat()] * len(df_raw)
        )

        # Optional local SHAP explanation for small batches / single-flow calls
        shap_model_name = (
            model_choice
            if model_choice in ("XGBoost", "MLP") and model_choice in self.artifacts.models
            else (
                "XGBoost"
                if "XGBoost" in self.artifacts.models
                else ("MLP" if "MLP" in self.artifacts.models else None)
            )
        )

        records: list[FlowPrediction] = []
        rows_out: list[dict[str, Any]] = []

        for idx in range(len(X_scaled)):
            p_class = pred_labels[idx]
            conf = float(confidences[idx])
            a_score = float(anom_res.scores[idx])
            a_flag = bool(anom_res.is_anomaly[idx])
            a_ratio = float(anom_res.severity_ratio[idx])

            t_level = get_threat_level(
                confidence=conf,
                attack_class=p_class,
                is_anomaly=a_flag,
                anomaly_ratio=a_ratio,
                min_actionable_confidence=min_actionable_confidence,
            )

            resp_dict = self.response_engine.apply_active_response(
                src_ip=src_ips[idx],
                attack_class=p_class,
                confidence=conf,
                model_used=used_label,
                is_anomaly=a_flag,
                anomaly_score=a_score,
                anomaly_ratio=a_ratio,
                min_actionable_confidence=min_actionable_confidence,
                persist_history=trigger_response,
            )

            contrib_feats: list[dict[str, Any]] = []
            if explain_top_k > 0 and shap_model_name is not None and idx < 5:
                bg_ref = X_background if X_background is not None else X_scaled
                cls_idx = int(np.argmax(probas[idx]))
                exp_res = explain_single_sample(
                    model_name=shap_model_name,
                    model=self.artifacts.models[shap_model_name],
                    x_single=X_scaled[idx],
                    X_background=bg_ref,
                    feature_names=self.artifacts.feature_names,
                    class_idx=cls_idx,
                    class_name=self.class_names[cls_idx],
                    waterfall_path=None,
                    max_display=explain_top_k,
                )
                contrib_feats = exp_res["top_features"]

            prob_map = {
                self.class_names[c_i]: round(float(probas[idx, c_i]), 6)
                for c_i in range(len(self.class_names))
            }
            records.append(
                FlowPrediction(
                    predicted_class=p_class,
                    confidence_score=round(conf, 6),
                    anomaly_score=round(a_score, 6),
                    is_anomaly=a_flag,
                    threat_level=t_level,
                    selected_model=used_label,
                    important_contributing_features=contrib_feats,
                    recommended_response=resp_dict,
                    class_probabilities=prob_map,
                )
            )
            rows_out.append(
                {
                    "timestamp": timestamps[idx],
                    "src_ip": src_ips[idx],
                    "dst_port": dst_ports[idx],
                    "attack_class": p_class,
                    "confidence": round(conf, 4),
                    "anomaly_score": round(a_score, 4),
                    "is_anomaly": a_flag,
                    "threat_level": t_level,
                    "model_used": used_label,
                    "action": resp_dict["action"],
                    "action_detail": resp_dict["action_detail"],
                    "blocked": resp_dict["blocked"],
                }
            )

        return pd.DataFrame(rows_out), records
