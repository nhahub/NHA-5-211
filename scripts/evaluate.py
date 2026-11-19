"""CLI evaluation, latency benchmarking, SHAP explainability, and academic report generation."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, ModelConfig, get_app_config, get_model_config
from src.detection.anomaly_detection import fit_classical_anomaly_baselines
from src.evaluation.benchmarking import benchmark_model_inference, benchmark_preprocessing_latency
from src.evaluation.metrics import (
    _binary_metrics,
    evaluate_anomaly_detector,
    evaluate_autoencoder,
    evaluate_supervised_model,
    evaluate_unseen_attack_protocol,
    paired_bootstrap_macro_f1_test,
)
from src.evaluation.protocols import (
    evaluate_by_day_protocol,
    evaluate_feature_ablation,
    evaluate_time_block_protocol,
    run_sample_size_sensitivity,
    run_xgboost_stratified_cv,
)
from src.evaluation.report_writer import format_ensemble_empirical_conclusion, write_academic_reports
from src.evaluation.visualizations import (
    plot_autoencoder_diagnostics,
    plot_confusion_matrices,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.explainability.shap_explainer import (
    compute_mlp_global_shap,
    compute_xgboost_global_shap,
    explain_single_sample,
)
from src.models.autoencoder import (
    compute_reconstruction_error,
    extract_benign_subset,
    find_benign_class_index,
)
from src.models.ensemble import HybridEnsembleEngine, StackingMetaLearner
from src.models.xgboost_model import expand_xgboost_probas
from src.utils.logging import get_logger
from src.utils.model_registry import ModelRegistry

logger = get_logger("smart_ids.scripts.evaluate")


def run_evaluation_suite(
    app_cfg: AppConfig | None = None,
    model_cfg: ModelConfig | None = None,
    skip_unseen: bool = False,
    skip_shap: bool = False,
    skip_protocols: bool = False,
) -> dict[str, Any]:
    """Evaluate all persisted models independently from training memory."""
    app_c = (app_cfg or get_app_config()).model_copy(deep=True)
    model_c = (model_cfg or get_model_config()).model_copy(deep=True)

    proc_dir = app_c.paths.resolve("processed_data_dir")
    models_dir = app_c.paths.resolve("models_dir")
    artifacts_dir = app_c.paths.resolve("artifacts_dir")
    reports_dir = app_c.paths.resolve("reports_dir")
    figures_dir = app_c.paths.resolve("figures_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    splits = np.load(proc_dir / "preprocessed_splits.npz")
    X_train_scaled = splits["X_train_scaled"]
    y_train = splits["y_train"]
    X_val_scaled = splits["X_val_scaled"]
    y_val = splits["y_val"]
    X_test_scaled = splits["X_test_scaled"]
    y_test = splits["y_test"]

    with (proc_dir / "preprocessing_audit.json").open("r", encoding="utf-8") as fh:
        prep_audit = json.load(fh)
    audit_split = prep_audit.get("split_strategy")
    if audit_split in {"random", "by_day", "time_block"}:
        app_c.data.split_strategy = audit_split

    train_diag: dict[str, Any] = {}
    diag_path = artifacts_dir / "training_diagnostics.json"
    if diag_path.exists():
        with diag_path.open("r", encoding="utf-8") as fh:
            train_diag = json.load(fh)

    registry = ModelRegistry(models_dir=models_dir, artifacts_dir=artifacts_dir)
    artifacts = registry.load_all(load_keras=True)
    class_names = [str(c) for c in artifacts.label_encoder.classes_]
    benign_idx, _ = find_benign_class_index(artifacts.label_encoder)
    y_test_binary = (y_test != benign_idx).astype(np.int64)

    num_features = X_test_scaled.shape[1]
    supervised_probas: dict[str, np.ndarray] = {}
    supervised_evals: dict[str, dict[str, Any]] = {}
    cm_norm_dict: dict[str, np.ndarray] = {}
    latency_table: dict[str, dict[str, float]] = {}

    boot_n = app_c.evaluation.bootstrap_n_resamples
    low_sup = app_c.evaluation.effective_reliable_support_threshold
    eval_seed = app_c.project.random_seed

    # 1. Evaluate & benchmark each supervised classifier
    if "MLP" in artifacts.models:
        p_mlp = np.asarray(artifacts.models["MLP"].predict(X_test_scaled, batch_size=1024, verbose=0))
        supervised_probas["MLP"] = p_mlp
        supervised_evals["MLP"] = evaluate_supervised_model(
            y_test,
            p_mlp,
            class_names,
            "MLP",
            bootstrap_n=boot_n,
            low_support_threshold=low_sup,
            seed=eval_seed,
        )
        cm_norm_dict["MLP"] = np.asarray(supervised_evals["MLP"]["confusion_matrix_normalized"])
        latency_table["MLP"] = benchmark_model_inference(
            artifacts.models["MLP"],
            X_test_scaled,
            "keras",
            model_path=models_dir / model_c.mlp.checkpoint_name,
            n_warmup=app_c.evaluation.latency_warmup_calls,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )

    if "CNN" in artifacts.models:
        X_test_cnn = X_test_scaled.reshape(-1, num_features, 1)
        p_cnn = np.asarray(artifacts.models["CNN"].predict(X_test_cnn, batch_size=1024, verbose=0))
        supervised_probas["CNN"] = p_cnn
        supervised_evals["CNN"] = evaluate_supervised_model(
            y_test,
            p_cnn,
            class_names,
            "1D-CNN",
            bootstrap_n=boot_n,
            low_support_threshold=low_sup,
            seed=eval_seed,
        )
        cm_norm_dict["CNN"] = np.asarray(supervised_evals["CNN"]["confusion_matrix_normalized"])
        latency_table["1D-CNN"] = benchmark_model_inference(
            artifacts.models["CNN"],
            X_test_scaled,
            "cnn",
            model_path=models_dir / model_c.cnn.checkpoint_name,
            n_warmup=app_c.evaluation.latency_warmup_calls,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )

    if "XGBoost" in artifacts.models:
        xgb_raw_proba = np.asarray(artifacts.models["XGBoost"].predict_proba(X_test_scaled))
        p_xgb = expand_xgboost_probas(xgb_raw_proba, np.unique(y_train).astype(np.int64), len(class_names))
        supervised_probas["XGBoost"] = p_xgb
        supervised_evals["XGBoost"] = evaluate_supervised_model(
            y_test,
            p_xgb,
            class_names,
            "XGBoost",
            bootstrap_n=boot_n,
            low_support_threshold=low_sup,
            seed=eval_seed,
        )
        cm_norm_dict["XGBoost"] = np.asarray(supervised_evals["XGBoost"]["confusion_matrix_normalized"])
        latency_table["XGBoost"] = benchmark_model_inference(
            artifacts.models["XGBoost"],
            X_test_scaled,
            "xgb",
            model_path=models_dir / model_c.xgboost.model_name,
            n_warmup=app_c.evaluation.latency_warmup_calls,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )

    # 2. Evaluate Ensemble Strategies on Test Set (trained/calibrated strictly on validation predictions)
    best_single = max(
        supervised_evals.keys(),
        key=lambda k: supervised_evals[k]["f1_macro"],
        default="XGBoost",
    )
    stacker = artifacts.stacker
    if stacker is None and len(supervised_probas) >= 2:
        val_probas: dict[str, np.ndarray] = {}
        if "MLP" in artifacts.models:
            val_probas["MLP"] = np.asarray(
                artifacts.models["MLP"].predict(X_val_scaled, batch_size=1024, verbose=0)
            )
        if "CNN" in artifacts.models:
            val_probas["CNN"] = np.asarray(
                artifacts.models["CNN"].predict(
                    X_val_scaled.reshape(-1, num_features, 1), batch_size=1024, verbose=0
                )
            )
        if "XGBoost" in artifacts.models:
            raw_v = np.asarray(artifacts.models["XGBoost"].predict_proba(X_val_scaled))
            val_probas["XGBoost"] = expand_xgboost_probas(
                raw_v, np.unique(y_train).astype(np.int64), len(class_names)
            )
        if len(val_probas) >= 2:
            stacker = StackingMetaLearner().fit(
                val_probas=val_probas,
                y_val=y_val,
                n_classes=len(class_names),
                calibrators=artifacts.calibrators or {},
                seed=eval_seed,
            )

    ensemble_engine = HybridEnsembleEngine(
        class_names=class_names,
        cfg=model_c.ensemble,
        best_model_name=best_single,
        benign_index=benign_idx,
        calibrators=artifacts.calibrators or {},
        stacker=stacker,
    )
    ensemble_comparison: dict[str, dict[str, Any]] = {}
    ensemble_preds: dict[str, np.ndarray] = {}
    for strat in (
        "best_single",
        "soft_voting",
        "confidence_weighted",
        "val_weighted_voting",
        "stacking",
    ):
        ens_probs, eff_name = ensemble_engine.combine_probabilities(supervised_probas, strategy=strat)
        ensemble_preds[strat] = np.argmax(ens_probs, axis=1)
        ens_eval = evaluate_supervised_model(
            y_test,
            ens_probs,
            class_names,
            f"Ensemble ({eff_name})",
            bootstrap_n=0,
            low_support_threshold=low_sup,
            seed=eval_seed,
        )
        ensemble_comparison[strat] = {
            "accuracy": ens_eval["accuracy"],
            "f1_macro": ens_eval["f1_macro"],
            "f1_macro_reliable": ens_eval["f1_macro_reliable"],
            "f1_weighted": ens_eval["f1_weighted"],
            "roc_auc_macro_ovr": ens_eval["roc_auc_macro_ovr"],
            "ece": ens_eval["ece"],
        }

    best_single_f1 = float(ensemble_comparison.get("best_single", {}).get("f1_macro", 0.0))
    fusion_keys = [k for k in ensemble_comparison if k != "best_single"]
    best_fusion = max(fusion_keys, key=lambda k: ensemble_comparison[k]["f1_macro"], default="soft_voting")
    best_fusion_f1 = float(ensemble_comparison.get(best_fusion, {}).get("f1_macro", 0.0))
    delta_ens = best_fusion_f1 - best_single_f1

    paired_n = int(boot_n) if boot_n > 0 else 1000
    paired_vs_best: dict[str, dict[str, Any]] = {}
    if "best_single" in ensemble_preds:
        for strat in ensemble_preds:
            if strat == "best_single":
                continue
            paired_vs_best[strat] = paired_bootstrap_macro_f1_test(
                y_true=y_test,
                y_pred_candidate=ensemble_preds[strat],
                y_pred_baseline=ensemble_preds["best_single"],
                num_classes=len(class_names),
                n_bootstrap=paired_n,
                seed=eval_seed,
            )

    paired_stacking_vs_all: dict[str, dict[str, Any]] = {}
    if "stacking" in ensemble_preds:
        for other in ensemble_preds:
            if other == "stacking":
                continue
            paired_stacking_vs_all[f"stacking_vs_{other}"] = paired_bootstrap_macro_f1_test(
                y_true=y_test,
                y_pred_candidate=ensemble_preds["stacking"],
                y_pred_baseline=ensemble_preds[other],
                num_classes=len(class_names),
                n_bootstrap=paired_n,
                seed=eval_seed,
            )

    ens_beats = bool(
        best_fusion_f1 > best_single_f1
        and paired_vs_best.get(best_fusion, {}).get("statistically_significant", False)
    )
    ensemble_analysis: dict[str, Any] = {
        "best_single_model": best_single,
        "best_single_f1_macro": round(best_single_f1, 6),
        "best_ensemble_strategy": best_fusion,
        "best_ensemble_f1_macro": round(best_fusion_f1, 6),
        "ensemble_beats_best_single": ens_beats,
        "stacking_vs_best_single_significant": bool(
            paired_vs_best.get("stacking", {}).get("statistically_significant", False)
        ),
        "delta_vs_best_single": round(delta_ens, 6),
        "val_weights": stacker.val_weights if stacker else {},
        "paired_bootstrap_vs_best_single": paired_vs_best,
        "paired_bootstrap_stacking_vs_all": paired_stacking_vs_all,
    }
    ensemble_analysis["honest_verdict"] = format_ensemble_empirical_conclusion(ensemble_analysis)

    # 3. Evaluate Autoencoder & Classical Benign-Only Anomaly Baselines + Unseen-Attack Protocol
    ae_eval: dict[str, Any] = {}
    ae_test_errors = np.zeros(len(X_test_scaled), dtype=np.float64)
    anomaly_evals: dict[str, dict[str, Any]] = {}
    baseline_anomaly_scores: dict[str, np.ndarray] = {}
    unseen_res: dict[str, Any] = {"executed": False}
    holdout_name = app_c.evaluation.unseen_attack_holdout_class
    holdout_idx = class_names.index(holdout_name) if holdout_name in class_names else None

    X_benign_val, _, _ = extract_benign_subset(X_val_scaled, y_val, artifacts.label_encoder)
    anom_val_benign_scores: dict[str, np.ndarray] = {}
    if "Autoencoder" in artifacts.models:
        ae_model = artifacts.models["Autoencoder"]
        benign_val_errors = compute_reconstruction_error(ae_model, X_benign_val)
        ae_test_errors = compute_reconstruction_error(ae_model, X_test_scaled)
        anom_val_benign_scores["Autoencoder"] = benign_val_errors
        ae_eval = evaluate_autoencoder(
            y_test_multiclass=y_test,
            recon_errors_test=ae_test_errors,
            benign_val_errors=benign_val_errors,
            threshold=artifacts.anomaly_threshold,
            class_names=class_names,
            benign_index=benign_idx,
            percentiles=app_c.evaluation.autoencoder_percentiles,
            bootstrap_n=boot_n,
            seed=eval_seed,
        )
        anomaly_evals["Autoencoder"] = ae_eval
        cm_norm_dict["Autoencoder"] = np.asarray(ae_eval["confusion_matrix_normalized"])
        latency_table["Autoencoder"] = benchmark_model_inference(
            ae_model,
            X_test_scaled,
            "ae",
            model_path=models_dir / model_c.autoencoder.checkpoint_name,
            n_warmup=app_c.evaluation.latency_warmup_calls,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )
        plot_autoencoder_diagnostics(
            recon_errors_test=ae_test_errors,
            y_test_binary=y_test_binary,
            threshold=artifacts.anomaly_threshold,
            percentile_sweep=ae_eval["percentile_sweep"],
            out_path=figures_dir / "autoencoder_diagnostics.png",
        )
        if not skip_unseen:
            unseen_res = evaluate_unseen_attack_protocol(
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                X_val_scaled=X_val_scaled,
                y_val=y_val,
                X_test_scaled=X_test_scaled,
                y_test=y_test,
                label_encoder=artifacts.label_encoder,
                holdout_class_name=holdout_name,
                epochs=10,
                seed=eval_seed,
            )

    # Evaluate benign-only neural variants (Deep Regularized Autoencoder & VAE) if present
    anom_var_meta = train_diag.get("anomaly_variants_meta", {})
    for var_name, ckpt_file in (
        ("Autoencoder (Deep Reg)", model_c.autoencoder.deep_reg_checkpoint_name),
        ("VAE", model_c.autoencoder.vae_checkpoint_name),
    ):
        if var_name not in artifacts.models:
            continue
        v_model = artifacts.models[var_name]
        v_val_err = compute_reconstruction_error(v_model, X_benign_val)
        v_test_err = compute_reconstruction_error(v_model, X_test_scaled)
        v_thresh = float(
            anom_var_meta.get(var_name, {}).get(
                "threshold",
                np.percentile(v_val_err, model_c.autoencoder.threshold_percentile)
                if len(v_val_err) > 0
                else artifacts.anomaly_threshold,
            )
        )
        anom_val_benign_scores[var_name] = v_val_err
        baseline_anomaly_scores[var_name] = v_test_err
        anomaly_evals[var_name] = evaluate_anomaly_detector(
            model_name=var_name,
            y_test_multiclass=y_test,
            anomaly_scores_test=v_test_err,
            benign_val_scores=v_val_err,
            threshold=v_thresh,
            class_names=class_names,
            benign_index=benign_idx,
            percentiles=app_c.evaluation.autoencoder_percentiles,
            bootstrap_n=boot_n,
            seed=eval_seed,
        )
        anomaly_evals[var_name]["fit_duration_sec"] = float(
            train_diag.get("training_durations_sec", {}).get(var_name, 0.0)
        )
        latency_table[var_name] = benchmark_model_inference(
            v_model,
            X_test_scaled,
            "ae",
            model_path=models_dir / ckpt_file,
            n_warmup=app_c.evaluation.latency_warmup_calls,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )

    # Evaluate Classical Benign-Only Anomaly Baselines (Isolation Forest, One-Class SVM, LOF)
    anomaly_baselines = artifacts.anomaly_baselines or {}
    if not anomaly_baselines and model_c.anomaly_baselines.enabled and len(X_train_scaled) > 0:
        X_benign_train, _, _ = extract_benign_subset(X_train_scaled, y_train, artifacts.label_encoder)
        anomaly_baselines = fit_classical_anomaly_baselines(
            X_benign_train=X_benign_train,
            X_benign_val=X_benign_val,
            cfg=model_c.anomaly_baselines,
            percentile=model_c.autoencoder.threshold_percentile,
            seed=eval_seed,
        )

    baseline_model_sizes: dict[str, float] = {}
    for b_name, base_obj in anomaly_baselines.items():
        val_scores = base_obj.score(X_benign_val)
        test_scores = base_obj.score(X_test_scaled)
        anom_val_benign_scores[b_name] = val_scores
        baseline_anomaly_scores[b_name] = test_scores
        anomaly_evals[b_name] = evaluate_anomaly_detector(
            model_name=b_name,
            y_test_multiclass=y_test,
            anomaly_scores_test=test_scores,
            benign_val_scores=val_scores,
            threshold=base_obj.threshold,
            class_names=class_names,
            benign_index=benign_idx,
            percentiles=app_c.evaluation.autoencoder_percentiles,
            bootstrap_n=boot_n,
            seed=eval_seed,
        )
        anomaly_evals[b_name]["fit_duration_sec"] = float(getattr(base_obj, "fit_duration_sec", 0.0))
        lat_b = benchmark_model_inference(
            base_obj.estimator,
            X_test_scaled,
            "sklearn_anomaly",
            model_path=None,
            n_warmup=app_c.evaluation.latency_warmup_calls,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )
        buf = io.BytesIO()
        joblib.dump(base_obj, buf)
        baseline_model_sizes[b_name] = round(len(buf.getvalue()) / (1024.0 * 1024.0), 3)
        lat_b["model_size_mb"] = baseline_model_sizes[b_name]
        latency_table[b_name] = lat_b

    # Evaluate Score-Level Anomaly Fusion (weights chosen strictly on non-test benign val + LOO attacks)
    fusion_obj = artifacts.anomaly_fusion
    all_test_scores = {"Autoencoder": ae_test_errors, **baseline_anomaly_scores}
    if fusion_obj is not None:
        needed_comps = [k for k, w in fusion_obj.weights.items() if w > 0]
        if needed_comps and all(k in all_test_scores and k in anom_val_benign_scores for k in needed_comps):
            fusion_name = getattr(fusion_obj, "name", "Score-Level Fusion (AE+OCSVM+IF)")
            f_val_scores = fusion_obj.score_from_components(anom_val_benign_scores)
            f_test_scores = fusion_obj.score_from_components(all_test_scores)
            baseline_anomaly_scores[fusion_name] = f_test_scores
            anomaly_evals[fusion_name] = evaluate_anomaly_detector(
                model_name=fusion_name,
                y_test_multiclass=y_test,
                anomaly_scores_test=f_test_scores,
                benign_val_scores=f_val_scores,
                threshold=float(fusion_obj.threshold),
                class_names=class_names,
                benign_index=benign_idx,
                percentiles=app_c.evaluation.autoencoder_percentiles,
                bootstrap_n=boot_n,
                seed=eval_seed,
            )

    # Summarize comparison across all benign-only anomaly detectors
    anom_val_sel = train_diag.get("anomaly_validation_selection", {})
    val_winner_name = str(
        anom_val_sel.get("best_overall_validation_detector")
        or (getattr(fusion_obj, "best_overall_validation_detector", "") if fusion_obj else "")
    )
    anomaly_detector_comparison: dict[str, Any] = {}
    if anomaly_evals:
        comp_rows = []
        for det_name, det_ev in anomaly_evals.items():
            holdout_rec = None
            holdout_auc = None
            if holdout_idx is not None:
                hold_cls_stats = det_ev.get("per_class_anomaly_detection_rate", {}).get(holdout_name)
                if hold_cls_stats:
                    holdout_rec = float(hold_cls_stats["flagged_rate"])
                eval_m = (y_test == benign_idx) | (y_test == holdout_idx)
                scores_for_auc = (
                    ae_test_errors if det_name == "Autoencoder" else baseline_anomaly_scores[det_name]
                )
                y_nov = (y_test[eval_m] == holdout_idx).astype(np.int64)
                if len(np.unique(y_nov)) > 1:
                    holdout_auc = float(
                        _binary_metrics(y_nov, scores_for_auc[eval_m], det_ev["threshold"])["roc_auc"]
                    )
            det_ci = det_ev.get("f1_ci95", {})
            comp_rows.append(
                {
                    "Detector": det_name,
                    "Accuracy": float(det_ev["accuracy"]),
                    "Precision": float(det_ev["precision"]),
                    "Recall": float(det_ev["recall"]),
                    "F1": float(det_ev["f1"]),
                    "F1_CI95_Low": round(float(det_ci.get("ci95_low", det_ev["f1"])), 4),
                    "F1_CI95_High": round(float(det_ci.get("ci95_high", det_ev["f1"])), 4),
                    "ROC_AUC": float(det_ev["roc_auc"]),
                    "PR_AUC": float(det_ev["pr_auc"]),
                    "FPR": float(det_ev["fpr"]),
                    "Holdout_Class_Recall": holdout_rec,
                    "Holdout_Class_ROC_AUC": holdout_auc,
                    "Val_Selected": "YES" if det_name == val_winner_name else "",
                }
            )
        best_f1_det = max(comp_rows, key=lambda r: r["F1"])
        best_auc_det = max(comp_rows, key=lambda r: r["ROC_AUC"])
        ae_row = next((r for r in comp_rows if r["Detector"] == "Autoencoder"), None)
        verdict = (
            f"Best binary F1 at {model_c.autoencoder.threshold_percentile:.0f}th-pct threshold: "
            f"{best_f1_det['Detector']} (F1={best_f1_det['F1']:.4f} "
            f"[{best_f1_det['F1_CI95_Low']:.4f}, {best_f1_det['F1_CI95_High']:.4f}], "
            f"Recall={best_f1_det['Recall']:.4f}, FPR={best_f1_det['FPR']:.4f}); "
            f"best overall ROC-AUC ranking: {best_auc_det['Detector']} (ROC-AUC={best_auc_det['ROC_AUC']:.4f}); "
            f"validation-selected option (leave-attack-out non-test): '{val_winner_name or best_f1_det['Detector']}'."
        )
        if ae_row and best_f1_det["Detector"] != "Autoencoder":
            verdict += (
                f" Baseline Autoencoder achieved F1={ae_row['F1']:.4f} / ROC-AUC={ae_row['ROC_AUC']:.4f}. "
                "All benign-only unsupervised options remain strictly supplementary zero-day signals and do NOT "
                "replace supervised multi-class XGBoost."
            )
        else:
            verdict += (
                " Even the best unsupervised option/fusion still trails supervised attribution by a wide margin "
                "and remains strictly a supplementary zero-day signal, not a replacement for XGBoost."
            )
        anomaly_detector_comparison = {
            "holdout_class": holdout_name,
            "best_by_f1": best_f1_det["Detector"],
            "best_by_roc_auc": best_auc_det["Detector"],
            "validation_winner": val_winner_name,
            "val_selection": anom_val_sel,
            "detectors": comp_rows,
            "verdict": verdict,
        }

    # 4. Preprocessing latency benchmark (isolated from model inference)
    prep_bench_df = pd.DataFrame(X_test_scaled[:10], columns=artifacts.feature_names)
    prep_latency = benchmark_preprocessing_latency(
        raw_df_sample=prep_bench_df,
        feature_names=artifacts.feature_names,
        imputer=artifacts.imputer,
        clipper=artifacts.outlier_clipper,
        scaler=artifacts.scaler,
        n_warmup=app_c.evaluation.latency_warmup_calls,
        n_runs=app_c.evaluation.latency_timed_runs,
    )

    # 5. Comparison CSV (exact contract expected by report & dashboard)
    durations = train_diag.get("training_durations_sec", {})
    imb_by_model = dict(
        train_diag.get("imbalance_strategies_by_model") or artifacts.model_imbalance_strategies or {}
    )
    fallback_imb = train_diag.get("imbalance_experiment", {}).get("selected_strategy", "none")
    rows_cmp: list[dict[str, Any]] = []
    name_map = {"MLP": "MLP", "CNN": "1D-CNN", "XGBoost": "XGBoost"}
    for key, display in name_map.items():
        if key not in supervised_evals:
            continue
        ev = supervised_evals[key]
        lat = latency_table.get(display, {})
        ci = ev.get("f1_macro_ci95", {})
        rows_cmp.append(
            {
                "Model": display,
                "Imbalance_Strategy": imb_by_model.get(key, fallback_imb),
                "Evaluation_Framing": ev["framing"],
                "Accuracy": round(ev["accuracy"], 4),
                "Precision_Weighted": round(ev["precision_weighted"], 4),
                "Recall_Weighted": round(ev["recall_weighted"], 4),
                "F1_Weighted": round(ev["f1_weighted"], 4),
                "F1_Macro": round(ev["f1_macro"], 4),
                "F1_Macro_Reliable": round(ev["f1_macro_reliable"], 4),
                "F1_Macro_CI95_Low": round(ci.get("ci95_low", ev["f1_macro"]), 4),
                "F1_Macro_CI95_High": round(ci.get("ci95_high", ev["f1_macro"]), 4),
                "ROC_AUC": round(ev["roc_auc_macro_ovr"], 4),
                "PR_AUC": round(ev["pr_auc_macro_ovr"], 4),
                "Latency_Median_us": round(lat.get("single_median_us", 0.0), 2),
                "Latency_P95_us": round(lat.get("single_p95_us", 0.0), 2),
                "Latency_Mean_us": round(lat.get("single_mean_us", 0.0), 2),
                "Samples_Per_Sec": round(lat.get("batch_samples_per_sec", 0.0), 1),
                "Model_Size_MB": lat.get("model_size_mb", 0.0),
                "Training_Duration_s": round(float(durations.get(key, 0.0)), 2),
            }
        )

    for det_name in anomaly_evals:
        det_ev = anomaly_evals[det_name]
        lat_det = latency_table.get(det_name, {})
        det_ci = det_ev.get("f1_ci95", {})
        fit_dur = durations.get(det_name, det_ev.get("fit_duration_sec", 0.0))
        rows_cmp.append(
            {
                "Model": det_name,
                "Imbalance_Strategy": "benign_only",
                "Evaluation_Framing": det_ev["framing"],
                "Accuracy": round(det_ev["accuracy"], 4),
                "Precision_Weighted": round(det_ev["precision"], 4),
                "Recall_Weighted": round(det_ev["recall"], 4),
                "F1_Weighted": round(det_ev["f1"], 4),
                "F1_Macro": round(det_ev["f1"], 4),
                "F1_Macro_Reliable": None,
                "F1_Macro_CI95_Low": round(det_ci.get("ci95_low", det_ev["f1"]), 4),
                "F1_Macro_CI95_High": round(det_ci.get("ci95_high", det_ev["f1"]), 4),
                "ROC_AUC": round(det_ev["roc_auc"], 4),
                "PR_AUC": round(det_ev["pr_auc"], 4),
                "Latency_Median_us": round(lat_det.get("single_median_us", 0.0), 2),
                "Latency_P95_us": round(lat_det.get("single_p95_us", 0.0), 2),
                "Latency_Mean_us": round(lat_det.get("single_mean_us", 0.0), 2),
                "Samples_Per_Sec": round(lat_det.get("batch_samples_per_sec", 0.0), 1),
                "Model_Size_MB": lat_det.get("model_size_mb", 0.0),
                "Training_Duration_s": round(float(fit_dur), 2),
            }
        )

    df_cmp = pd.DataFrame(rows_cmp)
    df_cmp.to_csv(reports_dir / "model_comparison.csv", index=False)

    # 6. Plots
    plot_confusion_matrices(cm_norm_dict, class_names, figures_dir / "confusion_matrices.png")
    plot_roc_curves(
        y_test=y_test,
        supervised_probas=supervised_probas,
        y_test_binary=y_test_binary,
        ae_errors=ae_test_errors,
        class_names=class_names,
        out_path=figures_dir / "roc_curves.png",
        baseline_anomaly_scores=baseline_anomaly_scores,
    )
    plot_precision_recall_curves(
        y_test=y_test,
        supervised_probas=supervised_probas,
        y_test_binary=y_test_binary,
        ae_errors=ae_test_errors,
        class_names=class_names,
        out_path=figures_dir / "pr_curves.png",
        baseline_anomaly_scores=baseline_anomaly_scores,
    )

    # 7. SHAP Explainability
    shap_summary: dict[str, Any] = {}
    xgb_present_cls = np.unique(y_train).astype(np.int64)
    xgb_class_names = [class_names[int(c)] for c in xgb_present_cls]
    if not skip_shap:
        if "XGBoost" in artifacts.models:
            xgb_shap_res, _, _ = compute_xgboost_global_shap(
                xgb_model=artifacts.models["XGBoost"],
                X_test=X_test_scaled,
                feature_names=artifacts.feature_names,
                class_names=xgb_class_names,
                max_samples=app_c.evaluation.shap_xgb_max_samples,
                summary_fig_path=figures_dir / "shap_xgb_summary.png",
            )
            shap_summary["XGBoost"] = {
                "explainer_type": xgb_shap_res.explainer_type,
                "n_samples_explained": xgb_shap_res.n_samples_explained,
                "elapsed_sec": xgb_shap_res.elapsed_sec,
                "top_features": xgb_shap_res.feature_importance[:20],
            }

        if "MLP" in artifacts.models:
            mlp_shap_res, mlp_bench = compute_mlp_global_shap(
                mlp_model=artifacts.models["MLP"],
                X_background=X_train_scaled,
                X_explain=X_test_scaled,
                feature_names=artifacts.feature_names,
                n_background=app_c.evaluation.shap_mlp_background_samples,
                n_explain=app_c.evaluation.shap_mlp_explain_samples,
                fig_path=figures_dir / "shap_mlp_importance.png",
                benchmark_kernel=True,
            )
            shap_summary["MLP"] = {
                "explainer_type": mlp_shap_res.explainer_type,
                "n_samples_explained": mlp_shap_res.n_samples_explained,
                "elapsed_sec": mlp_shap_res.elapsed_sec,
                "top_features": mlp_shap_res.feature_importance[:20],
                "explainer_benchmark": mlp_bench,
            }

            # Local waterfall on a true-positive critical attack sample (preferring trained classes)
            xgb_present_set = set(xgb_present_cls.tolist())
            attack_indices = [i for i, c in enumerate(class_names) if i != benign_idx]
            target_sample_idx = 0
            target_cls_idx = attack_indices[0] if attack_indices else benign_idx
            found_match = False
            if "MLP" in supervised_probas:
                mlp_preds = np.argmax(supervised_probas["MLP"], axis=1)
                preferred_attacks = [a for a in attack_indices if a in xgb_present_set] or attack_indices
                for a_idx in preferred_attacks:
                    matches = np.where((y_test == a_idx) & (mlp_preds == a_idx))[0]
                    if len(matches) > 0:
                        target_sample_idx = int(matches[0])
                        target_cls_idx = int(a_idx)
                        found_match = True
                        break
            use_xgb_local = "XGBoost" in artifacts.models and (
                target_cls_idx in xgb_present_set or benign_idx in xgb_present_set
            )
            if use_xgb_local:
                if target_cls_idx not in xgb_present_set:
                    target_cls_idx = benign_idx
                    if not found_match:
                        ben_matches = np.where(y_test == benign_idx)[0]
                        if len(ben_matches) > 0:
                            target_sample_idx = int(ben_matches[0])
                local_cls_idx = int(np.searchsorted(xgb_present_cls, target_cls_idx))
                local_model_name = "XGBoost"
                local_model_obj = artifacts.models["XGBoost"]
            else:
                local_cls_idx = target_cls_idx
                local_model_name = "MLP"
                local_model_obj = artifacts.models["MLP"]

            local_exp = explain_single_sample(
                model_name=local_model_name,
                model=local_model_obj,
                x_single=X_test_scaled[target_sample_idx],
                X_background=X_train_scaled,
                feature_names=artifacts.feature_names,
                class_idx=local_cls_idx,
                class_name=class_names[target_cls_idx],
                waterfall_path=figures_dir / "shap_waterfall.png",
            )
            shap_summary["local_example"] = {
                "sample_test_index": target_sample_idx,
                "true_class": class_names[int(y_test[target_sample_idx])],
                **{k: v for k, v in local_exp.items() if k != "all_shap_values"},
            }

        with (artifacts_dir / "shap_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(shap_summary, fh, indent=2)

    # 8. Secondary Evaluation Protocols (Open-Set Cross-Day by_day, Chronological time_block & Ablation)
    is_synth = bool(prep_audit.get("source", {}).get("is_synthetic", False))
    xgb_opt_diag = train_diag.get("xgboost_optimization", {})
    selected_xgb_params = (
        xgb_opt_diag.get("best_params") if xgb_opt_diag.get("selected_variant") == "tuned" else None
    )
    imb_strategy = train_diag.get("imbalance_experiment", {}).get("selected_strategy", "none")

    by_day_res: dict[str, Any] = {"executed": False, "reason": "Skipped via flag or config."}
    time_block_res: dict[str, Any] = {"executed": False, "reason": "Skipped via flag or config."}
    ablation_res: dict[str, Any] = {"executed": False, "reason": "Skipped via flag or config."}
    xgb_cv_res: dict[str, Any] = {"executed": False, "reason": "Skipped via flag or config."}
    sensitivity_res: dict[str, Any] = {"executed": False, "reason": "Skipped via flag or config."}
    if not skip_protocols:
        if app_c.evaluation.run_by_day_comparison:
            by_day_res = evaluate_by_day_protocol(
                app_cfg=app_c,
                model_cfg=model_c,
                xgb_params=selected_xgb_params,
                imbalance_strategy=imb_strategy,
                keras_epochs=app_c.evaluation.by_day_keras_epochs,
                force_synthetic=is_synth,
            )
        if app_c.evaluation.run_time_block_comparison:
            time_block_res = evaluate_time_block_protocol(
                app_cfg=app_c,
                model_cfg=model_c,
                xgb_params=selected_xgb_params,
                imbalance_strategy=imb_strategy,
                keras_epochs=app_c.evaluation.time_block_keras_epochs,
                force_synthetic=is_synth,
            )
        if app_c.evaluation.run_feature_ablation:
            ablation_res = evaluate_feature_ablation(
                app_cfg=app_c,
                model_cfg=model_c,
                drop_features=app_c.evaluation.ablation_drop_features,
                xgb_params=selected_xgb_params,
                imbalance_strategy=imb_strategy,
                force_synthetic=is_synth,
                compute_shap=not skip_shap,
            )
        if app_c.evaluation.run_xgboost_cv:
            if app_c.data.split_strategy == "by_day":
                xgb_cv_res = {
                    "executed": False,
                    "reason": "Skipped all-class stratified CV under open-set cross-day (by_day) primary split.",
                }
            else:
                xgb_cv_res = run_xgboost_stratified_cv(
                    app_cfg=app_c,
                    model_cfg=model_c,
                    xgb_params=selected_xgb_params,
                    imbalance_strategy=imb_strategy,
                    n_splits=app_c.evaluation.cv_n_splits,
                    force_synthetic=is_synth,
                )
        if app_c.evaluation.run_sample_size_sensitivity:
            if is_synth:
                sensitivity_res = {
                    "executed": False,
                    "reason": "Skipped sample-size sensitivity under synthetic mode (requires raw CICIDS2017 CSVs).",
                }
            else:
                sensitivity_res = run_sample_size_sensitivity(
                    app_cfg=app_c,
                    model_cfg=model_c,
                    xgb_params=selected_xgb_params,
                    imbalance_strategy=imb_strategy,
                    sample_sizes=app_c.evaluation.sensitivity_sample_sizes,
                    force_synthetic=False,
                )

    full_summary = {
        "dataset_source": prep_audit.get("source", {}),
        "split_strategy": prep_audit.get("split_strategy", app_c.data.split_strategy),
        "preprocessing_audit": prep_audit,
        "training_diagnostics": train_diag,
        "supervised_evaluations": supervised_evals,
        "ensemble_comparison": ensemble_comparison,
        "ensemble_analysis": ensemble_analysis,
        "autoencoder_evaluation": ae_eval,
        "anomaly_evaluations": anomaly_evals,
        "anomaly_detector_comparison": anomaly_detector_comparison,
        "unseen_attack_protocol": unseen_res,
        "by_day_protocol": by_day_res,
        "time_block_protocol": time_block_res,
        "feature_ablation": ablation_res,
        "xgboost_cross_validation": xgb_cv_res,
        "sample_size_sensitivity": sensitivity_res,
        "preprocessing_latency": prep_latency,
        "inference_latency": latency_table,
        "shap_summary": shap_summary,
    }
    with (reports_dir / "evaluation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(full_summary, fh, indent=2)

    write_academic_reports(full_summary, df_cmp, reports_dir)
    logger.info(
        "Evaluation complete. Saved model_comparison.csv, evaluation_summary.json, and markdown reports to %s",
        reports_dir,
    )
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SMART IDS v3 models.")
    parser.add_argument(
        "--skip-unseen", action="store_true", help="Skip leave-one-attack-out retraining protocol."
    )
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP generation.")
    parser.add_argument(
        "--skip-protocols",
        action="store_true",
        help="Skip secondary open-set cross-day (by_day), time_block, and feature ablation protocols.",
    )
    args = parser.parse_args()
    run_evaluation_suite(
        skip_unseen=args.skip_unseen,
        skip_shap=args.skip_shap,
        skip_protocols=args.skip_protocols,
    )


if __name__ == "__main__":
    main()
