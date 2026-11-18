"""CLI evaluation, latency benchmarking, SHAP explainability, and academic report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, ModelConfig, get_app_config, get_model_config
from src.evaluation.benchmarking import benchmark_model_inference, benchmark_preprocessing_latency
from src.evaluation.metrics import (
    evaluate_autoencoder,
    evaluate_supervised_model,
    evaluate_unseen_attack_protocol,
)
from src.evaluation.report_writer import write_academic_reports
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
from src.models.ensemble import HybridEnsembleEngine
from src.utils.logging import get_logger
from src.utils.model_registry import ModelRegistry

logger = get_logger("smart_ids.scripts.evaluate")


def run_evaluation_suite(
    app_cfg: AppConfig | None = None,
    model_cfg: ModelConfig | None = None,
    skip_unseen: bool = False,
    skip_shap: bool = False,
) -> dict[str, Any]:
    """Evaluate all persisted models independently from training memory."""
    app_c = app_cfg or get_app_config()
    model_c = model_cfg or get_model_config()

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

    # 1. Evaluate & benchmark each supervised classifier
    if "MLP" in artifacts.models:
        p_mlp = np.asarray(artifacts.models["MLP"].predict(X_test_scaled, batch_size=1024, verbose=0))
        supervised_probas["MLP"] = p_mlp
        supervised_evals["MLP"] = evaluate_supervised_model(y_test, p_mlp, class_names, "MLP")
        cm_norm_dict["MLP"] = np.asarray(supervised_evals["MLP"]["confusion_matrix_normalized"])
        latency_table["MLP"] = benchmark_model_inference(
            artifacts.models["MLP"],
            X_test_scaled,
            "keras",
            model_path=models_dir / model_c.mlp.checkpoint_name,
            n_warmup=app_c.evaluation.latency_warmup_runs,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )

    if "CNN" in artifacts.models:
        X_test_cnn = X_test_scaled.reshape(-1, num_features, 1)
        p_cnn = np.asarray(artifacts.models["CNN"].predict(X_test_cnn, batch_size=1024, verbose=0))
        supervised_probas["CNN"] = p_cnn
        supervised_evals["CNN"] = evaluate_supervised_model(y_test, p_cnn, class_names, "1D-CNN")
        cm_norm_dict["CNN"] = np.asarray(supervised_evals["CNN"]["confusion_matrix_normalized"])
        latency_table["1D-CNN"] = benchmark_model_inference(
            artifacts.models["CNN"],
            X_test_scaled,
            "cnn",
            model_path=models_dir / model_c.cnn.checkpoint_name,
            n_warmup=app_c.evaluation.latency_warmup_runs,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )

    if "XGBoost" in artifacts.models:
        p_xgb = np.asarray(artifacts.models["XGBoost"].predict_proba(X_test_scaled))
        supervised_probas["XGBoost"] = p_xgb
        supervised_evals["XGBoost"] = evaluate_supervised_model(y_test, p_xgb, class_names, "XGBoost")
        cm_norm_dict["XGBoost"] = np.asarray(supervised_evals["XGBoost"]["confusion_matrix_normalized"])
        latency_table["XGBoost"] = benchmark_model_inference(
            artifacts.models["XGBoost"],
            X_test_scaled,
            "xgb",
            model_path=models_dir / model_c.xgboost.model_name,
            n_warmup=app_c.evaluation.latency_warmup_runs,
            n_runs=app_c.evaluation.latency_timed_runs,
            batch_size=app_c.evaluation.batch_benchmark_size,
        )

    # 2. Evaluate Ensemble Strategies on Test Set (comparing best_single, soft_voting, confidence_weighted)
    best_single = max(
        supervised_evals.keys(),
        key=lambda k: supervised_evals[k]["f1_macro"],
        default="XGBoost",
    )
    ensemble_engine = HybridEnsembleEngine(
        class_names=class_names,
        cfg=model_c.ensemble,
        best_model_name=best_single,
        benign_index=benign_idx,
        calibrators=artifacts.calibrators or {},
    )
    ensemble_comparison: dict[str, dict[str, Any]] = {}
    for strat in ("best_single", "soft_voting", "confidence_weighted"):
        ens_probs, eff_name = ensemble_engine.combine_probabilities(supervised_probas, strategy=strat)
        ens_eval = evaluate_supervised_model(y_test, ens_probs, class_names, f"Ensemble ({eff_name})")
        ensemble_comparison[strat] = {
            "accuracy": ens_eval["accuracy"],
            "f1_macro": ens_eval["f1_macro"],
            "f1_weighted": ens_eval["f1_weighted"],
            "roc_auc_macro_ovr": ens_eval["roc_auc_macro_ovr"],
            "ece": ens_eval["ece"],
        }

    # 3. Evaluate Autoencoder Binary Anomaly & Unseen-Attack Protocol
    ae_eval: dict[str, Any] = {}
    ae_test_errors = np.zeros(len(X_test_scaled), dtype=np.float64)
    unseen_res: dict[str, Any] = {"executed": False}
    if "Autoencoder" in artifacts.models:
        ae_model = artifacts.models["Autoencoder"]
        X_benign_val, _, _ = extract_benign_subset(X_val_scaled, y_val, artifacts.label_encoder)
        benign_val_errors = compute_reconstruction_error(ae_model, X_benign_val)
        ae_test_errors = compute_reconstruction_error(ae_model, X_test_scaled)
        ae_eval = evaluate_autoencoder(
            y_test_multiclass=y_test,
            recon_errors_test=ae_test_errors,
            benign_val_errors=benign_val_errors,
            threshold=artifacts.anomaly_threshold,
            class_names=class_names,
            benign_index=benign_idx,
            percentiles=app_c.evaluation.autoencoder_percentiles,
        )
        cm_norm_dict["Autoencoder"] = np.asarray(ae_eval["confusion_matrix_normalized"])
        latency_table["Autoencoder"] = benchmark_model_inference(
            ae_model,
            X_test_scaled,
            "ae",
            model_path=models_dir / model_c.autoencoder.checkpoint_name,
            n_warmup=app_c.evaluation.latency_warmup_runs,
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
                holdout_class_name=app_c.evaluation.unseen_attack_holdout_class,
                epochs=10,
                seed=app_c.project.random_seed,
            )

    # 4. Preprocessing latency benchmark (isolated from model inference)
    prep_bench_df = pd.DataFrame(X_test_scaled[:10], columns=artifacts.feature_names)
    prep_latency = benchmark_preprocessing_latency(
        raw_df_sample=prep_bench_df,
        feature_names=artifacts.feature_names,
        imputer=artifacts.imputer,
        clipper=artifacts.outlier_clipper,
        scaler=artifacts.scaler,
        n_warmup=app_c.evaluation.latency_warmup_runs,
        n_runs=app_c.evaluation.latency_timed_runs,
    )

    # 5. Comparison CSV (exact contract expected by report & dashboard)
    durations = train_diag.get("training_durations_sec", {})
    rows_cmp: list[dict[str, Any]] = []
    name_map = {"MLP": "MLP", "CNN": "1D-CNN", "XGBoost": "XGBoost"}
    for key, display in name_map.items():
        if key not in supervised_evals:
            continue
        ev = supervised_evals[key]
        lat = latency_table.get(display, {})
        rows_cmp.append(
            {
                "Model": display,
                "Evaluation_Framing": ev["framing"],
                "Accuracy": round(ev["accuracy"], 4),
                "Precision_Weighted": round(ev["precision_weighted"], 4),
                "Recall_Weighted": round(ev["recall_weighted"], 4),
                "F1_Weighted": round(ev["f1_weighted"], 4),
                "F1_Macro": round(ev["f1_macro"], 4),
                "ROC_AUC": round(ev["roc_auc_macro_ovr"], 4),
                "PR_AUC": round(ev["pr_auc_macro_ovr"], 4),
                "Latency_Mean_us": round(lat.get("single_mean_us", 0.0), 2),
                "Latency_Median_us": round(lat.get("single_median_us", 0.0), 2),
                "Latency_P95_us": round(lat.get("single_p95_us", 0.0), 2),
                "Samples_Per_Sec": round(lat.get("batch_samples_per_sec", 0.0), 1),
                "Model_Size_MB": lat.get("model_size_mb", 0.0),
                "Training_Duration_s": round(float(durations.get(key, 0.0)), 2),
            }
        )

    if ae_eval:
        lat_ae = latency_table.get("Autoencoder", {})
        rows_cmp.append(
            {
                "Model": "Autoencoder",
                "Evaluation_Framing": ae_eval["framing"],
                "Accuracy": round(ae_eval["accuracy"], 4),
                "Precision_Weighted": round(ae_eval["precision"], 4),
                "Recall_Weighted": round(ae_eval["recall"], 4),
                "F1_Weighted": round(ae_eval["f1"], 4),
                "F1_Macro": round(ae_eval["f1"], 4),
                "ROC_AUC": round(ae_eval["roc_auc"], 4),
                "PR_AUC": round(ae_eval["pr_auc"], 4),
                "Latency_Mean_us": round(lat_ae.get("single_mean_us", 0.0), 2),
                "Latency_Median_us": round(lat_ae.get("single_median_us", 0.0), 2),
                "Latency_P95_us": round(lat_ae.get("single_p95_us", 0.0), 2),
                "Samples_Per_Sec": round(lat_ae.get("batch_samples_per_sec", 0.0), 1),
                "Model_Size_MB": lat_ae.get("model_size_mb", 0.0),
                "Training_Duration_s": round(float(durations.get("Autoencoder", 0.0)), 2),
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
    )
    plot_precision_recall_curves(
        y_test=y_test,
        supervised_probas=supervised_probas,
        y_test_binary=y_test_binary,
        ae_errors=ae_test_errors,
        class_names=class_names,
        out_path=figures_dir / "pr_curves.png",
    )

    # 7. SHAP Explainability
    shap_summary: dict[str, Any] = {}
    if not skip_shap:
        if "XGBoost" in artifacts.models:
            xgb_shap_res, _, _ = compute_xgboost_global_shap(
                xgb_model=artifacts.models["XGBoost"],
                X_test=X_test_scaled,
                feature_names=artifacts.feature_names,
                class_names=class_names,
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

            # Local waterfall on a true-positive critical attack sample
            attack_indices = [i for i, c in enumerate(class_names) if i != benign_idx]
            target_sample_idx = 0
            target_cls_idx = attack_indices[0] if attack_indices else 0
            if "MLP" in supervised_probas:
                mlp_preds = np.argmax(supervised_probas["MLP"], axis=1)
                for a_idx in attack_indices:
                    matches = np.where((y_test == a_idx) & (mlp_preds == a_idx))[0]
                    if len(matches) > 0:
                        target_sample_idx = int(matches[0])
                        target_cls_idx = int(a_idx)
                        break

            local_exp = explain_single_sample(
                model_name="XGBoost" if "XGBoost" in artifacts.models else "MLP",
                model=artifacts.models["XGBoost"]
                if "XGBoost" in artifacts.models
                else artifacts.models["MLP"],
                x_single=X_test_scaled[target_sample_idx],
                X_background=X_train_scaled,
                feature_names=artifacts.feature_names,
                class_idx=target_cls_idx,
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

    full_summary = {
        "dataset_source": prep_audit.get("source", {}),
        "preprocessing_audit": prep_audit,
        "training_diagnostics": train_diag,
        "supervised_evaluations": supervised_evals,
        "ensemble_comparison": ensemble_comparison,
        "autoencoder_evaluation": ae_eval,
        "unseen_attack_protocol": unseen_res,
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
    args = parser.parse_args()
    run_evaluation_suite(skip_unseen=args.skip_unseen, skip_shap=args.skip_shap)


if __name__ == "__main__":
    main()
