"""CLI entry point for end-to-end SMART IDS v3 preprocessing and model training."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_app_config, get_model_config
from src.data.preprocessing import run_preprocessing_pipeline
from src.evaluation.visualizations import plot_class_distributions, plot_training_histories
from src.training.trainer import train_all_models, train_xgboost_only_profiles
from src.utils.logging import get_logger
from src.utils.model_registry import ModelRegistry

logger = get_logger("smart_ids.scripts.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMART IDS v3 models end-to-end.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicitly run in synthetic demonstration mode even if raw CSVs exist.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override max training epochs for Keras models."
    )
    parser.add_argument("--optuna-trials", type=int, default=None, help="Override Optuna trials for XGBoost.")
    parser.add_argument("--max-rows", type=int, default=None, help="Override dataset sample limit.")
    parser.add_argument(
        "--split-strategy",
        choices=["random", "by_day", "time_block"],
        default=None,
        help="Override data.split_strategy from configs/config.yaml.",
    )
    parser.add_argument(
        "--only-xgboost",
        action="store_true",
        help="Train ONLY XGBoost across specified profiles while keeping MLP/1D-CNN/Autoencoder untouched.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="none,class_weight",
        help="Comma-separated XGBoost profiles to train when --only-xgboost is set (e.g. none,class_weight).",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Immediately run full evaluation, latency benchmarking, and SHAP reports after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_cfg = get_app_config().model_copy(deep=True)
    model_cfg = get_model_config().model_copy(deep=True)

    if args.split_strategy is not None:
        app_cfg.data.split_strategy = args.split_strategy
    if args.max_rows is not None:
        app_cfg.data.sample_size = args.max_rows
        app_cfg.data.max_rows_sample = args.max_rows
        app_cfg.data.synthetic_fallback.n_samples = args.max_rows

    if args.only_xgboost:
        req_profiles = [p.strip() for p in str(args.profiles).split(",") if p.strip()]
        xgb_res = train_xgboost_only_profiles(
            app_cfg=app_cfg,
            model_cfg=model_cfg,
            profiles=req_profiles or ["none", "class_weight"],
            optuna_trials_override=args.optuna_trials,
            force_synthetic=args.synthetic,
        )
        logger.info(
            "XGBoost-only profile training completed (active='%s', files=%s).",
            xgb_res["active_profile"],
            xgb_res["profile_files"],
        )
        if args.run_eval:
            from scripts.evaluate import run_evaluation_suite

            run_evaluation_suite(app_cfg=app_cfg, model_cfg=model_cfg)
        return

    logger.info("Starting SMART IDS v3 pipeline...")
    dataset = run_preprocessing_pipeline(app_cfg, force_synthetic=args.synthetic, save_outputs=True)

    figures_dir = app_cfg.paths.resolve("figures_dir")
    plot_class_distributions(
        pre_counts=dataset.audit_report["train_pre_smote_distribution"],
        post_counts=dataset.audit_report["train_post_smote_distribution"],
        out_path=figures_dir / "class_distribution.png",
    )

    outputs = train_all_models(
        dataset=dataset,
        app_cfg=app_cfg,
        model_cfg=model_cfg,
        epochs_override=args.epochs,
        optuna_trials_override=args.optuna_trials,
    )

    plot_training_histories(outputs.histories, out_path=figures_dir / "training_histories.png")

    registry = ModelRegistry(
        models_dir=app_cfg.paths.resolve("models_dir"),
        artifacts_dir=app_cfg.paths.resolve("artifacts_dir"),
    )
    registry.save_artifact_manifest()
    registry.save_manifest(
        {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dataset_source": dataset.source_info.to_dict(),
            "imbalance_strategy_selected": outputs.imbalance_experiment.selected_strategy,
            "imbalance_strategies_by_model": outputs.imbalance_strategies_by_model,
            "xgb_selected_variant": outputs.xgb_optimization.selected_variant,
            "best_supervised_model": outputs.best_supervised_model,
            "training_durations_sec": outputs.training_durations_sec,
            "anomaly_threshold": outputs.anomaly_threshold,
            "hardware_profile": {
                "tf_device": outputs.hardware_profile.tf_device,
                "xgb_device": outputs.hardware_profile.xgb_device,
            },
        }
    )
    logger.info("Training completed successfully. Artifacts saved to %s", app_cfg.paths.resolve("models_dir"))

    if args.run_eval:
        from scripts.evaluate import run_evaluation_suite

        run_evaluation_suite(app_cfg=app_cfg, model_cfg=model_cfg)


if __name__ == "__main__":
    main()
