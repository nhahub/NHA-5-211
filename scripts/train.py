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
from src.training.trainer import train_all_models
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
        "--run-eval",
        action="store_true",
        help="Immediately run full evaluation, latency benchmarking, and SHAP reports after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_cfg = get_app_config().model_copy(deep=True)
    model_cfg = get_model_config().model_copy(deep=True)

    if args.max_rows is not None:
        app_cfg.data.max_rows_sample = args.max_rows
        app_cfg.data.synthetic_fallback.n_samples = args.max_rows

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
    registry.save_manifest(
        {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dataset_source": dataset.source_info.to_dict(),
            "imbalance_strategy_selected": outputs.imbalance_experiment.selected_strategy,
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
