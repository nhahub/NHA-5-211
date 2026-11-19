# SMART IDS v3 — Automated Verification & Test Report

> Generated automatically by `scripts/generate_test_report.py` on `2026-11-19 18:49 UTC`.

## 1. Static Analysis & Formatting (`ruff 0.17.5`)
- **Command:** `ruff check src dashboard scripts tests app.py`
  - **Status:** `PASSED` (`exit=0`, `0.10s`) — `All checks passed!`
- **Command:** `ruff format --check src dashboard scripts tests app.py`
  - **Status:** `PASSED` (`exit=0`, `0.10s`) — `65 files already formatted`

---

## 2. Automated Test Suite (`pytest 9.2.0`)
- **Command:** `pytest tests/ -v --durations=0 --tb=short`
- **Summary:** **`42` collected — `42 passed, 0 failed, 0 skipped` (`110.33s` total, 42 passed / 0 failed / 0 error / 0 skipped)**

| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |
|---:|---|---|---|---:|---:|
| 1 | `tests/test_dashboard_components.py` | `test_dashboard_charts_and_badges_render_cleanly` | PASSED | 0.27 | 0.27 |
| 2 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_smoke_with_existing_artifacts` | PASSED | 3.27 | 3.27 |
| 3 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_graceful_degradation_without_models` | PASSED | 1.27 | 1.27 |
| 4 | `tests/test_data_pipeline.py` | `test_discover_csv_files_and_whitespace_stripping` | PASSED | 0.03 | 0.04 |
| 5 | `tests/test_data_pipeline.py` | `test_cleaning_inf_nan_duplicates_and_web_attack_mojibake` | PASSED | 0.02 | 0.02 |
| 6 | `tests/test_data_pipeline.py` | `test_schema_alignment_errors_and_coercion` | PASSED | 0.02 | 0.02 |
| 7 | `tests/test_data_pipeline.py` | `test_leakage_free_preprocessing_and_dynamic_smote_k` | PASSED | 0.11 | 0.11 |
| 8 | `tests/test_data_pipeline.py` | `test_post_clip_near_duplicate_removal_and_strict_leakage_error` | PASSED | 0.17 | 0.17 |
| 9 | `tests/test_data_pipeline.py` | `test_by_day_split_and_drop_features` | PASSED | 0.17 | 0.17 |
| 10 | `tests/test_data_pipeline.py` | `test_smote_multiplier_cap_and_min_real_samples_threshold` | PASSED | 0.00 | 0.00 |
| 11 | `tests/test_data_pipeline.py` | `test_time_block_split_timestamp_ordering_no_leakage_and_fallbacks` | PASSED | 0.09 | 0.09 |
| 12 | `tests/test_data_pipeline.py` | `test_real_timestamp_alignment_and_missing_dir_warning` | PASSED | 0.21 | 0.22 |
| 13 | `tests/test_data_pipeline.py` | `test_sample_size_config_label_correction_and_sensitivity_protocol` | PASSED | 0.74 | 0.74 |
| 14 | `tests/test_models_and_inference.py` | `test_all_four_models_build_train_save_reload_and_infer` | PASSED | 5.73 | 5.73 |
| 15 | `tests/test_models_and_inference.py` | `test_hybrid_ensemble_zero_day_override_and_missing_artifacts` | PASSED | 0.00 | 0.00 |
| 16 | `tests/test_models_and_inference.py` | `test_by_day_protocol_feature_ablation_and_report_tables` | PASSED | 14.54 | 14.55 |
| 17 | `tests/test_models_and_inference.py` | `test_imbalance_selection_bootstrap_ci_per_class_and_xgb_cv` | PASSED | 0.90 | 0.90 |
| 18 | `tests/test_models_and_inference.py` | `test_temperature_scaler_handles_float32_overconfidence` | PASSED | 0.01 | 0.01 |
| 19 | `tests/test_models_and_inference.py` | `test_optuna_cv_std_gating_and_stacking_meta_learner` | PASSED | 0.18 | 0.18 |
| 20 | `tests/test_models_and_inference.py` | `test_classical_anomaly_baselines_and_unified_evaluation` | PASSED | 0.25 | 0.25 |
| 21 | `tests/test_models_and_inference.py` | `test_paired_bootstrap_significance_and_report_conclusion` | PASSED | 0.03 | 0.03 |
| 22 | `tests/test_models_and_inference.py` | `test_train_all_models_unified_cv_regression` | PASSED | 41.76 | 41.76 |
| 23 | `tests/test_models_and_inference.py` | `test_imbalance_tie_breaking_with_synthetic_metric_dicts` | PASSED | 0.00 | 0.00 |
| 24 | `tests/test_models_and_inference.py` | `test_per_model_imbalance_selection_caps_epochs_and_reports` | PASSED | 16.57 | 16.57 |
| 25 | `tests/test_models_and_inference.py` | `test_small_class_reliability_oof_wilson_and_reliable_macro_f1` | PASSED | 0.75 | 0.75 |
| 26 | `tests/test_models_and_inference.py` | `test_time_block_failure_analysis_ks_drift_and_mitigation_tradeoffs` | PASSED | 0.03 | 0.03 |
| 27 | `tests/test_models_and_inference.py` | `test_deep_ae_vae_and_leave_attack_out_score_fusion` | PASSED | 5.61 | 5.62 |
| 28 | `tests/test_models_and_inference.py` | `test_optuna_budget_and_bounded_neural_loss_keep_revert` | PASSED | 6.73 | 6.73 |
| 29 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_partial_and_corrupt_artifacts` | PASSED | 0.15 | 0.15 |
| 30 | `tests/test_repo_and_edge_cases.py` | `test_batch_csv_scoring_schema_validation_and_reordering` | PASSED | 0.21 | 0.21 |
| 31 | `tests/test_repo_and_edge_cases.py` | `test_split_strategy_validation_and_error_paths` | PASSED | 0.19 | 0.20 |
| 32 | `tests/test_repo_and_edge_cases.py` | `test_direct_single_flow_benchmark_and_onnx_export_script` | PASSED | 0.93 | 0.93 |
| 33 | `tests/test_repo_and_edge_cases.py` | `test_ablation_preserves_exact_rows_even_with_port_only_differences` | PASSED | 0.54 | 0.55 |
| 34 | `tests/test_repo_and_edge_cases.py` | `test_latency_warmup_config_and_test_report_generator_parser` | PASSED | 0.01 | 0.01 |
| 35 | `tests/test_repo_and_edge_cases.py` | `test_manifest_sha256_match_mismatch_missing_and_lfs_skip` | PASSED | 0.02 | 0.02 |
| 36 | `tests/test_repo_and_edge_cases.py` | `test_download_models_fallback_and_hash_verification` | PASSED | 0.02 | 0.02 |
| 37 | `tests/test_repo_and_edge_cases.py` | `test_real_lfs_artifact_smoke` | PASSED | 0.19 | 0.19 |
| 38 | `tests/test_response_and_threat.py` | `test_threat_level_boundaries_and_benign_safety` | PASSED | 0.00 | 0.00 |
| 39 | `tests/test_response_and_threat.py` | `test_ip_validation_prevents_command_injection` | PASSED | 0.00 | 0.00 |
| 40 | `tests/test_response_and_threat.py` | `test_active_response_dry_run_idempotency_and_audit_log` | PASSED | 0.00 | 0.00 |
| 41 | `tests/test_response_and_threat.py` | `test_local_shap_explanation_xgboost` | PASSED | 0.30 | 0.30 |
| 42 | `tests/test_response_and_threat.py` | `test_response_command_previews_and_protected_ip_safeguards` | PASSED | 0.02 | 0.02 |
