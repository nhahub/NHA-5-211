# SMART IDS v3 — Automated Verification & Test Report

> Generated automatically by `scripts/generate_test_report.py` on `2026-11-21 10:54 UTC`.

## 1. Static Analysis & Formatting (`ruff 0.17.5`)
- **Command:** `ruff check src dashboard scripts tests app.py`
  - **Status:** `PASSED` (`exit=0`, `0.13s`) — `All checks passed!`
- **Command:** `ruff format --check src dashboard scripts tests app.py`
  - **Status:** `PASSED` (`exit=0`, `0.16s`) — `67 files already formatted`

---

## 2. CI Verification Matrix (Real Artifacts vs. LFS Pointer Stub Simulation)

| Suite Mode | Collected | Passed | Failed | Error | Skipped | Duration_s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Full Artifacts (local binaries) | 51 | 51 | 0 | 0 | 0 | 912.90 | PASSED |
| LFS Pointer Stubs (`lfs: false` mirror, 15 stubs) | 51 | 50 | 0 | 0 | 1 | 835.81 | PASSED |

---

## 3. Full-Artifact Test Suite (`pytest 9.2.0`)
- **Command:** `pytest tests/ -v --durations=0 --tb=short`
- **Summary:** **`51` collected — `51 passed` (`912.90s` total, 51 passed / 0 failed / 0 error / 0 skipped)**

| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |
|---:|---|---|---|---:|---:|
| 1 | `tests/test_dashboard_components.py` | `test_dashboard_charts_and_badges_render_cleanly` | PASSED | 1.90 | 1.90 |
| 2 | `tests/test_dashboard_components.py` | `test_model_artifact_summary_counts_and_hybrid_readiness` | PASSED | 0.01 | 0.01 |
| 3 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_smoke_with_existing_artifacts` | PASSED | 25.60 | 25.60 |
| 4 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_graceful_degradation_without_models` | PASSED | 8.60 | 8.62 |
| 5 | `tests/test_data_pipeline.py` | `test_discover_csv_files_and_whitespace_stripping` | PASSED | 0.18 | 0.27 |
| 6 | `tests/test_data_pipeline.py` | `test_cleaning_inf_nan_duplicates_and_web_attack_mojibake` | PASSED | 0.14 | 0.14 |
| 7 | `tests/test_data_pipeline.py` | `test_schema_alignment_errors_and_coercion` | PASSED | 0.18 | 0.18 |
| 8 | `tests/test_data_pipeline.py` | `test_leakage_free_preprocessing_and_dynamic_smote_k` | PASSED | 0.76 | 0.79 |
| 9 | `tests/test_data_pipeline.py` | `test_post_clip_near_duplicate_removal_and_strict_leakage_error` | PASSED | 1.09 | 1.12 |
| 10 | `tests/test_data_pipeline.py` | `test_by_day_split_and_drop_features` | PASSED | 0.93 | 0.95 |
| 11 | `tests/test_data_pipeline.py` | `test_smote_multiplier_cap_and_min_real_samples_threshold` | PASSED | 0.02 | 0.02 |
| 12 | `tests/test_data_pipeline.py` | `test_time_block_split_timestamp_ordering_no_leakage_and_fallbacks` | PASSED | 0.73 | 0.75 |
| 13 | `tests/test_data_pipeline.py` | `test_real_timestamp_alignment_and_missing_dir_warning` | PASSED | 1.17 | 1.20 |
| 14 | `tests/test_data_pipeline.py` | `test_sample_size_config_label_correction_and_sensitivity_protocol` | PASSED | 4.10 | 4.14 |
| 15 | `tests/test_models_and_inference.py` | `test_all_four_models_build_train_save_reload_and_infer` | PASSED | 42.68 | 42.71 |
| 16 | `tests/test_models_and_inference.py` | `test_hybrid_ensemble_zero_day_override_and_missing_artifacts` | PASSED | 0.01 | 0.02 |
| 17 | `tests/test_models_and_inference.py` | `test_by_day_protocol_feature_ablation_and_report_tables` | PASSED | 82.65 | 82.68 |
| 18 | `tests/test_models_and_inference.py` | `test_imbalance_selection_bootstrap_ci_per_class_and_xgb_cv` | PASSED | 4.71 | 4.74 |
| 19 | `tests/test_models_and_inference.py` | `test_temperature_scaler_handles_float32_overconfidence` | PASSED | 0.06 | 0.06 |
| 20 | `tests/test_models_and_inference.py` | `test_optuna_cv_std_gating_and_stacking_meta_learner` | PASSED | 2.73 | 2.74 |
| 21 | `tests/test_models_and_inference.py` | `test_classical_anomaly_baselines_and_unified_evaluation` | PASSED | 1.55 | 1.56 |
| 22 | `tests/test_models_and_inference.py` | `test_paired_bootstrap_significance_and_report_conclusion` | PASSED | 0.27 | 0.28 |
| 23 | `tests/test_models_and_inference.py` | `test_train_all_models_unified_cv_regression` | PASSED | 242.30 | 242.33 |
| 24 | `tests/test_models_and_inference.py` | `test_imbalance_tie_breaking_with_synthetic_metric_dicts` | PASSED | 0.00 | 0.00 |
| 25 | `tests/test_models_and_inference.py` | `test_per_model_imbalance_selection_caps_epochs_and_reports` | PASSED | 96.83 | 96.84 |
| 26 | `tests/test_models_and_inference.py` | `test_small_class_reliability_oof_wilson_and_reliable_macro_f1` | PASSED | 3.70 | 3.73 |
| 27 | `tests/test_models_and_inference.py` | `test_time_block_failure_analysis_ks_drift_and_mitigation_tradeoffs` | PASSED | 0.30 | 0.33 |
| 28 | `tests/test_models_and_inference.py` | `test_deep_ae_vae_and_leave_attack_out_score_fusion` | PASSED | 24.68 | 24.72 |
| 29 | `tests/test_models_and_inference.py` | `test_optuna_budget_and_bounded_neural_loss_keep_revert` | PASSED | 38.23 | 38.24 |
| 30 | `tests/test_models_and_inference.py` | `test_degenerate_fusion_labeling_and_na_unmeasured_values` | PASSED | 0.07 | 0.08 |
| 31 | `tests/test_models_and_inference.py` | `test_time_aware_xgboost_profile_selection_validation_only_and_reporting` | PASSED | 1.78 | 1.80 |
| 32 | `tests/test_models_and_inference.py` | `test_xgboost_downstream_refit_validation_only_keeps_mlp_cnn_calibrators` | PASSED | 281.42 | 281.45 |
| 33 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_partial_and_corrupt_artifacts` | PASSED | 1.11 | 1.15 |
| 34 | `tests/test_repo_and_edge_cases.py` | `test_batch_csv_scoring_schema_validation_and_reordering` | PASSED | 1.27 | 1.29 |
| 35 | `tests/test_repo_and_edge_cases.py` | `test_split_strategy_validation_and_error_paths` | PASSED | 1.13 | 1.16 |
| 36 | `tests/test_repo_and_edge_cases.py` | `test_direct_single_flow_benchmark_and_onnx_export_script` | PASSED | 6.89 | 6.90 |
| 37 | `tests/test_repo_and_edge_cases.py` | `test_ablation_preserves_exact_rows_even_with_port_only_differences` | PASSED | 2.63 | 2.64 |
| 38 | `tests/test_repo_and_edge_cases.py` | `test_latency_warmup_config_and_test_report_generator_parser` | PASSED | 0.15 | 0.15 |
| 39 | `tests/test_repo_and_edge_cases.py` | `test_manifest_sha256_match_mismatch_missing_and_lfs_skip` | PASSED | 0.18 | 0.19 |
| 40 | `tests/test_repo_and_edge_cases.py` | `test_download_models_fallback_and_hash_verification` | PASSED | 0.18 | 0.19 |
| 41 | `tests/test_repo_and_edge_cases.py` | `test_real_lfs_artifact_smoke` | PASSED | 1.00 | 1.00 |
| 42 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_stub_mode_graceful_degradation` | PASSED | 0.22 | 0.23 |
| 43 | `tests/test_repo_and_edge_cases.py` | `test_no_absolute_machine_paths_in_artifacts_and_sanitizer` | PASSED | 0.07 | 0.08 |
| 44 | `tests/test_repo_and_edge_cases.py` | `test_check_optional_data_present_and_missing` | PASSED | 0.04 | 0.05 |
| 45 | `tests/test_repo_and_edge_cases.py` | `test_readme_numbers_match_generated_reports_and_detect_drift` | PASSED | 0.07 | 0.08 |
| 46 | `tests/test_repo_and_edge_cases.py` | `test_deployment_profile_config_registry_manifest_and_only_xgboost` | PASSED | 2.09 | 2.11 |
| 47 | `tests/test_response_and_threat.py` | `test_threat_level_boundaries_and_benign_safety` | PASSED | 0.00 | 0.00 |
| 48 | `tests/test_response_and_threat.py` | `test_ip_validation_prevents_command_injection` | PASSED | 0.00 | 0.00 |
| 49 | `tests/test_response_and_threat.py` | `test_active_response_dry_run_idempotency_and_audit_log` | PASSED | 0.04 | 0.05 |
| 50 | `tests/test_response_and_threat.py` | `test_local_shap_explanation_xgboost` | PASSED | 5.98 | 6.00 |
| 51 | `tests/test_response_and_threat.py` | `test_response_command_previews_and_protected_ip_safeguards` | PASSED | 0.11 | 0.13 |

---

## 4. LFS Pointer Stub Simulation Suite (`pytest 9.2.0`, `lfs: false` Mirror)
- **Command:** `pytest tests/ -v --durations=0 --tb=short (stub-mode simulation)` (`15` LFS binaries replaced by pointer stubs)
- **Summary:** **`51` collected — `50 passed, 1 skipped` (`835.81s` total, 50 passed / 0 failed / 0 error / 1 skipped)**
- **Skipped Tests (stub detection):** `tests/test_repo_and_edge_cases.py::test_real_lfs_artifact_smoke`

| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |
|---:|---|---|---|---:|---:|
| 1 | `tests/test_dashboard_components.py` | `test_dashboard_charts_and_badges_render_cleanly` | PASSED | 1.82 | 1.82 |
| 2 | `tests/test_dashboard_components.py` | `test_model_artifact_summary_counts_and_hybrid_readiness` | PASSED | 0.07 | 0.07 |
| 3 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_smoke_with_existing_artifacts` | PASSED | 14.11 | 14.11 |
| 4 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_graceful_degradation_without_models` | PASSED | 6.80 | 6.83 |
| 5 | `tests/test_data_pipeline.py` | `test_discover_csv_files_and_whitespace_stripping` | PASSED | 0.11 | 0.18 |
| 6 | `tests/test_data_pipeline.py` | `test_cleaning_inf_nan_duplicates_and_web_attack_mojibake` | PASSED | 0.13 | 0.13 |
| 7 | `tests/test_data_pipeline.py` | `test_schema_alignment_errors_and_coercion` | PASSED | 0.13 | 0.13 |
| 8 | `tests/test_data_pipeline.py` | `test_leakage_free_preprocessing_and_dynamic_smote_k` | PASSED | 0.56 | 0.58 |
| 9 | `tests/test_data_pipeline.py` | `test_post_clip_near_duplicate_removal_and_strict_leakage_error` | PASSED | 0.94 | 0.96 |
| 10 | `tests/test_data_pipeline.py` | `test_by_day_split_and_drop_features` | PASSED | 0.84 | 0.87 |
| 11 | `tests/test_data_pipeline.py` | `test_smote_multiplier_cap_and_min_real_samples_threshold` | PASSED | 0.02 | 0.02 |
| 12 | `tests/test_data_pipeline.py` | `test_time_block_split_timestamp_ordering_no_leakage_and_fallbacks` | PASSED | 0.57 | 0.60 |
| 13 | `tests/test_data_pipeline.py` | `test_real_timestamp_alignment_and_missing_dir_warning` | PASSED | 1.02 | 1.04 |
| 14 | `tests/test_data_pipeline.py` | `test_sample_size_config_label_correction_and_sensitivity_protocol` | PASSED | 3.43 | 3.44 |
| 15 | `tests/test_models_and_inference.py` | `test_all_four_models_build_train_save_reload_and_infer` | PASSED | 37.31 | 37.34 |
| 16 | `tests/test_models_and_inference.py` | `test_hybrid_ensemble_zero_day_override_and_missing_artifacts` | PASSED | 0.01 | 0.02 |
| 17 | `tests/test_models_and_inference.py` | `test_by_day_protocol_feature_ablation_and_report_tables` | PASSED | 87.01 | 87.04 |
| 18 | `tests/test_models_and_inference.py` | `test_imbalance_selection_bootstrap_ci_per_class_and_xgb_cv` | PASSED | 4.62 | 4.65 |
| 19 | `tests/test_models_and_inference.py` | `test_temperature_scaler_handles_float32_overconfidence` | PASSED | 0.02 | 0.02 |
| 20 | `tests/test_models_and_inference.py` | `test_optuna_cv_std_gating_and_stacking_meta_learner` | PASSED | 2.60 | 2.61 |
| 21 | `tests/test_models_and_inference.py` | `test_classical_anomaly_baselines_and_unified_evaluation` | PASSED | 1.36 | 1.38 |
| 22 | `tests/test_models_and_inference.py` | `test_paired_bootstrap_significance_and_report_conclusion` | PASSED | 0.09 | 0.09 |
| 23 | `tests/test_models_and_inference.py` | `test_train_all_models_unified_cv_regression` | PASSED | 224.02 | 224.04 |
| 24 | `tests/test_models_and_inference.py` | `test_imbalance_tie_breaking_with_synthetic_metric_dicts` | PASSED | 0.00 | 0.00 |
| 25 | `tests/test_models_and_inference.py` | `test_per_model_imbalance_selection_caps_epochs_and_reports` | PASSED | 88.46 | 88.47 |
| 26 | `tests/test_models_and_inference.py` | `test_small_class_reliability_oof_wilson_and_reliable_macro_f1` | PASSED | 2.13 | 2.14 |
| 27 | `tests/test_models_and_inference.py` | `test_time_block_failure_analysis_ks_drift_and_mitigation_tradeoffs` | PASSED | 0.10 | 0.11 |
| 28 | `tests/test_models_and_inference.py` | `test_deep_ae_vae_and_leave_attack_out_score_fusion` | PASSED | 34.89 | 34.90 |
| 29 | `tests/test_models_and_inference.py` | `test_optuna_budget_and_bounded_neural_loss_keep_revert` | PASSED | 36.73 | 36.74 |
| 30 | `tests/test_models_and_inference.py` | `test_degenerate_fusion_labeling_and_na_unmeasured_values` | PASSED | 0.05 | 0.05 |
| 31 | `tests/test_models_and_inference.py` | `test_time_aware_xgboost_profile_selection_validation_only_and_reporting` | PASSED | 0.92 | 0.93 |
| 32 | `tests/test_models_and_inference.py` | `test_xgboost_downstream_refit_validation_only_keeps_mlp_cnn_calibrators` | PASSED | 247.61 | 247.62 |
| 33 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_partial_and_corrupt_artifacts` | PASSED | 1.02 | 1.05 |
| 34 | `tests/test_repo_and_edge_cases.py` | `test_batch_csv_scoring_schema_validation_and_reordering` | PASSED | 1.46 | 1.50 |
| 35 | `tests/test_repo_and_edge_cases.py` | `test_split_strategy_validation_and_error_paths` | PASSED | 1.03 | 1.06 |
| 36 | `tests/test_repo_and_edge_cases.py` | `test_direct_single_flow_benchmark_and_onnx_export_script` | PASSED | 6.58 | 6.59 |
| 37 | `tests/test_repo_and_edge_cases.py` | `test_ablation_preserves_exact_rows_even_with_port_only_differences` | PASSED | 2.62 | 2.65 |
| 38 | `tests/test_repo_and_edge_cases.py` | `test_latency_warmup_config_and_test_report_generator_parser` | PASSED | 0.13 | 0.13 |
| 39 | `tests/test_repo_and_edge_cases.py` | `test_manifest_sha256_match_mismatch_missing_and_lfs_skip` | PASSED | 0.16 | 0.17 |
| 40 | `tests/test_repo_and_edge_cases.py` | `test_download_models_fallback_and_hash_verification` | PASSED | 0.10 | 0.11 |
| 41 | `tests/test_repo_and_edge_cases.py` | `test_real_lfs_artifact_smoke` | SKIPPED | 0.00 | 0.00 |
| 42 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_stub_mode_graceful_degradation` | PASSED | 0.19 | 0.20 |
| 43 | `tests/test_repo_and_edge_cases.py` | `test_no_absolute_machine_paths_in_artifacts_and_sanitizer` | PASSED | 0.15 | 0.16 |
| 44 | `tests/test_repo_and_edge_cases.py` | `test_check_optional_data_present_and_missing` | PASSED | 0.05 | 0.06 |
| 45 | `tests/test_repo_and_edge_cases.py` | `test_readme_numbers_match_generated_reports_and_detect_drift` | PASSED | 0.12 | 0.13 |
| 46 | `tests/test_repo_and_edge_cases.py` | `test_deployment_profile_config_registry_manifest_and_only_xgboost` | PASSED | 2.08 | 2.10 |
| 47 | `tests/test_response_and_threat.py` | `test_threat_level_boundaries_and_benign_safety` | PASSED | 0.00 | 0.00 |
| 48 | `tests/test_response_and_threat.py` | `test_ip_validation_prevents_command_injection` | PASSED | 0.00 | 0.00 |
| 49 | `tests/test_response_and_threat.py` | `test_active_response_dry_run_idempotency_and_audit_log` | PASSED | 0.02 | 0.03 |
| 50 | `tests/test_response_and_threat.py` | `test_local_shap_explanation_xgboost` | PASSED | 4.32 | 4.33 |
| 51 | `tests/test_response_and_threat.py` | `test_response_command_previews_and_protected_ip_safeguards` | PASSED | 0.03 | 0.04 |
