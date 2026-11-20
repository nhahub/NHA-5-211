# SMART IDS v3 — Automated Verification & Test Report

> Generated automatically by `scripts/generate_test_report.py` on `2026-11-20 23:30 UTC`.

## 1. Static Analysis & Formatting (`ruff 0.17.5`)
- **Command:** `ruff check src dashboard scripts tests app.py`
  - **Status:** `PASSED` (`exit=0`, `0.11s`) — `All checks passed!`
- **Command:** `ruff format --check src dashboard scripts tests app.py`
  - **Status:** `PASSED` (`exit=0`, `0.12s`) — `67 files already formatted`

---

## 2. CI Verification Matrix (Real Artifacts vs. LFS Pointer Stub Simulation)

| Suite Mode | Collected | Passed | Failed | Error | Skipped | Duration_s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Full Artifacts (local binaries) | 51 | 51 | 0 | 0 | 0 | 555.18 | PASSED |
| LFS Pointer Stubs (`lfs: false` mirror, 15 stubs) | 51 | 50 | 0 | 0 | 1 | 536.89 | PASSED |

---

## 3. Full-Artifact Test Suite (`pytest 9.2.0`)
- **Command:** `pytest tests/ -v --durations=0 --tb=short`
- **Summary:** **`51` collected — `51 passed, 945 warnings` (`555.18s` total, 51 passed / 0 failed / 0 error / 0 skipped)**

| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |
|---:|---|---|---|---:|---:|
| 1 | `tests/test_dashboard_components.py` | `test_dashboard_charts_and_badges_render_cleanly` | PASSED | 0.98 | 0.98 |
| 2 | `tests/test_dashboard_components.py` | `test_model_artifact_summary_counts_and_hybrid_readiness` | PASSED | 0.00 | 0.00 |
| 3 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_smoke_with_existing_artifacts` | PASSED | 12.30 | 12.30 |
| 4 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_graceful_degradation_without_models` | PASSED | 4.00 | 4.01 |
| 5 | `tests/test_data_pipeline.py` | `test_discover_csv_files_and_whitespace_stripping` | PASSED | 0.09 | 0.13 |
| 6 | `tests/test_data_pipeline.py` | `test_cleaning_inf_nan_duplicates_and_web_attack_mojibake` | PASSED | 0.07 | 0.07 |
| 7 | `tests/test_data_pipeline.py` | `test_schema_alignment_errors_and_coercion` | PASSED | 0.07 | 0.07 |
| 8 | `tests/test_data_pipeline.py` | `test_leakage_free_preprocessing_and_dynamic_smote_k` | PASSED | 0.34 | 0.35 |
| 9 | `tests/test_data_pipeline.py` | `test_post_clip_near_duplicate_removal_and_strict_leakage_error` | PASSED | 0.53 | 0.54 |
| 10 | `tests/test_data_pipeline.py` | `test_by_day_split_and_drop_features` | PASSED | 0.43 | 0.44 |
| 11 | `tests/test_data_pipeline.py` | `test_smote_multiplier_cap_and_min_real_samples_threshold` | PASSED | 0.01 | 0.01 |
| 12 | `tests/test_data_pipeline.py` | `test_time_block_split_timestamp_ordering_no_leakage_and_fallbacks` | PASSED | 0.32 | 0.33 |
| 13 | `tests/test_data_pipeline.py` | `test_real_timestamp_alignment_and_missing_dir_warning` | PASSED | 0.63 | 0.64 |
| 14 | `tests/test_data_pipeline.py` | `test_sample_size_config_label_correction_and_sensitivity_protocol` | PASSED | 1.89 | 1.91 |
| 15 | `tests/test_models_and_inference.py` | `test_all_four_models_build_train_save_reload_and_infer` | PASSED | 20.19 | 20.20 |
| 16 | `tests/test_models_and_inference.py` | `test_hybrid_ensemble_zero_day_override_and_missing_artifacts` | PASSED | 0.01 | 0.02 |
| 17 | `tests/test_models_and_inference.py` | `test_by_day_protocol_feature_ablation_and_report_tables` | PASSED | 53.05 | 53.07 |
| 18 | `tests/test_models_and_inference.py` | `test_imbalance_selection_bootstrap_ci_per_class_and_xgb_cv` | PASSED | 2.42 | 2.43 |
| 19 | `tests/test_models_and_inference.py` | `test_temperature_scaler_handles_float32_overconfidence` | PASSED | 0.04 | 0.04 |
| 20 | `tests/test_models_and_inference.py` | `test_optuna_cv_std_gating_and_stacking_meta_learner` | PASSED | 1.26 | 1.27 |
| 21 | `tests/test_models_and_inference.py` | `test_classical_anomaly_baselines_and_unified_evaluation` | PASSED | 0.81 | 0.82 |
| 22 | `tests/test_models_and_inference.py` | `test_paired_bootstrap_significance_and_report_conclusion` | PASSED | 0.18 | 0.18 |
| 23 | `tests/test_models_and_inference.py` | `test_train_all_models_unified_cv_regression` | PASSED | 150.86 | 150.87 |
| 24 | `tests/test_models_and_inference.py` | `test_imbalance_tie_breaking_with_synthetic_metric_dicts` | PASSED | 0.00 | 0.00 |
| 25 | `tests/test_models_and_inference.py` | `test_per_model_imbalance_selection_caps_epochs_and_reports` | PASSED | 56.64 | 56.65 |
| 26 | `tests/test_models_and_inference.py` | `test_small_class_reliability_oof_wilson_and_reliable_macro_f1` | PASSED | 1.84 | 1.85 |
| 27 | `tests/test_models_and_inference.py` | `test_time_block_failure_analysis_ks_drift_and_mitigation_tradeoffs` | PASSED | 0.11 | 0.12 |
| 28 | `tests/test_models_and_inference.py` | `test_deep_ae_vae_and_leave_attack_out_score_fusion` | PASSED | 22.93 | 22.94 |
| 29 | `tests/test_models_and_inference.py` | `test_optuna_budget_and_bounded_neural_loss_keep_revert` | PASSED | 24.03 | 24.04 |
| 30 | `tests/test_models_and_inference.py` | `test_degenerate_fusion_labeling_and_na_unmeasured_values` | PASSED | 0.02 | 0.02 |
| 31 | `tests/test_models_and_inference.py` | `test_time_aware_xgboost_profile_selection_validation_only_and_reporting` | PASSED | 0.77 | 0.78 |
| 32 | `tests/test_models_and_inference.py` | `test_xgboost_downstream_refit_validation_only_keeps_mlp_cnn_calibrators` | PASSED | 178.88 | 178.89 |
| 33 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_partial_and_corrupt_artifacts` | PASSED | 0.52 | 0.53 |
| 34 | `tests/test_repo_and_edge_cases.py` | `test_batch_csv_scoring_schema_validation_and_reordering` | PASSED | 2.30 | 2.31 |
| 35 | `tests/test_repo_and_edge_cases.py` | `test_split_strategy_validation_and_error_paths` | PASSED | 0.60 | 0.61 |
| 36 | `tests/test_repo_and_edge_cases.py` | `test_direct_single_flow_benchmark_and_onnx_export_script` | PASSED | 3.54 | 3.54 |
| 37 | `tests/test_repo_and_edge_cases.py` | `test_ablation_preserves_exact_rows_even_with_port_only_differences` | PASSED | 1.44 | 1.45 |
| 38 | `tests/test_repo_and_edge_cases.py` | `test_latency_warmup_config_and_test_report_generator_parser` | PASSED | 0.06 | 0.06 |
| 39 | `tests/test_repo_and_edge_cases.py` | `test_manifest_sha256_match_mismatch_missing_and_lfs_skip` | PASSED | 0.08 | 0.09 |
| 40 | `tests/test_repo_and_edge_cases.py` | `test_download_models_fallback_and_hash_verification` | PASSED | 0.04 | 0.04 |
| 41 | `tests/test_repo_and_edge_cases.py` | `test_real_lfs_artifact_smoke` | PASSED | 0.54 | 0.54 |
| 42 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_stub_mode_graceful_degradation` | PASSED | 0.07 | 0.07 |
| 43 | `tests/test_repo_and_edge_cases.py` | `test_no_absolute_machine_paths_in_artifacts_and_sanitizer` | PASSED | 0.05 | 0.05 |
| 44 | `tests/test_repo_and_edge_cases.py` | `test_check_optional_data_present_and_missing` | PASSED | 0.02 | 0.02 |
| 45 | `tests/test_repo_and_edge_cases.py` | `test_readme_numbers_match_generated_reports_and_detect_drift` | PASSED | 0.03 | 0.03 |
| 46 | `tests/test_repo_and_edge_cases.py` | `test_deployment_profile_config_registry_manifest_and_only_xgboost` | PASSED | 1.15 | 1.16 |
| 47 | `tests/test_response_and_threat.py` | `test_threat_level_boundaries_and_benign_safety` | PASSED | 0.00 | 0.00 |
| 48 | `tests/test_response_and_threat.py` | `test_ip_validation_prevents_command_injection` | PASSED | 0.00 | 0.00 |
| 49 | `tests/test_response_and_threat.py` | `test_active_response_dry_run_idempotency_and_audit_log` | PASSED | 0.02 | 0.02 |
| 50 | `tests/test_response_and_threat.py` | `test_local_shap_explanation_xgboost` | PASSED | 0.79 | 0.79 |
| 51 | `tests/test_response_and_threat.py` | `test_response_command_previews_and_protected_ip_safeguards` | PASSED | 0.02 | 0.03 |

---

## 4. LFS Pointer Stub Simulation Suite (`pytest 9.2.0`, `lfs: false` Mirror)
- **Command:** `pytest tests/ -v --durations=0 --tb=short (stub-mode simulation)` (`15` LFS binaries replaced by pointer stubs)
- **Summary:** **`51` collected — `50 passed, 1 skipped, 945 warnings` (`536.89s` total, 50 passed / 0 failed / 0 error / 1 skipped)**
- **Skipped Tests (stub detection):** `tests/test_repo_and_edge_cases.py::test_real_lfs_artifact_smoke`

| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |
|---:|---|---|---|---:|---:|
| 1 | `tests/test_dashboard_components.py` | `test_dashboard_charts_and_badges_render_cleanly` | PASSED | 0.97 | 0.97 |
| 2 | `tests/test_dashboard_components.py` | `test_model_artifact_summary_counts_and_hybrid_readiness` | PASSED | 0.02 | 0.02 |
| 3 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_smoke_with_existing_artifacts` | PASSED | 8.06 | 8.06 |
| 4 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_graceful_degradation_without_models` | PASSED | 4.06 | 4.07 |
| 5 | `tests/test_data_pipeline.py` | `test_discover_csv_files_and_whitespace_stripping` | PASSED | 0.09 | 0.13 |
| 6 | `tests/test_data_pipeline.py` | `test_cleaning_inf_nan_duplicates_and_web_attack_mojibake` | PASSED | 0.07 | 0.07 |
| 7 | `tests/test_data_pipeline.py` | `test_schema_alignment_errors_and_coercion` | PASSED | 0.07 | 0.07 |
| 8 | `tests/test_data_pipeline.py` | `test_leakage_free_preprocessing_and_dynamic_smote_k` | PASSED | 0.51 | 0.53 |
| 9 | `tests/test_data_pipeline.py` | `test_post_clip_near_duplicate_removal_and_strict_leakage_error` | PASSED | 0.90 | 0.92 |
| 10 | `tests/test_data_pipeline.py` | `test_by_day_split_and_drop_features` | PASSED | 0.52 | 0.53 |
| 11 | `tests/test_data_pipeline.py` | `test_smote_multiplier_cap_and_min_real_samples_threshold` | PASSED | 0.01 | 0.01 |
| 12 | `tests/test_data_pipeline.py` | `test_time_block_split_timestamp_ordering_no_leakage_and_fallbacks` | PASSED | 0.37 | 0.38 |
| 13 | `tests/test_data_pipeline.py` | `test_real_timestamp_alignment_and_missing_dir_warning` | PASSED | 0.68 | 0.69 |
| 14 | `tests/test_data_pipeline.py` | `test_sample_size_config_label_correction_and_sensitivity_protocol` | PASSED | 2.01 | 2.03 |
| 15 | `tests/test_models_and_inference.py` | `test_all_four_models_build_train_save_reload_and_infer` | PASSED | 20.59 | 20.60 |
| 16 | `tests/test_models_and_inference.py` | `test_hybrid_ensemble_zero_day_override_and_missing_artifacts` | PASSED | 0.01 | 0.01 |
| 17 | `tests/test_models_and_inference.py` | `test_by_day_protocol_feature_ablation_and_report_tables` | PASSED | 50.23 | 50.24 |
| 18 | `tests/test_models_and_inference.py` | `test_imbalance_selection_bootstrap_ci_per_class_and_xgb_cv` | PASSED | 2.44 | 2.45 |
| 19 | `tests/test_models_and_inference.py` | `test_temperature_scaler_handles_float32_overconfidence` | PASSED | 0.04 | 0.04 |
| 20 | `tests/test_models_and_inference.py` | `test_optuna_cv_std_gating_and_stacking_meta_learner` | PASSED | 1.20 | 1.21 |
| 21 | `tests/test_models_and_inference.py` | `test_classical_anomaly_baselines_and_unified_evaluation` | PASSED | 0.82 | 0.83 |
| 22 | `tests/test_models_and_inference.py` | `test_paired_bootstrap_significance_and_report_conclusion` | PASSED | 0.16 | 0.17 |
| 23 | `tests/test_models_and_inference.py` | `test_train_all_models_unified_cv_regression` | PASSED | 149.96 | 149.97 |
| 24 | `tests/test_models_and_inference.py` | `test_imbalance_tie_breaking_with_synthetic_metric_dicts` | PASSED | 0.00 | 0.00 |
| 25 | `tests/test_models_and_inference.py` | `test_per_model_imbalance_selection_caps_epochs_and_reports` | PASSED | 56.72 | 56.73 |
| 26 | `tests/test_models_and_inference.py` | `test_small_class_reliability_oof_wilson_and_reliable_macro_f1` | PASSED | 1.95 | 1.96 |
| 27 | `tests/test_models_and_inference.py` | `test_time_block_failure_analysis_ks_drift_and_mitigation_tradeoffs` | PASSED | 0.12 | 0.14 |
| 28 | `tests/test_models_and_inference.py` | `test_deep_ae_vae_and_leave_attack_out_score_fusion` | PASSED | 20.74 | 20.75 |
| 29 | `tests/test_models_and_inference.py` | `test_optuna_budget_and_bounded_neural_loss_keep_revert` | PASSED | 25.45 | 25.46 |
| 30 | `tests/test_models_and_inference.py` | `test_degenerate_fusion_labeling_and_na_unmeasured_values` | PASSED | 0.04 | 0.04 |
| 31 | `tests/test_models_and_inference.py` | `test_time_aware_xgboost_profile_selection_validation_only_and_reporting` | PASSED | 1.10 | 1.12 |
| 32 | `tests/test_models_and_inference.py` | `test_xgboost_downstream_refit_validation_only_keeps_mlp_cnn_calibrators` | PASSED | 165.35 | 165.36 |
| 33 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_partial_and_corrupt_artifacts` | PASSED | 0.52 | 0.53 |
| 34 | `tests/test_repo_and_edge_cases.py` | `test_batch_csv_scoring_schema_validation_and_reordering` | PASSED | 2.32 | 2.33 |
| 35 | `tests/test_repo_and_edge_cases.py` | `test_split_strategy_validation_and_error_paths` | PASSED | 0.61 | 0.62 |
| 36 | `tests/test_repo_and_edge_cases.py` | `test_direct_single_flow_benchmark_and_onnx_export_script` | PASSED | 3.42 | 3.42 |
| 37 | `tests/test_repo_and_edge_cases.py` | `test_ablation_preserves_exact_rows_even_with_port_only_differences` | PASSED | 1.38 | 1.39 |
| 38 | `tests/test_repo_and_edge_cases.py` | `test_latency_warmup_config_and_test_report_generator_parser` | PASSED | 0.07 | 0.07 |
| 39 | `tests/test_repo_and_edge_cases.py` | `test_manifest_sha256_match_mismatch_missing_and_lfs_skip` | PASSED | 0.07 | 0.07 |
| 40 | `tests/test_repo_and_edge_cases.py` | `test_download_models_fallback_and_hash_verification` | PASSED | 0.05 | 0.05 |
| 41 | `tests/test_repo_and_edge_cases.py` | `test_real_lfs_artifact_smoke` | SKIPPED | 0.01 | 0.01 |
| 42 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_stub_mode_graceful_degradation` | PASSED | 0.08 | 0.08 |
| 43 | `tests/test_repo_and_edge_cases.py` | `test_no_absolute_machine_paths_in_artifacts_and_sanitizer` | PASSED | 0.10 | 0.10 |
| 44 | `tests/test_repo_and_edge_cases.py` | `test_check_optional_data_present_and_missing` | PASSED | 0.02 | 0.02 |
| 45 | `tests/test_repo_and_edge_cases.py` | `test_readme_numbers_match_generated_reports_and_detect_drift` | PASSED | 0.07 | 0.07 |
| 46 | `tests/test_repo_and_edge_cases.py` | `test_deployment_profile_config_registry_manifest_and_only_xgboost` | PASSED | 1.11 | 1.12 |
| 47 | `tests/test_response_and_threat.py` | `test_threat_level_boundaries_and_benign_safety` | PASSED | 0.00 | 0.00 |
| 48 | `tests/test_response_and_threat.py` | `test_ip_validation_prevents_command_injection` | PASSED | 0.00 | 0.00 |
| 49 | `tests/test_response_and_threat.py` | `test_active_response_dry_run_idempotency_and_audit_log` | PASSED | 0.02 | 0.02 |
| 50 | `tests/test_response_and_threat.py` | `test_local_shap_explanation_xgboost` | PASSED | 0.80 | 0.80 |
| 51 | `tests/test_response_and_threat.py` | `test_response_command_previews_and_protected_ip_safeguards` | PASSED | 0.03 | 0.03 |
