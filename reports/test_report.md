# SMART IDS v3 — Automated Verification & Test Report

> Generated automatically by `scripts/generate_test_report.py` on `2026-11-20 17:19 UTC`.

## 1. Static Analysis & Formatting (`ruff 0.17.5`)
- **Command:** `ruff check src dashboard scripts tests app.py`
  - **Status:** `PASSED` (`exit=0`, `0.12s`) — `All checks passed!`
- **Command:** `ruff format --check src dashboard scripts tests app.py`
  - **Status:** `PASSED` (`exit=0`, `0.10s`) — `67 files already formatted`

---

## 2. CI Verification Matrix (Real Artifacts vs. LFS Pointer Stub Simulation)

| Suite Mode | Collected | Passed | Failed | Error | Skipped | Duration_s | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Full Artifacts (local binaries) | 50 | 50 | 0 | 0 | 0 | 548.33 | PASSED |
| LFS Pointer Stubs (`lfs: false` mirror, 15 stubs) | 50 | 49 | 0 | 0 | 1 | 541.06 | PASSED |

---

## 3. Full-Artifact Test Suite (`pytest 9.2.0`)
- **Command:** `pytest tests/ -v --durations=0 --tb=short`
- **Summary:** **`50` collected — `50 passed, 979 warnings` (`548.33s` total, 50 passed / 0 failed / 0 error / 0 skipped)**

| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |
|---:|---|---|---|---:|---:|
| 1 | `tests/test_dashboard_components.py` | `test_dashboard_charts_and_badges_render_cleanly` | PASSED | 1.10 | 1.10 |
| 2 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_smoke_with_existing_artifacts` | PASSED | 12.37 | 12.37 |
| 3 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_graceful_degradation_without_models` | PASSED | 4.55 | 4.56 |
| 4 | `tests/test_data_pipeline.py` | `test_discover_csv_files_and_whitespace_stripping` | PASSED | 0.10 | 0.14 |
| 5 | `tests/test_data_pipeline.py` | `test_cleaning_inf_nan_duplicates_and_web_attack_mojibake` | PASSED | 0.08 | 0.08 |
| 6 | `tests/test_data_pipeline.py` | `test_schema_alignment_errors_and_coercion` | PASSED | 0.07 | 0.07 |
| 7 | `tests/test_data_pipeline.py` | `test_leakage_free_preprocessing_and_dynamic_smote_k` | PASSED | 0.35 | 0.36 |
| 8 | `tests/test_data_pipeline.py` | `test_post_clip_near_duplicate_removal_and_strict_leakage_error` | PASSED | 0.54 | 0.55 |
| 9 | `tests/test_data_pipeline.py` | `test_by_day_split_and_drop_features` | PASSED | 0.45 | 0.46 |
| 10 | `tests/test_data_pipeline.py` | `test_smote_multiplier_cap_and_min_real_samples_threshold` | PASSED | 0.01 | 0.01 |
| 11 | `tests/test_data_pipeline.py` | `test_time_block_split_timestamp_ordering_no_leakage_and_fallbacks` | PASSED | 0.35 | 0.36 |
| 12 | `tests/test_data_pipeline.py` | `test_real_timestamp_alignment_and_missing_dir_warning` | PASSED | 0.68 | 0.69 |
| 13 | `tests/test_data_pipeline.py` | `test_sample_size_config_label_correction_and_sensitivity_protocol` | PASSED | 2.04 | 2.05 |
| 14 | `tests/test_models_and_inference.py` | `test_all_four_models_build_train_save_reload_and_infer` | PASSED | 19.98 | 19.99 |
| 15 | `tests/test_models_and_inference.py` | `test_hybrid_ensemble_zero_day_override_and_missing_artifacts` | PASSED | 0.01 | 0.01 |
| 16 | `tests/test_models_and_inference.py` | `test_by_day_protocol_feature_ablation_and_report_tables` | PASSED | 53.87 | 53.88 |
| 17 | `tests/test_models_and_inference.py` | `test_imbalance_selection_bootstrap_ci_per_class_and_xgb_cv` | PASSED | 2.55 | 2.56 |
| 18 | `tests/test_models_and_inference.py` | `test_temperature_scaler_handles_float32_overconfidence` | PASSED | 0.04 | 0.04 |
| 19 | `tests/test_models_and_inference.py` | `test_optuna_cv_std_gating_and_stacking_meta_learner` | PASSED | 1.35 | 1.35 |
| 20 | `tests/test_models_and_inference.py` | `test_classical_anomaly_baselines_and_unified_evaluation` | PASSED | 0.87 | 0.88 |
| 21 | `tests/test_models_and_inference.py` | `test_paired_bootstrap_significance_and_report_conclusion` | PASSED | 0.15 | 0.15 |
| 22 | `tests/test_models_and_inference.py` | `test_train_all_models_unified_cv_regression` | PASSED | 149.85 | 149.86 |
| 23 | `tests/test_models_and_inference.py` | `test_imbalance_tie_breaking_with_synthetic_metric_dicts` | PASSED | 0.00 | 0.00 |
| 24 | `tests/test_models_and_inference.py` | `test_per_model_imbalance_selection_caps_epochs_and_reports` | PASSED | 58.46 | 58.47 |
| 25 | `tests/test_models_and_inference.py` | `test_small_class_reliability_oof_wilson_and_reliable_macro_f1` | PASSED | 1.98 | 1.99 |
| 26 | `tests/test_models_and_inference.py` | `test_time_block_failure_analysis_ks_drift_and_mitigation_tradeoffs` | PASSED | 0.13 | 0.14 |
| 27 | `tests/test_models_and_inference.py` | `test_deep_ae_vae_and_leave_attack_out_score_fusion` | PASSED | 21.09 | 21.10 |
| 28 | `tests/test_models_and_inference.py` | `test_optuna_budget_and_bounded_neural_loss_keep_revert` | PASSED | 27.80 | 27.81 |
| 29 | `tests/test_models_and_inference.py` | `test_degenerate_fusion_labeling_and_na_unmeasured_values` | PASSED | 0.07 | 0.07 |
| 30 | `tests/test_models_and_inference.py` | `test_time_aware_xgboost_profile_selection_validation_only_and_reporting` | PASSED | 1.06 | 1.07 |
| 31 | `tests/test_models_and_inference.py` | `test_xgboost_downstream_refit_validation_only_keeps_mlp_cnn_calibrators` | PASSED | 166.45 | 166.47 |
| 32 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_partial_and_corrupt_artifacts` | PASSED | 0.54 | 0.55 |
| 33 | `tests/test_repo_and_edge_cases.py` | `test_batch_csv_scoring_schema_validation_and_reordering` | PASSED | 2.34 | 2.35 |
| 34 | `tests/test_repo_and_edge_cases.py` | `test_split_strategy_validation_and_error_paths` | PASSED | 0.64 | 0.65 |
| 35 | `tests/test_repo_and_edge_cases.py` | `test_direct_single_flow_benchmark_and_onnx_export_script` | PASSED | 3.55 | 3.55 |
| 36 | `tests/test_repo_and_edge_cases.py` | `test_ablation_preserves_exact_rows_even_with_port_only_differences` | PASSED | 1.45 | 1.46 |
| 37 | `tests/test_repo_and_edge_cases.py` | `test_latency_warmup_config_and_test_report_generator_parser` | PASSED | 0.07 | 0.07 |
| 38 | `tests/test_repo_and_edge_cases.py` | `test_manifest_sha256_match_mismatch_missing_and_lfs_skip` | PASSED | 0.08 | 0.09 |
| 39 | `tests/test_repo_and_edge_cases.py` | `test_download_models_fallback_and_hash_verification` | PASSED | 0.05 | 0.05 |
| 40 | `tests/test_repo_and_edge_cases.py` | `test_real_lfs_artifact_smoke` | PASSED | 0.56 | 0.56 |
| 41 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_stub_mode_graceful_degradation` | PASSED | 0.08 | 0.08 |
| 42 | `tests/test_repo_and_edge_cases.py` | `test_no_absolute_machine_paths_in_artifacts_and_sanitizer` | PASSED | 0.05 | 0.05 |
| 43 | `tests/test_repo_and_edge_cases.py` | `test_check_optional_data_present_and_missing` | PASSED | 0.01 | 0.01 |
| 44 | `tests/test_repo_and_edge_cases.py` | `test_readme_numbers_match_generated_reports_and_detect_drift` | PASSED | 0.03 | 0.03 |
| 45 | `tests/test_repo_and_edge_cases.py` | `test_deployment_profile_config_registry_manifest_and_only_xgboost` | PASSED | 1.20 | 1.22 |
| 46 | `tests/test_response_and_threat.py` | `test_threat_level_boundaries_and_benign_safety` | PASSED | 0.00 | 0.00 |
| 47 | `tests/test_response_and_threat.py` | `test_ip_validation_prevents_command_injection` | PASSED | 0.00 | 0.00 |
| 48 | `tests/test_response_and_threat.py` | `test_active_response_dry_run_idempotency_and_audit_log` | PASSED | 0.02 | 0.02 |
| 49 | `tests/test_response_and_threat.py` | `test_local_shap_explanation_xgboost` | PASSED | 0.83 | 0.83 |
| 50 | `tests/test_response_and_threat.py` | `test_response_command_previews_and_protected_ip_safeguards` | PASSED | 0.03 | 0.03 |

---

## 4. LFS Pointer Stub Simulation Suite (`pytest 9.2.0`, `lfs: false` Mirror)
- **Command:** `pytest tests/ -v --durations=0 --tb=short (stub-mode simulation)` (`15` LFS binaries replaced by pointer stubs)
- **Summary:** **`50` collected — `49 passed, 1 skipped, 979 warnings` (`541.06s` total, 49 passed / 0 failed / 0 error / 1 skipped)**
- **Skipped Tests (stub detection):** `tests/test_repo_and_edge_cases.py::test_real_lfs_artifact_smoke`

| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |
|---:|---|---|---|---:|---:|
| 1 | `tests/test_dashboard_components.py` | `test_dashboard_charts_and_badges_render_cleanly` | PASSED | 0.96 | 0.96 |
| 2 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_smoke_with_existing_artifacts` | PASSED | 8.55 | 8.55 |
| 3 | `tests/test_dashboard_components.py` | `test_streamlit_apptest_graceful_degradation_without_models` | PASSED | 4.12 | 4.13 |
| 4 | `tests/test_data_pipeline.py` | `test_discover_csv_files_and_whitespace_stripping` | PASSED | 0.10 | 0.14 |
| 5 | `tests/test_data_pipeline.py` | `test_cleaning_inf_nan_duplicates_and_web_attack_mojibake` | PASSED | 0.07 | 0.07 |
| 6 | `tests/test_data_pipeline.py` | `test_schema_alignment_errors_and_coercion` | PASSED | 0.07 | 0.07 |
| 7 | `tests/test_data_pipeline.py` | `test_leakage_free_preprocessing_and_dynamic_smote_k` | PASSED | 0.36 | 0.37 |
| 8 | `tests/test_data_pipeline.py` | `test_post_clip_near_duplicate_removal_and_strict_leakage_error` | PASSED | 0.56 | 0.58 |
| 9 | `tests/test_data_pipeline.py` | `test_by_day_split_and_drop_features` | PASSED | 0.47 | 0.48 |
| 10 | `tests/test_data_pipeline.py` | `test_smote_multiplier_cap_and_min_real_samples_threshold` | PASSED | 0.01 | 0.01 |
| 11 | `tests/test_data_pipeline.py` | `test_time_block_split_timestamp_ordering_no_leakage_and_fallbacks` | PASSED | 0.35 | 0.36 |
| 12 | `tests/test_data_pipeline.py` | `test_real_timestamp_alignment_and_missing_dir_warning` | PASSED | 0.65 | 0.66 |
| 13 | `tests/test_data_pipeline.py` | `test_sample_size_config_label_correction_and_sensitivity_protocol` | PASSED | 2.21 | 2.23 |
| 14 | `tests/test_models_and_inference.py` | `test_all_four_models_build_train_save_reload_and_infer` | PASSED | 20.22 | 20.23 |
| 15 | `tests/test_models_and_inference.py` | `test_hybrid_ensemble_zero_day_override_and_missing_artifacts` | PASSED | 0.01 | 0.01 |
| 16 | `tests/test_models_and_inference.py` | `test_by_day_protocol_feature_ablation_and_report_tables` | PASSED | 51.05 | 51.06 |
| 17 | `tests/test_models_and_inference.py` | `test_imbalance_selection_bootstrap_ci_per_class_and_xgb_cv` | PASSED | 2.57 | 2.58 |
| 18 | `tests/test_models_and_inference.py` | `test_temperature_scaler_handles_float32_overconfidence` | PASSED | 0.03 | 0.03 |
| 19 | `tests/test_models_and_inference.py` | `test_optuna_cv_std_gating_and_stacking_meta_learner` | PASSED | 1.33 | 1.34 |
| 20 | `tests/test_models_and_inference.py` | `test_classical_anomaly_baselines_and_unified_evaluation` | PASSED | 0.79 | 0.80 |
| 21 | `tests/test_models_and_inference.py` | `test_paired_bootstrap_significance_and_report_conclusion` | PASSED | 0.17 | 0.17 |
| 22 | `tests/test_models_and_inference.py` | `test_train_all_models_unified_cv_regression` | PASSED | 152.36 | 152.37 |
| 23 | `tests/test_models_and_inference.py` | `test_imbalance_tie_breaking_with_synthetic_metric_dicts` | PASSED | 0.00 | 0.00 |
| 24 | `tests/test_models_and_inference.py` | `test_per_model_imbalance_selection_caps_epochs_and_reports` | PASSED | 56.62 | 56.63 |
| 25 | `tests/test_models_and_inference.py` | `test_small_class_reliability_oof_wilson_and_reliable_macro_f1` | PASSED | 1.78 | 1.79 |
| 26 | `tests/test_models_and_inference.py` | `test_time_block_failure_analysis_ks_drift_and_mitigation_tradeoffs` | PASSED | 0.12 | 0.15 |
| 27 | `tests/test_models_and_inference.py` | `test_deep_ae_vae_and_leave_attack_out_score_fusion` | PASSED | 21.99 | 22.00 |
| 28 | `tests/test_models_and_inference.py` | `test_optuna_budget_and_bounded_neural_loss_keep_revert` | PASSED | 24.17 | 24.18 |
| 29 | `tests/test_models_and_inference.py` | `test_degenerate_fusion_labeling_and_na_unmeasured_values` | PASSED | 0.04 | 0.04 |
| 30 | `tests/test_models_and_inference.py` | `test_time_aware_xgboost_profile_selection_validation_only_and_reporting` | PASSED | 1.34 | 1.35 |
| 31 | `tests/test_models_and_inference.py` | `test_xgboost_downstream_refit_validation_only_keeps_mlp_cnn_calibrators` | PASSED | 167.31 | 167.32 |
| 32 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_partial_and_corrupt_artifacts` | PASSED | 0.60 | 0.61 |
| 33 | `tests/test_repo_and_edge_cases.py` | `test_batch_csv_scoring_schema_validation_and_reordering` | PASSED | 2.42 | 2.44 |
| 34 | `tests/test_repo_and_edge_cases.py` | `test_split_strategy_validation_and_error_paths` | PASSED | 0.62 | 0.64 |
| 35 | `tests/test_repo_and_edge_cases.py` | `test_direct_single_flow_benchmark_and_onnx_export_script` | PASSED | 3.63 | 3.63 |
| 36 | `tests/test_repo_and_edge_cases.py` | `test_ablation_preserves_exact_rows_even_with_port_only_differences` | PASSED | 1.43 | 1.45 |
| 37 | `tests/test_repo_and_edge_cases.py` | `test_latency_warmup_config_and_test_report_generator_parser` | PASSED | 0.10 | 0.10 |
| 38 | `tests/test_repo_and_edge_cases.py` | `test_manifest_sha256_match_mismatch_missing_and_lfs_skip` | PASSED | 0.07 | 0.07 |
| 39 | `tests/test_repo_and_edge_cases.py` | `test_download_models_fallback_and_hash_verification` | PASSED | 0.05 | 0.05 |
| 40 | `tests/test_repo_and_edge_cases.py` | `test_real_lfs_artifact_smoke` | SKIPPED | 0.01 | 0.01 |
| 41 | `tests/test_repo_and_edge_cases.py` | `test_model_registry_stub_mode_graceful_degradation` | PASSED | 0.07 | 0.07 |
| 42 | `tests/test_repo_and_edge_cases.py` | `test_no_absolute_machine_paths_in_artifacts_and_sanitizer` | PASSED | 0.09 | 0.09 |
| 43 | `tests/test_repo_and_edge_cases.py` | `test_check_optional_data_present_and_missing` | PASSED | 0.03 | 0.03 |
| 44 | `tests/test_repo_and_edge_cases.py` | `test_readme_numbers_match_generated_reports_and_detect_drift` | PASSED | 0.08 | 0.08 |
| 45 | `tests/test_repo_and_edge_cases.py` | `test_deployment_profile_config_registry_manifest_and_only_xgboost` | PASSED | 1.16 | 1.18 |
| 46 | `tests/test_response_and_threat.py` | `test_threat_level_boundaries_and_benign_safety` | PASSED | 0.00 | 0.00 |
| 47 | `tests/test_response_and_threat.py` | `test_ip_validation_prevents_command_injection` | PASSED | 0.00 | 0.00 |
| 48 | `tests/test_response_and_threat.py` | `test_active_response_dry_run_idempotency_and_audit_log` | PASSED | 0.02 | 0.02 |
| 49 | `tests/test_response_and_threat.py` | `test_local_shap_explanation_xgboost` | PASSED | 0.85 | 0.85 |
| 50 | `tests/test_response_and_threat.py` | `test_response_command_previews_and_protected_ip_safeguards` | PASSED | 0.02 | 0.02 |
