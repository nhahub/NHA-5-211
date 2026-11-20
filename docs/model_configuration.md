# SMART IDS v3 — Model & Pipeline Configuration Reference

All system parameters are centralized in two Pydantic-validated YAML files:
- [`configs/config.yaml`](../configs/config.yaml) — Data engineering, evaluation, active response, and dashboard settings.
- [`configs/model_config.yaml`](../configs/model_config.yaml) — Hardware acceleration, model architectures, Optuna HPO, and ensemble fusion settings.

---

## 1. Pipeline Configuration (`configs/config.yaml`)

| Section | Key | Default | Engineering Purpose |
|---|---|---|---|
| `project` | `random_seed` | `42` | Global deterministic seed across Python, NumPy, TensorFlow, SMOTE, and Optuna |
| `data` | `max_rows_sample` | `200000` | Stratified downsample ceiling when loading multi-million-row raw CICIDS2017 CSVs |
| `data` | `merge_web_attacks` | `true` | Consolidates CICIDS2017 Web Attack sub-labels into canonical `Web Attack` |
| `data` | `nan_column_drop_threshold` | `0.50` | Drops columns exceeding 50% missingness (computed strictly on training split) |
| `data` | `outlier_clip_sigma` | `5.0` | Clips rate/duration artifacts to $\mu \pm 5\sigma$ using training statistics only |
| `data` | `split_strategy` | `random` | `random` (stratified), `time_block` (per-class chronological drift), or `by_day` (open-set cross-day) |
| `data.split` | `test_size`, `val_size_of_remainder`, `time_block_*` | `0.15`, `0.15`, `0.60`/`0.20` | Stratified `72.25%`/`12.75%`/`15.0%` (`random`) or chronological `60%`/`20%`/`20%` (`time_block`) |
| `deployment` | `xgboost_imbalance_profile` | `class_weight` | Active XGBoost artifact profile: `class_weight` → `models/xgb_model.json` (default) or `none` → `models/xgb_model_none.json`; selected on validation data only |
| `data.imbalance` | `strategy`, `imbalance_tie_tolerance` | `auto_compare`, `0.002` | Compares Unweighted, Class-Weighting, and SMOTE on validation Macro-F1; prefers `none > class_weight > smote` when tied within tolerance |
| `evaluation` | `latency_warmup_calls`, `latency_timed_runs` | `50`, `100` | Discards first N warm-up calls before `time.perf_counter()` single-flow latency timing |
| `evaluation` | `run_by_day_comparison`, `run_time_block_comparison` | `true`, `true` | Runs secondary open-set cross-day (`by_day`) and chronological drift (`time_block`) benchmarks |
| `evaluation` | `autoencoder_percentiles` | `[90, 95, 97, 99]` | Percentile thresholds evaluated for precision/recall/FPR trade-off analysis |
| `response` | `dry_run`, `rate_limit_pps` | `true`, `100` | Safety lock ensuring all firewall previews remain non-executed (`NOT EXECUTED (dry-run)`) |
| `security` | `verify_model_hashes`, `require_manifest`, `manifest_file` | `true`, `false`, `"MANIFEST.json"` | Verifies SHA-256 against `models/MANIFEST.json` before every `joblib.load`; refuses mismatches via `ModelIntegrityError` |
| `models_download` | `release_base_url`, `release_repo`, `release_tag` | `""`, `""`, `"models-v3"` | Optional GitHub Release fallback URL settings used by `scripts/download_models.py` |

---

## 2. Model Configuration (`configs/model_config.yaml`)

### A. Multi-Layer Perceptron (`mlp` — Neural Baseline for Comparison)
- `hidden_units: [512, 256, 128, 64]`
- `l2_reg: 0.0001`, `leaky_relu_slope: 0.1`, `dropout_rate: 0.4`
- `learning_rate: 0.001`, `clipnorm: 1.0`, `batch_size: 512`, `epochs: 50`
- Callbacks & Bounded Trials (`neural:`): `EarlyStopping(patience=8)`, `ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6)`; bounded `label_smoothing` / `focal_loss` trials capped by `neural.max_training_minutes: 3.0` and kept only when validation Macro-F1 beats baseline by `> 0.002`.

### B. One-Dimensional CNN (`cnn` — Neural Baseline for Comparison)
- Block 1: `block1_filters: 64`, `kernel_size: 3`, `pool_size: 2`, `dropout_block1: 0.3`
- Block 2: `block2_filters: 128`, `GlobalAveragePooling1D`, `dropout_block2: 0.4`
- Classification Head: `dense_units: 128`, `dropout_head: 0.3`
- Optimizer & Callbacks: identical to MLP (`Adam(lr=1e-3, clipnorm=1.0)`, `patience=8`), plus the same bounded `neural:` keep/revert gate.

### C. Undercomplete Autoencoder (`autoencoder`)
- `encoder_units: [256, 128, 64]`, `latent_dim: 32`, `decoder_units: [64, 128, 256]`
- Linear output reconstruction (`mse` loss), `learning_rate: 0.0005`
- `threshold_percentile: 95.0` calibrated strictly on `X_benign_val`
- Callbacks: `EarlyStopping(patience=10)`, `ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)`
- Phase 5 benign-only extensions (`try_variants: true`, `fusion_enabled: true`): Deeper L2/Dropout denoising Autoencoder (`best_autoencoder_deep.keras`), VAE (`best_vae.keras`), and leave-attack-out non-test score-level fusion (`AE + OCSVM + IF`).

### D. XGBoost & Optuna HPO (`xgboost` — Recommended Production Model)
- Report Baseline: `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `gamma=0.1`, `reg_alpha=0.01`, `reg_lambda=1.0`, `tree_method="hist"`, `early_stopping_rounds=20`
- Optuna Search: `n_trials=40`, `timeout_minutes=6.0` (`timeout_seconds=360`), TPE sampler minimizing validation `mlogloss`; `tuned` is selected only when validation Macro-F1 beats baseline by more than the 5-fold CV standard deviation (`> cv_macro_f1_std`), otherwise retaining `baseline`.
- Deployment profiles (`deployment.xgboost_imbalance_profile`): the shipped default is `class_weight` (`models/xgb_model.json`), chosen by chronological `time_block` validation Macro-F1 `0.8754` vs `0.8534` for `none` (`used_test_data = False`). The alternate `none` profile (`models/xgb_model_none.json`) is the ultra-low-FPR option chosen by random-split validation Macro-F1 `0.9518` vs `0.9446`. Both binaries are Git LFS-tracked and verified by `models/MANIFEST.json`.

### E. Multi-Model Decision & Anomaly Integration (`ensemble`)
- `strategy: "hybrid_anomaly_override"` (supports `best_single`, `soft_voting`, `confidence_weighted`, `val_weighted_voting`, `stacking`, `hybrid_anomaly_override`)
- `base_supervised_strategy: "best_single"` (prevents weak neural voters from diluting `best_single` XGBoost accuracy while retaining orthogonal Autoencoder novelty alerts)
- `supervised_weights`: fallback weights (`XGBoost: 0.45`, `MLP: 0.35`, `CNN: 0.20`), superseded by validation log-loss weighting (`val_weighted_voting`) or `LogisticRegression` meta-learning (`stacking`, trained strictly on validation log-probabilities)
- `calibrate_probabilities: true` (float64 bounded Brent Temperature Scaling fit strictly on validation probabilities)

> **Security Note (`*.pkl` Artifacts & SHA-256 Verification):** Serialized files (`models/*.pkl` loaded via `joblib` / `pickle`, including `scaler.pkl`, `imputer.pkl`, `outlier_clipper.pkl`, `label_encoder.pkl`, `calibrators.pkl`, `stacking_meta.pkl`, and `anomaly_baselines.pkl`) use Python pickle deserialization and **must only be loaded from trusted sources**. Training automatically generates `models/MANIFEST.json` containing SHA-256 checksums; when `security.verify_model_hashes: true` (default), `ModelRegistry` ([`src/utils/model_registry.py`](../src/utils/model_registry.py)) validates every digest before loading and raises `ModelIntegrityError` on any mismatch.
