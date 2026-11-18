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
| `data.split` | `test_size`, `val_size_of_remainder` | `0.15`, `0.15` | Stratified `72.25%` Train / `12.75%` Validation / `15.0%` Test split |
| `data.imbalance` | `strategy` | `auto_compare` | Empirically compares Unweighted, Class-Weighting, and SMOTE on validation Macro-F1 |
| `evaluation` | `latency_warmup_runs`, `latency_timed_runs` | `5`, `100` | Warm-up runs before repeated `time.perf_counter()` single-flow latency measurement |
| `evaluation` | `autoencoder_percentiles` | `[90, 95, 97, 99]` | Percentile thresholds evaluated for precision/recall/FPR trade-off analysis |
| `response` | `dry_run`, `rate_limit_pps` | `true`, `100` | Safety lock ensuring all `iptables` actions remain simulated |

---

## 2. Model Configuration (`configs/model_config.yaml`)

### A. Multi-Layer Perceptron (`mlp`)
- `hidden_units: [512, 256, 128, 64]`
- `l2_reg: 0.0001`, `leaky_relu_slope: 0.1`, `dropout_rate: 0.4`
- `learning_rate: 0.001`, `clipnorm: 1.0`, `batch_size: 512`, `epochs: 50`
- Callbacks: `EarlyStopping(patience=8)`, `ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6)`

### B. One-Dimensional CNN (`cnn`)
- Block 1: `block1_filters: 64`, `kernel_size: 3`, `pool_size: 2`, `dropout_block1: 0.3`
- Block 2: `block2_filters: 128`, `GlobalAveragePooling1D`, `dropout_block2: 0.4`
- Classification Head: `dense_units: 128`, `dropout_head: 0.3`
- Optimizer & Callbacks: identical to MLP (`Adam(lr=1e-3, clipnorm=1.0)`, `patience=8`)

### C. Undercomplete Autoencoder (`autoencoder`)
- `encoder_units: [256, 128, 64]`, `latent_dim: 32`, `decoder_units: [64, 128, 256]`
- Linear output reconstruction (`mse` loss), `learning_rate: 0.0005`
- `threshold_percentile: 95.0` calibrated strictly on `X_benign_val`
- Callbacks: `EarlyStopping(patience=10)`, `ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)`

### D. XGBoost & Optuna HPO (`xgboost`)
- Report Baseline: `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `gamma=0.1`, `reg_alpha=0.01`, `reg_lambda=1.0`, `tree_method="hist"`, `early_stopping_rounds=20`
- Optuna Search: `n_trials=12`, `timeout_seconds=180`, TPE sampler minimizing validation `mlogloss`; `tuned` is selected only when validation Macro-F1 beats baseline by more than the 5-fold CV standard deviation (`> cv_macro_f1_std`), otherwise retaining `baseline`.

### E. Hybrid Ensemble (`ensemble`)
- `strategy: "hybrid_anomaly_override"` (supports `best_single`, `soft_voting`, `confidence_weighted`, `val_weighted_voting`, `stacking`, `hybrid_anomaly_override`)
- `base_supervised_strategy: "best_single"` (prevents weak neural voters from diluting `best_single` XGBoost accuracy while retaining orthogonal Autoencoder novelty alerts)
- `supervised_weights`: fallback weights (`XGBoost: 0.45`, `MLP: 0.35`, `CNN: 0.20`), superseded by validation log-loss weighting (`val_weighted_voting`) or `LogisticRegression` meta-learning (`stacking`, trained strictly on validation log-probabilities)
- `calibrate_probabilities: true` (float64 bounded Brent Temperature Scaling fit strictly on validation probabilities)
