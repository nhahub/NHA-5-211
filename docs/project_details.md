# SMART IDS v3 — Problem Statement, Layer Specifications & Extended Notes

This document holds the detailed background narrative, model layer specifications, dependency matrix, troubleshooting guide, and future roadmap referenced from [`README.md`](../README.md).

---

## 1. Problem Statement & Core Objectives

Traditional signature-based Intrusion Detection Systems (e.g., Snort, Suricata) fail against polymorphic payloads and zero-day attacks lacking prior rules. Conversely, naive ML classifiers trained on heavily imbalanced network flow corpora (where benign flows dominate >80% of traffic) optimize overall accuracy by ignoring rare, high-impact attack categories (`Web Attack`, `FTP-Patator`, `SSH-Patator`), while closed-set supervised classifiers alone cannot detect novel unseen attacks.

### Project Objectives

1. **Zero-Leakage Cyber-Data Engineering:** Ingest raw CICFlowMeter CSVs, eliminate `±Inf`, `NaN`, duplicate rows, and structural identifiers (`Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Timestamp`), split **before** fitting imputers/clippers/scalers, deduplicate again after imputation and outlier clipping so cross-split feature collisions cannot enter validation/test or SMOTE, enforce disjointness via `strict_leakage_check`, and benchmark SMOTE vs. class weighting strictly on the training split.
2. **Multi-Paradigm Benchmarking:** Train, tune, and evaluate MLP, 1D-CNN, Autoencoder, XGBoost, and ensemble decision logic across accuracy, Macro/Weighted F1, ROC-AUC, PR-AUC, calibration (ECE/Brier), and single-flow latency.
3. **Orthogonal Zero-Day Novelty Tracking:** Separate closed-set multi-class classification from open-set benign-only reconstruction error, evaluating threshold percentile trade-offs (`90, 95, 97, 99`), classical anomaly baselines (Isolation Forest, One-Class SVM, Local Outlier Factor), and a Leave-One-Attack-Out unseen-attack protocol.
4. **Explainability & Safe Mitigation:** Provide game-theoretic SHAP attributions (`TreeExplainer` + `GradientExplainer`/`KernelExplainer` comparison) and a canonical 4-tier Active Response Engine operating in safe **DRY-RUN** simulation mode with strict `ipaddress` validation and append-only audit logging.
5. **SOC Operational Dashboard:** Deliver a 7-tab dark-navy Streamlit SOC platform supporting live single-flow inference, schema-validated batch CSV scoring, interactive SHAP drill-downs, and incident exports.

---

## 2. Detailed Model Specifications

1. **Multi-Layer Perceptron (MLP)** — Deep tabular classifier (`512 → 256 → 128 → 64`) regularized with Batch Normalization, LeakyReLU (`negative_slope=0.1`), Dropout (`0.4`), L2 decay (`1e-4`), and gradient clipping (`clipnorm=1.0`).
2. **One-Dimensional Convolutional Neural Network (1D-CNN)** — Dual Conv1D blocks (`64` and `128` filters, kernel size `3`), MaxPooling1D, GlobalAveragePooling1D, and dense classification head.
3. **Undercomplete Autoencoder (Unsupervised Zero-Day Tracker)** — Symmetric `256 → 128 → 64 → 32 → 64 → 128 → 256` reconstruction network trained **exclusively on `BENIGN` flows** under MSE loss, calibrated at benign validation reconstruction percentiles.
4. **XGBoost Classifier (Baseline & Optuna-Tuned)** — Regularized gradient-boosted decision trees (`hist`, `multi:softprob`) with CUDA/CPU auto-detection and Optuna TPE optimization gated against 5-fold cross-validation standard deviation.

See [`docs/model_configuration.md`](model_configuration.md) and [`docs/architecture.md`](architecture.md) for hyperparameters and pipeline diagrams.

---

## 3. Dependency Matrix (`Python 3.13.12`)

| Component | Version | Role |
|---|---|---|
| Python | `3.13.12` | Runtime interpreter |
| TensorFlow / Keras | `2.21.0` / `3.15.1` | MLP, 1D-CNN, Autoencoder |
| XGBoost | `3.4.1` | Gradient-boosted baseline & CUDA-accelerated Optuna tuning |
| Scikit-learn | `1.9.1` | Splitting, StandardScaler, LabelEncoder, metrics, anomaly baselines |
| Imbalanced-learn | `0.14.2` | Train-only SMOTE resampling |
| SHAP | `0.54.1` | `TreeExplainer`, `GradientExplainer`, `KernelExplainer`, waterfall plots |
| Optuna | `5.1.0` | TPE hyperparameter optimization |
| Pandas / NumPy / PyArrow | `3.1.1` / `2.5.5` / `26.0.1` | Vectorized tabular processing |
| Streamlit / Plotly | `1.67.0` / `7.1.0` | Interactive SOC dashboard |
| Matplotlib / Seaborn | `3.11.2` / `0.13.2` | Publication-grade figures |
| Pydantic / PyYAML / Joblib | `2.14.3` / `6.0.3` / `1.6.0` | Validated configuration & artifact persistence |
| pytest / Ruff | `9.2.0` / `0.17.5` | Automated testing, linting, and formatting |

---

## 4. ONNX Export Feasibility Notes

`scripts/export_onnx.py` provides optional conversion of trained Keras checkpoints (`best_mlp.keras`, `best_cnn.keras`, `best_autoencoder.keras`) using `tf2onnx` / `onnx` and optional numerical parity verification via `onnxruntime`.

- **Why optional:** At development time on **Python 3.13 + TensorFlow 2.21.0 + Keras 3.15.1 + NumPy 2.x**, `tf2onnx` and `onnx` are excluded from `requirements.txt` because `tf2onnx` release wheels lag Python 3.13 / NumPy 2.x / TensorFlow 2.21 compatibility.
- **Direct Runtime Alternative:** For production single-flow latency benchmarking and streaming inference without ONNX dependencies, [`src/evaluation/benchmarking.py`](../src/evaluation/benchmarking.py) and [`src/detection/inference.py`](../src/detection/inference.py) invoke warmed concrete `tf.function` callables for Keras models and `booster.inplace_predict(...)` (respecting `best_iteration`) for XGBoost.

---

## 5. Troubleshooting

- **TensorFlow uses CPU on Windows:** Official TensorFlow builds (`>=2.11`) dropped native Windows GPU support. Use WSL2 Ubuntu with `tensorflow[and-cuda]` if TensorFlow GPU acceleration is required; XGBoost automatically uses CUDA natively on Windows.
- **Uploaded CSV rejected in Predict Flow:** Ensure all numeric CICFlowMeter feature columns matching `models/feature_names.json` are present (header whitespace and extra metadata columns like `Source IP` / `Label` are handled automatically).
- **Dashboard shows missing models:** Run `python scripts/train.py --run-eval` first so artifacts populate `models/` and `reports/`.

---

## 6. Future Improvements

1. Streaming ingestion via Kafka/eBPF flow collectors.
2. Platform-agnostic SOAR/cloud security-group mitigation adapters.
3. Continual online drift monitoring and threshold recalibration.
4. Full ONNX Runtime / TensorRT deployment once Python 3.13 / TensorFlow 2.21 converter support stabilizes.
