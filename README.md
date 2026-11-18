# SMART IDS v3

**An Intelligent, Multi-Modal Hybrid Ensemble Architecture for Network Intrusion Detection and Zero-Day Anomaly Tracking**

Academic Graduation Project — Built strictly to the specification in `Smart_IDS_v3_Technical_Report.pdf` (Chapters 1–10) with zero-leakage preprocessing, GPU/CPU adaptive acceleration, Optuna optimization, SHAP explainability, dry-run active mitigation, automated testing, and a Security Operations Center (SOC) dashboard.

---

## 1. Project Overview

**SMART IDS v3** is an end-to-end network intrusion detection, zero-day anomaly tracking, explainable AI (XAI), and automated active-response prototype. Instead of relying on static byte signatures or a single classifier, SMART IDS v3 combines four distinct machine learning and deep learning paradigms:

1. **Multi-Layer Perceptron (MLP)** — Deep tabular classifier (`512 → 256 → 128 → 64`) regularized with Batch Normalization, LeakyReLU (`negative_slope=0.1`), Dropout (`0.4`), L2 decay (`1e-4`), and gradient clipping (`clipnorm=1.0`).
2. **One-Dimensional Convolutional Neural Network (1D-CNN)** — Dual Conv1D blocks (`64` and `128` filters, kernel size `3`), MaxPooling1D, GlobalAveragePooling1D, and dense classification head.
3. **Undercomplete Autoencoder (Unsupervised Zero-Day Tracker)** — Symmetric `256 → 128 → 64 → 32 → 64 → 128 → 256` reconstruction network trained **exclusively on `BENIGN` flows** under MSE loss, calibrated at the 95th percentile of benign validation reconstruction error.
4. **XGBoost Classifier (Baseline & Optuna-Tuned)** — Regularized gradient-boosted decision trees (`hist`, `multi:softprob`) optimized via Optuna TPE trials and accelerated on NVIDIA CUDA when available.

---

## 2. Problem Statement

Traditional signature-based Intrusion Detection Systems (e.g., Snort, Suricata) fail against polymorphic payloads and zero-day attacks lacking prior rules. Conversely, naive ML classifiers trained on heavily imbalanced network flow corpora (where benign flows dominate >80% of traffic) optimize overall accuracy by ignoring rare, high-impact attack categories (`Web Attack`, `FTP-Patator`, `SSH-Patator`), while closed-set supervised classifiers alone cannot detect novel unseen attacks.

---

## 3. Objectives

1. **Zero-Leakage Cyber-Data Engineering:** Ingest raw CICFlowMeter CSVs, eliminate `±Inf`, `NaN`, duplicate rows, and structural identifiers (`Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Timestamp`), split **before** fitting imputers/clippers/scalers, and experimentally validate SMOTE vs. class weighting strictly on the training split.
2. **Multi-Paradigm Benchmarking:** Implement, train, tune, and rigorously evaluate MLP, 1D-CNN, Autoencoder, XGBoost, and hybrid ensemble decision logic across accuracy, Macro/Weighted F1, ROC-AUC, PR-AUC, calibration (ECE/Brier), and single-flow latency.
3. **Orthogonal Zero-Day Novelty Tracking:** Separate closed-set multi-class classification from open-set benign-only reconstruction error, evaluating threshold percentile trade-offs (`90, 95, 97, 99`) and a Leave-One-Attack-Out unseen-attack protocol.
4. **Explainability & Safe Mitigation:** Provide game-theoretic SHAP attributions (`TreeExplainer` + `GradientExplainer`/`KernelExplainer` comparison) and a canonical 4-tier Active Response Engine operating in safe **DRY-RUN** simulation mode with strict `ipaddress` validation and append-only audit logging.
5. **SOC Operational Dashboard:** Deliver a 7-tab dark-navy Streamlit SOC platform supporting live single-flow inference, schema-validated batch CSV scoring, interactive SHAP drill-downs, and incident exports.

---

## 4. Architecture

See [`docs/architecture.md`](docs/architecture.md) for full Mermaid system and data-pipeline diagrams.

- **Supervised Attribution:** XGBoost + MLP + 1D-CNN compared individually and via `best_single`, `soft_voting`, `confidence_weighted` (with validation Temperature Scaling), and `hybrid_anomaly_override`.
- **Unsupervised Anomaly Signal:** The Autoencoder anomaly score (`MSE > threshold`) remains an independent orthogonal signal rather than being diluted into supervised class probabilities; if a flow is classified as `BENIGN` (or low-confidence) yet breaches the calibrated MSE threshold, the hybrid engine surfaces a `Zero-Day Anomaly (Novel)` flag.
- **Canonical Response Layer:** Shared across CLI, notebooks, and dashboard via [`src/response/response_engine.py`](src/response/response_engine.py).

---

## 5. Dataset Information

- **Primary Target Dataset:** **CICIDS2017** (Canadian Institute for Cybersecurity), comprising 8 daily CICFlowMeter CSV captures covering `BENIGN`, `DoS Hulk`, `PortScan`, `DDoS`, `DoS GoldenEye`, `FTP-Patator`, `SSH-Patator`, and `Web Attack` (plus rare classes merged/handled cleanly).
- **Cleaned Feature Schema:** **77** non-redundant numeric features (`Destination Port` + 76 CICFlowMeter flow, packet-length, IAT, TCP-flag, subflow, and active/idle statistics).
- **Synthetic Demonstration Mode:** When `data/raw/` contains no CSVs, [`src/data/loader.py`](src/data/loader.py) generates a schema-faithful synthetic demonstration dataset (`data/synthetic/cicids2017_synthetic_demo.csv`) mirroring CICIDS2017 class imbalance weights (`0.83 / 0.06 / 0.04 / 0.03 / 0.01 × 4`), `Inf`, `NaN`, duplicates, and extreme rate spikes. Synthetic runs are explicitly labeled everywhere and never conflated with real CICIDS2017 experimental results.

---

## 6. Technologies & Selected Versions

> **Python Compatibility Decision:** At development time, Python `3.14.5` and `3.13.12` were available on the host. Because the latest stable PyPI release of TensorFlow (**`2.21.0`**) supports Python `3.10–3.13` (while `2.22.0` remains a pre-release `rc0`), **Python `3.13.12`** was selected as the newest stable Python release supported reliably by the entire ML/DL stack. On native Windows, TensorFlow (`>=2.11`) executes on CPU while **XGBoost `3.4.1`** automatically utilizes **NVIDIA CUDA** (verified on the host RTX 3050 6GB GPU) with seamless CPU fallback.

| Component | Version | Role |
|---|---|---|
| Python | `3.13.12` | Runtime interpreter |
| TensorFlow / Keras | `2.21.0` / `3.15.1` | MLP, 1D-CNN, Autoencoder |
| XGBoost | `3.4.1` | Gradient-boosted baseline & CUDA-accelerated Optuna tuning |
| Scikit-learn | `1.9.1` | Splitting, StandardScaler, LabelEncoder, metrics, ROC/PR |
| Imbalanced-learn | `0.14.2` | Train-only SMOTE resampling |
| SHAP | `0.54.1` | `TreeExplainer`, `GradientExplainer`, `KernelExplainer`, waterfall plots |
| Optuna | `5.1.0` | TPE hyperparameter optimization |
| Pandas / NumPy / PyArrow | `3.1.1` / `2.5.5` / `26.0.1` | Vectorized tabular processing |
| Streamlit / Plotly | `1.67.0` / `7.1.0` | Interactive SOC dashboard |
| Matplotlib / Seaborn | `3.11.2` / `0.13.2` | Publication-grade figures |
| Pydantic / PyYAML / Joblib | `2.14.3` / `6.0.3` / `1.6.0` | Validated configuration & artifact persistence |
| pytest / Ruff | `9.2.0` / `0.17.5` | Automated testing, linting, and formatting |

---

## 7. Installation Instructions

```powershell
# 1. Create a Python 3.13 virtual environment
uv venv --python 3.13 .venv
# Or: py -3.13 -m venv .venv

# 2. Install locked compatible dependencies
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# Or: .venv\Scripts\pip.exe install -r requirements.txt

# 3. Copy environment template (optional)
copy .env.example .env
```

---

## 8. Dataset Preparation

1. Download the official **CICIDS2017** CSV archive (`MachineLearningCSV.zip` or `TrafficLabelling`) from:
   `https://www.unb.ca/cic/datasets/ids-2017.html`
2. Extract the 8 CSV exports into `data/raw/`.
3. If `data/raw/` is left empty, training and exploration automatically enter **Explicit Synthetic Demonstration Mode**.

---

## 9. Model Training

```powershell
# Train on raw CICIDS2017 CSVs in data/raw/ (or synthetic fallback if empty) and run full evaluation:
.venv\Scripts\python.exe scripts/train.py --run-eval

# Explicitly force synthetic demonstration mode with custom limits:
.venv\Scripts\python.exe scripts/train.py --synthetic --max-rows 60000 --epochs 50 --optuna-trials 12 --run-eval
```

Artifacts produced in `models/` and `data/processed/`:
- `best_mlp.keras`, `best_cnn.keras`, `best_autoencoder.keras`, `xgb_model.json`
- `scaler.pkl`, `imputer.pkl`, `outlier_clipper.pkl`, `label_encoder.pkl`, `calibrators.pkl`
- `feature_names.json`, `anomaly_threshold.json`, `training_histories.json`, `preprocessing_audit.json`

---

## 10. Model Evaluation & Measured Experimental Results

Run independent evaluation (reloading persisted models from disk) anytime via:

```powershell
.venv\Scripts\python.exe scripts/evaluate.py
```

### Measured Results (Genuine CICIDS2017 Benchmark Run: 2,830,743 raw rows → 200,000 stratified sample → 191,268 deduplicated rows; 77 features, 13 classes)

> **Dataset Provenance:** All 8 daily CICIDS2017 CSV exports in `data/raw/` (`138,190` train pre-SMOTE / `413,056` train post-SMOTE with `smote_cap_per_class=25,000` / `24,387` validation / `28,691` held-out test).

| Model | Framing | Accuracy | Precision (W) | Recall (W) | F1 (W) | F1 (Macro) | ROC-AUC | Mean Latency (µs) | Batch Throughput (flows/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Optuna)** | 13-class supervised | **0.9989** | **0.9989** | **0.9989** | **0.9989** | **0.9704** | **0.9995** | **516.82** | **263,650.3** |
| **1D-CNN** | 13-class supervised | 0.9849 | 0.9945 | 0.9849 | 0.9892 | 0.8281 | 0.9979 | 10,128.19 | 16,945.0 |
| **MLP** | 13-class supervised | 0.9633 | 0.9785 | 0.9633 | 0.9692 | 0.8076 | 0.9965 | 5,054.23 | 23,584.7 |
| **Autoencoder** | Binary anomaly (Benign vs Anomalous) | 0.8688 | 0.6863 | 0.5132 | 0.5873 | 0.5873 | 0.8393 | 6,335.78 | 24,713.7 |

Additional verified findings (details in [`reports/experimental_results.md`](reports/experimental_results.md) and [`reports/model_comparison_report.md`](reports/model_comparison_report.md)):
- **Isolated Single-Flow Preprocessing Latency:** `867.52 µs` mean (`820.70 µs` median, `1,052.05 µs` P95; `1,152.7` flows/sec).
- **Imbalance Experiment (Validation Split):** SMOTE (`Macro-F1 = 0.9432`, `Weighted-F1 = 0.9986`) vs. Class Weighting (`Macro-F1 = 0.9446`) vs. Unweighted (`Macro-F1 = 0.9518`).
- **XGBoost Optuna HPO (12 trials):** Improved validation `mlogloss` from `0.0052` to `0.0048` and `Macro-F1` from `0.9458` to `0.9465`.
- **1D-CNN vs. MLP & XGBoost:** On tabular flow statistics without genuine temporal sequence order, 1D-CNN required substantially higher training time (`2,604.57s` vs `361.81s` MLP / `149.38s` XGBoost) and inference latency (`10.13 ms` vs `0.52 ms` XGBoost) while trailing XGBoost by `14.23` Macro-F1 points (`0.8281` vs `0.9704`).
- **SHAP Trade-Off:** `TreeExplainer` explained 500 flows in `0.28s`; MLP `GradientExplainer` ran `2.83x` faster than `KernelExplainer` with `0.7860` Spearman rank agreement.

---

## 11. Dashboard Execution

```powershell
.venv\Scripts\python.exe scripts/run_dashboard.py
# Or directly:
.venv\Scripts\streamlit.exe run app.py
```

The 7 SOC tabs include: **Overview** (KPIs, Detection Log, Analytics, Threat Map), **Dataset Explorer**, **Model Performance**, **Predict Flow** (single-flow + schema-validated batch CSV), **Explainability** (global + live local SHAP), **Active Response** (dry-run simulator + incident export), and **System Settings**.

---

## 12. Project Structure

```text
IDS_Final/
├── app.py                          # Streamlit SOC dashboard entrypoint
├── pyproject.toml                  # Build metadata, Ruff & pytest configuration
├── requirements.txt                # Locked compatible dependency versions
├── .env.example                    # Environment configuration template
├── configs/
│   ├── config.yaml                 # Data, evaluation, response & UI config
│   └── model_config.yaml           # Model architectures, Optuna & ensemble config
├── docs/
│   ├── architecture.md             # Mermaid architecture & pipeline diagrams
│   ├── dataset_and_methodology.md  # Stack compatibility & scientific methodology
│   └── model_configuration.md      # Configuration schema reference
├── data/
│   ├── raw/                        # Place genuine CICIDS2017 CSVs here
│   ├── processed/                  # Preprocessed splits & preprocessing_audit.json
│   └── synthetic/                  # Explicitly labeled synthetic demo CSV
├── notebooks/                      # Modular notebooks 01 through 05
├── src/
│   ├── config.py                   # Pydantic configuration loader
│   ├── data/                       # loader, cleaner, feature_engineering, preprocessing
│   ├── models/                     # mlp, cnn, autoencoder, xgboost_model, ensemble
│   ├── training/                   # trainer, callbacks, optimization
│   ├── evaluation/                 # metrics, benchmarking, visualizations, report_writer
│   ├── explainability/             # shap_explainer
│   ├── detection/                  # inference, anomaly_detection, threat_scoring
│   ├── response/                   # response_engine (canonical shared mitigation)
│   └── utils/                      # logging, reproducibility, model_registry
├── dashboard/                      # Modular Streamlit components, pages & CSS theme
├── models/                         # Saved .keras, .json, .pkl model artifacts
├── artifacts/                      # Run manifest, training diagnostics & SHAP JSON
├── reports/                        # model_comparison.csv, evaluation_summary.json & figures/
├── scripts/                        # train.py, evaluate.py, run_dashboard.py
└── tests/                          # Comprehensive pytest suite
```

---

## 13. Troubleshooting

- **TensorFlow uses CPU on Windows:** Official TensorFlow builds (`>=2.11`) dropped native Windows GPU support. Use WSL2 Ubuntu with `tensorflow[and-cuda]` if TensorFlow GPU acceleration is required; XGBoost automatically uses CUDA natively on Windows.
- **Uploaded CSV rejected in Predict Flow:** Ensure all 77 numeric CICFlowMeter columns are present (header whitespace and extra metadata columns like `Source IP` / `Label` are handled automatically).
- **Dashboard shows "Not found" for models:** Run `.venv\Scripts\python.exe scripts/train.py --run-eval` first so artifacts populate `models/` and `reports/`.

---

## 14. Limitations

- **Academic Dry-Run Firewall Scope:** Active response executes strictly in dry-run simulation mode (`iptables` rules are logged, never applied to the host kernel).
- **Offline Batch Telemetry:** Live packet capture (`libpcap`/`eBPF`) and streaming CICFlowMeter extraction are outside project scope.
- **Unseen-Attack Overlap:** The unsupervised Autoencoder relies on reconstruction MSE divergence; stealthy attacks whose flow statistics closely resemble normal HTTP traffic (e.g., the 26 held-out `Web Attack` flows) exhibited `0.0000` recall at the 95th-percentile threshold (`0.5440` ROC-AUC) when isolated, whereas global anomaly detection across all attack classes achieved `0.8393` ROC-AUC (`0.5132` recall at `0.0521` FPR).

---

## 15. Future Improvements

1. Streaming ingestion via Kafka/eBPF flow collectors.
2. Platform-agnostic SOAR/cloud security-group mitigation adapters.
3. Continual online drift monitoring and threshold recalibration.
4. ONNX Runtime / TensorRT export for sub-millisecond neural inference.
