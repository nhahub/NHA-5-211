# SMART IDS v3

**An Intelligent, Multi-Modal Hybrid Ensemble Architecture for Network Intrusion Detection and Zero-Day Anomaly Tracking**

Academic Graduation Project — Built to the accompanying project technical specification (provided separately outside this repository) with split-isolated CICIDS2017 preprocessing (post-clipping cross-split deduplication and `strict_leakage_check` enforcement), CUDA/CPU adaptive XGBoost + TensorFlow/Keras models, classical anomaly baselines, SHAP explainability, dry-run active mitigation, CI automation, and a 7-tab Streamlit SOC dashboard.

---

## Summary & Architecture

**SMART IDS v3** integrates supervised intrusion attribution, unsupervised novelty tracking, explainable AI (XAI), and safe dry-run mitigation across modular packages:

- **Supervised Models:** `XGBoost` (CUDA/CPU, Optuna TPE), `MLP` (`512→256→128→64`), and `1D-CNN` evaluated independently and via calibration-aware ensembling (`best_single`, `val_weighted_voting`, validation `stacking`, and `hybrid_anomaly_override`).
- **Orthogonal Zero-Day Tracking:** `Autoencoder` trained exclusively on `BENIGN` traffic and benchmarked against **Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor** across identical benign validation percentiles (`90, 95, 97, 99`) plus a Leave-One-Attack-Out unseen-attack protocol.
- **Zero-Leakage Pipeline:** Stateless cleaning precedes stratified (`random`) or temporal (`by_day`) splitting; imputers, 5-sigma clippers, scalers, and SMOTE fit strictly on `train`, followed by post-clipping/post-scaling overlap removal enforced by `strict_leakage_check`.
- **Canonical Response & SOC UI:** Shared dry-run `ActiveResponseEngine` with `ipaddress` validation and append-only audit logs surfaced through `app.py`.

See [`docs/architecture.md`](docs/architecture.md), [`docs/dataset_and_methodology.md`](docs/dataset_and_methodology.md), [`docs/model_configuration.md`](docs/model_configuration.md), and [`docs/project_details.md`](docs/project_details.md) for detailed layer specs, dependency compatibility notes (`Python 3.13.12`), troubleshooting, and future roadmap.

---

## Quick Start

### Local Setup (Python 3.13)

```powershell
# 1. Create and populate Python 3.13 virtual environment
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# 2. Place official CICIDS2017 CSVs into data/raw/ (or omit for explicit synthetic demo mode)
# Source: https://www.unb.ca/cic/datasets/ids-2017.html

# 3. Train all models and regenerate evaluation reports & figures
.venv\Scripts\python.exe scripts/train.py --run-eval

# 4. Launch the SOC dashboard
.venv\Scripts\streamlit.exe run app.py
```

### Docker Execution (Dashboard)

```bash
docker build -t smart-ids-v3 .
docker run --rm -p 8501:8501 smart-ids-v3
```

### Independent Evaluation & Optional ONNX Export

```powershell
# Re-evaluate persisted models on disk and regenerate reports/
.venv\Scripts\python.exe scripts/evaluate.py

# Optional ONNX export check (skips gracefully if optional tf2onnx/onnx are not installed)
.venv\Scripts\python.exe scripts/export_onnx.py --verify
```

> **ONNX Export Feasibility:** `onnx` and `tf2onnx` are omitted from `requirements.txt` to avoid resolver conflicts on Python 3.13 + TensorFlow `2.21.0` + NumPy `2.x`. Single-flow latency paths in [`src/evaluation/benchmarking.py`](src/evaluation/benchmarking.py) and [`src/detection/inference.py`](src/detection/inference.py) instead use warmed concrete `tf.function` graphs (Keras) and `booster.inplace_predict(...)` (XGBoost). See [`docs/project_details.md`](docs/project_details.md#4-onnx-export-feasibility-notes).

---

## Measured Experimental Results & Evaluation Caveats

### Measured Results (Genuine CICIDS2017 Benchmark Run: 2,830,743 raw rows → 200,000 stratified sample → 191,268 deduplicated rows; 77 features, 13 classes)

> **Dataset Provenance:** All 8 daily CICIDS2017 CSV exports in `data/raw/` (`138,190` train pre-SMOTE / `413,056` train post-SMOTE / `24,387` validation / `28,691` held-out test). Full tables (per-class precision/recall/F1 + support, 95% bootstrap CIs, 5-fold XGBoost CV, `by_day` temporal protocol, and `Destination Port` ablation) live in [`reports/model_comparison_report.md`](reports/model_comparison_report.md) and [`reports/experimental_results.md`](reports/experimental_results.md).

| Model | Framing | Accuracy | Precision (W) | Recall (W) | F1 (W) | F1 (Macro) | ROC-AUC | Mean Latency (µs) | Batch Throughput (flows/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Optuna)** | 13-class supervised | **0.9989** | **0.9989** | **0.9989** | **0.9989** | **0.9704** | **0.9995** | **516.82** | **263,650.3** |
| **1D-CNN** | 13-class supervised | 0.9849 | 0.9945 | 0.9849 | 0.9892 | 0.8281 | 0.9979 | 10,128.19 | 16,945.0 |
| **MLP** | 13-class supervised | 0.9633 | 0.9785 | 0.9633 | 0.9692 | 0.8076 | 0.9965 | 5,054.23 | 23,584.7 |
| **Autoencoder** | Binary anomaly (Benign vs Anomalous) | 0.8688 | 0.6863 | 0.5132 | 0.5873 | 0.5873 | 0.8393 | 6,335.78 | 24,713.7 |

### Evaluation Caveats

- **Random Flow-Level Split Optimism (`random` vs. `by_day`):** Random flow splitting (`data.split_strategy: random`) shares flows from the same capture sessions across splits, yielding optimistic supervised metrics. `configs/config.yaml` supports chronological `split_strategy: by_day` (Mon–Wed train, part of Thu val, Thu+Fri test), reported separately alongside `random` in `reports/model_comparison_report.md`.
- **Shortcut-Feature Risk (`Destination Port`):** Baseline XGBoost global SHAP is dominated by `Destination Port` (`1.93` vs. `0.55` runner-up). Setting `data.drop_features: ["Destination Port"]` tests generalization without port memorization while preserving port metadata for the SOC map.
- **Statistically Unreliable Rare Classes:** `Heartbleed` (11 total) and `Infiltration` (36 total) yield tiny test supports (`< 30` flows); inspect per-class support, 95% bootstrap CIs, and 5-fold stratified CV in `reports/model_comparison_report.md`.
- **Known CICIDS2017 Labeling Issues (Engelen et al.):** Standard CICIDS2017 CSVs contain upstream flow-construction and labeling artifacts; treat cross-benchmark generalization cautiously without validating on corrected releases.

---

## How to Reproduce

To verify code style, tests, and end-to-end training reproducibility from a clean checkout:

```powershell
# Lint, formatting verification, and unit/smoke tests
.venv\Scripts\ruff.exe check src dashboard scripts tests app.py
.venv\Scripts\ruff.exe format --check src dashboard scripts tests app.py
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short

# End-to-end training + evaluation (synthetic quick sanity run or genuine data/raw/*.csv)
.venv\Scripts\python.exe scripts/train.py --synthetic --max-rows 20000 --epochs 5 --optuna-trials 3 --run-eval
```

All PRs and pushes to `main` run these Ruff and pytest checks automatically via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Repository Layout

```text
NHA-5-211/
├── .github/workflows/ci.yml   # GitHub Actions CI (Ruff + pytest on Python 3.13)
├── Dockerfile                 # Containerized Streamlit SOC deployment (port 8501)
├── LICENSE                    # MIT License
├── app.py                     # Streamlit SOC dashboard entrypoint (7 tabs)
├── configs/                   # YAML pipeline & hyperparameter configs
├── docs/                      # Architecture, methodology, specs & extended notes
├── data/                      # raw/ (CICIDS2017 CSVs), processed/, synthetic/
├── src/                       # Modular data, models, training, eval, XAI & response
├── dashboard/                 # Modular Streamlit pages, components & theme
├── scripts/                   # train.py, evaluate.py, run_dashboard.py, export_onnx.py
├── models/ & reports/         # Saved artifacts, CSV/JSON metrics, reports & figures
└── tests/                     # Pytest suite (pipeline, leakage, models, dashboard, CLI)
```

---

## Limitations & License

- **Dry-Run Mitigation Only:** `ActiveResponseEngine` logs simulated `iptables` rules without altering host firewall state.
- **Offline Batch Telemetry:** Live packet capture (`libpcap`/`eBPF`) and streaming CICFlowMeter extraction are outside project scope.
- **Supplementary Anomaly Detection:** Unsupervised detectors (`Autoencoder`, `IsolationForest`, `OneClassSVM`, `LOF`) struggle on stealthy application-layer flows mimicking benign HTTP statistics and act purely as supplementary zero-day alerts alongside supervised `XGBoost`.
- **License:** Distributed under the terms of the [MIT License](LICENSE).
