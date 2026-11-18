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

> **Dataset Provenance:** All 8 daily CICIDS2017 CSV exports in `data/raw/` (`138,190` train pre-SMOTE / `145,714` train post-SMOTE with `cap_per_class=5,000`, `max_multiplier=5.0`, `min_real_samples=50` / `24,112` validation / `28,324` held-out test; `642` post-clipping/scaling cross-split collisions removed; `leakage_checks.passed = true`). Full tables live in [`reports/model_comparison_report.md`](reports/model_comparison_report.md), [`reports/experimental_results.md`](reports/experimental_results.md), and [`reports/model_comparison.csv`](reports/model_comparison.csv).

| Model | Framing | Accuracy | Precision (W) | Recall (W) | F1 (W) | F1 (Macro) | ROC-AUC | Mean / Median Latency (µs) | Batch Throughput (flows/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Baseline)** | 13-class supervised | **0.9989** | **0.9989** | **0.9989** | **0.9988** | **0.9594** | **0.9996** | 2,301.30 / **514.00** | **234,907.3** |
| **1D-CNN** | 13-class supervised | 0.9904 | 0.9925 | 0.9904 | 0.9913 | 0.8447 | 0.9933 | 978.05 / 943.45 | 9,001.1 |
| **MLP** | 13-class supervised | 0.9767 | 0.9784 | 0.9767 | 0.9772 | 0.7592 | 0.9942 | **281.89** / **240.75** | 15,585.5 |
| **Autoencoder** | Binary anomaly (Benign vs Anomalous) | 0.8678 | 0.6890 | 0.5130 | 0.5881 | 0.5881 | 0.8384 | 693.31 / 671.85 | 16,344.5 |
| **One-Class SVM** | Binary anomaly (Benign vs Anomalous) | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5898 | 0.7467 | 251.27 / 164.10 | 18,114.6 |
| **LOF** | Binary anomaly (Benign vs Anomalous) | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4647 | 0.8607 | 1,598.18 / 1,401.95 | 35,777.2 |
| **Isolation Forest** | Binary anomaly (Benign vs Anomalous) | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4394 | 0.7710 | 9,790.02 / 9,254.10 | 65,464.5 |

- **Cross-Validation, Calibration & Stacking:** 5-fold stratified CV on `162,577` dev rows gave `0.9670 ± 0.0076` Macro-F1 for `XGBoost` (`95%` test bootstrap CI `[0.9000, 0.9833]`). Optuna tuning (`+0.0120` val Macro-F1) stayed within baseline CV std (`0.0470`), so CV-std gating retained `baseline`. Validation temperature scaling found `T = 0.8777` (`MLP`), `0.8749` (`1D-CNN`), and `0.9429` (`XGBoost`); validation log-probability `stacking` reached Test Macro-F1 **`0.9661`** (`+0.0067` over `XGBoost`), while unweighted/weighted probability voting trailed standalone `XGBoost` (`0.9000–0.9284`).
- **Temporal Generalization (`by_day`):** With Mon–Wed training and Thu–Fri unseen attack classes (`Bot`, `DDoS`, `Infiltration`, `PortScan`, `Web Attack`), closed-set multi-class Macro-F1 dropped to `0.0782` (`XGBoost`), while cross-day binary detection reached `F1 = 0.4676` (`Autoencoder`, ROC-AUC `0.8784`) and `0.4435` (`XGBoost 1 - P_Benign`, precision `0.9989`, recall `0.2851`).

### Evaluation Caveats

- **Random Flow-Level Split Optimism (`random` vs. `by_day`):** Random flow splitting (`data.split_strategy: random`) shares flows from the same capture sessions across splits, yielding optimistic supervised metrics. `configs/config.yaml` supports chronological `split_strategy: by_day` (Mon–Wed train, part of Thu val, Thu+Fri test), reported separately alongside `random` in `reports/model_comparison_report.md`.
- **Shortcut-Feature Risk (`Destination Port`):** Baseline XGBoost global SHAP is dominated by `Destination Port` (`1.0796` vs. `0.3769` for runner-up `min_seg_size_forward`). Setting `data.drop_features: ["Destination Port"]` tests generalization without port memorization while preserving port metadata for the SOC map (ablating it shifts XGBoost Test Macro-F1 from `0.9594` to `0.9500`, `delta = -0.0094`).
- **Statistically Unreliable Rare Classes:** `Heartbleed` (`2` test / `11` total), `Infiltration` (`5` test / `36` total), `Bot` (`21` test / `137` total), and `Web Attack` (`26` test / `176` total) have `< 30` held-out test flows and are flagged `Statistically Unreliable` alongside per-class support, 95% bootstrap CIs, and 5-fold stratified CV in `reports/model_comparison_report.md`.
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
