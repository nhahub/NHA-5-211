# SMART IDS v3

[![CI](https://github.com/nhahub/NHA-5-211/actions/workflows/ci.yml/badge.svg)](https://github.com/nhahub/NHA-5-211/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Multi-Model Network Intrusion Detection with Zero-Day Anomaly Tracking**

Academic Graduation Project — Built to the accompanying project technical specification (provided separately outside this repository) with split-isolated CICIDS2017 preprocessing (post-clipping cross-split deduplication and `strict_leakage_check` enforcement), CUDA/CPU adaptive XGBoost + TensorFlow/Keras models, classical anomaly baselines, SHAP explainability, dry-run active mitigation, CI automation, and a 7-tab Streamlit SOC dashboard.

---

## Summary & Architecture

**SMART IDS v3** integrates supervised intrusion attribution, unsupervised novelty tracking, explainable AI (XAI), and safe dry-run mitigation across modular packages:

- **Supervised Models:** `XGBoost` (CUDA/CPU, Optuna TPE), `MLP` (`512→256→128→64`), and `1D-CNN` evaluated independently and compared against ensemble rules (`best_single`, `val_weighted_voting`, validation `stacking`, and `hybrid_anomaly_override`, defaulting to standalone `XGBoost` when multi-model fusion shows no statistically significant gain).
- **Orthogonal Zero-Day Tracking:** `Autoencoder` trained exclusively on `BENIGN` traffic and benchmarked against **Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor** across identical benign validation percentiles (`90, 95, 97, 99`) plus a Leave-One-Attack-Out unseen-attack protocol.
- **Zero-Leakage Pipeline:** Stateless cleaning precedes stratified (`random`), chronological per-class (`time_block`), or open-set cross-day (`by_day`) splitting; imputers, 5-sigma clippers, scalers, and SMOTE fit strictly on `train`, followed by post-clipping/post-scaling overlap removal enforced by `strict_leakage_check`.
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

> **Dataset Provenance:** All 8 daily CICIDS2017 CSV exports in `data/raw/` (`138,190` train pre-SMOTE [`none` selected on `0.9518` vs `0.9518` tie within `0.0020` tolerance] / `24,112` validation / `28,324` held-out test; `642` post-clipping/scaling cross-split collisions removed; `leakage_checks.passed = true`). Full tables live in [`reports/model_comparison_report.md`](reports/model_comparison_report.md), [`reports/experimental_results.md`](reports/experimental_results.md), and [`reports/model_comparison.csv`](reports/model_comparison.csv).

| Model | Framing | Accuracy | Precision (W) | Recall (W) | F1 (W) | F1 (Macro) | ROC-AUC | Median / P95 Latency (µs) | Batch Throughput (flows/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Baseline)** | 13-class supervised | **0.9989** | **0.9989** | **0.9989** | **0.9988** | **0.9603** | **0.9995** | 444.40 / 743.52 | **257,069.0** |
| **1D-CNN** | 13-class supervised | 0.9946 | 0.9936 | 0.9946 | 0.9940 | 0.8499 | 0.9853 | 721.15 / 833.39 | 17,997.8 |
| **MLP** | 13-class supervised | 0.9780 | 0.9762 | 0.9780 | 0.9767 | 0.6728 | 0.9905 | 169.95 / **247.89** | 24,278.1 |
| **Autoencoder** | Binary anomaly (Benign vs Anomalous) | 0.8678 | 0.6890 | 0.5130 | 0.5881 | 0.5881 | 0.8384 | 201.60 / 318.80 | 23,245.0 |
| **One-Class SVM** | Binary anomaly (Benign vs Anomalous) | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5898 | 0.7467 | **142.25** / 288.95 | 22,511.5 |
| **LOF** | Binary anomaly (Benign vs Anomalous) | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4647 | 0.8607 | 1,122.45 / 1,569.15 | 41,648.9 |
| **Isolation Forest** | Binary anomaly (Benign vs Anomalous) | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4394 | 0.7710 | 7,671.55 / 8,682.40 | 63,630.7 |

- **Imbalance Selection, CV & Stacking Significance:** Because unweighted (`none`) and `smote` tied at `0.9518` validation Macro-F1 (within `imbalance_tie_tolerance=0.002`), tie-breaking selected the simplest strategy **`none`** (`1D-CNN` Test Macro-F1 rose to `0.8499`, while `MLP` fell to `0.6728` without oversampling). Under `none`, unified 5-fold stratified CV on the `162,577`-row dev pool yielded `0.9535 ± 0.0121` Macro-F1 for baseline `XGBoost`; Optuna's `+0.0059` val Macro-F1 delta (`0.9504 -> 0.9563`) fell inside `0.0121`, so gating retained `baseline` (`95%` test bootstrap CI `[0.9013, 0.9849]`). Validation temperature scaling found `T = 1.0142` (`MLP`), `0.9210` (`1D-CNN`), and `0.9095` (`XGBoost`). Validation `stacking` (`0.9606`) is comparable to `XGBoost` (`0.9603`), difference not statistically significant (paired bootstrap `n=1000`: observed delta `+0.0003`, mean diff `+0.0004`, `95%` CI `[-0.0069, +0.0086]` includes `0`), while voting variants (`0.8420–0.9271`) scored significantly lower.
- **Chronological Drift (`time_block`):** Splitting each class chronologically (earliest ~60% train / middle ~20% val / latest ~20% test so every class remains in training, `leakage_checks.passed = true`) shifted `XGBoost` Test Macro-F1 from `0.9603` (`random`) to `0.7518` (`95%` CI `[0.7098, 0.8250]`, Accuracy `0.9924`), `1D-CNN` to `0.4376` (`[0.4265, 0.4787]`, Accuracy `0.9521`), `MLP` to `0.2767` (`[0.2752, 0.3004]`, Accuracy `0.9502`), and benign-only `Autoencoder` to binary `F1 = 0.5003` (ROC-AUC `0.8263`), revealing temporal drift in low-support and multi-phase classes (`Bot`, `Heartbleed`, `Web Attack`).
- **Open-Set Cross-Day Detection (`by_day`):** By construction in CICIDS2017, all Thursday–Friday attack classes (`Bot`, `DDoS`, `Infiltration`, `PortScan`, `Web Attack`) never appear in Monday–Wednesday training data, making closed-set multi-class Macro-F1 inapplicable by design rather than ordinary generalization loss. Under binary novelty evaluation against unseen attacks, `XGBoost (1 - P_Benign)` reached `F1 = 0.4467` (precision `0.9991`, recall `0.2876` — missing `~71%` of unseen attack flows), `MLP (1 - P_Benign)` `0.4417` (precision `0.9942`, recall `0.2839`), and benign-only `Autoencoder` `F1 = 0.4676` (`ROC-AUC = 0.8784`).

### Evaluation Caveats

- **Random Flow-Level Split Optimism (`random` vs. `time_block` & `by_day`):** Random flow splitting (`data.split_strategy: random`) mixes adjacent flows from identical capture bursts across splits, yielding optimistic supervised metrics. `configs/config.yaml` supports both closed-set chronological drift (`split_strategy: time_block`, earliest ~60% / middle ~20% / latest ~20% per class) and open-set cross-day (`split_strategy: by_day`, Mon–Wed train, part of Thu val, Thu+Fri unseen attacks), both reported alongside `random` in [`reports/model_comparison_report.md`](reports/model_comparison_report.md).
- **Shortcut-Feature Risk (`Destination Port`):** Global SHAP on `XGBoost` is dominated by `Destination Port` (`0.8597` vs. `0.3706` for runner-up `Init_Win_bytes_backward`). Setting `data.drop_features: ["Destination Port"]` tests generalization without port memorization while preserving port metadata for the SOC map. When evaluated apples-to-apples on the identical `28,324` held-out test rows (verified via `sha256` row-index parity after full 77-feature deduplication), removing `Destination Port` shifts `XGBoost` Test Macro-F1 from `0.9603` to `0.9540` (`delta = -0.0063`).
- **Statistically Unreliable Rare Classes:** `Heartbleed` (`2` test / `11` total), `Infiltration` (`5` test / `36` total), `Bot` (`21` test / `137` total), and `Web Attack` (`26` test / `176` total) have `< 30` held-out test flows and are flagged `Statistically Unreliable` alongside per-class support, 95% bootstrap CIs, and 5-fold stratified CV in `reports/model_comparison_report.md`.
- **Known CICIDS2017 Labeling Issues (Engelen et al.):** Standard CICIDS2017 CSVs contain upstream flow-construction and labeling artifacts; treat cross-benchmark generalization cautiously without validating on corrected releases.

---

## How to Reproduce

To verify code style, tests, and end-to-end training reproducibility from a clean checkout:

```powershell
# Lint, formatting verification, and unit/smoke tests (or regenerate reports/test_report.md)
.venv\Scripts\ruff.exe check src dashboard scripts tests app.py
.venv\Scripts\ruff.exe format --check src dashboard scripts tests app.py
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
.venv\Scripts\python.exe scripts/generate_test_report.py

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
├── scripts/                   # train.py, evaluate.py, run_dashboard.py, export_onnx.py, generate_test_report.py
├── models/ & reports/         # Saved artifacts, CSV/JSON metrics, reports & figures
└── tests/                     # Pytest suite (pipeline, leakage, models, dashboard, CLI)
```

---

## Limitations, Security & License

- **Artifact Deserialization Security (`*.pkl` via `joblib`):** Serialized model and preprocessing files (`*.pkl` loaded via `joblib`) execute arbitrary code on deserialization and **must only be loaded from trusted sources**.

- **Dry-Run Mitigation Only:** `ActiveResponseEngine` logs simulated `iptables` rules without altering host firewall state.
- **Offline Batch Telemetry:** Live packet capture (`libpcap`/`eBPF`) and streaming CICFlowMeter extraction are outside project scope.
- **Open-Set Unseen-Attack Miss Rate:** On open-set cross-day evaluation (`by_day`), supervised models detect only `~28.8%` of unseen attack flows (`XGBoost (1 - P_Benign)` recall `0.2876` at precision `0.9991`; binary `XGBoost` recall `0.2690` at precision `0.9981`; `MLP` recall `0.2839` at precision `0.9942`); anomaly detectors are therefore kept as a supplementary zero-day signal.
- **Supplementary Anomaly Detection:** Unsupervised detectors (`Autoencoder`, `IsolationForest`, `OneClassSVM`, `LOF`) struggle on stealthy application-layer flows mimicking benign HTTP statistics and act purely as supplementary zero-day alerts alongside supervised `XGBoost`.
- **License:** Distributed under the terms of the [MIT License](LICENSE).
