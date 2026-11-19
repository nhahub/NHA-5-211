# SMART IDS v3

[![CI](https://github.com/nhahub/NHA-5-211/actions/workflows/ci.yml/badge.svg)](https://github.com/nhahub/NHA-5-211/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Multi-Model Network Intrusion Detection with Zero-Day Anomaly Tracking**

Academic Graduation Project — Built to the accompanying project technical specification (provided separately outside this repository) with split-isolated CICIDS2017 preprocessing (post-clipping cross-split deduplication and `strict_leakage_check` enforcement), CUDA/CPU adaptive XGBoost + TensorFlow/Keras models, classical anomaly baselines, SHAP explainability, dry-run active mitigation, CI automation, and a 7-tab Streamlit SOC dashboard.

---

## Summary & Architecture

**SMART IDS v3** integrates supervised intrusion attribution, unsupervised novelty tracking, explainable AI (XAI), and safe dry-run mitigation across modular packages:

- **Supervised Models:** **`XGBoost`** is the **recommended production classifier** (adaptive CUDA/CPU + CV-std-gated Optuna TPE). **`MLP`** (`512→256→128→64`) and **`1D-CNN`** serve strictly as **neural baselines for comparison**, evaluated independently with per-model imbalance strategy selection and bounded loss/LR regularization trials (`ReduceLROnPlateau`, `label_smoothing`, `focal_loss`), alongside ensemble fusion rules (`best_single`, `val_weighted_voting`, validation `stacking`, and `hybrid_anomaly_override`, defaulting to standalone `XGBoost` when multi-model fusion shows no statistically significant gain).
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
# 1a. Pull Git LFS model binaries first (recommended for full model inference)
git lfs pull

# 1b. Or download model assets from a GitHub Release URL (Git LFS quota-free fallback)
python scripts/download_models.py --repo <owner>/<repo> --tag models-v3

# 2. Standard build (succeeds in degraded demo/report mode if LFS stubs remain, printing a loud warning)
docker build -t smart-ids-v3 .

# 3. Strict production build (aborts Docker build if models/ contains unfetched LFS pointer stubs or missing weights)
docker build --build-arg REQUIRE_MODELS=true -t smart-ids-v3 .

docker run --rm -p 8501:8501 smart-ids-v3
```

## Model Artifacts (Git LFS & Release Fallback)

- Large/retrain-volatile model binaries (`*.keras`, `*.pkl`, `models/xgb_model.json`, `*.h5`, `*.onnx`, `*.npz`, `*.parquet`) are tracked via **Git LFS** in [`.gitattributes`](.gitattributes) so retraining never bloats Git object history going forward (historical commits remain untouched).
- On a fresh clone, fetch the binaries via Git LFS **or** the Release downloader before running inference or strict Docker builds:

```bash
# Option A: Git LFS
git lfs install
git lfs pull

# Option B: GitHub Release fallback (configured via `models_download` in configs/config.yaml)
.venv\Scripts\python.exe scripts/download_models.py --base-url https://github.com/<owner>/<repo>/releases/download/models-v3
```

- If `models/` is empty or contains unfetched Git LFS pointer stubs (for example, a Docker build without `git lfs pull` or an `lfs: false` checkout), the SOC dashboard degrades gracefully with a clear status banner, Dockerfile emits a prominent build-time warning unless `REQUIRE_MODELS=true`, and tests requiring full binaries skip with a clear reason. Core CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `lint-and-test` with `lfs: false` so GitHub LFS bandwidth quotas never block code checks, alongside an optional `workflow_dispatch`-only `lfs-artifact-smoke` job (`lfs: true`).

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

> **Dataset Provenance:** All 8 daily CICIDS2017 CSV exports in `data/raw/` (`138,190` train pre-SMOTE / `24,112` validation / `28,324` held-out test; `642` post-clipping/scaling cross-split collisions removed; `leakage_checks.passed = true` across `random`, `time_block`, and `by_day`). Per-model imbalance selection (`tolerance=0.0020`, `neural_epochs_cap=10`) picked `none` for `XGBoost`, `smote` for `MLP`, and `smote` for `1D-CNN`. Full tables live in [`reports/model_comparison_report.md`](reports/model_comparison_report.md), [`reports/experimental_results.md`](reports/experimental_results.md), and [`reports/model_comparison.csv`](reports/model_comparison.csv).

| Model | Imbalance | Framing | Accuracy | F1 (W) | F1 (Macro) | F1 (Reliable ≥30) | ROC-AUC | Median / P95 (µs) | Batch (flows/s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Baseline, Rec.)** | `none` | 13-class supervised | **0.9989** | **0.9988** | **0.9603** | **0.9972** | **0.9995** | 405.05 / 523.68 | **249,219.2** |
| **1D-CNN (Neural Baseline)** | `smote` | 13-class supervised | 0.9904 | 0.9913 | 0.8447 | 0.9562 | 0.9933 | 623.00 / 724.05 | 16,521.5 |
| **MLP (Neural Baseline)** | `smote` | 13-class supervised | 0.9766 | 0.9772 | 0.7934 | 0.9286 | 0.9962 | 173.85 / 221.99 | 24,457.6 |
| **Autoencoder** | `benign_only` | Binary anomaly | **0.8934** | **0.6926** | **0.6926** | N/A | **0.8676** | 198.70 / 280.10 | 17,682.5 |
| **Score-Level Fusion (degenerate: equals Autoencoder)** | `benign_only` | Binary anomaly (Val Sel.) | 0.8934 | 0.6926 | 0.6926 | N/A | 0.8676 | N/A | N/A |
| **VAE** | `benign_only` | Binary anomaly | 0.8693 | 0.5910 | 0.5910 | N/A | 0.7755 | 208.55 / 268.89 | 24,408.6 |
| **One-Class SVM** | `benign_only` | Binary anomaly | 0.8690 | 0.5898 | 0.5898 | N/A | 0.7467 | **136.45 / 166.06** | 23,264.0 |
| **Autoencoder (Deep Reg)** | `benign_only` | Binary anomaly | 0.8668 | 0.5831 | 0.5831 | N/A | 0.7908 | 233.50 / 308.93 | 22,696.0 |
| **LOF** | `benign_only` | Binary anomaly | 0.8421 | 0.4647 | 0.4647 | N/A | 0.8607 | 1,368.80 / 1,965.23 | 41,327.3 |
| **Isolation Forest** | `benign_only` | Binary anomaly | 0.8385 | 0.4394 | 0.4394 | N/A | 0.7710 | 6,958.00 / 7,768.49 | 79,549.0 |

> Validation selected a single-detector fusion; no gain from fusing detectors on this data.

- **Per-Model Imbalance Selection, Neural Trials, CV, Optuna & Stacking Significance:** Evaluating `{none, class_weight, smote}` per model lifted `MLP` Test Macro-F1 from Round 2's `0.6728` (Bot F1 `0.0000`) to `0.7934` (`smote` val Macro-F1 `0.7695` vs `none` `0.6924`; `1D-CNN` chose `smote` `0.7501`; `XGBoost` stayed `none` `0.9518`). Bounded neural loss regularization trials (`label_smoothing`, `focal_loss`, capped at `3.0` min) underperformed baseline `cross_entropy` (`MLP 0.7138/0.7618` vs `0.8134`; `1D-CNN 0.6562` vs `0.8701`, focal skipped at time budget) and were automatically reverted; both neural models are explicitly positioned as **neural baselines for comparison** with **`XGBoost`** recommended for production. Unified 5-fold CV on the `162,577`-row dev pool yielded `0.9535 ± 0.0121` Macro-F1 (`161,042`-row pooled OOF Macro-F1 `0.9550`, Reliable `0.9561`); `40` Optuna TPE trials reached val Macro-F1 `0.9561` vs `0.9504` (`+0.0056 ≤ 0.0121`), so gating retained `baseline` (`95%` test CI `[0.9013, 0.9849]`). Validation temperature scaling gave `T = 0.7948` (`MLP`), `0.8749` (`1D-CNN`), `0.9095` (`XGBoost`); validation `stacking` reached Test Macro-F1 `0.9725` (`F1_Macro_Reliable = 0.9921`), statistically indistinguishable from standalone `XGBoost` (`0.9603`, delta `+0.0123`, paired bootstrap `95%` CI `[-0.0040, +0.0538]` includes `0`), so `XGBoost` remains the default supervised engine.
- **Chronological Drift (`time_block`):** Evaluating the canonical models on per-class chronological splits (`time_block ordering = CSV row order (no real Timestamp)`, `leakage_checks.passed = true`) shifted `XGBoost` Test Macro-F1 from `0.9603` (`random`) to `0.7518` (`F1_Macro_Reliable = 0.9173`, `95%` CI `[0.7098, 0.8250]`, Accuracy `0.9924`), `1D-CNN` to `0.4376` (`Reliable 0.5689`), `MLP` to `0.2767` (`Reliable 0.3597`), and benign-only `Autoencoder` to binary `F1 = 0.5003` (`ROC-AUC = 0.8263`, `FPR = 0.0589`). Retraining from scratch directly under `split_strategy=time_block` (per-model imbalance `MLP=smote`, `CNN=class_weight`, `XGBoost=smote`, `leakage_checks.passed = true`) yielded `XGBoost` Test Macro-F1 `0.8427` (`Reliable 0.9288`), `MLP` `0.6703` (`Reliable 0.8067`), `1D-CNN` `0.3919` (`Reliable 0.5061`).
- **Open-Set Cross-Day Detection (`by_day`):** By construction in CICIDS2017, all five Thursday–Friday attack families (`Bot`, `DDoS`, `Infiltration`, `PortScan`, `Web Attack`) never appear in Monday–Wednesday training data (closed-set `XGBoost` Macro-F1 `0.0860` / `Reliable 0.1720`; full primary `by_day` retrain `0.0952` / `Reliable 0.1714`, `leakage_checks.passed = true`). Under binary novelty evaluation against unseen attacks, `XGBoost (1 - P_Benign)` reached recall `0.2876` (`F1 = 0.4467`, precision `0.9991`, `FPR = 0.0001` — missing `71.24%` of unseen attack flows), `MLP (1 - P_Benign)` recall `0.2839` (`F1 = 0.4417`), and benign-only `Autoencoder` recall `0.3419` (`F1 = 0.4676`, `ROC-AUC = 0.8784`, `FPR = 0.0549`).
- **Sample-Size Sensitivity (`200k` vs. `500k`):** Scaling `XGBoost` (`none`) from `200,000` requested rows (`191,268` deduplicated, `28,324` test) to `500,000` (`470,048` deduplicated, `69,026` test) shifted Macro-F1 from `0.9603` (`[0.9013, 0.9849]`, Reliable `0.9972`) to `0.9894` (`[0.9837, 0.9943]`, Reliable `0.9875`; `+0.0291` delta, overlapping `95%` CIs, `conclusions_change = False`).

### Evaluation Caveats

- **Random Flow-Level Split Optimism (`random` vs. `time_block` & `by_day`):** Random flow splitting (`data.split_strategy: random`) mixes adjacent flows from identical capture bursts across splits, yielding optimistic supervised metrics. `configs/config.yaml` supports both closed-set chronological drift (`split_strategy: time_block`, earliest ~60% / middle ~20% / latest ~20% per class) and open-set cross-day (`split_strategy: by_day`, Mon–Wed train, part of Thu val, Thu+Fri unseen attacks), both reported alongside `random` in [`reports/model_comparison_report.md`](reports/model_comparison_report.md).
- **Optional Real-Timestamp Alignment for `time_block` (`data.timestamp_source_dir`):** Standard `MachineLearningCVE` CSVs in `data/raw/` omit `Timestamp`, so `time_block` defaults to session + CSV row order (`time_block ordering = CSV row order (no real Timestamp)`). To sort strictly by true capture timestamps, download CICIDS2017 **`GeneratedLabelledFlows` / `TrafficLabelling_.zip`** from the [Canadian Institute for Cybersecurity CICIDS2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html), extract its daily CSVs into `data/TrafficLabelling/` (or configure `data.timestamp_source_dir` in `configs/config.yaml`). When present, [`align_real_timestamps`](src/data/loader.py) merges timestamps via `Flow ID` + 5-tuple (`Source IP`, `Destination IP`, `Source Port`, `Destination Port` + occurrence index) when present, or via deterministic session telemetry signature (`Source_Session_Day`, `Destination Port`, `Flow Duration`, `Total Fwd/Bwd Packets`, `Total Length of Fwd/Bwd Packets`, `Flow IAT Mean`, `Label`, `Occurrence_Index`) for `MachineLearningCVE`, logs matched/unmatched counts, and still excludes `Timestamp` from all model features. *(Note: `data/TrafficLabelling/` was not present locally during Round 3 training; real-timestamp `time_block` ordering is unit-tested on timestamped fixtures and not verified on full CICIDS2017 TrafficLabelling CSVs.)*
- **Shortcut-Feature Risk (`Destination Port`):** Global SHAP on `XGBoost` is dominated by `Destination Port` (`0.8610` vs. `0.3711` for runner-up `Init_Win_bytes_backward`). Setting `data.drop_features: ["Destination Port"]` tests generalization without port memorization while preserving port metadata for the SOC map. When evaluated apples-to-apples on the identical `28,324` held-out test rows (verified via `sha256` row-index parity after full 77-feature deduplication), removing `Destination Port` shifts `XGBoost` Test Macro-F1 from `0.9603` to `0.9540` (`delta = -0.0063`).
- **Statistically Unreliable Rare Classes (`Heartbleed`, `Infiltration`, `Bot`, `Web Attack`):** In CICIDS2017 held-out test partitions (~15% of the `200k` sample, and smaller still in `time_block`), **`Heartbleed`** (`2` test flows / `11` total), **`Infiltration`** (`5`–`7` test), **`Bot`** (`21`–`28` test), and **`Web Attack`** (`26` random test) fall below `evaluation.reliable_support_threshold = 30`. With `< 30` test flows, binomial confidence intervals span tens of percentage points and 1–2 misclassifications swing unweighted `F1_Macro` drastically. Consequently, every table reports **both** all-class `F1_Macro` and **`F1_Macro_Reliable`** (classes with test support `>= 30`) side-by-side, supplemented by a **pooled 5-fold CV Out-of-Fold (OOF)** table (`161,042` flows; `Bot` OOF `n=116` recall `0.6983 [0.6095, 0.7743]`, `Infiltration` `n=31` recall `0.7097 [0.5341, 0.8390]`, `Web Attack` `n=150` recall `0.9267 [0.8735, 0.9586]`, `Heartbleed` `n=9` recall `0.8889 [0.5650, 0.9801]`).
- **Known CICIDS2017 Labeling Issues & Optional Corrected-Label Support (Engelen et al., 2021):** Upstream CICIDS2017 CSVs exhibit flow-construction bugs, duplicated TCP teardown packets, infinite/NaN fields, and label assignments that can misrepresent DoS/Heartbleed/Bot behaviors (Engelen et al., *IEEE S&P Workshops*, 2021). Corrected label files are distributed by Distrinet KU Leuven (`https://intrusion-detection.distrinet-research.be/CNS2021/`). Download and configure `data.label_correction_file` in [`configs/config.yaml`](configs/config.yaml) (default `data/engelen2021_label_corrections.csv`) to relabel/filter matching flows cleanly via [`apply_label_corrections`](src/data/loader.py). *(Note: the external Engelen et al. CSV was absent locally and skipped cleanly during benchmark runs; relabeling is verified via unit-test fixtures rather than reported as a corrected-label benchmark.)*
- **Stratified Downsampling (`data.sample_size` / `data.use_full_dataset`):** Because the combined 8 CICIDS2017 CSVs span `2,830,743` raw flows, default training uses `data.sample_size: 200000` (`data.use_full_dataset: false`) via [`stratified_downsample`](src/data/loader.py): majority/mid-size classes are sampled proportionally while preserving at least `min(class_count, 50)` real rows for rare classes prior to deduplication. Setting `data.use_full_dataset: true` disables downsampling cleanly.
- **Limitations — Temporal Concept Drift, Unseen Attacks & Recall–FPR Trade-off:** Under `time_block`, `XGBoost` fails on `Bot` (`28/28` misclassified as `BENIGN`, recall `0.0000`, KS `1.0000` shift on `Flow Duration` and `Total Length of Fwd Packets`), `Heartbleed` (`2/2` -> `BENIGN`), `Web Attack` (recall `0.4000`, `45.7%` -> `BENIGN`), `Infiltration` (`0.4286`), and `DoS slowloris` (`0.7160`). Switching `XGBoost` to `class_weight` raised `time_block` failing-class mean recall from `0.4276` to `0.7816` (`Macro-F1 = 0.8512`) at Benign FPR `0.0144` vs `0.0001` baseline (`Bot` remained `0.000`), while on `by_day` unseen attacks lowering `1 - P(BENIGN)` at benign validation `P99` (`0.0020`) raised unseen focus recall from `0.1349` to `0.3412` (`FPR = 0.0103`). Full breakdown and KS tables live in [`reports/model_comparison_report.md`](reports/model_comparison_report.md).
- **Unsupervised Anomaly Detectors as Supplementary Zero-Day Signals Only:** Across all 7 benign-only detectors evaluated in [`reports/model_comparison_report.md`](reports/model_comparison_report.md), standard `Autoencoder` (`F1 = 0.6926 [0.6820, 0.7020]`, `ROC-AUC = 0.8676`) outperformed `VAE` (`0.5910`), `One-Class SVM` (`0.5898`), `Autoencoder (Deep Reg)` (`0.5831`), `LOF` (`0.4647`), and `Isolation Forest` (`0.4394`); non-test leave-attack-out validation selected `Score-Level Fusion` with weight `Autoencoder=1.00`. Under `time_block`, Autoencoder binary F1 dropped to `0.5003`. These remain **supplementary zero-day signals**, not replacements for supervised `XGBoost`.

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
├── .gitattributes             # Git LFS tracking for *.keras, *.pkl & large model files
├── .github/workflows/ci.yml   # GitHub Actions CI (lfs:false lint/pytest + manual LFS smoke job)
├── Dockerfile                 # Containerized Streamlit SOC deployment (REQUIRE_MODELS guard)
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

- **Artifact Deserialization Security & SHA-256 Verification (`*.pkl` via `joblib`):** Serialized model and preprocessing files (`*.pkl` loaded via `joblib`) execute arbitrary code on deserialization and **must only be loaded from trusted sources**. During training, `models/MANIFEST.json` is generated with SHA-256 digests of every model artifact; [`ModelRegistry`](src/utils/model_registry.py) (controlled by `security.verify_model_hashes: true` in [`configs/config.yaml`](configs/config.yaml)) verifies each artifact's SHA-256 digest before calling `joblib.load`, refuses to load tampered or mismatched files with `ModelIntegrityError`, and skips unfetched Git LFS pointer stubs gracefully.

- **Dry-Run Mitigation & Non-Executed Command Previews:** `ActiveResponseEngine` runs strictly in `NOT EXECUTED (dry-run)` preview mode, validating source IPs with `ipaddress`, refusing private/loopback/link-local/multicast/reserved/unspecified IP ranges, recording append-only JSON/text audit logs, and surfacing OS-specific firewall command previews (`iptables`/`nftables`, Windows `netsh`/`New-NetFirewallRule`). Live firewall execution is intentionally out of scope for this academic project.
- **Offline Batch Telemetry:** Live packet capture (`libpcap`/`eBPF`) and streaming CICFlowMeter extraction are outside project scope.
- **Open-Set Unseen-Attack Miss Rate:** On open-set cross-day evaluation (`by_day`), supervised models detect only `~28.8%` of unseen attack flows (`XGBoost (1 - P_Benign)` recall `0.2876` at precision `0.9991`; binary `XGBoost` recall `0.2690` at precision `0.9981`; `MLP` recall `0.2839` at precision `0.9942`); anomaly detectors are therefore kept as a supplementary zero-day signal.
- **Supplementary Anomaly Detection:** Unsupervised detectors (`Autoencoder`, `IsolationForest`, `OneClassSVM`, `LOF`) struggle on stealthy application-layer flows mimicking benign HTTP statistics and act purely as supplementary zero-day alerts alongside supervised `XGBoost`.
- **License:** Distributed under the terms of the [MIT License](LICENSE).
