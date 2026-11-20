# SMART IDS v3

[![CI](https://github.com/nhahub/NHA-5-211/actions/workflows/ci.yml/badge.svg)](https://github.com/nhahub/NHA-5-211/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**SMART IDS v3** is a modular CICIDS2017 network intrusion detection system pairing a production `XGBoost` multi-class classifier (alongside `MLP` and `1D-CNN` neural baselines) with benign-only zero-day anomaly tracking (`Autoencoder`, `VAE`, `One-Class SVM`, `Isolation Forest`, `LOF`), split-isolated preprocessing (`strict_leakage_check`), SHAP explainability, dry-run firewall response previews, and a 7-tab Streamlit SOC dashboard (`app.py`). Detailed architecture, preprocessing, model specs, and extended benchmarks live in [`docs/`](docs/results_details.md).

---

## Results Overview (`200k` CICIDS2017 Sample: `2,830,743` Raw → `191,268` Clean Flows, 77 Features, 13 Classes)

| Model | Imbalance | `random` `F1_Macro` (95% CI) | `time_block` `F1_Macro` (95% CI) | `by_day` Closed-Set `F1_Macro` |
|---|---|---:|---:|---:|
| **XGBoost (Rec.)** | `none` | **0.9603** `[0.9013, 0.9849]` | **0.7571** `[0.7087, 0.8316]` | **0.0860** |
| **1D-CNN (Baseline)** | `smote` | 0.8447 `[0.7649, 0.8819]` | 0.3073 `[0.3019, 0.3359]` | 0.0691 |
| **MLP (Baseline)** | `smote` | 0.7934 `[0.7544, 0.8176]` | 0.2770 `[0.2756, 0.3011]` | 0.0661 |

> All-class `F1_Macro` is the headline metric throughout; per-protocol `F1_Macro_Reliable` (`0.9972` / `0.9562` / `0.9286` on `random`), which excludes rare classes with `< 30` test flows, appears only alongside `F1_Macro` in [`docs/results_details.md`](docs/results_details.md) and [`reports/model_comparison_report.md`](reports/model_comparison_report.md).

### Key Findings

1. **Production Classifier:** `XGBoost` (`none`) leads `random` (`0.9603`) at `252,775` flows/s batch; `40` Optuna trials and `stacking` (`0.9725`, CI crossing `0`) brought no statistically significant gain, so baseline `XGBoost` remains default.
2. **Per-Model Imbalance Selection:** Evaluating `{none, class_weight, smote}` per model kept `XGBoost` on `none` and switched `MLP` and `1D-CNN` to `smote`, lifting `MLP` test `F1_Macro` from `0.6728` to `0.7934`.
3. **Drift & Unseen Attacks:** With real `Timestamp` aligned via `flow_telemetry_signature` (`2,830,471 / 2,830,743`, `99.99%` matched), `XGBoost` `time_block` `F1_Macro` shifted only slightly vs. the earlier row-order proxy (`0.7518` → `0.7571`, within CI) while `1D-CNN` dropped (`0.4376` → `0.3073`), showing neural baselines are unstable across splits; `Bot` recall remains `0.000` (`28/28` predicted `BENIGN`, `KS = 1.0` on `Flow Duration`), so it is real drift, not an ordering artifact, and `by_day` `XGBoost (1 - P_Benign)` detects only `28.76%` of unseen attacks (`F1 = 0.4467`).
4. **Deployment Profiles:** Saved `models/xgb_model.json` uses `'none'` (`time_block` Benign FPR `0.0002`, focus recall `0.4547`); `'class_weight'` **requires retraining** (`0.8646` `F1_Macro`, `0.7968` focus recall at `~1.4%` FPR), and `Bot` recall stays `0.000` on `time_block` (`~0.015` max on `by_day`).
5. **Anomaly Detectors (Supplementary):** Benign-only `Autoencoder` (`F1 = 0.6926`, `ROC-AUC = 0.8676`) leads all 6 individual baselines, while `Score-Level Fusion (degenerate: equals Autoencoder)` adds zero fusion gain (`weights: Autoencoder=1.00`, unmeasured cols `N/A`).

---

## Quick Start

### Local (Python 3.13)

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# Place the 8 CICIDS2017 CSVs into data/raw/ (or omit for synthetic demo mode)
.venv\Scripts\python.exe scripts/train.py --run-eval
.venv\Scripts\python.exe scripts/evaluate.py --time-block-only   # Optional: rerun time_block only
.venv\Scripts\streamlit.exe run app.py
```

### Docker & Git LFS Model Artifacts

```bash
git lfs install && git lfs pull   # Or: python scripts/download_models.py --repo <owner>/<repo> --tag models-v3
docker build -t smart-ids-v3 .    # Add --build-arg REQUIRE_MODELS=true to fail if LFS stubs remain
docker run --rm -p 8501:8501 smart-ids-v3
```

- Core CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `ruff check`, `ruff format --check`, and `pytest` with `lfs: false` (skipping binary smoke tests cleanly when LFS pointers are unfetched; full results in [`reports/test_report.md`](reports/test_report.md)).

---

## Limitations

- **Random Split Optimism:** Random flow splitting mixes burst neighbors, inflating supervised scores vs. chronological `time_block` (`0.7571`) and open-set `by_day` (`0.0860`). See [`docs/evaluation_caveats.md`](docs/evaluation_caveats.md) and [`docs/optional_data.md`](docs/optional_data.md).
- **Rare-Class Instability & Upstream Labels:** `Heartbleed`, `Infiltration`, `Bot`, and `Web Attack` have `< 30` test flows (`< 30` reliability threshold), and Engelen et al. (2021) label corrections were absent/not applied ([`docs/optional_data.md`](docs/optional_data.md)).
- **Shortcut-Feature & Zero-Day Limits:** Global SHAP relies strongly on `Destination Port` (`delta = -0.0063` when dropped), `by_day` misses `71.24%` of unseen attacks, and unsupervised detectors (`AE`/`VAE`/`OCSVM`/`IF`/`LOF`) are supplementary signals only.
- **Offline Batch & Dry-Run Response Scope:** Live `libpcap`/`eBPF` capture is out of scope, and `ActiveResponseEngine` outputs `NOT EXECUTED (dry-run)` firewall command previews rather than executing host rules ([`docs/evaluation_caveats.md`](docs/evaluation_caveats.md)).

---

## Security & License

- **Artifact Deserialization & Integrity:** Deserializing `*.pkl` files via `joblib` executes arbitrary code. `models/MANIFEST.json` protects against corrupted or accidentally swapped artifacts, but is NOT a defense against a malicious actor who can modify both the model and the manifest; only load artifacts from trusted sources.
- **License:** Distributed under the [MIT License](LICENSE). Full metric tables & caveats: [`docs/results_details.md`](docs/results_details.md) · [`docs/evaluation_caveats.md`](docs/evaluation_caveats.md) · [`docs/architecture.md`](docs/architecture.md) · [`docs/project_details.md`](docs/project_details.md).
