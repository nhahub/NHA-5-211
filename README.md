# SMART IDS v3

[![CI](https://github.com/nhahub/NHA-5-211/actions/workflows/ci.yml/badge.svg)](https://github.com/nhahub/NHA-5-211/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**SMART IDS v3** is a modular CICIDS2017 network intrusion detection system pairing a production `XGBoost` multi-class classifier (alongside `MLP` and `1D-CNN` neural baselines) with benign-only zero-day anomaly tracking (`Autoencoder`, `VAE`, `One-Class SVM`, `Isolation Forest`, `LOF`), split-isolated preprocessing (`strict_leakage_check`), SHAP explainability, dry-run firewall response previews, and a 7-tab Streamlit SOC dashboard (`app.py`). Detailed architecture, preprocessing, model specs, and extended benchmarks live in [`docs/`](docs/results_details.md).

---

## Results Overview (`200k` CICIDS2017 Sample: `2,830,743` Raw → `191,268` Clean Flows, 77 Features, 13 Classes)

| Model | Imbalance | `random` `F1_Macro` (95% CI) | `time_block` `F1_Macro` (95% CI) | `by_day` Closed-Set `F1_Macro` |
|---|---|---:|---:|---:|
| **XGBoost (Rec.)** | `class_weight` | **0.9661** `[0.9310, 0.9829]` | **0.8646** `[0.8460, 0.8763]` | **0.0716** |
| **1D-CNN (Baseline)** | `smote` | 0.8447 `[0.7649, 0.8819]` | 0.3073 `[0.3019, 0.3359]` | 0.0696 |
| **MLP (Baseline)** | `smote` | 0.7934 `[0.7544, 0.8176]` | 0.2770 `[0.2756, 0.3011]` | 0.0716 |

> All-class `F1_Macro` is the headline metric throughout; per-protocol `F1_Macro_Reliable` (`0.9931` / `0.9562` / `0.9286` on `random`), which excludes rare classes with `< 30` test flows, appears only alongside `F1_Macro` in [`docs/results_details.md`](docs/results_details.md) and [`reports/model_comparison_report.md`](reports/model_comparison_report.md).

### Key Findings

1. **Production Classifier:** `XGBoost` (`class_weight` deployment profile) leads `random` (`0.9661`) at `155,163` flows/s batch; `40` Optuna trials and `stacking` (`0.9660`, paired-bootstrap CI crossing `0`) brought no statistically significant gain, so plain `XGBoost` (no Optuna tuning or stacking) remains the production model.
2. **Per-Model Imbalance Selection:** On `random`-split validation, `{none, class_weight, smote}` selection kept `XGBoost` on `none` (`0.9518` vs `0.9446`) and switched `MLP` and `1D-CNN` to `smote`, lifting `MLP` test `F1_Macro` from `0.6728` to `0.7934`. The shipped deployment artifact is nevertheless trained with `class_weight`, because chronological `time_block` validation selects it (`0.8754` vs `0.8534`).
3. **Drift & Unseen Attacks:** With real `Timestamp` ordering, `XGBoost` `time_block` `F1_Macro` is `0.8646`; under open-set `by_day` binary evaluation `XGBoost (1 - P_Benign)` detects `28.29%` of unseen attacks (`F1 = 0.4407`).
4. **Deployment Profiles:** Both profiles are persisted and Git LFS-tracked — active `class_weight` (`models/xgb_model.json`) and alternate ultra-low-FPR `none` (`models/xgb_model_none.json`); see the dedicated section below.
5. **Anomaly Detectors (Supplementary):** Benign-only `Autoencoder` (`F1 = 0.6926`, `ROC-AUC = 0.8676`) leads all 6 individual baselines, while `Score-Level Fusion (degenerate: equals Autoencoder)` adds zero fusion gain (`weights: Autoencoder=1.00`, unmeasured cols `N/A`).

---

## XGBoost Deployment Profiles (`deployment.xgboost_imbalance_profile`)

Both profiles were trained on the same `random` split with the same seed and are selected **only on validation data** (`used_test_data = False`).

| Profile | Artifact | Selection Basis | `time_block` Test `F1_Macro` | Benign FPR | Attack Recall | Focus Mean Recall |
|---|---|---:|---:|---:|---:|
| **`class_weight` (default active)** | `models/xgb_model.json` | `time_block` val `0.8754` vs `0.8534` | **0.8646** `[0.8460, 0.8763]` | `0.0144` | `0.9796` | `0.7968` |
| `none` (alternate ultra-low-FPR) | `models/xgb_model_none.json` | random-split val `0.9518` vs `0.9446` | 0.7571 | `0.0002` | `0.9607` | `0.4547` |

On the held-out `time_block` test partition (`37,535` flows: `30,763` Benign, `6,772` attack), the default `class_weight` profile raises overall attack recall from `0.9607` to `0.9796` (≈`128` extra attacks caught) and six-class focus recall from `0.4547` to `0.7968`, at the cost of `0.0144` Benign FPR (≈`443` false alarms) versus `0.0002` (≈`6` false alarms) for `none`. Set `deployment.xgboost_imbalance_profile` to `"class_weight"` or `"none"` and restart; `ModelRegistry` resolves the matching artifact.

- **Honest caveat:** `Bot` recall remains `0.000` (`0 / 28` test flows) under both profiles because of severe temporal feature shift (`KS = 1.0000` on `Flow Duration`), and `Heartbleed` has only `2` test flows (`statistically_unreliable = true`).

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

- **Random Split Optimism & Real-Timestamp `time_block`:** Random flow splitting mixes burst neighbors; the committed `time_block` run now uses real `Timestamp` ordering from the optional `data/TrafficLabelling/` corpus (`match_rate = 0.9999`), which is local-only and intentionally untracked. See [`docs/optional_data.md`](docs/optional_data.md) and [`docs/evaluation_caveats.md`](docs/evaluation_caveats.md).
- **Rare-Class Instability & Upstream Labels:** `Heartbleed`, `Infiltration`, `Bot`, and `Web Attack` have `< 30` test flows (`< 30` reliability threshold), and Engelen et al. (2021) label corrections were absent/not applied ([`docs/optional_data.md`](docs/optional_data.md)).
- **Shortcut-Feature & Zero-Day Limits:** Global SHAP relies strongly on `Destination Port` (`delta = -0.0140` when dropped), `by_day` misses `71.71%` of unseen attacks, and unsupervised detectors (`AE`/`VAE`/`OCSVM`/`IF`/`LOF`) are supplementary signals only.
- **Offline Batch & Dry-Run Response Scope:** Live `libpcap`/`eBPF` capture is out of scope, and `ActiveResponseEngine` outputs `NOT EXECUTED (dry-run)` firewall command previews rather than executing host rules ([`docs/evaluation_caveats.md`](docs/evaluation_caveats.md)).

---

## Security & License

- **Artifact Deserialization & Integrity:** Deserializing `*.pkl` files via `joblib` executes arbitrary code. `models/MANIFEST.json` protects against corrupted or accidentally swapped artifacts, but is NOT a defense against a malicious actor who can modify both the model and the manifest; only load artifacts from trusted sources.
- **License:** Distributed under the [MIT License](LICENSE). Full metric tables & caveats: [`docs/results_details.md`](docs/results_details.md) · [`docs/evaluation_caveats.md`](docs/evaluation_caveats.md) · [`docs/architecture.md`](docs/architecture.md) · [`docs/project_details.md`](docs/project_details.md).
