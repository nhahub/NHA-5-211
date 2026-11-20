# SMART IDS v3 — Detailed Experimental Results

This document preserves the complete experimental breakdown referenced from [`README.md`](../README.md). Every metric below is sourced directly from generated evaluation outputs in [`reports/evaluation_summary.json`](../reports/evaluation_summary.json), [`reports/model_comparison.csv`](../reports/model_comparison.csv), and [`reports/model_comparison_report.md`](../reports/model_comparison_report.md).

---

## 1. Dataset Provenance & Primary Split (`random`)

- **Source Corpus:** All 8 daily CICIDS2017 `MachineLearningCVE` CSV exports in `data/raw/` spanning `2,830,743` raw flows (`77` non-redundant numeric features, `13` canonical traffic classes after Web Attack mojibake normalization and class consolidation).
- **Sampled & Cleaned Benchmark Pool:** Stratified downsample to `200,000` flows (`191,268` rows remaining after stateless deduplication of exact feature+label duplicates, `185` `inf` conversions, and `214` NaNs held for train-only median imputation).
- **Split-Isolated Partitions (`data.split_strategy: random`):**
  - Train (pre-SMOTE): `138,190` flows (`145,714` post-SMOTE with `k_neighbors=5`; SMOTE skipped on classes `< 30` rows (`8: 8`, `9: 26`)).
  - Validation: `24,112` flows (`274` post-clipping/scaling cross-split collisions removed).
  - Held-Out Test: `28,324` flows (`366` post-clipping/scaling cross-split collisions removed; `642` total post-clipping collisions removed across splits).
  - `leakage_checks.passed = True` (`0` train/val/test exact feature collisions; scaler fit strictly on `138,190` train rows).

---

## 2. Unified Model Benchmark (`random` Held-Out Test)

> All-class **`F1_Macro`** (with 95% bootstrap CI) is the headline multi-class metric. **`F1_Macro_Reliable`** excludes classes with held-out test support `< 30` (`Bot`, `Heartbleed`, `Infiltration`, `Web Attack`) and is shown alongside `F1_Macro` for reference only.

| Model | Imbalance | Framing | Accuracy | F1_Weighted | F1_Macro | 95% CI (F1_Macro / F1) | F1_Macro_Reliable | ROC-AUC | Median / P95 (µs) | Batch (flows/s) | Size (MB) | Train (s) |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **XGBoost (Rec.)** | `class_weight` | 13-class supervised | **0.9986** | **0.9986** | **0.9661** | `[0.9310, 0.9829]` | **0.9931** | **0.9995** | 754.90 / 1493.02 | **155,163.3** | 3.631 | 10.66 |
| **1D-CNN** | `smote` | 13-class supervised | 0.9904 | 0.9913 | 0.8447 | `[0.7649, 0.8819]` | 0.9562 | 0.9933 | 661.30 / 806.45 | 17,574.6 | 1.310 | 859.55 |
| **MLP** | `smote` | 13-class supervised | 0.9766 | 0.9772 | 0.7934 | `[0.7544, 0.8176]` | 0.9286 | 0.9962 | 155.00 / 270.41 | 24,487.8 | 2.534 | 57.84 |
| **Autoencoder** | `benign_only` | Binary anomaly | **0.8934** | **0.6926** | **0.6926** | `[0.6820, 0.7020]` | N/A | **0.8676** | 207.70 / 357.89 | 23,860.5 | 1.548 | 68.80 |
| **Score-Level Fusion (degenerate: equals Autoencoder)** | `benign_only` | Binary anomaly | 0.8934 | 0.6926 | 0.6926 | `[0.6820, 0.7020]` | N/A | 0.8676 | N/A | N/A | N/A | N/A |
| **VAE** | `benign_only` | Binary anomaly | 0.8693 | 0.5910 | 0.5910 | `[0.5780, 0.6025]` | N/A | 0.7755 | 250.75 / 373.46 | 24,299.1 | 1.571 | 33.46 |
| **One-Class SVM** | `benign_only` | Binary anomaly | 0.8690 | 0.5898 | 0.5898 | `[0.5770, 0.6020]` | N/A | 0.7467 | **140.85 / 166.06** | 16,548.0 | 0.478 | 1.49 |
| **Autoencoder (Deep Reg)** | `benign_only` | Binary anomaly | 0.8668 | 0.5831 | 0.5831 | `[0.5704, 0.5947]` | N/A | 0.7908 | 254.80 / 374.63 | 20,310.7 | 5.070 | 53.46 |
| **LOF** | `benign_only` | Binary anomaly | 0.8421 | 0.4647 | 0.4647 | `[0.4516, 0.4775]` | N/A | 0.8607 | 1,313.55 / 2,214.92 | 42,331.4 | 5.666 | 0.49 |
| **Isolation Forest** | `benign_only` | Binary anomaly | 0.8385 | 0.4394 | 0.4394 | `[0.4253, 0.4531]` | N/A | 0.7710 | 7,174.40 / 8,003.97 | 75,805.4 | 2.096 | 0.48 |

> **Degenerate Anomaly Fusion Note:** Validation selected a single-detector fusion (`Autoencoder=1.00`, `One-Class SVM=0.00`, `Isolation Forest=0.00`); no gain from fusing detectors on this data. Unmeasured fusion latency/size/duration fields are marked `N/A` rather than `0.0`. Across all 6 benign-only detectors on leave-attack-out validation (`AE` `0.1539`, `VAE` `0.1221`, `One-Class SVM` `0.1209`, `Autoencoder (Deep Reg)` `0.1202`, `LOF` `0.0969`, `Isolation Forest` `0.0812`), standard `Autoencoder` remained strongest.

---

## 3. Per-Model Imbalance Selection, Neural Trials, CV, Optuna & Ensemble Significance

- **Per-Model Imbalance Selection (`tolerance = 0.0020`, `neural_epochs_cap = 10`):**
  - `MLP` chose **`smote`** (val `F1_Macro = 0.7695` vs `none = 0.6924`, `class_weight = 0.4046`), lifting `MLP` held-out test `F1_Macro` from Round 2's `0.6728` (Bot F1 `0.0000`) to **`0.7934`**.
  - `1D-CNN` chose **`smote`** (val `F1_Macro = 0.7501` vs `none = 0.5094`, `class_weight = 0.4150`), reaching held-out test **`0.8447`**.
  - `XGBoost` stayed on **`none`** for the `random` split (val `F1_Macro = 0.9518` tied within tolerance `0.0020` against `smote = 0.9518`; `class_weight = 0.9446`). The shipped deployment artifact is nevertheless the `class_weight` profile because chronological `time_block` validation selects it (`0.8754` vs `0.8534`); both artifacts are persisted and selectable.
- **Bounded Neural Loss Regularization Trials (`max 3.0 min/model`):** Both `MLP` (`label_smoothing = 0.7138`, `focal_loss = 0.7618` vs baseline `cross_entropy = 0.8134`) and `1D-CNN` (`label_smoothing = 0.6562` vs baseline `0.8701`; `focal_loss` skipped at 3.0 min time budget) underperformed baseline `cross_entropy` and were automatically reverted. Both `MLP` (`512->256->128->64`) and `1D-CNN` are explicitly positioned as **neural baselines for comparison**, while **`XGBoost`** is recommended for production.
- **Unified 5-Fold Stratified CV & Optuna Gating (`XGBoost`):**
  - 5-fold stratified CV on the `166,926`-row development pool (`train+val`, held-out test excluded) yielded **`0.9535 ± 0.0121`** `F1_Macro` and **`159,908`-row pooled OOF `F1_Macro` `0.9613`** (`F1_Macro_Reliable = 0.9625`, `oof_total_rows = 159,908` post-collision dedup).
  - `40` Optuna TPE trials reached validation `F1_Macro` `0.9561` vs `baseline` `0.9504` (`+0.0056 <= 0.0121` CV std), so CV-std gating automatically retained `baseline`.
- **Temperature Scaling & Ensemble Paired Bootstrap:**
  - Validation temperature scaling gave `T = 0.7948` (`MLP`, ECE `-0.002535`), `T = 0.8749` (`1D-CNN`, ECE `-0.001259`), and `T = 0.9702` (`XGBoost`, ECE delta `+0.000072`, already well-calibrated near `T = 1.0`).
  - Ensemble comparison on test: `best_single (XGBoost)` `0.9661`, `soft_voting` `0.9193` (significant loss `-0.0468`), `confidence_weighted` `0.9536` (loss `-0.0125`, CI `[-0.0537, +0.0063]` includes `0`), `val_weighted_voting` `0.9225` (significant loss `-0.0436`), and validation `stacking` `0.9660` (`-0.0001`, paired bootstrap `95%` CI `[-0.0140, +0.0146]` crosses `0`, `statistically_significant = False`). Consequently `XGBoost` remains the default supervised production classifier.

---

## 4. Secondary Evaluation Protocols (`time_block`, `by_day`, Ablation, Unseen, Sensitivity)

### Chronological Drift (`time_block`)
- Ordering used real `Timestamp` values aligned from `data/TrafficLabelling/` (`time_block ordering = Timestamp`, `match_rate = 0.9999`, `leakage_checks.passed = True`). Partitions: train pre-SMOTE `115,669`, validation `37,420`, held-out test `37,535`.
- Evaluating canonical `random` split configuration under per-class chronological `time_block` shifted test **`F1_Macro`** sharply:
  - `XGBoost` (`class_weight` deployment profile): **`0.8646`** (`95%` CI `[0.8460, 0.8763]`, Accuracy `0.9834`, `F1_Weighted = 0.9876`, `F1_Macro_Reliable = 0.9240`).
  - `1D-CNN`: **`0.3073`** (`95%` CI `[0.3019, 0.3359]`, Accuracy `0.9250`, `F1_Macro_Reliable = 0.3995`).
  - `MLP`: **`0.2770`** (`95%` CI `[0.2756, 0.3011]`, Accuracy `0.9511`, `F1_Macro_Reliable = 0.3601`).
  - `Autoencoder`: binary `F1 = 0.4819` (`95%` CI `[0.4705, 0.4926]`, `ROC-AUC = 0.8151`, `FPR = 0.0623`).

### Open-Set Cross-Day Protocol (`by_day`: Mon–Wed Train / Thu Val / Thu–Fri Unseen Attacks)
- In CICIDS2017, all five Thursday–Friday attack families (`Bot`, `DDoS`, `Infiltration`, `PortScan`, `Web Attack`) never appear in Monday–Wednesday training (`leakage_checks.passed = True`). Partitions: train pre-SMOTE `112,124`, validation `15,537`, held-out test `63,001`.
  - Closed-set 13-class reference: `XGBoost` `F1_Macro = 0.0716` (`Acc=0.6856`, `F1_Macro_Reliable=0.1718`), `MLP` `0.0716`, `1D-CNN` `0.0696`.
  - Open-set binary novelty (`Benign vs. Attack`): `XGBoost (1 - P_Benign)` recall **`0.2829`** (`F1 = 0.4407`, precision `0.9970`, `FPR = 0.0004` — missing `71.71%` of unseen attacks), `XGBoost (Binary)` recall `0.2806` (`F1 = 0.4379`, precision `0.9968`), `MLP (1 - P_Benign)` recall `0.2841` (`F1 = 0.4420`, precision `0.9956`), `1D-CNN (1 - P_Benign)` recall `0.1663` (`F1 = 0.2812`), and benign-only `Autoencoder` recall **`0.3400`** (`F1 = 0.4663`, `ROC-AUC = 0.8760`, `FPR = 0.0542`).

### Recall-vs-FPR Mitigation Screening & Deployment Profiles

> **Important:** Full validation-screening tables and held-out test tables live in [`reports/model_comparison_report.md`](../reports/model_comparison_report.md). Test-partition mitigation tables are reported **for description only**; options were **not** selected on held-out test data. Validation ranking agreed with test ranking (`time_block` Spearman `rho = 0.9895`, top bounded-FPR option `XGBoost ('class_weight') (Argmax)`; `by_day` Spearman `rho = 0.8415`, top option `Lower P(Benign) Gate (1 - P_Benign > tau) (P99)`).

| Profile | Persistence Status | Selection Basis (`used_test_data = False`) | Random-Split Val `F1_Macro` | `time_block` Test `F1_Macro` | `time_block` Benign FPR | `time_block` Focus Mean Recall | Operational Trade-off |
|---|---|---|---:|---:|---:|---:|---|
| **Default active `'class_weight'`** | [`models/xgb_model.json`](../models/xgb_model.json) | `time_block` val `0.8754` vs `0.8534` | `0.9446` | `0.8646` | `0.0144` | `0.7968` | Higher recall on drifted classes (`DoS GoldenEye = 1.000`, `DoS slowloris = 0.975`, `Web Attack = 0.806`, `Heartbleed = 1.000`, `Infiltration = 1.000`) at ≈`443` false alarms on `30,763` benign test flows. |
| **Alternate `'none'` (ultra-low FPR)** | [`models/xgb_model_none.json`](../models/xgb_model_none.json) | random-split val `0.9518` vs `0.9446` | `0.9518` | `0.7571` | `0.0002` | `0.4547` | ≈`6` false alarms, but weaker drifted/rare recall (`Bot = 0.000`, `Heartbleed = 0.000`, `Web Attack = 0.528`, `Infiltration = 0.571`, `DoS GoldenEye = 0.829`, `DoS slowloris = 0.800`). |

- On the held-out `time_block` partition (`37,535` flows: `30,763` Benign, `6,772` attack), `class_weight` raises overall attack recall from `0.9607` to `0.9796` (≈`128` extra attacks caught) at `0.0144` versus `0.0002` Benign FPR.
- Choice between profiles depends on SOC operational tolerance for false alarms vs. missed drifted attacks. Under every mitigation option evaluated (`none`, `class_weight`, `Hybrid Override`, lowered `P(Benign)` gates), **`Bot` recall stays `0.000` on `time_block` (`0/28`, `KS = 1.0000` on `Flow Duration`) and `~0.015` at best on `by_day`**; `Heartbleed` has only `2` test flows and is `statistically_unreliable`.

### Feature Ablation (`Destination Port` Dropped), Leave-One-Attack-Out & Sample-Size Sensitivity
- **Feature Ablation (`Destination Port`):** Global SHAP on `XGBoost` is led by `Destination Port` (`1.2368` vs runner-up `Init_Win_bytes_backward` `0.4690`). Dropping `Destination Port` on identical `28,126` held-out test rows (`sha256: 6c72618b...` verified) shifted `XGBoost` Test `F1_Macro` from `0.9465` to `0.9325` (`delta = -0.0140`).
- **Leave-One-Attack-Out (`Web Attack` withheld from training, `26` test attack vs `23,112` benign):** Supervised closed-set recall on withheld `Web Attack` was `0.0`, while benign-only `Autoencoder` (`threshold = 0.0751` at P95) achieved `0.0` unseen recall (`ROC-AUC = 0.4750`, `FPR = 0.0509`), illustrating that stealthy web-attack flows mimic benign HTTP statistics.
- **Sample-Size Sensitivity (`200k` vs. `500k`):** Scaling `XGBoost` (`class_weight`) from `200,000` requested rows (`196,384` deduplicated, `28,126` test) to `500,000` (`488,531` deduplicated, `68,466` test) shifted `F1_Macro` from `0.9465` (`[0.9084, 0.9673]`) to `0.9736` (`[0.9651, 0.9800]`, `+0.0271` delta, overlapping 95% CIs, `conclusions_change = False`).
- **ONNX Export & Inference Latency Path:** `onnx` and `tf2onnx` are omitted from `requirements.txt` to avoid resolver conflicts on Python `3.13` + TensorFlow `2.21.0` + NumPy `2.x`; single-flow latency uses warmed `tf.function` graphs (Keras) and `booster.inplace_predict(...)` (XGBoost). See [`docs/project_details.md`](project_details.md#4-onnx-export-feasibility-notes).

---

## Using a Larger Sample

The committed production weights in [`models/`](../models/) were trained at the default stratified budget (`data.sample_size: 200000`, `data.use_full_dataset: false`) and have **not** been retrained here. In the evaluation sensitivity protocol (fit in-memory on `XGBoost` with the active `class_weight` profile), increasing the stratified sample from `200,000` (`196,384` clean flows) to `500,000` (`488,531` clean flows) yielded XGBoost Macro-F1 `0.9736 [0.9651, 0.9800]` vs `0.9465 [0.9084, 0.9673]` at `200k` (`F1_Macro_Reliable` `0.9688` vs `0.9854`), with overlapping 95% bootstrap CIs (`conclusions_change = False`).

### How to Retrain at `500k` or Full Dataset

1. **500k Stratified Sample (CLI override without editing YAML):**
   Keep `data.use_full_dataset: false` in [`configs/config.yaml`](../configs/config.yaml) and pass `--max-rows 500000`, or set `data.sample_size: 500000` in [`configs/config.yaml`](../configs/config.yaml):
   ```powershell
   .venv\Scripts\python.exe scripts/train.py --max-rows 500000 --run-eval
   ```
2. **Full `2,830,743`-Row Corpus (No Downsampling):**
   Set `data.use_full_dataset: true` in [`configs/config.yaml`](../configs/config.yaml) (which overrides `data.sample_size`) and run:
   ```powershell
   .venv\Scripts\python.exe scripts/train.py --run-eval
   ```

> **Expected Training-Time Warning:** At `200,000` requested rows, single baseline training runs measured in [`reports/model_comparison.csv`](../reports/model_comparison.csv) already required **`859.55s` (`14.3 min`)** for `1D-CNN` on CPU TensorFlow, **`10.66s`** for the `class_weight` `XGBoost` profile, `68.80s` for `Autoencoder`, `57.84s` for `MLP`, `53.46s` for `Autoencoder (Deep Reg)`, and `33.46s` for `VAE` — in addition to per-model imbalance screening, bounded loss trials, 40 Optuna trials, 5-fold stratified CV, and secondary protocol runs. Increasing the sample `2.5x` (`500k`) or `~14x` (full `2.83M` rows) scales CPU TensorFlow, CV, and neighbor-based anomaly detectors substantially; budget **1+ hour** at `500k` and **multiple hours / high RAM** for a full `2.83M`-row run on Windows CPU TensorFlow.
