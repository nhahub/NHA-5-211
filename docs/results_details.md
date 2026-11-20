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
| **XGBoost (Rec.)** | `none` | 13-class supervised | **0.9989** | **0.9988** | **0.9603** | `[0.9013, 0.9849]` | **0.9972** | **0.9995** | 344.20 / 485.61 | **252,775.0** | 2.954 | 320.33 |
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
  - `XGBoost` stayed on **`none`** (val `F1_Macro = 0.9518` tied within tolerance `0.0020` against `smote = 0.9518`; `class_weight = 0.9446`).
- **Bounded Neural Loss Regularization Trials (`max 3.0 min/model`):** Both `MLP` (`label_smoothing = 0.7138`, `focal_loss = 0.7618` vs baseline `cross_entropy = 0.8134`) and `1D-CNN` (`label_smoothing = 0.6562` vs baseline `0.8701`; `focal_loss` skipped at 3.0 min time budget) underperformed baseline `cross_entropy` and were automatically reverted. Both `MLP` (`512->256->128->64`) and `1D-CNN` are explicitly positioned as **neural baselines for comparison**, while **`XGBoost`** is recommended for production.
- **Unified 5-Fold Stratified CV & Optuna Gating (`XGBoost`):**
  - 5-fold stratified CV on the `162,577`-row development pool (`train+val`, held-out test excluded) yielded **`0.9535 ± 0.0121`** `F1_Macro` and **`161,042`-row pooled OOF `F1_Macro` `0.9550`** (`F1_Macro_Reliable = 0.9561`, `oof_total_rows = 161,042` post-collision dedup).
  - `40` Optuna TPE trials reached validation `F1_Macro` `0.9561` vs `baseline` `0.9504` (`+0.0056 <= 0.0121` CV std), so CV-std gating automatically retained `baseline`.
- **Temperature Scaling & Ensemble Paired Bootstrap:**
  - Validation temperature scaling gave `T = 0.7948` (`MLP`, ECE `-0.002535`), `T = 0.8749` (`1D-CNN`, ECE `-0.001259`), and `T = 0.9095` (`XGBoost`, ECE `-0.000253`).
  - Ensemble comparison on test: `best_single (XGBoost)` `0.9603`, `soft_voting` `0.9181` (significant loss `-0.0421`), `confidence_weighted` `0.9356` (significant loss `-0.0246`), `val_weighted_voting` `0.9184` (significant loss `-0.0419`), and validation `stacking` `0.9725` (`+0.0123`, paired bootstrap `95%` CI `[-0.0040, +0.0538]` crosses `0`, `statistically_significant = False`). Consequently `XGBoost` remains the default supervised production classifier.

---

## 4. Secondary Evaluation Protocols (`time_block`, `by_day`, Ablation, Unseen, Sensitivity)

### Chronological Drift (`time_block`)
- **`Timestamp` Alignment:** `time_block` ordering uses real `Timestamp` aligned via `flow_telemetry_signature` (`Source_Session_Day`, `Destination Port`, `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`, `Total Length of Fwd Packets`, `Total Length of Bwd Packets`, `Flow IAT Mean`, normalized `Label`, `Occurrence_Index`) from `data/TrafficLabelling/` (`8` CSVs; matched **`2,830,471 / 2,830,743`** = **`99.9904%`**, `272` unmatched; `Timestamp` excluded from model features).
- **Partitions (`split_strategy='time_block'`):** Train pre-SMOTE = `115,669` (`121,985` post-SMOTE), Validation = `37,420`, Held-Out Test = `37,535` (`5,760` post-clipping/scaling collisions removed; `Heartbleed` `n=11` low-support allocation `7` train / `2` val / `2` test; `leakage_checks.passed = True`).
- Evaluating canonical `random` split configuration under real-timestamp per-class chronological `time_block` yielded:
  - `XGBoost`: **`0.7571`** (`95%` CI **`[0.7087, 0.8316]`**, Accuracy `0.9922`, `F1_Weighted = 0.9916`, `F1_Macro_Reliable = 0.9116`, `ROC-AUC OVR = 0.9918`).
  - `1D-CNN`: **`0.3073`** (`95%` CI **`[0.3019, 0.3359]`**, Accuracy `0.9250`, `F1_Weighted = 0.9187`, `F1_Macro_Reliable = 0.3995`, `ROC-AUC OVR = 0.9405`).
  - `MLP`: **`0.2770`** (`95%` CI **`[0.2756, 0.3011]`**, Accuracy `0.9511`, `F1_Weighted = 0.9411`, `F1_Macro_Reliable = 0.3601`, `ROC-AUC OVR = 0.9526`).
  - `Autoencoder`: binary **`F1 = 0.4819`** (`95%` CI `[0.4705, 0.4926]`, Accuracy `0.8420`, `ROC-AUC = 0.8151`, `FPR = 0.0623`, threshold `0.1127`).
- `XGBoost` `time_block` `F1_Macro` changed only slightly vs. the earlier row-order proxy (`0.7518` -> `0.7571`, within CI), while the `1D-CNN` dropped (`0.4376` -> `0.3073`), showing neural baselines are unstable across splits. `Bot` recall remains `0.000` (all `28` test flows predicted `BENIGN`, `KS = 1.0` on `Flow Duration`), so it is real drift, not an ordering artifact.
- **Temporal Failure Breakdown (`XGBoost`, test recall `< 0.85`):** `Bot` (`28/28` -> `BENIGN`, recall `0.0000`, `KS = 1.0000` on `Flow Duration`, `Total Length of Fwd Packets`, `Fwd Packet Length Max/Mean`, `Flow Bytes/s`), `Heartbleed` (`2/2` -> `BENIGN`, recall `0.0000`, `KS = 1.0000` on `Fwd IAT Std` & `Init_Win_bytes_backward`), `Web Attack` (`36` test, recall `0.4167`, `47.2%` -> `BENIGN`, `KS = 0.4873` on `Flow IAT Mean` & `Bwd Packets/s`), `Infiltration` (`7` test, recall `0.5714`, `42.9%` -> `BENIGN`, `KS = 0.8182` on `Flow IAT Std` & `Flow IAT Max`), `DoS slowloris` (`80` test, recall `0.7000`, `20.0%` -> `BENIGN`, `KS = 0.2995` on `Idle Mean`), and `DoS GoldenEye` (`146` test, recall `0.8288`, `17.1%` -> `BENIGN`, `KS = 0.3100` on `Fwd Packet Length Mean`).
- *(Historical Reference Only — Prior Row-Order Primary Retrain):* An earlier full primary retrain with `split_strategy=time_block` prior to placing `data/TrafficLabelling/` (per-model selection `MLP=smote`, `CNN=class_weight`, `XGBoost=smote`) yielded `XGBoost` `0.8427`, `MLP` `0.6703`, `1D-CNN` `0.3919` under CSV row order; that full retrain was not repeated here.

### Open-Set Cross-Day Protocol (`by_day`: Mon–Wed Train / Thu Val / Thu–Fri Unseen Attacks)
- In CICIDS2017, all five Thursday–Friday attack families (`Bot`, `DDoS`, `Infiltration`, `PortScan`, `Web Attack`) never appear in Monday–Wednesday training (`leakage_checks.passed = True`):
  - Closed-set 13-class reference: `XGBoost` `F1_Macro = 0.0860` (`Acc=0.6868`; full primary `by_day` retrain `0.0952`), `1D-CNN` `0.0691`, `MLP` `0.0661`.
  - Open-set binary novelty (`Benign vs. Attack`): `XGBoost (1 - P_Benign)` recall **`0.2876`** (`F1 = 0.4467`, precision `0.9991`, `FPR = 0.0001` — missing `71.24%` of unseen attacks), `XGBoost (Binary)` recall `0.2690` (`F1 = 0.4238`, precision `0.9981`), `MLP (1 - P_Benign)` recall `0.2839` (`F1 = 0.4417`, precision `0.9942`), `1D-CNN (1 - P_Benign)` recall `0.1227` (`F1 = 0.2177`), and benign-only `Autoencoder` recall **`0.3419`** (`F1 = 0.4676`, `ROC-AUC = 0.8784`, `FPR = 0.0549`).

### Recall-vs-FPR Mitigation Screening & Deployment Profiles

> **Important:** Full 10-option validation-screening and held-out test tables live in [`reports/model_comparison_report.md`](../reports/model_comparison_report.md). Held-out test mitigation tables are reported **for description only**; options were **not** selected on test data. Validation ranking agreed strongly with test ranking (`time_block` Spearman **`rho = 0.9807`**, top bounded-FPR option `XGBoost ('class_weight') (Argmax)` on both splits; `by_day` Spearman `rho = 0.9756`, top option `Lower P(Benign) Gate (1 - P_Benign > tau) (P99)`).

| Profile | Persistence Status | Random `F1_Macro` | `time_block` Benign FPR | `time_block` Focus Mean Recall | `time_block` `F1_Macro` | Operational Trade-off |
|---|---|---:|---:|---:|---:|---|
| **Default `'none'`** | Persisted in [`models/xgb_model.json`](../models/xgb_model.json) | `0.9603` | `0.0002` | `0.4547` | `0.7571` | Lowest false positives on benign traffic (val FPR `0.0619`, focus recall `0.7405`, val `F1_Macro = 0.8534`), but weak recall on drifted/rare attacks (`Bot = 0.000`, `Heartbleed = 0.000`, `Web Attack = 0.528` in mitigation / `0.417` in model eval, `Infiltration = 0.571`, `DoS slowloris = 0.800` / `0.700`, `DoS GoldenEye = 0.829`). |
| **`'class_weight'`** | **Requires retraining** (`data.imbalance.strategy: class_weight`, `imbalance_selection.per_model: false`) | *(retrain required)* | `0.0144` (~`1.4%`) | `0.7968` | `0.8646` | Higher recall on drifted classes (val FPR `0.0803`, focus recall `0.9417`, val `F1_Macro = 0.8754`; test `Heartbleed = 1.000`, `Web Attack = 0.806`, `Infiltration = 1.000`, `DoS slowloris = 0.975`, `DoS GoldenEye = 1.000`) at ~`1.4%` Benign FPR (`72x` baseline false-alarm rate). |

- Key held-out descriptive alternatives in `time_block` (`XGBoost OR AE` vs `1 - P(Benign) > tau`): `Hybrid Override (P95)` reached focus recall `0.7401` at Benign FPR `0.0625` (val FPR `0.0842`, focus recall `0.9072`); `Hybrid Override (P99)` reached `0.7359` at Benign FPR `0.0468`; `Lower P(Benign) Gate (P90, tau=0.2100)` reached `0.5795` focus recall (`0.9833` overall attack recall) at Benign FPR `0.0006` (`P99`: focus recall `0.3527` at Benign FPR `0.0001`).
- Choice between profiles depends on SOC operational tolerance for false alarms vs. missed drifted attacks. Under every mitigation option evaluated (`none`, `class_weight`, `Hybrid Override`, lowered `P(Benign)` gates), **`Bot` recall stays `0.000` on `time_block` and `~0.015` at best on `by_day`**.

### Feature Ablation (`Destination Port` Dropped), Leave-One-Attack-Out & Sample-Size Sensitivity
- **Feature Ablation (`Destination Port`):** Global SHAP on `XGBoost` is led by `Destination Port` (`0.8610` vs runner-up `Init_Win_bytes_backward` `0.3711`). Dropping `Destination Port` on identical `28,324` held-out test rows (`sha256: 85c3896d...` verified) shifted `XGBoost` Test `F1_Macro` from `0.9603` to `0.9540` (`delta = -0.0063`).
- **Leave-One-Attack-Out (`Web Attack` withheld from training, `26` test attack vs `23,112` benign):** Supervised closed-set recall on withheld `Web Attack` was `0.0`, while benign-only `Autoencoder` (`threshold = 0.0744` at P95) achieved `0.0` unseen recall (`ROC-AUC = 0.5626`, `FPR = 0.0498`), illustrating that stealthy web-attack flows mimic benign HTTP statistics.
- **Sample-Size Sensitivity (`200k` vs. `500k`):** Scaling `XGBoost` (`none`) from `200,000` requested rows (`191,268` deduplicated, `28,324` test) to `500,000` (`470,048` deduplicated, `69,026` test) shifted `F1_Macro` from `0.9603` (`[0.9013, 0.9849]`) to `0.9894` (`[0.9837, 0.9943]`, `+0.0291` delta, overlapping 95% CIs, `conclusions_change = False`).
- **ONNX Export & Inference Latency Path:** `onnx` and `tf2onnx` are omitted from `requirements.txt` to avoid resolver conflicts on Python `3.13` + TensorFlow `2.21.0` + NumPy `2.x`; single-flow latency uses warmed `tf.function` graphs (Keras) and `booster.inplace_predict(...)` (XGBoost). See [`docs/project_details.md`](project_details.md#4-onnx-export-feasibility-notes).

---

## Using a Larger Sample

The committed production weights in [`models/`](../models/) were trained at the default stratified budget (`data.sample_size: 200000`, `data.use_full_dataset: false`) and have **not** been retrained here. In the evaluation sensitivity protocol (fit in-memory on `XGBoost`), increasing the stratified sample from `200,000` (`191,268` clean flows) to `500,000` (`470,048` clean flows) yielded XGBoost Macro-F1 `0.9894 [0.9837, 0.9943]` vs `0.9603 [0.9013, 0.9849]` at `200k` (`F1_Macro_Reliable` `0.9875` vs `0.9972`), with overlapping 95% bootstrap CIs (`conclusions_change = False`).

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

> **Expected Training-Time Warning:** At `200,000` requested rows, single baseline training runs measured in [`reports/model_comparison.csv`](../reports/model_comparison.csv) already required **`859.55s` (`14.3 min`)** for `1D-CNN` on CPU TensorFlow, **`320.33s` (`5.3 min`)** for `XGBoost`, `68.80s` for `Autoencoder`, `57.84s` for `MLP`, `53.46s` for `Autoencoder (Deep Reg)`, and `33.46s` for `VAE` — in addition to per-model imbalance screening, bounded loss trials, 40 Optuna trials, 5-fold stratified CV, and secondary protocol runs. Increasing the sample `2.5x` (`500k`) or `~14x` (full `2.83M` rows) scales CPU TensorFlow, CV, and neighbor-based anomaly detectors substantially; budget **1+ hour** at `500k` and **multiple hours / high RAM** for a full `2.83M`-row run on Windows CPU TensorFlow.
