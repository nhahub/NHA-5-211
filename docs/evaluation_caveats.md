# Evaluation Caveats, Limitations & Security Notes

This note consolidates methodological caveats, dataset constraints, and security boundaries referenced from [`README.md`](../README.md).

---

## 1. Data Splitting & Benchmark Optimism

- **Random Flow-Level Split Optimism (`random` vs. `time_block` & `by_day`):** Random flow splitting (`data.split_strategy: random`) mixes adjacent flows from identical capture bursts across train/validation/test splits, yielding optimistic supervised scores (`XGBoost` `class_weight` `F1_Macro = 0.9661` on `random` vs `0.8646` on `time_block` and `0.0716` on closed-set `by_day`). [`configs/config.yaml`](../configs/config.yaml) supports both closed-set chronological drift (`split_strategy: time_block`, earliest ~60% / middle ~20% / latest ~20% per class) and open-set cross-day (`split_strategy: by_day`, Mon–Wed train, half of Thu validation, Thu remainder + Fri test), both reported alongside `random` in [`reports/model_comparison_report.md`](../reports/model_comparison_report.md).
- **Real-Timestamp `time_block` Alignment (`data.timestamp_source_dir`):** Standard `MachineLearningCVE` CSVs in `data/raw/` omit `Timestamp`. The committed `time_block` run aligned real timestamps from the optional `data/TrafficLabelling/` corpus (`ordering = Timestamp`, `2,830,471 / 2,830,743` flows matched, `match_rate = 0.9999`); that corpus is **local-only and not tracked in git**. When it is absent, `time_block` falls back to session rank + CSV row index (`time_block ordering = CSV row order (no real Timestamp)`) with an explicit warning. Full download/placement instructions and the single-protocol `python scripts/evaluate.py --time-block-only` command live in [`docs/optional_data.md`](optional_data.md).
- **Stratified Downsampling (`data.sample_size` / `data.use_full_dataset`):** The 8 CICIDS2017 CSVs contain `2,830,743` raw flows; default configuration uses `data.sample_size: 200000` (`data.use_full_dataset: false`) via [`stratified_downsample`](../src/data/loader.py), sampling majority/mid-size classes proportionally while preserving `min(class_count, 50)` real rare-class rows prior to deduplication. Setting `data.use_full_dataset: true` disables downsampling cleanly.

---

## 2. Class Imbalance, Label Quality & Metric Reliability

- **Statistically Unreliable Rare Classes (`Heartbleed`, `Infiltration`, `Bot`, `Web Attack`):** In held-out test splits (`random`: `Heartbleed` `2` rows, `Infiltration` `5`, `Bot` `21`, `Web Attack` `26`; `time_block`: `Heartbleed` `2`, `Infiltration` `7`, `Bot` `28`, `Web Attack` `36`), all fall below `evaluation.reliable_support_threshold = 30`. Single misclassifications swing unweighted `F1_Macro` substantially and produce wide binomial recall intervals.
  - **`F1_Macro` vs. `F1_Macro_Reliable`:** All-class `F1_Macro` (with 95% bootstrap CI) is always the headline metric; `F1_Macro_Reliable` excludes classes with support `< 30` on that partition and is shown strictly next to `F1_Macro` with explicit footnotes listing excluded classes so low-support failures are never hidden.
  - **Pooled 5-Fold CV Out-of-Fold (OOF) Evaluation:** Every report table supplements single-split per-class metrics with pooled 5-fold CV OOF evaluation (`159,908` flows; rare-class OOF recall with 95% Wilson CIs: `Bot` `n=116` recall `0.9655 [0.9147, 0.9865]`, `Infiltration` `n=31` recall `0.9032 [0.7510, 0.9665]`, `Web Attack` `n=149` recall `0.9799 [0.9425, 0.9931]`, `Heartbleed` `n=9` recall `1.0000 [0.7009, 1.0000]`).
- **Known CICIDS2017 Labeling Issues (Engelen et al., 2021):** Upstream CICIDS2017 CSVs exhibit flow-construction bugs, duplicated TCP teardown packets, infinite/NaN fields, and contested labels (especially DoS/Heartbleed/Bot; Engelen et al., *IEEE S&P Workshops*, 2021). Corrected labels from Distrinet KU Leuven (`https://intrusion-detection.distrinet-research.be/CNS2021/`) are supported via `data.label_correction_file` (`data/engelen2021_label_corrections.csv`) and [`apply_label_corrections`](../src/data/loader.py). See [`docs/optional_data.md`](optional_data.md). *(Status: external Engelen file was absent locally and skipped cleanly; relabeling is unit-tested on fixtures and NOT VERIFIED on full CICIDS2017.)*
- **Shortcut-Feature Risk (`Destination Port`):** `Destination Port` dominates `XGBoost` global SHAP (`1.2368` vs `0.4690`). Evaluating with `Destination Port` removed across identical held-out rows shifted `XGBoost` Test `F1_Macro` by `-0.0140` (`0.9465` -> `0.9325`), showing performance is not solely port lookup while highlighting port sensitivity.

---

## 3. Operational Limits & Zero-Day Novelty Detection

- **Temporal Concept Drift (`time_block`):** Active `class_weight` `XGBoost` still collapses on `Bot` (`28/28` misclassified as `BENIGN`, recall `0.0000`, KS `1.0000` shift on `Flow Duration` & `Total Length of Fwd Packets`) and degrades on `Web Attack` (recall `0.6667`, `19.4%` -> `BENIGN`), `DoS slowloris` (`0.8125`, `10.0%` -> `DoS Slowhttptest`, `2.5%` -> `BENIGN`). The alternate `none` profile also loses `Heartbleed` (`0.000`), `Infiltration` (`0.571`), `DoS GoldenEye` (`0.829`), and `DoS slowloris` (`0.800`); `Heartbleed` has only `2` test flows (`statistically_unreliable`).
- **Open-Set Unseen-Attack Miss Rate (`by_day`):** Across unseen Thursday–Friday attack families, `XGBoost (1 - P_Benign)` detects only `28.29%` of unseen attack flows (`recall = 0.2829`, precision `0.9970`, `FPR = 0.0004`), missing `71.71%`.
- **Supplementary Role of Unsupervised Detectors:** Benign-only detectors (`Autoencoder` `F1 = 0.6926` random / `0.4819` `time_block` / `0.4663` `by_day`; `VAE = 0.5910`; `One-Class SVM = 0.5898`; `Isolation Forest = 0.4394`; `LOF = 0.4647`) struggle on stealthy application-layer flows mimicking benign HTTP statistics and act purely as **supplementary zero-day alerts**, never replacements for supervised detection.
- **Offline Telemetry & Dry-Run Active Response:** Live packet capture (`libpcap`/`eBPF`) and streaming CICFlowMeter extraction are outside project scope. `ActiveResponseEngine` runs strictly in `NOT EXECUTED (dry-run)` mode, validating public source IPs with `ipaddress` (blocking private/loopback/link-local/multicast/reserved/unspecified IPs), writing append-only logs, and generating OS firewall command previews (`iptables`/`nftables`, Windows `netsh`/`New-NetFirewallRule`) without executing host commands.

---

## 4. Artifact Integrity & Deserialization Security

- Serialized sklearn/joblib artifacts (`*.pkl`) execute arbitrary code during deserialization and **must only be loaded from trusted sources**.
- [`ModelRegistry`](../src/utils/model_registry.py) verifies SHA-256 digests against [`models/MANIFEST.json`](../models/MANIFEST.json) when `security.verify_model_hashes: true`.
- **Important Trust Boundary:** `models/MANIFEST.json` protects against corrupted or accidentally swapped artifacts, but is **NOT** a defense against a malicious actor who can modify both the model and the manifest; only load artifacts from trusted sources.
