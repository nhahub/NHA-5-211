# Evaluation Caveats, Limitations & Security Notes

This note consolidates methodological caveats, dataset constraints, and security boundaries referenced from [`README.md`](../README.md).

---

## 1. Data Splitting & Benchmark Optimism

- **Random Flow-Level Split Optimism (`random` vs. `time_block` & `by_day`):** Random flow splitting (`data.split_strategy: random`) mixes adjacent flows from identical capture bursts across train/validation/test splits, yielding optimistic supervised scores (`XGBoost` `F1_Macro = 0.9603` on `random` vs `0.7518` on `time_block` and `0.0860` on closed-set `by_day`). [`configs/config.yaml`](../configs/config.yaml) supports both closed-set chronological drift (`split_strategy: time_block`, earliest ~60% / middle ~20% / latest ~20% per class) and open-set cross-day (`split_strategy: by_day`, Mon–Wed train, half of Thu validation, Thu remainder + Fri test), both reported alongside `random` in [`reports/model_comparison_report.md`](../reports/model_comparison_report.md).
- **Optional Real-Timestamp Alignment for `time_block` (`data.timestamp_source_dir`):** Standard `MachineLearningCVE` CSVs in `data/raw/` omit `Timestamp`, so `time_block` falls back to session rank + CSV row index (`time_block ordering = CSV row order (no real Timestamp)`). Full download/placement instructions and the single-protocol `python scripts/evaluate.py --time-block-only` command live in [`docs/optional_data.md`](optional_data.md). *(Status: `data/TrafficLabelling/` was absent locally; real-timestamp `time_block` ordering is unit-tested on timestamped fixtures only and NOT VERIFIED on real CICIDS2017 TrafficLabelling files.)*
- **Stratified Downsampling (`data.sample_size` / `data.use_full_dataset`):** The 8 CICIDS2017 CSVs contain `2,830,743` raw flows; default configuration uses `data.sample_size: 200000` (`data.use_full_dataset: false`) via [`stratified_downsample`](../src/data/loader.py), sampling majority/mid-size classes proportionally while preserving `min(class_count, 50)` real rare-class rows prior to deduplication. Setting `data.use_full_dataset: true` disables downsampling cleanly.

---

## 2. Class Imbalance, Label Quality & Metric Reliability

- **Statistically Unreliable Rare Classes (`Heartbleed`, `Infiltration`, `Bot`, `Web Attack`):** In held-out test splits (~15% of `200k`, smaller under `time_block`), `Heartbleed` (`2` random test rows / `11` total), `Infiltration` (`7` test), `Bot` (`28` test), and `Web Attack` (`26` random test) fall below `evaluation.reliable_support_threshold = 30`. Single misclassifications swing unweighted `F1_Macro` substantially and produce wide binomial recall intervals.
  - **`F1_Macro` vs. `F1_Macro_Reliable`:** All-class `F1_Macro` (with 95% bootstrap CI) is always the headline metric; `F1_Macro_Reliable` excludes classes with support `< 30` on that partition and is shown strictly next to `F1_Macro` with explicit footnotes listing excluded classes so low-support failures are never hidden.
  - **Pooled 5-Fold CV Out-of-Fold (OOF) Evaluation:** Every report table supplements single-split per-class metrics with pooled 5-fold CV OOF evaluation (`161,042` flows; rare-class OOF recall with 95% Wilson CIs: `Bot` `n=116` recall `0.6983 [0.6095, 0.7743]`, `Infiltration` `n=31` recall `0.7097 [0.5341, 0.8390]`, `Web Attack` `n=150` recall `0.9267 [0.8735, 0.9586]`, `Heartbleed` `n=9` recall `0.8889 [0.5650, 0.9801]`).
- **Known CICIDS2017 Labeling Issues (Engelen et al., 2021):** Upstream CICIDS2017 CSVs exhibit flow-construction bugs, duplicated TCP teardown packets, infinite/NaN fields, and contested labels (especially DoS/Heartbleed/Bot; Engelen et al., *IEEE S&P Workshops*, 2021). Corrected labels from Distrinet KU Leuven (`https://intrusion-detection.distrinet-research.be/CNS2021/`) are supported via `data.label_correction_file` (`data/engelen2021_label_corrections.csv`) and [`apply_label_corrections`](../src/data/loader.py). See [`docs/optional_data.md`](optional_data.md). *(Status: external Engelen file was absent locally and skipped cleanly; relabeling is unit-tested on fixtures and NOT VERIFIED on full CICIDS2017.)*
- **Shortcut-Feature Risk (`Destination Port`):** `Destination Port` dominates `XGBoost` global SHAP (`0.8610` vs `0.3711`). Evaluating with `Destination Port` removed across identical held-out rows shifted `XGBoost` Test `F1_Macro` by `-0.0063` (`0.9603` -> `0.9540`), showing performance is not solely port lookup while highlighting port sensitivity.

---

## 3. Operational Limits & Zero-Day Novelty Detection

- **Temporal Concept Drift (`time_block`):** `XGBoost` (`none`) drops completely on `Bot` (`28/28` misclassified as `BENIGN`, recall `0.0000`, KS `1.0000` shift on `Flow Duration` & `Total Length of Fwd Packets`) and `Heartbleed` (`2/2` -> `BENIGN`), and degrades on `Web Attack` (recall `0.4000`, `45.7%` -> `BENIGN`), `Infiltration` (`0.4286`, `57.1%` -> `BENIGN`), and `DoS slowloris` (`0.7160`).
- **Open-Set Unseen-Attack Miss Rate (`by_day`):** Across unseen Thursday–Friday attack families, `XGBoost (1 - P_Benign)` detects only `28.76%` of unseen attack flows (`recall = 0.2876`, precision `0.9991`, `FPR = 0.0001`), missing `71.24%`.
- **Supplementary Role of Unsupervised Detectors:** Benign-only detectors (`Autoencoder` `F1 = 0.6926` random / `0.5003` `time_block` / `0.4676` `by_day`; `VAE = 0.5910`; `One-Class SVM = 0.5898`; `Isolation Forest = 0.4394`; `LOF = 0.4647`) struggle on stealthy application-layer flows mimicking benign HTTP statistics and act purely as **supplementary zero-day alerts**, never replacements for supervised detection.
- **Offline Telemetry & Dry-Run Active Response:** Live packet capture (`libpcap`/`eBPF`) and streaming CICFlowMeter extraction are outside project scope. `ActiveResponseEngine` runs strictly in `NOT EXECUTED (dry-run)` mode, validating public source IPs with `ipaddress` (blocking private/loopback/link-local/multicast/reserved/unspecified IPs), writing append-only logs, and generating OS firewall command previews (`iptables`/`nftables`, Windows `netsh`/`New-NetFirewallRule`) without executing host commands.

---

## 4. Artifact Integrity & Deserialization Security

- Serialized sklearn/joblib artifacts (`*.pkl`) execute arbitrary code during deserialization and **must only be loaded from trusted sources**.
- [`ModelRegistry`](../src/utils/model_registry.py) verifies SHA-256 digests against [`models/MANIFEST.json`](../models/MANIFEST.json) when `security.verify_model_hashes: true`.
- **Important Trust Boundary:** `models/MANIFEST.json` protects against corrupted or accidentally swapped artifacts, but is **NOT** a defense against a malicious actor who can modify both the model and the manifest; only load artifacts from trusted sources.
