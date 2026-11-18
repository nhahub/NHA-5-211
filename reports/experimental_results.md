# SMART IDS v3 — Experimental Results Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Data Engineering & Leakage Safeguard Audit

- **Original Dataset Dimensions:** `[200000, 79]`

- **Rows After Exact Deduplication:** `191,268` (removed `8,732` duplicates)

- **Infinity Values Converted to NaN:** `185`

- **Coerced Non-Numeric NaNs:** `0`

- **NaNs Imputed (Train-Only Median Fit):** train=`130`, val=`36`, test=`48`

- **Outlier Values Clipped (Train-Only 5-Sigma Bounds):** train=`37,467`, val=`6,337`, test=`7,437`

- **Removed Structural / Redundant Columns:** `['Fwd Header Length.1']`

- **Final Feature Count:** `77`

- **Leakage Verification Passed:** `False`


- **Partition Sizes:** Train (pre-SMOTE) = `138,190`, Train (post-SMOTE) = `413,056`, Validation = `24,387`, Test = `28,691`.


## 2. Class Imbalance Experiment (SMOTE vs. Class Weighting)

- **Selected Strategy:** `smote`

- **Justification:** SMOTE achieved validation Macro-F1=0.9432 (Weighted-F1=0.9986, Macro-Recall=0.9518) vs Class-Weight Macro-F1=0.9446 and Unweighted Macro-F1=0.9518. SMOTE is selected for deep/tree parity and balanced minority gradients.


## 3. Probability Calibration (Validation Temperature Scaling)

| Model | temperature | ece_before | ece_after | brier_before | brier_after |
| --- | --- | --- | --- | --- | --- |
| MLP | 1.0000 | 0.0208 | 0.0208 | 0.0520 | 0.0520 |
| CNN | 1.0000 | 0.0094 | 0.0094 | 0.0218 | 0.0218 |
| XGBoost | 1.0000 | 0.0008 | 0.0008 | 0.0022 | 0.0022 |


## 4. Autoencoder Zero-Day Novelty Detection & Percentile Sweep

- **95th Percentile Benign Validation Threshold:** `0.063031`
- **Binary Test Metrics:** Accuracy=`0.8688`, Precision=`0.6863`, Recall=`0.5132`, F1=`0.5873`, ROC-AUC=`0.8393`, FPR=`0.0521`, FNR=`0.4868`


| percentile | threshold | precision | recall | f1 | fpr | fnr |
| --- | --- | --- | --- | --- | --- | --- |
| 90.0000 | 0.0280 | 0.5494 | 0.5629 | 0.5560 | 0.1026 | 0.4371 |
| 95.0000 | 0.0630 | 0.6863 | 0.5132 | 0.5873 | 0.0521 | 0.4868 |
| 97.0000 | 0.0971 | 0.7605 | 0.4879 | 0.5944 | 0.0342 | 0.5121 |
| 99.0000 | 0.1698 | 0.8988 | 0.4425 | 0.5930 | 0.0111 | 0.5575 |


## 5. Unseen-Attack Holdout Protocol

- **Held-Out Attack Class:** `Web Attack` (`26` test flows)
- **Closed-Set Supervised Unseen-Class Recall:** `0.0000`
- **Benign-Only Autoencoder Unseen-Attack Recall:** `0.0000` (Benign vs Unseen ROC-AUC = `0.5440`)


## 6. SHAP Explainability Summary

- **XGBoost Top-5 (`TreeExplainer`, 0.28s):** `Destination Port` (1.9293), `Init_Win_bytes_backward` (0.5529), `min_seg_size_forward` (0.5354), `Init_Win_bytes_forward` (0.4501), `Flow IAT Min` (0.3253)

- **MLP Top-5 (`GradientExplainer`, 2.34s):** `PSH Flag Count` (0.0103), `ACK Flag Count` (0.0100), `Min Packet Length` (0.0071), `Down/Up Ratio` (0.0054), `Packet Length Variance` (0.0046)

- **MLP Explainer Trade-Off:** `GradientExplainer` speedup vs `KernelExplainer` = `2.83x`, Spearman rank correlation = `0.786`.
