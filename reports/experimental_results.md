# SMART IDS v3 — Experimental Results Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Data Engineering & Leakage Safeguard Audit

- **Split Strategy:** `random` (Stratified Random Split)

- **Explicitly Dropped Features (`drop_features`):** `[]`

- **Original Dataset Dimensions:** `[200000, 79]`

- **Rows After Exact Deduplication:** `191,268` (removed `8,732` duplicates)

- **Infinity Values Converted to NaN:** `185`

- **Coerced Non-Numeric NaNs:** `0`

- **NaNs Imputed (Train-Only Median Fit):** train=`130`, val=`36`, test=`48`

- **Outlier Values Clipped (Train-Only 5-Sigma Bounds):** train=`37,467`, val=`6,337`, test=`7,437`

- **Removed Structural / Redundant Columns:** `['Fwd Header Length.1']`

- **Final Feature Count:** `77`

- **Leakage Verification Passed:** `True`


- **Partition Sizes:** Train (pre-SMOTE) = `138,190`, Train (post-SMOTE) = `145,714`, Validation = `24,112`, Test = `28,324`.


## 2. Class Imbalance Experiment (Unweighted vs. Class Weighting vs. SMOTE)

| Strategy | val_macro_f1 | val_weighted_f1 | val_macro_recall | val_mlogloss |
| --- | --- | --- | --- | --- |
| Unweighted ('none') | 0.9518 | 0.9989 | 0.9323 | 0.0046 |
| Class Weighting ('class_weight') | 0.9446 | 0.9981 | 0.9583 | 0.0092 |
| Hybrid SMOTE ('smote') | 0.9518 | 0.9989 | 0.9439 | 0.0044 |


- **Selected Strategy:** `none`

- **Justification:** Tie within tolerance (tol=0.0020): Unweighted ('none') Macro-F1=0.9518, Hybrid SMOTE ('smote') Macro-F1=0.9518 fall within 0.0020 of peak validation Macro-F1=0.9518 (gap=0.0000 vs 'smote'); tie within tolerance, simplest strategy selected -> 'none' (Weighted-F1=0.9989, Macro-Recall=0.9323, mlogloss=0.0046; Hybrid SMOTE ('smote') Macro-F1=0.9518 (Weighted-F1=0.9989, mlogloss=0.0044); Class-Weight ('class_weight') Macro-F1=0.9446 (Weighted-F1=0.9981, mlogloss=0.0092)).


## 3. Probability Calibration (Validation Temperature Scaling)

| Model | temperature | ece_before | ece_after | ece_delta | brier_before | brier_after | nll_before | nll_after | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 1.0142 | 0.0081 | 0.0083 | 0.0002 | 0.0351 | 0.0351 | 0.0626 | 0.0626 | Already well-calibrated near T=1.0 |
| CNN | 0.9210 | 0.0024 | 0.0016 | -0.0008 | 0.0078 | 0.0077 | 0.0175 | 0.0173 | Recalibrated (T=0.9210, ECE delta=-0.000770) |
| XGBoost | 0.9095 | 0.0005 | 0.0003 | -0.0003 | 0.0017 | 0.0017 | 0.0042 | 0.0041 | Already well-calibrated near T=1.0 |


- **Calibration Evidence:** Models were already well-calibrated near `T = 1.0` with minimal NLL/ECE change.


## 4. Unsupervised Anomaly Detection & Classical Baselines

| Detector | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | FPR | Holdout_Class_Recall | Holdout_Class_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoencoder | 0.8678 | 0.6890 | 0.5130 | 0.5881 | 0.8384 | 0.6651 | 0.0522 | 0.0000 | 0.5852 |
| Isolation Forest | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.7710 | 0.5005 | 0.0499 | 0.0000 | 0.6101 |
| One-Class SVM | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.7467 | 0.5649 | 0.0504 | 0.0000 | 0.6385 |
| LOF | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.8607 | 0.5391 | 0.0520 | 0.1154 | 0.4188 |


- **Assessment:** Best binary F1 at 95th-pct threshold: One-Class SVM (F1=0.5898, Recall=0.5117, FPR=0.0504); best overall ROC-AUC ranking: LOF (ROC-AUC=0.8607). Autoencoder achieved F1=0.5881 / ROC-AUC=0.8384, confirming it works best as a supplementary anomaly signal alongside supervised XGBoost rather than as a standalone detector.


- **Autoencoder 95th Percentile Benign Validation Threshold:** `0.063726`
- **Autoencoder Binary Test Metrics:** Accuracy=`0.8678`, Precision=`0.6890`, Recall=`0.5130`, F1=`0.5881`, ROC-AUC=`0.8384`, FPR=`0.0522`, FNR=`0.4870`


| Detector | percentile | threshold | precision | recall | f1 | fpr | fnr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Autoencoder | 90.0000 | 0.0283 | 0.5523 | 0.5624 | 0.5573 | 0.1028 | 0.4376 |
| Autoencoder | 95.0000 | 0.0637 | 0.6890 | 0.5130 | 0.5881 | 0.0522 | 0.4870 |
| Autoencoder | 97.0000 | 0.0982 | 0.7622 | 0.4877 | 0.5948 | 0.0343 | 0.5123 |
| Autoencoder | 99.0000 | 0.1710 | 0.8995 | 0.4428 | 0.5935 | 0.0112 | 0.5572 |
| Isolation Forest | 90.0000 | -0.0495 | 0.5316 | 0.5184 | 0.5249 | 0.1030 | 0.4816 |
| Isolation Forest | 95.0000 | 0.0307 | 0.6085 | 0.3438 | 0.4394 | 0.0499 | 0.6562 |
| Isolation Forest | 97.0000 | 0.0638 | 0.7269 | 0.3360 | 0.4595 | 0.0285 | 0.6640 |
| Isolation Forest | 99.0000 | 0.1303 | 0.7730 | 0.1320 | 0.2255 | 0.0087 | 0.8680 |
| One-Class SVM | 90.0000 | -1.4622 | 0.5417 | 0.5336 | 0.5376 | 0.1018 | 0.4664 |
| One-Class SVM | 95.0000 | 0.0130 | 0.6960 | 0.5117 | 0.5898 | 0.0504 | 0.4883 |
| One-Class SVM | 97.0000 | 2.4644 | 0.7782 | 0.4847 | 0.5973 | 0.0312 | 0.5153 |
| One-Class SVM | 99.0000 | 10.3783 | 0.8891 | 0.3801 | 0.5325 | 0.0107 | 0.6199 |
| LOF | 90.0000 | 0.3579 | 0.6046 | 0.6663 | 0.6340 | 0.0983 | 0.3337 |
| LOF | 95.0000 | 1.1482 | 0.6178 | 0.3724 | 0.4647 | 0.0520 | 0.6276 |
| LOF | 97.0000 | 2.1977 | 0.6481 | 0.2611 | 0.3723 | 0.0320 | 0.7389 |
| LOF | 99.0000 | 6.2781 | 0.4178 | 0.0351 | 0.0648 | 0.0110 | 0.9649 |


## 5. Unseen-Attack Holdout Protocol

- **Held-Out Attack Class:** `Web Attack` (`26` test flows)
- **Closed-Set Supervised Unseen-Class Recall:** `0.0000`
- **Benign-Only Autoencoder Unseen-Attack Recall:** `0.0000` (Benign vs Unseen ROC-AUC = `0.5608`)


## 6. SHAP Explainability Summary

- **XGBoost Top-5 (`TreeExplainer`, 0.65s):** `Destination Port` (0.8597), `Init_Win_bytes_backward` (0.3706), `Init_Win_bytes_forward` (0.2930), `min_seg_size_forward` (0.2793), `Bwd Packets/s` (0.2136)

- **MLP Top-5 (`GradientExplainer`, 2.81s):** `ACK Flag Count` (0.0073), `Min Packet Length` (0.0054), `PSH Flag Count` (0.0054), `Flow IAT Std` (0.0049), `Packet Length Variance` (0.0043)

- **MLP Explainer Trade-Off:** `GradientExplainer` speedup vs `KernelExplainer` = `3.21x`, Spearman rank correlation = `0.6026`.
