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


- **Timestamp Alignment Audit:** status=`directory_not_found`, strategy=`none (CSV row order fallback)`, keys=`[]`, matched=`0`, unmatched=`2,830,743`.

- **Engelen et al. (2021) Label Correction Audit:** status=`file_not_found`, applied=`False`, matched=`0`, relabeled=`0`, dropped=`0`.

- **Partition Sizes:** Train (pre-SMOTE) = `138,190`, Train (post-SMOTE) = `145,714`, Validation = `24,112`, Test = `28,324`.


## 2. Class Imbalance Experiment (Unweighted vs. Class Weighting vs. SMOTE)

Selection mode: `per_model` (neural candidates capped at `10` epochs during selection; full training runs only for each model's winner).


| Model | Strategy | Val_Macro_F1 | Val_Weighted_F1 | Val_Macro_Recall | Val_MLogLoss | Selected |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | Unweighted ('none') | 0.6924 | 0.9763 | 0.6586 | 0.0626 |  |
| MLP | Class Weighting ('class_weight') | 0.4046 | 0.8575 | 0.9270 | 0.5528 |  |
| MLP | Hybrid SMOTE ('smote') | 0.7695 | 0.9788 | 0.7924 | 0.0569 | YES |
| 1D-CNN | Unweighted ('none') | 0.5094 | 0.9780 | 0.4634 | 0.0722 |  |
| 1D-CNN | Class Weighting ('class_weight') | 0.4150 | 0.8395 | 0.8840 | 0.4949 |  |
| 1D-CNN | Hybrid SMOTE ('smote') | 0.7501 | 0.9787 | 0.8455 | 0.0642 | YES |
| XGBoost | Unweighted ('none') | 0.9518 | 0.9989 | 0.9323 | 0.0046 | YES |
| XGBoost | Class Weighting ('class_weight') | 0.9446 | 0.9981 | 0.9583 | 0.0092 |  |
| XGBoost | Hybrid SMOTE ('smote') | 0.9518 | 0.9989 | 0.9439 | 0.0044 |  |


- **MLP (`smote`):** Data-driven selection chose 'smote' with highest validation Macro-F1=0.7695 (Weighted-F1=0.9788, Macro-Recall=0.7924, mlogloss=0.0569), beating runner-up 'none' by +0.0770 Macro-F1 (> tolerance=0.0020; Unweighted ('none') Macro-F1=0.6924 (Weighted-F1=0.9763, mlogloss=0.0626); Class-Weight ('class_weight') Macro-F1=0.4046 (Weighted-F1=0.8575, mlogloss=0.5528)).

- **1D-CNN (`smote`):** Data-driven selection chose 'smote' with highest validation Macro-F1=0.7501 (Weighted-F1=0.9787, Macro-Recall=0.8455, mlogloss=0.0642), beating runner-up 'none' by +0.2407 Macro-F1 (> tolerance=0.0020; Unweighted ('none') Macro-F1=0.5094 (Weighted-F1=0.9780, mlogloss=0.0722); Class-Weight ('class_weight') Macro-F1=0.4150 (Weighted-F1=0.8395, mlogloss=0.4949)).

- **XGBoost (`none`):** Tie within tolerance (tol=0.0020): Unweighted ('none') Macro-F1=0.9518, Hybrid SMOTE ('smote') Macro-F1=0.9518 fall within 0.0020 of peak validation Macro-F1=0.9518 (gap=0.0000 vs 'smote'); tie within tolerance, simplest strategy selected -> 'none' (Weighted-F1=0.9989, Macro-Recall=0.9323, mlogloss=0.0046; Hybrid SMOTE ('smote') Macro-F1=0.9518 (Weighted-F1=0.9989, mlogloss=0.0044); Class-Weight ('class_weight') Macro-F1=0.9446 (Weighted-F1=0.9981, mlogloss=0.0092)).

- **Overall Summary:** Per-model imbalance selection (independent validation Macro-F1 with tie_tolerance=0.0020, neural_epochs_cap=10): MLP -> 'smote', CNN -> 'smote', XGBoost -> 'none'.


## 3. Probability Calibration (Validation Temperature Scaling)

| Model | temperature | ece_before | ece_after | ece_delta | brier_before | brier_after | nll_before | nll_after | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.7948 | 0.0083 | 0.0057 | -0.0025 | 0.0299 | 0.0297 | 0.0498 | 0.0483 | Recalibrated (T=0.7948, ECE delta=-0.002535) |
| CNN | 0.8749 | 0.0045 | 0.0033 | -0.0013 | 0.0113 | 0.0111 | 0.0220 | 0.0218 | Recalibrated (T=0.8749, ECE delta=-0.001259) |
| XGBoost | 0.9095 | 0.0005 | 0.0003 | -0.0003 | 0.0017 | 0.0017 | 0.0042 | 0.0041 | Already well-calibrated near T=1.0 |


- **Calibration Evidence:** Bounded float64 Brent optimization recalibrated overconfident probabilities on `MLP` (T=0.7948, ECE 0.008277 -> 0.005742), `CNN` (T=0.8749, ECE 0.004522 -> 0.003263).


## 4. Unsupervised Anomaly Detection & Classical Baselines

| Detector | Accuracy | Precision | Recall | F1 | F1_CI95_Low | F1_CI95_High | ROC_AUC | PR_AUC | FPR | Holdout_Class_Recall | Holdout_Class_ROC_AUC | Val_Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoencoder | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 0.0522 | 0.0000 | 0.5816 |  |
| Autoencoder (Deep Reg) | 0.8668 | 0.6872 | 0.5063 | 0.5831 | 0.5704 | 0.5947 | 0.7908 | 0.6293 | 0.0520 | 0.0000 | 0.5371 |  |
| VAE | 0.8693 | 0.6964 | 0.5132 | 0.5910 | 0.5780 | 0.6025 | 0.7755 | 0.6231 | 0.0504 | 0.0000 | 0.5940 |  |
| Isolation Forest | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4253 | 0.4531 | 0.7710 | 0.5005 | 0.0499 | 0.0000 | 0.6101 |  |
| One-Class SVM | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5770 | 0.6020 | 0.7467 | 0.5649 | 0.0504 | 0.0000 | 0.6385 |  |
| LOF | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4516 | 0.4775 | 0.8607 | 0.5391 | 0.0520 | 0.1154 | 0.4188 |  |
| Score-Level Fusion (degenerate: equals Autoencoder) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 0.0522 | 0.0000 | 0.5816 | YES |


Validation selected a single-detector fusion; no gain from fusing detectors on this data.


- **Assessment:** Best binary F1 at 95th-pct threshold: Autoencoder (F1=0.6926 [0.6820, 0.7020], Recall=0.6523, FPR=0.0522); best overall ROC-AUC ranking: Autoencoder (ROC-AUC=0.8676); validation-selected option (leave-attack-out non-test): 'Score-Level Fusion (degenerate: equals Autoencoder)'. Validation selected a single-detector fusion; no gain from fusing detectors on this data.


- **Autoencoder 95th Percentile Benign Validation Threshold:** `0.023815`
- **Autoencoder Binary Test Metrics:** Accuracy=`0.8934`, Precision=`0.7382`, Recall=`0.6523`, F1=`0.6926`, ROC-AUC=`0.8676`, FPR=`0.0522`, FNR=`0.3477`


| Detector | percentile | threshold | precision | recall | f1 | fpr | fnr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Autoencoder | 90.0000 | 0.0127 | 0.6172 | 0.7187 | 0.6641 | 0.1005 | 0.2813 |
| Autoencoder | 95.0000 | 0.0238 | 0.7382 | 0.6523 | 0.6926 | 0.0522 | 0.3477 |
| Autoencoder | 97.0000 | 0.0386 | 0.7977 | 0.5652 | 0.6617 | 0.0323 | 0.4348 |
| Autoencoder | 99.0000 | 0.0994 | 0.8997 | 0.4472 | 0.5975 | 0.0112 | 0.5528 |
| Autoencoder (Deep Reg) | 90.0000 | 0.0698 | 0.5447 | 0.5393 | 0.5420 | 0.1017 | 0.4607 |
| Autoencoder (Deep Reg) | 95.0000 | 0.1434 | 0.6872 | 0.5063 | 0.5831 | 0.0520 | 0.4937 |
| Autoencoder (Deep Reg) | 97.0000 | 0.1997 | 0.7648 | 0.4785 | 0.5887 | 0.0332 | 0.5215 |
| Autoencoder (Deep Reg) | 99.0000 | 0.4471 | 0.8914 | 0.4252 | 0.5757 | 0.0117 | 0.5748 |
| VAE | 90.0000 | 0.0357 | 0.5459 | 0.5409 | 0.5434 | 0.1015 | 0.4591 |
| VAE | 95.0000 | 0.0705 | 0.6964 | 0.5132 | 0.5910 | 0.0504 | 0.4868 |
| VAE | 97.0000 | 0.0968 | 0.7747 | 0.4956 | 0.6045 | 0.0325 | 0.5044 |
| VAE | 99.0000 | 0.2107 | 0.9010 | 0.4298 | 0.5820 | 0.0106 | 0.5702 |
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
| Score-Level Fusion (degenerate: equals Autoencoder) | 90.0000 | 0.0090 | 0.6172 | 0.7187 | 0.6641 | 0.1005 | 0.2813 |
| Score-Level Fusion (degenerate: equals Autoencoder) | 95.0000 | 0.0179 | 0.7382 | 0.6523 | 0.6926 | 0.0522 | 0.3477 |
| Score-Level Fusion (degenerate: equals Autoencoder) | 97.0000 | 0.0297 | 0.7977 | 0.5652 | 0.6617 | 0.0323 | 0.4348 |
| Score-Level Fusion (degenerate: equals Autoencoder) | 99.0000 | 0.0782 | 0.8997 | 0.4472 | 0.5975 | 0.0112 | 0.5528 |


## 5. Unseen-Attack Holdout Protocol

- **Held-Out Attack Class:** `Web Attack` (`26` test flows)
- **Closed-Set Supervised Unseen-Class Recall:** `0.0000`
- **Benign-Only Autoencoder Unseen-Attack Recall:** `0.0000` (Benign vs Unseen ROC-AUC = `0.5626`)


## 6. SHAP Explainability Summary

- **XGBoost Top-5 (`TreeExplainer`, 0.46s):** `Destination Port` (0.8597), `Init_Win_bytes_backward` (0.3706), `Init_Win_bytes_forward` (0.2930), `min_seg_size_forward` (0.2793), `Bwd Packets/s` (0.2136)

- **MLP Top-5 (`GradientExplainer`, 2.43s):** `Min Packet Length` (0.0065), `ACK Flag Count` (0.0061), `PSH Flag Count` (0.0054), `Destination Port` (0.0037), `Packet Length Std` (0.0036)

- **MLP Explainer Trade-Off:** `GradientExplainer` speedup vs `KernelExplainer` = `2.86x`, Spearman rank correlation = `0.7402`.
