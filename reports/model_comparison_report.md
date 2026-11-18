# SMART IDS v3 — Model Comparison Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Unified Model Benchmark (Primary Split: `random`)

- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.

- **Unsupervised anomaly detectors (Autoencoder, Isolation Forest, One-Class SVM, LOF):** Trained strictly on benign flows and evaluated under identical binary novelty framing (Benign vs. Anomalous).


| Model | Evaluation_Framing | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | F1_Macro_CI95_Low | F1_Macro_CI95_High | ROC_AUC | PR_AUC | Latency_Mean_us | Latency_Median_us | Latency_P95_us | Samples_Per_Sec | Model_Size_MB | Training_Duration_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 13-class supervised | 0.9766 | 0.9790 | 0.9766 | 0.9772 | 0.7934 | 0.7544 | 0.8176 | 0.9962 | 0.8487 | 189.1100 | 179.9500 | 257.2800 | 24425.2000 | 2.5340 | 59.6300 |
| 1D-CNN | 13-class supervised | 0.9904 | 0.9925 | 0.9904 | 0.9913 | 0.8447 | 0.7649 | 0.8819 | 0.9933 | 0.8752 | 722.0100 | 699.2500 | 925.2500 | 16651.9000 | 1.3100 | 889.3800 |
| XGBoost | 13-class supervised | 0.9989 | 0.9989 | 0.9989 | 0.9989 | 0.9675 | 0.9307 | 0.9843 | 0.9995 | 0.9935 | 399.4300 | 369.9500 | 556.4600 | 370459.7000 | 2.0490 | 114.4200 |
| Autoencoder | Binary anomaly (Benign vs Anomalous) | 0.8678 | 0.6890 | 0.5130 | 0.5881 | 0.5881 | 0.5759 | 0.6001 | 0.8384 | 0.6651 | 256.9400 | 228.5000 | 418.7800 | 24771.3000 | 1.5480 | 35.8600 |
| Isolation Forest | Binary anomaly (Benign vs Anomalous) | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4394 | 0.4253 | 0.4531 | 0.7710 | 0.5005 | 7918.2300 | 7842.8000 | 8724.5100 | 74084.4000 | 2.0960 | 0.4900 |
| One-Class SVM | Binary anomaly (Benign vs Anomalous) | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5898 | 0.5770 | 0.6020 | 0.7467 | 0.5649 | 158.8100 | 137.9500 | 264.4700 | 22377.0000 | 0.4780 | 1.4700 |
| LOF | Binary anomaly (Benign vs Anomalous) | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4647 | 0.4516 | 0.4775 | 0.8607 | 0.5391 | 1262.7700 | 1188.9500 | 1629.3500 | 40543.4000 | 5.6660 | 0.5100 |


- **XGBoost Macro-F1 (95% Bootstrap CI, n=1000):** `0.9675` (`[0.9307, 0.9843]`, bootstrap std=`0.0139`)

- **1D-CNN Macro-F1 (95% Bootstrap CI, n=1000):** `0.8447` (`[0.7649, 0.8819]`, bootstrap std=`0.0299`)

- **MLP Macro-F1 (95% Bootstrap CI, n=1000):** `0.7934` (`[0.7544, 0.8176]`, bootstrap std=`0.0161`)

- **Autoencoder Binary F1 (95% Bootstrap CI, n=1000):** `0.5881` (`[0.5759, 0.6001]`, bootstrap std=`0.0061`)

- **Isolation Forest Binary F1 (95% Bootstrap CI, n=1000):** `0.4394` (`[0.4253, 0.4531]`, bootstrap std=`0.0069`)

- **One-Class SVM Binary F1 (95% Bootstrap CI, n=1000):** `0.5898` (`[0.5770, 0.6020]`, bootstrap std=`0.0062`)

- **LOF Binary F1 (95% Bootstrap CI, n=1000):** `0.4647` (`[0.4516, 0.4775]`, bootstrap std=`0.0066`)



## 2. Cross-Day Temporal Split Protocol (`by_day`: Mon–Wed Train / Thu Val / Thu–Fri Test)

- **Partitions (`split_strategy='by_day'`):** Train pre-SMOTE=`111,814`, Validation (Thursday half)=`15,408`, Test (Thursday remainder + Friday)=`63,207`.

- **Thursday–Friday Unseen Attack Classes:** `Bot`, `DDoS`, `Infiltration`, `PortScan`, `Web Attack`.

- **Protocol Interpretation:** In CICIDS2017, all attack families occurring on Thursday and Friday (Bot, DDoS, Infiltration, PortScan, Web Attack) are absent from Monday-Wednesday training data. Closed-set multi-class Macro-F1 therefore cannot attribute unseen Thursday-Friday labels, whereas the Cross-Day Binary table evaluates Benign-vs-Attack generalization directly.


### Closed-Set Multi-Class Evaluation (`by_day`)


| Model | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro_All | F1_Macro_Present_Test | ROC_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.6867 | 0.4944 | 0.6867 | 0.5749 | 0.0697 | 0.1395 | 0.5420 |
| MLP | 0.6853 | 0.5183 | 0.6853 | 0.5902 | 0.0661 | 0.1432 | 0.5444 |
| 1D-CNN | 0.6787 | 0.5118 | 0.6787 | 0.5835 | 0.0654 | 0.1416 | 0.6827 |


### Cross-Day Binary Intrusion Detection (`by_day`: Benign vs. Attack)


| Model | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (1 - P_Benign) | 0.7325 | 0.9962 | 0.1464 | 0.2553 | 0.7525 | 0.7287 | 0.0003 | 0.8536 |
| XGBoost (Binary) | 0.7172 | 0.9918 | 0.0977 | 0.1779 | 0.8456 | 0.7632 | 0.0004 | 0.9023 |
| MLP (1 - P_Benign) | 0.7756 | 0.9831 | 0.2883 | 0.4458 | 0.8579 | 0.7733 | 0.0023 | 0.7117 |
| 1D-CNN (1 - P_Benign) | 0.7638 | 0.9124 | 0.2717 | 0.4187 | 0.9272 | 0.8343 | 0.0119 | 0.7283 |
| Autoencoder | 0.7562 | 0.7396 | 0.3419 | 0.4676 | 0.8784 | 0.7022 | 0.0549 | 0.6581 |


## 3. Shortcut Feature Ablation (With vs. Without `Destination Port`)

| Variant | Features | Test_Rows | Val_Macro_F1 | Test_Accuracy | Test_F1_Weighted | Test_F1_Macro | Test_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (With Destination Port) | 77 | 28324 | 0.9625 | 0.9989 | 0.9989 | 0.9675 | 0.9995 |
| XGBoost (Without Destination Port) | 76 | 28324 | 0.9495 | 0.9987 | 0.9986 | 0.9537 | 0.9976 |


- **Apples-to-Apples Row Parity Verified:** Identical post-deduplication train/val/test rows (`Test_Rows=28,324`, `sha256=85c3896d03d7...`) evaluated after full 77-feature leakage verification.

- **Delta Test Macro-F1 (without - with):** `-0.0138` | **Delta Val Macro-F1:** `-0.0130` | **Delta Test Weighted-F1:** `-0.0002`.

- **Top-5 XGBoost SHAP Features After Removing `Destination Port`:** `Init_Win_bytes_forward` (0.6980), `min_seg_size_forward` (0.5503), `Flow IAT Min` (0.4499), `Init_Win_bytes_backward` (0.4401), `Bwd Packets/s` (0.2968).



## 4. Per-Class Precision, Recall, F1 & Support (Test Partition)

> **Statistical Reliability Note:** Classes marked `Statistically Unreliable` have low held-out test support (`< 30` flows); one or two misclassified flows swing per-class F1 sharply and drive Macro-F1 variance.


### XGBoost Per-Class Metrics


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9995 | 0.9993 | 0.9994 | 23112 | Adequate |
| Bot | 0.9412 | 0.7619 | 0.8421 | 21 | Statistically Unreliable |
| DDoS | 0.9993 | 0.9993 | 0.9993 | 1357 | Adequate |
| DoS GoldenEye | 1.0000 | 1.0000 | 1.0000 | 109 | Adequate |
| DoS Hulk | 0.9974 | 1.0000 | 0.9987 | 1883 | Adequate |
| DoS Slowhttptest | 0.9655 | 1.0000 | 0.9825 | 56 | Adequate |
| DoS slowloris | 1.0000 | 0.9508 | 0.9748 | 61 | Adequate |
| FTP-Patator | 1.0000 | 1.0000 | 1.0000 | 70 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.8000 | 0.8889 | 5 | Statistically Unreliable |
| PortScan | 0.9931 | 1.0000 | 0.9965 | 1576 | Adequate |
| SSH-Patator | 0.9783 | 0.9783 | 0.9783 | 46 | Adequate |
| Web Attack | 1.0000 | 0.8462 | 0.9167 | 26 | Statistically Unreliable |


### 1D-CNN Per-Class Metrics


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9961 | 0.9925 | 0.9943 | 23112 | Adequate |
| Bot | 0.7222 | 0.6190 | 0.6667 | 21 | Statistically Unreliable |
| DDoS | 0.9978 | 0.9985 | 0.9982 | 1357 | Adequate |
| DoS GoldenEye | 0.9160 | 1.0000 | 0.9561 | 109 | Adequate |
| DoS Hulk | 0.9778 | 0.9607 | 0.9692 | 1883 | Adequate |
| DoS Slowhttptest | 0.9032 | 1.0000 | 0.9492 | 56 | Adequate |
| DoS slowloris | 0.8923 | 0.9508 | 0.9206 | 61 | Adequate |
| FTP-Patator | 0.8961 | 0.9857 | 0.9388 | 70 | Adequate |
| Heartbleed | 0.6667 | 1.0000 | 0.8000 | 2 | Statistically Unreliable |
| Infiltration | 0.6000 | 0.6000 | 0.6000 | 5 | Statistically Unreliable |
| PortScan | 0.9937 | 1.0000 | 0.9968 | 1576 | Adequate |
| SSH-Patator | 0.8036 | 0.9783 | 0.8824 | 46 | Adequate |
| Web Attack | 0.1959 | 0.7308 | 0.3089 | 26 | Statistically Unreliable |


### MLP Per-Class Metrics


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9808 | 0.9910 | 0.9859 | 23112 | Adequate |
| Bot | 0.4615 | 0.2857 | 0.3529 | 21 | Statistically Unreliable |
| DDoS | 0.9971 | 0.9963 | 0.9967 | 1357 | Adequate |
| DoS GoldenEye | 0.9130 | 0.9633 | 0.9375 | 109 | Adequate |
| DoS Hulk | 0.9865 | 0.8933 | 0.9376 | 1883 | Adequate |
| DoS Slowhttptest | 0.9180 | 1.0000 | 0.9573 | 56 | Adequate |
| DoS slowloris | 0.9206 | 0.9508 | 0.9355 | 61 | Adequate |
| FTP-Patator | 0.7222 | 0.9286 | 0.8125 | 70 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 0.2143 | 0.6000 | 0.3158 | 5 | Statistically Unreliable |
| PortScan | 0.9743 | 0.8668 | 0.9174 | 1576 | Adequate |
| SSH-Patator | 0.8269 | 0.9348 | 0.8776 | 46 | Adequate |
| Web Attack | 0.1818 | 0.6923 | 0.2880 | 26 | Statistically Unreliable |


## 5. XGBoost 5-Fold Stratified Cross-Validation

- **Variant Evaluated:** `tuned` | **Development Pool Rows (Held-Out Test Excluded):** `162,577` | **Imbalance Strategy:** `smote`
- **Macro-F1 (Mean +/- Std):** `0.9641 ± 0.0125`
- **Weighted-F1 (Mean +/- Std):** `0.9988 ± 0.0003` | **Accuracy:** `0.9988 ± 0.0002`


| Fold | Train_Rows | Val_Rows | Accuracy | F1_Weighted | F1_Macro | ROC_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 137137 | 32200 | 0.9991 | 0.9991 | 0.9663 | 0.9994 |
| 2 | 137137 | 32310 | 0.9989 | 0.9989 | 0.9699 | 0.9987 |
| 3 | 137142 | 32201 | 0.9989 | 0.9989 | 0.9669 | 0.9995 |
| 4 | 137142 | 32248 | 0.9987 | 0.9987 | 0.9748 | 0.9991 |
| 5 | 137142 | 32083 | 0.9985 | 0.9985 | 0.9425 | 0.9994 |


## 6. Ensemble Strategy Comparison (Test Partition)

| Strategy | accuracy | f1_macro | f1_weighted | roc_auc_macro_ovr | ece |
| --- | --- | --- | --- | --- | --- |
| best_single | 0.9989 | 0.9675 | 0.9989 | 0.9995 | 0.0003 |
| soft_voting | 0.9959 | 0.9185 | 0.9958 | 0.9990 | 0.0119 |
| confidence_weighted | 0.9984 | 0.9454 | 0.9984 | 0.9990 | 0.0111 |
| val_weighted_voting | 0.9970 | 0.9173 | 0.9969 | 0.9989 | 0.0109 |
| stacking | 0.9988 | 0.9653 | 0.9988 | 0.9990 | 0.0007 |


### Paired Bootstrap Macro-F1 Significance vs. Best Single (`XGBoost`)


| Strategy_vs_BestSingle | Observed_Diff | Mean_Diff | CI95_Low | CI95_High | Statistically_Significant | N_Bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| soft_voting | -0.0490 | -0.0514 | -0.0959 | -0.0209 | True | 1000 |
| confidence_weighted | -0.0221 | -0.0239 | -0.0652 | -0.0029 | True | 1000 |
| val_weighted_voting | -0.0502 | -0.0529 | -0.0966 | -0.0238 | True | 1000 |
| stacking | -0.0022 | -0.0022 | -0.0139 | 0.0108 | False | 1000 |


### Paired Bootstrap Macro-F1 Comparison: Stacking vs. Other Strategies


| Comparison | Observed_Diff | Mean_Diff | CI95_Low | CI95_High | Statistically_Significant | N_Bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| stacking_vs_best_single | -0.0022 | -0.0022 | -0.0139 | 0.0108 | False | 1000 |
| stacking_vs_soft_voting | 0.0468 | 0.0492 | 0.0175 | 0.0958 | True | 1000 |
| stacking_vs_confidence_weighted | 0.0199 | 0.0217 | -0.0028 | 0.0639 | False | 1000 |
| stacking_vs_val_weighted_voting | 0.0480 | 0.0507 | 0.0194 | 0.0974 | True | 1000 |


- **Empirical Conclusion:** Ensemble fusion 'stacking' (Test Macro-F1=0.9653) is comparable to XGBoost, difference not statistically significant (XGBoost=0.9675; observed delta=-0.0022, paired bootstrap n=1000 mean diff=-0.0022, 95% CI=[-0.0139, +0.0108] includes 0; statistically_significant=False). Standalone XGBoost is kept as the supervised default with the Autoencoder orthogonal.


## 7. Unsupervised Anomaly Detector Baselines (Benign-Only Training)

| Detector | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | FPR | Holdout_Class_Recall | Holdout_Class_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoencoder | 0.8678 | 0.6890 | 0.5130 | 0.5881 | 0.8384 | 0.6651 | 0.0522 | 0.0000 | 0.5852 |
| Isolation Forest | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.7710 | 0.5005 | 0.0499 | 0.0000 | 0.6101 |
| One-Class SVM | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.7467 | 0.5649 | 0.0504 | 0.0000 | 0.6385 |
| LOF | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.8607 | 0.5391 | 0.0520 | 0.1154 | 0.4188 |


- **Verdict:** Best binary F1 at 95th-pct threshold: One-Class SVM (F1=0.5898, Recall=0.5117, FPR=0.0504); best overall ROC-AUC ranking: LOF (ROC-AUC=0.8607). Autoencoder achieved F1=0.5881 / ROC-AUC=0.8384, confirming it works best as a supplementary anomaly signal alongside supervised XGBoost rather than as a standalone detector.


### Percentile Threshold Trade-Off Sweep Across Detectors


| Detector | Percentile | Threshold | Precision | Recall | F1 | FPR | FNR |
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


## 8. XGBoost Baseline vs. Optuna Hyperparameter Optimization

| Variant | val_macro_f1 | val_weighted_f1 | val_macro_recall | val_mlogloss |
| --- | --- | --- | --- | --- |
| Baseline (300 trees, depth=6, lr=0.1) | 0.9505 | 0.9989 | 0.9439 | 0.0041 |
| Optuna Tuned (12 trials) | 0.9625 | 0.9990 | 0.9578 | 0.0039 |


- **Selected Variant:** `tuned` (Baseline CV Macro-F1 = `0.9670 ± 0.0076`, Val Macro-F1 delta=`+0.0120`, Meaningful Improvement=`True`)

- **Decision Basis:** Meaningful improvement: tuned val Macro-F1 (0.9625) beat baseline (0.9505) by +0.0120 (> CV std=0.0076).

- **How CV Std is Computed:** Unified 5-fold StratifiedKFold on the development pool (162,577 rows = train+val, held-out test excluded) with per-fold preprocessing and 'smote' imbalance handling (sample ddof=1 std across folds).



## 9. Architectural Notes on 1D-CNN vs. MLP & XGBoost

Tabular CICFlowMeter features do not form a physical time series; 1D convolutions scan adjacent engineered statistics (packet sizes, IATs, flags). Measured validation and test F1/latency above quantify whether 1D-CNN yields measurable benefit over the dense MLP and XGBoost.


## 10. Preprocessing vs. Model Inference Latency

- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** mean = `857.17 µs`, median = `801.75 µs`, P95 = `1108.55 µs` (`1166.6` flows/sec).
