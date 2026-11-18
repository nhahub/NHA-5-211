# SMART IDS v3 — Model Comparison Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Unified Model Benchmark (Primary Split: `random`)

- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.

- **Unsupervised anomaly detectors (Autoencoder, Isolation Forest, One-Class SVM, LOF):** Trained strictly on benign flows and evaluated under identical binary novelty framing (Benign vs. Anomalous).


| Model | Evaluation_Framing | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | F1_Macro_CI95_Low | F1_Macro_CI95_High | ROC_AUC | PR_AUC | Latency_Mean_us | Latency_Median_us | Latency_P95_us | Samples_Per_Sec | Model_Size_MB | Training_Duration_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 13-class supervised | 0.9780 | 0.9762 | 0.9780 | 0.9767 | 0.6728 | 0.6236 | 0.7067 | 0.9905 | 0.8117 | 205.2900 | 202.9000 | 273.4500 | 23816.4000 | 2.5340 | 33.5500 |
| 1D-CNN | 13-class supervised | 0.9946 | 0.9936 | 0.9946 | 0.9940 | 0.8499 | 0.7942 | 0.8827 | 0.9853 | 0.8735 | 718.6600 | 693.4000 | 943.2700 | 16872.9000 | 1.3100 | 858.3600 |
| XGBoost | 13-class supervised | 0.9989 | 0.9989 | 0.9989 | 0.9988 | 0.9603 | 0.9013 | 0.9849 | 0.9995 | 0.9773 | 592.8600 | 535.0500 | 744.7400 | 237598.0000 | 2.9540 | 119.3100 |
| Autoencoder | Binary anomaly (Benign vs Anomalous) | 0.8678 | 0.6890 | 0.5130 | 0.5881 | 0.5881 | 0.5759 | 0.6001 | 0.8384 | 0.6651 | 269.3200 | 249.1000 | 416.9000 | 24332.6000 | 1.5480 | 36.1000 |
| Isolation Forest | Binary anomaly (Benign vs Anomalous) | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4394 | 0.4253 | 0.4531 | 0.7710 | 0.5005 | 8052.4800 | 7982.1000 | 8787.9100 | 72477.6000 | 2.0960 | 0.5000 |
| One-Class SVM | Binary anomaly (Benign vs Anomalous) | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5898 | 0.5770 | 0.6020 | 0.7467 | 0.5649 | 145.2300 | 134.8500 | 176.2300 | 22546.5000 | 0.4780 | 1.4800 |
| LOF | Binary anomaly (Benign vs Anomalous) | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4647 | 0.4516 | 0.4775 | 0.8607 | 0.5391 | 1299.0700 | 1265.5000 | 1705.4500 | 40917.4000 | 5.6660 | 0.5300 |


- **XGBoost Macro-F1 (95% Bootstrap CI, n=1000):** `0.9603` (`[0.9013, 0.9849]`, bootstrap std=`0.0193`)

- **1D-CNN Macro-F1 (95% Bootstrap CI, n=1000):** `0.8499` (`[0.7942, 0.8827]`, bootstrap std=`0.0232`)

- **MLP Macro-F1 (95% Bootstrap CI, n=1000):** `0.6728` (`[0.6236, 0.7067]`, bootstrap std=`0.0223`)

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
| XGBoost | 0.6868 | 0.5185 | 0.6868 | 0.5909 | 0.0860 | 0.1434 | 0.5441 |
| MLP | 0.6864 | 0.5177 | 0.6864 | 0.5902 | 0.0661 | 0.1432 | 0.4874 |
| 1D-CNN | 0.6860 | 0.4871 | 0.6860 | 0.5697 | 0.0691 | 0.1382 | 0.6507 |


### Cross-Day Binary Intrusion Detection (`by_day`: Benign vs. Attack)


| Model | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (1 - P_Benign) | 0.7768 | 0.9991 | 0.2876 | 0.4467 | 0.7656 | 0.7168 | 0.0001 | 0.7124 |
| XGBoost (Binary) | 0.7709 | 0.9981 | 0.2690 | 0.4238 | 0.7559 | 0.6932 | 0.0002 | 0.7310 |
| MLP (1 - P_Benign) | 0.7752 | 0.9942 | 0.2839 | 0.4417 | 0.7706 | 0.7136 | 0.0008 | 0.7161 |
| 1D-CNN (1 - P_Benign) | 0.7239 | 0.9654 | 0.1227 | 0.2177 | 0.9353 | 0.8443 | 0.0020 | 0.8773 |
| Autoencoder | 0.7562 | 0.7396 | 0.3419 | 0.4676 | 0.8784 | 0.7022 | 0.0549 | 0.6581 |


## 3. Shortcut Feature Ablation (With vs. Without `Destination Port`)

| Variant | Features | Test_Rows | Val_Macro_F1 | Test_Accuracy | Test_F1_Weighted | Test_F1_Macro | Test_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (With Destination Port) | 77 | 28324 | 0.9504 | 0.9989 | 0.9988 | 0.9603 | 0.9995 |
| XGBoost (Without Destination Port) | 76 | 28324 | 0.9563 | 0.9987 | 0.9987 | 0.9540 | 0.9990 |


- **Apples-to-Apples Row Parity Verified:** Identical post-deduplication train/val/test rows (`Test_Rows=28,324`, `sha256=85c3896d03d7...`) evaluated after full 77-feature leakage verification.

- **Delta Test Macro-F1 (without - with):** `-0.0063` | **Delta Val Macro-F1:** `+0.0059` | **Delta Test Weighted-F1:** `-0.0002`.

- **Top-5 XGBoost SHAP Features After Removing `Destination Port`:** `Init_Win_bytes_backward` (0.4962), `Init_Win_bytes_forward` (0.4449), `min_seg_size_forward` (0.3634), `Flow IAT Min` (0.3315), `Bwd Packets/s` (0.2949).



## 4. Per-Class Precision, Recall, F1 & Support (Test Partition)

> **Statistical Reliability Note:** Classes marked `Statistically Unreliable` have low held-out test support (`< 30` flows); one or two misclassified flows swing per-class F1 sharply and drive Macro-F1 variance.


### XGBoost Per-Class Metrics


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9994 | 0.9992 | 0.9993 | 23112 | Adequate |
| Bot | 0.9412 | 0.7619 | 0.8421 | 21 | Statistically Unreliable |
| DDoS | 0.9993 | 0.9993 | 0.9993 | 1357 | Adequate |
| DoS GoldenEye | 1.0000 | 1.0000 | 1.0000 | 109 | Adequate |
| DoS Hulk | 0.9974 | 1.0000 | 0.9987 | 1883 | Adequate |
| DoS Slowhttptest | 1.0000 | 1.0000 | 1.0000 | 56 | Adequate |
| DoS slowloris | 1.0000 | 0.9836 | 0.9917 | 61 | Adequate |
| FTP-Patator | 1.0000 | 1.0000 | 1.0000 | 70 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.6000 | 0.7500 | 5 | Statistically Unreliable |
| PortScan | 0.9931 | 1.0000 | 0.9965 | 1576 | Adequate |
| SSH-Patator | 1.0000 | 0.9783 | 0.9890 | 46 | Adequate |
| Web Attack | 1.0000 | 0.8462 | 0.9167 | 26 | Statistically Unreliable |


### 1D-CNN Per-Class Metrics


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9953 | 0.9984 | 0.9968 | 23112 | Adequate |
| Bot | 0.7647 | 0.6190 | 0.6842 | 21 | Statistically Unreliable |
| DDoS | 0.9985 | 0.9985 | 0.9985 | 1357 | Adequate |
| DoS GoldenEye | 0.9811 | 0.9541 | 0.9674 | 109 | Adequate |
| DoS Hulk | 0.9934 | 0.9649 | 0.9790 | 1883 | Adequate |
| DoS Slowhttptest | 0.9492 | 1.0000 | 0.9739 | 56 | Adequate |
| DoS slowloris | 0.9833 | 0.9672 | 0.9752 | 61 | Adequate |
| FTP-Patator | 0.9714 | 0.9714 | 0.9714 | 70 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.4000 | 0.5714 | 5 | Statistically Unreliable |
| PortScan | 0.9899 | 1.0000 | 0.9949 | 1576 | Adequate |
| SSH-Patator | 0.9167 | 0.9565 | 0.9362 | 46 | Adequate |
| Web Attack | 0.0000 | 0.0000 | 0.0000 | 26 | Statistically Unreliable |


### MLP Per-Class Metrics


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9802 | 0.9934 | 0.9868 | 23112 | Adequate |
| Bot | 0.0000 | 0.0000 | 0.0000 | 21 | Statistically Unreliable |
| DDoS | 0.9898 | 0.9978 | 0.9938 | 1357 | Adequate |
| DoS GoldenEye | 0.9694 | 0.8716 | 0.9179 | 109 | Adequate |
| DoS Hulk | 0.9917 | 0.8917 | 0.9390 | 1883 | Adequate |
| DoS Slowhttptest | 0.9434 | 0.8929 | 0.9174 | 56 | Adequate |
| DoS slowloris | 0.9298 | 0.8689 | 0.8983 | 61 | Adequate |
| FTP-Patator | 0.9787 | 0.6571 | 0.7863 | 70 | Adequate |
| Heartbleed | 0.4000 | 1.0000 | 0.5714 | 2 | Statistically Unreliable |
| Infiltration | 0.0000 | 0.0000 | 0.0000 | 5 | Statistically Unreliable |
| PortScan | 0.9231 | 0.9067 | 0.9149 | 1576 | Adequate |
| SSH-Patator | 1.0000 | 0.6957 | 0.8205 | 46 | Adequate |
| Web Attack | 0.0000 | 0.0000 | 0.0000 | 26 | Statistically Unreliable |


## 5. XGBoost 5-Fold Stratified Cross-Validation

- **Variant Evaluated:** `baseline` | **Development Pool Rows (Held-Out Test Excluded):** `162,577` | **Imbalance Strategy:** `none`
- **Macro-F1 (Mean +/- Std):** `0.9535 ± 0.0121`
- **Weighted-F1 (Mean +/- Std):** `0.9988 ± 0.0002` | **Accuracy:** `0.9988 ± 0.0002`


| Fold | Train_Rows | Val_Rows | Accuracy | F1_Weighted | F1_Macro | ROC_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 130061 | 32200 | 0.9991 | 0.9990 | 0.9515 | 0.9993 |
| 2 | 130061 | 32310 | 0.9987 | 0.9987 | 0.9351 | 0.9991 |
| 3 | 130062 | 32201 | 0.9989 | 0.9988 | 0.9627 | 0.9995 |
| 4 | 130062 | 32248 | 0.9989 | 0.9989 | 0.9659 | 0.9993 |
| 5 | 130062 | 32083 | 0.9985 | 0.9985 | 0.9522 | 0.9993 |


## 6. Ensemble Strategy Comparison (Test Partition)

| Strategy | accuracy | f1_macro | f1_weighted | roc_auc_macro_ovr | ece |
| --- | --- | --- | --- | --- | --- |
| best_single | 0.9989 | 0.9603 | 0.9988 | 0.9995 | 0.0003 |
| soft_voting | 0.9970 | 0.8420 | 0.9964 | 0.9992 | 0.0120 |
| confidence_weighted | 0.9983 | 0.9271 | 0.9982 | 0.9993 | 0.0119 |
| val_weighted_voting | 0.9972 | 0.8558 | 0.9966 | 0.9992 | 0.0111 |
| stacking | 0.9988 | 0.9606 | 0.9987 | 0.9994 | 0.0007 |


### Paired Bootstrap Macro-F1 Significance vs. Best Single (`XGBoost`)


| Strategy_vs_BestSingle | Observed_Diff | Mean_Diff | CI95_Low | CI95_High | Statistically_Significant | N_Bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| soft_voting | -0.1183 | -0.1217 | -0.1737 | -0.0908 | True | 1000 |
| confidence_weighted | -0.0331 | -0.0344 | -0.0532 | -0.0162 | True | 1000 |
| val_weighted_voting | -0.1044 | -0.1063 | -0.1284 | -0.0864 | True | 1000 |
| stacking | 0.0003 | 0.0004 | -0.0069 | 0.0086 | False | 1000 |


### Paired Bootstrap Macro-F1 Comparison: Stacking vs. Other Strategies


| Comparison | Observed_Diff | Mean_Diff | CI95_Low | CI95_High | Statistically_Significant | N_Bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| stacking_vs_best_single | 0.0003 | 0.0004 | -0.0069 | 0.0086 | False | 1000 |
| stacking_vs_soft_voting | 0.1186 | 0.1222 | 0.0933 | 0.1743 | True | 1000 |
| stacking_vs_confidence_weighted | 0.0334 | 0.0349 | 0.0162 | 0.0549 | True | 1000 |
| stacking_vs_val_weighted_voting | 0.1047 | 0.1068 | 0.0883 | 0.1276 | True | 1000 |


- **Empirical Conclusion:** Ensemble fusion 'stacking' (Test Macro-F1=0.9606) is comparable to XGBoost, difference not statistically significant (XGBoost=0.9603; observed delta=+0.0003, paired bootstrap n=1000 mean diff=+0.0004, 95% CI=[-0.0069, +0.0086] includes 0; statistically_significant=False). Standalone XGBoost is kept as the supervised default with the Autoencoder orthogonal.


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
| Baseline (300 trees, depth=6, lr=0.1) | 0.9504 | 0.9988 | 0.9331 | 0.0042 |
| Optuna Tuned (12 trials) | 0.9563 | 0.9989 | 0.9381 | 0.0038 |


- **Selected Variant:** `baseline` (Baseline CV Macro-F1 = `0.9535 ± 0.0121`, Val Macro-F1 delta=`+0.0059`, Meaningful Improvement=`False`)

- **Decision Basis:** No meaningful improvement: Optuna tuned val Macro-F1 (0.9563) changed by +0.0059 vs baseline (0.9504), within baseline CV std=0.0121; retaining baseline.

- **How CV Std is Computed:** Unified 5-fold StratifiedKFold on the development pool (162,577 rows = train+val, held-out test excluded) with per-fold preprocessing and 'none' imbalance handling (sample ddof=1 std across folds).



## 9. Architectural Notes on 1D-CNN vs. MLP & XGBoost

Tabular CICFlowMeter features do not form a physical time series; 1D convolutions scan adjacent engineered statistics (packet sizes, IATs, flags). Measured validation and test F1/latency above quantify whether 1D-CNN yields measurable benefit over the dense MLP and XGBoost.


## 10. Preprocessing vs. Model Inference Latency

- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** mean = `838.22 µs`, median = `827.20 µs`, P95 = `953.77 µs` (`1193.0` flows/sec).
