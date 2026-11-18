# SMART IDS v3 — Model Comparison Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Unified Model Benchmark (Primary Split: `random`)

- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.

- **Unsupervised anomaly detectors (Autoencoder, Isolation Forest, One-Class SVM, LOF):** Trained strictly on benign flows and evaluated under identical binary novelty framing (Benign vs. Anomalous).


| Model | Evaluation_Framing | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | F1_Macro_CI95_Low | F1_Macro_CI95_High | ROC_AUC | PR_AUC | Latency_Mean_us | Latency_Median_us | Latency_P95_us | Samples_Per_Sec | Model_Size_MB | Training_Duration_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 13-class supervised | 0.9767 | 0.9784 | 0.9767 | 0.9772 | 0.7592 | 0.6849 | 0.7923 | 0.9942 | 0.8421 | 314.5700 | 270.7000 | 581.8300 | 13175.8000 | 2.5340 | 121.9000 |
| 1D-CNN | 13-class supervised | 0.9904 | 0.9925 | 0.9904 | 0.9913 | 0.8447 | 0.7649 | 0.8819 | 0.9933 | 0.8752 | 1036.0300 | 967.7000 | 1366.2800 | 11091.3000 | 1.3100 | 1031.3800 |
| XGBoost | 13-class supervised | 0.9989 | 0.9989 | 0.9989 | 0.9988 | 0.9594 | 0.9000 | 0.9833 | 0.9996 | 0.9939 | 2153.7900 | 578.3000 | 1586.6400 | 205731.3000 | 3.3380 | 224.9200 |
| Autoencoder | Binary anomaly (Benign vs Anomalous) | 0.8678 | 0.6890 | 0.5130 | 0.5881 | 0.5881 | 0.5759 | 0.6001 | 0.8384 | 0.6651 | 336.8800 | 270.2000 | 683.4700 | 14135.2000 | 1.5480 | 73.8500 |
| Isolation Forest | Binary anomaly (Benign vs Anomalous) | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4394 | 0.4253 | 0.4531 | 0.7710 | 0.5005 | 13371.0700 | 12691.2500 | 18341.9800 | 42900.5000 | 2.0960 | 0.8100 |
| One-Class SVM | Binary anomaly (Benign vs Anomalous) | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5898 | 0.5770 | 0.6020 | 0.7467 | 0.5649 | 314.6800 | 298.3500 | 574.6100 | 16024.8000 | 0.4780 | 2.5200 |
| LOF | Binary anomaly (Benign vs Anomalous) | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4647 | 0.4516 | 0.4775 | 0.8607 | 0.5391 | 1743.2100 | 1683.6500 | 2470.7600 | 34474.6000 | 5.6660 | 0.5700 |


- **XGBoost Macro-F1 (95% Bootstrap CI, n=1000):** `0.9594` (`[0.9000, 0.9833]`, bootstrap std=`0.0192`)

- **1D-CNN Macro-F1 (95% Bootstrap CI, n=1000):** `0.8447` (`[0.7649, 0.8819]`, bootstrap std=`0.0299`)

- **MLP Macro-F1 (95% Bootstrap CI, n=1000):** `0.7592` (`[0.6849, 0.7923]`, bootstrap std=`0.0266`)

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
| XGBoost | 0.6868 | 0.5179 | 0.6868 | 0.5905 | 0.0782 | 0.1433 | 0.5445 |
| MLP | 0.6853 | 0.5183 | 0.6853 | 0.5902 | 0.0661 | 0.1432 | 0.5444 |
| 1D-CNN | 0.6787 | 0.5118 | 0.6787 | 0.5835 | 0.0654 | 0.1416 | 0.6827 |


### Cross-Day Binary Intrusion Detection (`by_day`: Benign vs. Attack)


| Model | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (1 - P_Benign) | 0.7760 | 0.9989 | 0.2851 | 0.4435 | 0.7680 | 0.7006 | 0.0001 | 0.7149 |
| XGBoost (Binary) | 0.7726 | 0.9963 | 0.2747 | 0.4307 | 0.7672 | 0.6922 | 0.0005 | 0.7253 |
| MLP (1 - P_Benign) | 0.7756 | 0.9831 | 0.2883 | 0.4458 | 0.8579 | 0.7733 | 0.0023 | 0.7117 |
| 1D-CNN (1 - P_Benign) | 0.7638 | 0.9124 | 0.2717 | 0.4187 | 0.9272 | 0.8343 | 0.0119 | 0.7283 |
| Autoencoder | 0.7562 | 0.7396 | 0.3419 | 0.4676 | 0.8784 | 0.7022 | 0.0549 | 0.6581 |


## 3. Shortcut Feature Ablation (With vs. Without `Destination Port`)

| Variant | Features | Test_Rows | Val_Macro_F1 | Test_Accuracy | Test_F1_Weighted | Test_F1_Macro | Test_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (With Destination Port) | 77 | 28324 | 0.9505 | 0.9989 | 0.9988 | 0.9594 | 0.9996 |
| XGBoost (Without Destination Port) | 76 | 28324 | 0.9500 | 0.9987 | 0.9986 | 0.9540 | 0.9992 |


- **Apples-to-Apples Row Parity Verified:** Identical post-deduplication train/val/test rows (`Test_Rows=28,324`, `sha256=85c3896d03d7...`) evaluated after full 77-feature leakage verification.

- **Delta Test Macro-F1 (without - with):** `-0.0054` | **Delta Val Macro-F1:** `-0.0005` | **Delta Test Weighted-F1:** `-0.0002`.

- **Top-5 XGBoost SHAP Features After Removing `Destination Port`:** `min_seg_size_forward` (0.5056), `Init_Win_bytes_forward` (0.4793), `Init_Win_bytes_backward` (0.4162), `Flow IAT Min` (0.2998), `Fwd IAT Min` (0.2803).



## 4. Per-Class Precision, Recall, F1 & Support (Test Partition)

> **Statistical Reliability Note:** Classes marked `Statistically Unreliable` have low held-out test support (`< 30` flows); one or two misclassified flows swing per-class F1 sharply and drive Macro-F1 variance.


### XGBoost Per-Class Metrics


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9995 | 0.9993 | 0.9994 | 23112 | Adequate |
| Bot | 0.9444 | 0.8095 | 0.8718 | 21 | Statistically Unreliable |
| DDoS | 1.0000 | 0.9993 | 0.9996 | 1357 | Adequate |
| DoS GoldenEye | 1.0000 | 1.0000 | 1.0000 | 109 | Adequate |
| DoS Hulk | 0.9979 | 1.0000 | 0.9989 | 1883 | Adequate |
| DoS Slowhttptest | 0.9655 | 1.0000 | 0.9825 | 56 | Adequate |
| DoS slowloris | 1.0000 | 0.9508 | 0.9748 | 61 | Adequate |
| FTP-Patator | 0.9859 | 1.0000 | 0.9929 | 70 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.6000 | 0.7500 | 5 | Statistically Unreliable |
| PortScan | 0.9924 | 1.0000 | 0.9962 | 1576 | Adequate |
| SSH-Patator | 1.0000 | 0.9783 | 0.9890 | 46 | Adequate |
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
| BENIGN | 0.9843 | 0.9875 | 0.9859 | 23112 | Adequate |
| Bot | 0.0000 | 0.0000 | 0.0000 | 21 | Statistically Unreliable |
| DDoS | 0.9977 | 0.9683 | 0.9828 | 1357 | Adequate |
| DoS GoldenEye | 0.8651 | 1.0000 | 0.9277 | 109 | Adequate |
| DoS Hulk | 0.9906 | 0.8933 | 0.9394 | 1883 | Adequate |
| DoS Slowhttptest | 0.8750 | 1.0000 | 0.9333 | 56 | Adequate |
| DoS slowloris | 0.9500 | 0.9344 | 0.9421 | 61 | Adequate |
| FTP-Patator | 0.8228 | 0.9286 | 0.8725 | 70 | Adequate |
| Heartbleed | 0.6667 | 1.0000 | 0.8000 | 2 | Statistically Unreliable |
| Infiltration | 0.2727 | 0.6000 | 0.3750 | 5 | Statistically Unreliable |
| PortScan | 0.9125 | 0.9467 | 0.9293 | 1576 | Adequate |
| SSH-Patator | 0.8113 | 0.9348 | 0.8687 | 46 | Adequate |
| Web Attack | 0.2022 | 0.6923 | 0.3130 | 26 | Statistically Unreliable |


## 5. XGBoost 5-Fold Stratified Cross-Validation

- **Development Pool Rows (Held-Out Test Excluded):** `162,577` | **Imbalance Strategy:** `smote`
- **Macro-F1 (Mean +/- Std):** `0.9670 ± 0.0076`
- **Weighted-F1 (Mean +/- Std):** `0.9988 ± 0.0003` | **Accuracy:** `0.9988 ± 0.0003`


| Fold | Train_Rows | Val_Rows | Accuracy | F1_Weighted | F1_Macro | ROC_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 137137 | 32200 | 0.9992 | 0.9992 | 0.9666 | 0.9993 |
| 2 | 137137 | 32310 | 0.9988 | 0.9988 | 0.9677 | 0.9990 |
| 3 | 137142 | 32201 | 0.9990 | 0.9990 | 0.9712 | 0.9996 |
| 4 | 137142 | 32248 | 0.9986 | 0.9987 | 0.9747 | 0.9994 |
| 5 | 137142 | 32083 | 0.9985 | 0.9985 | 0.9547 | 0.9991 |


## 6. Ensemble Strategy Comparison (Test Partition)

| Strategy | accuracy | f1_macro | f1_weighted | roc_auc_macro_ovr | ece |
| --- | --- | --- | --- | --- | --- |
| best_single | 0.9989 | 0.9594 | 0.9988 | 0.9996 | 0.0004 |
| soft_voting | 0.9960 | 0.9000 | 0.9959 | 0.9990 | 0.0120 |
| confidence_weighted | 0.9983 | 0.9284 | 0.9983 | 0.9990 | 0.0115 |
| val_weighted_voting | 0.9961 | 0.9013 | 0.9960 | 0.9990 | 0.0118 |
| stacking | 0.9987 | 0.9661 | 0.9987 | 0.9991 | 0.0007 |


- **Empirical Conclusion:** Ensemble fusion 'stacking' achieved Test Macro-F1=0.9661, outperforming best_single (XGBoost=0.9594) by +0.0067.


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


- **Selected Variant:** `baseline` (Baseline CV Macro-F1 std=`0.0470`, Val Macro-F1 delta=`+0.0120`, Meaningful Improvement=`False`)

- **Decision Basis:** No meaningful improvement: Optuna tuned val Macro-F1 (0.9625) changed by +0.0120 vs baseline (0.9505), within baseline CV std=0.0470; retaining baseline.



## 9. Architectural Notes on 1D-CNN vs. MLP & XGBoost

Tabular CICFlowMeter features do not form a physical time series; 1D convolutions scan adjacent engineered statistics (packet sizes, IATs, flags). Measured validation and test F1/latency above quantify whether 1D-CNN yields measurable benefit over the dense MLP and XGBoost.


## 10. Preprocessing vs. Model Inference Latency

- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** mean = `1464.69 µs`, median = `1184.80 µs`, P95 = `2666.32 µs` (`682.7` flows/sec).
