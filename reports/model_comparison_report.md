# SMART IDS v3 — Model Comparison Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Unified Model Benchmark (Primary Split: `random`)

- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.

- **Unsupervised anomaly detectors (Autoencoder, Isolation Forest, One-Class SVM, LOF):** Trained strictly on benign flows and evaluated under identical binary novelty framing (Benign vs. Anomalous).


| Model | Imbalance_Strategy | Evaluation_Framing | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | F1_Macro_Reliable | F1_Macro_CI95_Low | F1_Macro_CI95_High | ROC_AUC | PR_AUC | Latency_Median_us | Latency_P95_us | Latency_Mean_us | Samples_Per_Sec | Model_Size_MB | Training_Duration_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | smote | 13-class supervised | 0.9766 | 0.9790 | 0.9766 | 0.9772 | 0.7934 | 0.9286 | 0.7544 | 0.8176 | 0.9962 | 0.8487 | 173.8500 | 221.9900 | 174.9600 | 24457.6000 | 2.5340 | 57.8400 |
| 1D-CNN | smote | 13-class supervised | 0.9904 | 0.9925 | 0.9904 | 0.9913 | 0.8447 | 0.9562 | 0.7649 | 0.8819 | 0.9933 | 0.8752 | 623.0000 | 724.0500 | 632.8100 | 16521.5000 | 1.3100 | 859.5500 |
| XGBoost | none | 13-class supervised | 0.9989 | 0.9989 | 0.9989 | 0.9988 | 0.9603 | 0.9972 | 0.9013 | 0.9849 | 0.9995 | 0.9773 | 405.0500 | 523.6800 | 459.0200 | 249219.2000 | 2.9540 | 320.3300 |
| Autoencoder | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6926 | N/A | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 198.7000 | 280.1000 | 215.2500 | 17682.5000 | 1.5480 | 68.8000 |
| Autoencoder (Deep Reg) | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8668 | 0.6872 | 0.5063 | 0.5831 | 0.5831 | N/A | 0.5704 | 0.5947 | 0.7908 | 0.6293 | 233.5000 | 308.9300 | 236.5200 | 22696.0000 | 5.0700 | 53.4600 |
| VAE | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8693 | 0.6964 | 0.5132 | 0.5910 | 0.5910 | N/A | 0.5780 | 0.6025 | 0.7755 | 0.6231 | 208.5500 | 268.8900 | 208.9900 | 24408.6000 | 1.5710 | 33.4600 |
| Isolation Forest | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4394 | N/A | 0.4253 | 0.4531 | 0.7710 | 0.5005 | 6958.0000 | 7768.4900 | 7054.1600 | 79549.0000 | 2.0960 | 0.4800 |
| One-Class SVM | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5898 | N/A | 0.5770 | 0.6020 | 0.7467 | 0.5649 | 136.4500 | 166.0600 | 145.0600 | 23264.0000 | 0.4780 | 1.4900 |
| LOF | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4647 | N/A | 0.4516 | 0.4775 | 0.8607 | 0.5391 | 1368.8000 | 1965.2300 | 1367.6700 | 41327.3000 | 5.6660 | 0.4900 |
| Score-Level Fusion (AE+OCSVM+IF) | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6926 | N/A | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


- **XGBoost Macro-F1 (95% Bootstrap CI, n=1000):** `0.9603` (`[0.9013, 0.9849]`, bootstrap std=`0.0193`)

- **1D-CNN Macro-F1 (95% Bootstrap CI, n=1000):** `0.8447` (`[0.7649, 0.8819]`, bootstrap std=`0.0299`)

- **MLP Macro-F1 (95% Bootstrap CI, n=1000):** `0.7934` (`[0.7544, 0.8176]`, bootstrap std=`0.0161`)

- **Autoencoder Binary F1 (95% Bootstrap CI, n=1000):** `0.6926` (`[0.6820, 0.7020]`, bootstrap std=`0.0053`)

- **Autoencoder (Deep Reg) Binary F1 (95% Bootstrap CI, n=1000):** `0.5831` (`[0.5704, 0.5947]`, bootstrap std=`0.0063`)

- **VAE Binary F1 (95% Bootstrap CI, n=1000):** `0.5910` (`[0.5780, 0.6025]`, bootstrap std=`0.0062`)

- **Isolation Forest Binary F1 (95% Bootstrap CI, n=1000):** `0.4394` (`[0.4253, 0.4531]`, bootstrap std=`0.0069`)

- **One-Class SVM Binary F1 (95% Bootstrap CI, n=1000):** `0.5898` (`[0.5770, 0.6020]`, bootstrap std=`0.0062`)

- **LOF Binary F1 (95% Bootstrap CI, n=1000):** `0.4647` (`[0.4516, 0.4775]`, bootstrap std=`0.0066`)

- **Score-Level Fusion (AE+OCSVM+IF) Binary F1 (95% Bootstrap CI, n=1000):** `0.6926` (`[0.6820, 0.7020]`, bootstrap std=`0.0053`)



## 2. Chronological Drift Evaluation (`time_block`: Per-Class Earliest ~60% Train / Next ~20% Val / Last ~20% Test)

- **Partitions (`split_strategy='time_block'`):** Train pre-SMOTE=`114,762`, Validation=`37,952`, Test=`37,862` (Ordering=`Capture_Order (capture session schedule + CSV row index; Timestamp column absent in MachineLearningCVE CSVs)`, `leakage_checks.passed=True`).

- **Warning:** `time_block ordering = CSV row order (no real Timestamp)`

- **Timestamp Alignment:** status=`directory_not_found`, strategy=`none (CSV row order fallback)`, keys=`[]`, matched=`0` / `2,830,743` (unmatched=`2,830,743`).

- **Low-Support / Tiny-Class Chronological Allocations:** `Heartbleed`: Low-support chronological split (n=11): 7 earliest train / 2 middle val / 2 latest test.

- **Protocol Interpretation:** Closed-set temporal drift split where every class is trained on its earliest ~60% of flows and evaluated on its latest ~20%, isolating concept drift over capture time without introducing unseen attack labels.


| Model | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | F1_Macro_Reliable | F1_Macro_CI95_Low | F1_Macro_CI95_High | ROC_AUC_OVR | PR_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.9924 | 0.9916 | 0.9924 | 0.9917 | 0.7518 | 0.9173 | 0.7098 | 0.8250 | 0.9929 | 0.8049 |
| 1D-CNN | 0.9521 | 0.9489 | 0.9521 | 0.9456 | 0.4376 | 0.5689 | 0.4265 | 0.4787 | 0.9700 | 0.7024 |
| MLP | 0.9502 | 0.9376 | 0.9502 | 0.9400 | 0.2767 | 0.3597 | 0.2752 | 0.3004 | 0.9231 | 0.4422 |


- **XGBoost (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.7518` (`[0.7098, 0.8250]`, bootstrap std=`0.0273`)

- **1D-CNN (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.4376` (`[0.4265, 0.4787]`, bootstrap std=`0.0137`)

- **MLP (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.2767` (`[0.2752, 0.3004]`, bootstrap std=`0.0081`)

- **Autoencoder (`time_block`, benign-only binary):** Accuracy=`0.8456`, F1=`0.5003` (`[0.4892, 0.5107]`, n=1000), ROC-AUC=`0.8263`, FPR=`0.0589`.



### XGBoost Per-Class Metrics (`time_block` Test Partition)


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9914 | 0.9999 | 0.9956 | 30907 | Adequate |
| Bot | 0.0000 | 0.0000 | 0.0000 | 28 | Statistically Unreliable |
| DDoS | 1.0000 | 0.9939 | 0.9970 | 1809 | Adequate |
| DoS GoldenEye | 0.9528 | 0.8288 | 0.8864 | 146 | Adequate |
| DoS Hulk | 0.9979 | 0.9376 | 0.9668 | 2516 | Adequate |
| DoS Slowhttptest | 0.9494 | 0.9868 | 0.9677 | 76 | Adequate |
| DoS slowloris | 1.0000 | 0.7160 | 0.8345 | 81 | Adequate |
| FTP-Patator | 1.0000 | 0.9785 | 0.9891 | 93 | Adequate |
| Heartbleed | 0.0000 | 0.0000 | 0.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.4286 | 0.6000 | 7 | Statistically Unreliable |
| PortScan | 1.0000 | 0.9952 | 0.9976 | 2100 | Adequate |
| SSH-Patator | 1.0000 | 1.0000 | 1.0000 | 62 | Adequate |
| Web Attack | 0.8235 | 0.4000 | 0.5385 | 35 | Adequate |


### Time-Block Error Analysis (Failing Classes with Recall < 0.85)


#### Confusion-Matrix Breakdown for Failing Classes


| Class | Test_Support | Recall | F1_Score | Predicted_As_Breakdown |
| --- | --- | --- | --- | --- |
| Bot | 28 | 0.0000 | 0.0000 | BENIGN: 28 (100.0%) |
| Heartbleed | 2 | 0.0000 | 0.0000 | BENIGN: 2 (100.0%) |
| Web Attack | 35 | 0.4000 | 0.5385 | BENIGN: 16 (45.7%), Web Attack: 14 (40.0%), DoS GoldenEye: 5 (14.3%) |
| Infiltration | 7 | 0.4286 | 0.6000 | BENIGN: 4 (57.1%), Infiltration: 3 (42.9%) |
| DoS slowloris | 81 | 0.7160 | 0.8345 | DoS slowloris: 58 (71.6%), BENIGN: 19 (23.5%), DoS Slowhttptest: 4 (4.9%) |
| DoS GoldenEye | 146 | 0.8288 | 0.8864 | DoS GoldenEye: 121 (82.9%), BENIGN: 25 (17.1%) |


#### Top Shifted Features Train-vs-Test per Failing Class (KS Statistic)


| Class | Train_Count | Test_Count | Rank | Feature | KS_Statistic | KS_PValue | Std_Mean_Shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bot | 82 | 28 | 1 | Flow Duration | 1.0000 | 0.0000 | 0.0287 |
| Bot | 82 | 28 | 2 | Total Length of Fwd Packets | 1.0000 | 0.0000 | -0.8171 |
| Bot | 82 | 28 | 3 | Fwd Packet Length Max | 1.0000 | 0.0000 | -0.4455 |
| Bot | 82 | 28 | 4 | Fwd Packet Length Mean | 1.0000 | 0.0000 | -0.5981 |
| Bot | 82 | 28 | 5 | Flow Bytes/s | 1.0000 | 0.0000 | -0.1109 |
| Heartbleed | 7 | 2 | 1 | Fwd IAT Std | 1.0000 | 0.0556 | -0.0005 |
| Heartbleed | 7 | 2 | 2 | Init_Win_bytes_backward | 1.0000 | 0.0556 | 0.0141 |
| Heartbleed | 7 | 2 | 3 | Total Backward Packets | 0.8571 | 0.1667 | -11.7189 |
| Heartbleed | 7 | 2 | 4 | Flow IAT Max | 0.8571 | 0.1667 | -0.0231 |
| Heartbleed | 7 | 2 | 5 | Fwd IAT Mean | 0.8571 | 0.1667 | -0.0001 |
| Web Attack | 106 | 35 | 1 | Flow IAT Mean | 0.5059 | 0.0000 | -0.1882 |
| Web Attack | 106 | 35 | 2 | Bwd Packets/s | 0.5059 | 0.0000 | 0.1001 |
| Web Attack | 106 | 35 | 3 | Flow IAT Std | 0.4965 | 0.0000 | -0.1177 |
| Web Attack | 106 | 35 | 4 | Flow Packets/s | 0.4965 | 0.0000 | 0.0230 |
| Web Attack | 106 | 35 | 5 | Fwd Packets/s | 0.4965 | 0.0000 | 0.0122 |
| Infiltration | 22 | 7 | 1 | Flow IAT Std | 0.8182 | 0.0004 | 1.4044 |
| Infiltration | 22 | 7 | 2 | Flow IAT Max | 0.8182 | 0.0004 | 1.4163 |
| Infiltration | 22 | 7 | 3 | Fwd IAT Max | 0.7208 | 0.0039 | 1.2435 |
| Infiltration | 22 | 7 | 4 | Idle Max | 0.7208 | 0.0039 | 1.2550 |
| Infiltration | 22 | 7 | 5 | Flow Packets/s | 0.6818 | 0.0062 | -0.0300 |
| DoS slowloris | 242 | 81 | 1 | Idle Mean | 0.2904 | 0.0001 | -0.6578 |
| DoS slowloris | 242 | 81 | 2 | Bwd IAT Min | 0.2865 | 0.0001 | -1.2123 |
| DoS slowloris | 242 | 81 | 3 | Bwd IAT Max | 0.2858 | 0.0001 | -1.2177 |
| DoS slowloris | 242 | 81 | 4 | Bwd IAT Total | 0.2741 | 0.0002 | -0.8278 |
| DoS slowloris | 242 | 81 | 5 | Bwd IAT Mean | 0.2741 | 0.0002 | -1.6249 |
| DoS GoldenEye | 436 | 146 | 1 | Fwd Packet Length Mean | 0.3100 | 0.0000 | 0.4889 |
| DoS GoldenEye | 436 | 146 | 2 | Avg Fwd Segment Size | 0.3100 | 0.0000 | 0.4889 |
| DoS GoldenEye | 436 | 146 | 3 | Fwd Packet Length Std | 0.2965 | 0.0000 | 0.4972 |
| DoS GoldenEye | 436 | 146 | 4 | Fwd Packet Length Max | 0.2850 | 0.0000 | 0.3080 |
| DoS GoldenEye | 436 | 146 | 5 | Total Length of Fwd Packets | 0.2758 | 0.0000 | 0.0212 |


### Time-Block Recall-vs-FPR Mitigation Experiment


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('none') | Argmax | argmax P(c) | 0.0001 | 0.9613 | 0.4276 | 0.9800 | 0.7518 | Bot=0.000, Heartbleed=0.000, Web Attack=0.543, Infiltration=0.429, DoS slowloris=0.765, DoS GoldenEye=0.829 |
| XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0144 | 0.9797 | 0.7816 | 0.9588 | 0.8512 | Bot=0.000, Heartbleed=1.000, Web Attack=0.714, Infiltration=1.000, DoS slowloris=0.975, DoS GoldenEye=1.000 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0475 | 0.0962 | 0.9699 | 0.7448 | 0.8091 | N/A | Bot=0.000, Heartbleed=1.000, Web Attack=0.543, Infiltration=1.000, DoS slowloris=0.926, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.2449 | 0.0005 | 0.9806 | 0.5616 | 0.9891 | N/A | Bot=0.000, Heartbleed=0.000, Web Attack=0.743, Infiltration=0.714, DoS slowloris=0.926, DoS GoldenEye=0.986 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1008 | 0.0590 | 0.9691 | 0.7448 | 0.8687 | N/A | Bot=0.000, Heartbleed=1.000, Web Attack=0.543, Infiltration=1.000, DoS slowloris=0.926, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.5884 | 0.0001 | 0.9554 | 0.3732 | 0.9770 | N/A | Bot=0.000, Heartbleed=0.000, Web Attack=0.514, Infiltration=0.143, DoS slowloris=0.753, DoS GoldenEye=0.829 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1385 | 0.0502 | 0.9687 | 0.7169 | 0.8840 | N/A | Bot=0.000, Heartbleed=1.000, Web Attack=0.543, Infiltration=0.857, DoS slowloris=0.901, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.5986 | 0.0001 | 0.9531 | 0.3732 | 0.9758 | N/A | Bot=0.000, Heartbleed=0.000, Web Attack=0.514, Infiltration=0.143, DoS slowloris=0.753, DoS GoldenEye=0.829 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.1652 | 0.0462 | 0.9687 | 0.7169 | 0.8911 | N/A | Bot=0.000, Heartbleed=1.000, Web Attack=0.543, Infiltration=0.857, DoS slowloris=0.901, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.6067 | 0.0001 | 0.9524 | 0.3684 | 0.9754 | N/A | Bot=0.000, Heartbleed=0.000, Web Attack=0.486, Infiltration=0.143, DoS slowloris=0.753, DoS GoldenEye=0.829 |


- **Mitigation Assessment (`time_block`):** XGBoost ('class_weight') (Argmax) raises focus attack recall from 0.4276 to 0.7816 with bounded Benign FPR=0.0144 vs baseline 0.0001.


## 3. Open-Set Cross-Day Protocol (`by_day`: Mon–Wed Train / Thu Val / Thu–Fri Unseen Attacks)

- **Partitions (`split_strategy='by_day'`):** Train pre-SMOTE=`111,814`, Validation (Thursday half)=`15,408`, Test (Thursday remainder + Friday)=`63,207` (`leakage_checks.passed=True`).

- **Thursday–Friday Unseen Attack Classes:** `Bot`, `DDoS`, `Infiltration`, `PortScan`, `Web Attack`.


> **Open-Set Interpretation:** Open-set cross-day protocol (by_day): by construction in CICIDS2017, all Thursday and Friday attack families (Bot, DDoS, Infiltration, PortScan, Web Attack) have zero training instances in Monday-Wednesday data. Closed-set multi-class Macro-F1 is therefore near zero by design rather than ordinary generalization degradation. The real empirical finding is that supervised XGBoost misses ~71% of unseen attack flows (binary recall=0.2876, precision=0.9991), which is why benign-only anomaly detectors are kept as a supplementary zero-day signal.


### Open-Set Cross-Day Binary Detection (`by_day`: Benign vs. Attack)


| Model | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (1 - P_Benign) | 0.7768 | 0.9991 | 0.2876 | 0.4467 | 0.7656 | 0.7168 | 0.0001 | 0.7124 |
| XGBoost (Binary) | 0.7709 | 0.9981 | 0.2690 | 0.4238 | 0.7559 | 0.6932 | 0.0002 | 0.7310 |
| MLP (1 - P_Benign) | 0.7752 | 0.9942 | 0.2839 | 0.4417 | 0.7706 | 0.7136 | 0.0008 | 0.7161 |
| 1D-CNN (1 - P_Benign) | 0.7239 | 0.9654 | 0.1227 | 0.2177 | 0.9353 | 0.8443 | 0.0020 | 0.8773 |
| Autoencoder | 0.7562 | 0.7396 | 0.3419 | 0.4676 | 0.8784 | 0.7022 | 0.0549 | 0.6581 |


### Closed-Set Multi-Class Reference Under Open-Set Split (`by_day`)


| Model | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro_All | F1_Macro_Reliable | F1_Macro_Present_Test | ROC_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.6868 | 0.5185 | 0.6868 | 0.5909 | 0.0860 | 0.1720 | 0.1434 | 0.5441 |
| MLP | 0.6864 | 0.5177 | 0.6864 | 0.5902 | 0.0661 | 0.1719 | 0.1432 | 0.4874 |
| 1D-CNN | 0.6860 | 0.4871 | 0.6860 | 0.5697 | 0.0691 | 0.1659 | 0.1382 | 0.6507 |


### Open-Set Cross-Day Recall-vs-FPR Mitigation Experiment (Thu–Fri Unseen Attacks)


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('none') | Argmax | argmax P(c) | 0.0001 | 0.2876 | 0.1349 | 0.4467 | 0.0860 | Bot=0.000, DDoS=0.628, Infiltration=0.000, PortScan=0.001, Web Attack=0.045 |
| XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0005 | 0.1711 | 0.1896 | 0.2920 | 0.0764 | Bot=0.000, DDoS=0.367, Infiltration=0.000, PortScan=0.002, Web Attack=0.580 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0616 | 0.0844 | 0.4449 | 0.3546 | 0.5458 | N/A | Bot=0.015, DDoS=0.629, Infiltration=0.778, PortScan=0.294, Web Attack=0.057 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.0001 | 0.1186 | 0.4818 | 0.4077 | 0.5532 | N/A | Bot=0.000, DDoS=1.000, Infiltration=0.000, PortScan=0.039, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1169 | 0.0550 | 0.3665 | 0.3139 | 0.4930 | N/A | Bot=0.015, DDoS=0.629, Infiltration=0.722, PortScan=0.147, Web Attack=0.057 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.0002 | 0.0553 | 0.4811 | 0.4074 | 0.6005 | N/A | Bot=0.000, DDoS=0.999, Infiltration=0.000, PortScan=0.038, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1252 | 0.0473 | 0.3596 | 0.3113 | 0.4915 | N/A | Bot=0.015, DDoS=0.629, Infiltration=0.722, PortScan=0.134, Web Attack=0.057 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.0005 | 0.0280 | 0.4776 | 0.4060 | 0.6207 | N/A | Bot=0.000, DDoS=0.996, Infiltration=0.000, PortScan=0.034, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.1888 | 0.0337 | 0.3303 | 0.2448 | 0.4704 | N/A | Bot=0.015, DDoS=0.628, Infiltration=0.444, PortScan=0.080, Web Attack=0.057 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.0020 | 0.0103 | 0.3290 | 0.3412 | 0.4868 | N/A | Bot=0.000, DDoS=0.680, Infiltration=0.000, PortScan=0.026, Web Attack=1.000 |


- **Mitigation Assessment (`by_day`):** Lower P(Benign) Gate (1 - P_Benign > tau) (P99) raises focus attack recall from 0.1349 to 0.3412 with bounded Benign FPR=0.0103 vs baseline 0.0001.


## 4. Shortcut Feature Ablation (With vs. Without `Destination Port`)

| Variant | Features | Test_Rows | Val_Macro_F1 | Test_Accuracy | Test_F1_Weighted | Test_F1_Macro | Test_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (With Destination Port) | 77 | 28324 | 0.9504 | 0.9989 | 0.9988 | 0.9603 | 0.9995 |
| XGBoost (Without Destination Port) | 76 | 28324 | 0.9563 | 0.9987 | 0.9987 | 0.9540 | 0.9990 |


- **Apples-to-Apples Row Parity Verified:** Identical post-deduplication train/val/test rows (`Test_Rows=28,324`, `sha256=85c3896d03d7...`) evaluated after full 77-feature leakage verification.

- **Delta Test Macro-F1 (without - with):** `-0.0063` | **Delta Val Macro-F1:** `+0.0059` | **Delta Test Weighted-F1:** `-0.0002`.

- **Top-5 XGBoost SHAP Features After Removing `Destination Port`:** `Init_Win_bytes_backward` (0.4962), `Init_Win_bytes_forward` (0.4449), `min_seg_size_forward` (0.3634), `Flow IAT Min` (0.3315), `Bwd Packets/s` (0.2949).



## 5. Per-Class Precision, Recall, F1 & Support (Test Partition)

> **Statistical Reliability Note:** Classes marked `Statistically Unreliable` have low held-out test support (`< 30` flows); one or two misclassified flows swing per-class F1 sharply and drive Macro-F1 variance.


### XGBoost Per-Class Metrics


| Class | Precision | Recall | Recall_CI95_Low | Recall_CI95_High | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9994 | 0.9992 | 0.9988 | 0.9995 | 0.9993 | 23112 | Adequate |
| Bot | 0.9412 | 0.7619 | 0.5491 | 0.8937 | 0.8421 | 21 | Statistically Unreliable |
| DDoS | 0.9993 | 0.9993 | 0.9958 | 0.9999 | 0.9993 | 1357 | Adequate |
| DoS GoldenEye | 1.0000 | 1.0000 | 0.9660 | 1.0000 | 1.0000 | 109 | Adequate |
| DoS Hulk | 0.9974 | 1.0000 | 0.9980 | 1.0000 | 0.9987 | 1883 | Adequate |
| DoS Slowhttptest | 1.0000 | 1.0000 | 0.9358 | 1.0000 | 1.0000 | 56 | Adequate |
| DoS slowloris | 1.0000 | 0.9836 | 0.9128 | 0.9971 | 0.9917 | 61 | Adequate |
| FTP-Patator | 1.0000 | 1.0000 | 0.9480 | 1.0000 | 1.0000 | 70 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 0.3424 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.6000 | 0.2307 | 0.8824 | 0.7500 | 5 | Statistically Unreliable |
| PortScan | 0.9931 | 1.0000 | 0.9976 | 1.0000 | 0.9965 | 1576 | Adequate |
| SSH-Patator | 1.0000 | 0.9783 | 0.8866 | 0.9962 | 0.9890 | 46 | Adequate |
| Web Attack | 1.0000 | 0.8462 | 0.6647 | 0.9385 | 0.9167 | 26 | Statistically Unreliable |


### 1D-CNN Per-Class Metrics


| Class | Precision | Recall | Recall_CI95_Low | Recall_CI95_High | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9961 | 0.9925 | 0.9913 | 0.9935 | 0.9943 | 23112 | Adequate |
| Bot | 0.7222 | 0.6190 | 0.4088 | 0.7925 | 0.6667 | 21 | Statistically Unreliable |
| DDoS | 0.9978 | 0.9985 | 0.9946 | 0.9996 | 0.9982 | 1357 | Adequate |
| DoS GoldenEye | 0.9160 | 1.0000 | 0.9660 | 1.0000 | 0.9561 | 109 | Adequate |
| DoS Hulk | 0.9778 | 0.9607 | 0.9509 | 0.9686 | 0.9692 | 1883 | Adequate |
| DoS Slowhttptest | 0.9032 | 1.0000 | 0.9358 | 1.0000 | 0.9492 | 56 | Adequate |
| DoS slowloris | 0.8923 | 0.9508 | 0.8651 | 0.9831 | 0.9206 | 61 | Adequate |
| FTP-Patator | 0.8961 | 0.9857 | 0.9234 | 0.9975 | 0.9388 | 70 | Adequate |
| Heartbleed | 0.6667 | 1.0000 | 0.3424 | 1.0000 | 0.8000 | 2 | Statistically Unreliable |
| Infiltration | 0.6000 | 0.6000 | 0.2307 | 0.8824 | 0.6000 | 5 | Statistically Unreliable |
| PortScan | 0.9937 | 1.0000 | 0.9976 | 1.0000 | 0.9968 | 1576 | Adequate |
| SSH-Patator | 0.8036 | 0.9783 | 0.8866 | 0.9962 | 0.8824 | 46 | Adequate |
| Web Attack | 0.1959 | 0.7308 | 0.5392 | 0.8630 | 0.3089 | 26 | Statistically Unreliable |


### MLP Per-Class Metrics


| Class | Precision | Recall | Recall_CI95_Low | Recall_CI95_High | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9808 | 0.9910 | 0.9897 | 0.9921 | 0.9859 | 23112 | Adequate |
| Bot | 0.4615 | 0.2857 | 0.1381 | 0.4996 | 0.3529 | 21 | Statistically Unreliable |
| DDoS | 0.9971 | 0.9963 | 0.9914 | 0.9984 | 0.9967 | 1357 | Adequate |
| DoS GoldenEye | 0.9130 | 0.9633 | 0.9094 | 0.9856 | 0.9375 | 109 | Adequate |
| DoS Hulk | 0.9865 | 0.8933 | 0.8785 | 0.9064 | 0.9376 | 1883 | Adequate |
| DoS Slowhttptest | 0.9180 | 1.0000 | 0.9358 | 1.0000 | 0.9573 | 56 | Adequate |
| DoS slowloris | 0.9206 | 0.9508 | 0.8651 | 0.9831 | 0.9355 | 61 | Adequate |
| FTP-Patator | 0.7222 | 0.9286 | 0.8434 | 0.9691 | 0.8125 | 70 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 0.3424 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 0.2143 | 0.6000 | 0.2307 | 0.8824 | 0.3158 | 5 | Statistically Unreliable |
| PortScan | 0.9743 | 0.8668 | 0.8491 | 0.8826 | 0.9174 | 1576 | Adequate |
| SSH-Patator | 0.8269 | 0.9348 | 0.8250 | 0.9776 | 0.8776 | 46 | Adequate |
| Web Attack | 0.1818 | 0.6923 | 0.5001 | 0.8350 | 0.2880 | 26 | Statistically Unreliable |


## 6. XGBoost 5-Fold Stratified Cross-Validation

- **Variant Evaluated:** `baseline` | **Development Pool Rows (Held-Out Test Excluded):** `162,577` | **Imbalance Strategy:** `none`
- **Fold Mean Macro-F1 (Mean +/- Std):** `0.9535 ± 0.0121`
- **Pooled OOF Macro-F1 (All Classes vs. Reliable `support >= 30`):** `0.9550` vs. `0.9561` (total OOF evaluated=`161,042`)
- **Weighted-F1 (Mean +/- Std):** `0.9988 ± 0.0002` | **Accuracy:** `0.9988 ± 0.0002`


| Fold | Train_Rows | Val_Rows | Accuracy | F1_Weighted | F1_Macro | F1_Macro_Reliable | ROC_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 130061 | 32200 | 0.9991 | 0.9990 | 0.9515 | 0.9940 | 0.9993 |
| 2 | 130061 | 32310 | 0.9987 | 0.9987 | 0.9351 | 0.9866 | 0.9991 |
| 3 | 130062 | 32201 | 0.9989 | 0.9988 | 0.9627 | 0.9881 | 0.9995 |
| 4 | 130062 | 32248 | 0.9989 | 0.9989 | 0.9659 | 0.9948 | 0.9993 |
| 5 | 130062 | 32083 | 0.9985 | 0.9985 | 0.9522 | 0.9829 | 0.9993 |


### Pooled Out-of-Fold (OOF) Per-Class Evaluation (5-Fold CV Dev Pool, Wilson 95% Recall CI)


> Each non-collision development-pool sample is scored out-of-fold exactly once, multiplying rare-class effective evaluation counts ~5x relative to the single 15% test split.


| Class | Precision | Recall | Recall_CI95_Low | Recall_CI95_High | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9994 | 0.9993 | 0.9992 | 0.9994 | 0.9994 | 131504 | Adequate |
| Bot | 0.8100 | 0.6983 | 0.6095 | 0.7743 | 0.7500 | 116 | Adequate |
| DDoS | 0.9996 | 0.9990 | 0.9979 | 0.9995 | 0.9993 | 7688 | Adequate |
| DoS GoldenEye | 0.9935 | 0.9919 | 0.9812 | 0.9965 | 0.9927 | 617 | Adequate |
| DoS Hulk | 0.9975 | 0.9994 | 0.9988 | 0.9997 | 0.9985 | 10677 | Adequate |
| DoS Slowhttptest | 0.9812 | 0.9812 | 0.9597 | 0.9914 | 0.9812 | 320 | Adequate |
| DoS slowloris | 0.9911 | 0.9767 | 0.9547 | 0.9881 | 0.9838 | 343 | Adequate |
| FTP-Patator | 1.0000 | 0.9975 | 0.9858 | 0.9996 | 0.9987 | 396 | Adequate |
| Heartbleed | 1.0000 | 0.8889 | 0.5650 | 0.9801 | 0.9412 | 9 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.7097 | 0.5341 | 0.8390 | 0.8302 | 31 | Adequate |
| PortScan | 0.9943 | 0.9988 | 0.9978 | 0.9993 | 0.9965 | 8928 | Adequate |
| SSH-Patator | 1.0000 | 0.9886 | 0.9670 | 0.9961 | 0.9943 | 263 | Adequate |
| Web Attack | 0.9720 | 0.9267 | 0.8735 | 0.9586 | 0.9488 | 150 | Adequate |


## 7. Ensemble Strategy Comparison (Test Partition)

| Strategy | accuracy | f1_macro | f1_macro_reliable | f1_weighted | roc_auc_macro_ovr | ece |
| --- | --- | --- | --- | --- | --- | --- |
| best_single | 0.9989 | 0.9603 | 0.9972 | 0.9988 | 0.9995 | 0.0003 |
| soft_voting | 0.9959 | 0.9181 | 0.9760 | 0.9958 | 0.9985 | 0.0119 |
| confidence_weighted | 0.9983 | 0.9356 | 0.9876 | 0.9983 | 0.9989 | 0.0111 |
| val_weighted_voting | 0.9970 | 0.9184 | 0.9786 | 0.9970 | 0.9985 | 0.0110 |
| stacking | 0.9988 | 0.9725 | 0.9921 | 0.9987 | 0.9988 | 0.0007 |


### Paired Bootstrap Macro-F1 Significance vs. Best Single (`XGBoost`)


| Strategy_vs_BestSingle | Observed_Diff | Mean_Diff | CI95_Low | CI95_High | Statistically_Significant | N_Bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| soft_voting | -0.0421 | -0.0434 | -0.0717 | -0.0197 | True | 1000 |
| confidence_weighted | -0.0246 | -0.0254 | -0.0421 | -0.0099 | True | 1000 |
| val_weighted_voting | -0.0419 | -0.0433 | -0.0717 | -0.0202 | True | 1000 |
| stacking | 0.0123 | 0.0138 | -0.0040 | 0.0538 | False | 1000 |


### Paired Bootstrap Macro-F1 Comparison: Stacking vs. Other Strategies


| Comparison | Observed_Diff | Mean_Diff | CI95_Low | CI95_High | Statistically_Significant | N_Bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| stacking_vs_best_single | 0.0123 | 0.0138 | -0.0040 | 0.0538 | False | 1000 |
| stacking_vs_soft_voting | 0.0544 | 0.0572 | 0.0259 | 0.1010 | True | 1000 |
| stacking_vs_confidence_weighted | 0.0369 | 0.0392 | 0.0140 | 0.0815 | True | 1000 |
| stacking_vs_val_weighted_voting | 0.0542 | 0.0571 | 0.0264 | 0.1035 | True | 1000 |


- **Empirical Conclusion:** Ensemble fusion 'stacking' (Test Macro-F1=0.9725) is comparable to XGBoost, difference not statistically significant (XGBoost=0.9603; observed delta=+0.0123, paired bootstrap n=1000 mean diff=+0.0138, 95% CI=[-0.0040, +0.0538] includes 0; statistically_significant=False). Standalone XGBoost is kept as the supervised default with the Autoencoder orthogonal.


## 8. Unsupervised Anomaly Detector Baselines & Options (Benign-Only Training)

> **Role Clarification:** Unsupervised detectors (including Autoencoder variants, classical baselines, and leave-attack-out score-level fusion) remain **supplementary zero-day signals**, NOT replacements for supervised multi-class XGBoost.


| Detector | Accuracy | Precision | Recall | F1 | F1_CI95_Low | F1_CI95_High | ROC_AUC | PR_AUC | FPR | Holdout_Class_Recall | Holdout_Class_ROC_AUC | Val_Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoencoder | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 0.0522 | 0.0000 | 0.5816 |  |
| Autoencoder (Deep Reg) | 0.8668 | 0.6872 | 0.5063 | 0.5831 | 0.5704 | 0.5947 | 0.7908 | 0.6293 | 0.0520 | 0.0000 | 0.5371 |  |
| VAE | 0.8693 | 0.6964 | 0.5132 | 0.5910 | 0.5780 | 0.6025 | 0.7755 | 0.6231 | 0.0504 | 0.0000 | 0.5940 |  |
| Isolation Forest | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4253 | 0.4531 | 0.7710 | 0.5005 | 0.0499 | 0.0000 | 0.6101 |  |
| One-Class SVM | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5770 | 0.6020 | 0.7467 | 0.5649 | 0.0504 | 0.0000 | 0.6385 |  |
| LOF | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4516 | 0.4775 | 0.8607 | 0.5391 | 0.0520 | 0.1154 | 0.4188 |  |
| Score-Level Fusion (AE+OCSVM+IF) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 0.0522 | 0.0000 | 0.5816 | YES |


- **Non-Test Validation Selection (Benign Percentile + Leave-Attack-Out Validation):** Best single=`Autoencoder`, Selected overall=`Score-Level Fusion (AE+OCSVM+IF)`, Fusion weights=[`Autoencoder`=1.00, `One-Class SVM`=0.00, `Isolation Forest`=0.00], Fusion threshold=`0.0179`.


### Anomaly Validation Selection Summary (Non-Test Leave-Attack-Out)


| Detector | Type | Weights | Val_LOO_F1 | Val_LOO_ROC_AUC | Val_Selection_Objective |
| --- | --- | --- | --- | --- | --- |
| Autoencoder | single | Autoencoder | 0.1539 | 0.8554 | 0.4549 |
| Autoencoder (Deep Reg) | single | Autoencoder (Deep Reg) | 0.1202 | 0.8017 | 0.3950 |
| VAE | single | VAE | 0.1221 | 0.8145 | 0.4001 |
| Isolation Forest | single | Isolation Forest | 0.0812 | 0.7910 | 0.3347 |
| One-Class SVM | single | One-Class SVM | 0.1209 | 0.7131 | 0.3792 |
| LOF | single | LOF | 0.0969 | 0.7384 | 0.3366 |
| Score-Level Fusion (AE+OCSVM+IF) | fusion | Autoencoder=1.00 | 0.1539 | 0.8554 | 0.4549 |


- **Verdict:** Best binary F1 at 95th-pct threshold: Autoencoder (F1=0.6926 [0.6820, 0.7020], Recall=0.6523, FPR=0.0522); best overall ROC-AUC ranking: Autoencoder (ROC-AUC=0.8676); validation-selected option (leave-attack-out non-test): 'Score-Level Fusion (AE+OCSVM+IF)'. Even the best unsupervised option/fusion still trails supervised attribution by a wide margin and remains strictly a supplementary zero-day signal, not a replacement for XGBoost.


### Percentile Threshold Trade-Off Sweep Across Detectors


| Detector | Percentile | Threshold | Precision | Recall | F1 | FPR | FNR |
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
| Score-Level Fusion (AE+OCSVM+IF) | 90.0000 | 0.0090 | 0.6172 | 0.7187 | 0.6641 | 0.1005 | 0.2813 |
| Score-Level Fusion (AE+OCSVM+IF) | 95.0000 | 0.0179 | 0.7382 | 0.6523 | 0.6926 | 0.0522 | 0.3477 |
| Score-Level Fusion (AE+OCSVM+IF) | 97.0000 | 0.0297 | 0.7977 | 0.5652 | 0.6617 | 0.0323 | 0.4348 |
| Score-Level Fusion (AE+OCSVM+IF) | 99.0000 | 0.0782 | 0.8997 | 0.4472 | 0.5975 | 0.0112 | 0.5528 |


## 9. XGBoost Sample-Size Sensitivity Experiment (`data.sample_size`)

- **Imbalance Strategy:** `none` | **Sizes Tested:** `[200000, 500000]` | **Memory-Limited:** `False`


| Requested_Sample_Size | Rows_After_Dedup | Train_Rows | Test_Rows | Accuracy | F1_Weighted | F1_Macro | F1_Macro_Reliable | CI95_Low | CI95_High | ROC_AUC | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200000 | 191268 | 138190 | 28324 | 0.9989 | 0.9988 | 0.9603 | 0.9972 | 0.9013 | 0.9849 | 0.9995 | completed |
| 500000 | 470048 | 339608 | 69026 | 0.9993 | 0.9993 | 0.9894 | 0.9875 | 0.9837 | 0.9943 | 0.9997 | completed |


- **Sensitivity Verdict:** Scaling stratified sample from 200,000 to 500,000 rows changed XGBoost F1_Macro by +0.0291 (0.9603 [0.9013, 0.9849] -> 0.9894 [0.9837, 0.9943]) and F1_Macro_Reliable by -0.0097 (95% CIs overlap=True); conclusions_change=False.


## 10. XGBoost Baseline vs. Optuna Hyperparameter Optimization

| Variant | val_macro_f1 | val_weighted_f1 | val_macro_recall | val_mlogloss |
| --- | --- | --- | --- | --- |
| Baseline (300 trees, depth=6, lr=0.1) | 0.9504 | 0.9988 | 0.9331 | 0.0042 |
| Optuna Tuned (40 trials) | 0.9561 | 0.9990 | 0.9398 | 0.0040 |


- **Selected Variant:** `baseline` (Baseline CV Macro-F1 = `0.9535 ± 0.0121`, Val Macro-F1 delta=`+0.0056`, Meaningful Improvement=`False`)

- **Decision Basis:** No meaningful improvement: Optuna tuned val Macro-F1 (0.9561) changed by +0.0056 vs baseline (0.9504), within baseline CV std=0.0121; retaining baseline.

- **How CV Std is Computed:** Unified 5-fold StratifiedKFold on the development pool (162,577 rows = train+val, held-out test excluded) with per-fold preprocessing and 'none' imbalance handling (sample ddof=1 std across folds).



## 11. Neural Baselines vs. Recommended Production Model (`XGBoost`)

- **Production Recommendation:** **`XGBoost`** is the recommended production classifier due to its stronger held-out Macro-F1, higher throughput (`>200k` flows/sec batched), and robustness without synthetic oversampling.
- **Role of `MLP` & `1D-CNN`:** `MLP` and `1D-CNN` are included as **neural baselines for comparison**; tabular CICFlowMeter features do not form a physical time series, so Conv1D kernels over adjacent engineered statistics do not surpass gradient-boosted trees.


### Bounded Neural Improvement Trials (`label_smoothing` & `focal_loss`)


| Model | loss_variant | status | val_macro_f1 | val_weighted_f1 | val_macro_recall | val_mlogloss | delta_val_macro_f1 | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | cross_entropy (baseline) | baseline | 0.8134 | 0.9778 | 0.8500 | 0.0498 | 0.0000 | 0.0000 |
| MLP | label_smoothing | reverted | 0.7138 | 0.9787 | 0.7875 | 0.0946 | -0.0996 | 31.3700 |
| MLP | focal_loss | reverted | 0.7618 | 0.9712 | 0.8436 | 0.1075 | -0.0516 | 31.1100 |
| 1D-CNN | cross_entropy (baseline) | baseline | 0.8701 | 0.9933 | 0.9120 | 0.0220 | 0.0000 | 0.0000 |
| 1D-CNN | label_smoothing | reverted | 0.6562 | 0.9696 | 0.7350 | 0.1153 | -0.2138 | 120.9400 |
| 1D-CNN | focal_loss | skipped_time_budget | N/A | N/A | N/A | N/A | N/A | 0.0000 |


- **MLP Decision:** Reverted to baseline 'cross_entropy' for MLP (baseline val Macro-F1=0.8134; no variant exceeded +0.0020 tolerance within bounded neural budget).

- **1D-CNN Decision:** Reverted to baseline 'cross_entropy' for CNN (baseline val Macro-F1=0.8701; no variant exceeded +0.0020 tolerance within bounded neural budget).



## 12. Preprocessing vs. Model Inference Latency

- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** median = `828.35 µs`, P95 = `1344.52 µs` (mean = `906.76 µs`, `50` warm-up calls discarded; `1102.8` flows/sec).
