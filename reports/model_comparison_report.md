# SMART IDS v3 — Model Comparison Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Unified Model Benchmark (Primary Split: `random`)

- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.

- **Unsupervised anomaly detectors (Autoencoder, Isolation Forest, One-Class SVM, LOF):** Trained strictly on benign flows and evaluated under identical binary novelty framing (Benign vs. Anomalous).


| Model | Imbalance_Strategy | Evaluation_Framing | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | F1_Macro_Reliable | F1_Macro_CI95_Low | F1_Macro_CI95_High | ROC_AUC | PR_AUC | Latency_Median_us | Latency_P95_us | Latency_Mean_us | Samples_Per_Sec | Model_Size_MB | Training_Duration_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | smote | 13-class supervised | 0.9766 | 0.9790 | 0.9766 | 0.9772 | 0.7934 | 0.9286 | 0.7544 | 0.8176 | 0.9962 | 0.8487 | 155.0000 | 270.4100 | 173.2300 | 24487.8000 | 2.5340 | 57.8400 |
| 1D-CNN | smote | 13-class supervised | 0.9904 | 0.9925 | 0.9904 | 0.9913 | 0.8447 | 0.9562 | 0.7649 | 0.8819 | 0.9933 | 0.8752 | 661.3000 | 806.4500 | 689.6400 | 17574.6000 | 1.3100 | 859.5500 |
| XGBoost | none | 13-class supervised | 0.9989 | 0.9989 | 0.9989 | 0.9988 | 0.9603 | 0.9972 | 0.9013 | 0.9849 | 0.9995 | 0.9773 | 344.2000 | 485.6100 | 387.6300 | 252775.0000 | 2.9540 | 320.3300 |
| Autoencoder | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6926 | N/A | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 207.7000 | 357.8900 | 223.0700 | 23860.5000 | 1.5480 | 68.8000 |
| Autoencoder (Deep Reg) | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8668 | 0.6872 | 0.5063 | 0.5831 | 0.5831 | N/A | 0.5704 | 0.5947 | 0.7908 | 0.6293 | 254.8000 | 374.6300 | 272.5700 | 20310.7000 | 5.0700 | 53.4600 |
| VAE | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8693 | 0.6964 | 0.5132 | 0.5910 | 0.5910 | N/A | 0.5780 | 0.6025 | 0.7755 | 0.6231 | 250.7500 | 373.4600 | 262.3800 | 24299.1000 | 1.5710 | 33.4600 |
| Isolation Forest | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4394 | N/A | 0.4253 | 0.4531 | 0.7710 | 0.5005 | 7174.4000 | 8003.9700 | 7260.6900 | 75805.4000 | 2.0960 | 0.4800 |
| One-Class SVM | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5898 | N/A | 0.5770 | 0.6020 | 0.7467 | 0.5649 | 140.8500 | 167.0400 | 145.8000 | 16548.0000 | 0.4780 | 1.4900 |
| LOF | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4647 | N/A | 0.4516 | 0.4775 | 0.8607 | 0.5391 | 1313.5500 | 2214.9200 | 1426.5600 | 42331.4000 | 5.6660 | 0.4900 |
| Score-Level Fusion (degenerate: equals Autoencoder) | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6926 | N/A | 0.6820 | 0.7020 | 0.8676 | 0.7290 | N/A | N/A | N/A | N/A | N/A | N/A |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: Bot, Heartbleed, Infiltration, Web Attack.


Validation selected a single-detector fusion; no gain from fusing detectors on this data.


- **XGBoost Macro-F1 (95% Bootstrap CI, n=1000):** `0.9603` (`[0.9013, 0.9849]`, bootstrap std=`0.0193`)

- **1D-CNN Macro-F1 (95% Bootstrap CI, n=1000):** `0.8447` (`[0.7649, 0.8819]`, bootstrap std=`0.0299`)

- **MLP Macro-F1 (95% Bootstrap CI, n=1000):** `0.7934` (`[0.7544, 0.8176]`, bootstrap std=`0.0161`)

- **Autoencoder Binary F1 (95% Bootstrap CI, n=1000):** `0.6926` (`[0.6820, 0.7020]`, bootstrap std=`0.0053`)

- **Autoencoder (Deep Reg) Binary F1 (95% Bootstrap CI, n=1000):** `0.5831` (`[0.5704, 0.5947]`, bootstrap std=`0.0063`)

- **VAE Binary F1 (95% Bootstrap CI, n=1000):** `0.5910` (`[0.5780, 0.6025]`, bootstrap std=`0.0062`)

- **Isolation Forest Binary F1 (95% Bootstrap CI, n=1000):** `0.4394` (`[0.4253, 0.4531]`, bootstrap std=`0.0069`)

- **One-Class SVM Binary F1 (95% Bootstrap CI, n=1000):** `0.5898` (`[0.5770, 0.6020]`, bootstrap std=`0.0062`)

- **LOF Binary F1 (95% Bootstrap CI, n=1000):** `0.4647` (`[0.4516, 0.4775]`, bootstrap std=`0.0066`)

- **Score-Level Fusion (degenerate: equals Autoencoder) Binary F1 (95% Bootstrap CI, n=1000):** `0.6926` (`[0.6820, 0.7020]`, bootstrap std=`0.0053`)



## 2. Chronological Drift Evaluation (`time_block`: Per-Class Earliest ~60% Train / Next ~20% Val / Last ~20% Test)

- **Partitions (`split_strategy='time_block'`):** Train pre-SMOTE=`115,669`, Validation=`37,420`, Test=`37,535` (Ordering=`Timestamp`, `leakage_checks.passed=True`).

- **Timestamp Alignment:** status=`aligned`, strategy=`flow_telemetry_signature`, keys=`['Source_Session_Day', 'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow IAT Mean', 'Label', 'Occurrence_Index']`, matched=`2,830,471` / `2,830,743` (unmatched=`272`).

- **Low-Support / Tiny-Class Chronological Allocations:** `Heartbleed`: Low-support chronological split (n=11): 7 earliest train / 2 middle val / 2 latest test.

- **Protocol Interpretation:** Closed-set temporal drift split where every class is trained on its earliest ~60% of flows and evaluated on its latest ~20%, isolating concept drift over capture time without introducing unseen attack labels.


| Model | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | F1_Macro_Reliable | F1_Macro_CI95_Low | F1_Macro_CI95_High | ROC_AUC_OVR | PR_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.9922 | 0.9914 | 0.9922 | 0.9916 | 0.7571 | 0.9116 | 0.7087 | 0.8316 | 0.9918 | 0.7987 |
| 1D-CNN | 0.9250 | 0.9287 | 0.9250 | 0.9187 | 0.3073 | 0.3995 | 0.3019 | 0.3359 | 0.9405 | 0.4047 |
| MLP | 0.9511 | 0.9380 | 0.9511 | 0.9411 | 0.2770 | 0.3601 | 0.2756 | 0.3011 | 0.9526 | 0.4331 |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: Bot, Heartbleed, Infiltration.


- **XGBoost (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.7571` (`[0.7087, 0.8316]`, bootstrap std=`0.0280`)

- **1D-CNN (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.3073` (`[0.3019, 0.3359]`, bootstrap std=`0.0095`)

- **MLP (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.2770` (`[0.2756, 0.3011]`, bootstrap std=`0.0081`)

- **Autoencoder (`time_block`, benign-only binary):** Accuracy=`0.8420`, F1=`0.4819` (`[0.4705, 0.4926]`, n=1000), ROC-AUC=`0.8151`, FPR=`0.0623`.



### XGBoost Per-Class Metrics (`time_block` Test Partition)


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9914 | 0.9998 | 0.9956 | 30763 | Adequate |
| Bot | 0.0000 | 0.0000 | 0.0000 | 28 | Statistically Unreliable |
| DDoS | 1.0000 | 0.9934 | 0.9967 | 1809 | Adequate |
| DoS GoldenEye | 0.9603 | 0.8288 | 0.8897 | 146 | Adequate |
| DoS Hulk | 0.9978 | 0.9367 | 0.9663 | 2463 | Adequate |
| DoS Slowhttptest | 0.9012 | 0.9865 | 0.9419 | 74 | Adequate |
| DoS slowloris | 0.9655 | 0.7000 | 0.8116 | 80 | Adequate |
| FTP-Patator | 1.0000 | 1.0000 | 1.0000 | 96 | Adequate |
| Heartbleed | 0.0000 | 0.0000 | 0.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.5714 | 0.7273 | 7 | Statistically Unreliable |
| PortScan | 1.0000 | 0.9949 | 0.9975 | 1978 | Adequate |
| SSH-Patator | 1.0000 | 0.9245 | 0.9608 | 53 | Adequate |
| Web Attack | 0.8333 | 0.4167 | 0.5556 | 36 | Adequate |


### Time-Block Error Analysis (Failing Classes with Recall < 0.85)


#### Confusion-Matrix Breakdown for Failing Classes


| Class | Test_Support | Recall | F1_Score | Predicted_As_Breakdown |
| --- | --- | --- | --- | --- |
| Bot | 28 | 0.0000 | 0.0000 | BENIGN: 28 (100.0%) |
| Heartbleed | 2 | 0.0000 | 0.0000 | BENIGN: 2 (100.0%) |
| Web Attack | 36 | 0.4167 | 0.5556 | BENIGN: 17 (47.2%), Web Attack: 15 (41.7%), DoS GoldenEye: 4 (11.1%) |
| Infiltration | 7 | 0.5714 | 0.7273 | Infiltration: 4 (57.1%), BENIGN: 3 (42.9%) |
| DoS slowloris | 80 | 0.7000 | 0.8116 | DoS slowloris: 56 (70.0%), BENIGN: 16 (20.0%), DoS Slowhttptest: 8 (10.0%) |
| DoS GoldenEye | 146 | 0.8288 | 0.8897 | DoS GoldenEye: 121 (82.9%), BENIGN: 25 (17.1%) |


#### Top Shifted Features Train-vs-Test per Failing Class (KS Statistic)


| Class | Train_Count | Test_Count | Rank | Feature | KS_Statistic | KS_PValue | Std_Mean_Shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bot | 81 | 28 | 1 | Flow Duration | 1.0000 | 0.0000 | 0.0290 |
| Bot | 81 | 28 | 2 | Total Length of Fwd Packets | 1.0000 | 0.0000 | -0.8294 |
| Bot | 81 | 28 | 3 | Fwd Packet Length Max | 1.0000 | 0.0000 | -0.4557 |
| Bot | 81 | 28 | 4 | Fwd Packet Length Mean | 1.0000 | 0.0000 | -0.6118 |
| Bot | 81 | 28 | 5 | Flow Bytes/s | 1.0000 | 0.0000 | -0.1159 |
| Heartbleed | 7 | 2 | 1 | Fwd IAT Std | 1.0000 | 0.0556 | -0.0005 |
| Heartbleed | 7 | 2 | 2 | Init_Win_bytes_backward | 1.0000 | 0.0556 | 0.0142 |
| Heartbleed | 7 | 2 | 3 | Total Backward Packets | 0.8571 | 0.1667 | -11.8875 |
| Heartbleed | 7 | 2 | 4 | Flow IAT Max | 0.8571 | 0.1667 | -0.0232 |
| Heartbleed | 7 | 2 | 5 | Fwd IAT Mean | 0.8571 | 0.1667 | -0.0001 |
| Web Attack | 105 | 36 | 1 | Flow IAT Mean | 0.4873 | 0.0000 | -0.1820 |
| Web Attack | 105 | 36 | 2 | Bwd Packets/s | 0.4873 | 0.0000 | 0.0912 |
| Web Attack | 105 | 36 | 3 | Fwd IAT Min | 0.4786 | 0.0000 | -0.0001 |
| Web Attack | 105 | 36 | 4 | Flow Packets/s | 0.4778 | 0.0000 | 0.0210 |
| Web Attack | 105 | 36 | 5 | Flow IAT Std | 0.4778 | 0.0000 | -0.1127 |
| Infiltration | 22 | 7 | 1 | Flow IAT Std | 0.8182 | 0.0004 | 1.4126 |
| Infiltration | 22 | 7 | 2 | Flow IAT Max | 0.8182 | 0.0004 | 1.4214 |
| Infiltration | 22 | 7 | 3 | Fwd IAT Max | 0.7208 | 0.0039 | 1.2480 |
| Infiltration | 22 | 7 | 4 | Idle Max | 0.7208 | 0.0039 | 1.2595 |
| Infiltration | 22 | 7 | 5 | Flow Packets/s | 0.6818 | 0.0062 | -0.0294 |
| DoS slowloris | 242 | 80 | 1 | Idle Mean | 0.2995 | 0.0000 | -0.6642 |
| DoS slowloris | 242 | 80 | 2 | Bwd IAT Max | 0.2962 | 0.0000 | -1.2590 |
| DoS slowloris | 242 | 80 | 3 | Bwd IAT Min | 0.2949 | 0.0000 | -1.2758 |
| DoS slowloris | 242 | 80 | 4 | Idle Max | 0.2830 | 0.0001 | -0.8034 |
| DoS slowloris | 242 | 80 | 5 | Bwd IAT Total | 0.2824 | 0.0001 | -0.8715 |
| DoS GoldenEye | 436 | 146 | 1 | Fwd Packet Length Mean | 0.3100 | 0.0000 | 0.4979 |
| DoS GoldenEye | 436 | 146 | 2 | Avg Fwd Segment Size | 0.3100 | 0.0000 | 0.4979 |
| DoS GoldenEye | 436 | 146 | 3 | Fwd Packet Length Std | 0.2965 | 0.0000 | 0.5055 |
| DoS GoldenEye | 436 | 146 | 4 | Fwd Packet Length Max | 0.2850 | 0.0000 | 0.3137 |
| DoS GoldenEye | 436 | 146 | 5 | Total Length of Fwd Packets | 0.2758 | 0.0000 | 0.0215 |


### Time-Block Recall-vs-FPR Mitigation Experiment


#### time_block Validation Partition Screening


> Evaluated strictly on the validation partition (options screened without touching test data).


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('none') | Argmax | argmax P(c) | 0.0619 | 0.9952 | 0.7405 | 0.8767 | 0.8534 | Bot=0.500, Heartbleed=0.000, Web Attack=0.943, Infiltration=1.000, DoS slowloris=1.000, DoS GoldenEye=1.000 |
| XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0803 | 0.9972 | 0.9417 | 0.8471 | 0.8754 | Bot=0.679, Heartbleed=1.000, Web Attack=0.971, Infiltration=1.000, DoS slowloris=1.000, DoS GoldenEye=1.000 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0591 | 0.1320 | 0.9969 | 0.9072 | 0.7717 | N/A | Bot=0.500, Heartbleed=1.000, Web Attack=0.943, Infiltration=1.000, DoS slowloris=1.000, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.2100 | 0.1000 | 0.9967 | 0.7643 | 0.8166 | N/A | Bot=0.643, Heartbleed=0.000, Web Attack=0.943, Infiltration=1.000, DoS slowloris=1.000, DoS GoldenEye=1.000 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1127 | 0.0842 | 0.9968 | 0.9072 | 0.8407 | N/A | Bot=0.500, Heartbleed=1.000, Web Attack=0.943, Infiltration=1.000, DoS slowloris=1.000, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.6658 | 0.0496 | 0.9929 | 0.6098 | 0.8972 | N/A | Bot=0.357, Heartbleed=0.000, Web Attack=0.914, Infiltration=0.429, DoS slowloris=1.000, DoS GoldenEye=0.959 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1214 | 0.0814 | 0.9968 | 0.9072 | 0.8452 | N/A | Bot=0.500, Heartbleed=1.000, Web Attack=0.943, Infiltration=1.000, DoS slowloris=1.000, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.6695 | 0.0290 | 0.9929 | 0.6098 | 0.9359 | N/A | Bot=0.357, Heartbleed=0.000, Web Attack=0.914, Infiltration=0.429, DoS slowloris=1.000, DoS GoldenEye=0.959 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.2067 | 0.0718 | 0.9968 | 0.9072 | 0.8609 | N/A | Bot=0.500, Heartbleed=1.000, Web Attack=0.943, Infiltration=1.000, DoS slowloris=1.000, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.6697 | 0.0086 | 0.9929 | 0.6098 | 0.9777 | N/A | Bot=0.357, Heartbleed=0.000, Web Attack=0.914, Infiltration=0.429, DoS slowloris=1.000, DoS GoldenEye=0.959 |


#### time_block Held-Out Test Partition (Descriptive Only)


> **Evaluated on the held-out test partition for description only; options were NOT selected on test data.**


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('none') | Argmax | argmax P(c) | 0.0002 | 0.9607 | 0.4547 | 0.9795 | 0.7571 | Bot=0.000, Heartbleed=0.000, Web Attack=0.528, Infiltration=0.571, DoS slowloris=0.800, DoS GoldenEye=0.829 |
| XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0144 | 0.9796 | 0.7968 | 0.9581 | 0.8646 | Bot=0.000, Heartbleed=1.000, Web Attack=0.806, Infiltration=1.000, DoS slowloris=0.975, DoS GoldenEye=1.000 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0591 | 0.0974 | 0.9681 | 0.7401 | 0.8032 | N/A | Bot=0.000, Heartbleed=1.000, Web Attack=0.528, Infiltration=1.000, DoS slowloris=0.912, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.2100 | 0.0006 | 0.9833 | 0.5795 | 0.9903 | N/A | Bot=0.000, Heartbleed=0.000, Web Attack=0.722, Infiltration=0.857, DoS slowloris=0.925, DoS GoldenEye=0.973 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1127 | 0.0625 | 0.9680 | 0.7401 | 0.8597 | N/A | Bot=0.000, Heartbleed=1.000, Web Attack=0.528, Infiltration=1.000, DoS slowloris=0.912, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.6658 | 0.0001 | 0.9396 | 0.3527 | 0.9686 | N/A | Bot=0.000, Heartbleed=0.000, Web Attack=0.444, Infiltration=0.143, DoS slowloris=0.700, DoS GoldenEye=0.829 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1214 | 0.0601 | 0.9680 | 0.7401 | 0.8638 | N/A | Bot=0.000, Heartbleed=1.000, Web Attack=0.528, Infiltration=1.000, DoS slowloris=0.912, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.6695 | 0.0001 | 0.9395 | 0.3527 | 0.9686 | N/A | Bot=0.000, Heartbleed=0.000, Web Attack=0.444, Infiltration=0.143, DoS slowloris=0.700, DoS GoldenEye=0.829 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.2067 | 0.0468 | 0.9677 | 0.7359 | 0.8876 | N/A | Bot=0.000, Heartbleed=1.000, Web Attack=0.528, Infiltration=1.000, DoS slowloris=0.887, DoS GoldenEye=1.000 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.6697 | 0.0001 | 0.9395 | 0.3527 | 0.9686 | N/A | Bot=0.000, Heartbleed=0.000, Web Attack=0.444, Infiltration=0.143, DoS slowloris=0.700, DoS GoldenEye=0.829 |


- **Validation vs. Test Ranking Agreement:** Validation ranking agrees with test (Spearman rho=0.9807; top bounded-FPR option is 'XGBoost ('class_weight') (Argmax)' on both splits).

- **Mitigation Assessment (`time_block`):** Descriptively on the held-out test partition (not used for selection), XGBoost ('class_weight') (Argmax) shows focus attack recall 0.7968 at Benign FPR=0.0144 vs baseline 0.4547 (Benign FPR=0.0002).


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


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, Heartbleed, Infiltration, SSH-Patator.


### Open-Set Cross-Day Recall-vs-FPR Mitigation Experiment (Thu–Fri Unseen Attacks)


#### by_day Validation Partition Screening


> Evaluated strictly on the validation partition (options screened without touching test data).


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('none') | Argmax | argmax P(c) | 0.0005 | 0.0283 | 0.0170 | 0.0517 | 0.1661 | Infiltration=0.000, Web Attack=0.034 |
| XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0008 | 0.3396 | 0.2046 | 0.4675 | 0.1247 | Infiltration=0.000, Web Attack=0.409 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0616 | 0.1003 | 0.2075 | 0.5228 | 0.0265 | N/A | Infiltration=1.000, Web Attack=0.045 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.0001 | 0.1000 | 0.8585 | 0.5834 | 0.1054 | N/A | Infiltration=0.167, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1169 | 0.0503 | 0.1792 | 0.4394 | 0.0425 | N/A | Infiltration=0.833, Web Attack=0.045 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.0002 | 0.0501 | 0.8396 | 0.5278 | 0.1852 | N/A | Infiltration=0.056, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1252 | 0.0303 | 0.1792 | 0.4394 | 0.0645 | N/A | Infiltration=0.833, Web Attack=0.045 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.0005 | 0.0301 | 0.8302 | 0.5000 | 0.2691 | N/A | Infiltration=0.000, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.1888 | 0.0103 | 0.1415 | 0.3283 | 0.1075 | N/A | Infiltration=0.611, Web Attack=0.045 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.0020 | 0.0101 | 0.8302 | 0.5000 | 0.5057 | N/A | Infiltration=0.000, Web Attack=1.000 |


#### by_day Held-Out Test Partition (Descriptive Only)


> **Evaluated on the held-out test partition for description only; options were NOT selected on test data.**


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


- **Validation vs. Test Ranking Agreement:** Validation ranking agrees with test (Spearman rho=0.9756; top bounded-FPR option is 'Lower P(Benign) Gate (1 - P_Benign > tau) (P99)' on both splits).

- **Mitigation Assessment (`by_day`):** Descriptively on the held-out test partition (not used for selection), Lower P(Benign) Gate (1 - P_Benign > tau) (P99) shows focus attack recall 0.3412 at Benign FPR=0.0103 vs baseline 0.1349 (Benign FPR=0.0001).


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


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: Bot, Heartbleed, Infiltration.


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


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: Bot, Heartbleed, Infiltration, Web Attack.


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
| Score-Level Fusion (degenerate: equals Autoencoder) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 0.0522 | 0.0000 | 0.5816 | YES |


Validation selected a single-detector fusion; no gain from fusing detectors on this data.


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


- **Verdict:** Best binary F1 at 95th-pct threshold: Autoencoder (F1=0.6926 [0.6820, 0.7020], Recall=0.6523, FPR=0.0522); best overall ROC-AUC ranking: Autoencoder (ROC-AUC=0.8676); validation-selected option (leave-attack-out non-test): 'Score-Level Fusion (degenerate: equals Autoencoder)'. Validation selected a single-detector fusion; no gain from fusing detectors on this data.


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
| Score-Level Fusion (degenerate: equals Autoencoder) | 90.0000 | 0.0090 | 0.6172 | 0.7187 | 0.6641 | 0.1005 | 0.2813 |
| Score-Level Fusion (degenerate: equals Autoencoder) | 95.0000 | 0.0179 | 0.7382 | 0.6523 | 0.6926 | 0.0522 | 0.3477 |
| Score-Level Fusion (degenerate: equals Autoencoder) | 97.0000 | 0.0297 | 0.7977 | 0.5652 | 0.6617 | 0.0323 | 0.4348 |
| Score-Level Fusion (degenerate: equals Autoencoder) | 99.0000 | 0.0782 | 0.8997 | 0.4472 | 0.5975 | 0.0112 | 0.5528 |


## 9. XGBoost Sample-Size Sensitivity Experiment (`data.sample_size`)

- **Imbalance Strategy:** `none` | **Sizes Tested:** `[200000, 500000]` | **Memory-Limited:** `False`


| Requested_Sample_Size | Rows_After_Dedup | Train_Rows | Test_Rows | Accuracy | F1_Weighted | F1_Macro | F1_Macro_Reliable | CI95_Low | CI95_High | ROC_AUC | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200000 | 191268 | 138190 | 28324 | 0.9989 | 0.9988 | 0.9603 | 0.9972 | 0.9013 | 0.9849 | 0.9995 | completed |
| 500000 | 470048 | 339608 | 69026 | 0.9993 | 0.9993 | 0.9894 | 0.9875 | 0.9837 | 0.9943 | 0.9997 | completed |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: N=200,000: Bot, Heartbleed, Infiltration, Web Attack; N=500,000: Heartbleed, Infiltration.


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
| 1D-CNN | focal_loss | skipped_time_budget | N/A | N/A | N/A | N/A | N/A | N/A |


- **MLP Decision:** Reverted to baseline 'cross_entropy' for MLP (baseline val Macro-F1=0.8134; no variant exceeded +0.0020 tolerance within bounded neural budget).

- **1D-CNN Decision:** Reverted to baseline 'cross_entropy' for CNN (baseline val Macro-F1=0.8701; no variant exceeded +0.0020 tolerance within bounded neural budget).



## 12. Preprocessing vs. Model Inference Latency

- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** median = `814.00 µs`, P95 = `1057.99 µs` (mean = `855.75 µs`, `50` warm-up calls discarded; `1168.6` flows/sec).
