# SMART IDS v3 — Model Comparison Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Unified Model Benchmark (Primary Split: `random`)

- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.

- **Unsupervised anomaly detectors (Autoencoder, Isolation Forest, One-Class SVM, LOF):** Trained strictly on benign flows and evaluated under identical binary novelty framing (Benign vs. Anomalous).


| Model | Imbalance_Strategy | Evaluation_Framing | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | F1_Macro_Reliable | F1_Macro_CI95_Low | F1_Macro_CI95_High | ROC_AUC | PR_AUC | Latency_Median_us | Latency_P95_us | Latency_Mean_us | Samples_Per_Sec | Model_Size_MB | Training_Duration_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | smote | 13-class supervised | 0.9766 | 0.9790 | 0.9766 | 0.9772 | 0.7934 | 0.9286 | 0.7544 | 0.8176 | 0.9962 | 0.8487 | 225.8000 | 407.2900 | 246.9700 | 16829.8000 | 2.5340 | 57.8400 |
| 1D-CNN | smote | 13-class supervised | 0.9904 | 0.9925 | 0.9904 | 0.9913 | 0.8447 | 0.9562 | 0.7649 | 0.8819 | 0.9933 | 0.8752 | 1088.3000 | 1347.6300 | 1122.5400 | 10290.2000 | 1.3100 | 859.5500 |
| XGBoost | class_weight | 13-class supervised | 0.9986 | 0.9987 | 0.9986 | 0.9986 | 0.9661 | 0.9931 | 0.9310 | 0.9829 | 0.9995 | 0.9926 | 754.9000 | 1493.0200 | 818.3900 | 155163.3000 | 3.6310 | 10.6600 |
| Autoencoder | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6926 | N/A | 0.6820 | 0.7020 | 0.8676 | 0.7290 | 290.0500 | 481.8000 | 323.1500 | 16666.1000 | 1.5480 | 68.8000 |
| Autoencoder (Deep Reg) | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8668 | 0.6872 | 0.5063 | 0.5831 | 0.5831 | N/A | 0.5704 | 0.5947 | 0.7908 | 0.6293 | 482.1000 | 854.4900 | 518.6000 | 16079.5000 | 5.0700 | 53.4600 |
| VAE | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8693 | 0.6964 | 0.5132 | 0.5910 | 0.5910 | N/A | 0.5780 | 0.6025 | 0.7755 | 0.6231 | 364.1500 | 698.4000 | 408.2800 | 15674.7000 | 1.5710 | 33.4600 |
| Isolation Forest | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8385 | 0.6085 | 0.3438 | 0.4394 | 0.4394 | N/A | 0.4253 | 0.4531 | 0.7710 | 0.5005 | 10112.1000 | 17538.4800 | 11043.3300 | 54771.2000 | 2.0960 | 0.4800 |
| One-Class SVM | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8690 | 0.6960 | 0.5117 | 0.5898 | 0.5898 | N/A | 0.5770 | 0.6020 | 0.7467 | 0.5649 | 271.5500 | 586.6700 | 297.1200 | 16475.7000 | 0.4780 | 1.4900 |
| LOF | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8421 | 0.6178 | 0.3724 | 0.4647 | 0.4647 | N/A | 0.4516 | 0.4775 | 0.8607 | 0.5391 | 1400.7000 | 2170.0200 | 1480.6300 | 35230.9000 | 5.6660 | 0.4900 |
| Score-Level Fusion (degenerate: equals Autoencoder) | benign_only | Binary anomaly (Benign vs Anomalous) | 0.8934 | 0.7382 | 0.6523 | 0.6926 | 0.6926 | N/A | 0.6820 | 0.7020 | 0.8676 | 0.7290 | N/A | N/A | N/A | N/A | N/A | N/A |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: Bot, Heartbleed, Infiltration, Web Attack.


Validation selected a single-detector fusion; no gain from fusing detectors on this data.


- **XGBoost Macro-F1 (95% Bootstrap CI, n=1000):** `0.9661` (`[0.9310, 0.9829]`, bootstrap std=`0.0139`)

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
| XGBoost | 0.9834 | 0.9920 | 0.9834 | 0.9876 | 0.8646 | 0.9240 | 0.8460 | 0.8763 | 0.9927 | 0.8828 |
| 1D-CNN | 0.9250 | 0.9287 | 0.9250 | 0.9187 | 0.3073 | 0.3995 | 0.3019 | 0.3359 | 0.9405 | 0.4047 |
| MLP | 0.9511 | 0.9380 | 0.9511 | 0.9411 | 0.2770 | 0.3601 | 0.2756 | 0.3011 | 0.9526 | 0.4331 |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: Bot, Heartbleed, Infiltration.


- **XGBoost (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.8646` (`[0.8460, 0.8763]`, bootstrap std=`0.0078`)

- **1D-CNN (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.3073` (`[0.3019, 0.3359]`, bootstrap std=`0.0095`)

- **MLP (`time_block`) Macro-F1 (95% Bootstrap CI, n=1000):** `0.2770` (`[0.2756, 0.3011]`, bootstrap std=`0.0081`)

- **Autoencoder (`time_block`, benign-only binary):** Accuracy=`0.8420`, F1=`0.4819` (`[0.4705, 0.4926]`, n=1000), ROC-AUC=`0.8151`, FPR=`0.0623`.



### Time-Aware XGBoost Profile Selection (Validation-Only)


> Selected strictly on chronological `time_block` **validation** Macro-F1 with simplicity tie-breaking (`tie_tolerance=0.0020`, `used_test_data=False`, validation flows=`37,420`, ordering=`Timestamp`).


| Profile | Val_Macro_F1 | Val_Weighted_F1 | Val_Macro_Recall | Val_MLogLoss | Val_Benign_FPR | Selected |
| --- | --- | --- | --- | --- | --- | --- |
| none | 0.8534 | 0.9556 | 0.8633 | 0.0872 | 0.0619 |  |
| class_weight | 0.8754 | 0.9443 | 0.9566 | 0.3263 | 0.0803 | YES |


- **Selected XGBoost Deployment Profile (`time_block` Val):** `class_weight`

- **Selection Justification:** Data-driven selection chose 'class_weight' with highest validation Macro-F1=0.8754 (Weighted-F1=0.9443, Macro-Recall=0.9566, mlogloss=0.3263), beating runner-up 'none' by +0.0220 Macro-F1 (> tolerance=0.0020; Unweighted ('none') Macro-F1=0.8534 (Weighted-F1=0.9556, mlogloss=0.0872)). Validation Benign FPR (time_block): 'class_weight'=0.0803 vs 'none'=0.0619. Random-split validation context: 'none' Macro-F1=0.9518 vs 'class_weight' Macro-F1=0.9446.


### XGBoost Per-Class Metrics (`time_block` Test Partition)


| Class | Precision | Recall | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9955 | 0.9856 | 0.9905 | 30763 | Adequate |
| Bot | 0.0000 | 0.0000 | 0.0000 | 28 | Statistically Unreliable |
| DDoS | 0.9967 | 0.9967 | 0.9967 | 1809 | Adequate |
| DoS GoldenEye | 0.8848 | 1.0000 | 0.9389 | 146 | Adequate |
| DoS Hulk | 0.9716 | 0.9578 | 0.9646 | 2463 | Adequate |
| DoS Slowhttptest | 0.6518 | 0.9865 | 0.7849 | 74 | Adequate |
| DoS slowloris | 0.9559 | 0.8125 | 0.8784 | 80 | Adequate |
| FTP-Patator | 0.9796 | 1.0000 | 0.9897 | 96 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 1.0000 | 1.0000 | 7 | Statistically Unreliable |
| PortScan | 0.9995 | 0.9949 | 0.9972 | 1978 | Adequate |
| SSH-Patator | 1.0000 | 0.9245 | 0.9608 | 53 | Adequate |
| Web Attack | 0.8276 | 0.6667 | 0.7385 | 36 | Adequate |


### Time-Block Error Analysis (Active Deployment Profile Failing Classes with Recall < 0.85)


> The error breakdown below reflects the active XGBoost deployment profile only; the mitigation focus set is the union of classes failing under either candidate profile (`none` / `class_weight`).


#### Confusion-Matrix Breakdown for Failing Classes


| Class | Test_Support | Recall | F1_Score | Predicted_As_Breakdown |
| --- | --- | --- | --- | --- |
| Bot | 28 | 0.0000 | 0.0000 | BENIGN: 28 (100.0%) |
| Web Attack | 36 | 0.6667 | 0.7385 | Web Attack: 24 (66.7%), BENIGN: 7 (19.4%), DoS GoldenEye: 5 (13.9%) |
| DoS slowloris | 80 | 0.8125 | 0.8784 | DoS slowloris: 65 (81.2%), DoS Slowhttptest: 8 (10.0%), DoS GoldenEye: 4 (5.0%), BENIGN: 2 (2.5%) |


#### Top Shifted Features Train-vs-Test per Failing Class (KS Statistic)


| Class | Train_Count | Test_Count | Rank | Feature | KS_Statistic | KS_PValue | Std_Mean_Shift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bot | 81 | 28 | 1 | Flow Duration | 1.0000 | 0.0000 | 0.0290 |
| Bot | 81 | 28 | 2 | Total Length of Fwd Packets | 1.0000 | 0.0000 | -0.8294 |
| Bot | 81 | 28 | 3 | Fwd Packet Length Max | 1.0000 | 0.0000 | -0.4557 |
| Bot | 81 | 28 | 4 | Fwd Packet Length Mean | 1.0000 | 0.0000 | -0.6118 |
| Bot | 81 | 28 | 5 | Flow Bytes/s | 1.0000 | 0.0000 | -0.1159 |
| Web Attack | 105 | 36 | 1 | Flow IAT Mean | 0.4873 | 0.0000 | -0.1820 |
| Web Attack | 105 | 36 | 2 | Bwd Packets/s | 0.4873 | 0.0000 | 0.0912 |
| Web Attack | 105 | 36 | 3 | Fwd IAT Min | 0.4786 | 0.0000 | -0.0001 |
| Web Attack | 105 | 36 | 4 | Flow Packets/s | 0.4778 | 0.0000 | 0.0210 |
| Web Attack | 105 | 36 | 5 | Flow IAT Std | 0.4778 | 0.0000 | -0.1127 |
| DoS slowloris | 242 | 80 | 1 | Idle Mean | 0.2995 | 0.0000 | -0.6642 |
| DoS slowloris | 242 | 80 | 2 | Bwd IAT Max | 0.2962 | 0.0000 | -1.2590 |
| DoS slowloris | 242 | 80 | 3 | Bwd IAT Min | 0.2949 | 0.0000 | -1.2758 |
| DoS slowloris | 242 | 80 | 4 | Idle Max | 0.2830 | 0.0001 | -0.8034 |
| DoS slowloris | 242 | 80 | 5 | Bwd IAT Total | 0.2824 | 0.0001 | -0.8715 |


### Time-Block Recall-vs-FPR Mitigation Experiment


#### time_block Validation Partition Screening


> Evaluated strictly on the validation partition (options screened without touching test data).


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0803 | 0.9972 | 0.9417 | 0.8471 | 0.8754 | Bot=0.679, DoS GoldenEye=1.000, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.971 |
| XGBoost ('none') | Argmax | argmax P(c) | 0.0619 | 0.9952 | 0.7405 | 0.8767 | 0.8534 | Bot=0.500, DoS GoldenEye=1.000, DoS slowloris=1.000, Heartbleed=0.000, Infiltration=1.000, Web Attack=0.943 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0591 | 0.1468 | 0.9981 | 0.9417 | 0.7531 | N/A | Bot=0.679, DoS GoldenEye=1.000, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.971 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.2534 | 0.0972 | 0.9999 | 1.0000 | 0.8223 | N/A | Bot=1.000, DoS GoldenEye=1.000, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1127 | 0.1258 | 0.9980 | 0.9417 | 0.7806 | N/A | Bot=0.679, DoS GoldenEye=1.000, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.971 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.9931 | 0.0483 | 0.9834 | 0.9229 | 0.8949 | N/A | Bot=0.643, DoS GoldenEye=0.952, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.943 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1214 | 0.1069 | 0.9980 | 0.9417 | 0.8070 | N/A | Bot=0.679, DoS GoldenEye=1.000, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.971 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.9935 | 0.0296 | 0.9833 | 0.9229 | 0.9299 | N/A | Bot=0.643, DoS GoldenEye=0.952, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.943 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.2067 | 0.0888 | 0.9980 | 0.9417 | 0.8342 | N/A | Bot=0.679, DoS GoldenEye=1.000, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.971 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.9955 | 0.0090 | 0.9790 | 0.9229 | 0.9697 | N/A | Bot=0.643, DoS GoldenEye=0.952, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.943 |


#### time_block Held-Out Test Partition (Descriptive Only)


> **Evaluated on the held-out test partition for description only; options were NOT selected on test data.**


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0144 | 0.9796 | 0.7968 | 0.9581 | 0.8646 | Bot=0.000, DoS GoldenEye=1.000, DoS slowloris=0.975, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.806 |
| XGBoost ('none') | Argmax | argmax P(c) | 0.0002 | 0.9607 | 0.4547 | 0.9795 | 0.7571 | Bot=0.000, DoS GoldenEye=0.829, DoS slowloris=0.800, Heartbleed=0.000, Infiltration=0.571, Web Attack=0.528 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0591 | 0.1056 | 0.9808 | 0.7968 | 0.7972 | N/A | Bot=0.000, DoS GoldenEye=1.000, DoS slowloris=0.975, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.806 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.2534 | 0.0401 | 0.9994 | 1.0000 | 0.9162 | N/A | Bot=1.000, DoS GoldenEye=1.000, DoS slowloris=1.000, Heartbleed=1.000, Infiltration=1.000, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1127 | 0.0712 | 0.9808 | 0.7968 | 0.8513 | N/A | Bot=0.000, DoS GoldenEye=1.000, DoS slowloris=0.975, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.806 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.9931 | 0.0002 | 0.8407 | 0.6120 | 0.9131 | N/A | Bot=0.000, DoS GoldenEye=1.000, DoS slowloris=0.700, Heartbleed=0.500, Infiltration=1.000, Web Attack=0.472 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1214 | 0.0688 | 0.9808 | 0.7968 | 0.8553 | N/A | Bot=0.000, DoS GoldenEye=1.000, DoS slowloris=0.975, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.806 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.9935 | 0.0002 | 0.8404 | 0.6120 | 0.9129 | N/A | Bot=0.000, DoS GoldenEye=1.000, DoS slowloris=0.700, Heartbleed=0.500, Infiltration=1.000, Web Attack=0.472 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.2067 | 0.0573 | 0.9808 | 0.7968 | 0.8753 | N/A | Bot=0.000, DoS GoldenEye=1.000, DoS slowloris=0.975, Heartbleed=1.000, Infiltration=1.000, Web Attack=0.806 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.9955 | 0.0001 | 0.8342 | 0.5837 | 0.9093 | N/A | Bot=0.000, DoS GoldenEye=0.973, DoS slowloris=0.700, Heartbleed=0.500, Infiltration=0.857, Web Attack=0.472 |


- **Validation vs. Test Ranking Agreement:** Validation ranking partially agrees with test (Spearman rho=0.9895; validation top bounded-FPR option is 'Lower P(Benign) Gate (1 - P_Benign > tau) (P90)' vs descriptive test top 'Lower P(Benign) Gate (1 - P_Benign > tau) (P95)').

- **Mitigation Assessment (`time_block`):** Descriptively on held-out test, raising recall above baseline (0.7968 at Benign FPR=0.0144; peak option 'Lower P(Benign) Gate (1 - P_Benign > tau)' [P90] reaches 1.0000 at Benign FPR=0.0401) incurs an elevated false-positive rate cost (> +2% FPR); the active production profile is therefore kept unchanged and trade-offs are reported transparently.


## 3. Open-Set Cross-Day Protocol (`by_day`: Mon–Wed Train / Thu Val / Thu–Fri Unseen Attacks)

- **Partitions (`split_strategy='by_day'`):** Train pre-SMOTE=`112,124`, Validation (Thursday half)=`15,537`, Test (Thursday remainder + Friday)=`63,001` (`leakage_checks.passed=True`).

- **Thursday–Friday Unseen Attack Classes:** `Bot`, `DDoS`, `Infiltration`, `PortScan`, `Web Attack`.


> **Open-Set Interpretation:** Open-set cross-day protocol (by_day): by construction in CICIDS2017, all Thursday and Friday attack families (Bot, DDoS, Infiltration, PortScan, Web Attack) have zero training instances in Monday-Wednesday data. Closed-set multi-class Macro-F1 is therefore near zero by design rather than ordinary generalization degradation. The real empirical finding is that supervised XGBoost misses ~72% of unseen attack flows (binary recall=0.2829, precision=0.9970), which is why benign-only anomaly detectors are kept as a supplementary zero-day signal.


### Open-Set Cross-Day Binary Detection (`by_day`: Benign vs. Attack)


| Model | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (1 - P_Benign) | 0.7745 | 0.9970 | 0.2829 | 0.4407 | 0.7155 | 0.6683 | 0.0004 | 0.7171 |
| XGBoost (Binary) | 0.7737 | 0.9968 | 0.2806 | 0.4379 | 0.7427 | 0.7075 | 0.0004 | 0.7194 |
| MLP (1 - P_Benign) | 0.7747 | 0.9956 | 0.2841 | 0.4420 | 0.8059 | 0.7237 | 0.0006 | 0.7159 |
| 1D-CNN (1 - P_Benign) | 0.7328 | 0.9081 | 0.1663 | 0.2812 | 0.9304 | 0.8240 | 0.0077 | 0.8337 |
| Autoencoder | 0.7555 | 0.7417 | 0.3400 | 0.4663 | 0.8760 | 0.6982 | 0.0542 | 0.6600 |


### Closed-Set Multi-Class Reference Under Open-Set Split (`by_day`)


| Model | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro_All | F1_Macro_Reliable | F1_Macro_Present_Test | ROC_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.6856 | 0.5162 | 0.6856 | 0.5890 | 0.0716 | 0.1718 | 0.1431 | 0.5358 |
| MLP | 0.6855 | 0.5164 | 0.6855 | 0.5891 | 0.0716 | 0.1718 | 0.1431 | 0.5678 |
| 1D-CNN | 0.6816 | 0.4940 | 0.6816 | 0.5728 | 0.0696 | 0.1670 | 0.1392 | 0.6586 |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, Heartbleed, Infiltration, SSH-Patator.


### Open-Set Cross-Day Recall-vs-FPR Mitigation Experiment (Thu–Fri Unseen Attacks)


#### by_day Validation Partition Screening


> Evaluated strictly on the validation partition (options screened without touching test data).


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0005 | 0.2736 | 0.1648 | 0.4056 | 0.1425 | Infiltration=0.000, Web Attack=0.330 |
| XGBoost ('none') | Argmax | argmax P(c) | 0.0004 | 0.0472 | 0.0284 | 0.0855 | 0.1424 | Infiltration=0.000, Web Attack=0.057 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0611 | 0.1004 | 0.4340 | 0.6149 | 0.0541 | N/A | Infiltration=0.889, Web Attack=0.341 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.0004 | 0.1000 | 0.8585 | 0.5834 | 0.1046 | N/A | Infiltration=0.167, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1170 | 0.0505 | 0.4245 | 0.5871 | 0.0968 | N/A | Infiltration=0.833, Web Attack=0.341 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.0010 | 0.0500 | 0.8396 | 0.5278 | 0.1841 | N/A | Infiltration=0.056, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1310 | 0.0305 | 0.4245 | 0.5871 | 0.1449 | N/A | Infiltration=0.833, Web Attack=0.341 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.0020 | 0.0300 | 0.8302 | 0.5000 | 0.2679 | N/A | Infiltration=0.000, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.1679 | 0.0105 | 0.3491 | 0.3649 | 0.2426 | N/A | Infiltration=0.389, Web Attack=0.341 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.0080 | 0.0100 | 0.8302 | 0.5000 | 0.5043 | N/A | Infiltration=0.000, Web Attack=1.000 |


#### by_day Held-Out Test Partition (Descriptive Only)


> **Evaluated on the held-out test partition for description only; options were NOT selected on test data.**


| Option | Benign_Val_Pct | Threshold | Benign_FPR | Overall_Attack_Recall | Focus_Mean_Recall | Binary_F1 | Multiclass_Macro_F1 | Focus_Class_Recalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline XGBoost ('class_weight') | Argmax | argmax P(c) | 0.0003 | 0.2828 | 0.2137 | 0.4407 | 0.0716 | Bot=0.000, DDoS=0.611, Infiltration=0.000, PortScan=0.003, Web Attack=0.455 |
| XGBoost ('none') | Argmax | argmax P(c) | 0.0001 | 0.2881 | 0.1485 | 0.4472 | 0.0782 | Bot=0.000, DDoS=0.628, Infiltration=0.000, PortScan=0.001, Web Attack=0.114 |
| Hybrid Override (XGBoost OR AE) | P90 | AE>0.0611 | 0.0866 | 0.4384 | 0.4555 | 0.5387 | N/A | Bot=0.015, DDoS=0.631, Infiltration=0.889, PortScan=0.277, Web Attack=0.466 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P90 | 1-P(B)>0.0004 | 0.1200 | 0.4825 | 0.4981 | 0.5531 | N/A | Bot=0.007, DDoS=1.000, Infiltration=0.444, PortScan=0.039, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P95 | AE>0.1170 | 0.0545 | 0.3653 | 0.4167 | 0.4922 | N/A | Bot=0.015, DDoS=0.627, Infiltration=0.833, PortScan=0.143, Web Attack=0.466 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P95 | 1-P(B)>0.0010 | 0.0635 | 0.4805 | 0.4293 | 0.5935 | N/A | Bot=0.000, DDoS=0.999, Infiltration=0.111, PortScan=0.037, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P97 | AE>0.1310 | 0.0471 | 0.3497 | 0.4108 | 0.4815 | N/A | Bot=0.015, DDoS=0.627, Infiltration=0.833, PortScan=0.114, Web Attack=0.466 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P97 | 1-P(B)>0.0020 | 0.0351 | 0.4479 | 0.4041 | 0.5876 | N/A | Bot=0.000, DDoS=0.930, Infiltration=0.056, PortScan=0.034, Web Attack=1.000 |
| Hybrid Override (XGBoost OR AE) | P99 | AE>0.1679 | 0.0366 | 0.3086 | 0.3843 | 0.4445 | N/A | Bot=0.015, DDoS=0.626, Infiltration=0.778, PortScan=0.037, Web Attack=0.466 |
| Lower P(Benign) Gate (1 - P_Benign > tau) | P99 | 1-P(B)>0.0080 | 0.0119 | 0.3681 | 0.3585 | 0.5281 | N/A | Bot=0.000, DDoS=0.771, Infiltration=0.000, PortScan=0.021, Web Attack=1.000 |


- **Validation vs. Test Ranking Agreement:** Validation ranking agrees with test (Spearman rho=0.8415; top bounded-FPR option is 'Lower P(Benign) Gate (1 - P_Benign > tau) (P99)' on both splits).

- **Mitigation Assessment (`by_day`):** Descriptively on the held-out test partition (not used for selection), Lower P(Benign) Gate (1 - P_Benign > tau) (P99) shows focus attack recall 0.3585 at Benign FPR=0.0119 vs baseline 0.2137 (Benign FPR=0.0003).


## 4. Shortcut Feature Ablation (With vs. Without `Destination Port`)

| Variant | Features | Test_Rows | Val_Macro_F1 | Test_Accuracy | Test_F1_Weighted | Test_F1_Macro | Test_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (With Destination Port) | 77 | 28126 | 0.9751 | 0.9980 | 0.9982 | 0.9465 | 0.9996 |
| XGBoost (Without Destination Port) | 76 | 28126 | 0.9349 | 0.9975 | 0.9977 | 0.9325 | 0.9992 |


- **Apples-to-Apples Row Parity Verified:** Identical post-deduplication train/val/test rows (`Test_Rows=28,126`, `sha256=6c72618b9e1f...`) evaluated after full 77-feature leakage verification.

- **Delta Test Macro-F1 (without - with):** `-0.0140` | **Delta Val Macro-F1:** `-0.0402` | **Delta Test Weighted-F1:** `-0.0005`.

- **Top-5 XGBoost SHAP Features After Removing `Destination Port`:** `Init_Win_bytes_backward` (0.7080), `min_seg_size_forward` (0.6315), `Init_Win_bytes_forward` (0.5243), `Bwd Packets/s` (0.3662), `Bwd Packet Length Mean` (0.3573).



## 5. Per-Class Precision, Recall, F1 & Support (Test Partition)

> **Statistical Reliability Note:** Classes marked `Statistically Unreliable` have low held-out test support (`< 30` flows); one or two misclassified flows swing per-class F1 sharply and drive Macro-F1 variance.


### XGBoost Per-Class Metrics


| Class | Precision | Recall | Recall_CI95_Low | Recall_CI95_High | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9998 | 0.9987 | 0.9981 | 0.9991 | 0.9993 | 23112 | Adequate |
| Bot | 0.7500 | 1.0000 | 0.8454 | 1.0000 | 0.8571 | 21 | Statistically Unreliable |
| DDoS | 0.9985 | 0.9993 | 0.9958 | 0.9999 | 0.9989 | 1357 | Adequate |
| DoS GoldenEye | 0.9909 | 1.0000 | 0.9660 | 1.0000 | 0.9954 | 109 | Adequate |
| DoS Hulk | 0.9958 | 1.0000 | 0.9980 | 1.0000 | 0.9979 | 1883 | Adequate |
| DoS Slowhttptest | 1.0000 | 1.0000 | 0.9358 | 1.0000 | 1.0000 | 56 | Adequate |
| DoS slowloris | 0.9677 | 0.9836 | 0.9128 | 0.9971 | 0.9756 | 61 | Adequate |
| FTP-Patator | 0.9722 | 1.0000 | 0.9480 | 1.0000 | 0.9859 | 70 | Adequate |
| Heartbleed | 1.0000 | 1.0000 | 0.3424 | 1.0000 | 1.0000 | 2 | Statistically Unreliable |
| Infiltration | 1.0000 | 0.8000 | 0.3755 | 0.9638 | 0.8889 | 5 | Statistically Unreliable |
| PortScan | 0.9924 | 1.0000 | 0.9976 | 1.0000 | 0.9962 | 1576 | Adequate |
| SSH-Patator | 1.0000 | 0.9783 | 0.8866 | 0.9962 | 0.9890 | 46 | Adequate |
| Web Attack | 0.9545 | 0.8077 | 0.6212 | 0.9149 | 0.8750 | 26 | Statistically Unreliable |


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

- **Variant Evaluated:** `baseline` | **Development Pool Rows (Held-Out Test Excluded):** `166,926` | **Imbalance Strategy:** `class_weight`
- **Fold Mean Macro-F1 (Mean +/- Std):** `0.9599 ± 0.0134`
- **Pooled OOF Macro-F1 (All Classes vs. Reliable `support >= 30`):** `0.9613` vs. `0.9625` (total OOF evaluated=`159,908`)
- **Weighted-F1 (Mean +/- Std):** `0.9986 ± 0.0002` | **Accuracy:** `0.9985 ± 0.0002`


| Fold | Train_Rows | Val_Rows | Accuracy | F1_Weighted | F1_Macro | F1_Macro_Reliable | ROC_AUC_OVR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 130696 | 31926 | 0.9983 | 0.9984 | 0.9552 | 0.9941 | 0.9995 |
| 2 | 130647 | 32020 | 0.9982 | 0.9983 | 0.9390 | 0.9840 | 0.9993 |
| 3 | 130632 | 31941 | 0.9986 | 0.9986 | 0.9663 | 0.9893 | 0.9992 |
| 4 | 130656 | 31958 | 0.9987 | 0.9988 | 0.9653 | 0.9887 | 0.9989 |
| 5 | 130655 | 32063 | 0.9987 | 0.9987 | 0.9736 | 0.9923 | 0.9996 |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: Bot, Heartbleed, Infiltration, Web Attack.


### Pooled Out-of-Fold (OOF) Per-Class Evaluation (5-Fold CV Dev Pool, Wilson 95% Recall CI)


> Each non-collision development-pool sample is scored out-of-fold exactly once, multiplying rare-class effective evaluation counts ~5x relative to the single 15% test split.


| Class | Precision | Recall | Recall_CI95_Low | Recall_CI95_High | F1_Score | Support | Reliability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BENIGN | 0.9999 | 0.9985 | 0.9983 | 0.9987 | 0.9992 | 130678 | Adequate |
| Bot | 0.5803 | 0.9655 | 0.9147 | 0.9865 | 0.7249 | 116 | Adequate |
| DDoS | 0.9986 | 0.9993 | 0.9985 | 0.9997 | 0.9990 | 7688 | Adequate |
| DoS GoldenEye | 0.9840 | 0.9968 | 0.9882 | 0.9991 | 0.9903 | 616 | Adequate |
| DoS Hulk | 0.9965 | 0.9994 | 0.9988 | 0.9997 | 0.9980 | 10542 | Adequate |
| DoS Slowhttptest | 0.9692 | 0.9844 | 0.9640 | 0.9933 | 0.9767 | 320 | Adequate |
| DoS slowloris | 0.9852 | 0.9823 | 0.9619 | 0.9919 | 0.9838 | 339 | Adequate |
| FTP-Patator | 0.9948 | 0.9974 | 0.9855 | 0.9995 | 0.9961 | 387 | Adequate |
| Heartbleed | 0.9000 | 1.0000 | 0.7009 | 1.0000 | 0.9474 | 9 | Statistically Unreliable |
| Infiltration | 0.9655 | 0.9032 | 0.7510 | 0.9665 | 0.9333 | 31 | Adequate |
| PortScan | 0.9938 | 0.9990 | 0.9981 | 0.9995 | 0.9964 | 8802 | Adequate |
| SSH-Patator | 0.9828 | 0.9870 | 0.9625 | 0.9956 | 0.9849 | 231 | Adequate |
| Web Attack | 0.9542 | 0.9799 | 0.9425 | 0.9931 | 0.9669 | 149 | Adequate |


## 7. Ensemble Strategy Comparison (Test Partition)

| Strategy | accuracy | f1_macro | f1_macro_reliable | f1_weighted | roc_auc_macro_ovr | ece |
| --- | --- | --- | --- | --- | --- | --- |
| best_single | 0.9986 | 0.9661 | 0.9931 | 0.9986 | 0.9995 | 0.0005 |
| soft_voting | 0.9959 | 0.9193 | 0.9748 | 0.9958 | 0.9989 | 0.0124 |
| confidence_weighted | 0.9985 | 0.9536 | 0.9852 | 0.9985 | 0.9990 | 0.0115 |
| val_weighted_voting | 0.9971 | 0.9225 | 0.9774 | 0.9970 | 0.9989 | 0.0113 |
| stacking | 0.9986 | 0.9660 | 0.9921 | 0.9986 | 0.9987 | 0.0008 |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: Bot, Heartbleed, Infiltration, Web Attack.


### Paired Bootstrap Macro-F1 Significance vs. Best Single (`XGBoost`)


| Strategy_vs_BestSingle | Observed_Diff | Mean_Diff | CI95_Low | CI95_High | Statistically_Significant | N_Bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| soft_voting | -0.0468 | -0.0492 | -0.0915 | -0.0207 | True | 1000 |
| confidence_weighted | -0.0125 | -0.0138 | -0.0537 | 0.0063 | False | 1000 |
| val_weighted_voting | -0.0436 | -0.0460 | -0.0870 | -0.0175 | True | 1000 |
| stacking | -0.0001 | -0.0001 | -0.0140 | 0.0146 | False | 1000 |


### Paired Bootstrap Macro-F1 Comparison: Stacking vs. Other Strategies


| Comparison | Observed_Diff | Mean_Diff | CI95_Low | CI95_High | Statistically_Significant | N_Bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| stacking_vs_best_single | -0.0001 | -0.0001 | -0.0140 | 0.0146 | False | 1000 |
| stacking_vs_soft_voting | 0.0467 | 0.0491 | 0.0172 | 0.0931 | True | 1000 |
| stacking_vs_confidence_weighted | 0.0124 | 0.0138 | -0.0100 | 0.0565 | False | 1000 |
| stacking_vs_val_weighted_voting | 0.0435 | 0.0459 | 0.0138 | 0.0906 | True | 1000 |


- **Empirical Conclusion:** Ensemble fusion 'stacking' (Test Macro-F1=0.9660) is comparable to XGBoost, difference not statistically significant (XGBoost=0.9661; observed delta=-0.0001, paired bootstrap n=1000 mean diff=-0.0001, 95% CI=[-0.0140, +0.0146] includes 0; statistically_significant=False). Standalone XGBoost is kept as the supervised default with the Autoencoder orthogonal.


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

- **Imbalance Strategy:** `class_weight` | **Sizes Tested:** `[200000, 500000]` | **Memory-Limited:** `False`


| Requested_Sample_Size | Rows_After_Dedup | Train_Rows | Test_Rows | Accuracy | F1_Weighted | F1_Macro | F1_Macro_Reliable | CI95_Low | CI95_High | ROC_AUC | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200000 | 196384 | 138714 | 28126 | 0.9980 | 0.9982 | 0.9465 | 0.9854 | 0.9084 | 0.9673 | 0.9996 | completed |
| 500000 | 488531 | 341704 | 68466 | 0.9986 | 0.9987 | 0.9736 | 0.9688 | 0.9651 | 0.9800 | 0.9995 | completed |


> *Note:* F1_Macro_Reliable excludes classes with support below 30 on the evaluated partition and is always shown next to the all-class F1_Macro; use F1_Macro (with its CI) as the headline number. Excluded classes: N=200,000: Bot, Heartbleed, Infiltration, Web Attack; N=500,000: Heartbleed, Infiltration.


- **Sensitivity Verdict:** Scaling stratified sample from 200,000 to 500,000 rows changed XGBoost F1_Macro by +0.0271 (0.9465 [0.9084, 0.9673] -> 0.9736 [0.9651, 0.9800]) and F1_Macro_Reliable by -0.0166 (95% CIs overlap=True); conclusions_change=False.


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

- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** median = `1336.50 µs`, P95 = `2423.61 µs` (mean = `1430.88 µs`, `50` warm-up calls discarded; `698.9` flows/sec).
