# SMART IDS v3 — Model Comparison Report

> **Mode Notice — Genuine CICIDS2017 Benchmark Run:** All metrics below were measured on held-out CICIDS2017 evaluation data.

## 1. Unified Four-Model Benchmark

- **Supervised models (MLP, 1D-CNN, XGBoost):** Evaluated on multi-class attack attribution using weighted and macro metrics.

- **Unsupervised Autoencoder:** Evaluated separately under binary novelty framing (Benign vs. Anomalous).


| Model | Evaluation_Framing | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro | ROC_AUC | PR_AUC | Latency_Mean_us | Latency_Median_us | Latency_P95_us | Samples_Per_Sec | Model_Size_MB | Training_Duration_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 13-class supervised | 0.9633 | 0.9785 | 0.9633 | 0.9692 | 0.8076 | 0.9965 | 0.8919 | 5054.2300 | 5001.1500 | 5684.6500 | 23584.7000 | 2.5340 | 361.8100 |
| 1D-CNN | 13-class supervised | 0.9849 | 0.9945 | 0.9849 | 0.9892 | 0.8281 | 0.9979 | 0.9089 | 10128.1900 | 10166.8500 | 11056.8500 | 16945.0000 | 1.3100 | 2604.5700 |
| XGBoost | 13-class supervised | 0.9989 | 0.9989 | 0.9989 | 0.9989 | 0.9704 | 0.9995 | 0.9908 | 516.8200 | 501.1500 | 621.4900 | 263650.3000 | 2.7270 | 149.3800 |
| Autoencoder | Binary anomaly (Benign vs Anomalous) | 0.8688 | 0.6863 | 0.5132 | 0.5873 | 0.5873 | 0.8393 | 0.6642 | 6335.7800 | 6304.5000 | 6938.9500 | 24713.7000 | 1.5480 | 36.5400 |


## 2. Ensemble Strategy Comparison (Test Partition)

| Strategy | accuracy | f1_macro | f1_weighted | roc_auc_macro_ovr | ece |
| --- | --- | --- | --- | --- | --- |
| best_single | 0.9989 | 0.9704 | 0.9989 | 0.9995 | 0.0004 |
| soft_voting | 0.9891 | 0.8554 | 0.9926 | 0.9992 | 0.0128 |
| confidence_weighted | 0.9946 | 0.8892 | 0.9957 | 0.9993 | 0.0111 |


## 3. XGBoost Baseline vs. Optuna Hyperparameter Optimization

| Variant | val_macro_f1 | val_weighted_f1 | val_macro_recall | val_mlogloss |
| --- | --- | --- | --- | --- |
| Baseline (300 trees, depth=6, lr=0.1) | 0.9458 | 0.9987 | 0.9518 | 0.0052 |
| Optuna Tuned (12 trials) | 0.9465 | 0.9986 | 0.9518 | 0.0048 |


- **Selected Variant:** `tuned`


## 4. Architectural Notes on 1D-CNN vs. MLP & XGBoost

Tabular CICFlowMeter features do not form a physical time series; 1D convolutions scan adjacent engineered statistics (packet sizes, IATs, flags). Measured validation and test F1/latency above quantify whether 1D-CNN yields measurable benefit over the dense MLP and XGBoost.


## 5. Preprocessing vs. Model Inference Latency

- **Isolated Single-Flow Preprocessing (Imputer + 5-Sigma Clipper + StandardScaler):** mean = `867.52 µs`, median = `820.70 µs`, P95 = `1052.05 µs` (`1152.7` flows/sec).
