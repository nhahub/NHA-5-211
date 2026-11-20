# SMART IDS v3 — System Architecture & Data Engineering Pipeline

## 1. End-to-End Hybrid Architecture Diagram

```mermaid
flowchart TB
    subgraph Ingestion["1. Data Ingestion & Provenance"]
        RAW["CICIDS2017 CSVs (data/raw/*.csv)"]
        SYN["Synthetic Demo Fallback (data/synthetic/*.csv)"]
        RAW --> LOADER["CSV Auto-Discovery & Header Stripper"]
        SYN --> LOADER
    end

    subgraph Cleaning["2. Zero-Leakage Data Engineering"]
        LOADER --> STATELESS["Stateless Cleaning: ID Removal, Label Merge, Inf->NaN, Deduplication"]
        STATELESS --> SPLIT["Stratified Split (72.25% Train / 12.75% Val / 15% Test)"]
        SPLIT --> FIT["Fit on Train ONLY: Drop >50% NaN, Median Imputer, 5-Sigma Clipper, StandardScaler"]
        FIT --> TRANSFORM["Transform Train / Val / Test (Val & Test kept naturally imbalanced)"]
        TRANSFORM --> SMOTE["SMOTE on Train Only (dynamic k_neighbors <= 5)"]
    end

    subgraph Ensemble["3. Multi-Modal Detection Ensemble"]
        SMOTE --> MLP["MLP (512-256-128-64, BN, LeakyReLU, Dropout 0.4, L2)"]
        SMOTE --> CNN["1D-CNN (Conv1D 64/128, MaxPool, GAP, Dense 128)"]
        SMOTE --> XGB["XGBoost (Baseline + Optuna TPE Tuning, CUDA/CPU)"]
        TRANSFORM -->|BENIGN Train Only| AE["Autoencoder (256-128-64-32-64-128-256, MSE)"]
        AE --> THRESH["95th Percentile Benign Validation Threshold"]
    end

    subgraph Hybrid["4. Calibration, XAI & Active Mitigation"]
        MLP & CNN & XGB --> CAL["Validation Temperature Scaling & Decision Layer"]
        THRESH --> ANOM["Orthogonal Anomaly Signal (MSE > Threshold)"]
        CAL & ANOM --> SCORING["4-Tier Threat Scoring (Benign Safety Guard)"]
        CAL --> SHAP["SHAP XAI (TreeExplainer + Gradient/KernelExplainer)"]
        SCORING --> RESP["Active Response Engine (IP Gate, Idempotency, Dry-Run iptables DROP)"]
    end

    subgraph SOC["5. Operational Interface"]
        RESP & SHAP & Hybrid --> DASH["Streamlit SOC Dashboard (7 Tabs)"]
    end
```

---

## 2. Leakage-Free Data Engineering Flow

```mermaid
flowchart LR
    A["Raw CSV Rows (82+ cols)"] --> B["Strip Column Whitespace & Normalize Web Attack Labels"]
    B --> C["Exact Row Deduplication & Structural ID Drop (IP, Timestamp, Flow ID, Source Port)"]
    C --> D["Convert +Inf/-Inf -> NaN & Coerce Non-Numeric Strings"]
    D --> E["Stratified Split BEFORE Any Statistical Fit"]
    E -->|Train 72.25%| F["Fit Median Imputer, 5-Sigma Clipper, StandardScaler"]
    E -->|Val 12.75% & Test 15%| G["Transform Only Using Train Parameters"]
    F --> H["Apply SMOTE Strictly on Standardized Train Split"]
    F -->|Filter BENIGN Rows| I["Train Unsupervised Autoencoder"]
    G -->|Benign Val Subset| J["Calibrate 95th Percentile Anomaly Threshold"]
```

> **Chronological `time_block` Alignment:** Before dropping `Timestamp` from model features, `time_block` evaluation aligns real capture `Timestamp` values from `data/TrafficLabelling/` onto `MachineLearningCVE` flows via [`align_real_timestamps`](../src/data/loader.py) (`flow_telemetry_signature`, `2,830,471 / 2,830,743` = `99.99%` matched; Train `115,669` / Val `37,420` / Test `37,535`, `leakage_checks.passed = True`).

---

## 3. Model Architecture Specifications

| Model | Learning Paradigm | Topology / Hyperparameters | Training Target |
|---|---|---|---|
| **MLP** | Supervised Deep Neural Network (Neural Baseline) | `Input(77) -> [Dense(512/256/128/64, no bias, L2=1e-4) -> BN -> LeakyReLU(0.1) -> Dropout(0.4)] -> Softmax` | Per-model selected strategy (`smote` or `class_weight`/`none`) on multi-class flows (13 CICIDS2017 classes) |
| **1D-CNN** | Supervised Convolutional Network (Neural Baseline) | `Input(77,1) -> Conv1D(64,k=3)x2 + MaxPool(2) + Drop(0.3) -> Conv1D(128,k=3)x2 + GAP + Drop(0.4) -> Dense(128) + Drop(0.3) -> Softmax` | Per-model selected strategy on multi-class flows (13 CICIDS2017 classes) |
| **Autoencoder** | Unsupervised Reconstruction | `Encoder(256->128->64->32) -> Decoder(64->128->256->77 linear)`, MSE loss, `Adam(lr=5e-4)` | **BENIGN training flows only** (zero attack labels) |
| **XGBoost** | Gradient-Boosted Decision Trees (Recommended Production) | Baseline (`300` trees, `depth=6`, `lr=0.1`, `subsample=0.8`, `colsample=0.8`, `hist`) vs CV-std-gated 40-trial Optuna TPE | Per-model selected strategy (`none`) on 13 CICIDS2017 classes |
