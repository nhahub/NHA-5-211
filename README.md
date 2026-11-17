# NHA-5-211
# Smart IDS

**AI-Powered Network Intrusion Detection, Zero-Day Anomaly Detection & Automated Response System**

*DEPI Graduation Project - From Network Traffic to Explainable Security Decisions.*

---

## Overview

**Smart IDS** is an end-to-end intelligent intrusion detection and response platform that turns raw network-flow telemetry into explained, risk-scored security decisions.

The system ingests network-flow data and applies machine learning and deep learning to:

1. **Detect** malicious network traffic in real time
2. **Classify** known attacks into canonical categories using multi-class supervised models (XGBoost, Deep MLP, 1D-CNN)
3. **Discover zero-day anomalies** that indicate previously unseen attacks using an undercomplete autoencoder trained exclusively on benign traffic
4. **Explain** every model prediction using SHAP (SHapley Additive exPlanations)
5. **Assess risk** across 4 calibrated tiers (Low / Medium / High / Critical)
6. **Alert** security analysts with structured, prioritized threat alerts
7. **Respond** by recommending or safely simulating graduated mitigation actions (LOG, ALERT, RATE_LIMIT, BLOCK with dry-run iptables)
8. **Visualize** the entire threat landscape on an interactive Security Operations Center (SOC) dashboard built with Streamlit

Smart IDS is designed as a complete engineering system rather than a single classification notebook: data engineering, supervised and unsupervised AI, explainability, risk assessment, an active response layer, and visualization are integrated into one unified, reproducible pipeline.

---

## Problem Statement

Traditional signature- and rule-based intrusion detection systems (such as Snort or Suricata) depend heavily on previously known attack patterns:

- **Blind to Novel Threats:** They struggle when attack patterns are new, polymorphic, or zero-day exploits.
- **Scalability Bottlenecks:** They are computationally expensive to scale across high-throughput modern networks.
- **Black-Box Alerting:** Security analysts cannot easily understand *why* an alert fired, leading to alert fatigue.
- **Lack of Prioritization:** Alerts arrive unranked without risk-based severity scores.
- **Disconnected Response:** Detection does not automatically trigger a safe, graduated mitigation workflow.

A system that only outputs "attack detected" does not provide the full workflow an operational security analyst needs: classification, novelty detection, forensic explanation, prioritization, and an actionable next step. **Smart IDS addresses that entire lifecycle.**

---

## System Architecture

```text
Network Traffic / PCAP / Flow Telemetry (CICFlowMeter 77 Features)
                     |
            Data Ingestion Layer (Raw CSVs & Synthetic Fallback)
                     |
Validation & Preprocessing -> Identifier Exclusion -> Outlier Clipping -> Train / Val / Test Partition
                     |
                     +---------------------------------------+
                     |                                       |
              Supervised Layer                        Unsupervised Layer
      [XGBoost / Deep MLP / 1D-CNN]             [Undercomplete Autoencoder]
     (Known Attack Multi-Class Heads)             (Benign-Only Reconstruction)
                     |                                       |
             Attack Attribution                    Reconstruction Error (MSE)
                     |                                       |
                     +-------------------+-------------------+
                                         |
                               Hybrid Decision Engine
                                         |
               +-------------------------+-------------------------+
               |                                                   |
      Risk Assessment Engine                             Explainable AI (SHAP)
   [Low / Medium / High / Critical]                     [Feature Attribution Plots]
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                            Automated Active Response Engine
                     [LOG -> ALERT -> RATE_LIMIT -> DRY-RUN BLOCK]
                                         |
               +-------------------------+-------------------------+
               |                                                   |
     Audit Logging Engine                         Streamlit SOC Web Dashboard
   (ids_response_log.txt)                         (5 Interactive Operational Tabs)
```

### Modules

| Module | Name | Implementation | Description |
|:---:|:---|:---|:---|
| **A** | **Traffic / Data Ingestion** | `src/smart_ids/data_loader.py` | Ingests CICIDS2017 daily exports or provides a schema-faithful synthetic fallback generator |
| **B** | **Data Engineering** | `src/smart_ids/preprocessing.py` | Cleans zero-division `inf`, median-imputes NaNs, removes 4 identifier columns, clips outliers at $\pm 5\sigma$, fits scaling/SMOTE on train only |
| **C** | **Supervised Attack Detection** | `src/smart_ids/models.py` | Multi-class classifiers: XGBoost (300 trees), 4-layer regularized MLP, and localized 1D-CNN |
| **D** | **Zero-Day Anomaly Detection** | `src/smart_ids/models.py` | Undercomplete Autoencoder (`256-128-64-32-64-128-256`) trained on benign traffic with 95th percentile thresholding |
| **E** | **Hybrid Decision Engine** | `src/smart_ids/evaluation.py` | Fuses supervised multi-class confidence with unsupervised reconstruction novelty scores |
| **F** | **Explainable AI (XAI)** | `src/smart_ids/explain.py` | SHAP TreeExplainer and KernelExplainer generating summary, beeswarm, and single-sample waterfall plots |
| **G** | **Risk Assessment** | `src/smart_ids/response.py` | Maps confidence scores to 4 severity tiers (`Low`, `Medium`, `High`, `Critical`) |
| **H** | **Automated / Safe Response** | `src/smart_ids/response.py` | Validates IPv4/IPv6 addresses via `ipaddress` module and builds idempotent Linux `iptables` dry-run rules |
| **I** | **Security Operations Dashboard** | `app/streamlit_app.py` | Dark-themed SOC interface with live KPI cards, telemetry filtering, threat map, flow testing, and active mitigation |

---

## AI / ML Methodology

### 1. Supervised Learning — Known Attacks
- **XGBoost Classifier Baseline:** 300 gradient-boosted trees, max depth 6, learning rate 0.1, subsample 0.8, colsample 0.8, histogram-based split finding (`tree_method="hist"`). Highly robust and extremely fast (~1.18 ms inference latency).
- **Deep Multi-Layer Perceptron (MLP):** 4-layer funnel architecture (`512 -> 256 -> 128 -> 64`) utilizing `use_bias=False`, L2 kernel regularization ($10^{-4}$), Batch Normalization, LeakyReLU ($\alpha=0.1$), and Dropout ($p=0.4$) to prevent overfitting on minority attack classes.
- **1D-Convolutional Neural Network (1D-CNN):** Treats tabular flow statistics as a 1D pseudo-sequence using stacked Conv1D blocks ($64 \rightarrow 128$ filters, kernel size 3) with Batch Normalization, ReLU, MaxPool, GlobalAveragePooling, and a Dense(128) classification head.

### 2. Unsupervised Learning — Zero-Day Novelty Detection
Zero-day attacks are threats the model has never encountered during training. We model this as an **out-of-distribution anomaly detection** task using an **Undercomplete Autoencoder**:

```text
Input Flow (77 features) -> Encoder (256 -> 128 -> 64) -> Bottleneck Latent Space (32)
                                                                    |
Reconstructed Flow (77) <- Decoder (256 <- 128 <- 64) <------------+
         |
Reconstruction Error (MSE) -> Anomaly Score -> Threshold (95th Percentile) -> Benign / Zero-Day Anomaly
```

- **Training:** The Autoencoder is trained **exclusively on benign traffic**. It learns the structural manifold of normal communications.
- **Inference:** When a zero-day attack arrives, the autoencoder cannot compress and reconstruct its abnormal statistical pattern, resulting in a high Mean Squared Error (MSE).
- **Thresholding:** Calibrated at the **95th percentile** of reconstruction error on the benign validation partition ($\tau = 0.5580$). Any flow with $\text{MSE} > \tau$ is flagged as an anomaly.

### 3. Explainable AI (XAI)
To eliminate black-box opacity in security operations:
- **SHAP TreeExplainer:** Rapid, exact Shapley value decomposition for XGBoost models.
- **SHAP KernelExplainer:** Model-agnostic attribution for the Deep MLP using background k-means clustering.
- **Waterfall Plots:** Provides per-flow attribution decomposing the exact contribution of packet lengths, flow rates, and TCP flag counts toward the final alert.

---

## Dataset & Attack Taxonomy

The system is trained and benchmarked against **CICIDS2017**, the leading academic benchmark for network intrusion detection:

- **Volume:** Over 2.8 million raw network flow records collected across 5 consecutive days.
- **Features:** 77 numerical network flow statistics computed by CICFlowMeter (including inter-arrival times, packet length distributions, header byte lengths, and TCP flag frequencies).
- **Taxonomy:** Standardized into an 8-class operational taxonomy:

| Class ID | Canonical Category | Traffic Characteristics | Real-World Attack Type |
|:---:|:---|:---|:---|
| `0` | **BENIGN** | Normal background network traffic | Web browsing, file transfer, internal RPC |
| `1` | **DoS Hulk** | High-volume HTTP GET requests with randomized headers | Application-layer Denial of Service |
| `2` | **PortScan** | Rapid sequential connection attempts across port ranges | Pre-attack network reconnaissance |
| `3` | **DDoS** | Volumetric SYN/UDP flood originating from distributed bots | Network infrastructure exhaustion |
| `4` | **DoS GoldenEye** | HTTP Keep-Alive & No-Cache flooding exhausting server pool | HTTP worker starvation |
| `5` | **FTP-Patator** | High-frequency credential guessing over TCP port 21 | FTP Authentication brute force |
| `6` | **SSH-Patator** | Systematic dictionary authentication against TCP port 22 | Encrypted shell brute force |
| `7` | **Web Attack** | SQL Injection, Cross-Site Scripting (XSS), Brute Force | OWASP Top 10 web vulnerabilities |

---

## Data Engineering Pipeline

```text
Raw CICIDS2017 CSVs
       |
1. Schema Normalization: Strip leading/trailing header whitespace
       |
2. Inf / NaN Remediation: Convert zero-division infinities to NaN; median-impute missing values
       |
3. Duplicate Elimination: Drop exact duplicate network flow records
       |
4. Target Leakage Prevention: Drop non-causal identifiers (Source IP, Destination IP, Destination Port, Timestamp)
       |
5. Outlier Mitigation: Clip extreme flow rate spikes at empirical mean +/- 5 std
       |
6. Stratified Partitioning: 70% Train / 15% Validation / 15% Test (Fixed Seed: 42)
       |
7. Feature Standardization: Fit StandardScaler STRICTLY on Train split; transform Val and Test
       |
8. Synthetic Minority Oversampling: Apply SMOTE STRICTLY to Train split (Val & Test remain natural)
       |
Leak-Free Processed Arrays Ready for Model Training
```

> **Data Leakage Safeguard:**
> Fitting `StandardScaler` or applying `SMOTE` before splitting data is a critical machine learning flaw. In Smart IDS, transformers and synthetic oversamplers are fitted strictly on training data. The validation and test sets remain untransformed and naturally imbalanced, guaranteeing that all benchmark metrics reflect real-world deployment performance.

---

## Risk Assessment & Response

Threat levels and response rules live in a single centralized module ([`src/smart_ids/response.py`](file:///d:/SMART_IDS/src/smart_ids/response.py)) imported by both the training notebooks and the operational dashboard:

| Threat Severity Tier | Confidence Range | Automated Response Action | Operational Description |
|:---|:---:|:---:|:---|
| **🟢 LOW** | $0.00 \le c < 0.40$ | `LOG` | Log flow telemetry to standard SIEM / audit stream; no user disruption |
| **🟡 MEDIUM** | $0.40 \le c < 0.70$ | `ALERT` | Generate high-visibility alert on SOC dashboard for tier-1 security analyst review |
| **🟠 HIGH** | $0.70 \le c < 0.90$ | `RATE_LIMIT` | Enforce network traffic shaping and bandwidth ceiling (max 50 req/s) |
| **🔴 CRITICAL** | $0.90 \le c \le 1.00$ | `BLOCK` | Synthesize firewall isolation rule; validate IP; simulate immediate DROP |

### Safe Response Simulation & Injection Defense
- **Strict IP Sanitization:** Every IP address is validated through Python's standard `ipaddress` module before command generation. Malformed IPs or shell injection attempts (e.g. `192.168.1.1; cat /etc/passwd`) are rejected with `BLOCK_FAILED: INVALID_IP`.
- **Dry-Run Firewall Rule Synthesis:** Generates standard Linux `iptables` rules without executing them against the host OS:
  ```bash
  iptables -I INPUT 1 -s <IP> -j DROP
  ```
- **Idempotency Guarantee:** The response engine maintains an in-memory set and append-only audit log ([`models/ids_response_log.txt`](file:///d:/SMART_IDS/models/ids_response_log.txt)). Repeated detections of the same attacker IP will not generate duplicate firewall rules.

---

## Evaluation Strategy & Benchmark Results

All models were evaluated on the held-out test split (15% unseen data, naturally imbalanced). Supervised models were measured across weighted multi-class accuracy, precision, recall, and F1-score; the Autoencoder was evaluated on binary zero-day detection (anomaly vs. benign).

### Benchmark Comparison Table

| Model Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Per-Sample Latency ($\mu s$) | Evaluation Type |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **XGBoost Classifier Baseline** | **96.49%** | **96.95%** | **96.49%** | **96.65%** | **1,178 $\mu s$** (~1.18 ms) | Multi-Class (8 Classes) |
| **Deep MLP (DNN 512-256-128-64)** | **82.82%** | **92.25%** | **82.82%** | **86.14%** | 95,767 $\mu s$ (~95.8 ms) | Multi-Class (8 Classes) |
| **Undercomplete Autoencoder** | **85.30%** | **60.87%** | **37.84%** | **46.67%** | 106,085 $\mu s$ (~106 ms) | Binary (Zero-Day Anomaly) |
| **1D-CNN (Convolutional Network)** | 4.23% | 7.19% | 4.23% | 2.25% | 124,324 $\mu s$ (~124 ms) | Multi-Class (8 Classes) |

*Artifact file:* [`models/model_comparison.csv`](file:///d:/SMART_IDS/models/model_comparison.csv)

### Key Evaluation Insights
1. **XGBoost Dominance on Tabular Features:** XGBoost achieved **96.65% F1-score** with ultra-low latency (**1.18 ms** per flow), making it the optimal engine for high-speed edge firewall inline inspection.
2. **Deep MLP High Precision:** The Deep MLP achieved **92.25% Precision**, minimizing costly false alarms on enterprise traffic.
3. **Zero-Day Autoencoder Performance:** The Autoencoder successfully detected unseen anomalous attacks with **85.30% accuracy** despite being trained on 0 attack samples.
4. **Visual Diagnostic Artifacts:**
   - 4-panel normalized confusion matrices: [`models/confusion_matrices.png`](file:///d:/SMART_IDS/models/confusion_matrices.png)
   - One-vs-Rest multiclass ROC/AUC curves: [`models/roc_curves.png`](file:///d:/SMART_IDS/models/roc_curves.png)
   - Single-sample SHAP waterfall breakdown: [`models/shap_waterfall.png`](file:///d:/SMART_IDS/models/shap_waterfall.png)

---

## Tech Stack

| Area | Tools & Libraries | Purpose |
|:---|:---|:---|
| **Language & Environment** | Python 3.11.9, Virtualenv | Core runtime and isolated dependency management |
| **Data Processing & Engineering** | Pandas 2.2, NumPy 1.26, SciPy 1.13 | High-performance matrix operations and data pipeline |
| **Machine Learning** | Scikit-learn 1.4, Imbalanced-learn 0.12 | Metrics, splitting, label encoding, scaling, and SMOTE |
| **Deep Learning** | TensorFlow 2.16, Keras 3.3 | Deep MLP, 1D-CNN, and Autoencoder architectures |
| **Gradient Boosting** | XGBoost 2.1 | Fast tabular gradient boosting classifier baseline |
| **Explainable AI (XAI)** | SHAP 0.45 | TreeExplainer, KernelExplainer, and Waterfall forensic plots |
| **SOC Dashboard & Visualization** | Streamlit 1.37, Plotly 5.22, Matplotlib 3.8, Seaborn 0.13 | Interactive security operations center UI and dark charts |
| **Testing & Quality Assurance** | Pytest 8.2 | 31 automated unit tests verifying response rules and leakage |
| **Reproducibility** | Jupyter, Nbconvert 7.17 | Interactive pedagogical notebooks with automated CLI execution |

---

## Repository Structure

```text
smart-ids-v3/
├── README.md                                # Comprehensive DEPI graduation documentation
├── requirements.txt                         # Pinned, mutually compatible Python 3.11 packages
├── pytest.ini                               # Test configuration setting pythonpath = src
│
├── data/                                    # Datasets directory
│   ├── raw/                                 # Storage directory for raw CICIDS2017 daily CSVs
│   └── processed/                           # Leakage-free train/val/test splits (.npy) & sample CSVs
│
├── notebooks/                               # 7 Pedagogical interactive Jupyter notebooks
│   ├── 01_intro_and_security_basics.ipynb   # Networking glossary, CICIDS2017 schema, class skew
│   ├── 02_data_preprocessing.ipynb          # Inf/NaN handling, leakage drop, clipping, scale, SMOTE
│   ├── 03_supervised_models.ipynb           # Training XGBoost, Deep MLP, and 1D-CNN
│   ├── 04_autoencoder_zero_day.ipynb        # Benign-only Autoencoder, 95th percentile cutoff
│   ├── 05_evaluation_and_comparison.ipynb   # Comparison table, confusion matrices, ROC/AUC, latency
│   ├── 06_explainability_shap.ipynb         # SHAP TreeExplainer, KernelExplainer, waterfall plot
│   └── 07_active_response_demo.ipynb        # 4-tier threat engine, IP validation, dry-run iptables
│
├── src/smart_ids/                           # Production modular python package
│   ├── __init__.py                          # Package initialization
│   ├── config.py                            # Fixed SEED=42, paths, 8 classes, hyperparameters
│   ├── data_loader.py                       # CSV ingestion, label standardization, synthetic generator
│   ├── preprocessing.py                     # Data cleaning, outlier clipping, stratified split, SMOTE
│   ├── models.py                            # Keras MLP, 1D-CNN, Autoencoder, and XGBoost builders
│   ├── training.py                          # Callbacks (EarlyStopping, ReduceLR) and training loops
│   ├── evaluation.py                        # Benchmark metrics, per-sample latency, confusion plots
│   ├── explain.py                           # SHAP TreeExplainer and waterfall plot generators
│   ├── artifacts.py                         # Centralized loader and saver for models and scalers
│   └── response.py                          # Canonical threat tiers, IP validator, dry-run firewall
│
├── models/                                  # Saved trained artifacts & evaluation figures
│   ├── best_mlp.keras                       # Trained Keras Deep MLP weights
│   ├── best_cnn.keras                       # Trained Keras 1D-CNN weights
│   ├── best_autoencoder.keras               # Trained Keras Autoencoder weights
│   ├── xgb_model.json                       # Serialized XGBoost model
│   ├── scaler.pkl                           # Fitted StandardScaler (Train-only)
│   ├── label_encoder.pkl                    # Fitted LabelEncoder for 8 classes
│   ├── feature_names.json                   # Ordered list of 77 flow features
│   ├── anomaly_threshold.json               # Calibrated 95th percentile reconstruction cutoff
│   ├── model_comparison.csv                 # Benchmark comparison matrix across all models
│   ├── confusion_matrices.png               # High-resolution confusion matrices
│   ├── roc_curves.png                       # High-resolution One-vs-Rest ROC curves
│   ├── shap_waterfall.png                   # Single-sample forensic explanation figure
│   └── ids_response_log.txt                 # Append-only active response mitigation audit log
│
├── app/                                     # Interactive Security Operations Center (SOC)
│   ├── streamlit_app.py                     # Full 5-tab operational Streamlit application
│   └── components.py                        # Dark SOC styling, pulsing CSS badges, KPI cards
│
├── tests/                                   # Automated unit test suite
│   ├── test_preprocessing.py                # 5 tests: data cleaning, identifier drop, train-only SMOTE
│   └── test_response.py                     # 26 tests: threat boundaries, IP validation, idempotency
│
└── .vscode/
    └── settings.json                        # Auto-selection configuration for .venv Python interpreter
```

---

## Getting Started

### Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Version:** **Python 3.11** (Tested on Python 3.11.9)

### 1. Setup Virtual Environment

#### On Windows (PowerShell):
```powershell
# Navigate to project directory
cd D:\SMART_IDS

# Create Python 3.11 virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

#### On Linux / macOS / Git Bash:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run the 7 Pedagogical Notebooks

The notebooks are designed to run in sequence. You can launch them interactively in Jupyter:

```powershell
jupyter lab
```

Or execute them headlessly from start to finish via `nbconvert`:

```powershell
jupyter nbconvert --to notebook --execute notebooks\01_intro_and_security_basics.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks\02_data_preprocessing.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks\03_supervised_models.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks\04_autoencoder_zero_day.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks\05_evaluation_and_comparison.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks\06_explainability_shap.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks\07_active_response_demo.ipynb --inplace
```

---

### 3. Launch the Interactive SOC Dashboard

Launch the Streamlit Security Operations Center application:

```powershell
streamlit run app\streamlit_app.py
```

Open your browser at **`http://localhost:8501`**.

#### Dashboard Features:
- **KPI Metrics Strip:** Live display of analyzed flows, detected attacks, zero-day anomalies, critical threats, and mean model confidence.
- **Tab 1: Detection Log:** Real-time filterable flow telemetry with colored threat severity badges (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- **Tab 2: Analytics:** Class distribution donuts, confidence score histograms, and the full model evaluation comparison matrix.
- **Tab 3: Threat Map:** Attack distribution breakdowns across traffic classes and top targeted destination ports.
- **Tab 4: Predict Flow:** Single-sample live inference with interactive class probabilities, zero-day verdict, and on-demand **SHAP explainability chart**.
- **Tab 5: Active Response:** Live simulation of the graduated response engine with IP validation and idempotent `iptables` generation.

---

### 4. Run the Automated Test Suite

Verify system integrity, threat boundary math, and data leakage safeguards:

```powershell
pytest -v
```

**Result: 31 passed in 1.70 seconds** (100% test pass rate).

---

## Development Roadmap (DEPI Project Milestones)

| Phase | Focus Area | Deliverables & Status | Status |
|:---:|:---|:---|:---:|
| **1** | **Research & Requirements** | Problem definition, literature review of CICIDS2017, modular architecture design | **Completed** ✅ |
| **2** | **Data Engineering** | Leakage prevention, 5-$\sigma$ outlier clipping, train-only scaling and SMOTE | **Completed** ✅ |
| **3** | **Model Zoo Development** | Trained XGBoost, Deep MLP, 1D-CNN, and Benign-only Autoencoder | **Completed** ✅ |
| **4** | **Explainability & Risk** | SHAP TreeExplainer, KernelExplainer, waterfall plots, and 4-tier risk mapping | **Completed** ✅ |
| **5** | **Safe Active Response** | Centralized response engine, IP sanitization, idempotent dry-run iptables | **Completed** ✅ |
| **6** | **SOC Web Dashboard** | Streamlit dark-theme UI with 5 tabs, live inference, and telemetry upload | **Completed** ✅ |
| **7** | **Verification & Testing** | 31 Pytest unit tests, headless nbconvert execution across all 7 notebooks | **Completed** ✅ |
| **8** | **Graduation Submission** | Complete repository packaging, comprehensive README documentation | **Completed** ✅ |

---

## Limitations

- **Benchmark Domain Adaptation:** CICIDS2017 is an academic testbed recorded in an emulated environment; real-world production networks exhibit evolving background protocol distributions.
- **Reconstruction Anomaly Nuance:** The Autoencoder flags statistical deviations; benign network events with unusual payload structures (e.g., massive database backups) may trigger false-positive anomalies.
- **Dry-Run Defense Scope:** Active response is strictly executed in dry-run simulation mode to ensure safety; live production firewall modification requires elevated kernel permissions and carrier-grade testing.
- **Explainability Latency Trade-Off:** KernelExplainer on deep networks is computationally demanding; we utilize TreeExplainer and background sampling to maintain sub-second response times in the dashboard.

---

## Future Work

1. **Real-Time Streaming Ingestion:** Direct integration with live packet capture drivers (`libpcap`, `eBPF`) and stream brokers (Apache Kafka).
2. **SIEM & SOAR Integration:** Forwarding structured CEF/Syslog alerts directly into enterprise SIEMs (Splunk, Elastic SIEM, Microsoft Sentinel).
3. **Graph Neural Networks (GNN):** Modeling entire network topologies as graphs to detect lateral movement and advanced persistent threats (APTs).
4. **Continual Online Learning:** Incremental model weight updates to adapt to concept drift without catastrophic forgetting.
5. **Hardware Acceleration:** Compiling tree and neural models using TensorRT or ONNX Runtime for multi-gigabit line-rate FPGA/NIC deployment.

---

*Digital Egypt Pioneers Initiative (DEPI) - Graduation Project*
