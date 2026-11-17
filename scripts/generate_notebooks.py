"""Generator for the 7 Smart IDS v3 pedagogical notebooks.

Creates standard nbformat v4 Jupyter Notebooks with structured headings,
markdown explanations, security concept callouts, and clean modular code cells.
"""

from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_cell(cell_type: str, source: str) -> nbf.NotebookNode:
    """Create a code or markdown cell."""
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(source)
    else:
        return nbf.v4.new_code_cell(source)


def build_notebook_01() -> nbf.NotebookNode:
    """Notebook 01: Introduction & Security Basics."""
    nb = nbf.v4.new_notebook()
    cells = [
        create_cell("markdown", """# Smart IDS v3 — Notebook 01: Introduction & Security Basics
An Intelligent Network Intrusion Detection System for AI Engineers

---

### What You Will Learn
1. **The Intrusion Detection Problem**: Why traditional signature-based firewalls fail against modern cyber attacks.
2. **Security Concepts for AI Engineers**: Translating networking jargon (flows, 5-tuples, packets) into tabular ML representations.
3. **Attack Taxonomy**: Understanding the 8 attack families in the CICIDS2017 benchmark dataset.
4. **Data Ingestion**: Loading raw daily CSV files or generating a high-fidelity synthetic fallback.
5. **Exploratory Data Analysis (EDA)**: Investigating feature distributions, zero-division rate artifacts, and severe class imbalance.

---

### Prerequisites
- Python 3.11, pandas, matplotlib, and scikit-learn basics.
- **Zero prior knowledge of networking or cybersecurity is required!** All security concepts are introduced from first principles with ML analogies.
"""),

        create_cell("markdown", """## 1. Security Concepts for AI Engineers

Before looking at data, let's establish a shared vocabulary between cybersecurity and machine learning.

> **Security Concept: What is a Network Flow?**
> In computer networking, data moves in chunks called **packets**. A **flow** is a bidirectional conversation between two computers (for example, your laptop requesting a web page from a web server). 
> 
> *Machine Learning Analogy:* A network flow is simply **one row in a tabular dataset**! Each column is a summary statistic computed over that conversation (such as duration, total bytes sent, or average packet size).

> **Security Concept: What is the 5-Tuple?**
> Every flow is uniquely identified by five network properties:
> 1. Source IP (who sent it)
> 2. Destination IP (who received it)
> 3. Source Port (the application port on the sender)
> 4. Destination Port (the target service, e.g., Port 80 for HTTP, Port 22 for SSH)
> 5. Protocol (TCP or UDP)

> **Security Concept: What is CICFlowMeter?**
> CICFlowMeter is a network feature-extraction tool developed by the Canadian Institute for Cybersecurity. It listens to raw network packet streams and calculates **76 numerical statistical features** for each flow.

> **Security Concept: Attack Taxonomy (The 8 Classes)**
> - **BENIGN**: Normal, safe network traffic (web browsing, file transfers, email).
> - **DoS Hulk**: Denial-of-Service attack using high-volume HTTP request flooding to overwhelm a web server.
> - **PortScan**: Reconnaissance attack where an adversary probes hundreds of ports to see which doors are open.
> - **DDoS**: Distributed Denial-of-Service attack where traffic is coordinated across thousands of compromised computers.
> - **DoS GoldenEye**: Denial-of-Service attack that slowly consumes server connection pools with keep-alive headers.
> - **FTP-Patator**: Automated dictionary brute-force password guessing against an FTP file server (Port 21).
> - **SSH-Patator**: Automated dictionary brute-force password guessing against an encrypted SSH shell (Port 22).
> - **Web Attack**: SQL injection or cross-site scripting (XSS) attempting to steal data or compromise a web app.
"""),

        create_cell("markdown", """## 2. Setting Up the Environment and Reproducibility

Let's import our libraries, verify paths, and set a fixed seed (42) so that all results are 100% reproducible.
"""),

        create_cell("code", """import sys
from pathlib import Path

cur = Path.cwd()
root = cur if (cur / 'src').exists() else cur.parent
if str(root / 'src') not in sys.path:
    sys.path.insert(0, str(root / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from smart_ids.config import (
    CLASS_NAMES,
    CLASS_WEIGHTS,
    DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
)
from smart_ids.data_loader import load_cicids2017

np.random.seed(SEED)
pd.set_option('display.max_columns', 15)
print(f"Environment initialized. Seed fixed to {SEED}.")
"""),

        create_cell("markdown", """## 3. Loading the Benchmark Dataset

We load the CICIDS2017 dataset using `load_cicids2017`. If the 8 daily CSV files are placed in `data/raw/`, they are merged and sampled. If they are missing, the pipeline automatically activates a schema-identical synthetic fallback generator.

**What we do and why:**
We load a representative sample so that students can run the entire pipeline smoothly.
"""),

        create_cell("code", """# Load dataset (or generate synthetic fallback if raw CSVs are not present)
# For fast execution and learning, sample ~25,000 rows; can be scaled to 200,000 for final production benchmark
df_raw = load_cicids2017(data_dir=RAW_DATA_DIR, sample_size=25000, random_state=SEED)
print(f"Dataset Loaded Successfully!")
print(f"Total Rows: {df_raw.shape[0]:,}")
print(f"Total Columns: {df_raw.shape[1]}")
df_raw.head(3)
"""),

        create_cell("markdown", """## 4. Exploratory Data Analysis: Class Distribution

Now let's examine the class distribution of our network traffic.

> **Common Mistake: Naive Accuracy on Imbalanced Data**
> In real-world enterprise networks, over 80% to 99% of all traffic is normal (BENIGN). If a dataset is 83% BENIGN, a dummy model that always guesses "BENIGN" achieves **83% accuracy while detecting ZERO attacks**!
> 
> In cybersecurity, a **False Negative** (missing a real attack) is catastrophic: an adversary breaches your network undetected. Never use raw accuracy alone to evaluate an Intrusion Detection System.
"""),

        create_cell("code", """# Compute class distribution counts and percentages
class_counts = df_raw['Label'].value_counts()
class_pcts = df_raw['Label'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({'Flow Count': class_counts, 'Percentage (%)': class_pcts})
print("--- Class Distribution ---")
print(dist_df.round(2))

# Visualize class distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts.values, y=class_counts.index, palette='viridis')
plt.title("Empirical Class Distribution in Network Telemetry", fontsize=13, fontweight='bold')
plt.xlabel("Number of Flows", fontsize=11)
plt.ylabel("Traffic Class", fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
"""),

        create_cell("markdown", """**What did we learn?**
The data is severely imbalanced: BENIGN traffic dominates (~83%), while critical attacks like Web Attack, FTP-Patator, and SSH-Patator represent around 1% each. We must use techniques like SMOTE and weighted evaluation metrics to handle this asymmetry.
"""),

        create_cell("markdown", """## 5. Investigating Data Quality Pathologies

Network flow features generated by CICFlowMeter suffer from a known data-quality pathology: **infinite values (`+inf`, `-inf`)**.

> **Security Concept: Why do infinite values appear in network data?**
> Rate features such as `Flow Bytes/s` and `Flow Packets/s` are calculated by dividing the total bytes by `Flow Duration`. If a flow starts and ends instantly (duration = 0 microseconds), the computer divides by zero, resulting in `+inf` or `-inf`.

Let's detect how many infinite and missing values exist in the raw data.
"""),

        create_cell("code", """# Count infinite values across numerical features
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
num_infs = int(np.isinf(df_raw[numeric_cols].to_numpy()).sum())
num_nans = int(df_raw[numeric_cols].isna().sum().sum())
num_dupes = int(df_raw.duplicated().sum())

print(f"Data Quality Diagnostics:")
print(f" - Infinite values detected: {num_infs:,}")
print(f" - Missing values (NaN) detected: {num_nans:,}")
print(f" - Exact duplicate rows detected: {num_dupes:,}")
"""),

        create_cell("markdown", """## 6. Persisting Raw Dataset Sample for Downstream Notebooks

To ensure all subsequent notebooks (02 through 07) use the exact same data split, we save this raw sample to `data/processed/cicids2017_raw_sample.parquet`.
"""),

        create_cell("code", """# Cell 5 in Notebook 01
processed_raw_path = PROCESSED_DATA_DIR / 'cicids2017_raw_sample.csv'
processed_raw_path.parent.mkdir(parents=True, exist_ok=True)
df_raw.to_csv(processed_raw_path, index=False)
print(f"Saved dataset snapshot to: {processed_raw_path}")
"""),

        create_cell("markdown", """---
## Summary and Next Notebook

In this notebook, we learned:
- How network packets are grouped into bidirectional **flows** (tabular rows).
- The 8 attack categories in the CICIDS2017 dataset.
- Why naive accuracy is misleading on heavily imbalanced network traffic.
- Why zero-duration flows produce mathematical `inf` values.

**Next up:** In **Notebook 02: Data Preprocessing Pipeline**, we will clean infinite values, drop identifier columns to prevent data leakage, clip extreme outliers, split into train/val/test, scale features, and apply SMOTE to rebalance our training data!
""")
    ]
    nb.cells = cells
    return nb


def build_notebook_02() -> nbf.NotebookNode:
    """Notebook 02: Data Preprocessing Pipeline."""
    nb = nbf.v4.new_notebook()
    cells = [
        create_cell("markdown", """# Smart IDS v3 — Notebook 02: Data Preprocessing Pipeline
Cleaning, Leakage Prevention, Splitting, Scaling, and SMOTE Rebalancing

---

### What You Will Learn
1. **Handling Zero-Division**: Converting `inf` / `-inf` to NaN and imputing with robust column medians.
2. **Preventing Target Leakage**: Dropping non-causal network identifiers (`Source IP`, `Destination IP`, `Timestamp`, etc.).
3. **Outlier Mitigation**: Clipping rate spikes at empirical $\mu \pm 5\sigma$ to stabilize neural network gradients.
4. **Stratified Splitting**: Creating a leak-free 70% Train / 15% Validation / 15% Test partition.
5. **Proper Feature Scaling**: Fitting `StandardScaler` strictly on the training set.
6. **Class Rebalancing with SMOTE**: Synthesizing minority attack samples exclusively on the training partition.

---

### Prerequisites
- Completion of Notebook 01.
- Understanding of scikit-learn transformers (`fit` vs `transform`).
"""),

        create_cell("markdown", """## 1. Security Concept: Target Leakage in Network Security

> **Security Concept: What is Target Leakage in Cybersecurity?**
> In the CICIDS2017 testbed, attackers operated from fixed IP addresses (e.g. `192.168.10.50`), and victim servers had fixed IPs. Furthermore, certain attacks only occurred on Friday afternoon.
> 
> If we leave `Source IP`, `Destination IP`, or `Timestamp` in the feature matrix, machine learning models will simply **memorize the IP addresses and clock times**! The model will achieve 99.9% accuracy in the testbed, but when deployed into your actual company network, it will fail completely because the company's IP addresses and timestamps are different.
> 
> *Rule:* We must strictly drop all identifier columns. The model must learn the **statistical behavior** of the attack (packet lengths, inter-arrival times, flag ratios), not who sent it.
"""),

        create_cell("code", """import sys
from pathlib import Path

cur = Path.cwd()
root = cur if (cur / 'src').exists() else cur.parent
if str(root / 'src') not in sys.path:
    sys.path.insert(0, str(root / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from smart_ids.config import (
    CLASS_NAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    TEST_SIZE,
    VAL_SIZE_OF_REMAINDER,
)
from smart_ids.preprocessing import (
    clean_cicids_dataframe,
    remove_identifier_columns,
    clip_outliers,
    split_data,
    scale_features,
    apply_smote_rebalancing,
)
from smart_ids.artifacts import (
    save_scaler,
    save_label_encoder,
    save_feature_names,
)

np.random.seed(SEED)
print(f"Preprocessing environment initialized. Random seed: {SEED}")
"""),

        create_cell("markdown", """## 2. Step 1: Cleaning Infinities, Imputing Missing Values, Deduplicating

**What we do and why:**
We replace `np.inf` and `-np.inf` with `np.nan`. If any feature has > 50% missing values, it is dropped. Remaining missing values are filled with the **feature median** (the median is resistant to extreme skew, unlike the mean). Finally, duplicate flows are removed.
"""),

        create_cell("code", """# Load raw dataset snapshot saved by Notebook 01
raw_csv_path = PROCESSED_DATA_DIR / 'cicids2017_raw_sample.csv'
df_raw = pd.read_csv(raw_csv_path)
print(f"Loaded raw snapshot: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

# Clean infinities, NaNs, and duplicates
df_clean, audit = clean_cicids_dataframe(df_raw)
print("\\n--- Data Cleaning Audit Report ---")
for k, v in audit.items():
    print(f"  {k}: {v}")
"""),

        create_cell("markdown", """## 3. Step 2: Feature Selection (Dropping Identifiers)

**What we do and why:**
We remove columns containing keywords like `'ip'`, `'timestamp'`, `'flow id'`, `'src'`, `'dst'` to prevent target leakage.
"""),

        create_cell("code", """df_features, dropped_cols = remove_identifier_columns(df_clean)
print(f"Dropped {len(dropped_cols)} identifier columns: {dropped_cols}")
print(f"Remaining feature matrix shape: {df_features.shape}")
"""),

        create_cell("markdown", """## 4. Step 3: Outlier Clipping ($\mu \pm 5\sigma$)

**What we do and why:**
In network telemetry, anomalous network glitches can produce rate values like $10^{12}$ bytes/sec. These massive numbers cause exploding gradients during neural network training. We clip all features to $[\mu - 5\sigma, \mu + 5\sigma]$.
"""),

        create_cell("code", """df_clipped = clip_outliers(df_features, sigma_multiple=5.0)
print("Outlier clipping complete. Feature values are now bounded within +/- 5 standard deviations.")
"""),

        create_cell("markdown", """## 5. Step 4: Stratified Train / Validation / Test Splitting

> **Common Mistake: Data Leakage Before Splitting**
> A very common beginner mistake is to fit a scaler or run SMOTE on the entire dataset *before* creating train and test sets. This leaks future test information into the training data and produces invalid, overly optimistic test scores!
> 
> *The Golden Rule:* Split first. Fit transformers and oversamplers **on the training partition ONLY**.

We perform a stratified split:
- **Test Set**: 15% of the total dataset.
- **Validation Set**: 15% of the remainder (~12.75% of total).
- **Train Set**: The remaining ~72.25%.
"""),

        create_cell("code", """X_train_df, X_val_df, X_test_df, y_train_raw, y_val_raw, y_test_raw = split_data(
    df_clipped,
    target_col='Label',
    test_size=TEST_SIZE,
    val_size_of_remainder=VAL_SIZE_OF_REMAINDER,
    random_state=SEED,
)

# Encode categorical labels to integers 0..7
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)

y_train = label_encoder.transform(y_train_raw)
y_val = label_encoder.transform(y_val_raw)
y_test = label_encoder.transform(y_test_raw)

feature_names = list(X_train_df.columns)
print(f"Feature count: {len(feature_names)}")
print(f"Classes encoded: {list(zip(range(len(CLASS_NAMES)), CLASS_NAMES))}")
"""),

        create_cell("markdown", """## 6. Step 5: Feature Standardization (Fit on Train Only)

**What we do and why:**
We fit `StandardScaler` strictly on `X_train_df`, then transform `X_val_df` and `X_test_df`.
"""),

        create_cell("code", """X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
    X_train_df, X_val_df, X_test_df
)

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_val_scaled shape:   {X_val_scaled.shape}")
print(f"X_test_scaled shape:  {X_test_scaled.shape}")
print(f"Train Mean (first 3 features): {X_train_scaled[:, :3].mean(axis=0).round(4)}")
print(f"Train Std (first 3 features):  {X_train_scaled[:, :3].std(axis=0).round(4)}")
"""),

        create_cell("markdown", """## 7. Step 6: SMOTE Class Rebalancing (Train Only)

> **Common Mistake: Applying SMOTE to Validation or Test Data**
> Validation and test sets must represent the **real world**, which is naturally imbalanced! If you apply SMOTE to validation or test data, you are testing your model on synthetic data that doesn't exist in reality.
> 
> Apply SMOTE **exclusively to the training partition**.

We use a dynamic $k$-neighbor guard: $k = \max(1, \min(5, \text{smallest\_class\_size} - 1))$.
"""),

        create_cell("code", """# Pre-SMOTE distribution
pre_counts = pd.Series(y_train).value_counts().sort_index()

# Apply SMOTE exclusively on training partition
X_train_sm, y_train_sm = apply_smote_rebalancing(
    X_train_scaled, y_train, random_state=SEED
)

# Post-SMOTE distribution
post_counts = pd.Series(y_train_sm).value_counts().sort_index()

smote_comparison = pd.DataFrame({
    'Class Name': CLASS_NAMES,
    'Pre-SMOTE Train': pre_counts.values,
    'Post-SMOTE Train': post_counts.values,
})
print("--- Training Partition SMOTE Rebalancing ---")
print(smote_comparison)
"""),

        create_cell("markdown", """## 8. Persisting Preprocessing Artifacts

We save the fitted `StandardScaler`, `LabelEncoder`, feature name list, and preprocessed NumPy arrays so downstream notebooks can load them directly.
"""),

        create_cell("code", """save_scaler(scaler)
save_label_encoder(label_encoder)
save_feature_names(feature_names)

# Save processed arrays for instant loading in Notebooks 03, 04, 05, 06
np.save(PROCESSED_DATA_DIR / 'X_train_sm.npy', X_train_sm)
np.save(PROCESSED_DATA_DIR / 'y_train_sm.npy', y_train_sm)
np.save(PROCESSED_DATA_DIR / 'X_train_scaled.npy', X_train_scaled)
np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train)
np.save(PROCESSED_DATA_DIR / 'X_val_scaled.npy', X_val_scaled)
np.save(PROCESSED_DATA_DIR / 'y_val.npy', y_val)
np.save(PROCESSED_DATA_DIR / 'X_test_scaled.npy', X_test_scaled)
np.save(PROCESSED_DATA_DIR / 'y_test.npy', y_test)

print("All preprocessing artifacts and NumPy splits saved successfully!")
"""),

        create_cell("markdown", """---
## Summary and Next Notebook

In this notebook, we built a leak-free data engineering pipeline:
- Converted zero-division infinities to NaN and median-imputed them.
- Excluded IP and timestamp columns to eliminate target leakage.
- Clipped features at $\pm 5\sigma$ to protect gradient descent.
- Split data into train (70%), validation (15%), and test (15%).
- Fit `StandardScaler` and `SMOTE` **on the training data only**.

**Next up:** In **Notebook 03: Supervised Multi-Class Models**, we will train our XGBoost baseline, deep Multi-Layer Perceptron (MLP), and 1D-Convolutional Neural Network (1D-CNN)!
""")
    ]
    nb.cells = cells
    return nb


def build_notebook_03() -> nbf.NotebookNode:
    """Notebook 03: Supervised Multi-Class Models."""
    nb = nbf.v4.new_notebook()
    cells = [
        create_cell("markdown", """# Smart IDS v3 — Notebook 03: Supervised Multi-Class Models
XGBoost Baseline, Deep Multi-Layer Perceptron (MLP), and 1D-CNN

---

### What You Will Learn
1. **Tree-Based Baseline**: Training an Extreme Gradient Boosting (XGBoost) classifier with histogram split finding.
2. **Deep MLP Architecture**: Building a 4-layer regularized funnel (512-256-128-64) with LeakyReLU, BatchNorm, L2, and Dropout.
3. **1D-CNN Architecture**: Reinterpreting tabular flow features as pseudo-sequences for localized pattern detection.
4. **Adaptive Training Callbacks**: Using `EarlyStopping` (patience 8), `ReduceLROnPlateau`, and `ModelCheckpoint`.
5. **Convergence Analysis**: Comparing learning curves and loss stabilization across architectures.

---

### Prerequisites
- Completion of Notebooks 01 and 02.
- Basic understanding of neural network layers (Dense, Conv1D, BatchNormalization).
"""),

        create_cell("markdown", """## 1. The Supervised Learning Strategy

In this notebook, we train 3 distinct supervised models:
1. **XGBoost Classifier**: Fast, battle-tested decision-tree ensemble baseline for tabular data.
2. **Deep Multi-Layer Perceptron (MLP)**: Deep feedforward neural network designed to capture non-linear relationships across all features.
3. **1D-Convolutional Neural Network (1D-CNN)**: Slides 1D convolutional kernels across the feature vector to detect localized co-occurrences (e.g. adjacent packet timing and flag statistics).
"""),

        create_cell("code", """import sys
from pathlib import Path

cur = Path.cwd()
root = cur if (cur / 'src').exists() else cur.parent
if str(root / 'src') not in sys.path:
    sys.path.insert(0, str(root / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from smart_ids.config import (
    CLASS_NAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
)
from smart_ids.models import build_mlp, build_cnn, build_xgboost
from smart_ids.training import train_mlp, train_cnn, train_xgboost, get_callbacks
from smart_ids.artifacts import save_xgboost_model

# Set seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"TensorFlow {tf.__version__} initialized.")
"""),

        create_cell("markdown", """## 2. Loading Preprocessed Data Splits

We load the scaled and SMOTE-rebalanced training data, plus the naturally imbalanced validation and test sets saved by Notebook 02.
"""),

        create_cell("code", """X_train_sm = np.load(PROCESSED_DATA_DIR / 'X_train_sm.npy')
y_train_sm = np.load(PROCESSED_DATA_DIR / 'y_train_sm.npy')
X_val_scaled = np.load(PROCESSED_DATA_DIR / 'X_val_scaled.npy')
y_val = np.load(PROCESSED_DATA_DIR / 'y_val.npy')
X_test_scaled = np.load(PROCESSED_DATA_DIR / 'X_test_scaled.npy')
y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')

# One-hot encode targets for categorical cross-entropy neural network loss
num_classes = len(CLASS_NAMES)
y_train_cat = to_categorical(y_train_sm, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

n_features = X_train_sm.shape[1]
print(f"Input Features: {n_features}")
print(f"Training Samples (Balanced via SMOTE): {X_train_sm.shape[0]:,}")
print(f"Validation Samples (Naturally Imbalanced): {X_val_scaled.shape[0]:,}")
print(f"Test Samples (Naturally Imbalanced):       {X_test_scaled.shape[0]:,}")
"""),

        create_cell("markdown", """## 3. Model 1: Extreme Gradient Boosting (XGBoost) Baseline

**What we do and why:**
We train an `XGBClassifier` with 300 estimators, tree depth 6, and histogram split finding (`tree_method='hist'`). We supply the validation set to monitor early stopping with a patience of 20 rounds.
"""),

        create_cell("code", """xgb_clf = build_xgboost(num_classes=num_classes, random_state=SEED)
xgb_trained = train_xgboost(xgb_clf, X_train_sm, y_train_sm, X_val_scaled, y_val)

# Save trained XGBoost model to JSON
save_xgboost_model(xgb_trained)
print("XGBoost Baseline trained and saved to models/xgb_model.json.")
"""),

        create_cell("markdown", """## 4. Model 2: Deep Multi-Layer Perceptron (MLP)

**What we do and why:**
We build a 4-layer funnel architecture: `512 -> 256 -> 128 -> 64`.
Each block applies:
1. `Dense(units, L2=1e-4, use_bias=False)`
2. `BatchNormalization()` (stabilizes gradient flow)
3. `LeakyReLU(alpha=0.1)` (prevents dying ReLU neurons)
4. `Dropout(0.4)` (prevents co-adaptation and overfitting)
5. `Adam` optimizer with `clipnorm=1.0` (gradient clipping).
"""),

        create_cell("code", """mlp_model = build_mlp(input_dim=n_features, num_classes=num_classes)
mlp_callbacks = get_callbacks("mlp", patience_es=8, checkpoint_name="best_mlp.keras")

mlp_trained, mlp_history = train_mlp(
    mlp_model,
    X_train_sm,
    y_train_cat,
    X_val_scaled,
    y_val_cat,
    epochs=15,
    batch_size=1024,
    callbacks=mlp_callbacks,
)
print("MLP Training complete. Best weights saved to models/best_mlp.keras.")
"""),

        create_cell("markdown", """## 5. Model 3: 1D-Convolutional Neural Network (1D-CNN)

**What we do and why:**
We reshape the input features to shape `(n_samples, n_features, 1)`. 
Conv1D kernels slide across the feature sequence to capture statistical co-occurrences between adjacent engineered features (like forward/backward packet sizes and inter-arrival times).
"""),

        create_cell("code", """cnn_model = build_cnn(input_dim=n_features, num_classes=num_classes)
cnn_callbacks = get_callbacks("cnn", patience_es=4, checkpoint_name="best_cnn.keras")

cnn_trained, cnn_history = train_cnn(
    cnn_model,
    X_train_sm,
    y_train_cat,
    X_val_scaled,
    y_val_cat,
    epochs=12,
    batch_size=2048,
    callbacks=cnn_callbacks,
)
print("1D-CNN Training complete. Best weights saved to models/best_cnn.keras.")
"""),

        create_cell("markdown", """## 6. Comparing Neural Training Convergence Histories

Let's visualize the loss and accuracy convergence profiles of the MLP and 1D-CNN across epochs.
"""),

        create_cell("code", """fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# MLP Loss
axes[0, 0].plot(mlp_history.history['loss'], label='Train Loss', color='#2563eb')
axes[0, 0].plot(mlp_history.history['val_loss'], label='Val Loss', linestyle='--', color='#f59e0b')
axes[0, 0].set_title('MLP — Categorical Cross-Entropy Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MLP Accuracy
axes[0, 1].plot(mlp_history.history['accuracy'], label='Train Acc', color='#2563eb')
axes[0, 1].plot(mlp_history.history['val_accuracy'], label='Val Acc', linestyle='--', color='#10b981')
axes[0, 1].set_title('MLP — Classification Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# CNN Loss
axes[1, 0].plot(cnn_history.history['loss'], label='Train Loss', color='#7c3aed')
axes[1, 0].plot(cnn_history.history['val_loss'], label='Val Loss', linestyle='--', color='#f59e0b')
axes[1, 0].set_title('1D-CNN — Categorical Cross-Entropy Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# CNN Accuracy
axes[1, 1].plot(cnn_history.history['accuracy'], label='Train Acc', color='#7c3aed')
axes[1, 1].plot(cnn_history.history['val_accuracy'], label='Val Acc', linestyle='--', color='#10b981')
axes[1, 1].set_title('1D-CNN — Classification Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""),

        create_cell("markdown", """---
## Summary and Next Notebook

In this notebook:
- We trained an **XGBoost classifier** baseline with early stopping.
- We constructed and trained a regularized **Deep MLP** (512-256-128-64).
- We trained a **1D-CNN** with hierarchical convolutional blocks and global average pooling.
- All three models were persisted to `models/` for downstream benchmarking.

**Next up:** In **Notebook 04: Autoencoder Zero-Day Anomaly Detection**, we will explore why supervised models fail against unseen attacks, and how to train an unsupervised Autoencoder on benign traffic to detect zero-day threats!
""")
    ]
    nb.cells = cells
    return nb


def build_notebook_04() -> nbf.NotebookNode:
    """Notebook 04: Autoencoder Zero-Day Detector."""
    nb = nbf.v4.new_notebook()
    cells = [
        create_cell("markdown", """# Smart IDS v3 — Notebook 04: Unsupervised Zero-Day Anomaly Detection
Autoencoder Architecture, Benign-Only Training, and Anomaly Threshold Calibration

---

### What You Will Learn
1. **The Closed-Set Dilemma**: Why supervised classifiers fail against new, unseen cyber threats.
2. **Zero-Day Attacks**: Defining zero-day vulnerabilities in security operations.
3. **ML Analogy**: Using an Undercomplete Autoencoder as an anomaly detector.
4. **Benign-Only Training**: Training solely on normal traffic under a Mean Squared Error (MSE) objective.
5. **Threshold Calibration**: Calibrating the decision boundary at the 95th percentile of benign validation reconstruction error.
6. **Binary Evaluation**: Assessing Anomaly vs Benign performance on held-out test data.

---

### Prerequisites
- Completion of Notebooks 01 through 03.
- Basic understanding of autoencoders (Encoder, Bottleneck Latent Space, Decoder).
"""),

        create_cell("markdown", """## 1. Security Concept: The Zero-Day Problem & Closed-Set Supervised Learning

> **Security Concept: What is a Zero-Day Threat?**
> A **zero-day** is a vulnerability or exploit that has never been seen before. There are zero days of warning, zero public signatures, and zero labeled examples in training datasets.
> 
> *The Closed-Set Assumption in Supervised Learning:*
> Supervised models (like MLP, CNN, and XGBoost) can only classify attacks they were explicitly trained to recognize. If a completely new attack arrives, a supervised model will be forced to guess one of its existing labels (or mistake it for Benign).
> 
> *The Solution: Anomaly Detection via Autoencoder:*
> Instead of learning what attacks look like, we train an Autoencoder to learn what **BENIGN (normal) traffic** looks like.
> When normal traffic passes through, the Autoencoder reconstructs it with low error.
> When an abnormal zero-day attack arrives, the Autoencoder fails to reconstruct it, producing a **high reconstruction error**!
"""),

        create_cell("code", """import sys
from pathlib import Path

cur = Path.cwd()
root = cur if (cur / 'src').exists() else cur.parent
if str(root / 'src') not in sys.path:
    sys.path.insert(0, str(root / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from smart_ids.config import (
    CLASS_NAMES,
    BENIGN_CLASS_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
)
from smart_ids.models import build_autoencoder
from smart_ids.training import train_autoencoder, get_callbacks
from smart_ids.evaluation import (
    compute_reconstruction_error,
    calibrate_anomaly_threshold,
    evaluate_autoencoder_binary,
)
from smart_ids.artifacts import save_anomaly_threshold

np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"Autoencoder environment initialized with seed {SEED}.")
"""),

        create_cell("markdown", """## 2. Isolating Benign Traffic for Self-Supervised Training

**What we do and why:**
We filter our scaled training and validation datasets to keep **ONLY samples where Label == 'BENIGN'** (class index 0). All attacks are strictly excluded from the Autoencoder's training!
"""),

        create_cell("code", """# Load scaled splits from Notebook 02
X_train_scaled = np.load(PROCESSED_DATA_DIR / 'X_train_scaled.npy')
y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
X_val_scaled = np.load(PROCESSED_DATA_DIR / 'X_val_scaled.npy')
y_val = np.load(PROCESSED_DATA_DIR / 'y_val.npy')
X_test_scaled = np.load(PROCESSED_DATA_DIR / 'X_test_scaled.npy')
y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')

# Filter for Benign only (Class Index 0)
benign_idx = 0
train_benign_mask = (y_train == benign_idx)
val_benign_mask = (y_val == benign_idx)

X_benign_train = X_train_scaled[train_benign_mask]
X_benign_val = X_val_scaled[val_benign_mask]

n_features = X_benign_train.shape[1]
print(f"Benign Training Samples:   {X_benign_train.shape[0]:,}")
print(f"Benign Validation Samples: {X_benign_val.shape[0]:,}")
"""),

        create_cell("markdown", """## 3. Building and Training the Undercomplete Autoencoder

**What we do and why:**
We construct the Autoencoder:
- **Encoder**: `Dense(256) -> BN -> ReLU -> Dense(128) -> BN -> ReLU -> Dense(64) -> Latent(32)`.
- **Decoder**: `Dense(64) -> BN -> ReLU -> Dense(128) -> BN -> ReLU -> Dense(256) -> Dense(n_features, linear)`.
Linear activation is essential on the reconstruction layer because features were standardized to real-valued $z$-scores ($\mu=0, \sigma=1$).
"""),

        create_cell("code", """autoencoder, encoder, decoder = build_autoencoder(
    input_dim=n_features,
    latent_dim=32,
    learning_rate=5e-4,
)

ae_callbacks = get_callbacks(
    "autoencoder",
    patience_es=10,
    checkpoint_name="best_autoencoder.keras",
)

autoencoder_trained, ae_history = train_autoencoder(
    autoencoder,
    X_benign_train,
    X_benign_val,
    epochs=15,
    batch_size=1024,
    callbacks=ae_callbacks,
)
print("Autoencoder training complete. Saved to models/best_autoencoder.keras.")
"""),

        create_cell("markdown", """## 4. Calibrating the Anomaly Detection Threshold

**What we do and why:**
We compute the reconstruction error (Mean Squared Error) for each flow in `X_benign_val`. 
We set `ANOMALY_THRESHOLD` to the **95th percentile** of these benign errors.
This mathematically guarantees that the expected false-positive rate on known benign traffic is bounded at ~5%.
"""),

        create_cell("code", """# Calibrate 95th percentile threshold on unseen benign validation flows
threshold = calibrate_anomaly_threshold(autoencoder_trained, X_benign_val, percentile=95.0)

# Save threshold metadata to models/anomaly_threshold.json
save_anomaly_threshold(threshold, percentile=95.0)
print(f"Calibrated Anomaly Decision Threshold: {threshold:.6f}")
"""),

        create_cell("markdown", """## 5. Visualizing Reconstruction Error Distributions

Let's compare the reconstruction error distributions of **Benign Test Flows** vs **Attack Test Flows**.
"""),

        create_cell("code", """test_errors = compute_reconstruction_error(autoencoder_trained, X_test_scaled)
is_attack_test = (y_test != benign_idx)

benign_test_errors = test_errors[~is_attack_test]
attack_test_errors = test_errors[is_attack_test]

plt.figure(figsize=(10, 5))
plt.hist(benign_test_errors, bins=50, range=(0, threshold * 3), alpha=0.6, label='Benign Test Flows', color='#10b981', density=True)
plt.hist(attack_test_errors, bins=50, range=(0, threshold * 3), alpha=0.6, label='Attack Test Flows', color='#ef4444', density=True)
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold (95th %ile) = {threshold:.4f}')

plt.title("Reconstruction Error Distribution: Benign vs Attacks", fontsize=13, fontweight='bold')
plt.xlabel("Reconstruction Error (MSE)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),

        create_cell("markdown", """## 6. Binary Evaluation on Held-Out Test Data

**What we do and why:**
Any flow with reconstruction error > `ANOMALY_THRESHOLD` is predicted as `1` (Anomaly).
We evaluate binary Accuracy, Precision, Recall, and F1 score against ground truth `(y_test != BENIGN)`.
"""),

        create_cell("code", """y_pred_binary = (test_errors > threshold).astype(int)
y_true_binary = is_attack_test.astype(int)

ae_metrics = evaluate_autoencoder_binary(y_true_binary, y_pred_binary)
print("--- Autoencoder Binary Anomaly Evaluation ---")
for k, v in ae_metrics.items():
    print(f"  {k}: {v if isinstance(v, str) else round(v, 4)}")
"""),

        create_cell("markdown", """---
## Summary and Next Notebook

In this notebook:
- We trained an **Undercomplete Autoencoder** exclusively on benign network traffic.
- We calibrated a **95th percentile reconstruction threshold** on unseen benign validation data.
- We confirmed that malicious traffic exhibits substantially higher reconstruction error.
- We evaluated the Autoencoder under a **binary framing** (Normal vs Anomaly).

**Next up:** In **Notebook 05: Evaluation and Comparison**, we will benchmark per-sample inference latency for all 4 models, compare their performance side-by-side, generate confusion matrices, and plot multi-class One-vs-Rest ROC curves!
""")
    ]
    nb.cells = cells
    return nb


def build_notebook_05() -> nbf.NotebookNode:
    """Notebook 05: Evaluation and Comparison."""
    nb = nbf.v4.new_notebook()
    cells = [
        create_cell("markdown", """# Smart IDS v3 — Notebook 05: Comprehensive Model Evaluation & Benchmarking
Multi-Model Comparison Table, Microsecond Latency, Normalized Confusion Matrices, and ROC/AUC

---

### What You Will Learn
1. **Real-Time Latency Benchmarking**: Measuring per-sample inference latency in microseconds with untimed warm-up runs.
2. **Side-by-Side Comparison**: Contrasting XGBoost, MLP, 1D-CNN (multi-class) with the Autoencoder (binary anomaly).
3. **Normalized Confusion Matrices**: Plotting a 2x2 grid showing conditional probabilities $P(\\text{pred} \\mid \\text{true})$.
4. **One-vs-Rest ROC/AUC Curves**: Computing per-class and macro-averaged ROC curves.
5. **Persisting Results**: Saving `models/model_comparison.csv` and evaluation figures.

---

### Prerequisites
- Completion of Notebooks 01 through 04.
- All 4 models trained and saved in `models/`.
"""),

        create_cell("markdown", """## 1. Security Concept: The Operational Latency Budget

> **Security Concept: Why does inference latency matter in network security?**
> A modern 10-Gigabit-per-second enterprise link processes over **1 million packets per second**.
> If an intrusion detection model takes 100 milliseconds to analyze a flow, incoming queues will instantly overflow, dropping packets and causing network downtime.
> In a production Intrusion Prevention System (IPS), models must evaluate flows within **microseconds** or small milliseconds.
"""),

        create_cell("code", """import sys
from pathlib import Path

cur = Path.cwd()
root = cur if (cur / 'src').exists() else cur.parent
if str(root / 'src') not in sys.path:
    sys.path.insert(0, str(root / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from smart_ids.config import (
    CLASS_NAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
)
from smart_ids.artifacts import (
    load_all_artifacts,
    load_anomaly_threshold,
)
from smart_ids.evaluation import (
    measure_latency,
    compute_reconstruction_error,
    evaluate_supervised,
    evaluate_autoencoder_binary,
    plot_confusion_matrices,
    plot_roc_multiclass,
)

np.random.seed(SEED)
print("Evaluation environment initialized.")
"""),

        create_cell("markdown", """## 2. Loading All Trained Models and Test Telemetry

We load all 4 saved models (MLP, 1D-CNN, XGBoost, Autoencoder) using `load_all_artifacts()`, along with the held-out test data.
"""),

        create_cell("code", """artifacts = load_all_artifacts(MODELS_DIR)
print(f"Loaded artifacts: {list(artifacts.keys())}")

mlp_model = artifacts['MLP']
cnn_model = artifacts['CNN']
xgb_model = artifacts['XGBoost']
ae_model = artifacts['Autoencoder']
threshold_data = artifacts['threshold_data']
threshold = threshold_data['anomaly_threshold']

# Load test data
X_test_scaled = np.load(PROCESSED_DATA_DIR / 'X_test_scaled.npy')
y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')
n_test = len(y_test)
print(f"Test dataset: {n_test:,} samples across {len(CLASS_NAMES)} classes.")
"""),

        create_cell("markdown", """## 3. Benchmarking Per-Sample Inference Latency ($\mu s$)

**What we do and why:**
We run 1 untimed warm-up call (to warm JIT compilers and cache lines), then measure the exact wall-clock time of 100 repeated single-sample predictions using Python's high-resolution `time.perf_counter()`.
"""),

        create_cell("code", """latency_mlp = measure_latency(mlp_model, X_test_scaled, model_type='keras', n_runs=100)
latency_cnn = measure_latency(cnn_model, X_test_scaled, model_type='cnn', n_runs=100)
latency_xgb = measure_latency(xgb_model, X_test_scaled, model_type='xgboost', n_runs=100)
latency_ae  = measure_latency(ae_model,  X_test_scaled, model_type='ae', n_runs=100)

latency_df = pd.DataFrame({
    'Model': ['MLP (DNN)', '1D-CNN', 'XGBoost Baseline', 'Autoencoder (Zero-Day)'],
    'Latency (us)': [latency_mlp, latency_cnn, latency_xgb, latency_ae],
    'Throughput (flows/sec)': [1e6 / latency_mlp, 1e6 / latency_cnn, 1e6 / latency_xgb, 1e6 / latency_ae],
})
print("--- Inference Latency Benchmarks ---")
print(latency_df.round(2))
"""),

        create_cell("markdown", """## 4. Evaluating All Models & Building the Comparison Table

**What we do and why:**
We compute predictions on the test set for all 4 models:
- Supervised models: Weighted Precision, Recall, F1, and Accuracy.
- Autoencoder: Binary Precision, Recall, F1, and Accuracy against `(y_test != 0)`.
We consolidate these into `model_comparison.csv`.
"""),

        create_cell("code", """# Predictions
# MLP
y_proba_mlp = mlp_model.predict(X_test_scaled, verbose=0)
y_pred_mlp = np.argmax(y_proba_mlp, axis=1)

# 1D-CNN
X_test_cnn = X_test_scaled.reshape(n_test, X_test_scaled.shape[1], 1)
y_proba_cnn = cnn_model.predict(X_test_cnn, verbose=0)
y_pred_cnn = np.argmax(y_proba_cnn, axis=1)

# XGBoost
y_proba_xgb = xgb_model.predict_proba(X_test_scaled)
y_pred_xgb = np.argmax(y_proba_xgb, axis=1)

# Autoencoder
test_errors = compute_reconstruction_error(ae_model, X_test_scaled)
y_pred_ae = (test_errors > threshold).astype(int)
y_true_binary = (y_test != 0).astype(int)

# Compute metrics
res_mlp = evaluate_supervised(y_test, y_pred_mlp, latency_us=latency_mlp)
res_cnn = evaluate_supervised(y_test, y_pred_cnn, latency_us=latency_cnn)
res_xgb = evaluate_supervised(y_test, y_pred_xgb, latency_us=latency_xgb)
res_ae  = evaluate_autoencoder_binary(y_true_binary, y_pred_ae, latency_us=latency_ae)

comparison_rows = [
    {'Model': 'MLP (DNN)', **res_mlp},
    {'Model': '1D-CNN', **res_cnn},
    {'Model': 'XGBoost Baseline', **res_xgb},
    {'Model': 'Autoencoder (Zero-Day)', **res_ae},
]
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)

print("--- Unified Model Comparison Table ---")
print(comparison_df.round(4))
"""),

        create_cell("markdown", """## 5. Normalized Confusion Matrices (2x2 Grid)

**What we do and why:**
We plot row-normalized confusion matrices for all 4 models. Each cell shows the probability $P(\\text{Predicted Class} \\mid \\text{True Class})$. Row normalization ensures rare attack classes are clearly visible despite class imbalance.
"""),

        create_cell("code", """cms = {
    'MLP': confusion_matrix(y_test, y_pred_mlp),
    '1D-CNN': confusion_matrix(y_test, y_pred_cnn),
    'XGBoost': confusion_matrix(y_test, y_pred_xgb),
    'Autoencoder': confusion_matrix(y_true_binary, y_pred_ae),
}

fig_cm = plot_confusion_matrices(
    cms,
    class_names=CLASS_NAMES,
    save_path=MODELS_DIR / 'confusion_matrices.png',
)
plt.show()
"""),

        create_cell("markdown", """## 6. Receiver Operating Characteristic (ROC) & AUC Analysis

**What we do and why:**
We compute One-vs-Rest ROC curves with macro-average AUC for the multi-class models, and a direct ROC curve for the Autoencoder's continuous reconstruction error.
"""),

        create_cell("code", """fig, axes = plt.subplots(2, 2, figsize=(15, 12))

plot_roc_multiclass(y_test, y_proba_mlp, CLASS_NAMES, "MLP (One-vs-Rest ROC)", axes[0, 0])
plot_roc_multiclass(y_test, y_proba_cnn, CLASS_NAMES, "1D-CNN (One-vs-Rest ROC)", axes[0, 1])
plot_roc_multiclass(y_test, y_proba_xgb, CLASS_NAMES, "XGBoost (One-vs-Rest ROC)", axes[1, 0])
plot_roc_multiclass(y_true_binary, test_errors, ["Benign", "Attack"], "Autoencoder (Reconstruction Error ROC)", axes[1, 1], binary=True)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()
"""),

        create_cell("markdown", """---
## Summary and Next Notebook

In this notebook:
- We benchmarked **microsecond inference latency** across all four models.
- We generated the **unified model comparison table** (`models/model_comparison.csv`).
- We plotted **normalized confusion matrices** and **One-vs-Rest ROC curves**.

**Next up:** In **Notebook 06: Explainable AI via SHAP**, we will break open the "black box" and explain why our models make their predictions using game-theoretic Shapley values!
""")
    ]
    nb.cells = cells
    return nb


def build_notebook_06() -> nbf.NotebookNode:
    """Notebook 06: SHAP Interpretability Layer."""
    nb = nbf.v4.new_notebook()
    cells = [
        create_cell("markdown", """# Smart IDS v3 — Notebook 06: Explainable AI & Feature Attribution via SHAP
TreeExplainer, KernelExplainer, and Single-Sample Waterfall Decomposition

---

### What You Will Learn
1. **The Need for Explainability in SOCs**: Why black-box predictions are an operational liability in incident response.
2. **Game Theory Foundations**: How Shapley values fairly distribute credit among features.
3. **SHAP TreeExplainer**: Fast, exact global feature importance and attack beeswarm plots for XGBoost.
4. **SHAP KernelExplainer**: Model-agnostic attribution for the deep MLP with defensive multi-class shape handling.
5. **Waterfall Decomposition**: Dissecting a single critical detection event into its exact additive feature contributions.

---

### Prerequisites
- Completion of Notebooks 01 through 05.
- Understanding of feature importance concepts.
"""),

        create_cell("markdown", """## 1. Security Concept: The Black-Box Problem in Security Operations

> **Security Concept: Why Black-Box AI Fails in a SOC**
> If an automated IDS triggers a high-severity alert and instructs a firewall to block an IP address, human security analysts must immediately answer:
> 1. *Why* did the model flag this flow?
> 2. Is this an actual zero-day attack, or did a regular user just download a large video file?
> 
> Without explainability, analysts experience **alert fatigue** or disregard the AI's recommendations. SHAP (SHapley Additive exPlanations) provides mathematically guaranteed, human-auditable explanations for every single decision.
"""),

        create_cell("code", """import sys
from pathlib import Path

cur = Path.cwd()
root = cur if (cur / 'src').exists() else cur.parent
if str(root / 'src') not in sys.path:
    sys.path.insert(0, str(root / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from smart_ids.config import (
    CLASS_NAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
)
from smart_ids.artifacts import (
    load_xgboost_model,
    load_feature_names,
)
from smart_ids.explain import (
    compute_shap_xgboost,
    compute_shap_mlp,
    generate_waterfall_plot,
)

np.random.seed(SEED)
print(f"SHAP version: {shap.__version__}")
"""),

        create_cell("markdown", """## 2. Loading Models, Test Data, and Feature Schema

We load the trained XGBoost model, the Keras MLP, and the standardized test features.
"""),

        create_cell("code", """xgb_model = load_xgboost_model(MODELS_DIR / 'xgb_model.json')
mlp_model = tf.keras.models.load_model(MODELS_DIR / 'best_mlp.keras')
feature_names = load_feature_names(MODELS_DIR / 'feature_names.json')

X_test_scaled = np.load(PROCESSED_DATA_DIR / 'X_test_scaled.npy')
y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')

print(f"Loaded {len(feature_names)} feature names.")
print(f"Test samples available: {len(X_test_scaled):,}")
"""),

        create_cell("markdown", """## 3. TreeExplainer for XGBoost: Global Feature Importance

**What we do and why:**
`TreeExplainer` exploits tree structure to compute exact Shapley values in polynomial time. We evaluate 500 test samples and plot:
1. Global feature importance bar chart (ranking features across all attack classes).
2. Beeswarm summary plot for DoS Hulk (showing whether high or low values trigger the detection).
"""),

        create_cell("code", """explainer_xgb, shap_values_xgb, X_shap_df = compute_shap_xgboost(
    xgb_model,
    X_test_scaled,
    feature_names,
    n_samples=500,
)

# 1. Global Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb, X_shap_df, plot_type='bar', show=False, max_display=15)
plt.title("XGBoost — Global Feature Importance (Top 15 Features)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""),

        create_cell("markdown", """## 4. Beeswarm Summary Plot for DoS Hulk (Class Index 1)

**What we do and why:**
In the beeswarm plot:
- Each dot is a flow.
- Red color indicates a high feature value, blue indicates a low value.
- Horizontal position indicates whether that feature increases (right) or decreases (left) the probability of DoS Hulk.
"""),

        create_cell("code", """# Extract SHAP values for class 1 (DoS Hulk)
if isinstance(shap_values_xgb, list):
    shap_vals_class1 = shap_values_xgb[1]
elif isinstance(shap_values_xgb, np.ndarray) and len(shap_values_xgb.shape) == 3:
    shap_vals_class1 = shap_values_xgb[:, :, 1]
else:
    shap_vals_class1 = shap_values_xgb

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_class1, X_shap_df, show=False, max_display=12)
plt.title("SHAP Beeswarm: Directional Feature Impact for DoS Hulk", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""),

        create_cell("markdown", """## 5. KernelExplainer for Deep MLP (Model-Agnostic Attribution)

**What we do and why:**
Unlike trees, neural networks do not have an exact tree algorithm. We use `KernelExplainer`, which perturbs features and measures the change in model output.
To keep the computation fast and responsive, we use a 10-sample reference background and explain 20 test flows.
"""),

        create_cell("code", """explainer_mlp, mean_abs_shap_mlp, _ = compute_shap_mlp(
    mlp_model,
    X_test_scaled,
    feature_names,
    n_background=10,
    n_samples=20,
)

# Plot top 15 features for MLP
top_indices = np.argsort(mean_abs_shap_mlp)[-15:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_indices)), mean_abs_shap_mlp[top_indices], color='#f97316')
plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
plt.xlabel("Mean |SHAP Value| (Across Classes)", fontsize=11)
plt.title("MLP — Top 15 Influential Features via KernelExplainer", fontsize=13, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""),

        create_cell("markdown", """## 6. Single-Sample Waterfall Decomposition: Explaining a Critical Attack Flow

**What we do and why:**
When a critical intrusion occurs, security analysts need a forensic report explaining that specific flow.
We programmatically locate a true positive attack sample and generate a **Waterfall Plot** showing how each feature shifted the prediction from the base value to the final attack confidence.
"""),

        create_cell("code", """# Find a genuine True Positive attack sample (e.g., DoS Hulk, class 1)
y_pred_xgb = np.argmax(xgb_model.predict_proba(X_test_scaled[:500]), axis=1)
target_class = 1  # DoS Hulk

tp_indices = np.where((y_test[:500] == target_class) & (y_pred_xgb == target_class))[0]
sample_idx = tp_indices[0] if len(tp_indices) > 0 else 0

sample_vector = X_test_scaled[sample_idx]
print(f"Explaining flow #{sample_idx}: True Class='{CLASS_NAMES[y_test[sample_idx]]}', Predicted='{CLASS_NAMES[y_pred_xgb[sample_idx]]}'")

fig_waterfall = generate_waterfall_plot(
    explainer_xgb,
    sample_vector,
    feature_names,
    target_class_idx=target_class,
    max_display=12,
    save_path=MODELS_DIR / 'shap_waterfall.png',
)
plt.show()
"""),

        create_cell("markdown", """---
## Summary and Next Notebook

In this notebook:
- We computed exact Shapley values for the **XGBoost ensemble** using `TreeExplainer`.
- We computed model-agnostic feature attributions for the **Deep MLP** using `KernelExplainer`.
- We generated a **Waterfall Plot** (`models/shap_waterfall.png`) decomposing a specific attack detection down to individual features.

**Next up:** In **Notebook 07: Active Response Engine Simulation**, we will translate model confidence scores into graduated defensive countermeasures (LOG, ALERT, RATE_LIMIT, BLOCK) and simulate firewall rule creation!
""")
    ]
    nb.cells = cells
    return nb


def build_notebook_07() -> nbf.NotebookNode:
    """Notebook 07: Active Response Demo."""
    nb = nbf.v4.new_notebook()
    cells = [
        create_cell("markdown", """# Smart IDS v3 — Notebook 07: Automated Mitigation Engine & Active Response Simulation
Threat Tiers, IP Validation, Idempotent Dry-Run Blocking, and SOC Audit Logging

---

### What You Will Learn
1. **The Detection-to-Response Gap**: Why alerts without automated mitigation fail against automated attacks.
2. **Four-Tier Threat Severity Taxonomy**: Mapping confidence scores to risk-proportionate mitigation actions:
   - Low (0.00 – 0.40): LOG
   - Medium (0.40 – 0.70): ALERT
   - High (0.70 – 0.90): RATE_LIMIT
   - Critical (0.90 – 1.01): BLOCK
3. **Defensive Programming in Cybersecurity**: Validating IP addresses with `ipaddress` to prevent command injection.
4. **Firewall Rule Construction**: Understanding Linux `iptables` DROP rules (`-I INPUT 1 -s <IP> -j DROP`).
5. **Idempotency & Simulation**: Ensuring dry-run safety and preventing duplicate firewall degradation.
6. **Append-Only Audit Logging**: Tracking security responses in `ids_response_log.txt`.

---

### Prerequisites
- Completion of Notebooks 01 through 06.
- Understanding of the Single Canonical Module design (`src/smart_ids/response.py`).
"""),

        create_cell("markdown", """## 1. Security Concepts: Automated Mitigation & Command Injection Defense

> **Security Concept: What is iptables and why DROP instead of REJECT?**
> `iptables` is the Linux kernel's packet filtering firewall. The command:
> `iptables -I INPUT 1 -s 198.51.100.2 -j DROP`
> - `-I INPUT 1`: Inserts this rule at the very top (position 1) of the inbound packet chain, guaranteeing immediate precedence.
> - `-j DROP`: Silently discards any packet from this source IP with **zero reply**.
> 
> In contrast, `-j REJECT` sends an ICMP "port unreachable" error packet back to the sender. This tells an attacker that an active firewall is blocking them. `DROP` leaves them hanging, wasting their connection time and leaking zero information.

> **Security Concept: Defensive IP Validation against Command Injection**
> If an attacker submits a malicious flow with an IP like:
> `192.168.1.1; rm -rf /`
> and we interpolate that into a system shell command, the server executes arbitrary malware!
> 
> *The Defense:* We pass every candidate IP through Python's standard `ipaddress.ip_address()` validator. If it fails to parse as a valid IPv4 or IPv6 address, the block is aborted immediately with action code `BLOCK_FAILED`.
"""),

        create_cell("code", """import sys
from pathlib import Path

cur = Path.cwd()
root = cur if (cur / 'src').exists() else cur.parent
if str(root / 'src') not in sys.path:
    sys.path.insert(0, str(root / 'src'))

from smart_ids.config import (
    LOGS_FILE,
    THREAT_ACTIONS,
    THREAT_COLORS,
    THREAT_THRESHOLDS,
)
from smart_ids.response import (
    apply_active_response,
    get_threat_level,
    reset_blocked_ips,
    validate_ip,
)

# Reset blocked IP cache for this demonstration session
reset_blocked_ips()
print(f"Active Response Engine initialized.")
print(f"Audit log destination: {LOGS_FILE}")
"""),

        create_cell("markdown", """## 2. Testing the 4-Tier Severity Mapping

**What we do and why:**
We verify that confidence scores correctly map to the 4 severity tiers:
- $0.25 \to$ Low
- $0.55 \to$ Medium
- $0.80 \to$ High
- $0.95 \to$ Critical
- Exactly $1.00 \to$ Critical (ensured by the interval $[0.90, 1.01)$).
"""),

        create_cell("code", """test_confidences = [0.15, 0.40, 0.65, 0.70, 0.85, 0.90, 0.99, 1.00]

print("--- Confidence to Severity Mapping ---")
for conf in test_confidences:
    tier = get_threat_level(conf)
    action = THREAT_ACTIONS[tier]
    print(f"Confidence: {conf:4.2f} -> Tier: {tier:8s} -> Action: {action}")
"""),

        create_cell("markdown", """## 3. Testing Defensive IP Address Validation

**What we do and why:**
We test `validate_ip` against valid IPv4 and IPv6 addresses, as well as malformed injection payloads.
"""),

        create_cell("code", """test_ips = [
    "192.168.1.100",
    "10.0.0.1",
    "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
    "INVALID_IP",
    "192.168.1.1; cat /etc/passwd",
    "999.999.999.999",
    "",
]

print("--- IP Address Validation Checks ---")
for candidate in test_ips:
    valid = validate_ip(candidate)
    status = "VALID" if valid else "REJECTED (SAFE)"
    print(f"[{status:15s}] '{candidate}'")
"""),

        create_cell("markdown", """## 4. Simulating Graduated Incident Scenarios

Now let's simulate realistic detection scenarios across all 4 threat tiers.
"""),

        create_cell("code", """scenarios = [
    {"ip": "10.0.0.15", "class": "BENIGN", "conf": 0.22, "model": "SmartIDS_MLP"},
    {"ip": "198.51.100.42", "class": "PortScan", "conf": 0.58, "model": "SmartIDS_1DCNN"},
    {"ip": "203.0.113.88", "class": "SSH-Patator", "conf": 0.82, "model": "XGBoost"},
    {"ip": "192.0.2.1", "class": "DoS Hulk", "conf": 0.96, "model": "SmartIDS_MLP"},
    {"ip": "INVALID_INJECTION_IP", "class": "DDoS", "conf": 0.98, "model": "SmartIDS_1DCNN"},
]

print("--- Simulated Incident Response Execution ---")
for s in scenarios:
    res = apply_active_response(
        src_ip=s["ip"],
        attack_class=s["class"],
        confidence=s["conf"],
        model_name=s["model"],
        dry_run=True,
    )
    print(f"\\nIncident: [{res['threat_level']}] from {res['src_ip']}")
    print(f"  Class: {res['attack_class']} (Confidence: {res['confidence']:.2f})")
    print(f"  Decision: {res['action']}")
    print(f"  Detail:   {res['action_detail']}")
"""),

        create_cell("markdown", """## 5. Idempotency Gate Verification

> **Security Concept: What is Idempotency in Firewall Management?**
> If an attacker bombards a server with 50,000 requests in 2 seconds, the IDS will detect 50,000 malicious flows from the same IP.
> If the engine adds a new `iptables` rule 50,000 times, the firewall's memory and packet traversal performance will collapse!
> 
> *The Solution:* The engine maintains an in-memory set `BLOCKED_IPS`. If an IP is already blocked, subsequent calls immediately return without issuing redundant rules.
"""),

        create_cell("code", """attacker_ip = "192.0.2.1"

# Second call for the same IP
repeat_res = apply_active_response(
    src_ip=attacker_ip,
    attack_class="DoS Hulk",
    confidence=0.99,
    dry_run=True,
)

print(f"Repeat Block Attempt for {attacker_ip}:")
print(f"  Action: {repeat_res['action']}")
print(f"  Detail: {repeat_res['action_detail']}")
"""),

        create_cell("markdown", """## 6. Inspecting the Audit Trail (`ids_response_log.txt`)

Let's read the last 10 entries from our durable, append-only security log file.
"""),

        create_cell("code", """if LOGS_FILE.exists():
    with open(LOGS_FILE, 'r', encoding='utf-8') as f:
        log_lines = f.readlines()
    print(f"--- Last {min(8, len(log_lines))} Log Entries from {LOGS_FILE.name} ---")
    for line in log_lines[-8:]:
        print(line.strip())
else:
    print("Log file will be created upon first event.")
"""),

        create_cell("markdown", """---
## Summary & Research Series Conclusion

Congratulations! Across the 7 notebooks in the **Smart IDS v3** curriculum, you have:
1. Explored network flows and the CICIDS2017 benchmark dataset.
2. Cleaned zero-division infinities and built a leak-free scaling + SMOTE pipeline.
3. Trained three supervised architectures: XGBoost, Deep MLP, and 1D-CNN.
4. Built an unsupervised Autoencoder for zero-day anomaly detection.
5. Benchmarked microsecond latency, normalized confusion matrices, and ROC curves.
6. Decoded model predictions using SHAP game-theoretic feature attribution.
7. Deployed an active response engine with injection-safe, idempotent firewall simulation.

All trained models and metadata are saved in `models/` and ready for the interactive **Streamlit SOC Dashboard** (`app/streamlit_app.py`)!
""")
    ]
    nb.cells = cells
    return nb


def main():
    """Generate all 7 notebooks."""
    notebooks = [
        ("01_intro_and_security_basics.ipynb", build_notebook_01()),
        ("02_data_preprocessing.ipynb", build_notebook_02()),
        ("03_supervised_models.ipynb", build_notebook_03()),
        ("04_autoencoder_zero_day.ipynb", build_notebook_04()),
        ("05_evaluation_and_comparison.ipynb", build_notebook_05()),
        ("06_explainability_shap.ipynb", build_notebook_06()),
        ("07_active_response_demo.ipynb", build_notebook_07()),
    ]

    for fname, nb in notebooks:
        target_path = NOTEBOOKS_DIR / fname
        with open(target_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generated notebook: {target_path}")


if __name__ == "__main__":
    main()
