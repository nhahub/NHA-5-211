"""Centralized configuration for Smart IDS v3.

Defines reproducible seeds, file paths, taxonomy constants, model
hyperparameters, and the Active Response threat severity mapping.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress verbose warnings across execution environments
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# --- Reproducibility Seed ---
SEED: int = 42

# --- File System Paths ---
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
APP_DIR: Path = BASE_DIR / "app"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
LOGS_FILE: Path = BASE_DIR / "ids_response_log.txt"

# Ensure essential output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Attack Taxonomy (8 Classes) ---
CLASS_NAMES: List[str] = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "Web Attack",
]

BENIGN_CLASS_NAME: str = "BENIGN"

# Representative class distribution weights (matching empirical CICIDS2017 skew)
CLASS_WEIGHTS: Dict[str, float] = {
    "BENIGN": 0.83,
    "DoS Hulk": 0.06,
    "PortScan": 0.04,
    "DDoS": 0.03,
    "DoS GoldenEye": 0.01,
    "FTP-Patator": 0.01,
    "SSH-Patator": 0.01,
    "Web Attack": 0.01,
}

# --- Data Cleaning and Feature Filtering ---
NAN_DROP_THRESHOLD: float = 0.50  # Drop column if > 50% NaN
OUTLIER_SIGMA_MULTIPLE: float = 5.0  # Clip values outside mean +/- 5 * std

# Explicit structural identifier keywords to drop to avoid target leakage
IDENTIFIER_KEYWORDS: List[str] = ["ip", "timestamp", "flow id", "src", "dst"]

# Exact column names to exclude if present (defensive explicit exclusion list)
EXPLICIT_EXCLUDE_COLS: List[str] = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
    "External IP",
]

# --- Train / Validation / Test Splitting ---
TEST_SIZE: float = 0.15
VAL_SIZE_OF_REMAINDER: float = 0.15  # 15% of (1 - 0.15) = 12.75% of total dataset

# --- Active Response Engine Threat Severity Mapping ---
# Closed-lower, open-upper intervals [lo, hi). Upper bound is 1.01 to safely include 1.0.
THREAT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Low": (0.00, 0.40),
    "Medium": (0.40, 0.70),
    "High": (0.70, 0.90),
    "Critical": (0.90, 1.01),
}

THREAT_ACTIONS: Dict[str, str] = {
    "Low": "LOG",
    "Medium": "ALERT",
    "High": "RATE_LIMIT",
    "Critical": "BLOCK",
}

THREAT_COLORS: Dict[str, str] = {
    "Low": "#10b981",       # Emerald green
    "Medium": "#f59e0b",    # Amber
    "High": "#f97316",      # Vivid orange
    "Critical": "#ef4444",  # Crimson red
}

THREAT_ICONS: Dict[str, str] = {
    "Low": "🟢",
    "Medium": "🟡",
    "High": "🟠",
    "Critical": "🔴",
}

# Rate limit ceiling string
RATE_LIMIT_CEILING: str = "100 pps"

# --- Supervised Model Hyperparameters ---
# MLP
MLP_UNITS: List[int] = [512, 256, 128, 64]
MLP_L2: float = 1e-4
MLP_LEAKY_ALPHA: float = 0.1
MLP_DROPOUT: float = 0.4
MLP_LR: float = 1e-3
MLP_CLIPNORM: float = 1.0
MLP_BATCH_SIZE: int = 512
MLP_MAX_EPOCHS: int = 50

# 1D-CNN
CNN_FILTERS_BLOCK1: int = 64
CNN_FILTERS_BLOCK2: int = 128
CNN_KERNEL_SIZE: int = 3
CNN_L2: float = 1e-4
CNN_HEAD_UNITS: int = 128
CNN_LR: float = 1e-3
CNN_BATCH_SIZE: int = 512
CNN_MAX_EPOCHS: int = 50

# Autoencoder (Zero-Day Anomaly Detector)
AE_ENCODER_UNITS: List[int] = [256, 128, 64]
AE_LATENT_DIM: int = 32
AE_DECODER_UNITS: List[int] = [64, 128, 256]
AE_LR: float = 5e-4
AE_BATCH_SIZE: int = 512
AE_MAX_EPOCHS: int = 50
AE_PERCENTILE: float = 95.0

# XGBoost Baseline
XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 6
XGB_LEARNING_RATE: float = 0.1
XGB_SUBSAMPLE: float = 0.8
XGB_COLSAMPLE_BYTREE: float = 0.8
XGB_MIN_CHILD_WEIGHT: int = 5
XGB_GAMMA: float = 0.1
XGB_REG_ALPHA: float = 0.01
XGB_REG_LAMBDA: float = 1.0
XGB_TREE_METHOD: str = "hist"
XGB_EARLY_STOPPING_ROUNDS: int = 20

# Callback Patience settings
EARLY_STOPPING_PATIENCE_SUPERVISED: int = 8
EARLY_STOPPING_PATIENCE_UNSUPERVISED: int = 10
REDUCE_LR_PATIENCE: int = 4
REDUCE_LR_FACTOR: float = 0.5
MIN_LR: float = 1e-6

# Latency benchmarking
LATENCY_BENCHMARK_RUNS: int = 100
