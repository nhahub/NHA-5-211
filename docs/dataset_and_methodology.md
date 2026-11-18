# SMART IDS v3 — Methodology, Compatibility & Scientific Notes

## 1. Technology Stack Selection & Compatibility Constraints

At development time (November 2026), Python **3.14.5** and **3.13.12** are installed on the host workstation. Per the project compatibility mandate:
- **Target evaluated:** Python 3.14.
- **Blocking constraint verified:** The latest stable PyPI release of TensorFlow is **`tensorflow==2.21.0`** (March 2026), which supports Python **3.10–3.13** (`2.22.0` remains at `2.22.0rc0` pre-release).
- **Selected Python:** **`3.13.12`** — the newest stable Python release supported cleanly across the entire required stack (`tensorflow==2.21.0`, `keras==3.15.1`, `scikit-learn==1.9.1`, `xgboost==3.4.1`, `shap==0.54.1`, `pandas==3.1.1`, `numpy==2.5.5`, `imbalanced-learn==0.14.2`, `optuna==5.1.0`, `streamlit==1.67.0`, `plotly==7.1.0`, `pydantic==2.14.3`, `pytest==9.2.0`, `ruff==0.17.5`).
- **Windows GPU/CPU Acceleration:** Native TensorFlow GPU support on Windows was discontinued after TF 2.10 (requiring WSL2 Linux for TF GPU). Consequently, `src/utils/reproducibility.py` configures **TensorFlow on CPU** while verifying and enabling **NVIDIA CUDA (`device="cuda"`) for XGBoost** on the host **NVIDIA GeForce RTX 3050 6GB GPU**, with automatic CPU fallback whenever CUDA is unavailable.

---

## 2. Methodological Improvements Over the Baseline Report

1. **Elimination of Pre-Split Data Leakage (Report Ch. 3):**
   - *Original Report:* Applied median imputation and 5-sigma clipping prior to train/val/test splitting.
   - *SMART IDS v3 Fix:* Performs deterministic stateless cleaning first (`strip_columns`, Web Attack label merging, deduplication, structural ID drop, `Inf -> NaN`, numeric coercion), then executes the **stratified split (`72.25% / 12.75% / 15%`) BEFORE fitting** `HighMissingColumnDropper`, `MedianImputerTransformer`, `SigmaOutlierClipper`, and `StandardScaler`. Validation and test splits remain untouched and naturally imbalanced.
2. **Dynamic Class Label Alignment (Report Ch. 6.2):**
   - Eliminates hardcoded prevalence-ordered `CLASS_NAMES` in favor of reading `label_encoder.classes_` directly across training, evaluation, inference, and dashboard layers.
3. **Experimental Validation of Imbalance Strategy (SMOTE vs. Class-Weighting):**
   - Instead of blindly assuming SMOTE outperforms class weighting, `compare_smote_vs_class_weight` benchmarks unweighted, inverse-frequency class-weighted, and SMOTE-resampled training on the naturally imbalanced validation split.
4. **Canonical Shared Response Engine & Benign Safety Guard (Report Ch. 7):**
   - Consolidates duplicated notebook/dashboard logic into `src/response/response_engine.py`, and fixes a critical safety flaw where high-confidence `BENIGN` classifications would trigger `Critical`/`BLOCK` under naive confidence thresholding.
5. **Architectural Honesty on 1D-CNN & Zero-Day Autoencoder:**
   - Explicitly documents that CICFlowMeter tabular feature order is not a genuine temporal signal, evaluating 1D-CNN empirically against MLP and XGBoost.
   - Evaluates the benign-trained Autoencoder across percentile thresholds (`90, 95, 97, 99`) and an explicit **Leave-One-Attack-Out novelty protocol** rather than claiming unrestricted zero-day immunity from single-split metrics.

---

## 3. Dataset Setup Workflow (CICIDS2017 vs. Synthetic Demonstration Mode)

1. **Genuine CICIDS2017 Execution:**
   - Download `MachineLearningCSV.zip` or `TrafficLabelling` CSVs from UNB CIC: https://www.unb.ca/cic/datasets/ids-2017.html
   - Place the 8 daily CSV files into `data/raw/`.
   - Run `python scripts/train.py --run-eval`.
2. **Synthetic Demonstration Mode (Automatic Fallback when `data/raw/` is empty):**
   - `src/data/loader.py` generates a schema-faithful 77-feature corpus (`data/synthetic/cicids2017_synthetic_demo.csv`) mirroring CICIDS2017 class imbalance weights (`0.83, 0.06, 0.04, 0.03, 0.01 x 4`), duplicates, `Inf`, `NaN`, and extreme rate spikes.
   - Every dashboard banner, manifest, and report explicitly labels synthetic runs so they are never confused with genuine CICIDS2017 benchmark results.
