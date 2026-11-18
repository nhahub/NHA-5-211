# Dataset Directory (`data/`)

## Subdirectories
- `raw/` — Place the original **CICIDS2017** daily CSV files here (from `MachineLearningCSV.zip` or `TrafficLabelling` CSVs).
- `processed/` — Stores preprocessed train/validation/test arrays, metadata, and the preprocessing audit report (`preprocessing_audit.json`).
- `synthetic/` — Stores explicitly labeled synthetic demonstration datasets when the original CICIDS2017 dataset is absent.

## Obtaining Genuine CICIDS2017 Files
1. Download the official CICIDS2017 dataset from the Canadian Institute for Cybersecurity (University of New Brunswick):
   https://www.unb.ca/cic/datasets/ids-2017.html
2. Extract the 8 CSV exports directly into `data/raw/`:
   - `Monday-WorkingHours.pcap_ISCX.csv`
   - `Tuesday-WorkingHours.pcap_ISCX.csv`
   - `Wednesday-workingHours.pcap_ISCX.csv`
   - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
   - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
   - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
3. Run `python scripts/train.py` (it automatically detects CSVs in `data/raw/`).

## Synthetic Demonstration Mode
If `data/raw/` contains no CSV files, `src/data/loader.py` can generate an explicitly labeled **synthetic demonstration corpus** (`data/synthetic/cicids2017_synthetic_demo.csv`) mirroring the 77-feature CICFlowMeter schema, 8-class imbalance weights (`0.83 / 0.06 / 0.04 / 0.03 / 0.01 x 4`), duplicates, `inf`, and `NaN` artifacts.

> **Important Scientific Notice:** Metrics produced in Synthetic Demonstration Mode measure pipeline correctness and model behavior on synthetic distributions and must NEVER be presented as genuine CICIDS2017 experimental results.
