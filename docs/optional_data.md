# Optional CICIDS2017 Datasets (Real Timestamps & Corrected Labels)

This repository runs the main benchmark against the 8 `MachineLearningCVE` CSV files in `data/raw/`. Two optional upstream extensions are configured via [`configs/config.yaml`](../configs/config.yaml) without code changes:

1. **Real Capture Timestamps for `time_block` (`TrafficLabelling_.zip` / `GeneratedLabelledFlows`) — VERIFIED for `time_block` on `timestamp-eval`**
   - **Why optional:** Standard `MachineLearningCVE` CSVs in `data/raw/` omit the `Timestamp` column. When `data/TrafficLabelling/*.csv` is present, chronological `time_block` evaluation aligns real `Timestamp` values (`2,830,471 / 2,830,743` = `99.9904%` matched via `flow_telemetry_signature`, `272` unmatched; only missing checkouts fall back to CSV row order).
   - **Where to download:** Download **`GeneratedLabelledFlows` / `TrafficLabelling_.zip`** from the UNB Canadian Institute for Cybersecurity CICIDS2017 page: <https://www.unb.ca/cic/datasets/ids-2017.html>.
   - **Where to place:** Extract all 8 daily CSVs into `data/TrafficLabelling/` (matching `data.timestamp_source_dir: "data/TrafficLabelling"`):
     - `Monday-WorkingHours.pcap_ISCX.csv`
     - `Tuesday-WorkingHours.pcap_ISCX.csv`
     - `Wednesday-workingHours.pcap_ISCX.csv`
     - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
     - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
     - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
     - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
     - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
   - **How alignment works:** [`align_real_timestamps`](../src/data/loader.py) aligns `Timestamp` onto the main flows via shared `Flow ID` + 5-tuple (`Source IP`, `Destination IP`, `Source Port`, `Destination Port` + occurrence index) or deterministic session telemetry signatures (`Source_Session_Day`, `Destination Port`, `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`, `Total Length of Fwd Packets`, `Total Length of Bwd Packets`, `Flow IAT Mean`, normalized `Label`, occurrence index), logs exact matched/unmatched counts, and excludes `Timestamp` from model features.

2. **Engelen et al. (2021) Corrected Labels (`data/engelen2021_label_corrections.csv`) — NOT APPLIED (absent locally)**
   - **Why optional:** Original CICIDS2017 labels contain documented flow-construction and ground-truth errors affecting classes such as DoS/Heartbleed/Bot (Engelen et al., *IEEE S&P Workshops*, 2021).
   - **Where to download:** Distrinet Research Group (KU Leuven), CICIDS2017 Troubleshooting (`CNS2021`): <https://intrusion-detection.distrinet-research.be/CNS2021/>.
   - **Where to place:** Save the corrected CSV or Parquet file at `data/engelen2021_label_corrections.csv` (or point `data.label_correction_file` in [`configs/config.yaml`](../configs/config.yaml) to your local file).

---

## Quick Verification & Affected Re-Evaluation Command

Check which files are currently present or missing:

```bash
python scripts/check_optional_data.py
```

Once `data/TrafficLabelling/*.csv` (and optionally `data/engelen2021_label_corrections.csv`) are in place, re-run **only** the `time_block` evaluation without retraining any saved primary models:

```bash
python scripts/evaluate.py --time-block-only
```

- `python scripts/evaluate.py --time-block-only` reloads the raw corpus, aligns real `Timestamp` values (and applies Engelen corrections within the `time_block` protocol if present), updates `time_block_protocol` inside [`reports/evaluation_summary.json`](../reports/evaluation_summary.json), and regenerates the markdown reports without touching [`models/`](../models/).
- If you also want the saved primary `random`-split production models in [`models/`](../models/) retrained on Engelen corrected labels, a full retrain is required:

```bash
python scripts/train.py --run-eval
```
