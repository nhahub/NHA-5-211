"""Check presence of optional CICIDS2017 TrafficLabelling and Engelen label-correction data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, get_app_config, to_repo_relative_path

EXPECTED_TRAFFICLABELLING_FILES: tuple[tuple[str, str], ...] = (
    ("Monday-WorkingHours.pcap_ISCX.csv", "monday"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "tuesday"),
    ("Wednesday-workingHours.pcap_ISCX.csv", "wednesday"),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "webattack"),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "infilt"),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", "friday-workinghours-morning"),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "portscan"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "ddos"),
)

TRAFFICLABELLING_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
ENGELEN_URL = "https://intrusion-detection.distrinet-research.be/CNS2021/"


def _resolve_under(raw_spec: str | None, root: Path) -> Path | None:
    if not raw_spec or not str(raw_spec).strip():
        return None
    p = Path(str(raw_spec).strip())
    return p if p.is_absolute() else (root / p).resolve()


def _csv_has_timestamp_header(csv_path: Path) -> bool:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with csv_path.open("r", encoding=enc, errors="ignore") as fh:
                header_line = fh.readline()
            cols = [c.strip().lower() for c in header_line.split(",")]
            return "timestamp" in cols
        except OSError:
            return False
    return False


def check_optional_data_status(
    app_cfg: AppConfig | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Inspect optional CICIDS2017 timestamp CSVs and Engelen et al. (2021) label corrections."""
    cfg = app_cfg or get_app_config()

    ts_spec = cfg.data.timestamp_source_dir or "data/TrafficLabelling"
    ts_dir = _resolve_under(ts_spec, root)
    ts_exists = bool(ts_dir and ts_dir.exists() and ts_dir.is_dir())
    found_csvs = sorted(p for p in ts_dir.rglob("*.csv") if p.is_file()) if ts_exists and ts_dir else []
    found_names = [p.name for p in found_csvs]
    lower_names = [n.lower() for n in found_names]

    present_sessions: list[str] = []
    missing_sessions: list[str] = []
    for canonical_name, token in EXPECTED_TRAFFICLABELLING_FILES:
        matched = next(
            (name for name, low in zip(found_names, lower_names, strict=False) if token in low), None
        )
        if matched is not None:
            present_sessions.append(matched)
        else:
            missing_sessions.append(canonical_name)

    files_with_timestamp = [p.name for p in found_csvs if _csv_has_timestamp_header(p)]

    corr_spec = cfg.data.label_correction_file or "data/engelen2021_label_corrections.csv"
    corr_path = _resolve_under(corr_spec, root)
    corr_exists = bool(corr_path and corr_path.exists() and corr_path.is_file())

    return {
        "timestamp_source": {
            "configured_path": to_repo_relative_path(ts_dir, root=root) if ts_dir else ts_spec,
            "directory_exists": ts_exists,
            "csv_count": len(found_csvs),
            "present_files": found_names,
            "present_canonical_sessions": present_sessions,
            "missing_canonical_sessions": missing_sessions,
            "files_with_timestamp_column": files_with_timestamp,
            "ready": bool(found_csvs and files_with_timestamp),
        },
        "label_correction": {
            "configured_path": to_repo_relative_path(corr_path, root=root) if corr_path else corr_spec,
            "file_exists": corr_exists,
            "ready": corr_exists,
        },
        "all_optional_present": bool(found_csvs and files_with_timestamp and corr_exists),
    }


def format_optional_data_report(status: dict[str, Any]) -> str:
    """Render a concise human-readable diagnostic and rerun command report."""
    ts = status["timestamp_source"]
    lc = status["label_correction"]
    ts_state = "PRESENT" if ts["ready"] else "MISSING"
    lc_state = "PRESENT" if lc["ready"] else "MISSING"

    lines = [
        "SMART IDS v3 — Optional Data Status",
        "===================================",
        "",
        f"1. Real-Timestamp CSVs (TrafficLabelling / GeneratedLabelledFlows): {ts_state}",
        f"   Configured folder : {ts['configured_path']}",
        f"   Directory exists  : {ts['directory_exists']}",
        f"   CSV files found   : {ts['csv_count']}",
        f"   Files w/ Timestamp: {len(ts['files_with_timestamp_column'])}",
        f"   Present sessions  : {', '.join(ts['present_canonical_sessions']) if ts['present_canonical_sessions'] else '(none)'}",
        f"   Missing sessions  : {', '.join(ts['missing_canonical_sessions']) if ts['missing_canonical_sessions'] else '(none)'}",
        f"   Download source   : CICIDS2017 GeneratedLabelledFlows / TrafficLabelling_.zip ({TRAFFICLABELLING_URL})",
        f"   Target placement  : Extract the CSVs directly into {ts['configured_path']}/",
        "",
        f"2. Engelen et al. (2021) Corrected Labels: {lc_state}",
        f"   Configured file   : {lc['configured_path']}",
        f"   File exists       : {lc['file_exists']}",
        f"   Download source   : Distrinet KU Leuven CNS2021 CICIDS2017 Troubleshooting ({ENGELEN_URL})",
        f"   Target placement  : Save CSV/Parquet at {lc['configured_path']} (or update data.label_correction_file in configs/config.yaml)",
        "",
        "Affected Re-Evaluation Commands (NO full retrain for time_block):",
        "  - Re-run ONLY time_block evaluation (applies real Timestamps + Engelen corrections to time_block):",
        "      python scripts/evaluate.py --time-block-only",
        "  - Optional (only if you also want primary saved `random` models retrained on Engelen corrected labels):",
        "      python scripts/train.py --run-eval",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check optional TrafficLabelling timestamp CSVs and Engelen label corrections."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if either optional dataset source is missing.",
    )
    args = parser.parse_args()
    status = check_optional_data_status()
    print(format_optional_data_report(status))
    if args.strict and not status["all_optional_present"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
