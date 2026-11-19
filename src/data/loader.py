"""CICIDS2017 CSV auto-discovery, loading, and explicit synthetic fallback generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, DataConfig
from src.data.cleaner import CHRON_ORDER_COLUMN, SOURCE_FILE_COLUMN, normalize_label_text
from src.utils.logging import get_logger

logger = get_logger("smart_ids.data.loader")

# Exact 77 non-redundant numeric features from CICFlowMeter / CICIDS2017
# (Destination Port + 76 flow statistics, excluding duplicate 'Fwd Header Length.1' and IDs).
CICIDS2017_FEATURES: list[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


@dataclass
class DatasetSourceInfo:
    """Provenance record distinguishing real CICIDS2017 data from synthetic demos."""

    source_type: str  # "cicids2017_raw" or "synthetic_demonstration"
    is_synthetic: bool
    files_loaded: list[str]
    raw_row_count: int
    sampled_row_count: int
    warning_message: str | None = None
    timestamp_alignment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_csv_files(raw_dir: Path) -> list[Path]:
    """Discover all CSV files inside raw_dir in deterministic alphabetical order."""
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*.csv") if p.is_file())


def _read_single_csv(path: Path) -> pd.DataFrame:
    """Read a CICIDS2017 CSV resiliently across UTF-8 / Latin-1 encodings."""
    last_exc: Exception | None = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            df.columns = df.columns.astype(str).str.strip()
            return df
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise ValueError(f"Unable to decode CSV file {path}: {last_exc}")


def stratified_downsample(
    df: pd.DataFrame,
    label_col: str,
    max_rows: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample a large dataframe proportionally while keeping rare classes intact."""
    if len(df) <= max_rows:
        return df

    counts = df[label_col].value_counts()
    fractions = counts / len(df)
    target_per_class = (fractions * max_rows).round().astype(int)
    # Preserve at least up to all available rows for small minority classes
    target_per_class = np.minimum(counts, np.maximum(target_per_class, np.minimum(counts, 50)))

    sampled_parts = []
    for cls_name, n_take in target_per_class.items():
        subset = df[df[label_col] == cls_name]
        if len(subset) > int(n_take):
            sampled_parts.append(subset.sample(n=int(n_take), random_state=seed))
        else:
            sampled_parts.append(subset)

    combined = pd.concat(sampled_parts, ignore_index=True)
    if len(combined) > max_rows:
        # Trim majority class slightly if floor raised total above max_rows
        excess = len(combined) - max_rows
        majority_label = counts.index[0]
        maj_idx = combined.index[combined[label_col] == majority_label]
        if len(maj_idx) > excess + 100:
            rng = np.random.default_rng(seed)
            drop_idx = rng.choice(maj_idx, size=excess, replace=False)
            combined = combined.drop(index=drop_idx).reset_index(drop=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_synthetic_cicids2017(
    n_samples: int = 60000,
    seed: int = 42,
    class_weights: dict[str, float] | None = None,
    duplicate_fraction: float = 0.004,
    inf_fraction: float = 0.008,
    nan_fraction: float = 0.003,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Generate an explicitly labeled, schema-faithful synthetic CICIDS2017 demo dataset.

    Each of the 8 classes receives realistic, partially overlapping statistical
    telemetry patterns (ports, packet sizes, IATs, TCP flag ratios, active/idle times)
    plus injected CICFlowMeter pathologies (Inf, NaN, duplicates, rate spikes).
    """
    rng = np.random.default_rng(seed)
    weights = class_weights or {
        "BENIGN": 0.83,
        "DoS Hulk": 0.06,
        "PortScan": 0.04,
        "DDoS": 0.03,
        "DoS GoldenEye": 0.01,
        "FTP-Patator": 0.01,
        "SSH-Patator": 0.01,
        "Web Attack": 0.01,
    }
    classes = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs /= probs.sum()

    labels = rng.choice(classes, size=n_samples, p=probs)
    n_feat = len(CICIDS2017_FEATURES)

    # Build baseline multivariate matrix with realistic positive heavy-tailed scale
    data = np.zeros((n_samples, n_feat), dtype=np.float64)
    feat_idx = {name: idx for idx, name in enumerate(CICIDS2017_FEATURES)}

    # Profile definitions: (port_choices, dur_mu, fwd_pkt_mu, bwd_pkt_mu, pkt_len_mu, syn_mu, ack_mu, psh_mu, fin_mu, active_mu, idle_mu)
    profiles: dict[
        str, tuple[list[int], float, float, float, float, float, float, float, float, float, float]
    ] = {
        "BENIGN": ([80, 443, 53, 123, 445, 22, 8080], 11.2, 2.0, 2.1, 5.1, 0.4, 2.5, 0.9, 0.3, 9.2, 12.1),
        "DoS Hulk": ([80, 443], 13.8, 2.3, 1.8, 6.2, 1.3, 2.8, 2.0, 1.5, 11.5, 14.5),
        "PortScan": (
            [21, 22, 23, 25, 80, 110, 139, 443, 445, 3389, 8080],
            5.4,
            0.5,
            0.4,
            3.6,
            1.6,
            0.6,
            0.2,
            0.1,
            4.0,
            5.0,
        ),
        "DDoS": ([80, 443], 13.0, 2.4, 1.2, 6.4, 1.1, 3.2, 1.0, 0.8, 10.8, 13.9),
        "DoS GoldenEye": ([80, 443], 14.4, 2.1, 1.5, 5.9, 1.0, 2.4, 1.7, 0.6, 12.0, 15.2),
        "FTP-Patator": ([21], 10.6, 2.5, 2.6, 4.5, 1.0, 3.0, 2.4, 1.0, 8.8, 10.2),
        "SSH-Patator": ([22], 12.0, 3.1, 3.2, 5.0, 1.1, 3.6, 2.8, 1.1, 9.5, 11.1),
        "Web Attack": ([80, 443, 8080], 12.3, 2.6, 2.3, 5.5, 1.0, 3.0, 2.2, 0.8, 10.0, 12.4),
    }

    for cls_name, profile in profiles.items():
        mask = labels == cls_name
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        (
            ports,
            dur_mu,
            fwd_mu,
            bwd_mu,
            len_mu,
            syn_mu,
            ack_mu,
            psh_mu,
            fin_mu,
            act_mu,
            idle_mu,
        ) = profile

        # Introduce realistic multi-modal sub-clusters for benign traffic and noise
        if cls_name == "PortScan":
            data[mask, feat_idx["Destination Port"]] = rng.integers(1, 10000, size=cnt)
        else:
            data[mask, feat_idx["Destination Port"]] = rng.choice(ports, size=cnt)

        flow_dur = np.exp(rng.normal(dur_mu, 1.35, size=cnt))
        fwd_pkts = np.maximum(1.0, np.round(np.exp(rng.normal(fwd_mu, 0.85, size=cnt))))
        bwd_pkts = np.maximum(0.0, np.round(np.exp(rng.normal(bwd_mu, 0.90, size=cnt))))
        pkt_mean = np.exp(rng.normal(len_mu, 0.75, size=cnt))
        pkt_std = pkt_mean * rng.uniform(0.15, 0.95, size=cnt)
        fwd_len_tot = fwd_pkts * pkt_mean * rng.uniform(0.6, 1.4, size=cnt)
        bwd_len_tot = bwd_pkts * pkt_mean * rng.uniform(0.5, 1.6, size=cnt)

        data[mask, feat_idx["Flow Duration"]] = flow_dur
        data[mask, feat_idx["Total Fwd Packets"]] = fwd_pkts
        data[mask, feat_idx["Total Backward Packets"]] = bwd_pkts
        data[mask, feat_idx["Total Length of Fwd Packets"]] = fwd_len_tot
        data[mask, feat_idx["Total Length of Bwd Packets"]] = bwd_len_tot

        data[mask, feat_idx["Fwd Packet Length Max"]] = pkt_mean * rng.uniform(1.2, 2.8, size=cnt)
        data[mask, feat_idx["Fwd Packet Length Min"]] = pkt_mean * rng.uniform(0.05, 0.45, size=cnt)
        data[mask, feat_idx["Fwd Packet Length Mean"]] = pkt_mean * rng.uniform(0.8, 1.2, size=cnt)
        data[mask, feat_idx["Fwd Packet Length Std"]] = pkt_std
        data[mask, feat_idx["Bwd Packet Length Max"]] = pkt_mean * rng.uniform(1.1, 3.2, size=cnt)
        data[mask, feat_idx["Bwd Packet Length Min"]] = pkt_mean * rng.uniform(0.02, 0.4, size=cnt)
        data[mask, feat_idx["Bwd Packet Length Mean"]] = pkt_mean * rng.uniform(0.7, 1.4, size=cnt)
        data[mask, feat_idx["Bwd Packet Length Std"]] = pkt_std * rng.uniform(0.8, 1.3, size=cnt)

        dur_sec = np.maximum(flow_dur / 1e6, 1e-5)
        data[mask, feat_idx["Flow Bytes/s"]] = (fwd_len_tot + bwd_len_tot) / dur_sec
        data[mask, feat_idx["Flow Packets/s"]] = (fwd_pkts + bwd_pkts) / dur_sec
        data[mask, feat_idx["Fwd Packets/s"]] = fwd_pkts / dur_sec
        data[mask, feat_idx["Bwd Packets/s"]] = bwd_pkts / dur_sec

        iat_mean = flow_dur / np.maximum(fwd_pkts + bwd_pkts - 1.0, 1.0)
        data[mask, feat_idx["Flow IAT Mean"]] = iat_mean
        data[mask, feat_idx["Flow IAT Std"]] = iat_mean * rng.uniform(0.2, 1.6, size=cnt)
        data[mask, feat_idx["Flow IAT Max"]] = iat_mean * rng.uniform(1.5, 4.5, size=cnt)
        data[mask, feat_idx["Flow IAT Min"]] = iat_mean * rng.uniform(0.01, 0.3, size=cnt)

        data[mask, feat_idx["Fwd IAT Total"]] = flow_dur * rng.uniform(0.6, 1.0, size=cnt)
        data[mask, feat_idx["Fwd IAT Mean"]] = iat_mean * rng.uniform(0.7, 1.3, size=cnt)
        data[mask, feat_idx["Fwd IAT Std"]] = iat_mean * rng.uniform(0.2, 1.5, size=cnt)
        data[mask, feat_idx["Fwd IAT Max"]] = iat_mean * rng.uniform(1.2, 4.0, size=cnt)
        data[mask, feat_idx["Fwd IAT Min"]] = iat_mean * rng.uniform(0.01, 0.35, size=cnt)

        data[mask, feat_idx["Bwd IAT Total"]] = flow_dur * rng.uniform(0.3, 0.95, size=cnt)
        data[mask, feat_idx["Bwd IAT Mean"]] = iat_mean * rng.uniform(0.5, 1.4, size=cnt)
        data[mask, feat_idx["Bwd IAT Std"]] = iat_mean * rng.uniform(0.1, 1.4, size=cnt)
        data[mask, feat_idx["Bwd IAT Max"]] = iat_mean * rng.uniform(1.0, 3.8, size=cnt)
        data[mask, feat_idx["Bwd IAT Min"]] = iat_mean * rng.uniform(0.01, 0.3, size=cnt)

        data[mask, feat_idx["Fwd PSH Flags"]] = rng.binomial(1, min(0.95, psh_mu / 3.0), size=cnt)
        data[mask, feat_idx["Bwd PSH Flags"]] = 0.0
        data[mask, feat_idx["Fwd URG Flags"]] = rng.binomial(1, 0.01, size=cnt)
        data[mask, feat_idx["Bwd URG Flags"]] = 0.0
        data[mask, feat_idx["Fwd Header Length"]] = fwd_pkts * 20.0 + rng.normal(0, 4.0, size=cnt)
        data[mask, feat_idx["Bwd Header Length"]] = bwd_pkts * 20.0 + rng.normal(0, 4.0, size=cnt)

        data[mask, feat_idx["Min Packet Length"]] = pkt_mean * rng.uniform(0.05, 0.4, size=cnt)
        data[mask, feat_idx["Max Packet Length"]] = pkt_mean * rng.uniform(1.3, 3.5, size=cnt)
        data[mask, feat_idx["Packet Length Mean"]] = pkt_mean
        data[mask, feat_idx["Packet Length Std"]] = pkt_std
        data[mask, feat_idx["Packet Length Variance"]] = np.square(pkt_std)

        data[mask, feat_idx["FIN Flag Count"]] = np.round(np.clip(rng.normal(fin_mu, 0.5, size=cnt), 0, 4))
        data[mask, feat_idx["SYN Flag Count"]] = np.round(np.clip(rng.normal(syn_mu, 0.6, size=cnt), 0, 5))
        data[mask, feat_idx["RST Flag Count"]] = rng.binomial(
            1, 0.15 if cls_name == "PortScan" else 0.05, size=cnt
        )
        data[mask, feat_idx["PSH Flag Count"]] = np.round(np.clip(rng.normal(psh_mu, 0.7, size=cnt), 0, 6))
        data[mask, feat_idx["ACK Flag Count"]] = np.round(np.clip(rng.normal(ack_mu, 0.8, size=cnt), 0, 12))
        data[mask, feat_idx["URG Flag Count"]] = rng.binomial(1, 0.03, size=cnt)
        data[mask, feat_idx["CWE Flag Count"]] = 0.0
        data[mask, feat_idx["ECE Flag Count"]] = rng.binomial(1, 0.02, size=cnt)

        data[mask, feat_idx["Down/Up Ratio"]] = np.round(bwd_pkts / np.maximum(fwd_pkts, 1.0), 2)
        data[mask, feat_idx["Average Packet Size"]] = (fwd_len_tot + bwd_len_tot) / (fwd_pkts + bwd_pkts)
        data[mask, feat_idx["Avg Fwd Segment Size"]] = data[mask, feat_idx["Fwd Packet Length Mean"]]
        data[mask, feat_idx["Avg Bwd Segment Size"]] = data[mask, feat_idx["Bwd Packet Length Mean"]]

        # Bulk averages (mostly sparse 0 in CICIDS2017)
        for bulk_col in (
            "Fwd Avg Bytes/Bulk",
            "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk",
            "Bwd Avg Packets/Bulk",
            "Bwd Avg Bulk Rate",
        ):
            data[mask, feat_idx[bulk_col]] = 0.0

        data[mask, feat_idx["Subflow Fwd Packets"]] = fwd_pkts
        data[mask, feat_idx["Subflow Fwd Bytes"]] = fwd_len_tot
        data[mask, feat_idx["Subflow Bwd Packets"]] = bwd_pkts
        data[mask, feat_idx["Subflow Bwd Bytes"]] = bwd_len_tot

        data[mask, feat_idx["Init_Win_bytes_forward"]] = rng.choice([256, 512, 8192, 29200, 65535], size=cnt)
        data[mask, feat_idx["Init_Win_bytes_backward"]] = rng.choice([-1, 0, 256, 29200, 65535], size=cnt)
        data[mask, feat_idx["act_data_pkt_fwd"]] = np.maximum(0.0, fwd_pkts - 1.0)
        data[mask, feat_idx["min_seg_size_forward"]] = rng.choice([20, 32, 40], size=cnt)

        act_mean = np.exp(rng.normal(act_mu, 1.1, size=cnt))
        idle_mean = np.exp(rng.normal(idle_mu, 1.2, size=cnt))
        data[mask, feat_idx["Active Mean"]] = act_mean
        data[mask, feat_idx["Active Std"]] = act_mean * rng.uniform(0.0, 0.5, size=cnt)
        data[mask, feat_idx["Active Max"]] = act_mean * rng.uniform(1.0, 2.0, size=cnt)
        data[mask, feat_idx["Active Min"]] = act_mean * rng.uniform(0.3, 1.0, size=cnt)
        data[mask, feat_idx["Idle Mean"]] = idle_mean
        data[mask, feat_idx["Idle Std"]] = idle_mean * rng.uniform(0.0, 0.4, size=cnt)
        data[mask, feat_idx["Idle Max"]] = idle_mean * rng.uniform(1.0, 1.8, size=cnt)
        data[mask, feat_idx["Idle Min"]] = idle_mean * rng.uniform(0.4, 1.0, size=cnt)

    df = pd.DataFrame(data, columns=CICIDS2017_FEATURES)

    # Inject synthetic structural identifiers (to verify identifier removal & Threat Map metadata)
    internal_ips = [f"192.168.10.{i}" for i in range(10, 60)]
    external_ips = [f"203.0.113.{i}" for i in range(10, 45)] + [f"198.51.100.{i}" for i in range(5, 35)]
    src_ips = np.where(
        labels == "BENIGN", rng.choice(internal_ips, size=n_samples), rng.choice(external_ips, size=n_samples)
    )
    dst_ips = rng.choice(["192.168.10.50", "192.168.10.51", "172.16.0.1"], size=n_samples)
    base_ts = datetime(2017, 7, 7, 9, 0, 0, tzinfo=UTC)
    timestamps = [
        (base_ts + timedelta(milliseconds=100 * i)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for i in range(n_samples)
    ]

    df.insert(
        0,
        "Flow ID",
        [f"{s}-{d}-{int(p)}-6" for s, d, p in zip(src_ips, dst_ips, df["Destination Port"], strict=False)],
    )
    df.insert(1, "Source IP", src_ips)
    df.insert(2, "Destination IP", dst_ips)
    df.insert(3, "Source Port", rng.integers(1024, 65535, size=n_samples))
    df.insert(4, "Timestamp", timestamps)
    df["Label"] = labels

    # Inject realistic CICFlowMeter quirks: Inf in rate columns, NaNs, and extreme outliers
    n_inf = int(n_samples * inf_fraction)
    if n_inf > 0:
        inf_rows = rng.choice(n_samples, size=n_inf, replace=False)
        half = n_inf // 2
        df.loc[inf_rows[:half], "Flow Bytes/s"] = np.inf
        df.loc[inf_rows[half:], "Flow Packets/s"] = -np.inf

    n_nan = int(n_samples * nan_fraction)
    if n_nan > 0:
        nan_rows = rng.choice(n_samples, size=n_nan, replace=False)
        df.loc[nan_rows, "Flow Bytes/s"] = np.nan

    # Inject extreme rate spike outliers (~1e10) to validate 5-sigma outlier clipping
    spike_rows = rng.choice(n_samples, size=max(10, n_samples // 500), replace=False)
    df.loc[spike_rows, "Flow Bytes/s"] = 1e11

    # Inject exact duplicates
    n_dup = int(n_samples * duplicate_fraction)
    if n_dup > 0:
        dup_rows = df.sample(n=n_dup, random_state=seed)
        df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Saved synthetic demonstration dataset (%d rows) to %s", len(df), save_path)

    return df


_TS_RELEVANT_COLUMNS_LOWER: frozenset[str] = frozenset(
    {
        "timestamp",
        "flow id",
        "source ip",
        "destination ip",
        "source port",
        "destination port",
        "flow duration",
        "total fwd packets",
        "total backward packets",
        "total length of fwd packets",
        "total length of bwd packets",
        "flow iat mean",
        "label",
    }
)


def _resolve_timestamp_dir(raw_spec: str | None) -> Path | None:
    if raw_spec is None:
        return None
    text = str(raw_spec).strip()
    if not text:
        return None
    candidate = Path(text)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _day_session_key(filename: str) -> str:
    stem = Path(str(filename)).name.lower().split(".pcap")[0]
    for prefix in ("workinghours-", "workinghours"):
        stem = stem.replace(prefix, "")
    return stem.strip("_- ") or Path(str(filename)).stem.lower()


def _normalize_key_column(series: pd.Series, is_numeric: bool) -> pd.Series:
    if is_numeric:
        vals = pd.to_numeric(series, errors="coerce").fillna(-1.0)
        return vals.round(4).astype(np.float64)
    return series.astype(str).str.strip()


def align_real_timestamps(
    df_main: pd.DataFrame,
    timestamp_source_dir: Path | str | None,
    label_col: str = "Label",
    source_file_col: str = SOURCE_FILE_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align real CICIDS2017 TrafficLabelling/GeneratedLabelledFlows timestamps to main flows.

    Join strategy priority:
    1. Native ``Timestamp`` column already present with non-null values.
    2. Direct identifier keys shared by both datasets (``Flow ID``, ``Source IP``,
       ``Destination IP``, ``Source Port``, ``Destination Port`` + per-key occurrence index).
    3. MachineLearningCVE telemetry signature fallback when IPs/Flow ID are absent:
       day/session stem + ``Destination Port``, ``Flow Duration``, ``Total Fwd Packets``,
       ``Total Backward Packets``, ``Total Length of Fwd Packets``, ``Total Length of Bwd Packets``,
       ``Flow IAT Mean``, normalized ``Label``, and per-signature occurrence index.

    Returns the aligned DataFrame (with ``Timestamp`` kept strictly for non-feature metadata)
    plus an explicit alignment audit dictionary reporting join keys, matched rows, and
    unmatched rows. Never raises when ``timestamp_source_dir`` is absent or empty.
    """
    n_total = len(df_main)
    ts_dir = (
        timestamp_source_dir
        if isinstance(timestamp_source_dir, Path)
        else _resolve_timestamp_dir(timestamp_source_dir)
    )

    existing_ts_col = next(
        (c for c in df_main.columns if str(c).strip().lower() == "timestamp"),
        None,
    )
    if existing_ts_col is not None and n_total > 0 and df_main[existing_ts_col].notna().any():
        matched = int(df_main[existing_ts_col].notna().sum())
        return df_main, {
            "status": "native_column",
            "source_dir": str(ts_dir) if ts_dir else None,
            "directory_exists": bool(ts_dir and ts_dir.exists()),
            "csv_files_found": 0,
            "real_timestamps_available": True,
            "join_strategy": "native Timestamp column already present in raw CSVs",
            "join_keys_used": [existing_ts_col],
            "total_rows": n_total,
            "matched_rows": matched,
            "unmatched_rows": int(n_total - matched),
            "match_rate": round(matched / max(n_total, 1), 6),
        }

    if ts_dir is None or not ts_dir.exists():
        logger.warning(
            "Timestamp source directory '%s' not found; time_block ordering will fall back to "
            "CSV row order (no real Timestamp).",
            ts_dir,
        )
        return df_main, {
            "status": "directory_not_found",
            "source_dir": str(ts_dir) if ts_dir else None,
            "directory_exists": False,
            "csv_files_found": 0,
            "real_timestamps_available": False,
            "join_strategy": "none (CSV row order fallback)",
            "join_keys_used": [],
            "total_rows": n_total,
            "matched_rows": 0,
            "unmatched_rows": n_total,
            "match_rate": 0.0,
        }

    ts_csvs = discover_csv_files(ts_dir)
    if not ts_csvs:
        logger.warning(
            "No CSV files found in timestamp_source_dir=%s; time_block ordering = CSV row order (no real Timestamp).",
            ts_dir,
        )
        return df_main, {
            "status": "no_csv_files",
            "source_dir": str(ts_dir),
            "directory_exists": True,
            "csv_files_found": 0,
            "real_timestamps_available": False,
            "join_strategy": "none (CSV row order fallback)",
            "join_keys_used": [],
            "total_rows": n_total,
            "matched_rows": 0,
            "unmatched_rows": n_total,
            "match_rate": 0.0,
        }

    ts_frames: list[pd.DataFrame] = []
    for p in ts_csvs:
        part = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                part = pd.read_csv(
                    p,
                    encoding=encoding,
                    low_memory=False,
                    usecols=lambda c: str(c).strip().lower() in _TS_RELEVANT_COLUMNS_LOWER,
                )
                part.columns = part.columns.astype(str).str.strip()
                break
            except UnicodeDecodeError:
                continue
        if part is not None and "Timestamp" in part.columns and len(part) > 0:
            part["__ts_day_key__"] = _day_session_key(p.name)
            ts_frames.append(part)

    if not ts_frames:
        logger.warning(
            "CSV files in %s did not contain a valid 'Timestamp' column; using CSV row order fallback.",
            ts_dir,
        )
        return df_main, {
            "status": "no_timestamp_column",
            "source_dir": str(ts_dir),
            "directory_exists": True,
            "csv_files_found": len(ts_csvs),
            "real_timestamps_available": False,
            "join_strategy": "none (CSV row order fallback)",
            "join_keys_used": [],
            "total_rows": n_total,
            "matched_rows": 0,
            "unmatched_rows": n_total,
            "match_rate": 0.0,
        }

    ts_df = pd.concat(ts_frames, ignore_index=True)

    id_priority = ("Flow ID", "Source IP", "Destination IP", "Source Port", "Destination Port")
    shared_id_keys = [col for col in id_priority if col in df_main.columns and col in ts_df.columns]

    sig_priority = (
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow IAT Mean",
    )
    shared_sig_keys = [col for col in sig_priority if col in df_main.columns and col in ts_df.columns]

    numeric_keys = {
        "Source Port",
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow IAT Mean",
    }

    if "Flow ID" in shared_id_keys or ("Source IP" in shared_id_keys and "Destination IP" in shared_id_keys):
        join_keys = list(shared_id_keys)
        join_strategy = "5tuple_flow_id"
    else:
        join_keys = list(shared_sig_keys)
        join_strategy = "flow_telemetry_signature"

    if not join_keys:
        logger.warning("No shared join keys found between raw CSVs and %s; skipping timestamp merge.", ts_dir)
        return df_main, {
            "status": "no_shared_keys",
            "source_dir": str(ts_dir),
            "directory_exists": True,
            "csv_files_found": len(ts_csvs),
            "real_timestamps_available": False,
            "join_strategy": "none (no overlapping keys)",
            "join_keys_used": [],
            "total_rows": n_total,
            "matched_rows": 0,
            "unmatched_rows": n_total,
            "match_rate": 0.0,
        }

    left_keys = pd.DataFrame(index=df_main.index)
    right_keys = pd.DataFrame(index=ts_df.index)
    used_key_names: list[str] = []

    if join_strategy == "flow_telemetry_signature" and source_file_col in df_main.columns:
        left_keys["__k_day"] = df_main[source_file_col].map(_day_session_key)
        right_keys["__k_day"] = ts_df["__ts_day_key__"]
        used_key_names.append("Source_Session_Day")

    for col in join_keys:
        is_num = col in numeric_keys
        k_col = f"__k_{col}"
        left_keys[k_col] = _normalize_key_column(df_main[col], is_num)
        right_keys[k_col] = _normalize_key_column(ts_df[col], is_num)
        used_key_names.append(col)

    if label_col in df_main.columns and "Label" in ts_df.columns:
        left_keys["__k_label"] = df_main[label_col].map(normalize_label_text)
        right_keys["__k_label"] = ts_df["Label"].map(normalize_label_text)
        used_key_names.append("Label")

    merge_cols = list(left_keys.columns)
    left_keys["__occ__"] = left_keys.groupby(merge_cols, sort=False).cumcount()
    right_keys["__occ__"] = right_keys.groupby(merge_cols, sort=False).cumcount()
    right_keys["Timestamp"] = ts_df["Timestamp"].astype(str).str.strip().values
    used_key_names.append("Occurrence_Index")

    merge_on = [*merge_cols, "__occ__"]
    right_dedup = right_keys.drop_duplicates(subset=merge_on, keep="first")
    left_keys["__row_pos__"] = np.arange(len(df_main), dtype=np.int64)
    merged_pos = left_keys.merge(right_dedup, on=merge_on, how="left").sort_values("__row_pos__")

    out_df = df_main.copy()
    aligned_ts = merged_pos["Timestamp"].to_numpy()
    out_df["Timestamp"] = pd.Series(aligned_ts, index=out_df.index)
    valid_mask = (
        out_df["Timestamp"].notna()
        & out_df["Timestamp"].astype(str).str.strip().ne("")
        & ~out_df["Timestamp"].astype(str).str.strip().str.lower().isin({"nan", "nat", "none"})
    )
    matched_rows = int(valid_mask.sum())
    unmatched_rows = int(n_total - matched_rows)
    match_rate = round(matched_rows / max(n_total, 1), 6)
    final_df = out_df if matched_rows > 0 else df_main

    logger.info(
        "Aligned real timestamps from %s via %s keys (%s): matched=%d / %d (unmatched=%d, rate=%.2f%%)",
        ts_dir,
        join_strategy,
        ", ".join(used_key_names),
        matched_rows,
        n_total,
        unmatched_rows,
        match_rate * 100.0,
    )
    return final_df, {
        "status": "aligned" if matched_rows > 0 else "zero_matches",
        "source_dir": str(ts_dir),
        "directory_exists": True,
        "csv_files_found": len(ts_csvs),
        "real_timestamps_available": matched_rows > 0,
        "join_strategy": join_strategy,
        "join_keys_used": used_key_names,
        "total_rows": n_total,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "match_rate": match_rate,
    }


def _session_chronological_rank(filename: str) -> int:
    """Return CICIDS2017 capture session order (Monday -> Friday afternoon)."""
    name = Path(str(filename)).name.lower()
    if "monday" in name:
        return 0
    if "tuesday" in name:
        return 1
    if "wednesday" in name:
        return 2
    if "thursday" in name:
        return 3 if "morning" in name else 4
    if "friday" in name:
        if "morning" in name:
            return 5
        if "portscan" in name:
            return 6
        if "ddos" in name:
            return 7
        return 8
    return 9


def load_cicids2017_corpus(
    raw_dir: Path,
    synthetic_dir: Path,
    data_cfg: DataConfig,
    seed: int = 42,
    force_synthetic: bool = False,
) -> tuple[pd.DataFrame, DatasetSourceInfo]:
    """Load real CICIDS2017 CSVs from raw_dir, or fall back to labeled synthetic demo mode."""
    csv_files = [] if force_synthetic else discover_csv_files(raw_dir)

    if csv_files:
        logger.info("Discovered %d raw CSV file(s) in %s", len(csv_files), raw_dir)
        frames: list[pd.DataFrame] = []
        for csv_path in csv_files:
            part = _read_single_csv(csv_path)
            logger.info("Loaded %-55s -> %d rows x %d cols", csv_path.name, len(part), part.shape[1])
            part[SOURCE_FILE_COLUMN] = csv_path.name
            session_rank = _session_chronological_rank(csv_path.name)
            part[CHRON_ORDER_COLUMN] = (session_rank + 1) * 100_000_000 + np.arange(len(part), dtype=np.int64)
            frames.append(part)

        df = pd.concat(frames, ignore_index=True)
        label_col = data_cfg.label_column
        df, ts_alignment = align_real_timestamps(
            df_main=df,
            timestamp_source_dir=data_cfg.timestamp_source_dir,
            label_col=label_col,
        )
        raw_count = len(df)

        if data_cfg.max_rows_sample and len(df) > data_cfg.max_rows_sample and label_col in df.columns:
            df = stratified_downsample(df, label_col=label_col, max_rows=data_cfg.max_rows_sample, seed=seed)
            logger.info("Stratified downsample from %d to %d rows", raw_count, len(df))

        info = DatasetSourceInfo(
            source_type="cicids2017_raw",
            is_synthetic=False,
            files_loaded=[p.name for p in csv_files],
            raw_row_count=raw_count,
            sampled_row_count=len(df),
            warning_message=None,
            timestamp_alignment=ts_alignment,
        )
        return df, info

    if not data_cfg.synthetic_fallback.enabled:
        raise FileNotFoundError(
            f"No CICIDS2017 CSV files found in {raw_dir} and synthetic fallback is disabled."
        )

    logger.warning(
        "No raw CICIDS2017 CSVs found in %s. Entering EXPLICIT SYNTHETIC DEMONSTRATION MODE.",
        raw_dir,
    )
    syn_path = synthetic_dir / "cicids2017_synthetic_demo.csv"
    df = generate_synthetic_cicids2017(
        n_samples=data_cfg.synthetic_fallback.n_samples,
        seed=seed,
        class_weights=data_cfg.synthetic_fallback.class_weights,
        duplicate_fraction=data_cfg.synthetic_fallback.duplicate_fraction,
        inf_fraction=data_cfg.synthetic_fallback.inf_fraction,
        nan_fraction=data_cfg.synthetic_fallback.nan_fraction,
        save_path=syn_path,
    )
    info = DatasetSourceInfo(
        source_type="synthetic_demonstration",
        is_synthetic=True,
        files_loaded=[syn_path.name],
        raw_row_count=len(df),
        sampled_row_count=len(df),
        warning_message=(
            "SYNTHETIC DEMONSTRATION MODE ACTIVE: Original CICIDS2017 CSVs were not present "
            "in data/raw/. All metrics reflect schema-faithful synthetic demonstration data "
            "rather than genuine CICIDS2017 benchmark performance."
        ),
    )
    return df, info
