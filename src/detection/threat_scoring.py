"""Threat severity taxonomy mapping supervised confidence and anomaly status."""

from __future__ import annotations

THREAT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Low": (0.0, 0.40),
    "Medium": (0.40, 0.70),
    "High": (0.70, 0.90),
    "Critical": (0.90, 1.01),
}

THREAT_COLORS: dict[str, str] = {
    "Low": "#22c55e",
    "Medium": "#eab308",
    "High": "#f97316",
    "Critical": "#ef4444",
}

THREAT_ICONS: dict[str, str] = {
    "Low": "🟢",
    "Medium": "🟡",
    "High": "🟠",
    "Critical": "🔴",
}

THREAT_ORDER: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def get_threat_level(
    confidence: float,
    attack_class: str | None = None,
    is_anomaly: bool = False,
    anomaly_ratio: float = 0.0,
    thresholds: dict[str, tuple[float, float]] | None = None,
    min_actionable_confidence: float | None = None,
) -> str:
    """Map prediction confidence, class identity, and anomaly signal to 4-tier severity.

    Safety Refinement over naive threshold check:
    - Non-anomalous BENIGN traffic is always classified as 'Low' severity regardless
      of how confident the classifier is that the traffic is benign.
    - If traffic is predicted BENIGN by closed-set models BUT flagged as anomalous by the
      unsupervised Autoencoder, severity scales with the reconstruction-error ratio.
    - If min_actionable_confidence is set (e.g. from dashboard slider), attack detections
      below that floor are capped at 'Low' to avoid noisy escalations.
    """
    conf = float(max(0.0, min(1.0, confidence)))
    bands = thresholds or THREAT_THRESHOLDS

    if attack_class is not None and attack_class.strip().lower() == "benign":
        if not is_anomaly:
            return "Low"
        # Unsupervised anomaly escalation on otherwise benign-looking traffic
        if anomaly_ratio >= 4.0:
            return "Critical"
        if anomaly_ratio >= 2.0:
            return "High"
        return "Medium"

    if min_actionable_confidence is not None and conf < float(min_actionable_confidence):
        return "Low"

    level_out = "Critical"
    for level, (lo, hi) in bands.items():
        if lo <= conf < hi:
            level_out = level
            break

    if is_anomaly and anomaly_ratio >= 3.0 and THREAT_ORDER.get(level_out, 0) < THREAT_ORDER["High"]:
        level_out = "High"
    return level_out
