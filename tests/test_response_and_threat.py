"""Tests for threat scoring, IP validation, active response simulation, and SHAP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

from src.detection.threat_scoring import get_threat_level
from src.explainability.shap_explainer import explain_single_sample
from src.response.response_engine import (
    ActiveResponseEngine,
    build_firewall_command_preview,
    is_protected_ip,
    validate_ip,
)


def test_threat_level_boundaries_and_benign_safety() -> None:
    # Exact boundary tests for attack classes
    assert get_threat_level(0.0, attack_class="DDoS") == "Low"
    assert get_threat_level(0.39, attack_class="DDoS") == "Low"
    assert get_threat_level(0.40, attack_class="DDoS") == "Medium"
    assert get_threat_level(0.69, attack_class="DDoS") == "Medium"
    assert get_threat_level(0.70, attack_class="DDoS") == "High"
    assert get_threat_level(0.89, attack_class="DDoS") == "High"
    assert get_threat_level(0.90, attack_class="DDoS") == "Critical"
    assert get_threat_level(1.00, attack_class="DDoS") == "Critical"

    # Safety check: non-anomalous BENIGN at 0.999 confidence MUST stay Low
    assert get_threat_level(0.999, attack_class="BENIGN", is_anomaly=False) == "Low"

    # Anomalous BENIGN escalates via reconstruction error ratio
    assert get_threat_level(0.95, attack_class="BENIGN", is_anomaly=True, anomaly_ratio=1.5) == "Medium"
    assert get_threat_level(0.95, attack_class="BENIGN", is_anomaly=True, anomaly_ratio=2.5) == "High"
    assert get_threat_level(0.95, attack_class="BENIGN", is_anomaly=True, anomaly_ratio=5.0) == "Critical"


def test_ip_validation_prevents_command_injection() -> None:
    assert validate_ip("192.168.10.5") is True
    assert validate_ip("2001:db8::1") is True
    assert validate_ip("INVALID_IP") is False
    assert validate_ip("192.168.1.1; rm -rf /") is False
    assert validate_ip("127.0.0.1 && calc.exe") is False
    assert validate_ip("") is False


def test_active_response_dry_run_idempotency_and_audit_log(tmp_path: Path) -> None:
    log_path = tmp_path / "ids_response_log.txt"
    hist_path = tmp_path / "incident_history.json"
    engine = ActiveResponseEngine(dry_run=True, log_file=log_path, history_file=hist_path)

    # 1. Critical valid IP -> DRY RUN BLOCK
    r1 = engine.apply_active_response("203.0.113.10", "DDoS", 0.96, "XGBoost")
    assert r1["threat_level"] == "Critical"
    assert r1["action"] == "BLOCK"
    assert r1["blocked"] is True
    assert "[DRY RUN]" in r1["action_detail"]
    assert "203.0.113.10" in engine.blocked_ips

    # 2. Same IP again -> idempotent skip
    r2 = engine.apply_active_response("203.0.113.10", "DDoS", 0.98, "XGBoost")
    assert r2["idempotent_skip"] is True
    assert r2["blocked"] is True

    # 3. Malformed IP injection attempt -> BLOCK_FAILED
    r3 = engine.apply_active_response("INVALID_IP;whoami", "SSH-Patator", 0.95, "MLP")
    assert r3["action"] == "BLOCK_FAILED"
    assert r3["blocked"] is False

    # 4. High tier -> RATE_LIMIT
    r4 = engine.apply_active_response("198.51.100.22", "PortScan", 0.80, "CNN")
    assert r4["threat_level"] == "High"
    assert r4["action"] == "RATE_LIMIT"
    assert r4["blocked"] is False

    # 5. Audit file & history persisted
    assert log_path.exists()
    assert hist_path.exists()
    assert len(engine.history) == 4


def test_local_shap_explanation_xgboost(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, eval_metric="logloss")
    clf.fit(X, y)

    feat_names = [f"f_{i}" for i in range(5)]
    wf_path = tmp_path / "waterfall.png"
    res = explain_single_sample(
        model_name="XGBoost",
        model=clf,
        x_single=X[0],
        X_background=X,
        feature_names=feat_names,
        class_idx=int(y[0]),
        class_name="Attack" if y[0] == 1 else "Benign",
        waterfall_path=wf_path,
        max_display=5,
    )
    assert res["explainer"] == "TreeExplainer"
    assert len(res["top_features"]) == 5
    assert wf_path.exists()


def test_response_command_previews_and_protected_ip_safeguards(tmp_path: Path) -> None:
    """Firewall previews generate Linux/Windows strings in NOT EXECUTED dry-run mode and protect local ranges."""
    for protected_ip in (
        "127.0.0.1",
        "::1",
        "10.0.0.1",
        "172.16.10.4",
        "192.168.1.50",
        "169.254.1.1",
        "224.0.0.1",
        "0.0.0.0",
    ):
        is_prot, _ = is_protected_ip(protected_ip)
        assert is_prot is True

    assert is_protected_ip("203.0.113.55")[0] is False
    assert is_protected_ip("8.8.8.8")[0] is False

    log_path = tmp_path / "resp.log"
    hist_path = tmp_path / "hist.json"
    engine = ActiveResponseEngine(dry_run=True, log_file=log_path, history_file=hist_path)

    # Protected internal IP must NEVER be blocked or added to blocked_ips
    prot_res = engine.apply_active_response("192.168.1.50", "DDoS", 0.99, "XGBoost")
    assert prot_res["blocked"] is False
    assert prot_res["action"] == "BLOCK_REFUSED_PROTECTED"
    assert prot_res["execution_status"] == "NOT EXECUTED (dry-run)"
    assert "192.168.1.50" not in engine.blocked_ips
    assert "refused_reason" in prot_res["command_preview"]

    # Valid external/test-net IPs produce full Linux & Windows preview commands tagged NOT EXECUTED (dry-run)
    v4_res = engine.apply_active_response("203.0.113.55", "DDoS", 0.97, "XGBoost")
    assert v4_res["action"] == "BLOCK"
    assert v4_res["execution_status"] == "NOT EXECUTED (dry-run)"
    prev = v4_res["command_preview"]
    assert prev["status"] == "NOT EXECUTED (dry-run)"
    assert "iptables" in prev["linux_iptables"] and "203.0.113.55" in prev["linux_iptables"]
    assert "nft add rule" in prev["linux_nftables"]
    assert "netsh advfirewall" in prev["windows_netsh"]
    assert "New-NetFirewallRule" in prev["windows_powershell"]

    v6_prev = build_firewall_command_preview("2001:db8::42", action_type="BLOCK")
    assert "ip6tables" in v6_prev["linux_iptables"]
    assert "ip6 saddr" in v6_prev["linux_nftables"]

    # Audit log explicitly records NOT EXECUTED (dry-run)
    assert "NOT EXECUTED (dry-run)" in log_path.read_text(encoding="utf-8")
