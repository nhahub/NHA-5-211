"""Canonical Active Response Engine shared across notebooks, CLI, and Streamlit dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import ipaddress
import json
import logging
from pathlib import Path
import subprocess
from typing import Any

from src.detection.threat_scoring import get_threat_level
from src.utils.logging import get_logger


def validate_ip(ip: str) -> bool:
    """Strictly validate an IPv4 or IPv6 address before command construction."""
    if not isinstance(ip, str):
        return False
    candidate = ip.strip()
    if not candidate:
        return False
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


@dataclass
class ResponseDecision:
    """Nine-field canonical response decision contract plus anomaly metadata."""

    timestamp: str
    src_ip: str
    attack_class: str
    confidence: float
    threat_level: str
    model_used: str
    action: str
    action_detail: str
    blocked: bool
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    idempotent_skip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActiveResponseEngine:
    """Graduated four-tier automated mitigation simulator with durable audit logging."""

    def __init__(
        self,
        dry_run: bool = True,
        rate_limit_pps: int = 100,
        log_file: Path | None = None,
        history_file: Path | None = None,
        enforce_academic_dry_run_only: bool = True,
    ) -> None:
        # Enforce strict academic prototype safety: never execute real host firewall commands
        # unless explicitly disabled in controlled environments.
        self.enforce_academic_dry_run_only = enforce_academic_dry_run_only
        self.dry_run = True if enforce_academic_dry_run_only else bool(dry_run)
        self.rate_limit_pps = int(rate_limit_pps)
        self.blocked_ips: set[str] = set()
        self.history: list[dict[str, Any]] = []
        self.log_file = Path(log_file) if log_file else Path("reports/ids_response_log.txt")
        self.history_file = Path(history_file) if history_file else Path("reports/incident_history.json")
        self.logger: logging.Logger = get_logger("SmartIDS_Response", log_file=self.log_file)

    def block_ip_iptables(self, src_ip: str, dry_run: bool | None = None) -> tuple[bool, str, bool]:
        """Construct and dry-run simulate (or safely execute) iptables INPUT DROP rule.

        Returns:
            (blocked_success, detail_message, idempotent_skip)
        """
        if not validate_ip(src_ip):
            self.logger.error(
                "Validation failed: invalid IP address '%s' rejected before command construction.", src_ip
            )
            return False, f"Invalid source IP '{src_ip}' rejected by ipaddress validation gate.", False

        ip_clean = str(ipaddress.ip_address(src_ip.strip()))
        if ip_clean in self.blocked_ips:
            msg = f"Idempotent skip: {ip_clean} already present in BLOCKED_IPS registry."
            self.logger.info(msg)
            return True, msg, True

        effective_dry_run = (
            True
            if self.enforce_academic_dry_run_only
            else (self.dry_run if dry_run is None else bool(dry_run))
        )
        command = ["iptables", "-I", "INPUT", "1", "-s", ip_clean, "-j", "DROP"]

        if effective_dry_run:
            msg = f"[DRY RUN] Would execute: {' '.join(command)}"
            self.blocked_ips.add(ip_clean)
            self.logger.critical("BLOCK (DRY RUN) | src_ip=%s | cmd=%s", ip_clean, " ".join(command))
            return True, msg, False

        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
            self.blocked_ips.add(ip_clean)
            msg = f"Executed firewall rule: {' '.join(command)}"
            self.logger.critical("BLOCK (LIVE) | src_ip=%s | cmd=%s", ip_clean, " ".join(command))
            return True, msg, False
        except subprocess.CalledProcessError as exc:
            msg = f"iptables block failed for {ip_clean}: {exc.stderr.strip()}"
            self.logger.error(msg)
            return False, msg, False
        except FileNotFoundError:
            msg = f"iptables executable not found (non-Linux host); simulated block logged for {ip_clean}."
            self.logger.error(msg)
            return False, msg, False

    def apply_active_response(
        self,
        src_ip: str,
        attack_class: str,
        confidence: float,
        model_used: str = "Hybrid Ensemble",
        dry_run: bool | None = None,
        is_anomaly: bool = False,
        anomaly_score: float = 0.0,
        anomaly_ratio: float = 0.0,
        min_actionable_confidence: float | None = None,
        persist_history: bool = True,
    ) -> dict[str, Any]:
        """Evaluate threat severity and emit a structured, audited mitigation decision."""
        conf = float(max(0.0, min(1.0, confidence)))
        threat_level = get_threat_level(
            confidence=conf,
            attack_class=attack_class,
            is_anomaly=is_anomaly,
            anomaly_ratio=anomaly_ratio,
            min_actionable_confidence=min_actionable_confidence,
        )

        blocked = False
        idempotent_skip = False
        if threat_level == "Low":
            action = "LOG"
            action_detail = f"Passive statistical logging for '{attack_class}' (confidence={conf:.4f})."
        elif threat_level == "Medium":
            action = "ALERT"
            action_detail = (
                f"SIEM/SOC analyst review flagged for '{attack_class}' from {src_ip} "
                f"(confidence={conf:.4f}, anomaly={is_anomaly})."
            )
            self.logger.info("ALERT | src_ip=%s | class=%s | conf=%.4f", src_ip, attack_class, conf)
        elif threat_level == "High":
            action = "RATE_LIMIT"
            if not validate_ip(src_ip):
                action = "RATE_LIMIT_FAILED"
                action_detail = f"Invalid source IP '{src_ip}' rejected before rate-limiting."
                self.logger.error("RATE_LIMIT_FAILED | invalid_ip=%s | class=%s", src_ip, attack_class)
            else:
                action_detail = (
                    f"[DRY RUN] Rate-limiting applied to {src_ip} ({self.rate_limit_pps} pps ceiling) "
                    f"for '{attack_class}' (confidence={conf:.4f})."
                )
                self.logger.warning(
                    "RATE_LIMIT | src_ip=%s | ceiling=%dpps | class=%s | conf=%.4f",
                    src_ip,
                    self.rate_limit_pps,
                    attack_class,
                    conf,
                )
        else:
            # Critical tier
            if not validate_ip(src_ip):
                action = "BLOCK_FAILED"
                action_detail = f"Invalid source IP '{src_ip}' rejected by IP validation gate."
                self.logger.error(
                    "BLOCK_FAILED | invalid_ip=%s | class=%s | conf=%.4f", src_ip, attack_class, conf
                )
            else:
                ok, detail, idempotent_skip = self.block_ip_iptables(src_ip, dry_run=dry_run)
                action = "BLOCK" if ok else "BLOCK_FAILED"
                action_detail = detail
                blocked = ok

        decision = ResponseDecision(
            timestamp=datetime.now(UTC).isoformat(),
            src_ip=str(src_ip),
            attack_class=str(attack_class),
            confidence=round(conf, 4),
            threat_level=threat_level,
            model_used=str(model_used),
            action=action,
            action_detail=action_detail,
            blocked=blocked,
            is_anomaly=bool(is_anomaly),
            anomaly_score=round(float(anomaly_score), 6),
            idempotent_skip=idempotent_skip,
        ).to_dict()

        self.history.append(decision)
        if persist_history:
            self.save_history()
        return decision

    def unblock_ip(self, src_ip: str) -> bool:
        """Remove an IP from the simulated block registry."""
        if src_ip in self.blocked_ips:
            self.blocked_ips.remove(src_ip)
            self.logger.info("Simulated unblock removed %s from BLOCKED_IPS.", src_ip)
            return True
        return False

    def clear_history(self) -> None:
        """Clear in-memory response history and blocked IP set."""
        self.blocked_ips.clear()
        self.history.clear()

    def save_history(self) -> None:
        """Persist response event history as JSON for SOC export and audit."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with self.history_file.open("w", encoding="utf-8") as fh:
            json.dump(self.history[-500:], fh, indent=2)

    def load_history(self) -> list[dict[str, Any]]:
        """Load persisted response events if present on disk."""
        if self.history_file.exists():
            try:
                with self.history_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    self.history = data
            except Exception:
                pass
        return self.history
