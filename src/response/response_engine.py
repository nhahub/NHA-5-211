"""Canonical Active Response Engine shared across notebooks, CLI, and Streamlit dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import ipaddress
import json
import logging
from pathlib import Path
import subprocess
from typing import Any

from src.detection.threat_scoring import get_threat_level
from src.utils.logging import get_logger

_DOCUMENTATION_TEST_NETS = (
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("2001:db8::/32"),
)


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


def is_protected_ip(ip: str) -> tuple[bool, str]:
    """Return (is_protected, reason) for private, loopback, link-local, multicast, reserved, or unspecified IPs."""
    if not validate_ip(ip):
        return False, "invalid"
    ip_obj = ipaddress.ip_address(ip.strip())
    if ip_obj.is_loopback:
        return True, "loopback"
    if ip_obj.is_link_local:
        return True, "link-local"
    if ip_obj.is_multicast:
        return True, "multicast"
    if ip_obj.is_unspecified:
        return True, "unspecified"
    if ip_obj.is_reserved:
        return True, "reserved"
    if ip_obj.is_private and not any(ip_obj in net for net in _DOCUMENTATION_TEST_NETS):
        return True, "private"
    return False, "routable_or_doc_testnet"


def build_firewall_command_preview(
    src_ip: str,
    action_type: str = "BLOCK",
    rate_limit_pps: int = 100,
) -> dict[str, Any]:
    """Build non-executed OS-specific firewall command previews for Linux and Windows."""
    if not validate_ip(src_ip):
        raise ValueError(f"Cannot build firewall preview for invalid IP '{src_ip}'.")

    ip_obj = ipaddress.ip_address(src_ip.strip())
    ip_clean = str(ip_obj)
    protected, reason = is_protected_ip(ip_clean)
    if protected:
        return {
            "status": "NOT EXECUTED (dry-run)",
            "ip": ip_clean,
            "refused_reason": f"Protected {reason} IP range is never targeted.",
        }

    safe_tag = ip_clean.replace(":", "_").replace(".", "_")
    iptables_bin = "ip6tables" if ip_obj.version == 6 else "iptables"
    nft_family = "ip6" if ip_obj.version == 6 else "ip"
    pps = max(1, int(rate_limit_pps))

    if action_type == "RATE_LIMIT":
        return {
            "status": "NOT EXECUTED (dry-run)",
            "ip": ip_clean,
            "linux_iptables": (
                f"sudo {iptables_bin} -I INPUT 1 -s {ip_clean} -m limit --limit {pps}/second -j ACCEPT"
            ),
            "linux_nftables": (
                f"sudo nft add rule inet filter input {nft_family} saddr {ip_clean} "
                f"limit rate {pps}/second accept"
            ),
            "windows_netsh": (
                f"# Windows Defender Firewall lacks native per-IP pps rate limit; "
                f"optional QoS policy tag: SMART_IDS_RATELIMIT_{safe_tag}_{pps}pps"
            ),
            "windows_powershell": (
                f"New-NetQosPolicy -Name 'SMART_IDS_RATELIMIT_{safe_tag}' "
                f"-IPSrcPrefixMatchCondition {ip_clean} -ThrottleRateActionBitsPerSecond {pps * 12000}"
            ),
        }

    return {
        "status": "NOT EXECUTED (dry-run)",
        "ip": ip_clean,
        "linux_iptables": f"sudo {iptables_bin} -I INPUT 1 -s {ip_clean} -j DROP",
        "linux_nftables": f"sudo nft add rule inet filter input {nft_family} saddr {ip_clean} drop",
        "windows_netsh": (
            f'netsh advfirewall firewall add rule name="SMART_IDS_BLOCK_{safe_tag}" '
            f"dir=in action=block remoteip={ip_clean}"
        ),
        "windows_powershell": (
            f"New-NetFirewallRule -DisplayName 'SMART_IDS_BLOCK_{safe_tag}' "
            f"-Direction Inbound -RemoteAddress {ip_clean} -Action Block"
        ),
    }


@dataclass
class ResponseDecision:
    """Canonical response decision contract plus anomaly metadata and non-executed previews."""

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
    execution_status: str = "NOT EXECUTED (dry-run)"
    command_preview: dict[str, Any] = field(default_factory=dict)

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
        protected, protected_reason = is_protected_ip(ip_clean)
        if protected:
            msg = (
                f"Refused block for protected {protected_reason} IP '{ip_clean}' "
                "(private/loopback/link-local/multicast/reserved ranges are never blocked) "
                "[NOT EXECUTED (dry-run)]."
            )
            self.logger.warning(
                "BLOCK_REFUSED_PROTECTED | src_ip=%s | reason=%s | NOT EXECUTED (dry-run)",
                ip_clean,
                protected_reason,
            )
            return False, msg, False

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
            msg = f"[DRY RUN] NOT EXECUTED (dry-run) | Preview: {' '.join(command)}"
            self.blocked_ips.add(ip_clean)
            self.logger.critical(
                "BLOCK (DRY RUN) | src_ip=%s | status=NOT EXECUTED (dry-run) | cmd=%s",
                ip_clean,
                " ".join(command),
            )
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
        execution_status = "NOT EXECUTED (dry-run)"
        command_preview: dict[str, Any] = {}
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
            if not validate_ip(src_ip):
                action = "RATE_LIMIT_FAILED"
                action_detail = f"Invalid source IP '{src_ip}' rejected before rate-limiting."
                self.logger.error("RATE_LIMIT_FAILED | invalid_ip=%s | class=%s", src_ip, attack_class)
            else:
                protected, protected_reason = is_protected_ip(src_ip)
                command_preview = build_firewall_command_preview(
                    src_ip,
                    action_type="RATE_LIMIT",
                    rate_limit_pps=self.rate_limit_pps,
                )
                if protected:
                    action = "RATE_LIMIT_REFUSED_PROTECTED"
                    action_detail = (
                        f"[NOT EXECUTED (dry-run)] Refused rate-limit for protected {protected_reason} "
                        f"IP '{src_ip}' (confidence={conf:.4f})."
                    )
                    self.logger.warning(
                        "RATE_LIMIT_REFUSED_PROTECTED | src_ip=%s | reason=%s | status=NOT EXECUTED (dry-run)",
                        src_ip,
                        protected_reason,
                    )
                else:
                    action = "RATE_LIMIT"
                    action_detail = (
                        f"[DRY RUN] NOT EXECUTED (dry-run) | Rate-limiting preview ({self.rate_limit_pps} pps) "
                        f"generated for {src_ip} ('{attack_class}', confidence={conf:.4f})."
                    )
                    self.logger.warning(
                        "RATE_LIMIT | src_ip=%s | ceiling=%dpps | class=%s | conf=%.4f | status=NOT EXECUTED (dry-run)",
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
                protected, protected_reason = is_protected_ip(src_ip)
                command_preview = build_firewall_command_preview(src_ip, action_type="BLOCK")
                if protected:
                    action = "BLOCK_REFUSED_PROTECTED"
                    action_detail = (
                        f"[NOT EXECUTED (dry-run)] Refused block for protected {protected_reason} "
                        f"IP '{src_ip}' (confidence={conf:.4f})."
                    )
                    self.logger.warning(
                        "BLOCK_REFUSED_PROTECTED | src_ip=%s | reason=%s | status=NOT EXECUTED (dry-run)",
                        src_ip,
                        protected_reason,
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
            execution_status=execution_status,
            command_preview=command_preview,
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
