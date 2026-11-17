"""Centralized Active Response Engine for Smart IDS v3.

Translates model predictions and confidence scores into graduated, risk-proportionate
mitigation actions:
- Low (0.00 - 0.40): LOG (statistical recording)
- Medium (0.40 - 0.70): ALERT (SIEM / SOC dispatch)
- High (0.70 - 0.90): RATE_LIMIT (traffic shaping / throttling)
- Critical (0.90 - 1.01): BLOCK (firewall DROP rule simulation)

Enforces strict IP validation to prevent command injection, maintains an idempotent
in-memory set of blocked IPs, and logs all events to an append-only audit trail.
This module is the SINGLE CANONICAL SOURCE OF TRUTH imported by both notebooks
and the Streamlit dashboard.
"""

from datetime import datetime
import ipaddress
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from smart_ids.config import (
    LOGS_FILE,
    RATE_LIMIT_CEILING,
    THREAT_ACTIONS,
    THREAT_THRESHOLDS,
)

# Persistent in-memory cache for idempotency
BLOCKED_IPS: Set[str] = set()

# Dedicated response logger appending to ids_response_log.txt
response_logger = logging.getLogger("SmartIDS_Response")
response_logger.setLevel(logging.INFO)

# Attach FileHandler if not already configured
if not response_logger.handlers:
    file_handler = logging.FileHandler(LOGS_FILE, mode="a", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    response_logger.addHandler(file_handler)


def get_threat_level(confidence: float) -> str:
    """Map a confidence score to one of 4 graduated threat severity tiers.

    Uses half-open intervals [lo, hi). Upper bound for Critical is 1.01 to
    safely accommodate confidence scores of exactly 1.0 due to floating-point rounding.

    Args:
        confidence: Prediction confidence score between 0.0 and 1.0.

    Returns:
        Threat level string: 'Low', 'Medium', 'High', or 'Critical'.
    """
    conf = float(confidence)
    for level, (lo, hi) in THREAT_THRESHOLDS.items():
        if lo <= conf < hi:
            return level
    # Fallback guard for confidence >= 1.01
    return "Critical" if conf >= 0.90 else "Low"


def validate_ip(ip: str) -> bool:
    """Verify that a string represents a syntactically valid IPv4 or IPv6 address.

    Defense-in-depth gate preventing shell metacharacters and command injection
    from propagating into downstream mitigation commands.

    Args:
        ip: Candidate IP address string.

    Returns:
        True if valid IP address, False otherwise.
    """
    if not isinstance(ip, str) or not ip.strip():
        return False
    try:
        ipaddress.ip_address(ip.strip())
        return True
    except ValueError:
        return False


def block_ip_iptables(
    src_ip: str,
    dry_run: bool = True,
) -> Tuple[bool, str, List[str]]:
    """Construct and simulate Linux netfilter iptables DROP firewall rule.

    Command format: ['iptables', '-I', 'INPUT', '1', '-s', <src_ip>, '-j', 'DROP']
    - -I INPUT 1 ensures top-priority precedence in the INPUT chain.
    - -j DROP silently discards packets without acknowledging the attacker.
    - Dry-run simulation guarantees host safety while fully exercising logic.
    - Idempotency check prevents duplicate redundant rules for the same IP.

    Args:
        src_ip: Validated source IP address.
        dry_run: If True, simulates execution without executing system shell commands.

    Returns:
        Tuple of (success_bool, detail_message, command_args_list).
    """
    command = ["iptables", "-I", "INPUT", "1", "-s", src_ip, "-j", "DROP"]

    # Idempotency gate
    if src_ip in BLOCKED_IPS:
        msg = f"IP {src_ip} is already blocked. Skipping redundant rule (idempotent)."
        response_logger.info(msg)
        return True, msg, command

    if dry_run:
        BLOCKED_IPS.add(src_ip)
        msg = f"[DRY RUN] Simulated iptables command: {' '.join(command)}"
        response_logger.critical(
            "ACTION=BLOCK [SIMULATED] | IP=%s | RULE='%s'",
            src_ip,
            " ".join(command),
        )
        return True, msg, command
    else:
        # In this project, active response is strictly simulated for safety across all OS platforms
        BLOCKED_IPS.add(src_ip)
        msg = f"[SIMULATION ONLY] iptables execution bypassed for platform safety: {' '.join(command)}"
        response_logger.critical("ACTION=BLOCK [SIMULATION_SAFEGUARD] | IP=%s", src_ip)
        return True, msg, command


def apply_active_response(
    src_ip: str,
    attack_class: str,
    confidence: float,
    model_name: str = "SmartIDS_MLP",
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Execute graduated mitigation decision pipeline for an analyzed flow.

    Returns a standardized 9-field response dictionary.

    Args:
        src_ip: Source IP of the detected flow.
        attack_class: Predicted class label.
        confidence: Model confidence score (0.0 to 1.0).
        model_name: Name of detecting model.
        dry_run: If True, operates in simulation mode.

    Returns:
        Dict[str, Any]: Standardized response dictionary containing:
            - timestamp: ISO 8601 string
            - src_ip: Source IP
            - attack_class: Classified attack
            - confidence: Rounded float (4 decimals)
            - threat_level: 'Low', 'Medium', 'High', or 'Critical'
            - model_name: Detecting model identifier
            - action: 'LOG', 'ALERT', 'RATE_LIMIT', 'BLOCK', or 'BLOCK_FAILED'
            - action_detail: Action explanation
            - blocked: Boolean flag indicating if blocked
    """
    threat_level = get_threat_level(confidence)
    base_action = THREAT_ACTIONS.get(threat_level, "LOG")
    now_iso = datetime.now().isoformat()
    conf_rounded = round(float(confidence), 4)

    is_valid_ip = validate_ip(src_ip)

    if threat_level == "Low":
        action = "LOG"
        detail = f"Flow logged for baseline statistical analysis. Confidence {conf_rounded:.2f} < 0.40."
        response_logger.info("ACTION=LOG | IP=%s | Class=%s | Conf=%.4f", src_ip, attack_class, conf_rounded)
        blocked = False

    elif threat_level == "Medium":
        action = "ALERT"
        detail = f"Security event dispatched to SIEM and SOC dashboard. Class: {attack_class}."
        response_logger.info("ACTION=ALERT | IP=%s | Class=%s | Conf=%.4f", src_ip, attack_class, conf_rounded)
        blocked = False

    elif threat_level == "High":
        action = "RATE_LIMIT"
        detail = f"Rate limiting rule enforced: traffic from {src_ip} restricted to {RATE_LIMIT_CEILING}."
        response_logger.warning("ACTION=RATE_LIMIT | IP=%s | Limit=%s", src_ip, RATE_LIMIT_CEILING)
        blocked = False

    else:  # Critical
        if not is_valid_ip:
            action = "BLOCK_FAILED"
            detail = f"IP validation failed for candidate '{src_ip}'. Blocking aborted to prevent injection."
            response_logger.error("ACTION=BLOCK_FAILED | Invalid IP='%s'", src_ip)
            blocked = False
        else:
            success, msg, _ = block_ip_iptables(src_ip, dry_run=dry_run)
            action = "BLOCK"
            detail = msg
            blocked = success

    return {
        "timestamp": now_iso,
        "src_ip": src_ip,
        "attack_class": attack_class,
        "confidence": conf_rounded,
        "threat_level": threat_level,
        "model_name": model_name,
        "action": action,
        "action_detail": detail,
        "blocked": blocked,
    }


def reset_blocked_ips() -> None:
    """Clear in-memory blocked IP set (useful for test resets)."""
    BLOCKED_IPS.clear()
