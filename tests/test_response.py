"""Unit tests for the Active Response Engine and defensive security safeguards."""

import pytest
from smart_ids.response import (
    apply_active_response,
    block_ip_iptables,
    get_threat_level,
    reset_blocked_ips,
    validate_ip,
)


class TestThreatSeverityBoundaries:
    """Test get_threat_level boundary behavior across all 4 tiers."""

    def test_low_tier(self):
        assert get_threat_level(0.0) == "Low"
        assert get_threat_level(0.20) == "Low"
        assert get_threat_level(0.3999) == "Low"

    def test_medium_tier(self):
        assert get_threat_level(0.40) == "Medium"
        assert get_threat_level(0.55) == "Medium"
        assert get_threat_level(0.6999) == "Medium"

    def test_high_tier(self):
        assert get_threat_level(0.70) == "High"
        assert get_threat_level(0.80) == "High"
        assert get_threat_level(0.8999) == "High"

    def test_critical_tier(self):
        assert get_threat_level(0.90) == "Critical"
        assert get_threat_level(0.95) == "Critical"
        assert get_threat_level(0.9999) == "Critical"

    def test_exact_one_point_zero_boundary(self):
        """Ensure saturated softmax outputs of exactly 1.0 map to Critical."""
        assert get_threat_level(1.0) == "Critical"
        assert get_threat_level(1.000) == "Critical"


class TestIPValidationSafeguard:
    """Test validate_ip against legitimate addresses and command injection vectors."""

    @pytest.mark.parametrize(
        "valid_ip",
        [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.254",
            "8.8.8.8",
            "127.0.0.1",
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
            "::1",
        ],
    )
    def test_valid_ips(self, valid_ip):
        assert validate_ip(valid_ip) is True

    @pytest.mark.parametrize(
        "invalid_ip",
        [
            "INVALID_IP",
            "999.999.999.999",
            "192.168.1.1; cat /etc/passwd",
            "192.168.1.1 | rm -rf /",
            "192.168.1",
            "256.100.0.1",
            "",
            "   ",
            None,
            12345,
        ],
    )
    def test_invalid_ips(self, invalid_ip):
        assert validate_ip(invalid_ip) is False


class TestActiveResponseLogic:
    """Test response dictionary structure, idempotency, and injection handling."""

    def setup_method(self):
        reset_blocked_ips()

    def test_response_dictionary_schema(self):
        res = apply_active_response(
            src_ip="192.168.1.50",
            attack_class="PortScan",
            confidence=0.55,
            model_name="SmartIDS_MLP",
            dry_run=True,
        )
        expected_keys = {
            "timestamp",
            "src_ip",
            "attack_class",
            "confidence",
            "threat_level",
            "model_name",
            "action",
            "action_detail",
            "blocked",
        }
        assert set(res.keys()) == expected_keys
        assert res["threat_level"] == "Medium"
        assert res["action"] == "ALERT"
        assert res["blocked"] is False

    def test_critical_valid_ip_blocks(self):
        res = apply_active_response(
            src_ip="203.0.113.19",
            attack_class="DoS Hulk",
            confidence=0.98,
            dry_run=True,
        )
        assert res["threat_level"] == "Critical"
        assert res["action"] == "BLOCK"
        assert res["blocked"] is True
        assert "[DRY RUN]" in res["action_detail"]

    def test_critical_invalid_ip_fails_gracefully(self):
        """Verify command injection attempts are blocked with BLOCK_FAILED."""
        res = apply_active_response(
            src_ip="MALICIOUS_IP; reboot",
            attack_class="DDoS",
            confidence=0.99,
            dry_run=True,
        )
        assert res["threat_level"] == "Critical"
        assert res["action"] == "BLOCK_FAILED"
        assert res["blocked"] is False
        assert "IP validation failed" in res["action_detail"]

    def test_idempotency_prevents_duplicate_rules(self):
        ip = "198.51.100.77"
        # First block attempt
        success1, msg1, _ = block_ip_iptables(ip, dry_run=True)
        assert success1 is True
        assert "[DRY RUN]" in msg1

        # Second block attempt for identical IP
        success2, msg2, _ = block_ip_iptables(ip, dry_run=True)
        assert success2 is True
        assert "already blocked" in msg2.lower()
