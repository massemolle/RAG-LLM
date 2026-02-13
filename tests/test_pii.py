"""
Tests for the unified PII detection and redaction module (utils.pii).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pii import (
    PII_PATTERNS,
    detect_pii_regex,
    redact_pii_regex,
)


class TestPIIPatterns:
    """Verify that the canonical pattern list covers all expected PII types."""

    def test_pattern_names(self):
        names = {name for name, _ in PII_PATTERNS}
        expected = {
            "email", "phone_lux", "phone_intl", "credit_card",
            "ssn", "iban", "ip_address", "passport", "api_key",
        }
        assert expected.issubset(names), f"Missing patterns: {expected - names}"


class TestDetectPIIRegex:
    """Test regex-based PII detection."""

    @pytest.mark.parametrize("text,expected_type", [
        ("Contact john@example.com for info", "email"),
        ("Call +352 621 123 456 now", "phone_lux"),
        ("Call +1-555-123-4567 now", "phone_intl"),
        ("Card: 4111-1111-1111-1111", "credit_card"),
        ("SSN: 123-45-6789", "ssn"),
        ("IBAN: LU280019400644750000", "iban"),
        ("Server at 192.168.1.1 is down", "ip_address"),
        ("Passport AB1234567", "passport"),
        ("api_key = abcdef1234567890abcdef", "api_key"),
    ])
    def test_detect_known_pii(self, text, expected_type):
        entities, redacted = detect_pii_regex(text)
        types_found = {e["type"] for e in entities}
        assert expected_type in types_found, (
            f"Expected '{expected_type}' in {types_found} for: {text}"
        )

    def test_no_false_positive_on_clean_text(self):
        clean = "The quick brown fox jumps over the lazy dog."
        entities, redacted = detect_pii_regex(clean)
        assert len(entities) == 0
        assert redacted == clean


class TestRedactPIIRegex:
    """Test regex-based PII redaction."""

    def test_email_redacted(self):
        result = redact_pii_regex("Send to alice@corp.lu please")
        assert "[REDACTED_EMAIL]" in result
        assert "alice@corp.lu" not in result

    def test_credit_card_redacted(self):
        result = redact_pii_regex("Card 4111 1111 1111 1111 on file")
        assert "[REDACTED_CREDIT_CARD]" in result
        assert "4111" not in result

    def test_multiple_pii_types(self):
        text = "Email: test@x.com, SSN: 123-45-6789"
        result = redact_pii_regex(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_SSN]" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
