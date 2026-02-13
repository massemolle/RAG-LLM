"""
Tests for the data classification module (rag.classification).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.classification import (
    DataClassification,
    classify_document,
    is_ingestible,
    rejection_reason,
    INGESTIBLE,
)


class TestDataClassification:
    """Test the DataClassification enum."""

    def test_all_levels_exist(self):
        assert DataClassification.PUBLIC.value == "public"
        assert DataClassification.ENTITY_INTERNAL.value == "entity_internal"
        assert DataClassification.GROUP_INTERNAL.value == "group_internal"
        assert DataClassification.CLASSIFIED.value == "classified"
        assert DataClassification.SECRET.value == "secret"


class TestClassifyDocument:
    """Test document classification logic."""

    def test_explicit_level_takes_priority(self):
        result = classify_document("/some/path/doc.txt", explicit_level="secret")
        assert result == DataClassification.SECRET

    def test_explicit_level_case_insensitive(self):
        result = classify_document("/doc.txt", explicit_level="PUBLIC")
        assert result == DataClassification.PUBLIC

    def test_folder_inference_secret(self):
        result = classify_document("/data/secret/credentials.txt")
        assert result == DataClassification.SECRET

    def test_folder_inference_classified(self):
        result = classify_document("/data/classified/report.pdf")
        assert result == DataClassification.CLASSIFIED

    def test_folder_inference_public(self):
        result = classify_document("/data/public/readme.txt")
        assert result == DataClassification.PUBLIC

    def test_folder_inference_confidential(self):
        result = classify_document("/data/confidential/doc.pdf")
        assert result == DataClassification.CLASSIFIED

    def test_default_classification(self):
        result = classify_document("/some/random/path/doc.txt")
        assert result == DataClassification.ENTITY_INTERNAL

    def test_unknown_explicit_falls_back(self):
        result = classify_document("/data/secret/doc.txt", explicit_level="unknown_level")
        # Falls back to folder inference -> secret
        assert result == DataClassification.SECRET


class TestIsIngestible:
    """Test ingestion eligibility."""

    @pytest.mark.parametrize("level,expected", [
        (DataClassification.PUBLIC, True),
        (DataClassification.ENTITY_INTERNAL, True),
        (DataClassification.GROUP_INTERNAL, True),
        (DataClassification.CLASSIFIED, False),
        (DataClassification.SECRET, False),
    ])
    def test_ingestible_levels(self, level, expected):
        assert is_ingestible(level) == expected


class TestRejectionReason:
    """Test rejection reason messages."""

    def test_secret_rejection(self):
        reason = rejection_reason(DataClassification.SECRET)
        assert "secret" in reason.lower()
        assert "allowed" in reason.lower()

    def test_classified_rejection(self):
        reason = rejection_reason(DataClassification.CLASSIFIED)
        assert "classified" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
