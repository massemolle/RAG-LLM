"""
Tests for the secure ingestion pipeline (rag.ingest).

Uses temporary directories and files to avoid touching real data.
"""

import json
import os
import pytest
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingest import run_ingest


class TestRunIngest:
    """Test the run_ingest pipeline."""

    @pytest.fixture
    def tmp_workspace(self, tmp_path):
        """Create a temporary workspace with source docs and index dir."""
        src = tmp_path / "data"
        src.mkdir()
        idx = tmp_path / "index"
        idx.mkdir()
        return src, idx

    def _write_txt(self, src_dir, filename, content):
        """Helper to write a .txt file in the source directory."""
        path = src_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_ingest_clean_document(self, tmp_workspace, monkeypatch):
        src, idx = tmp_workspace
        self._write_txt(src, "clean.txt", "Embeddings are vector representations.")

        # Patch the index paths to use our temp dir
        monkeypatch.setattr("rag.ingest.os.makedirs", lambda *a, **kw: None)

        result = run_ingest(
            src=str(src),
            collection="test",
            chunk_size=100,
            chunk_overlap=20,
        )

        assert result["status"] == "completed"
        assert result["files_processed"] >= 1
        assert result["files_rejected"] == 0

    def test_ingest_rejects_secret_folder(self, tmp_workspace, monkeypatch):
        src, idx = tmp_workspace
        secret_dir = src / "secret"
        secret_dir.mkdir()
        self._write_txt(secret_dir, "creds.txt", "password=hunter2")

        result = run_ingest(
            src=str(src),
            collection="test",
            chunk_size=100,
            chunk_overlap=20,
        )

        assert result["files_rejected"] >= 1

    def test_ingest_explicit_classification(self, tmp_workspace, monkeypatch):
        src, idx = tmp_workspace
        self._write_txt(src, "doc.txt", "Normal document content.")

        result = run_ingest(
            src=str(src),
            collection="test",
            classification_level="public",
            chunk_size=100,
            chunk_overlap=20,
        )

        assert result["status"] == "completed"

    def test_ingest_returns_scan_summary(self, tmp_workspace, monkeypatch):
        src, idx = tmp_workspace
        self._write_txt(src, "test.txt", "Simple test document.")

        result = run_ingest(
            src=str(src),
            collection="test",
            chunk_size=100,
            chunk_overlap=20,
        )

        assert "scan_summary" in result
        assert isinstance(result["scan_summary"], list)

    def test_ingest_empty_directory(self, tmp_workspace, monkeypatch):
        src, idx = tmp_workspace
        # No files in src

        result = run_ingest(
            src=str(src),
            collection="test",
        )

        assert result["files_processed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
