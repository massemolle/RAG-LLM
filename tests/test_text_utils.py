"""
Tests for shared text processing utilities (utils.text).
"""

import os
import pytest
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text import (
    sanitize_llm_output,
    strip_role_markers,
    sha256_hash,
    md5_hash,
    read_document,
    chunk_text,
)


class TestSanitizeLLMOutput:
    """Test LLM output sanitization."""

    def test_strips_im_end_tokens(self):
        assert "<|im_end|>" not in sanitize_llm_output("Hello<|im_end|>")

    def test_strips_question_answer_prefix(self):
        result = sanitize_llm_output("Answer: The sky is blue.")
        assert not result.startswith("Answer:")

    def test_de_stutters(self):
        result = sanitize_llm_output("the the quick brown fox")
        assert "the the" not in result

    def test_normalises_whitespace(self):
        result = sanitize_llm_output("Hello   world\t\tthere")
        assert "  " not in result
        assert "\t" not in result

    def test_empty_string(self):
        assert sanitize_llm_output("") == ""


class TestStripRoleMarkers:
    """Test role marker removal."""

    def test_strips_inst(self):
        assert "[INST]" not in strip_role_markers("[INST] Hello [/INST]")

    def test_strips_system_colon(self):
        result = strip_role_markers("SYSTEM: You are helpful")
        assert "SYSTEM:" not in result

    def test_strips_chat_markers(self):
        result = strip_role_markers("### Human: What is AI?")
        assert "### Human:" not in result

    def test_strips_html_comments(self):
        result = strip_role_markers("Hello <!-- hidden --> world")
        assert "hidden" not in result


class TestHashing:
    """Test hashing utilities."""

    def test_sha256_deterministic(self):
        assert sha256_hash("hello") == sha256_hash("hello")

    def test_sha256_different_inputs(self):
        assert sha256_hash("hello") != sha256_hash("world")

    def test_sha256_length(self):
        assert len(sha256_hash("test")) == 64

    def test_md5_deterministic(self):
        assert md5_hash("hello") == md5_hash("hello")

    def test_md5_length(self):
        assert len(md5_hash("test")) == 32


class TestReadDocument:
    """Test document reading."""

    def test_read_txt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello world")
            f.flush()
            result = read_document(f.name)
        os.unlink(f.name)
        assert result == "Hello world"

    def test_read_md_file(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Title\nContent")
            f.flush()
            result = read_document(f.name)
        os.unlink(f.name)
        assert "Title" in result

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("data")
            f.flush()
            result = read_document(f.name)
        os.unlink(f.name)
        assert result == ""


class TestChunkText:
    """Test text chunking."""

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_multiple_chunks(self):
        text = "A" * 2000
        chunks = chunk_text(text, size=800, overlap=120)
        assert len(chunks) > 1

    def test_overlap_works(self):
        text = "ABCDEFGHIJ" * 100  # 1000 chars
        chunks = chunk_text(text, size=500, overlap=100)
        # Second chunk should start 400 chars in (500 - 100)
        assert len(chunks) >= 2
        assert chunks[1][:100] == chunks[0][400:500]

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_text("   \n\n  ")
        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
