"""Tests for audio.py — chunking and encoding helpers."""

import base64
import os
import tempfile

import pytest

from audio import BYTES_PER_MS, chunk_audio, encode_chunk, load_pcm_files


# ---------------------------------------------------------------------------
# chunk_audio
# ---------------------------------------------------------------------------

class TestChunkAudio:
    def test_exact_division(self):
        """When data length is exact multiple of chunk size, no short tail."""
        bpm = BYTES_PER_MS[24000]  # 48
        chunk_ms = 20
        chunk_bytes = bpm * chunk_ms  # 960
        data = b"\x00" * (chunk_bytes * 3)
        chunks = chunk_audio(data, chunk_ms, sample_rate=24000)
        assert len(chunks) == 3
        assert all(len(c) == chunk_bytes for c in chunks)

    def test_short_tail(self):
        """Last chunk can be shorter than the full chunk size."""
        bpm = BYTES_PER_MS[24000]
        chunk_ms = 20
        chunk_bytes = bpm * chunk_ms
        data = b"\x00" * (chunk_bytes * 2 + 100)
        chunks = chunk_audio(data, chunk_ms, sample_rate=24000)
        assert len(chunks) == 3
        assert len(chunks[-1]) == 100

    def test_empty_data(self):
        chunks = chunk_audio(b"", 20, sample_rate=24000)
        assert chunks == []

    def test_16k_sample_rate(self):
        bpm = BYTES_PER_MS[16000]  # 32
        chunk_ms = 50
        chunk_bytes = bpm * chunk_ms  # 1600
        data = b"\xAB" * chunk_bytes
        chunks = chunk_audio(data, chunk_ms, sample_rate=16000)
        assert len(chunks) == 1
        assert len(chunks[0]) == chunk_bytes

    def test_unsupported_rate_raises(self):
        with pytest.raises(ValueError, match="Unsupported sample_rate"):
            chunk_audio(b"\x00" * 100, 20, sample_rate=44100)

    def test_zero_chunk_ms_raises(self):
        with pytest.raises(ValueError, match="chunk_ms must be > 0"):
            chunk_audio(b"\x00" * 100, 0, sample_rate=24000)


# ---------------------------------------------------------------------------
# encode_chunk
# ---------------------------------------------------------------------------

class TestEncodeChunk:
    def test_roundtrip(self):
        original = b"\x01\x02\x03\x04"
        encoded = encode_chunk(original)
        assert isinstance(encoded, str)
        assert base64.b64decode(encoded) == original

    def test_empty(self):
        assert encode_chunk(b"") == ""


# ---------------------------------------------------------------------------
# load_pcm_files
# ---------------------------------------------------------------------------

class TestLoadPcmFiles:
    def test_loads_sorted(self, tmp_path):
        (tmp_path / "b.raw").write_bytes(b"\x02" * 10)
        (tmp_path / "a.raw").write_bytes(b"\x01" * 5)
        buffers = load_pcm_files(str(tmp_path))
        assert len(buffers) == 2
        assert buffers[0] == b"\x01" * 5  # 'a' comes first
        assert buffers[1] == b"\x02" * 10

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .raw PCM files"):
            load_pcm_files(str(tmp_path))

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pcm_files("/nonexistent/path/12345")
