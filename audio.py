"""Audio chunking and base64 encoding for Voice Live API streaming."""

import base64
import glob
import os


# PCM16 mono: bytes_per_ms = sample_rate * 2 / 1000
BYTES_PER_MS = {
    16000: 32,  # 16kHz: 16000 * 2 / 1000
    24000: 48,  # 24kHz: 24000 * 2 / 1000
}


def load_pcm_files(audio_dir: str) -> list[bytes]:
    """Load all .raw PCM files from *audio_dir*, sorted by name."""
    pattern = os.path.join(audio_dir, "*.raw")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No .raw PCM files found in {audio_dir!r}. "
            "Run scripts/audio_preprocess.py first."
        )
    buffers: list[bytes] = []
    for p in paths:
        with open(p, "rb") as f:
            buffers.append(f.read())
    return buffers


def chunk_audio(pcm_data: bytes, chunk_ms: int, sample_rate: int = 24000) -> list[bytes]:
    """Split *pcm_data* into chunks of *chunk_ms* milliseconds.

    Returns a list of byte chunks.  The last chunk may be shorter.
    """
    bpm = BYTES_PER_MS.get(sample_rate)
    if bpm is None:
        raise ValueError(
            f"Unsupported sample_rate {sample_rate}; supported: {sorted(BYTES_PER_MS)}"
        )
    chunk_bytes = bpm * chunk_ms
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_ms must be > 0, got {chunk_ms}")
    return [pcm_data[i : i + chunk_bytes] for i in range(0, len(pcm_data), chunk_bytes)]


def encode_chunk(chunk: bytes) -> str:
    """Base64-encode a raw PCM chunk for the Voice Live API."""
    return base64.b64encode(chunk).decode("ascii")
