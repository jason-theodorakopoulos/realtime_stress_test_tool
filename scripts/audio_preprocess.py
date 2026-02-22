#!/usr/bin/env python3
"""Convert MP3 files to raw PCM16 mono at a given sample rate.

Usage:
    python scripts/audio_preprocess.py [--input-dir ./audio] [--output-dir ./audio_pcm24k] [--rate 24000]

Requires ``pydub`` and ``ffmpeg`` (or ``avconv``) installed on PATH.
"""

import argparse
import glob
import os
import sys

from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize

# Target peak headroom in dB (keeps peaks just below 0 dBFS).
DEFAULT_HEADROOM_DB = 1.0


def convert_mp3_to_pcm(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 24000,
    do_normalize: bool = True,
    headroom_db: float = DEFAULT_HEADROOM_DB,
) -> list[str]:
    """Convert every MP3 in *input_dir* to raw PCM16 mono in *output_dir*.

    When *do_normalize* is True (default) each file is peak-normalised so
    the loudest sample sits *headroom_db* dB below 0 dBFS.  This ensures
    the audio is loud enough for the realtime model to understand.

    Returns list of output file paths.
    """
    pattern = os.path.join(input_dir, "*.mp3")
    mp3_files = sorted(glob.glob(pattern))
    if not mp3_files:
        print(f"No .mp3 files found in {input_dir!r}", file=sys.stderr)
        return []

    os.makedirs(output_dir, exist_ok=True)
    outputs: list[str] = []

    for mp3_path in mp3_files:
        basename = os.path.splitext(os.path.basename(mp3_path))[0]
        out_path = os.path.join(output_dir, f"{basename}.raw")

        seg = AudioSegment.from_mp3(mp3_path)
        before_dbfs = seg.dBFS
        seg = seg.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)

        if do_normalize:
            seg = pydub_normalize(seg, headroom=headroom_db)

        raw_data = seg.raw_data
        with open(out_path, "wb") as f:
            f.write(raw_data)

        duration_s = len(seg) / 1000.0
        norm_info = f", {before_dbfs:.1f} -> {seg.dBFS:.1f} dBFS" if do_normalize else ""
        print(
            f"  {os.path.basename(mp3_path)} -> {os.path.basename(out_path)} "
            f"({duration_s:.1f}s, {len(raw_data)} bytes{norm_info})"
        )
        outputs.append(out_path)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Convert MP3 to raw PCM16.")
    parser.add_argument(
        "--input-dir", default="./audio", help="Directory with .mp3 files"
    )
    parser.add_argument(
        "--output-dir", default="./audio_pcm24k", help="Output directory for .raw files"
    )
    parser.add_argument(
        "--rate", type=int, default=24000, help="Target sample rate (16000 or 24000)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip peak-normalization (audio is used as-is)",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=DEFAULT_HEADROOM_DB,
        help=f"Peak headroom in dB below 0 dBFS (default {DEFAULT_HEADROOM_DB})",
    )
    args = parser.parse_args()

    norm_label = "(no normalization)" if args.no_normalize else f"(normalize, headroom={args.headroom} dB)"
    print(f"Converting MP3 files in {args.input_dir} -> {args.output_dir} at {args.rate} Hz {norm_label}")
    results = convert_mp3_to_pcm(
        args.input_dir,
        args.output_dir,
        args.rate,
        do_normalize=not args.no_normalize,
        headroom_db=args.headroom,
    )
    if results:
        print(f"Done — {len(results)} file(s) converted.")
    else:
        print("No files converted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
