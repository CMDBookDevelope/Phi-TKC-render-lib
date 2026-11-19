#!/usr/bin/env python3
"""Reproduce the immutable layout/time planner model (not neural weights)."""

import argparse
import struct
from pathlib import Path


# Spatial coordinates are the game area's half-width; timing is in seconds.
# This hand-authored ergonomic policy uses geometry, recent travel, chords and
# hold reach. It does not claim training on human-labelled hand sequences.
PARAMETERS = (
    ("home", 0.45),
    ("spatial", 2.0),
    ("travel", 0.35),
    ("reuse", 0.9),
    ("reuse_window", 0.18),
    ("relax_time", 1.0),
    ("chord_epsilon", 0.003),
    ("hold_reach", 0.30),
    ("hold_penalty", 8.0),
    ("chord_extra", 0.6),
)


def build():
    data = b"PHITKHP\0" + struct.pack("<II", 1, len(PARAMETERS))
    data += struct.pack("<10f", *(value for _, value in PARAMETERS))
    checksum = 0x811C9DC5
    for byte in data:
        checksum = ((checksum ^ byte) * 0x01000193) & 0xFFFFFFFF
    return data + struct.pack("<I", checksum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets/phitk_ai_model.bin")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = build()
    if args.check:
        if args.output.read_bytes() != data:
            raise SystemExit("The packaged hand model differs from the reproducible policy")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    print(f"Layout/time model v1: {len(data)} bytes; {args.output}")
