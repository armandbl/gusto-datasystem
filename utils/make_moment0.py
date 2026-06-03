#!/usr/bin/env python3
"""Create a moment-0 map from a 3D FITS cube."""

from __future__ import annotations

import argparse
from pathlib import Path

from astropy.io import fits

from measure_mixer_crosscorr_moment0 import build_moment0_map, load_cube, moment0_header


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a moment-0 FITS image from a cube")
    parser.add_argument("--input", required=True, help="Input 3D FITS cube")
    parser.add_argument("--output", required=True, help="Output moment-0 FITS path")
    parser.add_argument("--ext", type=int, default=0, help="FITS extension index to read")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    cube, header = load_cube(input_path, ext=args.ext)
    moment0 = build_moment0_map(cube)
    out_header = moment0_header(header)
    out_header["HISTORY"] = f"Moment 0 image generated from {input_path.name}"
    fits.writeto(output_path, moment0, out_header, overwrite=True)

    print(f"Wrote moment-0 FITS: {output_path}")


if __name__ == "__main__":
    main()
