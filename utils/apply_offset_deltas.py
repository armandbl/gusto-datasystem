#!/usr/bin/env python3
"""Apply summed delta offsets to a calibration table without AS_MEASURED rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_deltas(path: Path) -> dict[str, tuple[float, float]]:
    deltas: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pix_label = str(row.get("pix_label", "")).strip()
            if not pix_label:
                continue
            sum_az = float(row.get("sum_az_deg", 0.0) or 0.0)
            sum_el = float(row.get("sum_el_deg", 0.0) or 0.0)
            deltas[pix_label] = (sum_az, sum_el)
    return deltas


def apply_deltas_to_offsets(offsets_file: Path, deltas: dict[str, tuple[float, float]]) -> str:
    lines = offsets_file.read_text(encoding="utf-8").splitlines()
    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        output_lines.append(line)
        if not stripped or stripped.startswith("[") or stripped.upper().startswith("PIX"):
            continue

        cols = stripped.split()
        if len(cols) < 4:
            continue

        pix_label = cols[0]
        if cols[3].upper() not in {"THEORY", "FIDUCIAL"}:
            continue

        delta = deltas.get(pix_label)
        if delta is None:
            continue

        base_az = float(cols[1])
        base_el = float(cols[2])
        new_az = base_az + delta[0]
        new_el = base_el + delta[1]
        output_lines.append(f"{pix_label}\t{new_az:.6f}\t{new_el:.6f}\tDELTA_APPLIED")

    return "\n".join(output_lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply summed delta offsets to a calibration file")
    parser.add_argument("--offsets", required=True, help="Base offsets text file")
    parser.add_argument("--deltas", required=True, help="Delta CSV file from measure_mixer_crosscorr.py")
    parser.add_argument("--output", required=True, help="Output offsets file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    offsets_path = Path(args.offsets).resolve()
    deltas_path = Path(args.deltas).resolve()
    output_path = Path(args.output).resolve()

    deltas = read_deltas(deltas_path)
    output_text = apply_deltas_to_offsets(offsets_path, deltas)
    output_path.write_text(output_text, encoding="utf-8")

    print(f"Wrote adjusted offsets file: {output_path}")


if __name__ == "__main__":
    main()
