#!/usr/bin/env python3
"""Verify alignment quality by cross-correlating gridded mixer cubes.

Measures the residual pixel offset between each target mixer and its
reference mixer in a set of gridded cubes.  After a successful alignment
iteration the measured offsets should be close to (0, 0).  Results are
appended to a tracking CSV so you can compare pipeline runs over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import the heavy-lifting functions directly from the measurement script so
# we stay in sync and avoid code duplication.
# ---------------------------------------------------------------------------
_UTILS = Path(__file__).resolve().parent
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from measure_mixer_crosscorr import (  # type: ignore[import-not-found]
    JobResult,
    build_moment0_map,
    find_latest_run_dir,
    galactic_offset_to_azel,
    header_float,
    load_cube,
    measure_shift_integer,
    moment0_header,
    parse_line_and_mixer_from_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_reference_and_targets(
    run_dir: Path,
    line: str,
    reference_mixer: int,
    target_mixers: list[int],
) -> dict[int, tuple[Path, Path]]:
    """Return ``{target_mixer: (ref_cube_path, tgt_cube_path)}``."""
    line = line.upper()
    cubes: dict[int, Path] = {}

    for fpath in sorted(run_dir.glob("*.fits")):
        p_line, p_mixer = parse_line_and_mixer_from_name(fpath)
        if p_line == line and p_mixer is not None:
            cubes[p_mixer] = fpath

    ref_path = cubes.get(reference_mixer)
    if ref_path is None:
        raise FileNotFoundError(
            f"No cube found for {line} reference mixer M{reference_mixer}"
            f" in {run_dir}"
        )

    pairs: dict[int, tuple[Path, Path]] = {}
    for mx in target_mixers:
        tgt_path = cubes.get(mx)
        if tgt_path is None:
            print(f"  [skip] No cube for {line} M{mx} — skipping")
            continue
        pairs[mx] = (ref_path, tgt_path)

    return pairs


def compute_offset_metrics(
    ref_cube: np.ndarray,
    tgt_cube: np.ndarray,
    ref_header: Any,
    observer: tuple[float, float, float, str] | None,
) -> dict[str, object]:
    """Cross-correlate two cubes and return pixel / angular offset metrics."""

    nchan = min(ref_cube.shape[0], tgt_cube.shape[0])
    ref = ref_cube[:nchan]
    tgt = tgt_cube[:nchan]

    map_ref = build_moment0_map(ref)
    map_tgt = build_moment0_map(tgt)

    lag_x, lag_y, _corr = measure_shift_integer(map_ref, map_tgt)
    dx_pix = -float(lag_x)
    dy_pix = -float(lag_y)

    offset_pix = float(np.sqrt(dx_pix**2 + dy_pix**2))

    az_deg = float("nan")
    el_deg = float("nan")
    offset_arcsec = float("nan")

    if observer is not None:
        cdelt1 = header_float(ref_header, "CDELT1", 0.0)
        cdelt2 = header_float(ref_header, "CDELT2", 0.0)
        dlon_deg = dx_pix * cdelt1
        dlat_deg = dy_pix * cdelt2
        ref_glon = header_float(ref_header, "CRVAL1", 0.0)
        ref_glat = header_float(ref_header, "CRVAL2", 0.0)
        try:
            az_deg, el_deg = galactic_offset_to_azel(
                observer[0], observer[1], observer[2], observer[3],
                ref_glon, ref_glat, dlon_deg, dlat_deg,
            )
            offset_arcsec = float(np.sqrt(az_deg**2 + el_deg**2)) * 3600.0
        except Exception:
            az_deg = float("nan")
            el_deg = float("nan")
            offset_arcsec = float("nan")

    return {
        "lag_x": lag_x,
        "lag_y": lag_y,
        "dx_pix": dx_pix,
        "dy_pix": dy_pix,
        "offset_pix": offset_pix,
        "az_deg": az_deg,
        "el_deg": el_deg,
        "offset_arcsec": offset_arcsec,
    }


# ---------------------------------------------------------------------------
# CSV tracking
# ---------------------------------------------------------------------------

TRACKING_FIELDS = [
    "timestamp",
    "run_label",
    "source",
    "line",
    "reference_mixer",
    "target_mixer",
    "dx_pix",
    "dy_pix",
    "offset_pix",
    "az_deg",
    "el_deg",
    "offset_arcsec",
    "pass_pix",
    "pass_arcsec",
]


def append_tracking_row(csv_path: Path, row: dict[str, object]) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TRACKING_FIELDS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify alignment by cross-correlating gridded cubes"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source name (e.g. G337). When given, auto-discovers the latest "
             "run directory under --data-root/<source>/.",
    )
    parser.add_argument(
        "--data-root",
        default="Data/level2",
        help="Root data directory for Level-2 cubes (default: Data/level2)",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the run directory (optional if --source is given)",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file with observer metadata",
    )
    parser.add_argument(
        "--label",
        help="Human-readable label for this pipeline run (auto-derived if omitted)",
    )
    parser.add_argument(
        "--output-csv",
        default="Perso/alignment_verification_log.csv",
        help="Path to the tracking CSV (default: Perso/alignment_verification_log.csv)",
    )
    parser.add_argument(
        "--threshold-pix", type=float, default=1.0,
        help="Pixel offset threshold for pass/fail (default: 1.0 pix)",
    )
    parser.add_argument(
        "--threshold-arcsec", type=float, default=30.0,
        help="Arcsecond offset threshold for pass/fail (default: 30 arcsec)",
    )
    parser.add_argument(
        "--line-targets",
        help='JSON string with line→target config, e.g. \'{"CII": [8, [5]], "NII": [3, [2,6]]}\'',
    )
    return parser


def resolve_line_targets(
    config: dict[str, object] | None,
    cli_targets: str | None,
) -> dict[str, tuple[int, list[int]]]:
    """Return ``{line: (reference_mixer, [target_mixer, ...])}``."""
    if cli_targets:
        raw = json.loads(cli_targets)
        return {
            line.upper(): (int(v[0]), [int(m) for m in v[1]])
            for line, v in raw.items()
        }
    if config and "line_targets" in config:
        lt = config["line_targets"]
        result: dict[str, tuple[int, list[int]]] = {}
        for line, entry in lt.items():
            line = line.upper()
            ref = int(entry.get("target_mixer", 8 if line == "CII" else 3))
            mixers = [int(m) for m in entry.get("mixers", entry.get("mixer", []))]
            result[line] = (ref, mixers)
        return result
    # Defaults
    return {
        "CII": (8, [5]),
        "NII": (3, [2, 6]),
    }


def resolve_observer(
    config: dict[str, object] | None,
) -> tuple[float, float, float, str] | None:
    if config:
        obs = config.get("observer")
        if isinstance(obs, dict):
            lat = obs.get("lat_deg")
            lon = obs.get("lon_deg")
            alt = obs.get("alt_m")
            time = obs.get("obs_time_utc")
            if all(v is not None for v in (lat, lon, alt, time)):
                return (float(lat), float(lon), float(alt), str(time))
    return None


def main() -> None:
    args = build_parser().parse_args()

    # Resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.source:
        repo_root = _UTILS.parent
        source_dir = (repo_root / args.data_root / args.source).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        run_dir = find_latest_run_dir(source_dir, "latest")
    else:
        build_parser().error("Either --run-dir or --source is required")

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config: dict[str, object] | None = None
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    # Determine source name from directory path  (e.g. …/G337/run 12 → G337)
    source = args.source or run_dir.parent.name
    label = args.label or run_dir.name
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    line_targets = resolve_line_targets(config, args.line_targets)
    observer = resolve_observer(config)

    repo_root = _UTILS.parent
    output_csv = (repo_root / args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Run dir:      {run_dir}")
    print(f"Source:       {source}")
    print(f"Label:        {label}")
    print(f"Tracking CSV: {output_csv}")
    print(f"Thresholds:   {args.threshold_pix:.1f} pix  /  {args.threshold_arcsec:.0f} arcsec")
    print(f"Observer:     {'configured' if observer else 'NONE (pixel-only scoring)'}")
    print()

    total_pairs = 0
    passed_pix = 0
    passed_arcsec = 0

    for line, (ref_mx, tgt_mxs) in sorted(line_targets.items()):
        pairs = find_reference_and_targets(run_dir, line, ref_mx, tgt_mxs)
        if not pairs:
            print(f"[{line}] No target cubes found — skipping\n")
            continue

        for tgt_mx, (ref_path, tgt_path) in sorted(pairs.items()):
            total_pairs += 1
            print(f"[{line} M{tgt_mx} vs M{ref_mx}] ", end="", flush=True)

            ref_cube, ref_header = load_cube(ref_path)
            tgt_cube, _ = load_cube(tgt_path)

            metrics = compute_offset_metrics(ref_cube, tgt_cube, ref_header, observer)

            pass_pix = bool(metrics["offset_pix"] <= args.threshold_pix)
            pass_arcsec = bool(
                not np.isnan(metrics["offset_arcsec"])
                and metrics["offset_arcsec"] <= args.threshold_arcsec
            )

            if pass_pix:
                passed_pix += 1
            if pass_arcsec:
                passed_arcsec += 1

            status_pix = "PASS" if pass_pix else "FAIL"
            status_arcsec = "PASS" if pass_arcsec else ("FAIL" if not np.isnan(metrics["offset_arcsec"]) else "N/A")
            print(
                f"dx={metrics['dx_pix']:+.1f} pix  "
                f"dy={metrics['dy_pix']:+.1f} pix  "
                f"|offset|={metrics['offset_pix']:.2f} pix [{status_pix}]  "
                f"AZ={metrics['az_deg']:+.5f}°  EL={metrics['el_deg']:+.5f}°  "
                f"|offset|={metrics['offset_arcsec']:.0f}\" [{status_arcsec}]"
            )

            row: dict[str, object] = {
                "timestamp": timestamp,
                "run_label": label,
                "source": source,
                "line": line,
                "reference_mixer": ref_mx,
                "target_mixer": tgt_mx,
                "dx_pix": f"{metrics['dx_pix']:.3f}",
                "dy_pix": f"{metrics['dy_pix']:.3f}",
                "offset_pix": f"{metrics['offset_pix']:.3f}",
                "az_deg": f"{metrics['az_deg']:.6f}" if not np.isnan(metrics["az_deg"]) else "",
                "el_deg": f"{metrics['el_deg']:.6f}" if not np.isnan(metrics["el_deg"]) else "",
                "offset_arcsec": f"{metrics['offset_arcsec']:.1f}" if not np.isnan(metrics["offset_arcsec"]) else "",
                "pass_pix": pass_pix,
                "pass_arcsec": pass_arcsec,
            }
            append_tracking_row(output_csv, row)

        print()

    print(f"Summary: {passed_pix}/{total_pairs} pixel-passes, "
          f"{passed_arcsec}/{total_pairs} arcsec-passes")
    print(f"Results appended to {output_csv}")


if __name__ == "__main__":
    main()
