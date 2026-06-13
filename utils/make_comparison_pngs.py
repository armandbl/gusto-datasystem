#!/usr/bin/env python3
"""Render 3-panel comparison PNGs at specific velocity slices from two FITS cubes.

For each requested velocity the script picks the nearest channel in both cubes
and saves a 3-panel PNG (Reference | Target | Difference) in the run's Compare
directory.

Example:
  source Perso/gusto_workon.sh
  python utils/make_comparison_pngs.py \
    --ref "Data/level2/G337/run 18/G337_CII_8_reference.fits" \
    --tgt "Data/level2/G337/run 18/G337_CII_5_matched.fits" \
    --velocities -160 -120 -80 -40 0

Output goes to:
  Data/level2/G337/run 18/Compare/comparison_CII_M8_vs_M5/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz_helpers import (
    intensity_limits,
    line_from_filename,
    load_cube,
    mixer_from_filename,
    spectral_axis_mps,
    velocity_label,
)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_comparison_png(
    ref_slice: np.ndarray,
    tgt_slice: np.ndarray,
    diff_slice: np.ndarray,
    velocity_mps: float,
    ref_label: str,
    tgt_label: str,
    vmin: float,
    vmax: float,
    out_path: Path,
) -> None:
    """Render a single 3-panel PNG: Reference | Target | Difference."""

    diff_abs_max = max(abs(np.nanmin(diff_slice)), abs(np.nanmax(diff_slice)))
    if diff_abs_max == 0 or not np.isfinite(diff_abs_max):
        diff_abs_max = vmax * 0.1

    dpi = 120
    fig = plt.figure(figsize=(18, 5.5), dpi=dpi)
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 0.04, 1, 0.04, 1, 0.04])

    ax0 = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[0, 2])
    cax1 = fig.add_subplot(gs[0, 3])
    ax2 = fig.add_subplot(gs[0, 4])
    cax2 = fig.add_subplot(gs[0, 5])

    # Panel 1: Reference
    im0 = ax0.imshow(ref_slice, origin="lower", cmap="inferno",
                     vmin=vmin, vmax=vmax)
    ax0.set_title(f"Reference\n{ref_label}", fontsize=10)
    fig.colorbar(im0, cax=cax0)

    # Panel 2: Target
    im1 = ax1.imshow(tgt_slice, origin="lower", cmap="inferno",
                     vmin=vmin, vmax=vmax)
    ax1.set_title(f"Target\n{tgt_label}", fontsize=10)
    fig.colorbar(im1, cax=cax1)

    # Panel 3: Difference
    im2 = ax2.imshow(diff_slice, origin="lower", cmap="RdBu_r",
                     vmin=-diff_abs_max, vmax=diff_abs_max)
    ax2.set_title("Difference\n(target − reference)", fontsize=10)
    fig.colorbar(im2, cax=cax2)

    for ax in (ax0, ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    vel_kms = velocity_mps / 1000.0
    fig.suptitle(f"v = {vel_kms:+.3f} km/s  ({velocity_mps:+.0f} m/s)",
                 fontsize=14, y=0.98)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render 3-panel comparison PNGs at specific velocity slices."
    )
    parser.add_argument(
        "--ref", required=True, type=Path,
        help="Path to the reference FITS cube.",
    )
    parser.add_argument(
        "--tgt", required=True, type=Path,
        help="Path to the target FITS cube (to compare against reference).",
    )
    parser.add_argument(
        "--velocities", nargs="+", type=float, required=True,
        help="Velocity values in m/s, e.g. --velocities -160 -120 -80 -40 0",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <ref_parent>/Compare/comparison_<LINE>_M<A>_vs_M<B>/).",
    )
    parser.add_argument(
        "--imin", type=float, default=None,
        help="Override the minimum intensity for the color scale.",
    )
    parser.add_argument(
        "--imax", type=float, default=None,
        help="Override the maximum intensity for the color scale.",
    )
    parser.add_argument(
        "--percentile", type=float, nargs=2, default=[5, 99.5],
        help="Percentile range for auto color scaling (default: 5 99.5).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ref_path = args.ref.resolve()
    tgt_path = args.tgt.resolve()

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference cube not found: {ref_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Target cube not found: {tgt_path}")

    # --- Load cubes ---
    print(f"Loading reference: {ref_path.name}")
    ref_cube, ref_header = load_cube(ref_path)
    print(f"Loading target:    {tgt_path.name}")
    tgt_cube, _ = load_cube(tgt_path)

    if ref_cube.shape != tgt_cube.shape:
        raise ValueError(
            f"Cube shape mismatch: ref={ref_cube.shape}  tgt={tgt_cube.shape}"
        )

    # --- Spectral axis ---
    nchan = ref_cube.shape[0]
    vel_mps = spectral_axis_mps(ref_header, nchan)

    # --- Map velocities to channels ---
    velocities: List[float] = sorted(args.velocities)
    chan_map: List[Tuple[float, int, float]] = []  # (target_mps, chan, actual_mps)
    seen: set[int] = set()

    for target_vel in velocities:
        idx = int(np.argmin(np.abs(vel_mps - target_vel)))
        if idx not in seen:
            seen.add(idx)
            chan_map.append((target_vel, idx, float(vel_mps[idx])))

    if not chan_map:
        raise ValueError(
            f"No channels found for velocities {velocities}. "
            f"Cube spectral extent: [{vel_mps[0]:.0f}, {vel_mps[-1]:.0f}] m/s"
        )

    print(f"\nVelocity slices ({len(chan_map)}):")
    for target_vel, idx, actual_mps in chan_map:
        print(f"  {target_vel:+.0f} m/s  →  chan {idx}  "
              f"(actual {actual_mps:+.1f} m/s = {actual_mps/1000:+.3f} km/s)")

    # --- Line detection ---
    ref_line = line_from_filename(ref_path) or "???"
    tgt_line = line_from_filename(tgt_path) or "???"
    line = ref_line if ref_line != "???" else tgt_line
    ref_mx = mixer_from_filename(ref_path) or "?"
    tgt_mx = mixer_from_filename(tgt_path) or "?"

    # --- Intensity limits ---
    vmin_intensity, vmax_intensity = intensity_limits(line)
    if args.imin is not None:
        vmin_intensity = args.imin
    if args.imax is not None:
        vmax_intensity = args.imax
    if vmin_intensity == -1.0 and vmax_intensity == 1.0 and line == "???":
        valid = np.isfinite(ref_cube)
        if valid.any():
            vmin_intensity, vmax_intensity = np.percentile(
                ref_cube[valid],
                [float(args.percentile[0]), float(args.percentile[1])],
            )

    print(f"Intensity limits: [{vmin_intensity:.2f}, {vmax_intensity:.2f}]")

    # --- Output directory ---
    if args.output_dir:
        out_dir = args.output_dir.resolve()
    else:
        compare_dir = ref_path.parent / "Compare"
        slug = f"comparison_{line}_M{ref_mx}_vs_M{tgt_mx}"
        out_dir = compare_dir / slug

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # --- Labels ---
    ref_label = f"{line} M{ref_mx}"
    tgt_label = f"{line} M{tgt_mx}"

    # --- Render ---
    print(f"\nRendering {len(chan_map)} PNGs ...")
    for i, (target_vel, chan, actual_mps) in enumerate(chan_map):
        ref_slice = ref_cube[chan]
        tgt_slice = tgt_cube[chan]
        diff_slice = tgt_slice - ref_slice

        safe_vel = velocity_label(target_vel)
        fname = f"comparison_{line}_M{ref_mx}_vs_M{tgt_mx}_v{safe_vel}mps.png"
        out_path = out_dir / fname

        render_comparison_png(
            ref_slice=ref_slice,
            tgt_slice=tgt_slice,
            diff_slice=diff_slice,
            velocity_mps=actual_mps,
            ref_label=ref_label,
            tgt_label=tgt_label,
            vmin=vmin_intensity,
            vmax=vmax_intensity,
            out_path=out_path,
        )
        print(f"  [{i+1}/{len(chan_map)}] {out_path.name}")

    print(f"\nDone — {len(chan_map)} PNGs written to {out_dir}/")


if __name__ == "__main__":
    main()
