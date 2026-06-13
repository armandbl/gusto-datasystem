#!/usr/bin/env python3
"""Make a 3-panel comparison movie from two FITS cubes over a velocity range.

Renders every channel between --vmin and --vmax as a 3-panel PNG
(Reference | Target | Difference), then stitches them into an MP4 with ffmpeg.

Example:
  source Perso/gusto_workon.sh
  python utils/make_comparison_movie.py \
    --ref "Data/level2/G337/run 18/G337_CII_8_reference.fits" \
    --tgt "Data/level2/G337/run 18/G337_CII_5_matched.fits" \
    --vmin -160 --vmax 0

Output goes to:
  Data/level2/G337/run 18/Compare/comparison_CII_M8_vs_M5.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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
)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(
    ref_slice: np.ndarray,
    tgt_slice: np.ndarray,
    diff_slice: np.ndarray,
    velocity_kms: float,
    ref_label: str,
    tgt_label: str,
    vmin: float,
    vmax: float,
    out_path: Path,
) -> None:
    """Render a single 3-panel PNG frame."""

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

    fig.suptitle(f"v = {velocity_kms:+.3f} km/s", fontsize=14, y=0.98)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make a 3-panel comparison movie from two FITS cubes "
                    "over a velocity range."
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
        "--vmin", required=True, type=float,
        help="Minimum velocity (m/s) for the movie range.",
    )
    parser.add_argument(
        "--vmax", required=True, type=float,
        help="Maximum velocity (m/s) for the movie range.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output movie path (default: <ref_parent>/Compare/comparison_<LINE>_M<A>_vs_M<B>.mp4).",
    )
    parser.add_argument(
        "--framerate", type=float, default=3,
        help="Frames per second in the output movie (default: 3).",
    )
    parser.add_argument(
        "--keep-frames", action="store_true",
        help="Keep the intermediate PNG frames after stitching.",
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
        help="Percentile range for auto color scaling when line is unknown (default: 5 99.5).",
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="Wait for Enter key before exiting (keeps terminal open in VS Code).",
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

    vmin_mps, vmax_mps = sorted([args.vmin, args.vmax])
    chan_mask = (vel_mps >= vmin_mps) & (vel_mps <= vmax_mps)
    chan_indices = np.where(chan_mask)[0]

    if len(chan_indices) == 0:
        raise ValueError(
            f"No channels in range [{vmin_mps:.0f}, {vmax_mps:.0f}] m/s. "
            f"Cube spectral extent: [{vel_mps[0]:.0f}, {vel_mps[-1]:.0f}] m/s"
        )

    print(f"\nSpectral range: {vmin_mps:.0f} → {vmax_mps:.0f} m/s  "
          f"({len(chan_indices)} channels)")

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

    # --- Output path ---
    if args.output:
        output_path = args.output.resolve()
    else:
        compare_dir = ref_path.parent / "Compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        output_path = compare_dir / f"comparison_{line}_M{ref_mx}_vs_M{tgt_mx}.mp4"

    if not str(output_path).endswith(".mp4"):
        output_path = output_path.with_suffix(".mp4")

    # --- Labels ---
    ref_label = f"{line} M{ref_mx}"
    tgt_label = f"{line} M{tgt_mx}"
    print(f"Reference: {ref_label}  ({ref_path.name})")
    print(f"Target:    {tgt_label}  ({tgt_path.name})")
    print(f"Output:    {output_path}")

    # --- Frames directory ---
    frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # --- Render frames ---
    print(f"\nRendering {len(chan_indices)} frames to {frames_dir}/ ...")
    frame_paths: List[Path] = []

    for frame_idx, chan in enumerate(chan_indices):
        ref_slice = ref_cube[chan]
        tgt_slice = tgt_cube[chan]
        diff_slice = tgt_slice - ref_slice

        vel_kms = vel_mps[chan] / 1000.0

        frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
        render_frame(
            ref_slice=ref_slice,
            tgt_slice=tgt_slice,
            diff_slice=diff_slice,
            velocity_kms=vel_kms,
            ref_label=ref_label,
            tgt_label=tgt_label,
            vmin=vmin_intensity,
            vmax=vmax_intensity,
            out_path=frame_path,
        )
        frame_paths.append(frame_path)

        if (frame_idx + 1) % 20 == 0 or frame_idx == len(chan_indices) - 1:
            print(f"  {frame_idx + 1}/{len(chan_indices)} frames rendered")

    # --- Stitch with ffmpeg ---
    # The "pad" filter ensures even pixel dimensions (libx264 requires this).
    print(f"\nStitching with ffmpeg ({len(chan_indices)} frames @ {args.framerate} fps) ...")
    input_pattern = str(frames_dir / "frame_%05d.png")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(args.framerate),
        "-i", input_pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        str(output_path),
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"\nERROR: ffmpeg failed with exit code {result.returncode}")
            print(result.stderr)
            print(f"\nPNG frames are still available at: {frames_dir}/")
            if args.pause:
                input("\nPress Enter to exit...")
            sys.exit(1)
    except FileNotFoundError:
        print("\nERROR: ffmpeg is not installed.")
        print("Install it with:  sudo apt install ffmpeg")
        print(f"\nPNG frames are available at: {frames_dir}/")
        if args.pause:
            input("\nPress Enter to exit...")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("\nERROR: ffmpeg timed out after 10 minutes")
        print(f"PNG frames are available at: {frames_dir}/")
        if args.pause:
            input("\nPress Enter to exit...")
        sys.exit(1)

    # --- Cleanup ---
    if not args.keep_frames:
        print("Cleaning up intermediate PNGs ...")
        for fp in frame_paths:
            fp.unlink(missing_ok=True)
        try:
            frames_dir.rmdir()
        except OSError:
            pass
    else:
        print(f"Intermediate frames kept at: {frames_dir}/")

    # --- Final output ---
    file_size_kb = output_path.stat().st_size / 1024
    print(f"\n{'='*60}")
    print(f"DONE — movie written to: {output_path}")
    print(f"Size: {file_size_kb:.0f} KB  |  Frames: {len(chan_indices)}  "
          f"|  Framerate: {args.framerate} fps")
    print(f"{'='*60}")

    if args.pause:
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
