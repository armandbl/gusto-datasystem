#!/usr/bin/env python3
"""Create blinking GIFs from 2–3 FITS cubes at specific velocity channels.

For each requested velocity, extracts the nearest channel from every cube and
creates an animated GIF that alternates between the views — replacing the DS9
blink feature for quick visual alignment checks.

Example:
  source Perso/gusto_workon.sh
  python utils/make_blink_gif.py \
      --cubes "Data/level2/G337/run 18/G337_CII_8_reference.fits" \
              "Data/level2/G337/run 18/G337_CII_5_matched.fits" \
      --velocities -70 -38 -40 -100 -20 -120

Output goes to:
  <cube_parent>/Compare/blink_<LINE>_M<A>_M<B>_v<vel>kms.gif
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from viz_helpers import (
    intensity_limits,
    line_from_filename,
    load_cube,
    mixer_from_filename,
    spatial_plot_metadata,
    spectral_axis_mps,
    velocity_slug,
)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame_to_pil(
    data: np.ndarray,
    header: fits.Header,
    label: str,
    velocity_mps: float,
    vmin: float,
    vmax: float,
    cmap: str = "inferno",
    figsize: Tuple[float, float] = (8, 6.5),
    dpi: int = 120,
) -> Image.Image:
    """Render a single 2D slice to a PIL Image.

    The velocity is shown in a suptitle so the user always knows which channel
    they are looking at, even when the GIF is shared standalone.
    """
    extent, xlabel, ylabel = spatial_plot_metadata(header, data)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=extent)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Intensity")
    ax.set_title(label, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    vel_kms = velocity_mps / 1000.0
    fig.suptitle(f"v = {vel_kms:+.2f} km/s  ({velocity_mps:+.0f} m/s)",
                 fontsize=13, y=0.99)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# GIF creation
# ---------------------------------------------------------------------------

def make_blink_gif(
    slices: List[Tuple[np.ndarray, fits.Header, str]],
    velocity_mps: float,
    vmin: float,
    vmax: float,
    output_path: Path,
    duration_ms: int = 500,
    cmap: str = "inferno",
) -> None:
    """Create an animated GIF that blinks between multiple 2D slices.

    Parameters
    ----------
    slices : list of (data, header, label)
        One entry per cube.  All slices must share the same velocity channel.
    velocity_mps : float
        Actual velocity of this channel (m/s), for the suptitle.
    vmin, vmax : float
        Shared intensity limits applied to every frame.
    output_path : Path
        Where to write the .gif file.
    duration_ms : int
        Milliseconds each frame is displayed.
    cmap : str
        Matplotlib colormap name.
    """
    frames: List[Image.Image] = []
    for data, header, label in slices:
        img = render_frame_to_pil(
            data=data,
            header=header,
            label=label,
            velocity_mps=velocity_mps,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        frames.append(img)

    # Save as animated GIF — infinite loop
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create blinking GIFs from 2–3 FITS cubes at specific "
                    "velocity channels."
    )
    parser.add_argument(
        "--cubes", nargs="+", type=Path, required=True,
        help="2 or 3 FITS cube paths to blink between.",
    )
    parser.add_argument(
        "--velocities", nargs="+", type=float, required=True,
        help="Velocity values in m/s, e.g. --velocities -70 -38 -40",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: Compare/ in the first cube's parent).",
    )
    parser.add_argument(
        "--duration", type=int, default=500,
        help="Milliseconds per frame in the GIF (default: 500).",
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
        "--cmap", type=str, default="inferno",
        help="Matplotlib colormap (default: inferno).",
    )
    parser.add_argument(
        "--percentile", type=float, nargs=2, default=[5, 99.5],
        help="Percentile range for auto color scaling when line is unknown "
             "(default: 5 99.5).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="If cubes have different spatial shapes, trim all to the smallest "
             "common dimensions instead of aborting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # --- Validate cube count ---
    if len(args.cubes) < 2 or len(args.cubes) > 3:
        raise ValueError(f"Expected 2 or 3 cubes, got {len(args.cubes)}")

    # --- Resolve and validate paths ---
    cube_paths = [p.resolve() for p in args.cubes]
    for p in cube_paths:
        if not p.exists():
            raise FileNotFoundError(f"Cube not found: {p}")

    # --- Load cubes ---
    print("Loading cubes:")
    cube_info: List[Tuple[np.ndarray, fits.Header, str, Path, str, str]] = []
    # (data, header, label, path, line, mixer)

    for cp in cube_paths:
        data, header = load_cube(cp)
        line = line_from_filename(cp) or "???"
        mixer = mixer_from_filename(cp) or "?"
        label = f"{line} M{mixer}"
        print(f"  {cp.name}  →  {label}  shape={data.shape}")
        cube_info.append((data, header, label, cp, line, mixer))

    # --- Verify spatial shapes match (or crop with --force) ---
    ref_data, ref_header, ref_label, ref_path, _, _ = cube_info[0]
    shapes = [(ref_data.shape, ref_label, ref_path)]
    for data, _, label, cp, _, _ in cube_info[1:]:
        shapes.append((data.shape, label, cp))

    all_same = all(s[0] == shapes[0][0] for s in shapes[1:])
    if not all_same:
        if args.force:
            # Trim all cubes to the smallest common spatial dimensions.
            # Only the Y/X axes (1,2) are cropped; the velocity axis (0)
            # must still be identical.
            nchan = ref_data.shape[0]
            min_ny = min(s[0][1] for s in shapes)
            min_nx = min(s[0][2] for s in shapes)
            for s in shapes[1:]:
                if s[0][0] != nchan:
                    raise ValueError(
                        f"Velocity axis mismatch: {shapes[0][0][0]} vs {s[0][0]} "
                        f"channels. --force requires identical velocity axes.\n"
                        f"  {shapes[0][1]}: {shapes[0][0]}\n"
                        f"  {s[1]}: {s[0]}"
                    )
            print(f"\n⚠  Spatial shapes differ — cropping all to {min_ny}×{min_nx} "
                  f"(smallest common).")
            for i, (shape, label, cp) in enumerate(shapes):
                if shape[1] != min_ny or shape[2] != min_nx:
                    print(f"     {label}: {shape[1]}×{shape[2]} → {min_ny}×{min_nx}")
                    # Update cube_info in place
                    cropped = cube_info[i][0][:, :min_ny, :min_nx].copy()
                    cube_info[i] = (cropped,) + cube_info[i][1:]
        else:
            print(f"\nERROR: cubes have different spatial shapes:")
            for shape, label, cp in shapes:
                print(f"  {label}: {shape[1]}×{shape[2]}  ({cp.name})")
            print(f"\nRe-run with --force to trim all cubes to the smallest "
                  f"common dimensions.")
            raise SystemExit(1)

    ref_data, ref_header, _, _, _, _ = cube_info[0]  # may have changed after crop

    # --- Determine line ---
    line = "???"
    for _, _, _, _, l, _ in cube_info:
        if l != "???":
            line = l
            break

    # --- Intensity limits (shared across all GIFs) ---
    vmin, vmax = intensity_limits(line)
    if args.imin is not None:
        vmin = args.imin
    if args.imax is not None:
        vmax = args.imax
    if vmin == -1.0 and vmax == 1.0 and line == "???":
        all_valid = []
        for data, _, _, _, _, _ in cube_info:
            valid = data[np.isfinite(data)]
            if valid.size > 0:
                all_valid.append(valid.ravel())
        if all_valid:
            combined = np.concatenate(all_valid)
            vmin = float(np.percentile(combined, args.percentile[0]))
            vmax = float(np.percentile(combined, args.percentile[1]))
    print(f"Intensity limits: [{vmin:.2f}, {vmax:.2f}]")

    # --- Output directory ---
    if args.output_dir:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = cube_paths[0].parent / "Compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # --- Spectral axis (from first cube) ---
    nchan = ref_data.shape[0]
    vel_mps = spectral_axis_mps(ref_header, nchan)

    # --- Mixer slug for filenames ---
    mixers = [m for _, _, _, _, _, m in cube_info]
    mixer_slug = "_".join(f"M{m}" for m in mixers)

    # --- Render one GIF per velocity ---
    print(f"\nRendering {len(args.velocities)} GIFs "
          f"({args.duration} ms/frame, {len(cube_info)} frames each):")

    for target_vel in sorted(args.velocities):
        chan = int(np.argmin(np.abs(vel_mps - target_vel)))
        actual_mps = float(vel_mps[chan])

        # Build slice list for this channel
        slices: List[Tuple[np.ndarray, fits.Header, str]] = []
        for data, header, label, _, _, _ in cube_info:
            slices.append((data[chan], header, label))

        # Filename
        vel_str = velocity_slug(target_vel)
        fname = f"blink_{line}_{mixer_slug}_{vel_str}mps.gif"
        out_path = out_dir / fname

        make_blink_gif(
            slices=slices,
            velocity_mps=actual_mps,
            vmin=vmin,
            vmax=vmax,
            output_path=out_path,
            duration_ms=args.duration,
            cmap=args.cmap,
        )

        vel_kms = actual_mps / 1000.0
        target_kms = target_vel / 1000.0
        print(f"  {fname}  (target={target_kms:+.1f} km/s, "
              f"actual={vel_kms:+.3f} km/s, chan={chan})")

    # --- Summary ---
    total_frames = len(args.velocities) * len(cube_info)
    print(f"\nDone — {len(args.velocities)} GIFs ({total_frames} total frames) "
          f"written to {out_dir}/")


if __name__ == "__main__":
    main()
