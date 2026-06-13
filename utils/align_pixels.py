#!/usr/bin/env python3
"""Interactive (or scripted) pixel alignment of two FITS images.

Optimises the Galactic (lon, lat) shift that minimises the absolute
difference between two images, then converts the result to AZ/EL offsets
using the full astropy coordinate chain.

Supports both an interactive rectangle-selection mode (for exploratory
work) and a fully non-interactive CLI mode (for scripting / pipelines).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

# Use the canonical coordinate conversion from the alignment pipeline
# instead of maintaining a duplicate here.
_UTILS = Path(__file__).resolve().parent
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from measure_mixer_crosscorr import galactic_offset_to_azel  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------


def load_fits(file_path: Path) -> tuple[np.ndarray, WCS]:
    """Load a 2-D FITS image, returning ``(data, wcs)``."""
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
    return data, wcs


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------


def click_rectangle(ax: plt.Axes, title: str) -> list[tuple[float, float]]:
    """Let the user select two corners of a rectangle by clicking."""
    ax.set_title(title)
    coords = plt.ginput(2, timeout=120)
    plt.close()
    return coords  # type: ignore[no-any-return]


def crop_image(
    data: np.ndarray,
    wcs: WCS,
    coords: list[tuple[float, float]],
) -> tuple[np.ndarray, WCS]:
    """Crop *data* and *wcs* to the selected rectangular region."""
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x_min, x_max = int(min(x1, x2)), int(max(x1, x2))
    y_min, y_max = int(min(y1, y2)), int(max(y1, y2))

    cropped_data = data[y_min:y_max, x_min:x_max]
    cropped_wcs = wcs.deepcopy()
    cropped_wcs.wcs.crpix[0] -= x_min
    cropped_wcs.wcs.crpix[1] -= y_min
    cropped_wcs.wcs.set()

    return cropped_data, cropped_wcs


# ---------------------------------------------------------------------------
# Alignment optimisation
# ---------------------------------------------------------------------------


def align_images(
    data1: np.ndarray,
    wcs1: WCS,
    data2: np.ndarray,
    wcs2: WCS,
    initial_shift: tuple[float, float] = (0.0, 0.0),
):
    """Optimise the Galactic (lon, lat) shift to minimise image residuals."""

    def misalignment_error(shift: np.ndarray) -> float:
        lon_shift, lat_shift = float(shift[0]), float(shift[1])
        wcs2_shifted = wcs2.deepcopy()
        wcs2_shifted.wcs.crval[0] += lon_shift
        wcs2_shifted.wcs.crval[1] += lat_shift
        wcs2_shifted.wcs.set()

        y_indices, x_indices = np.indices(data2.shape)
        world_coords = wcs2_shifted.wcs_pix2world(x_indices, y_indices, 0)
        pix_coords = wcs1.wcs_world2pix(world_coords[0], world_coords[1], 0)

        valid = (
            np.isfinite(pix_coords[0])
            & np.isfinite(pix_coords[1])
            & (pix_coords[0] >= 0)
            & (pix_coords[0] <= (data1.shape[1] - 1))
            & (pix_coords[1] >= 0)
            & (pix_coords[1] <= (data1.shape[0] - 1))
            & np.isfinite(data2)
        )
        if not np.any(valid):
            return np.inf

        aligned = map_coordinates(
            data1,
            [pix_coords[1], pix_coords[0]],
            order=1,
            mode="constant",
            cval=0.0,
        )

        valid &= np.isfinite(aligned)
        if not np.any(valid):
            return np.inf
        return float(np.sum(np.abs(data2[valid] - aligned[valid])))

    return minimize(misalignment_error, np.array(initial_shift), method="Nelder-Mead")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align two FITS images by optimising Galactic coordinate shifts "
                    "and computing AZ/EL offsets.",
    )
    parser.add_argument(
        "file1", nargs="?", type=Path,
        help="Path to the first (reference) FITS image.",
    )
    parser.add_argument(
        "file2", nargs="?", type=Path,
        help="Path to the second (target) FITS image.",
    )
    parser.add_argument(
        "--file1", dest="f1", type=Path,
        help="Path to the first (reference) FITS image (named form).",
    )
    parser.add_argument(
        "--file2", dest="f2", type=Path,
        help="Path to the second (target) FITS image (named form).",
    )
    parser.add_argument(
        "--time",
        default="2024-02-04 13:02:32",
        help="Observer time in UTC (default: %(default)s)",
    )
    parser.add_argument(
        "--lat", type=float, default=-74.735938,
        help="Observer latitude in degrees (default: %(default)s)",
    )
    parser.add_argument(
        "--lon", type=float, default=61.630017,
        help="Observer longitude in degrees (default: %(default)s)",
    )
    parser.add_argument(
        "--alt", type=float, default=35172.0,
        help="Observer altitude in meters (default: %(default)s)",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip rectangle selection — align the full images.",
    )
    parser.add_argument(
        "--rectangle", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
        help="Pre-set rectangle corners (overrides interactive selection).",
    )
    parser.add_argument(
        "--output-plot", type=Path,
        help="Save the first image with rectangle overlay to this path.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve file paths — positional or named
    file1 = args.file1 or args.f1
    file2 = args.file2 or args.f2

    if file1 is None:
        file1 = Path(input("Enter the path to the first FITS file: ") or "../level2/src/align_b2m8.fits")
    if file2 is None:
        file2 = Path(input("Enter the path to the second FITS file: ") or "../level2/src/align_b2m5.fits")

    data1, wcs1 = load_fits(file1)
    data2, wcs2 = load_fits(file2)

    # --- Rectangle selection ---
    if args.rectangle:
        coords = [(args.rectangle[0], args.rectangle[1]),
                  (args.rectangle[2], args.rectangle[3])]
    elif args.non_interactive:
        # Use full image — no cropping
        coords = None
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(data1, origin="lower", cmap="gray")
        coords = click_rectangle(ax, "Click two corners to define a rectangle")
        print(f"Selected rectangle: {coords}")

        if args.output_plot:
            fig.savefig(args.output_plot)
            print(f"Rectangle selection saved to {args.output_plot}")

    # --- Crop or use full ---
    if coords is not None and len(coords) >= 2:
        cropped_data1, cropped_wcs1 = crop_image(data1, wcs1, coords)
        cropped_data2, cropped_wcs2 = crop_image(data2, wcs2, coords)
    else:
        cropped_data1, cropped_wcs1 = data1, wcs1
        cropped_data2, cropped_wcs2 = data2, wcs2

    # --- Alignment ---
    result = align_images(cropped_data1, cropped_wcs1, cropped_data2, cropped_wcs2)
    shift = result.x
    print(f"Optimal Galactic lon/lat shifts (deg): lon={shift[0]:.6f}, lat={shift[1]:.6f}")
    if not result.success:
        print(f"WARNING: minimizer did not fully converge: {result.message}")

    # --- Reference point (centre of cropped region) ---
    y_center = (cropped_data1.shape[0] - 1) / 2.0
    x_center = (cropped_data1.shape[1] - 1) / 2.0
    ref_lon, ref_lat = cropped_wcs1.wcs_pix2world(x_center, y_center, 0)
    print(f"Reference Galactic coords: l={ref_lon:.6f}°, b={ref_lat:.6f}°")

    # --- Observer info ---
    obs_time = args.time
    obs_lat = args.lat
    obs_lon = args.lon
    obs_alt = args.alt

    if not args.non_interactive and not args.rectangle:
        obs_time = input(f"Observer time [YYYY-MM-DD HH:MM:SS] ({obs_time}): ") or obs_time
        try:
            obs_lat = float(input(f"Observer latitude ({obs_lat}): ") or obs_lat)
        except ValueError:
            pass
        try:
            obs_lon = float(input(f"Observer longitude ({obs_lon}): ") or obs_lon)
        except ValueError:
            pass
        try:
            obs_alt = float(input(f"Observer altitude ({obs_alt}): ") or obs_alt)
        except ValueError:
            pass

    print(f"Observer: time={obs_time}, lat={obs_lat:.4f}°, lon={obs_lon:.4f}°, alt={obs_alt:.0f} m")

    # --- AZ/EL conversion ---
    az, el = galactic_offset_to_azel(
        obs_lat, obs_lon, obs_alt, obs_time,
        float(ref_lon), float(ref_lat),
        float(shift[0]), float(shift[1]),
    )
    print(f"Azimuth  offset: {az:.4f}°")
    print(f"Elevation offset: {el:.4f}°")


if __name__ == "__main__":
    main()
