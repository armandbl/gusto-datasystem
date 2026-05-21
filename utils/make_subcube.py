#!/usr/bin/env python3
"""
Create a spatial/spectral subcube around a Galactic coordinate and velocity range.

Example:
  python utils/make_subcube.py \
    --input data/original_cube.fits \
    --output data/subcube.fits \
    --vmin -10 --vmax 25 \
    --l 49.5 --b -0.2 \
    --radius-arcmin 5
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a subcube from a FITS data cube using Galactic coordinates and a velocity range."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input FITS data cube.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the output subcube FITS file.",
    )
    parser.add_argument(
        "--vmin",
        required=True,
        type=float,
        help="Minimum velocity (km/s).",
    )
    parser.add_argument(
        "--vmax",
        required=True,
        type=float,
        help="Maximum velocity (km/s).",
    )
    parser.add_argument(
        "--l",
        required=True,
        type=float,
        help="Galactic longitude (deg).",
    )
    parser.add_argument(
        "--b",
        required=True,
        type=float,
        help="Galactic latitude (deg).",
    )

    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        "--radius-pix",
        type=int,
        help="Half-size of the spatial cutout in pixels.",
    )
    size_group.add_argument(
        "--radius-arcmin",
        type=float,
        help="Half-size of the spatial cutout in arcminutes.",
    )

    parser.add_argument(
        "--ext",
        type=int,
        default=0,
        help="FITS extension index (default: 0).",
    )

    return parser.parse_args()


def _axis_matches_lon(phys_type: str) -> bool:
    phys_type = phys_type.lower()
    return (
        "longitude" in phys_type
        or phys_type.endswith(".lon")
        or phys_type.endswith(":lon")
    )


def _axis_matches_lat(phys_type: str) -> bool:
    phys_type = phys_type.lower()
    return (
        "latitude" in phys_type
        or phys_type.endswith(".lat")
        or phys_type.endswith(":lat")
    )


def _axis_matches_spectral(phys_type: str) -> bool:
    phys_type = phys_type.lower()
    return any(token in phys_type for token in ("spectral", "frequency", "velocity"))


def _world_values_for_wcs(
    wcs: WCS, gal_l_deg: float, gal_b_deg: float, vel: u.Quantity
) -> Tuple[float, ...]:
    """Return world values in the order expected by WCS.world_to_array_index_values()."""

    # WCS expects world values in the order of world_axis_physical_types
    values = []
    sky = SkyCoord(l=gal_l_deg * u.deg, b=gal_b_deg * u.deg, frame="galactic")

    for phys_type, unit_str in zip(wcs.world_axis_physical_types, wcs.world_axis_units):
        if phys_type is None:
            raise ValueError("WCS has an undefined world axis; cannot map coordinates reliably.")

        unit = u.Unit(unit_str) if unit_str else None

        if _axis_matches_lon(phys_type):
            val = sky.l.to(unit).value if unit else sky.l.deg
        elif _axis_matches_lat(phys_type):
            val = sky.b.to(unit).value if unit else sky.b.deg
        elif _axis_matches_spectral(phys_type):
            val = vel.to(unit).value if unit else vel.value
        else:
            raise ValueError(
                f"Unsupported world axis type '{phys_type}'. Expected lon/lat/spectral axes."
            )

        values.append(val)

    return tuple(values)


def _radius_pixels(wcs: WCS, radius_arcmin: float) -> int:
    """Estimate a pixel radius from an arcminute radius using celestial WCS."""
    cel = wcs.celestial
    # proj_plane_pixel_scales gives degrees/pixel for each axis
    scales_deg = proj_plane_pixel_scales(cel) * u.deg
    # Use the mean scale for a square-ish pixel estimate
    mean_scale = np.mean(scales_deg).to(u.arcmin)
    radius_pix = int(np.ceil((radius_arcmin * u.arcmin / mean_scale).value))
    return max(radius_pix, 1)


def _clamp_slice(start: int, end: int, size: int) -> slice:
    start = max(start, 0)
    end = min(end, size)
    return slice(start, end)


def main() -> None:
    args = _parse_args()

    with fits.open(args.input) as hdul:
        hdu = hdul[args.ext]
        data = hdu.data
        header = hdu.header

    if data is None:
        raise ValueError("Input FITS extension has no data.")

    wcs = WCS(header)
    if wcs.naxis < 3:
        raise ValueError("Expected a 3D FITS cube (spectral + 2 spatial axes).")

    vmin = args.vmin * u.km / u.s
    vmax = args.vmax * u.km / u.s

    world_min = _world_values_for_wcs(wcs, args.l, args.b, vmin)
    world_max = _world_values_for_wcs(wcs, args.l, args.b, vmax)

    idx_min = wcs.world_to_array_index_values(*world_min)
    idx_max = wcs.world_to_array_index_values(*world_max)

    # Identify spectral axis as the axis whose index differs between vmin and vmax
    diff_axes = [i for i, (a, b) in enumerate(zip(idx_min, idx_max)) if a != b]
    if len(diff_axes) != 1:
        raise ValueError(
            "Could not uniquely determine spectral axis. Ensure vmin/vmax are valid and distinct."
        )

    spectral_axis = diff_axes[0]
    spatial_axes = [i for i in range(len(idx_min)) if i != spectral_axis]

    if args.radius_pix is not None:
        radius_pix = args.radius_pix
    else:
        radius_pix = _radius_pixels(wcs, args.radius_arcmin)

    # Build slices in numpy (array) order
    slices = [slice(None)] * data.ndim

    # Spectral slice
    zmin = min(idx_min[spectral_axis], idx_max[spectral_axis])
    zmax = max(idx_min[spectral_axis], idx_max[spectral_axis]) + 1
    slices[spectral_axis] = _clamp_slice(zmin, zmax, data.shape[spectral_axis])

    # Spatial slices
    for axis in spatial_axes:
        center = idx_min[axis]
        start = center - radius_pix
        end = center + radius_pix + 1
        slices[axis] = _clamp_slice(start, end, data.shape[axis])

    subcube = data[tuple(slices)]

    # Update header/WCS for the subcube
    try:
        sub_wcs = wcs.slice(tuple(slices))
        sub_header = sub_wcs.to_header()
    except Exception:
        # Fallback: adjust CRPIX for each axis
        sub_header = header.copy()
        for i, slc in enumerate(slices, start=1):
            if isinstance(slc, slice):
                offset = slc.start or 0
                key = f"CRPIX{i}"
                if key in sub_header:
                    sub_header[key] = sub_header[key] - offset

    fits.writeto(args.output, subcube, sub_header, overwrite=True)


if __name__ == "__main__":
    main()
