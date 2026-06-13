#!/usr/bin/env python3
"""Shared visualization and file-discovery helpers for GUSTO analysis scripts.

Functions extracted from ``make_blink_gif.py``, ``make_comparison_pngs.py``,
``make_comparison_movie.py``, and ``extract_cube_slices_png.py`` to eliminate
copy-paste duplication.  Also supplies the canonical ``find_latest_run_dir``
used by ``measure_mixer_crosscorr.py``, ``make_difference_cube.py``, and
``extract_cube_slices_png.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# Common regex patterns
# ---------------------------------------------------------------------------

LINE_PATTERN = re.compile(r"(CII|NII)", re.IGNORECASE)
MIXER_PATTERN = re.compile(r"_(\d+)", re.IGNORECASE)
RUN_PATTERN = re.compile(r"^run\s+(\d+)$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Spectral axis
# ---------------------------------------------------------------------------


def spectral_axis_mps(header: fits.Header, nchan: int) -> np.ndarray:
    """Return spectral axis in meters per second (m/s).

    Assumes ``CRVAL3`` / ``CDELT3`` are in the units specified by ``CUNIT3``.
    GUSTO cubes frequently use values between about −200 and 200, consistent
    with m/s rather than km/s.
    """
    crval = float(header.get("CRVAL3", 0.0))
    cdelt = float(header.get("CDELT3", 1.0))
    crpix = float(header.get("CRPIX3", 1.0))
    cunit = str(header.get("CUNIT3", "")).strip().lower()

    axis = crval + ((np.arange(nchan) + 1.0) - crpix) * cdelt

    if "km/s" in cunit or "kms" in cunit:
        return axis * 1000.0

    return axis


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def line_from_filename(file_path: Path) -> Optional[str]:
    """Extract the spectral line (``'CII'`` or ``'NII'``) from a filename."""
    match = LINE_PATTERN.search(file_path.name)
    if not match:
        return None
    return match.group(1).upper()


def mixer_from_filename(file_path: Path) -> Optional[str]:
    """Extract the mixer number as a string from a filename."""
    match = MIXER_PATTERN.search(file_path.name)
    if not match:
        return None
    return match.group(1)


# ---------------------------------------------------------------------------
# Display intensity limits
# ---------------------------------------------------------------------------


def intensity_limits(line: str) -> Tuple[float, float]:
    """Default display intensity limits in Kelvin-like units for *line*."""
    if line == "CII":
        return (-1.0, 6.0)
    elif line == "NII":
        return (-1.0, 2.0)
    return (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Cube I/O
# ---------------------------------------------------------------------------


def load_cube(path: Path) -> Tuple[np.ndarray, fits.Header]:
    """Load a 3-D FITS cube, squeeze degenerate axes, and return ``(data, header)``."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None:
        raise ValueError(f"No data in {path}")

    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube in {path}, got shape={data.shape}")

    return np.array(data, dtype=float), header


# ---------------------------------------------------------------------------
# Spatial metadata for ``imshow``
# ---------------------------------------------------------------------------


def spatial_plot_metadata(
    header: fits.Header, frame: np.ndarray
) -> Tuple[List[float], str, str]:
    """Return ``(extent, xlabel, ylabel)`` for ``ax.imshow``."""
    ny, nx = frame.shape
    crval1 = float(header.get("CRVAL1", 0.0))
    crpix1 = float(header.get("CRPIX1", 1.0))
    cdelt1 = float(header.get("CDELT1", 1.0))
    ctype1 = str(header.get("CTYPE1", "X")).split("-")[0].strip() or "X"
    cunit1 = str(header.get("CUNIT1", "deg")).strip() or "deg"

    crval2 = float(header.get("CRVAL2", 0.0))
    crpix2 = float(header.get("CRPIX2", 1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))
    ctype2 = str(header.get("CTYPE2", "Y")).split("-")[0].strip() or "Y"
    cunit2 = str(header.get("CUNIT2", "deg")).strip() or "deg"

    x0 = crval1 + ((1.0 - crpix1) * cdelt1)
    x1 = crval1 + ((float(nx) - crpix1) * cdelt1)
    y0 = crval2 + ((1.0 - crpix2) * cdelt2)
    y1 = crval2 + ((float(ny) - crpix2) * cdelt2)

    extent = [x0, x1, y0, y1]
    xlabel = f"{ctype1} [{cunit1}]"
    ylabel = f"{ctype2} [{cunit2}]"
    return extent, xlabel, ylabel


# ---------------------------------------------------------------------------
# Velocity → filename tokens
# ---------------------------------------------------------------------------


def velocity_label(value_mps: float) -> str:
    """Convert a velocity in m/s to a safe filename token (PNG naming).

    Example: *+120.5* → ``"p120.5"``, *−40.0* → ``"m040.0"``.
    """
    return f"{value_mps:+06.1f}".replace("+", "p").replace("-", "m")


def velocity_slug(vel_mps: float) -> str:
    """Convert a velocity in m/s to a safe filename token (GIF naming).

    Example: *+120.5* → ``"vp120p5"``, *−40.0* → ``"vm040p0"``.
    """
    return (
        f"v{vel_mps:+.1f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )


# ---------------------------------------------------------------------------
# Run-directory discovery
# ---------------------------------------------------------------------------


def find_latest_run_dir(source_dir: Path, run_selector: str = "latest") -> Path:
    """Return the ``run N`` directory under *source_dir*.

    Parameters
    ----------
    source_dir : Path
        Directory containing ``run N`` subdirectories.
    run_selector : str
        ``"latest"`` picks the highest-numbered run; anything else is
        treated as a literal run number (e.g. ``"12"`` → ``run 12``).
    """
    if run_selector.lower() != "latest":
        run_dir = source_dir / f"run {run_selector}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    runs: list[tuple[int, Path]] = []
    for child in source_dir.iterdir():
        if child.is_dir():
            m = RUN_PATTERN.match(child.name)
            if m:
                runs.append((int(m.group(1)), child))
    if not runs:
        raise FileNotFoundError(f"No run directories found in {source_dir}")
    return sorted(runs, key=lambda item: item[0])[-1][1]
