#!/usr/bin/env python3
"""Apply fractional mixer-offset corrections to SDFITS files.

Reads per-mixer calibration offsets from a text table (same format as
``offsets.txt``) and applies a fractional correction to the RA/DEC of
every row in a FITS file, updating the file in-place.

This is a legacy utility superseded by the :ref:`alignment pipeline`
(``measure_mixer_crosscorr`` → ``apply_offset_deltas`` → ``runGUSTO``).
"""

from __future__ import annotations

import datetime
import importlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

# Use configargparse if available, plain argparse otherwise.
try:
    _argparse_backend = importlib.import_module("configargparse")
except ModuleNotFoundError:
    import argparse as _argparse_backend  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Calibration table reader
# ---------------------------------------------------------------------------


def get_cal_mixer_offsets(
    calib_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read mixer offsets from *calib_file*.

    Returns
    -------
    azoffs : np.ndarray
        Azimuth offsets in **arcminutes**.
    eloffs : np.ndarray
        Elevation offsets in **arcminutes**.
    bm : np.ndarray
        String array of ``(mixer_label, offset_type)`` pairs,
        shape ``(N, 2)``.
    """
    bm: list[tuple[str, str]] = []
    azoffs: list[float] = []
    eloffs: list[float] = []

    with calib_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("B"):
                # header / non-data
                if not stripped.startswith("B"):
                    continue
            else:
                continue

            cols = stripped.split("\t")
            if len(cols) < 4:
                continue

            bm.append((cols[0], cols[3].strip()))
            azoffs.append(float(cols[1]))
            eloffs.append(float(cols[2]))

    return (
        np.array(azoffs) * 60.0,   # → arcmin
        np.array(eloffs) * 60.0,
        np.array(bm),
    )


# ---------------------------------------------------------------------------
# FITS history helpers
# ---------------------------------------------------------------------------


def history_test(hdr: fits.Header, phrase: str, verbose: bool = False) -> bool:
    """Return True if *phrase* is present in the HISTORY cards of *hdr*."""
    history_list = hdr.get("HISTORY")
    if history_list is None:
        if verbose:
            print(f"HISTORY not in header: applying '{phrase}'")
        return False

    if phrase in history_list:
        if verbose:
            print(f"'{phrase}' present in HISTORY")
        return True
    else:
        if verbose:
            print(f"'{phrase}' not found in HISTORY")
        return False


# ---------------------------------------------------------------------------
# Offset application
# ---------------------------------------------------------------------------


def update_mixer_offset(
    foff: Sequence[float],
    hdr0: fits.Header,
    ra: np.ndarray,
    dec: np.ndarray,
    obstime: Time,
    band: int,
    mix: int,
    calib_file: Path,
    verbose: bool = False,
) -> SkyCoord:
    """Apply a fractional mixer offset to an RA/DEC position.

    Parameters
    ----------
    foff : (faz, falt)
        Fractional scale factors for azimuth and altitude offsets.
    hdr0 : fits.Header
        Primary header containing ``GON_LAT``, ``GON_LON``, ``GON_ALT``.
    ra, dec : np.ndarray
        Right ascension and declination in degrees.
    obstime : Time
        Observation time.
    band : int
        Band number (1 = NII, 2 = CII).
    mix : int
        Mixer number (1-8).
    calib_file : Path
        Path to the calibration offsets file.
    verbose : bool
        Print additional diagnostics.
    """
    balloon = EarthLocation(
        lat=hdr0["GON_LAT"] * u.deg,
        lon=hdr0["GON_LON"] * u.deg,
        height=hdr0["GON_ALT"] * u.m,
    )

    faz = float(foff[0])
    falt = float(foff[1])

    aa = AltAz(location=balloon, obstime=obstime)
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    altaz = coord.transform_to(aa)

    azoffset, aloffset, bm = get_cal_mixer_offsets(calib_file)
    mname = f"B{band}M{mix}"
    indx = np.argwhere(bm[:, 0] == mname).flatten()

    if len(indx) == 0:
        raise ValueError(f"Mixer {mname} not found in {calib_file}")

    # Use the last entry to get AS_MEASURED values if present
    azoffm = float(azoffset[indx[-1]]) / 60.0   # arcmin → deg
    aloffm = float(aloffset[indx[-1]]) / 60.0

    if band == 1:
        naz = (altaz.az.deg - azoffm + faz * azoffm) * u.deg
        nalt = (altaz.alt.deg - aloffm + falt * aloffm) * u.deg
    else:
        # Band 2 is not updated by this legacy utility.
        naz = altaz.az.deg * u.deg
        nalt = altaz.alt.deg * u.deg

    ncc = SkyCoord(AltAz(az=naz, alt=nalt, obstime=obstime, location=balloon))
    return ncc.transform_to("icrs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(args: Optional[list[str]] = None) -> None:
    parser = _argparse_backend.ArgumentParser(
        prog="update_mixer_offset",
        description="Update RA/DEC positions in GUSTO data files "
                    "by applying fractional mixer offsets.",
    )
    parser.add_argument("-v", action="version", version="Version 0.1.0")
    parser.add_argument(
        "-o", "--foff", nargs=2, metavar="FRAC",
        help="Scale factors for calibration offsets (AZ ALT)",
    )
    parser.add_argument(
        "-r", "--fileroot", required=True,
        help="Name (glob prefix) of data file(s) to update.",
    )
    parser.add_argument(
        "-d", "--directory",
        default="/data/scratch/GUSTO/gusto-datasystem/Data/level1/",
        help="Path to data directory (default: %(default)s)",
    )
    parser.add_argument(
        "-c", "--calib-file",
        default="/data/scratch/GUSTO/gusto-datasystem/calib/cal_offsets.txt",
        help="Path to calibration offsets file (default: %(default)s)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Dry-run: only report which scans would be updated.",
    )

    parsed = parser.parse_args(args)

    if parsed.foff is None:
        parser.error("--foff is required (e.g. --foff 1.0 0.5)")

    fileroot = parsed.fileroot
    directory = Path(parsed.directory)
    calib_file = Path(parsed.calib_file)
    foff = np.array(parsed.foff, dtype=float)
    check_only: bool = parsed.check

    pattern = f"{fileroot}*.fits"
    to_process = sorted(directory.glob(pattern))

    history_phrase = "Mixer Offsets Updated for Band 1"
    utc0 = np.datetime64("1970-01-01T00:00:00", "s")

    for infile in to_process:
        with fits.open(infile, mode="update") as hdu:
            hdr0 = hdu[0].header

            if history_test(hdr0, history_phrase, verbose=True):
                print(f"Offsets already applied: {infile.name} — skipping")
                continue

            if check_only:
                print(f"[dry-run] Would update: {infile.name}")
                continue

            data = hdu[1].data
            mixers = np.unique(data["mixer"])
            band = int(hdr0["BAND"])

            otime = float(np.median(data["unixtime"]))
            obstime = utc0 + np.timedelta64(int(otime * 1000), "ms")

            for mx in mixers:
                qmx = data["mixer"] == mx
                nradec = update_mixer_offset(
                    foff=foff,
                    hdr0=hdr0,
                    ra=data["RA"][qmx],
                    dec=data["DEC"][qmx],
                    obstime=obstime,
                    band=band,
                    mix=mx,
                    calib_file=calib_file,
                )
                data["RA"][qmx] = np.array(nradec.ra.deg)
                data["DEC"][qmx] = np.array(nradec.dec.deg)

            print(f"Updating {infile.name}")
            hdr0.add_history(history_phrase)
            hdu.flush()


if __name__ == "__main__":
    main()
