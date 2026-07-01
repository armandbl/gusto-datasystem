#!/usr/bin/env python3
"""Measure integer-pixel shifts on moment-0 maps and write delta-only outputs.

Handles the full Galactic→AZ/EL coordinate conversion using astropy so that
measured offsets are correctly expressed in the focal-plane Azimuth/Elevation
frame expected by the calibration table and the L10 pointing pipeline.
"""

from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz_helpers import find_latest_run_dir  # canonical implementation


@dataclass
class Job:
    source: str
    line: str
    target_mixer: int
    mixer: int
    run: str


@dataclass
class JobResult:
    source: str
    run_dir: str
    line: str
    target_mixer: int
    mixer: int
    reference_cube: str
    target_cube: str
    reference_moment0: str
    target_moment0: str
    shift_x: int
    shift_y: int
    dx_pix: float
    dy_pix: float
    az_deg: float
    el_deg: float
    coord_method: str
    status: str
    correlation_png: str
    # Uncertainty fields (1-σ)
    sigma_x_pix: float = float("nan")
    sigma_y_pix: float = float("nan")
    sigma_az_deg: float = float("nan")
    sigma_el_deg: float = float("nan")


# ---------------------------------------------------------------------------
# Observer metadata helpers
# ---------------------------------------------------------------------------

def galactic_offset_to_azel(
    observer_lat: float,
    observer_lon: float,
    observer_alt: float,
    obs_time_utc: str,
    ref_glon: float,
    ref_glat: float,
    dlon_deg: float,
    dlat_deg: float,
) -> tuple[float, float]:
    """Convert a Galactic (lon, lat) offset to an AZ/EL offset.

    Uses the full astropy coordinate transformation:
    Galactic(l,b) → ICRS → AltAz, then computes the AZ/EL difference.

    This is the same logic as ``align_pixels.calculate_az_el_offsets``,
    inlined here to avoid importing the interactive UI module.
    """
    location = EarthLocation(
        lat=observer_lat * u.deg,
        lon=observer_lon * u.deg,
        height=observer_alt * u.m,
    )
    obstime = Time(obs_time_utc)
    aa_frame = AltAz(location=location, obstime=obstime)

    ref = SkyCoord(l=ref_glon * u.deg, b=ref_glat * u.deg, frame="galactic")
    shifted = SkyCoord(
        l=(ref_glon + dlon_deg) * u.deg,
        b=(ref_glat + dlat_deg) * u.deg,
        frame="galactic",
    )

    ref_aa = ref.transform_to(aa_frame)
    shifted_aa = shifted.transform_to(aa_frame)

    daz = float((shifted_aa.az - ref_aa.az).wrap_at(180 * u.deg).deg)
    del_ = float((shifted_aa.alt - ref_aa.alt).deg)
    return daz, del_


def _auto_detect_from_level1(
    data_root: Path,
    source: str,
    line: str,
    ref_glon: float | None = None,
    ref_glat: float | None = None,
) -> tuple[float, float, float, str] | None:
    """Extract observer metadata from Level-1 FITS files.

    Globs ``Data/level1/{source}/{line}_*_L10.fits``, reads GON_LAT /
    GON_LON / GON_ALT from the primary header and computes the median
    UNIXTIME from the binary table.

    When *ref_glon* and *ref_glat* are provided, selects the OTF leg
    closest to that sky position instead of using the median of all data.
    This gives the observer state at the subcube centre rather than the
    flight-wide average, reducing systematic AZ/EL conversion error from
    ~0.6\" RMS to ~0.1\" (compared to the per-scan-mean ground truth).
    """
    level1_dir = data_root.parent / "level1" / source
    pattern = str(level1_dir / f"{line}_*_L10.fits")
    l1_files = sorted(_glob.glob(pattern))

    if not l1_files:
        print(f"  [auto-detect] No Level-1 files found for {source}/{line} "
              f"at {pattern}")
        return None

    use_position = ref_glon is not None and ref_glat is not None

    if use_position:
        # Find the OTF leg closest to the subcube centre

        ref_coord = SkyCoord(l=ref_glon * u.deg, b=ref_glat * u.deg,
                              frame="galactic")
        best_sep = float("inf")
        best_lat: float | None = None
        best_lon: float | None = None
        best_alt: float | None = None
        best_ut: float | None = None

        for l1_path in l1_files:
            try:
                with fits.open(l1_path) as hdul:
                    hdr = hdul[0].header
                    data = hdul[1].data
                    osel = data["scan_type"] == "OTF"
                    if not np.any(osel):
                        continue
                    coords = SkyCoord(
                        ra=data["RA"][osel] * u.deg,
                        dec=data["DEC"][osel] * u.deg,
                        frame="icrs",
                    )
                    seps = ref_coord.separation(coords)
                    idx = int(np.argmin(seps))
                    sep_deg = float(seps[idx].deg)
                    if sep_deg < best_sep:
                        best_sep = sep_deg
                        best_lat = float(hdr["GON_LAT"])
                        best_lon = float(hdr["GON_LON"])
                        best_alt = float(hdr["GON_ALT"])
                        best_ut = float(data["UNIXTIME"][osel][idx])
            except Exception:
                continue

        if best_ut is None:
            print(f"  [auto-detect] No OTF legs found near "
                  f"({ref_glon:.2f}, {ref_glat:.2f}) for {source}/{line}")
            return None

        obs_time_iso = Time(best_ut, format="unix").iso
        print(f"  [auto-detect] {source}/{line} at "
              f"({ref_glon:.2f}, {ref_glat:.2f}): "
              f"closest leg {best_sep * 3600:.0f}\" away, "
              f"lat={best_lat:.4f} lon={best_lon:.4f} alt={best_alt:.1f}m, "
              f"obs_time={obs_time_iso}")
        return (best_lat, best_lon, best_alt, obs_time_iso)

    # Fallback: median of all data (original behaviour)
    lats: list[float] = []
    lons: list[float] = []
    alts: list[float] = []
    utimes: list[float] = []

    for l1_path in l1_files:
        try:
            with fits.open(l1_path) as hdul:
                hdr = hdul[0].header
                lats.append(float(hdr["GON_LAT"]))
                lons.append(float(hdr["GON_LON"]))
                alts.append(float(hdr["GON_ALT"]))

                data = hdul[1].data
                osel = data["scan_type"] == "OTF"
                if np.any(osel):
                    utimes.append(float(np.median(data["UNIXTIME"][osel])))
        except Exception:
            continue

    if not utimes:
        print(f"  [auto-detect] No valid Level-1 rows for {source}/{line}")
        return None

    med_lat = float(np.median(lats))
    med_lon = float(np.median(lons))
    med_alt = float(np.median(alts))
    med_unix = float(np.median(utimes))
    obs_time_iso = Time(med_unix, format="unix").iso

    print(f"  [auto-detect] {source}/{line}: "
          f"{len(l1_files)} L1 files, "
          f"lat={med_lat:.4f} lon={med_lon:.4f} alt={med_alt:.1f}m, "
          f"obs_time={obs_time_iso}")
    return (med_lat, med_lon, med_alt, obs_time_iso)


def get_observer_metadata(
    config: dict[str, object],
    data_root: Path,
    source: str,
    line: str,
    ref_glon: float | None = None,
    ref_glat: float | None = None,
) -> tuple[float, float, float, str] | None:
    """Resolve observer metadata via config or auto-detection.

    Returns ``(lat_deg, lon_deg, alt_m, obs_time_utc_iso)`` or *None*.

    When *ref_glon* / *ref_glat* are provided (e.g. CRVAL1 / CRVAL2 from
    the cube header), auto-detection picks the OTF leg closest to that
    sky position rather than the flight-wide median.
    """
    # Priority 1: explicit config section
    observer_cfg = config.get("observer")
    if isinstance(observer_cfg, dict):
        lat = observer_cfg.get("lat_deg")
        lon = observer_cfg.get("lon_deg")
        alt = observer_cfg.get("alt_m")
        time = observer_cfg.get("obs_time_utc")
        if all(v is not None for v in (lat, lon, alt, time)):
            return (float(lat), float(lon), float(alt), str(time))

    # Priority 2: auto-detect from Level-1 telemetry
    if config.get("auto_detect_observer", False):
        return _auto_detect_from_level1(data_root, source, line,
                                        ref_glon=ref_glon, ref_glat=ref_glat)

    print(f"  [metadata] No observer config for {source}/{line}; "
          f"set 'auto_detect_observer': true or add 'observer' section")
    return None


# ---------------------------------------------------------------------------
# File / data helpers
# ---------------------------------------------------------------------------

def _compute_azel_to_pixel_jacobian(
    observer: tuple[float, float, float, str],
    ref_header: fits.Header,
) -> np.ndarray | None:
    """Compute the 2×2 Jacobian ∂(x_pix, y_pix)/∂(az, el) at the reference position.

    Uses the L10 correction path (ICRS → AltAz → add offset → ICRS → WCS pixel)
    to numerically compute how pixel coordinates respond to AZ/EL perturbations.

    Returns the Jacobian matrix ``J`` where::

        [dx_pix]   J   [daz]
        [dy_pix] =   · [del_]

    or *None* if the WCS or observer data is invalid.
    """
    try:
        w = WCS(ref_header).celestial
    except Exception:
        return None

    ref_glon = header_float(ref_header, "CRVAL1", 0.0)
    ref_glat = header_float(ref_header, "CRVAL2", 0.0)

    location = EarthLocation(
        lat=observer[0] * u.deg,
        lon=observer[1] * u.deg,
        height=observer[2] * u.m,
    )
    obstime = Time(observer[3])
    aa_frame = AltAz(location=location, obstime=obstime)

    # Reference position in ICRS → AltAz
    ref_gal = SkyCoord(l=ref_glon * u.deg, b=ref_glat * u.deg, frame="galactic")
    ref_icrs = ref_gal.transform_to("icrs")
    ref_altaz = ref_icrs.transform_to(aa_frame)

    # CRPIX for pixel reference
    crpix1 = header_float(ref_header, "CRPIX1", 1.0)
    crpix2 = header_float(ref_header, "CRPIX2", 1.0)

    def _pixel_shift(daz: float, del_: float) -> tuple[float, float]:
        """Apply (daz, del_) via the L10 correction path, return pixel shift."""
        naz = ref_altaz.az + daz * u.deg
        nalt = ref_altaz.alt + del_ * u.deg
        corr_icrs = SkyCoord(
            AltAz(az=naz, alt=nalt, obstime=obstime, location=location)
        ).transform_to("icrs")
        cx, cy = w.world_to_pixel(corr_icrs)
        return float(cx - crpix1), float(cy - crpix2)

    # Numerical differentiation with a small perturbation (≈ 1 arcsec)
    eps = 1.0 / 3600.0  # degrees

    # Column 0: ∂(pix)/∂(az)
    x_az_p, y_az_p = _pixel_shift(+eps, 0.0)
    x_az_m, y_az_m = _pixel_shift(-eps, 0.0)
    j00 = (x_az_p - x_az_m) / (2 * eps)
    j10 = (y_az_p - y_az_m) / (2 * eps)

    # Column 1: ∂(pix)/∂(el)
    x_el_p, y_el_p = _pixel_shift(0.0, +eps)
    x_el_m, y_el_m = _pixel_shift(0.0, -eps)
    j01 = (x_el_p - x_el_m) / (2 * eps)
    j11 = (y_el_p - y_el_m) / (2 * eps)

    J = np.array([[j00, j01], [j10, j11]], dtype=float)
    return J


def pixel_offset_to_azel(
    dx_pix: float,
    dy_pix: float,
    ref_header: fits.Header,
    observer: tuple[float, float, float, str] | None,
) -> tuple[float, float, str]:
    """Convert a pixel offset to an AZ/EL angular offset.

    When observer metadata is available, computes the 2×2 Jacobian
    ∂(pix)/∂(az, el) along the L10 correction path and solves the coupled
    linear system to decouple AZ and EL corrections — preventing the
    coordinate-axis cancellation that can stall convergence in one pixel
    direction.

    Without observer metadata falls back to the direct CDELT-based
    conversion (``dlon_deg = dx_pix × cdelt1``, etc.).

    Returns ``(az_deg, el_deg, method)`` where *method* is
    ``"jacobian"`` (decoupled, preferred), ``"astropy"`` (uncoupled
    Galactic→AltAz conversion), or ``"cdelt_fallback"`` (no observer).
    """
    if observer is not None:
        J = _compute_azel_to_pixel_jacobian(observer, ref_header)
        if J is not None and _jacobian_is_reliable(J):
            # Solve: J · [daz, del_] = [-dx_pix, -dy_pix]
            # (we want to CANCEL the measured offset)
            target = np.array([-dx_pix, -dy_pix], dtype=float)
            try:
                solution = np.linalg.solve(J, target)
                return float(solution[0]), float(solution[1]), "jacobian"
            except np.linalg.LinAlgError:
                pass  # fall through to uncoupled method

        # Jacobian failed — fall back to uncoupled Galactic→AltAz
        cdelt1 = header_float(ref_header, "CDELT1", 0.0)
        cdelt2 = header_float(ref_header, "CDELT2", 0.0)
        dlon_deg = dx_pix * cdelt1
        dlat_deg = dy_pix * cdelt2
        ref_glon = header_float(ref_header, "CRVAL1", 0.0)
        ref_glat = header_float(ref_header, "CRVAL2", 0.0)
        az_deg, el_deg = galactic_offset_to_azel(
            observer[0], observer[1], observer[2], observer[3],
            ref_glon, ref_glat, dlon_deg, dlat_deg,
        )
        return az_deg, el_deg, "astropy"

    cdelt1 = header_float(ref_header, "CDELT1", 0.0)
    cdelt2 = header_float(ref_header, "CDELT2", 0.0)
    dlon_deg = dx_pix * cdelt1
    dlat_deg = dy_pix * cdelt2
    return dlon_deg, dlat_deg, "cdelt_fallback"


def propagate_pixel_uncertainty_to_azel(
    sigma_x_pix: float,
    sigma_y_pix: float,
    cov_xy_pix: float,
    ref_header: fits.Header,
    observer: tuple[float, float, float, str] | None,
    coord_method: str,
    dx_pix: float = 0.0,
    dy_pix: float = 0.0,
) -> tuple[float, float]:
    """Propagate 1-σ pixel uncertainties to AZ/EL via the appropriate Jacobian.

    Uses the same coordinate path as ``pixel_offset_to_azel``:
    - ``"jacobian"`` — full J = ∂(pix)/∂(az,el)  →  Cov_azel = J⁻¹ Cov_pix (J⁻¹)ᵀ
    - ``"astropy"`` — numerical Jacobian of the Galactic→AltAz transform
    - ``"cdelt_fallback"`` — simple CDELT scaling

    Returns ``(sigma_az_deg, sigma_el_deg)``.
    """
    if not np.isfinite(sigma_x_pix) or not np.isfinite(sigma_y_pix):
        return float("nan"), float("nan")

    if coord_method == "jacobian" and observer is not None:
        J = _compute_azel_to_pixel_jacobian(observer, ref_header)
        if J is not None and _jacobian_is_reliable(J):
            return _propagate_covariance_through_jacobian(
                J, sigma_x_pix, sigma_y_pix, cov_xy_pix,
            )

    if coord_method in ("jacobian", "astropy") and observer is not None:
        # Numerical Jacobian of Galactic→AltAz at the measured offset
        cdelt1 = header_float(ref_header, "CDELT1", 0.0)
        cdelt2 = header_float(ref_header, "CDELT2", 0.0)
        ref_glon = header_float(ref_header, "CRVAL1", 0.0)
        ref_glat = header_float(ref_header, "CRVAL2", 0.0)

        dlon_deg = dx_pix * cdelt1
        dlat_deg = dy_pix * cdelt2

        eps = 1.0 / 3600.0  # 1 arcsec in degrees

        def _azel_at(dl: float, db: float) -> tuple[float, float]:
            return galactic_offset_to_azel(
                observer[0], observer[1], observer[2], observer[3],
                ref_glon, ref_glat, dl, db,
            )

        az0, el0 = _azel_at(dlon_deg, dlat_deg)
        az_dl, el_dl = _azel_at(dlon_deg + eps, dlat_deg)
        az_db, el_db = _azel_at(dlon_deg, dlat_deg + eps)

        # Jacobian ∂(az,el)/∂(dlon,dlat)
        J_gal = np.array([
            [(az_dl - az0) / eps, (az_db - az0) / eps],
            [(el_dl - el0) / eps, (el_db - el0) / eps],
        ], dtype=float)

        # Pixel covariance → (dlon, dlat) covariance (CDELT scaling)
        cov_dl_dlat = np.array([
            [sigma_x_pix ** 2 * cdelt1 ** 2, cov_xy_pix * cdelt1 * cdelt2],
            [cov_xy_pix * cdelt1 * cdelt2, sigma_y_pix ** 2 * cdelt2 ** 2],
        ], dtype=float)

        try:
            cov_azel = J_gal @ cov_dl_dlat @ J_gal.T
            sigma_az = float(np.sqrt(max(cov_azel[0, 0], 0)))
            sigma_el = float(np.sqrt(max(cov_azel[1, 1], 0)))
            return sigma_az, sigma_el
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")

    # cdelt_fallback — simple scaling
    cdelt1 = header_float(ref_header, "CDELT1", 0.0)
    cdelt2 = header_float(ref_header, "CDELT2", 0.0)
    sigma_az = sigma_x_pix * abs(cdelt1)
    sigma_el = sigma_y_pix * abs(cdelt2)
    return sigma_az, sigma_el


def parse_line_and_mixer_from_name(file_path: Path) -> tuple[str | None, int | None]:
    tokens = file_path.stem.split("_")
    for i, tok in enumerate(tokens):
        up = tok.upper()
        if up in {"CII", "NII"} and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.isdigit():
                return up, int(nxt)

    name = file_path.name.upper()
    line = None
    if "_CII_" in name:
        line = "CII"
    elif "_NII_" in name:
        line = "NII"

    m = re.search(r"_(\d+)(?:_|\.FITS$)", name)
    mixer = int(m.group(1)) if m else None
    return line, mixer


def select_cube(run_dir: Path, line: str, mixer: int) -> Path:
    line = line.upper()
    candidates: list[Path] = []
    for file_path in sorted(run_dir.glob("*.fits")):
        p_line, p_mixer = parse_line_and_mixer_from_name(file_path)
        if p_line == line and p_mixer == mixer:
            candidates.append(file_path)

    if not candidates:
        raise FileNotFoundError(f"No cube found in {run_dir} for line={line} mixer={mixer}")
    if len(candidates) == 1:
        return candidates[0]

    if mixer == 8:
        refs = [p for p in candidates if "reference" in p.name.lower()]
        if refs:
            return sorted(refs)[-1]
    else:
        matched = [p for p in candidates if "matched" in p.name.lower()]
        if matched:
            return sorted(matched)[-1]

    return sorted(candidates, key=lambda p: len(p.name))[0]


def load_cube(path: Path, ext: int = 0) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        primary = hdul[ext]
        data = np.squeeze(cast(Any, primary).data)
        header = cast(fits.Header, cast(Any, primary).header)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube in {path}, got shape={data.shape}")
    return np.array(data, dtype=float), header


def build_moment0_map(cube: np.ndarray) -> np.ndarray:
    return np.nansum(cube, axis=0)


def moment0_header(header: fits.Header) -> fits.Header:
    try:
        return WCS(header).celestial.to_header()
    except Exception:
        return header.copy()


def save_moment0_products(
    cube: np.ndarray,
    header: fits.Header,
    fits_path: Path,
    png_path: Path,
    title: str,
    dx_pix: float | None = None,
    dy_pix: float | None = None,
    peak_x: float | None = None,
    peak_y: float | None = None,
) -> np.ndarray:
    """Save moment-0 map as FITS + annotated PNG.

    When ``dx_pix`` / ``dy_pix`` are provided, the PNG includes:
    - A dashed crosshair at the image centre (reference point)
    - A red X marking the measured correlation-peak position
    """
    moment0 = build_moment0_map(cube)
    out_header = moment0_header(header)
    out_header["HISTORY"] = f"Moment 0 image generated by measure_mixer_crosscorr: {title}"
    fits.writeto(fits_path, moment0, out_header, overwrite=True)

    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(moment0, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    plt.colorbar(im, ax=ax, label="Moment 0")

    ny, nx = moment0.shape
    cx, cy = nx / 2.0, ny / 2.0

    # --- reference crosshair at image centre ---
    ax.axvline(cx, color="white", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(cy, color="white", linestyle="--", linewidth=1.0, alpha=0.7)

    # --- measured-offset overlay ---
    if dx_pix is not None and dy_pix is not None:
        if peak_x is None:
            peak_x = cx + dx_pix
        if peak_y is None:
            peak_y = cy + dy_pix

        ax.plot(peak_x, peak_y, "rx", markersize=16, markeredgewidth=2.5)

    fig.tight_layout()
    fig.savefig(str(png_path))
    plt.close(fig)
    return moment0


def prep_map(image: np.ndarray) -> np.ndarray:
    out = np.array(image, dtype=float)
    median = np.nanmedian(out)
    if np.isfinite(median):
        out = out - median
    out[~np.isfinite(out)] = 0.0
    return out


def measure_shift_integer(
    reference_map: np.ndarray, target_map: np.ndarray,
) -> tuple[int, int, float, np.ndarray]:
    """Cross-correlate two moment-0 maps via FFT convolution.

    Returns ``(lag_x, lag_y, peak_value, corr)`` where *lag_?* are the
    integer pixel lags of the correlation peak relative to centre and
    *peak_value* is the maximum correlation coefficient.
    """
    ref = prep_map(reference_map)
    tgt = prep_map(target_map)
    corr = fftconvolve(ref, tgt[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    center_y, center_x = (s // 2 for s in corr.shape)
    lag_y = int(peak_y - center_y)
    lag_x = int(peak_x - center_x)
    return lag_x, lag_y, float(corr[peak_y, peak_x]), corr


def _fallback_uncertainty(
    corr: np.ndarray,
    peak_y: int,
    peak_x: int,
) -> tuple[float, float, float]:
    """Estimate peak-position uncertainty from the correlation width and SNR.

    Uses the matched-filter relation:  σ_peak ≈ σ_corr / SNR

    where σ_corr is the correlation-peak width (from FWHM / 2.355) and
    SNR = (peak_value − background) / noise_rms.

    This is a model-free fallback — it needs no Gaussian fit and works on
    any correlation surface with a well-defined peak.
    """
    ny, nx = corr.shape
    peak_val = float(corr[peak_y, peak_x])

    # --- noise floor: RMS of outer region (> ¼ of the image from peak) ---
    dy, dx = np.mgrid[0:ny, 0:nx]
    dist = np.sqrt((dx - peak_x) ** 2 + (dy - peak_y) ** 2)
    outer_mask = dist > max(nx, ny) / 4.0
    if np.sum(outer_mask) < 20:
        noise_rms = float(np.nanstd(corr))
    else:
        noise_rms = float(np.nanstd(corr[outer_mask]))

    if noise_rms < 1e-15:
        noise_rms = 1e-15

    bg = float(np.median(corr[outer_mask])) if np.sum(outer_mask) >= 20 else 0.0
    snr = (peak_val - bg) / noise_rms

    # --- FWHM along x and y through the peak ---
    half_max = bg + (peak_val - bg) / 2.0

    # x-direction
    x_profile = corr[peak_y, :] - bg
    above = np.where(x_profile >= half_max - bg)[0]
    if len(above) >= 2:
        fwhm_x = float(above[-1] - above[0] + 1)
    else:
        # No clear half-max crossing — use half the image as a crude bound
        fwhm_x = float(nx) / 2.0

    # y-direction
    y_profile = corr[:, peak_x] - bg
    above = np.where(y_profile >= half_max - bg)[0]
    if len(above) >= 2:
        fwhm_y = float(above[-1] - above[0] + 1)
    else:
        fwhm_y = float(ny) / 2.0

    sigma_corr_x = fwhm_x / 2.355
    sigma_corr_y = fwhm_y / 2.355

    # Matched-filter relation: σ_peak = σ_corr / SNR
    sigma_x = sigma_corr_x / max(snr, 1.0)
    sigma_y = sigma_corr_y / max(snr, 1.0)

    return sigma_x, sigma_y, 0.0


def measure_shift_with_uncertainty(
    corr: np.ndarray,
    peak_y: int,
    peak_x: int,
    half_window: int = 5,
) -> tuple[float, float, float, float, float]:
    """Sub-pixel shift and formal uncertainty from a 2-D Gaussian fit to the correlation peak.

    Fits ``A * exp(-((x-x0)²/(2σx²) + (y-y0)²/(2σy²))) + B`` directly to
    the correlation values *C(x,y)* (not log-transformed) in a ±*half_window*
    pixel window.  Uses ``scipy.optimize.curve_fit`` whose output parameter
    covariance is a proper estimator covariance under the assumption of
    i.i.d. Gaussian noise on the correlation values.

    Returns ``(x0, y0, sigma_x, sigma_y, cov_xy)`` where:

    * **x0, y0** — sub-pixel peak position in correlation-surface array
      coordinates (NOT lags).  Convert to lags via
      ``lag_x = x0 - corr.shape[1] // 2``.
    * **sigma_x** — 1-σ uncertainty on *x0* from the fit covariance [pixels]
    * **sigma_y** — 1-σ uncertainty on *y0* from the fit covariance [pixels]
    * **cov_xy** — covariance between *x0* and *y0* [pixels²]

    On fit failure falls back to ``_fallback_uncertainty()`` which estimates
    σ from the correlation peak width and signal-to-noise ratio
    (σ ≈ FWHM / (2.355 × SNR)) — no free parameters, driven by the data.
    """
    ny, nx = corr.shape
    y0 = max(0, peak_y - half_window)
    y1 = min(ny, peak_y + half_window + 1)
    x0 = max(0, peak_x - half_window)
    x1 = min(nx, peak_x + half_window + 1)

    region = corr[y0:y1, x0:x1]
    y_idx = np.arange(y0, y1, dtype=float)
    x_idx = np.arange(x0, x1, dtype=float)
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = region.ravel()

    # Initial guesses
    amp0 = float(corr[peak_y, peak_x] - np.median(region))
    bg0 = float(np.median(region))
    # Initial guesses — estimate sigma from FWHM of the peak profile
    # (much better than a hardcoded 2.0 px for extended sources)
    _xprof = region[peak_y - y0, :]
    _yprof = region[:, peak_x - x0]
    _half = (amp0 + bg0) / 2.0 if np.isfinite(amp0 + bg0) else amp0 / 2.0 + bg0
    _above_x = np.where(_xprof >= _half)[0]
    _above_y = np.where(_yprof >= _half)[0]
    if len(_above_x) >= 2:
        sx0 = max(float(_above_x[-1] - _above_x[0] + 1) / 2.355, 0.5)
    else:
        sx0 = 2.0
    if len(_above_y) >= 2:
        sy0 = max(float(_above_y[-1] - _above_y[0] + 1) / 2.355, 0.5)
    else:
        sy0 = 2.0

    # ponytail: sigma bound is 2× half_window — extended sources (G337)
    #           can have correlation peaks wider than 5 px.
    _sigma_max = float(half_window * 2)

    def _gaussian_2d(xy, amplitude, xc, yc, sigma_x, sigma_y, bg):
        x, y = xy
        return amplitude * np.exp(
            -((x - xc) ** 2 / (2 * sigma_x ** 2) + (y - yc) ** 2 / (2 * sigma_y ** 2))
        ) + bg

    p0 = [amp0, float(peak_x), float(peak_y), sx0, sy0, bg0]
    bounds = (
        [0.0, x_idx[0], y_idx[0], 0.5, 0.5, -np.inf],
        [np.inf, x_idx[-1], y_idx[-1], _sigma_max, _sigma_max, np.inf],
    )

    try:
        popt, pcov = curve_fit(
            _gaussian_2d,
            (x_flat, y_flat),
            z_flat,
            p0=p0,
            bounds=bounds,
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        # Fit failed — return conservative beam-scale fallback
        return float(peak_x), float(peak_y), *_fallback_uncertainty(corr, peak_y, peak_x)

    x0_fit = float(popt[1])
    y0_fit = float(popt[2])

    # Check that the parameter covariance is usable
    if np.linalg.cond(pcov) > 1e6 or not np.all(np.isfinite(pcov)):
        return x0_fit, y0_fit, *_fallback_uncertainty(corr, peak_y, peak_x)

    sigma_x = float(np.sqrt(max(pcov[1, 1], 0)))
    sigma_y = float(np.sqrt(max(pcov[2, 2], 0)))
    cov_xy = float(pcov[1, 2])

    return x0_fit, y0_fit, sigma_x, sigma_y, cov_xy


def _jacobian_is_reliable(J: np.ndarray, cond_threshold: float = 10.0) -> bool:
    """Check whether a 2×2 Jacobian is well-conditioned for inversion.

    Returns ``True`` if both ``|det(J)| > 1e-12`` and
    ``cond(J) < cond_threshold``.
    """
    return bool(abs(np.linalg.det(J)) > 1e-12 and np.linalg.cond(J) < cond_threshold)


def _propagate_covariance_through_jacobian(
    J: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    cov_xy: float,
) -> tuple[float, float]:
    """Propagate pixel covariance through the Jacobian to AZ/EL space.

    ``Cov_azel = J⁻¹ · Cov_pix · (J⁻¹)ᵀ``

    Returns ``(sigma_az_deg, sigma_el_deg)``.
    """
    if not _jacobian_is_reliable(J):
        return float("nan"), float("nan")
    cov_pix = np.array([[sigma_x ** 2, cov_xy], [cov_xy, sigma_y ** 2]], dtype=float)
    try:
        J_inv = np.linalg.inv(J)
        cov_azel = J_inv @ cov_pix @ J_inv.T
        sigma_az = float(np.sqrt(max(cov_azel[0, 0], 0)))
        sigma_el = float(np.sqrt(max(cov_azel[1, 1], 0)))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return sigma_az, sigma_el


def header_float(header: fits.Header, key: str, default: float = 0.0) -> float:
    value = header.get(key, default)
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float(default)


def _cube_center_galactic(
    header: fits.Header,
    shape: tuple[int, ...],
) -> tuple[float, float]:
    """Return the Galactic (l, b) at the spatial centre pixel of a cube.

    Uses the celestial WCS to convert the centre pixel to Galactic
    coordinates.  This is the true subcube centre, unlike CRVAL which
    reflects the original (pre-sliced) reference position.
    """
    # shape is (nchan, ny, nx) in numpy; celestial axes are (nx, ny)
    cy, cx = shape[1] // 2, shape[2] // 2
    w = WCS(header).celestial
    lon, lat = w.wcs_pix2world(cx, cy, 0)
    return float(lon), float(lat)


def fmt_uncertainty(val: float, sigfigs: int = 2) -> str:
    """Format an uncertainty value for display.

    Uses scientific notation for small values (``abs(val) < 1e-3``)
    where decimal places become unreadable.
    """
    if not np.isfinite(val):
        return "nan"
    if abs(val) < 1e-3 and val != 0.0:
        return f"{val:.{sigfigs - 1}e}"
    return f"{val:.6f}"


def save_correlation_png(
    corr: np.ndarray,
    out_path: Path,
    title: str,
    peak_x: int | None = None,
    peak_y: int | None = None,
    dx_pix: float | None = None,
    dy_pix: float | None = None,
    sigma_x_pix: float | None = None,
    sigma_y_pix: float | None = None,
    sigma_az_deg: float | None = None,
    sigma_el_deg: float | None = None,
) -> None:
    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("X lag index")
    ax.set_ylabel("Y lag index")
    plt.colorbar(im, ax=ax, label="Correlation")

    # Auto-detect peak if not explicitly provided
    if peak_x is None or peak_y is None:
        peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
        peak_x = int(peak_x)
        peak_y = int(peak_y)

    ax.plot(peak_x, peak_y, "rx", markersize=14, markeredgewidth=2.5)

    # Uncertainty ellipse on peak if available
    if (sigma_x_pix is not None and sigma_y_pix is not None
            and np.isfinite(sigma_x_pix) and np.isfinite(sigma_y_pix)):
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(
            (peak_x, peak_y),
            width=2 * sigma_x_pix,
            height=2 * sigma_y_pix,
            angle=0,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )
        ax.add_patch(ellipse)

    cy_c, cx_c = (s // 2 for s in corr.shape)
    ax.set_xlim(cx_c - 75, cx_c + 75)
    ax.set_ylim(cy_c - 75, cy_c + 75)

    # Info box with uncertainty
    info_lines = []
    if dx_pix is not None and dy_pix is not None:
        info_lines.append(f"Pixel: ({dx_pix:+.1f}, {dy_pix:+.1f}) pix")
    if sigma_x_pix is not None and sigma_y_pix is not None:
        info_lines.append(
            f"σ: ({fmt_uncertainty(sigma_x_pix)}, {fmt_uncertainty(sigma_y_pix)}) pix"
        )
    if sigma_az_deg is not None and sigma_el_deg is not None:
        info_lines.append(
            f"σ AZ/EL: ({fmt_uncertainty(sigma_az_deg)}, {fmt_uncertainty(sigma_el_deg)})°"
        )
    if info_lines:
        ax.text(
            0.02, 0.98, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=7, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            zorder=13,
        )

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def pix_label_from_line_mixer(line: str, mixer: int) -> str:
    band = "B2" if line.upper() == "CII" else "B1"
    return f"{band}M{mixer}"


def list_sources(data_root: Path) -> list[str]:
    return sorted(child.name for child in data_root.iterdir() if child.is_dir())


def compare_dir_for_run(run_dir: Path) -> Path:
    compare_dir = run_dir / "Compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "moment0").mkdir(parents=True, exist_ok=True)
    return compare_dir


def parse_line_target_config(config: dict[str, object], line: str) -> tuple[int, list[int]]:
    defaults = {"CII": (8, [5]), "NII": (3, [2, 6])}
    default_target, default_mixers = defaults[line]

    line_targets = config.get("line_targets")
    if not isinstance(line_targets, dict):
        return default_target, default_mixers

    entry = line_targets.get(line, {})
    if not isinstance(entry, dict):
        return default_target, default_mixers

    target_mixer = int(entry.get("target_mixer", default_target))
    if "mixers" in entry:
        mixers = [int(m) for m in entry["mixers"]]
    elif "mixer" in entry:
        mixers = [int(entry["mixer"])]
    else:
        mixers = list(default_mixers)
    return target_mixer, mixers


def parse_jobs(config: dict[str, object], default_run: str, data_root: Path) -> list[Job]:
    raw_jobs = config.get("jobs")
    if isinstance(raw_jobs, list) and raw_jobs:
        jobs: list[Job] = []
        for entry in raw_jobs:
            if not isinstance(entry, dict):
                raise ValueError("Each entry in 'jobs' must be an object")

            source = str(entry["source"]).strip()
            line = str(entry.get("line", "CII")).strip().upper()
            if line not in {"CII", "NII"}:
                raise ValueError(f"Unsupported line '{line}' in job: {entry}")

            run = str(entry.get("run", default_run)).strip()
            target_mixer = int(entry.get("target_mixer", 8 if line == "CII" else 2))

            if "mixers" in entry:
                mixers = [int(m) for m in entry["mixers"]]
            elif "mixer" in entry:
                mixers = [int(entry["mixer"])]
            else:
                raise ValueError(f"Job must provide 'mixer' or 'mixers': {entry}")

            for mixer in mixers:
                if mixer == 0 or mixer == target_mixer:
                    continue
                jobs.append(Job(source=source, line=line, target_mixer=target_mixer, mixer=mixer, run=run))

        if not jobs:
            raise ValueError("No valid non-zero mixers left to process after filtering")
        return jobs

    jobs: list[Job] = []
    selected_sources = config.get("sources")
    if isinstance(selected_sources, list) and selected_sources:
        sources = [str(source).strip() for source in selected_sources]
    elif isinstance(config.get("source"), str) and str(config.get("source")).strip():
        sources = [str(config.get("source")).strip()]
    else:
        sources = list_sources(data_root)

    selected_lines = config.get("lines")
    if isinstance(selected_lines, list) and selected_lines:
        lines = [str(line).strip().upper() for line in selected_lines if str(line).strip()]
    else:
        lines = ["CII", "NII"]

    for source in sources:
        for line in lines:
            target_mixer, mixers = parse_line_target_config(config, line)
            for mixer in mixers:
                if mixer == 0 or mixer == target_mixer:
                    continue
                jobs.append(Job(source=source, line=line, target_mixer=target_mixer, mixer=mixer, run=default_run))

    if not jobs:
        raise ValueError("No valid jobs generated from config and available sources")
    return jobs


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_job(data_root: Path, job: Job, config: dict[str, object]) -> JobResult:
    source_dir = data_root / job.source
    run_dir = find_latest_run_dir(source_dir, job.run)

    reference_cube_path = select_cube(run_dir, job.line, job.target_mixer)
    target_cube_path = select_cube(run_dir, job.line, job.mixer)

    ref_cube, ref_header = load_cube(reference_cube_path)
    tgt_cube, _ = load_cube(target_cube_path)

    if ref_cube.shape[1:] != tgt_cube.shape[1:]:
        raise ValueError(f"Spatial shape mismatch: ref={ref_cube.shape[1:]} target={tgt_cube.shape[1:]}")

    nchan = min(ref_cube.shape[0], tgt_cube.shape[0])
    ref_cube = ref_cube[:nchan]
    tgt_cube = tgt_cube[:nchan]

    compare_dir = compare_dir_for_run(run_dir)
    moment0_dir = compare_dir / "moment0"

    reference_moment0_path = moment0_dir / f"moment0_{job.source}_{job.line}_M{job.target_mixer}_reference.fits"
    target_moment0_path = moment0_dir / f"moment0_{job.source}_{job.line}_M{job.mixer}_target.fits"
    reference_moment0_png = moment0_dir / f"moment0_{job.source}_{job.line}_M{job.target_mixer}_reference.png"
    target_moment0_png = moment0_dir / f"moment0_{job.source}_{job.line}_M{job.mixer}_target.png"

    # Build moment-0 maps first so we can measure the shift, then save
    # the PNGs *with* the measured offset overlaid.
    map_ref = build_moment0_map(ref_cube)
    map_tgt = build_moment0_map(tgt_cube)

    lag_x_int, lag_y_int, peak_val, corr = measure_shift_integer(map_ref, map_tgt)

    # Sub-pixel Gaussian fit for both offset AND formal uncertainty
    peak_y_c = corr.shape[0] // 2 + lag_y_int
    peak_x_c = corr.shape[1] // 2 + lag_x_int
    x0_sub, y0_sub, sigma_x_pix, sigma_y_pix, cov_xy_pix = measure_shift_with_uncertainty(
        corr, peak_y_c, peak_x_c,
    )

    center_y, center_x = corr.shape[0] // 2, corr.shape[1] // 2
    lag_x = x0_sub - center_x
    lag_y = y0_sub - center_y
    dx_pix = -float(lag_x)
    dy_pix = -float(lag_y)

    # Save reference moment-0 (FITS + PNG with centre crosshair only)
    _ref_map = save_moment0_products(
        ref_cube,
        ref_header,
        reference_moment0_path,
        reference_moment0_png,
        title=f"{job.source} {job.line} M{job.target_mixer} moment0 (reference)",
    )

    # Save target moment-0 (FITS + PNG with centre crosshair AND offset marker)
    _tgt_map = save_moment0_products(
        tgt_cube,
        ref_header,
        target_moment0_path,
        target_moment0_png,
        title=f"{job.source} {job.line} M{job.mixer} moment0",
        dx_pix=dx_pix,
        dy_pix=dy_pix,
    )

    # ------------------------------------------------------------------
    # Convert the measured *Galactic* pixel offset to true AZ / EL
    # ------------------------------------------------------------------
    # Use the subcube spatial centre (WCS pixel centre → world), not CRVAL
    # which points to the full-cube reference position.
    ref_glon, ref_glat = _cube_center_galactic(ref_header, ref_cube.shape)
    obs = get_observer_metadata(config, data_root, job.source, job.line,
                                ref_glon=ref_glon, ref_glat=ref_glat)
    az_deg, el_deg, coord_method = pixel_offset_to_azel(
        dx_pix, dy_pix, ref_header, obs,
    )

    # Propagate pixel uncertainty to AZ/EL
    sigma_az_deg, sigma_el_deg = propagate_pixel_uncertainty_to_azel(
        sigma_x_pix, sigma_y_pix, cov_xy_pix,
        ref_header, obs, coord_method,
        dx_pix=dx_pix, dy_pix=dy_pix,
    )

    # --- Systematic uncertainty floor ---
    # Optional: add a configurable floor (arcsec) for unmodeled systematics.
    # Default 0.0 — only set when an empirical upper bound is measured
    # (e.g. via split-half reproducibility test across independent scan sets).
    _sys_floor_arcsec = float(config.get("systematic_floor_arcsec", 0.0))
    _sys_floor_deg = _sys_floor_arcsec / 3600.0
    if _sys_floor_deg > 0:
        if np.isfinite(sigma_az_deg):
            sigma_az_deg = float(np.sqrt(sigma_az_deg ** 2 + _sys_floor_deg ** 2))
        if np.isfinite(sigma_el_deg):
            sigma_el_deg = float(np.sqrt(sigma_el_deg ** 2 + _sys_floor_deg ** 2))

    if coord_method == "cdelt_fallback":
        print(
            f"  WARNING: No observer metadata for {job.source}/{job.line}. "
            f"Using direct Galactic→AZ/EL conversion (may be rotated). "
            f"Set 'auto_detect_observer': true or add an 'observer' section "
            f"to the config for physically correct AZ/EL offsets."
        )

    # Compute Galactic degree offsets from pixel offsets
    cdelt1 = header_float(ref_header, "CDELT1", 0.0)
    cdelt2 = header_float(ref_header, "CDELT2", 0.0)
    dlon_deg = dx_pix * cdelt1
    dlat_deg = dy_pix * cdelt2

    corr_png = compare_dir / f"crosscorr_{job.source}_{job.line}_M{job.target_mixer}_vs_M{job.mixer}.png"
    save_correlation_png(
        corr,
        corr_png,
        title=(
            f"{job.source} {job.line} M{job.target_mixer} vs M{job.mixer}"
        ),
        dx_pix=dx_pix,
        dy_pix=dy_pix,
        sigma_x_pix=sigma_x_pix,
        sigma_y_pix=sigma_y_pix,
        sigma_az_deg=sigma_az_deg,
        sigma_el_deg=sigma_el_deg,
    )

    return JobResult(
        source=job.source,
        run_dir=str(run_dir),
        line=job.line,
        target_mixer=job.target_mixer,
        mixer=job.mixer,
        reference_cube=reference_cube_path.name,
        target_cube=target_cube_path.name,
        reference_moment0=str(reference_moment0_path),
        target_moment0=str(target_moment0_path),
        shift_x=lag_x_int,
        shift_y=lag_y_int,
        dx_pix=dx_pix,
        dy_pix=dy_pix,
        az_deg=az_deg,
        el_deg=el_deg,
        coord_method=coord_method,
        status="OK",
        correlation_png=str(corr_png),
        sigma_x_pix=sigma_x_pix,
        sigma_y_pix=sigma_y_pix,
        sigma_az_deg=sigma_az_deg,
        sigma_el_deg=sigma_el_deg,
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

# Offset-type priority matching getMixerOffsets() in L10_pointing.py
_EFFECTIVE_PRIORITY = ("AS_MEASURED", "FIDUCIAL", "THEORY")


def _read_effective_offsets(offsets_path: Path) -> dict[str, tuple[float, float]]:
    """Read the currently-effective (az, el) for each mixer from an offsets file.

    Uses the same priority logic as ``getMixerOffsets()``:
    AS_MEASURED > FIDUCIAL > THEORY, last-in-type wins.
    """
    raw: dict[str, list[tuple[float, float, str]]] = {}
    for line in offsets_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or stripped.upper().startswith("PIX"):
            continue
        cols = stripped.split()
        if len(cols) < 4:
            continue
        try:
            az = float(cols[1])
            el = float(cols[2])
        except ValueError:
            continue
        pix_label = cols[0]
        etype = cols[3].upper()
        raw.setdefault(pix_label, []).append((az, el, etype))

    effective: dict[str, tuple[float, float]] = {}
    for pix_label, entries in raw.items():
        for ptype in _EFFECTIVE_PRIORITY:
            matches = [(az, el) for az, el, et in entries if et == ptype]
            if matches:
                effective[pix_label] = matches[-1]
                break
    return effective


def write_csv(rows: list[dict[str, object]], out_path: Path, fieldnames: list[str]) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_delta_csv(rows: list[JobResult], out_path: Path,
                    offsets_file: Path | None = None) -> None:
    grouped: dict[str, list[JobResult]] = {}
    for row in rows:
        if row.status != "OK":
            continue
        pix_label = pix_label_from_line_mixer(row.line, row.mixer)
        grouped.setdefault(pix_label, []).append(row)

    # Read current effective offsets if available (for relationship columns)
    effective_offsets: dict[str, tuple[float, float]] = {}
    if offsets_file is not None and offsets_file.exists():
        effective_offsets = _read_effective_offsets(offsets_file)

    delta_rows: list[dict[str, object]] = []
    for pix_label, items in sorted(grouped.items()):
        n = len(items)
        mean_az_deg = float(np.mean([item.az_deg for item in items]))
        mean_el_deg = float(np.mean([item.el_deg for item in items]))
        mean_dx = float(np.mean([item.dx_pix for item in items]))
        mean_dy = float(np.mean([item.dy_pix for item in items]))

        # Aggregate uncertainties: σ_mean = √(Σ σ_i²) / N  (independent measurements)
        az_uncs = [item.sigma_az_deg for item in items
                   if np.isfinite(item.sigma_az_deg)]
        el_uncs = [item.sigma_el_deg for item in items
                   if np.isfinite(item.sigma_el_deg)]
        sigma_az = (float(np.sqrt(sum(u ** 2 for u in az_uncs)) / max(len(az_uncs), 1))
                    if az_uncs else float("nan"))
        sigma_el = (float(np.sqrt(sum(u ** 2 for u in el_uncs)) / max(len(el_uncs), 1))
                    if el_uncs else float("nan"))

        # Report the dominant conversion method used across contributors.
        methods = [item.coord_method for item in items]
        unique_method = methods[0] if len(set(methods)) == 1 else "mixed"

        # Current effective offset and proposed new absolute offset
        old = effective_offsets.get(pix_label)
        old_az = old[0] if old else float("nan")
        old_el = old[1] if old else float("nan")
        new_az = old_az + mean_az_deg if np.isfinite(old_az) else float("nan")
        new_el = old_el + mean_el_deg if np.isfinite(old_el) else float("nan")

        delta_rows.append(
            {
                "pix_label": pix_label,
                "line": items[0].line,
                "target_mixer": items[0].target_mixer,
                "source_count": n,
                "mean_dx_pix": mean_dx,
                "mean_dy_pix": mean_dy,
                "residual_az_deg": mean_az_deg,
                "residual_el_deg": mean_el_deg,
                "sigma_az_deg": sigma_az,
                "sigma_el_deg": sigma_el,
                "effective_az_deg": old_az,
                "effective_el_deg": old_el,
                "proposed_az_deg": new_az,
                "proposed_el_deg": new_el,
                "coord_method": unique_method,
                "contributors": ",".join(item.source for item in items),
            }
        )

    write_csv(
        delta_rows,
        out_path,
        [
            "pix_label",
            "line",
            "target_mixer",
            "source_count",
            "mean_dx_pix",
            "mean_dy_pix",
            "residual_az_deg",
            "residual_el_deg",
            "sigma_az_deg",
            "sigma_el_deg",
            "effective_az_deg",
            "effective_el_deg",
            "proposed_az_deg",
            "proposed_el_deg",
            "coord_method",
            "contributors",
        ],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure moment-0 cross-correlation shifts and write delta-only outputs")
    parser.add_argument("--config", required=True, help="Path to JSON config file for batch processing")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    data_root_cfg = str(config.get("data_root", "Data/level2"))
    data_root = (repo_root / data_root_cfg).resolve() if not Path(data_root_cfg).is_absolute() else Path(data_root_cfg)

    default_run = str(config.get("run", "latest"))
    measurements_cfg = str(config.get("measurements_output", config.get("csv_output", "Data/level2/crosscorr_measurements.csv")))
    measurements_out = (repo_root / measurements_cfg).resolve() if not Path(measurements_cfg).is_absolute() else Path(measurements_cfg)

    delta_cfg = str(config.get("delta_output", "Data/level2/crosscorr_deltas.csv"))
    delta_out = (repo_root / delta_cfg).resolve() if not Path(delta_cfg).is_absolute() else Path(delta_cfg)

    jobs = parse_jobs(config, default_run=default_run, data_root=data_root)

    results: list[JobResult] = []
    for job in jobs:
        try:
            row = process_job(data_root=data_root, job=job, config=config)
            print(
                f"[{job.source} {job.line} M{job.mixer}] "
                f"dx={row.dx_pix:+.1f} pix dy={row.dy_pix:+.1f} pix "
                f"AZ={row.az_deg:+.6f}±{fmt_uncertainty(row.sigma_az_deg)} "
                f"EL={row.el_deg:+.6f}±{fmt_uncertainty(row.sigma_el_deg)} "
                f"({row.coord_method})"
            )
            results.append(row)
        except Exception as exc:
            results.append(
                JobResult(
                    source=job.source,
                    run_dir="",
                    line=job.line,
                    target_mixer=job.target_mixer,
                    mixer=job.mixer,
                    reference_cube="",
                    target_cube="",
                    reference_moment0="",
                    target_moment0="",
                    shift_x=0,
                    shift_y=0,
                    dx_pix=float("nan"),
                    dy_pix=float("nan"),
                    az_deg=float("nan"),
                    el_deg=float("nan"),
                    coord_method="error",
                    status=f"ERROR: {exc}",
                    correlation_png="",
                )
            )
            print(f"[{job.source} {job.line} M{job.mixer}] ERROR: {exc}")

    write_csv(
        [row.__dict__ for row in results],
        measurements_out,
        [
            "source",
            "run_dir",
            "line",
            "target_mixer",
            "mixer",
            "reference_cube",
            "target_cube",
            "reference_moment0",
            "target_moment0",
            "shift_x",
            "shift_y",
            "dx_pix",
            "dy_pix",
            "az_deg",
            "el_deg",
            "sigma_x_pix",
            "sigma_y_pix",
            "sigma_az_deg",
            "sigma_el_deg",
            "coord_method",
            "status",
            "correlation_png",
        ],
    )
    # Resolve offsets file path for relationship columns
    offsets_cfg = str(config.get("offsets_file", ""))
    offsets_path: Path | None = None
    if offsets_cfg:
        offsets_path = ((repo_root / offsets_cfg).resolve()
                        if not Path(offsets_cfg).is_absolute()
                        else Path(offsets_cfg))
        if not offsets_path.exists():
            print(f"NOTE: offsets_file '{offsets_path}' not found — "
                  f"skipping effective/proposed columns in delta CSV")
            offsets_path = None

    write_delta_csv(results, delta_out, offsets_file=offsets_path)

    print(f"Saved measurements CSV: {measurements_out}")
    print(f"Saved delta CSV: {delta_out}")

    # ── Relationship summary ──────────────────────────────────────────
    if offsets_path is not None:
        eff = _read_effective_offsets(offsets_path)
        if eff:
            print(f"\n{'='*70}")
            print("OFFSET RELATIONSHIP SUMMARY")
            print(f"{'='*70}")
            print(f"  Offsets file: {offsets_path}")
            print(f"  Formula:  proposed_AS_MEASURED = effective_offset + residual")
            print(f"  (The cross-corr measures the RESIDUAL shift; the offsets")
            print(f"   file stores the ABSOLUTE offset per mixer.)")
            print(f"  {'Mixer':8s} {'Effective':>24s}  {'Residual':>24s}  {'Proposed':>24s}")
            print(f"  {'':8s} {'AZ (deg)':>12s} {'EL (deg)':>12s}  "
                  f"{'AZ (deg)':>12s} {'EL (deg)':>12s}  "
                  f"{'AZ (deg)':>12s} {'EL (deg)':>12s}")
            print(f"  {'-'*8} {'-'*24}  {'-'*24}  {'-'*24}")
            for r in results:
                if r.status != "OK":
                    continue
                pl = pix_label_from_line_mixer(r.line, r.mixer)
                old = eff.get(pl)
                old_az = old[0] if old else float("nan")
                old_el = old[1] if old else float("nan")
                new_az = old_az + r.az_deg if np.isfinite(old_az) else float("nan")
                new_el = old_el + r.el_deg if np.isfinite(old_el) else float("nan")
                print(f"  {pl:8s} {old_az:12.6f} {old_el:12.6f}  "
                      f"{r.az_deg:12.6f} {r.el_deg:12.6f}  "
                      f"{new_az:12.6f} {new_el:12.6f}")
            print(f"  {'='*70}")


if __name__ == "__main__":
    main()
