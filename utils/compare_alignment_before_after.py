#!/usr/bin/env python3
"""Compare cross-correlation alignment before and after offset correction.

For a given source and set of mixer pairs, measures cross-correlation
shifts on the cubes in a run directory and generates:

1. **Annotated cross-correlation maps** — PNG + FITS showing the full 2-D
   correlation surface with zero-lag centre ("C") and correlation peak ("P")
   marked, plus an offset arrow and info box with pixel/angular shifts.
2. **Before/after comparison figures** — when ``--compare-with`` is passed,
   a 2-panel figure per mixer pair is saved.

Usage
-----

.. code-block:: bash

    # ----  Before correction (theory-only offsets)  ----
    python utils/compare_alignment_before_after.py \\
        --config utils/measure_mixer_crosscorr_config_obj.json \\
        --run obj1_theory \\
        --label theory \\
        --output-dir Data/level2/G337/alignment_comparison/theory

    # ----  After correction (measured offsets applied)  ----
    python utils/compare_alignment_before_after.py \\
        --config utils/measure_mixer_crosscorr_config_obj.json \\
        --run obj1_measured \\
        --label measured \\
        --output-dir Data/level2/G337/alignment_comparison/measured \\
        --compare-with Data/level2/G337/alignment_comparison/theory
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

# Ensure utils/ is importable regardless of CWD (needed for viz_helpers)
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from scipy.signal import fftconvolve

from viz_helpers import find_latest_run_dir  # canonical implementation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LINE_TARGETS: dict[str, dict[str, object]] = {
    "CII": {"target_mixer": 8, "mixers": [5]},
    "NII": {"target_mixer": 3, "mixers": [2, 6]},
}

# ---------------------------------------------------------------------------
# File / data helpers (mirrored from measure_mixer_crosscorr)
# ---------------------------------------------------------------------------


def header_float(header: fits.Header, key: str, default: float = 0.0) -> float:
    value = header.get(key, default)
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float(default)


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
        raise FileNotFoundError(
            f"No cube found in {run_dir} for line={line} mixer={mixer}"
        )
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


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------


def prep_map(image: np.ndarray) -> np.ndarray:
    out = np.array(image, dtype=float)
    median = np.nanmedian(out)
    if np.isfinite(median):
        out = out - median
    out[~np.isfinite(out)] = 0.0
    return out


def measure_shift_integer(
    reference_map: np.ndarray, target_map: np.ndarray
) -> tuple[int, int, float, np.ndarray]:
    """Cross-correlate two moment-0 maps.

    Returns
    -------
    lag_x, lag_y : int
        Integer pixel lags (peak relative to centre).
    peak_value : float
        Maximum correlation value.
    corr : np.ndarray
        Full 2-D correlation surface.
    """
    ref = prep_map(reference_map)
    tgt = prep_map(target_map)
    corr = fftconvolve(ref, tgt[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    center_y, center_x = (s // 2 for s in corr.shape)
    lag_x = int(peak_x - center_x)
    lag_y = int(peak_y - center_y)
    return lag_x, lag_y, float(corr[peak_y, peak_x]), corr


# ---------------------------------------------------------------------------
# Observer metadata / coordinate conversion
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
    data_root: Path, source: str, line: str
) -> tuple[float, float, float, str] | None:
    import glob as _glob

    level1_dir = data_root.parent / "level1" / source
    pattern = str(level1_dir / f"{line}_*_L10.fits")
    l1_files = sorted(_glob.glob(pattern))
    if not l1_files:
        return None
    lats, lons, alts, utimes = [], [], [], []
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
        return None
    return (
        float(np.median(lats)),
        float(np.median(lons)),
        float(np.median(alts)),
        Time(float(np.median(utimes)), format="unix").iso,
    )


def get_observer_metadata(
    config: dict[str, object], data_root: Path, source: str, line: str
) -> tuple[float, float, float, str] | None:
    observer_cfg = config.get("observer")
    if isinstance(observer_cfg, dict):
        lat = observer_cfg.get("lat_deg")
        lon = observer_cfg.get("lon_deg")
        alt = observer_cfg.get("alt_m")
        time = observer_cfg.get("obs_time_utc")
        if all(v is not None for v in (lat, lon, alt, time)):
            return (float(lat), float(lon), float(alt), str(time))
    if config.get("auto_detect_observer", False):
        return _auto_detect_from_level1(data_root, source, line)
    return None


def pixel_offset_to_azel(
    dx_pix: float,
    dy_pix: float,
    ref_header: fits.Header,
    observer: tuple[float, float, float, str] | None,
) -> tuple[float, float, str]:
    cdelt1 = header_float(ref_header, "CDELT1", 0.0)
    cdelt2 = header_float(ref_header, "CDELT2", 0.0)
    dlon_deg = dx_pix * cdelt1
    dlat_deg = dy_pix * cdelt2
    if observer is not None:
        ref_glon = header_float(ref_header, "CRVAL1", 0.0)
        ref_glat = header_float(ref_header, "CRVAL2", 0.0)
        az_deg, el_deg = galactic_offset_to_azel(
            observer[0], observer[1], observer[2], observer[3],
            ref_glon, ref_glat, dlon_deg, dlat_deg,
        )
        return az_deg, el_deg, "astropy"
    return dlon_deg, dlat_deg, "cdelt_fallback"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _add_centre_crosshair(
    ax: plt.Axes, cx: float, cy: float, color: str = "white", **kwargs
) -> None:
    """Draw dashed crosshair lines through (cx, cy)."""
    defaults = dict(linestyle="--", linewidth=1.0, alpha=0.7)
    defaults.update(kwargs)
    ax.axvline(cx, color=color, **defaults)
    ax.axhline(cy, color=color, **defaults)


def _add_peak_marker(
    ax: plt.Axes, px: float, py: float, label: str = "Peak", **kwargs
) -> None:
    """Draw a red cross at the peak position with an annotation."""
    defaults = dict(
        markersize=14, markeredgewidth=2.5, marker="+", color="red", zorder=10
    )
    defaults.update(kwargs)
    ax.plot(px, py, **defaults)


def _add_offset_arrow(
    ax: plt.Axes,
    cx: float,
    cy: float,
    px: float,
    py: float,
    dx: float,
    dy: float,
) -> None:
    """Draw an arrow from centre to peak, annotated with the pixel offset."""
    ax.annotate(
        "",
        xy=(px, py),
        xytext=(cx, cy),
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            lw=2.0,
            alpha=0.9,
            connectionstyle="arc3,rad=0",
        ),
        zorder=11,
    )
    # Annotate the offset value near the midpoint
    mid_x, mid_y = (cx + px) / 2, (cy + py) / 2
    ax.annotate(
        f"({dx:+.1f}, {dy:+.1f}) pix",
        xy=(mid_x, mid_y),
        fontsize=9,
        fontweight="bold",
        color="red",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        zorder=12,
    )


def plot_correlation_map(
    corr: np.ndarray,
    out_path: Path,
    source: str,
    line: str,
    ref_mixer: int,
    tgt_mixer: int,
    label: str,
    lag_x: int,
    lag_y: int,
    dx_pix: float,
    dy_pix: float,
    peak_value: float,
    dlon_deg: float | None = None,
    dlat_deg: float | None = None,
    az_deg: float | None = None,
    el_deg: float | None = None,
) -> None:
    """Save a highly annotated cross-correlation map.

    Marks:
    - The zero-lag centre (white dashed crosshair)
    - The correlation peak (red cross)
    - An arrow from centre → peak with the pixel offset
    - Info box with peak value and converted angles
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(corr, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(
        f"{source}  {line}  M{ref_mixer} vs M{tgt_mixer}\n"
        f"Cross-correlation  [{label}]",
        fontsize=11,
    )
    ax.set_xlabel("X lag index")
    ax.set_ylabel("Y lag index")
    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.85)

    cy, cx = (s // 2 for s in corr.shape)
    peak_y = cy + lag_y
    peak_x = cx + lag_x

    # Centre crosshair
    _add_centre_crosshair(ax, cx, cy, color="white")

    # Peak marker
    _add_peak_marker(ax, peak_x, peak_y)

    # Offset arrow
    _add_offset_arrow(ax, cx, cy, peak_x, peak_y, dx_pix, dy_pix)

    # Info box
    lines = [
        f"Peak value: {peak_value:.3g}",
        f"Peak position: ({peak_x}, {peak_y})",
        f"Centre (zero-lag): ({cx}, {cy})",
        f"Pixel shift: ({dx_pix:+.1f}, {dy_pix:+.1f}) pix",
        f"Offset magnitude: {np.sqrt(dx_pix**2 + dy_pix**2):.1f} pix",
    ]
    if dlon_deg is not None and dlat_deg is not None:
        lines.append(f"Galactic: ({dlon_deg:+.4f}, {dlat_deg:+.4f})°")
    if az_deg is not None and el_deg is not None:
        lines.append(f"AZ/EL: ({az_deg:+.5f}, {el_deg:+.5f})°")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
        zorder=13,
    )

    # Zoom annotation showing centre and peak in context
    ax.annotate(
        "C",
        xy=(cx, cy),
        fontsize=12,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle,pad=0.2", facecolor="black", alpha=0.5),
        zorder=13,
    )
    ax.annotate(
        "P",
        xy=(peak_x, peak_y),
        fontsize=12,
        fontweight="bold",
        color="red",
        ha="center",
        va="center",
        xytext=(8, 8),
        textcoords="offset points",
        bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", alpha=0.85),
        zorder=13,
    )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_before_after_comparison(
    corr_before: np.ndarray,
    corr_after: np.ndarray,
    results_before: dict[str, object],
    results_after: dict[str, object],
    out_path: Path,
) -> None:
    """Create a 2-panel before/after comparison figure.

    Layout:
        left:   Corr map BEFORE (theory)
        right:  Corr map AFTER  (measured)
    """
    source = str(results_before.get("source", ""))
    line = str(results_before.get("line", ""))
    ref_mixer = int(results_before.get("ref_mixer", 0))
    tgt_mixer = int(results_before.get("tgt_mixer", 0))

    fig, (ax_before, ax_after) = plt.subplots(
        1, 2, figsize=(17, 7.5), dpi=150
    )

    cy_c, cx_c = (s // 2 for s in corr_before.shape)

    def _plot_one(
        ax: plt.Axes,
        corr: np.ndarray,
        res: dict[str, object],
        subtitle: str,
    ) -> None:
        im = ax.imshow(corr, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(
            f"Cross-correlation  [{subtitle}]", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("X lag index")
        ax.set_ylabel("Y lag index")
        plt.colorbar(im, ax=ax, label="Correlation", shrink=0.88)

        dx = float(res.get("dx_pix", 0))
        dy = float(res.get("dy_pix", 0))
        lag_x = int(res.get("lag_x", 0))
        lag_y = int(res.get("lag_y", 0))
        peak_y = cy_c + lag_y
        peak_x = cx_c + lag_x

        _add_centre_crosshair(ax, cx_c, cy_c, color="white")
        _add_peak_marker(ax, peak_x, peak_y)
        _add_offset_arrow(ax, cx_c, cy_c, peak_x, peak_y, dx, dy)

        offset_mag = np.sqrt(dx**2 + dy**2)
        dlon = res.get("dlon_deg")
        dlat = res.get("dlat_deg")

        lines = [
            f"Pixel shift: ({dx:+.1f}, {dy:+.1f}) pix",
            f"Offset magnitude: {offset_mag:.1f} pix",
        ]
        if dlon is not None and dlat is not None:
            lines.append(f"Galactic: ({float(dlon):+.4f}, {float(dlat):+.4f})°")
        az = res.get("az_deg")
        el = res.get("el_deg")
        if az is not None and el is not None:
            lines.append(f"AZ/EL: ({float(az):+.5f}, {float(el):+.5f})°")

        ax.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
            zorder=13,
        )

        # Centre + peak labels
        ax.annotate(
            "C", xy=(cx_c, cy_c),
            fontsize=11, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="black", alpha=0.5),
            zorder=13,
        )
        ax.annotate(
            "P", xy=(peak_x, peak_y),
            fontsize=11, fontweight="bold", color="red",
            ha="center", va="center",
            xytext=(8, 8), textcoords="offset points",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", alpha=0.85),
            zorder=13,
        )

    _plot_one(ax_before, corr_before, results_before, "BEFORE (theory)")
    _plot_one(ax_after, corr_after, results_after, "AFTER (measured)")

    fig.suptitle(
        f"{source}  {line}  M{ref_mixer} vs M{tgt_mixer} — Alignment Before/After",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_one_mixer_pair(
    data_root: Path,
    run_dir: Path,
    source: str,
    line: str,
    ref_mixer: int,
    tgt_mixer: int,
    config: dict[str, object],
    output_dir: Path,
    label: str,
) -> dict[str, object]:
    """Measure cross-correlation for one mixer pair and save all outputs.

    Returns a dict with all measurements for later comparison.
    """
    line = line.upper()
    ref_path = select_cube(run_dir, line, ref_mixer)
    tgt_path = select_cube(run_dir, line, tgt_mixer)

    ref_cube, ref_header = load_cube(ref_path)
    tgt_cube, _ = load_cube(tgt_path)

    if ref_cube.shape[1:] != tgt_cube.shape[1:]:
        raise ValueError(
            f"Spatial shape mismatch: ref={ref_cube.shape[1:]} "
            f"target={tgt_cube.shape[1:]}"
        )

    nchan = min(ref_cube.shape[0], tgt_cube.shape[0])
    ref_cube = ref_cube[:nchan]
    tgt_cube = tgt_cube[:nchan]

    map_ref = build_moment0_map(ref_cube)
    map_tgt = build_moment0_map(tgt_cube)

    # Cross-correlation
    lag_x, lag_y, peak_val, corr = measure_shift_integer(map_ref, map_tgt)
    dx_pix = -float(lag_x)
    dy_pix = -float(lag_y)

    # Coordinate conversion
    obs = get_observer_metadata(config, data_root, source, line)
    az_deg, el_deg, coord_method = pixel_offset_to_azel(
        dx_pix, dy_pix, ref_header, obs
    )

    cdelt1 = header_float(ref_header, "CDELT1", 0.0)
    cdelt2 = header_float(ref_header, "CDELT2", 0.0)
    dlon_deg = dx_pix * cdelt1
    dlat_deg = dy_pix * cdelt2

    if coord_method == "cdelt_fallback":
        print(
            f"  WARNING: No observer metadata for {source}/{line}. "
            f"Using direct CDELT conversion (may be rotated)."
        )

    pair_tag = f"{source}_{line}_M{ref_mixer}_vs_M{tgt_mixer}"

    # Save correlation map
    corr_png = output_dir / f"crosscorr_{pair_tag}.png"
    plot_correlation_map(
        corr,
        corr_png,
        source,
        line,
        ref_mixer,
        tgt_mixer,
        label,
        lag_x,
        lag_y,
        dx_pix,
        dy_pix,
        peak_val,
        dlon_deg=dlon_deg,
        dlat_deg=dlat_deg,
        az_deg=az_deg,
        el_deg=el_deg,
    )

    # Save correlation surface as FITS for potential reuse
    corr_fits = output_dir / f"crosscorr_{pair_tag}.fits"
    fits.writeto(corr_fits, corr, overwrite=True)

    result: dict[str, object] = {
        "source": source,
        "line": line,
        "ref_mixer": ref_mixer,
        "tgt_mixer": tgt_mixer,
        "ref_cube": ref_path.name,
        "tgt_cube": tgt_path.name,
        "lag_x": lag_x,
        "lag_y": lag_y,
        "dx_pix": dx_pix,
        "dy_pix": dy_pix,
        "peak_value": peak_val,
        "dlon_deg": dlon_deg,
        "dlat_deg": dlat_deg,
        "az_deg": az_deg,
        "el_deg": el_deg,
        "coord_method": coord_method,
        "corr_fits": str(corr_fits),
        "corr_png": str(corr_png),
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result


def process_all_pairs(
    data_root: Path,
    run_dir: Path,
    source: str,
    config: dict[str, object],
    output_dir: Path,
    label: str,
) -> list[dict[str, object]]:
    """Process all mixer pairs for a source.

    Reads line/mixer configuration from *config*.
    """
    line_targets = config.get("line_targets")
    if isinstance(line_targets, dict):
        pairs: list[tuple[str, int, int]] = []
        for line, entry in line_targets.items():
            if not isinstance(entry, dict):
                continue
            ref = int(entry.get("target_mixer", 8 if line.upper() == "CII" else 3))
            mixers = entry.get("mixers")
            if isinstance(mixers, list):
                mixer_list = [int(m) for m in mixers]
            elif "mixer" in entry:
                mixer_list = [int(entry["mixer"])]
            else:
                mixer_list = [5] if line.upper() == "CII" else [2, 6]
            for m in mixer_list:
                if m != 0 and m != ref:
                    pairs.append((line.upper(), ref, m))
    else:
        pairs = [
            ("CII", 8, 5),
            ("NII", 3, 2),
            ("NII", 3, 6),
        ]

    results: list[dict[str, object]] = []
    for line, ref_mixer, tgt_mixer in pairs:
        print(f"  [{label}] {source} {line} M{ref_mixer} vs M{tgt_mixer} ...")
        try:
            res = process_one_mixer_pair(
                data_root=data_root,
                run_dir=run_dir,
                source=source,
                line=line,
                ref_mixer=ref_mixer,
                tgt_mixer=tgt_mixer,
                config=config,
                output_dir=output_dir,
                label=label,
            )
            print(
                f"    dx={res['dx_pix']:+.1f} pix  "
                f"dy={res['dy_pix']:+.1f} pix  "
                f"AZ={res['az_deg']:+.6f}°  "
                f"EL={res['el_deg']:+.6f}°"
            )
            results.append(res)
        except Exception as exc:
            print(f"    ERROR: {exc}")

    return results


def generate_comparisons(
    before_dir: Path,
    after_dir: Path,
    comparison_dir: Path,
) -> None:
    """Load before/after results and generate comparison figures."""
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load results from JSON files
    import json as _json

    before_json = before_dir / "results.json"
    after_json = after_dir / "results.json"

    if not before_json.exists():
        print(f"Before results not found: {before_json}")
        return
    if not after_json.exists():
        print(f"After results not found: {after_json}")
        return

    before_results: list[dict[str, object]] = _json.loads(
        before_json.read_text(encoding="utf-8")
    )
    after_results: list[dict[str, object]] = _json.loads(
        after_json.read_text(encoding="utf-8")
    )

    # Index by (source, line, ref_mixer, tgt_mixer)
    def _key(r: dict[str, object]) -> tuple[str, str, int, int]:
        return (
            str(r["source"]),
            str(r["line"]),
            int(r["ref_mixer"]),
            int(r["tgt_mixer"]),
        )

    after_index = {_key(r): r for r in after_results}

    for b_res in before_results:
        k = _key(b_res)
        a_res = after_index.get(k)
        if a_res is None:
            print(f"  No after result for {k}, skipping comparison")
            continue

        source = str(b_res["source"])
        line = str(b_res["line"])
        ref_m = int(b_res["ref_mixer"])
        tgt_m = int(b_res["tgt_mixer"])
        pair_tag = f"{source}_{line}_M{ref_m}_vs_M{tgt_m}"

        # Load correlation FITS files
        corr_before_path = Path(str(b_res.get("corr_fits", "")))
        corr_after_path = Path(str(a_res.get("corr_fits", "")))

        if not corr_before_path.exists() or not corr_after_path.exists():
            print(f"  Missing correlation FITS for {pair_tag}")
            continue

        with fits.open(corr_before_path) as hdul:
            corr_before = np.array(hdul[0].data, dtype=float)
        with fits.open(corr_after_path) as hdul:
            corr_after = np.array(hdul[0].data, dtype=float)

        # Generate comparison figure
        comp_png = comparison_dir / f"comparison_{pair_tag}.png"
        print(f"  Generating comparison: {comp_png.name}")
        plot_before_after_comparison(
            corr_before=corr_before,
            corr_after=corr_after,
            results_before=b_res,
            results_after=a_res,
            out_path=comp_png,
        )

    print(f"Comparison figures saved to: {comparison_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare cross-correlation alignment before and after offset correction"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config file (same format as measure_mixer_crosscorr)",
    )
    parser.add_argument(
        "--run",
        default="latest",
        help="Run directory selector (e.g. 'obj1', 'latest', '12')",
    )
    parser.add_argument(
        "--label",
        default="unknown",
        help="Label for this state (e.g. 'theory', 'measured')",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for results (created if needed)",
    )
    parser.add_argument(
        "--compare-with",
        default=None,
        help="Path to a previous output directory (with results.json) for before/after comparison",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override source(s) from config (comma-separated)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    data_root_cfg = str(config.get("data_root", "Data/level2"))
    data_root = (
        (repo_root / data_root_cfg).resolve()
        if not Path(data_root_cfg).is_absolute()
        else Path(data_root_cfg)
    )

    # Resolve sources
    if args.source:
        sources = [s.strip() for s in args.source.split(",") if s.strip()]
    elif isinstance(config.get("sources"), list):
        sources = [str(s).strip() for s in config["sources"]]
    elif isinstance(config.get("source"), str) and str(config.get("source")).strip():
        sources = [str(config.get("source")).strip()]
    else:
        sources = sorted(
            child.name for child in data_root.iterdir() if child.is_dir()
        )

    output_dir = (
        Path(args.output_dir).resolve()
        if Path(args.output_dir).is_absolute()
        else (repo_root / args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    label = args.label

    # ------------------------------------------------------------------
    # Process each source
    # ------------------------------------------------------------------
    all_results: list[dict[str, object]] = []
    for source in sources:
        source_dir = data_root / source
        if not source_dir.is_dir():
            print(f"Source directory not found: {source_dir}, skipping")
            continue

        try:
            run_dir = find_latest_run_dir(source_dir, args.run)
        except FileNotFoundError:
            print(f"Run '{args.run}' not found under {source_dir}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {source}  |  run: {run_dir.name}  |  label: {label}")
        print(f"{'='*60}")

        results = process_all_pairs(
            data_root=data_root,
            run_dir=run_dir,
            source=source,
            config=config,
            output_dir=output_dir,
            label=label,
        )
        all_results.extend(results)

    # Save results manifest
    manifest_path = output_dir / "results.json"
    manifest_path.write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults manifest saved to: {manifest_path}")
    print(f"Output directory: {output_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY [{label}]")
    print(f"{'='*60}")
    for r in all_results:
        print(
            f"  {r['source']:6s} {r['line']:3s} "
            f"M{r['ref_mixer']} vs M{r['tgt_mixer']}:  "
            f"dx={r['dx_pix']:+.1f}  dy={r['dy_pix']:+.1f} pix  "
            f"|offset|={np.sqrt(float(r['dx_pix'])**2 + float(r['dy_pix'])**2):.1f} pix  "
            f"AZ={r['az_deg']:+.6f}°  EL={r['el_deg']:+.6f}°"
        )

    # ------------------------------------------------------------------
    # Generate comparisons if requested
    # ------------------------------------------------------------------
    if args.compare_with:
        compare_with = (
            Path(args.compare_with).resolve()
            if Path(args.compare_with).is_absolute()
            else (repo_root / args.compare_with).resolve()
        )
        comparison_dir = output_dir.parent / "comparison_before_after"
        print(f"\n{'='*60}")
        print(f"Generating before/after comparisons")
        print(f"  Before: {compare_with}")
        print(f"  After:  {output_dir}")
        print(f"{'='*60}")
        generate_comparisons(compare_with, output_dir, comparison_dir)


if __name__ == "__main__":
    main()
