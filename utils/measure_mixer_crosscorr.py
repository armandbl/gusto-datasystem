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
from scipy.signal import fftconvolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_PATTERN = re.compile(r"^run\s+(\d+)$", re.IGNORECASE)


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
) -> tuple[float, float, float, str] | None:
    """Extract observer metadata from Level-1 FITS files.

    Globs ``Data/level1/{source}/{line}_*_L10.fits``, reads GON_LAT /
    GON_LON / GON_ALT from the primary header and computes the median
    UNIXTIME from the binary table.
    """
    level1_dir = data_root.parent / "level1" / source
    pattern = str(level1_dir / f"{line}_*_L10.fits")
    l1_files = sorted(_glob.glob(pattern))

    if not l1_files:
        print(f"  [auto-detect] No Level-1 files found for {source}/{line} "
              f"at {pattern}")
        return None

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
) -> tuple[float, float, float, str] | None:
    """Resolve observer metadata via config or auto-detection.

    Returns ``(lat_deg, lon_deg, alt_m, obs_time_utc_iso)`` or *None*.
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
        return _auto_detect_from_level1(data_root, source, line)

    print(f"  [metadata] No observer config for {source}/{line}; "
          f"set 'auto_detect_observer': true or add 'observer' section")
    return None


# ---------------------------------------------------------------------------
# File / data helpers
# ---------------------------------------------------------------------------

def find_latest_run_dir(source_dir: Path, run_selector: str) -> Path:
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


def save_moment0_products(cube: np.ndarray, header: fits.Header, fits_path: Path, png_path: Path, title: str) -> np.ndarray:
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


def measure_shift_integer(reference_map: np.ndarray, target_map: np.ndarray) -> tuple[int, int, np.ndarray]:
    ref = prep_map(reference_map)
    tgt = prep_map(target_map)
    corr = fftconvolve(ref, tgt[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    center_y, center_x = (s // 2 for s in corr.shape)
    lag_y = int(peak_y - center_y)
    lag_x = int(peak_x - center_x)
    return lag_x, lag_y, corr


def header_float(header: fits.Header, key: str, default: float = 0.0) -> float:
    value = header.get(key, default)
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float(default)


def save_correlation_png(
    corr: np.ndarray,
    out_path: Path,
    title: str,
    peak_x: int | None = None,
    peak_y: int | None = None,
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

    center_y, center_x = (s // 2 for s in corr.shape)
    dx = center_x - peak_x
    dy = center_y - peak_y
    ax.plot(peak_x, peak_y, "r+", markersize=14, markeredgewidth=2.5)
    ax.annotate(
        f"peak=({peak_x},{peak_y})\ndx={dx:+d}, dy={dy:+d}",
        xy=(peak_x, peak_y),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
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

    map_ref = save_moment0_products(
        ref_cube,
        ref_header,
        reference_moment0_path,
        reference_moment0_png,
        title=f"{job.source} {job.line} M{job.target_mixer} moment0",
    )
    map_tgt = save_moment0_products(
        tgt_cube,
        ref_header,
        target_moment0_path,
        target_moment0_png,
        title=f"{job.source} {job.line} M{job.mixer} moment0",
    )

    lag_x, lag_y, corr = measure_shift_integer(map_ref, map_tgt)
    dx_pix = -float(lag_x)
    dy_pix = -float(lag_y)

    # ------------------------------------------------------------------
    # Convert the measured *Galactic* pixel offset to true AZ / EL
    # ------------------------------------------------------------------
    cdelt1 = header_float(ref_header, "CDELT1", 0.0)
    cdelt2 = header_float(ref_header, "CDELT2", 0.0)

    dlon_deg = dx_pix * cdelt1   # Galactic longitude offset  (deg)
    dlat_deg = dy_pix * cdelt2   # Galactic latitude offset   (deg)

    obs = get_observer_metadata(config, data_root, job.source, job.line)
    if obs is not None:
        ref_glon = header_float(ref_header, "CRVAL1", 0.0)
        ref_glat = header_float(ref_header, "CRVAL2", 0.0)
        az_deg, el_deg = galactic_offset_to_azel(
            obs[0], obs[1], obs[2], obs[3],
            ref_glon, ref_glat, dlon_deg, dlat_deg,
        )
        coord_method = "astropy"
    else:
        print(
            f"  WARNING: No observer metadata for {job.source}/{job.line}. "
            f"Using direct Galactic→AZ/EL conversion (may be rotated). "
            f"Set 'auto_detect_observer': true or add an 'observer' section "
            f"to the config for physically correct AZ/EL offsets."
        )
        az_deg = dlon_deg
        el_deg = dlat_deg
        coord_method = "cdelt_fallback"

    corr_png = compare_dir / f"crosscorr_{job.source}_{job.line}_M{job.target_mixer}_vs_M{job.mixer}.png"
    save_correlation_png(corr, corr_png, title=f"{job.source} {job.line} M{job.target_mixer} vs M{job.mixer} | lag=({lag_x},{lag_y})")

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
        shift_x=lag_x,
        shift_y=lag_y,
        dx_pix=dx_pix,
        dy_pix=dy_pix,
        az_deg=az_deg,
        el_deg=el_deg,
        coord_method=coord_method,
        status="OK",
        correlation_png=str(corr_png),
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict[str, object]], out_path: Path, fieldnames: list[str]) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_delta_csv(rows: list[JobResult], out_path: Path) -> None:
    grouped: dict[str, list[JobResult]] = {}
    for row in rows:
        if row.status != "OK":
            continue
        pix_label = pix_label_from_line_mixer(row.line, row.mixer)
        grouped.setdefault(pix_label, []).append(row)

    delta_rows: list[dict[str, object]] = []
    for pix_label, items in sorted(grouped.items()):
        mean_az_deg = float(np.mean([item.az_deg for item in items]))
        mean_el_deg = float(np.mean([item.el_deg for item in items]))
        # Report the dominant conversion method used across contributors.
        methods = [item.coord_method for item in items]
        unique_method = methods[0] if len(set(methods)) == 1 else "mixed"
        delta_rows.append(
            {
                "pix_label": pix_label,
                "line": items[0].line,
                "target_mixer": items[0].target_mixer,
                "source_count": len(items),
                "mean_dx_pix": float(np.mean([item.dx_pix for item in items])),
                "mean_dy_pix": float(np.mean([item.dy_pix for item in items])),
                "mean_az_deg": mean_az_deg,
                "mean_el_deg": mean_el_deg,
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
            "mean_az_deg",
            "mean_el_deg",
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
                f"AZ={row.az_deg:+.6f} EL={row.el_deg:+.6f} "
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
            "coord_method",
            "status",
            "correlation_png",
        ],
    )
    write_delta_csv(results, delta_out)

    print(f"Saved measurements CSV: {measurements_out}")
    print(f"Saved delta CSV: {delta_out}")


if __name__ == "__main__":
    main()
