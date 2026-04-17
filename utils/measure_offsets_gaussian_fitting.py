import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
import matplotlib
from matplotlib.patches import Ellipse

# Force non-interactive backend so this works in terminal sessions.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_PATTERN = re.compile(r"^run\s+(\d+)$", re.IGNORECASE)
LINE_PATTERN = re.compile(r"(CII|NII)", re.IGNORECASE)
MIXER_PATTERN = re.compile(r"_(\d+)", re.IGNORECASE)


@dataclass
class LoadedCube:
    """In-memory representation of one spectral cube and its velocity axis."""

    path: Path
    data: np.ndarray
    header: fits.Header
    vel_axis_kms: np.ndarray


@dataclass
class VelocityConfig:
    """Per-velocity settings resolved from config file and/or CLI options."""

    velocities: list[float]
    velocity_hints: dict[float, tuple[float, float]]
    velocity_weights_config: dict[float, float]
    velocity_sizes_config: dict[float, tuple[float, float]]
    velocity_rho_config: dict[float, float]
    velocity_center_shift_config: dict[float, float]
    velocity_theta_shift_config: dict[float, float]
    velocity_fix_shape_config: dict[float, bool]


@dataclass
class OffsetAccumulator:
    """Collects per-velocity measurements and debug rows for final aggregation."""

    dx_pix_values: list[float]
    dy_pix_values: list[float]
    az_deg_values: list[float]
    el_deg_values: list[float]
    velocity_weights: list[float]
    debug_rows: list[dict[str, object]]


def new_offset_accumulator() -> OffsetAccumulator:
    """Create an empty accumulator for offset statistics and debug output."""

    return OffsetAccumulator(
        dx_pix_values=[],
        dy_pix_values=[],
        az_deg_values=[],
        el_deg_values=[],
        velocity_weights=[],
        debug_rows=[],
    )


def find_run_dir(base_dir: Path, run: str) -> Path:
    """Return a run directory by explicit number or latest available run."""

    if run.lower() == "latest":
        candidates = []
        for child in base_dir.iterdir():
            if not child.is_dir():
                continue
            match = RUN_PATTERN.match(child.name)
            if match:
                candidates.append((int(match.group(1)), child))
        if not candidates:
            raise FileNotFoundError(f"No run directories found under {base_dir}")
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    run_dir = base_dir / f"run {run}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return run_dir


def spectral_axis_kms(header: fits.Header, nchan: int) -> np.ndarray:
    """Build spectral axis from FITS header and normalize it to km/s when needed."""

    crval = float(header.get("CRVAL3", 0.0))
    cdelt = float(header.get("CDELT3", 1.0))
    crpix = float(header.get("CRPIX3", 1.0))
    cunit = str(header.get("CUNIT3", "")).strip().lower()

    axis = crval + ((np.arange(nchan) + 1.0) - crpix) * cdelt

    if "km/s" in cunit or "kms" in cunit:
        return axis

    if "m/s" in cunit or "ms-1" in cunit:
        if np.nanmax(np.abs(axis)) > 2000.0:
            return axis / 1000.0
        return axis

    if np.nanmax(np.abs(axis)) > 2000.0:
        return axis / 1000.0
    return axis


def line_from_filename(file_path: Path) -> str | None:
    """Extract spectral line tag (CII/NII) from a cube filename."""

    match = LINE_PATTERN.search(file_path.name)
    if not match:
        return None
    return match.group(1).upper()


def mixer_from_filename(file_path: Path) -> str | None:
    """Extract mixer number token from a cube filename."""

    match = MIXER_PATTERN.search(file_path.name)
    if not match:
        return None
    return match.group(1)


def load_cube_data(cube_path: Path) -> LoadedCube:
    """Load a FITS cube once and precompute its velocity axis."""

    with fits.open(cube_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None or data.ndim < 3:
        raise ValueError(f"Expected 3D cube in {cube_path}")

    if data.ndim == 4:
        data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError(f"Unsupported cube shape {data.shape} in {cube_path}")

    cube_data = np.array(data, dtype=float)
    vel_axis = spectral_axis_kms(header, cube_data.shape[0])
    return LoadedCube(path=cube_path, data=cube_data, header=header, vel_axis_kms=vel_axis)


def extract_slice_from_loaded_cube(cube: LoadedCube, target_vel: float) -> tuple[float, np.ndarray]:
    """Extract the nearest velocity channel frame from a preloaded cube."""

    idx = int(np.argmin(np.abs(cube.vel_axis_kms - target_vel)))
    actual_vel = float(cube.vel_axis_kms[idx])
    frame = np.array(cube.data[idx, :, :], dtype=float)
    return actual_vel, frame


def find_cube_for_line_and_mixer(cubes: list[Path], line: str, mixer: str) -> Path:
    """Select exactly one cube matching spectral line and mixer identifier."""

    matches = [
        cube
        for cube in cubes
        if line_from_filename(cube) == line and mixer_from_filename(cube) == mixer
    ]
    if not matches:
        raise FileNotFoundError(f"No cube found for line={line}, mixer={mixer}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple cubes found for line={line}, mixer={mixer}: {[m.name for m in matches]}"
        )
    return matches[0]


def safe_peak_xy(frame: np.ndarray) -> tuple[float, float]:
    """Return global peak position (x, y) in a frame with finite-value checks."""

    if not np.isfinite(frame).any():
        raise ValueError("Frame contains no finite values")
    iy, ix = np.unravel_index(np.nanargmax(frame), frame.shape)
    return float(ix), float(iy)


def find_peak_in_region(frame: np.ndarray, x_center: float, y_center: float, search_radius: int = 30) -> tuple[float, float]:
    """Find the peak within a circular region around (x_center, y_center)."""
    ny, nx = frame.shape
    xc = int(round(x_center))
    yc = int(round(y_center))
    
    # Extract the search region (clamped to frame bounds)
    x0 = max(0, xc - search_radius)
    x1 = min(nx, xc + search_radius + 1)
    y0 = max(0, yc - search_radius)
    y1 = min(ny, yc + search_radius + 1)
    
    cutout = frame[y0:y1, x0:x1]
    
    if not np.isfinite(cutout).any():
        raise ValueError("No finite values in search region")
    
    # Find peak in the cutout
    iy, ix = np.unravel_index(np.nanargmax(cutout), cutout.shape)
    
    # Convert back to full-frame coordinates
    x_peak = float(x0 + ix)
    y_peak = float(y0 + iy)
    return x_peak, y_peak


def gaussian_init_from_xy_correlation(
    x_sigma_xy: float,
    y_sigma_xy: float,
    rho_xy: float,
) -> tuple[float, float, float]:
    """Convert (sigma_x, sigma_y, rho_xy) in map axes into principal-axis Gaussian params.

    Returns (x_stddev_init, y_stddev_init, theta_init_rad) suitable for astropy Gaussian2D.
    """
    sx = float(abs(x_sigma_xy)) if np.isfinite(x_sigma_xy) and x_sigma_xy > 0.0 else 2.5
    sy = float(abs(y_sigma_xy)) if np.isfinite(y_sigma_xy) and y_sigma_xy > 0.0 else 2.5
    rho = float(np.clip(rho_xy, -0.99, 0.99)) if np.isfinite(rho_xy) else 0.0

    cov = np.array(
        [
            [sx * sx, rho * sx * sy],
            [rho * sx * sy, sy * sy],
        ],
        dtype=float,
    )

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = float(np.sqrt(max(eigvals[0], 1e-6)))
    minor = float(np.sqrt(max(eigvals[1], 1e-6)))
    theta = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return major, minor, theta


def fit_gaussian_centroid(
    frame: np.ndarray,
    x_guess: float,
    y_guess: float,
    half_window: int,
    x_stddev_init: float = 2.5,
    y_stddev_init: float = 2.5,
    rho_xy_init: float = 0.0,
    size_bounds_factor: float = 3.0,
    center_max_shift_pix: float = 8.0,
    theta_max_shift_deg: float = 45.0,
    fix_shape: bool = False,
) -> tuple[float, float, dict[str, float]]:
    """Fit a local 2D Gaussian plus constant background and return fitted centroid."""

    ny, nx = frame.shape
    xc = int(round(x_guess))
    yc = int(round(y_guess))

    x0 = max(0, xc - half_window)
    x1 = min(nx, xc + half_window + 1)
    y0 = max(0, yc - half_window)
    y1 = min(ny, yc + half_window + 1)

    cutout = np.array(frame[y0:y1, x0:x1], dtype=float)
    if cutout.size == 0:
        raise ValueError("Empty cutout for Gaussian fit")

    if np.isfinite(cutout).any():
        fill_value = float(np.nanmedian(cutout[np.isfinite(cutout)]))
    else:
        fill_value = 0.0
    cutout = np.nan_to_num(cutout, nan=fill_value)

    yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]

    local_x_guess = np.clip(x_guess - x0, 0.0, cutout.shape[1] - 1.0)
    local_y_guess = np.clip(y_guess - y0, 0.0, cutout.shape[0] - 1.0)

    c0 = float(np.nanmedian(cutout))
    amp0 = float(np.nanmax(cutout) - c0)
    if not np.isfinite(amp0) or amp0 <= 0.0:
        amp0 = float(np.nanmax(np.abs(cutout)))
    if not np.isfinite(amp0) or amp0 == 0.0:
        amp0 = 1.0

    x_init, y_init, theta_init = gaussian_init_from_xy_correlation(
        x_stddev_init,
        y_stddev_init,
        rho_xy_init,
    )
    bound_factor = (
        float(size_bounds_factor)
        if np.isfinite(size_bounds_factor) and size_bounds_factor > 1.0
        else 3.0
    )

    model = models.Const2D(amplitude=c0) + models.Gaussian2D(
        amplitude=amp0,
        x_mean=local_x_guess,
        y_mean=local_y_guess,
        x_stddev=x_init,
        y_stddev=y_init,
        theta=theta_init,
    )

    model[1].x_mean.bounds = (0.0, cutout.shape[1] - 1.0)
    model[1].y_mean.bounds = (0.0, cutout.shape[0] - 1.0)
    if np.isfinite(center_max_shift_pix) and center_max_shift_pix > 0.0:
        shift = float(center_max_shift_pix)
        x_center_low = max(0.0, local_x_guess - shift)
        x_center_high = min(cutout.shape[1] - 1.0, local_x_guess + shift)
        y_center_low = max(0.0, local_y_guess - shift)
        y_center_high = min(cutout.shape[0] - 1.0, local_y_guess + shift)
        if x_center_low < x_center_high:
            model[1].x_mean.bounds = (x_center_low, x_center_high)
        if y_center_low < y_center_high:
            model[1].y_mean.bounds = (y_center_low, y_center_high)

    theta_range = np.deg2rad(theta_max_shift_deg) if np.isfinite(theta_max_shift_deg) else np.pi
    if theta_range > 0.0 and theta_range < np.pi:
        theta_low = theta_init - theta_range
        theta_high = theta_init + theta_range
        if theta_low >= -np.pi and theta_high <= np.pi:
            model[1].theta.bounds = (theta_low, theta_high)

    if fix_shape:
        model[1].x_stddev.fixed = True
        model[1].y_stddev.fixed = True
        model[1].theta.fixed = True

    x_lower = max(0.6, x_init / bound_factor)
    x_upper = min(max(1.0, cutout.shape[1]), x_init * bound_factor)
    if x_lower >= x_upper:
        x_lower, x_upper = 0.6, max(1.0, cutout.shape[1])

    y_lower = max(0.6, y_init / bound_factor)
    y_upper = min(max(1.0, cutout.shape[0]), y_init * bound_factor)
    if y_lower >= y_upper:
        y_lower, y_upper = 0.6, max(1.0, cutout.shape[0])

    model[1].x_stddev.bounds = (x_lower, x_upper)
    model[1].y_stddev.bounds = (y_lower, y_upper)

    fit_model = fitting.LevMarLSQFitter()(model, xx, yy, cutout)

    x_fit = float(fit_model[1].x_mean.value) + float(x0)
    y_fit = float(fit_model[1].y_mean.value) + float(y0)
    fit_info = {
        "x_stddev_pix": float(fit_model[1].x_stddev.value),
        "y_stddev_pix": float(fit_model[1].y_stddev.value),
        "theta_rad": float(fit_model[1].theta.value),
    }
    return x_fit, y_fit, fit_info


def pixel_to_world_xy(header: fits.Header, x_pix: float, y_pix: float) -> tuple[float, float]:
    """Convert zero-based pixel coordinates to world coordinates from linear FITS WCS terms."""
    crval1 = float(header.get("CRVAL1", 0.0))
    crpix1 = float(header.get("CRPIX1", 1.0))
    cdelt1 = float(header.get("CDELT1", 1.0))

    crval2 = float(header.get("CRVAL2", 0.0))
    crpix2 = float(header.get("CRPIX2", 1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))

    x_world = crval1 + (((x_pix + 1.0) - crpix1) * cdelt1)
    y_world = crval2 + (((y_pix + 1.0) - crpix2) * cdelt2)
    return float(x_world), float(y_world)


def world_xy_to_pixel(header: fits.Header, x_world: float, y_world: float) -> tuple[float, float]:
    """Convert world coordinates (e.g. Galactic) to zero-based pixel coordinates."""
    crval1 = float(header.get("CRVAL1", 0.0))
    crpix1 = float(header.get("CRPIX1", 1.0))
    cdelt1 = float(header.get("CDELT1", 1.0))

    crval2 = float(header.get("CRVAL2", 0.0))
    crpix2 = float(header.get("CRPIX2", 1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))

    x_pix = ((x_world - crval1) / cdelt1) + crpix1 - 1.0
    y_pix = ((y_world - crval2) / cdelt2) + crpix2 - 1.0
    return float(x_pix), float(y_pix)


def read_velocity_config(config_file: Path) -> dict[float, dict[str, object]]:
    """Read velocity configuration from a text file.

    Format (tab-separated or whitespace):
        velocity_kms [lon_deg lat_deg [weight [x_sigma_pix [y_sigma_pix [rho_xy
                     [center_max_shift_pix [theta_max_shift_deg [fix_shape]]]]]]]]

    Lines starting with # or empty lines are ignored.

    Optional trailing fields:
        center_max_shift_pix: per-velocity max centroid shift from initial guess.
        theta_max_shift_deg: per-velocity max orientation shift from initialized theta.
        fix_shape: per-velocity boolean (1/0, true/false, yes/no, y/n).
    """
    config: dict[float, dict[str, object]] = {}
    with open(config_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 1:
                continue
            try:
                vel = float(cols[0])
                entry = {"vel": vel}
                if len(cols) >= 3:
                    entry["lon"] = float(cols[1])
                    entry["lat"] = float(cols[2])
                if len(cols) >= 4:
                    entry["weight"] = float(cols[3])
                if len(cols) >= 5:
                    entry["x_sigma_pix"] = float(cols[4])
                    if len(cols) >= 6:
                        entry["y_sigma_pix"] = float(cols[5])
                    else:
                        entry["y_sigma_pix"] = float(cols[4])
                if len(cols) >= 7:
                    entry["rho_xy"] = float(cols[6])
                if len(cols) >= 8:
                    entry["center_max_shift_pix"] = float(cols[7])
                if len(cols) >= 9:
                    entry["theta_max_shift_deg"] = float(cols[8])
                if len(cols) >= 10:
                    fix_shape_token = cols[9].strip().lower()
                    if fix_shape_token in {"1", "true", "yes", "y"}:
                        entry["fix_shape"] = True
                    elif fix_shape_token in {"0", "false", "no", "n"}:
                        entry["fix_shape"] = False
                config[vel] = entry
            except (ValueError, IndexError):
                continue
    return config


def robust_median_and_scatter(values: list[float]) -> tuple[float, float]:
    """Compute robust center and scatter using median and MAD-to-sigma scaling."""

    arr = np.array(values, dtype=float)
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    sigma = 1.4826 * mad
    return med, sigma


def weighted_mean_and_scatter(values: list[float], weights: list[float]) -> tuple[float, float]:
    """Compute weighted mean and weighted RMS scatter for finite positive weights."""

    arr = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)

    valid = np.isfinite(arr) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        raise ValueError("No valid weighted values available")

    arr = arr[valid]
    w = w[valid]

    mean = float(np.average(arr, weights=w))
    variance = float(np.average((arr - mean) ** 2, weights=w))
    sigma = float(np.sqrt(max(variance, 0.0)))
    return mean, sigma


def spatial_plot_metadata(header: fits.Header, frame: np.ndarray) -> tuple[list[float], str, str]:
    """Extract spatial extent and axis labels from FITS header."""
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


def save_gaussian_fit_comparison(
    ref_frame: np.ndarray,
    tgt_frame: np.ndarray,
    x_ref: float,
    y_ref: float,
    x_tgt: float,
    y_tgt: float,
    ref_fit_info: dict[str, float],
    tgt_fit_info: dict[str, float],
    header: fits.Header,
    target_vel: float,
    ref_cube_name: str,
    tgt_cube_name: str,
    run_dir: Path,
) -> Path:
    """Save comparison image showing reference and target frames with gaussian fit centers marked."""
    extent, xlabel, ylabel = spatial_plot_metadata(header, ref_frame)

    cdelt1 = float(header.get("CDELT1", 1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))
    x_ref_w, y_ref_w = pixel_to_world_xy(header, x_ref, y_ref)
    x_tgt_w, y_tgt_w = pixel_to_world_xy(header, x_tgt, y_tgt)

    # True 2-sigma contour diameter: 2 * (2*sigma) = 4*sigma.
    ref_w = 4.0 * abs(ref_fit_info["x_stddev_pix"] * cdelt1)
    ref_h = 4.0 * abs(ref_fit_info["y_stddev_pix"] * cdelt2)
    tgt_w = 4.0 * abs(tgt_fit_info["x_stddev_pix"] * cdelt1)
    tgt_h = 4.0 * abs(tgt_fit_info["y_stddev_pix"] * cdelt2)
    ref_theta_deg = np.degrees(ref_fit_info["theta_rad"])
    tgt_theta_deg = np.degrees(tgt_fit_info["theta_rad"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    
    # Determine vmin/vmax from both frames combined for consistent scaling
    all_data = np.concatenate([ref_frame.ravel(), tgt_frame.ravel()])
    vmin = float(np.nanpercentile(all_data, 2))
    vmax = float(np.nanpercentile(all_data, 98))
    
    # Reference frame
    im1 = axes[0].imshow(ref_frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].plot(x_ref_w, y_ref_w, "r+", markersize=15, markeredgewidth=2, label="Fit center")
    axes[0].add_patch(
        Ellipse(
            (x_ref_w, y_ref_w),
            width=ref_w,
            height=ref_h,
            angle=ref_theta_deg,
            edgecolor="white",
            facecolor="none",
            linewidth=1.5,
            linestyle="--",
            label="2-sigma Gaussian",
        )
    )
    axes[0].set_title(f"Reference: {ref_cube_name}")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="upper right")
    fig.colorbar(im1, ax=axes[0], label="Intensity")
    
    # Target frame
    im2 = axes[1].imshow(tgt_frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
    axes[1].plot(x_tgt_w, y_tgt_w, "r+", markersize=15, markeredgewidth=2, label="Fit center")
    axes[1].add_patch(
        Ellipse(
            (x_tgt_w, y_tgt_w),
            width=tgt_w,
            height=tgt_h,
            angle=tgt_theta_deg,
            edgecolor="white",
            facecolor="none",
            linewidth=1.5,
            linestyle="--",
            label="2-sigma Gaussian",
        )
    )
    axes[1].set_title(f"Target: {tgt_cube_name}")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend(loc="upper right")
    fig.colorbar(im2, ax=axes[1], label="Intensity")
    
    safe_vel = f"{target_vel:+06.1f}".replace("+", "p").replace("-", "m")
    fig.suptitle(f"Gaussian Fit Comparison: v={target_vel:.1f} km/s | offset=({x_tgt-x_ref:+.2f}, {y_tgt-y_ref:+.2f}) pix", fontsize=12)
    
    compare_dir = run_dir / "Compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    out_path = compare_dir / f"GaussianFit_v{safe_vel}kms.png"
    
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    
    return out_path


def save_velocity_debug_table(debug_rows: list[dict[str, object]], run_dir: Path) -> Path:
    """Write per-velocity diagnostic rows to CSV in the run Compare directory."""

    compare_dir = run_dir / "Compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    out_path = compare_dir / "GaussianFit_velocity_debug.csv"

    fieldnames = [
        "status",
        "requested_velocity_kms",
        "ref_channel_velocity_kms",
        "tgt_channel_velocity_kms",
        "weight",
        "ref_peak_x",
        "ref_peak_y",
        "ref_fit_x",
        "ref_fit_y",
        "ref_fit_sigx_pix",
        "ref_fit_sigy_pix",
        "tgt_fit_x",
        "tgt_fit_y",
        "tgt_fit_sigx_pix",
        "tgt_fit_sigy_pix",
        "dx_pix",
        "dy_pix",
        "az_deg",
        "el_deg",
        "ref_cube",
        "tgt_cube",
        "message",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in debug_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    return out_path


def update_offsets_file(offsets_file: Path, pix_label: str, az_deg: float, el_deg: float) -> None:
    """Insert or update one AS_MEASURED row for a mixer in offsets.txt."""

    lines = offsets_file.read_text(encoding="utf-8").splitlines()
    new_line = f"{pix_label}\t{az_deg:.6f}\t{el_deg:.6f}\tAS_MEASURED"

    updated = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or stripped.startswith("PIX"):
            continue
        cols = stripped.split()
        if len(cols) >= 4 and cols[0] == pix_label and cols[3].upper() == "AS_MEASURED":
            lines[i] = new_line
            updated = True
            break

    if not updated:
        while lines and not lines[-1].strip():
            lines.pop()
        lines.append("")
        lines.append(new_line)

    offsets_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser for offset measurement."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure mixer offsets with 2D Gaussian centroid fitting and write AS_MEASURED values into offsets.txt"
        )
    )
    parser.add_argument("--source", "-s", default="G337", help="Source under Data/level2")
    parser.add_argument(
        "--run",
        "-n",
        default="latest",
        help='Run number (for "run N") or "latest". Ignored if --run-dir is provided.',
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit run directory path. Example: Data/level2/G337/run 2",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to velocity config file with format: velocity_kms [lon_deg lat_deg [weight [x_sigma_pix [y_sigma_pix [rho_xy [center_max_shift_pix [theta_max_shift_deg [fix_shape]]]]]]]].",
    )
    parser.add_argument("--pattern", default="*.fits", help="Glob pattern for FITS cubes inside run directory.")
    parser.add_argument(
        "-b",
        "--line",
        default="CII",
        choices=["CII", "NII"],
        help="Spectral line to use (CII->B2, NII->B1).",
    )
    parser.add_argument(
        "-f",
        "--fiducial-mixer",
        default="8",
        help="Fiducial mixer number in selected line/band. Default is 8.",
    )
    parser.add_argument(
        "-t",
        "--target-mixer",
        default="5",
        help="Target mixer number in selected line/band. Default is 5.",
    )
    parser.add_argument(
        "-w",
        "--fit-window",
        type=int,
        default=12,
        help="Half-size of fit window in pixels around peak.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["uniform", "peak"],
        default="uniform",
        help="How to score each velocity when config weight is not provided. 'peak' uses the reference-frame peak signal above background.",
    )
    parser.add_argument(
        "--gauss-size-pix",
        type=float,
        default=2.5,
        help="Approximate Gaussian sigma (pixels) used as initial guess when no per-velocity size is set.",
    )
    parser.add_argument(
        "--gauss-size-bounds-factor",
        type=float,
        default=3.0,
        help="Allowed fit range around initial sigma: [sigma/factor, sigma*factor]. Must be > 1.",
    )
    parser.add_argument(
        "--gauss-rho-init",
        type=float,
        default=0.0,
        help="Initial x-y correlation factor (rho in [-1,1]) for Gaussian shape when no per-velocity value is set.",
    )
    parser.add_argument(
        "--center-max-shift-pix",
        type=float,
        default=8.0,
        help="Maximum allowed centroid shift (pixels) from the initial guess during fit.",
    )
    parser.add_argument(
        "--theta-max-shift-deg",
        type=float,
        default=45.0,
        help="Maximum allowed orientation shift (degrees) from initialized theta. Set >=180 to effectively disable.",
    )
    parser.add_argument(
        "--fix-shape",
        action="store_true",
        help="Keep x/y sigma and theta fixed to priors; only centroid and amplitude/background are fitted.",
    )
    parser.add_argument(
        "-o",
        "--offsets-file",
        default="src/GUSTO_Pipeline/calib/offsets.txt",
        help="Path to offsets.txt. Default: src/GUSTO_Pipeline/calib/offsets.txt",
    )
    return parser


def resolve_run_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    """Resolve run directory from explicit path or source/run selectors."""

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = find_run_dir(repo_root / "Data" / "level2" / args.source, str(args.run))

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return run_dir


def resolve_offsets_file(args: argparse.Namespace, repo_root: Path) -> Path:
    """Resolve output offsets.txt path from CLI or repository default."""

    if args.offsets_file:
        return Path(args.offsets_file).resolve()
    return (repo_root / "src" / "GUSTO_Pipeline" / "calib" / "offsets.txt").resolve()


def resolve_config_file_path(config_arg: str, repo_root: Path) -> Path:
    """Resolve config file path across absolute, repo-relative, cwd, and calib fallback."""

    config_file_path = Path(config_arg)
    if config_file_path.is_absolute():
        return config_file_path.resolve()
    repo_relative = repo_root / config_arg
    if repo_relative.exists():
        return repo_relative.resolve()
    if config_file_path.exists():
        return config_file_path.resolve()

    calib_path = repo_root / "src" / "GUSTO_Pipeline" / "calib" / config_arg
    if calib_path.exists():
        return calib_path.resolve()

    raise FileNotFoundError(
        f"Config file not found. Tried:\n"
        f"  {config_file_path.resolve()}\n"
        f"  {repo_relative}\n"
        f"  {calib_path}"
    )


def load_velocity_config_from_args(args: argparse.Namespace, repo_root: Path) -> VelocityConfig:
    """Load velocity list and optional per-velocity priors from config file or CLI."""

    config_file = resolve_config_file_path(args.config_file, repo_root)
    config_dict = read_velocity_config(config_file)
    velocities = sorted(config_dict.keys())
    if not velocities:
        raise ValueError(f"No valid velocities found in config file: {config_file}")

    velocity_hints: dict[float, tuple[float, float]] = {}
    velocity_weights_config: dict[float, float] = {}
    velocity_sizes_config: dict[float, tuple[float, float]] = {}
    velocity_rho_config: dict[float, float] = {}
    velocity_center_shift_config: dict[float, float] = {}
    velocity_theta_shift_config: dict[float, float] = {}
    velocity_fix_shape_config: dict[float, bool] = {}

    for vel, entry in config_dict.items():
        if "lon" in entry and "lat" in entry:
            velocity_hints[vel] = (float(entry["lon"]), float(entry["lat"]))
        if "weight" in entry:
            velocity_weights_config[vel] = float(entry["weight"])
        if "x_sigma_pix" in entry and "y_sigma_pix" in entry:
            velocity_sizes_config[vel] = (
                float(entry["x_sigma_pix"]),
                float(entry["y_sigma_pix"]),
            )
        if "rho_xy" in entry:
            velocity_rho_config[vel] = float(entry["rho_xy"])
        if "center_max_shift_pix" in entry:
            velocity_center_shift_config[vel] = float(entry["center_max_shift_pix"])
        if "theta_max_shift_deg" in entry:
            velocity_theta_shift_config[vel] = float(entry["theta_max_shift_deg"])
        if "fix_shape" in entry:
            velocity_fix_shape_config[vel] = bool(entry["fix_shape"])

    print(f"Loaded {len(velocities)} velocities from config file: {config_file}")
    print(f"  Velocities: {velocities}")
    if velocity_hints:
        print(f"  Per-velocity Galactic center hints available for {len(velocity_hints)} velocity(s)")
    if velocity_sizes_config:
        print(f"  Per-velocity Gaussian size priors available for {len(velocity_sizes_config)} velocity(s)")
    if velocity_rho_config:
        print(f"  Per-velocity Gaussian correlation priors available for {len(velocity_rho_config)} velocity(s)")
    if velocity_center_shift_config:
        print(f"  Per-velocity centroid-shift limits available for {len(velocity_center_shift_config)} velocity(s)")
    if velocity_theta_shift_config:
        print(f"  Per-velocity theta-shift limits available for {len(velocity_theta_shift_config)} velocity(s)")
    if velocity_fix_shape_config:
        print(f"  Per-velocity fix-shape flags available for {len(velocity_fix_shape_config)} velocity(s)")

    return VelocityConfig(
        velocities=velocities,
        velocity_hints=velocity_hints,
        velocity_weights_config=velocity_weights_config,
        velocity_sizes_config=velocity_sizes_config,
        velocity_rho_config=velocity_rho_config,
        velocity_center_shift_config=velocity_center_shift_config,
        velocity_theta_shift_config=velocity_theta_shift_config,
        velocity_fix_shape_config=velocity_fix_shape_config,
    )


def resolve_base_velocity_weight(index: int, vel: float, velocity_config: VelocityConfig) -> float:
    """Return explicit per-velocity weight from config/CLI, or default 1.0."""
    _ = index
    if vel in velocity_config.velocity_weights_config:
        return velocity_config.velocity_weights_config[vel]
    return 1.0


def choose_peak_guess(vel: float, ref_frame: np.ndarray, ref_header: fits.Header, velocity_config: VelocityConfig) -> tuple[float, float]:
    """Pick initial peak guess globally, or near per-velocity Galactic hint when available."""

    x_peak, y_peak = safe_peak_xy(ref_frame)
    if vel not in velocity_config.velocity_hints:
        return x_peak, y_peak

    lon_hint, lat_hint = velocity_config.velocity_hints[vel]
    try:
        x_hint, y_hint = world_xy_to_pixel(ref_header, lon_hint, lat_hint)
        if 0 <= x_hint < ref_frame.shape[1] and 0 <= y_hint < ref_frame.shape[0]:
            x_peak, y_peak = find_peak_in_region(ref_frame, x_hint, y_hint, search_radius=30)
            print(f"  Found peak near Galactic hint for v={vel:.1f}: x={x_peak:.1f}, y={y_peak:.1f}")
        else:
            print(f"  Galactic hint for v={vel:.1f} outside frame bounds, using global auto-peak")
    except Exception as exc:
        print(f"  Could not search around Galactic hint for v={vel:.1f}: {exc}, using global auto-peak")
    return x_peak, y_peak


def gaussian_priors_for_velocity(args: argparse.Namespace, vel: float, velocity_config: VelocityConfig) -> tuple[float, float, float]:
    """Resolve Gaussian size and correlation priors for one velocity channel."""

    if vel in velocity_config.velocity_sizes_config:
        x_sigma_init, y_sigma_init = velocity_config.velocity_sizes_config[vel]
    else:
        x_sigma_init = float(args.gauss_size_pix)
        y_sigma_init = float(args.gauss_size_pix)

    if vel in velocity_config.velocity_rho_config:
        rho_init = float(np.clip(velocity_config.velocity_rho_config[vel], -0.99, 0.99))
    else:
        rho_init = float(np.clip(args.gauss_rho_init, -0.99, 0.99))
    return x_sigma_init, y_sigma_init, rho_init


def compute_velocity_weight(
    args: argparse.Namespace,
    index: int,
    vel: float,
    ref_frame: np.ndarray,
    velocity_config: VelocityConfig,
) -> float:
    """Compute final weight, optionally using peak SNR mode when no explicit weight exists."""

    base_weight = resolve_base_velocity_weight(index, vel, velocity_config)
    if vel in velocity_config.velocity_weights_config:
        return base_weight
    if args.weight_mode == "peak":
        background = float(np.nanmedian(ref_frame))
        signal = float(np.nanmax(ref_frame) - background)
        noise = float(np.nanmedian(np.abs(ref_frame - background)))
        return signal / noise if np.isfinite(noise) and noise > 0.0 else max(signal, 1.0)
    return base_weight


def fit_controls_for_velocity(
    args: argparse.Namespace,
    vel: float,
    velocity_config: VelocityConfig,
) -> tuple[float, float, bool]:
    """Resolve fit-control options (center shift, theta shift, fix-shape) for one velocity."""
    if vel in velocity_config.velocity_center_shift_config:
        center_max_shift = float(velocity_config.velocity_center_shift_config[vel])
    else:
        center_max_shift = float(args.center_max_shift_pix)

    if vel in velocity_config.velocity_theta_shift_config:
        theta_max_shift = float(velocity_config.velocity_theta_shift_config[vel])
    else:
        theta_max_shift = float(args.theta_max_shift_deg)

    if vel in velocity_config.velocity_fix_shape_config:
        fix_shape = bool(velocity_config.velocity_fix_shape_config[vel])
    else:
        fix_shape = bool(args.fix_shape)

    return center_max_shift, theta_max_shift, fix_shape


def process_velocity(
    args: argparse.Namespace,
    index: int,
    vel: float,
    ref_cube_data: LoadedCube,
    tgt_cube_data: LoadedCube,
    run_dir: Path,
    velocity_config: VelocityConfig,
    accumulator: OffsetAccumulator,
) -> None:
    """Process one velocity: slice, fit, compute offsets, weight, plots, and debug row."""

    try:
        v_ref, ref_frame = extract_slice_from_loaded_cube(ref_cube_data, vel)
        v_tgt, tgt_frame = extract_slice_from_loaded_cube(tgt_cube_data, vel)

        if ref_frame.shape != tgt_frame.shape:
            message = f"shape mismatch {ref_frame.shape} vs {tgt_frame.shape}"
            accumulator.debug_rows.append(
                {
                    "status": "skipped",
                    "requested_velocity_kms": vel,
                    "ref_channel_velocity_kms": v_ref,
                    "tgt_channel_velocity_kms": v_tgt,
                    "weight": resolve_base_velocity_weight(index, vel, velocity_config),
                    "ref_cube": ref_cube_data.path.name,
                    "tgt_cube": tgt_cube_data.path.name,
                    "message": message,
                }
            )
            print(f"Skipping v={vel:.1f}: {message}")
            return

        x_peak, y_peak = choose_peak_guess(vel, ref_frame, ref_cube_data.header, velocity_config)
        x_sigma_init, y_sigma_init, rho_init = gaussian_priors_for_velocity(args, vel, velocity_config)
        center_max_shift, theta_max_shift, fix_shape = fit_controls_for_velocity(args, vel, velocity_config)

        x_ref, y_ref, ref_fit_info = fit_gaussian_centroid(
            ref_frame,
            x_peak,
            y_peak,
            int(args.fit_window),
            x_stddev_init=x_sigma_init,
            y_stddev_init=y_sigma_init,
            rho_xy_init=rho_init,
            size_bounds_factor=float(args.gauss_size_bounds_factor),
            center_max_shift_pix=center_max_shift,
            theta_max_shift_deg=theta_max_shift,
            fix_shape=fix_shape,
        )
        x_tgt, y_tgt, tgt_fit_info = fit_gaussian_centroid(
            tgt_frame,
            x_ref,
            y_ref,
            int(args.fit_window),
            x_stddev_init=x_sigma_init,
            y_stddev_init=y_sigma_init,
            rho_xy_init=rho_init,
            size_bounds_factor=float(args.gauss_size_bounds_factor),
            center_max_shift_pix=center_max_shift,
            theta_max_shift_deg=theta_max_shift,
            fix_shape=fix_shape,
        )

        dx_pix = x_tgt - x_ref
        dy_pix = y_tgt - y_ref
        az_deg = dx_pix * float(ref_cube_data.header.get("CDELT1", 0.0))
        el_deg = dy_pix * float(ref_cube_data.header.get("CDELT2", 0.0))

        accumulator.dx_pix_values.append(dx_pix)
        accumulator.dy_pix_values.append(dy_pix)
        accumulator.az_deg_values.append(az_deg)
        accumulator.el_deg_values.append(el_deg)

        weight = compute_velocity_weight(args, index, vel, ref_frame, velocity_config)
        accumulator.velocity_weights.append(weight)

        try:
            save_gaussian_fit_comparison(
                ref_frame,
                tgt_frame,
                x_ref,
                y_ref,
                x_tgt,
                y_tgt,
                ref_fit_info,
                tgt_fit_info,
                ref_cube_data.header,
                vel,
                ref_cube_data.path.name,
                tgt_cube_data.path.name,
                run_dir,
            )
        except Exception as exc_img:
            print(f"Warning: Could not save comparison image for v={vel:.1f}: {exc_img}")

        accumulator.debug_rows.append(
            {
                "status": "ok",
                "requested_velocity_kms": vel,
                "ref_channel_velocity_kms": v_ref,
                "tgt_channel_velocity_kms": v_tgt,
                "weight": weight,
                "ref_peak_x": x_peak,
                "ref_peak_y": y_peak,
                "ref_fit_x": x_ref,
                "ref_fit_y": y_ref,
                "ref_fit_sigx_pix": ref_fit_info["x_stddev_pix"],
                "ref_fit_sigy_pix": ref_fit_info["y_stddev_pix"],
                "tgt_fit_x": x_tgt,
                "tgt_fit_y": y_tgt,
                "tgt_fit_sigx_pix": tgt_fit_info["x_stddev_pix"],
                "tgt_fit_sigy_pix": tgt_fit_info["y_stddev_pix"],
                "dx_pix": dx_pix,
                "dy_pix": dy_pix,
                "az_deg": az_deg,
                "el_deg": el_deg,
                "ref_cube": ref_cube_data.path.name,
                "tgt_cube": tgt_cube_data.path.name,
                "message": "",
            }
        )

        print(
            f"v_target={vel:.1f} km/s v_ref={v_ref:.2f} v_tgt={v_tgt:.2f} "
            f"weight={weight:.3f} dx={dx_pix:+.3f} pix dy={dy_pix:+.3f} pix "
            f"dAZ={az_deg:+.6f} deg dEL={el_deg:+.6f} deg"
        )
    except Exception as exc:
        accumulator.debug_rows.append(
            {
                "status": "error",
                "requested_velocity_kms": vel,
                "ref_cube": ref_cube_data.path.name,
                "tgt_cube": tgt_cube_data.path.name,
                "message": str(exc),
                "weight": resolve_base_velocity_weight(index, vel, velocity_config),
            }
        )
        print(f"Skipping v={vel:.1f}: {exc}")


def summarize_offsets(args: argparse.Namespace, velocity_config: VelocityConfig, accumulator: OffsetAccumulator) -> tuple[float, float, float, float, float, float, float, float, str]:
    """Aggregate per-velocity offsets using weighted mean or robust median mode."""

    if not accumulator.az_deg_values:
        raise RuntimeError("No valid velocities produced Gaussian-fit offsets")

    use_weighted = args.weight_mode == "peak" or bool(velocity_config.velocity_weights_config)
    if use_weighted:
        dx_med, dx_sig = weighted_mean_and_scatter(accumulator.dx_pix_values, accumulator.velocity_weights)
        dy_med, dy_sig = weighted_mean_and_scatter(accumulator.dy_pix_values, accumulator.velocity_weights)
        az_med, az_sig = weighted_mean_and_scatter(accumulator.az_deg_values, accumulator.velocity_weights)
        el_med, el_sig = weighted_mean_and_scatter(accumulator.el_deg_values, accumulator.velocity_weights)
        summary_label = "weighted mean"
    else:
        dx_med, dx_sig = robust_median_and_scatter(accumulator.dx_pix_values)
        dy_med, dy_sig = robust_median_and_scatter(accumulator.dy_pix_values)
        az_med, az_sig = robust_median_and_scatter(accumulator.az_deg_values)
        el_med, el_sig = robust_median_and_scatter(accumulator.el_deg_values)
        summary_label = "robust median"
    return dx_med, dx_sig, dy_med, dy_sig, az_med, az_sig, el_med, el_sig, summary_label


def main() -> None:
    """CLI entry point for Gaussian-fit mixer offset measurement."""

    parser = build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    run_dir = resolve_run_dir(args, repo_root)
    cubes = sorted(run_dir.glob(args.pattern))
    if not cubes:
        raise FileNotFoundError(f"No FITS files found in {run_dir} matching {args.pattern}")

    offsets_file = resolve_offsets_file(args, repo_root)
    velocity_config = load_velocity_config_from_args(args, repo_root)

    ref_cube = find_cube_for_line_and_mixer(cubes, args.line, str(args.fiducial_mixer))
    tgt_cube = find_cube_for_line_and_mixer(cubes, args.line, str(args.target_mixer))
    ref_cube_data = load_cube_data(ref_cube)
    tgt_cube_data = load_cube_data(tgt_cube)

    accumulator = new_offset_accumulator()

    print(f"Run directory: {run_dir}")
    print(f"Reference cube: {ref_cube.name}")
    print(f"Target cube: {tgt_cube.name}")

    for index, vel in enumerate(velocity_config.velocities):
        process_velocity(
            args=args,
            index=index,
            vel=vel,
            ref_cube_data=ref_cube_data,
            tgt_cube_data=tgt_cube_data,
            run_dir=run_dir,
            velocity_config=velocity_config,
            accumulator=accumulator,
        )

    dx_med, dx_sig, dy_med, dy_sig, az_med, az_sig, el_med, el_sig, summary_label = summarize_offsets(
        args,
        velocity_config,
        accumulator,
    )

    band = "B2" if args.line == "CII" else "B1"
    pix_label = f"{band}M{args.target_mixer}"

    debug_table_file = save_velocity_debug_table(accumulator.debug_rows, run_dir)
    update_offsets_file(offsets_file, pix_label, az_med, el_med)

    print("\nGaussian-fit offset summary")
    print(f"Combination method: {summary_label}")
    print(f"Line={args.line} Fiducial={band}M{args.fiducial_mixer} Target={pix_label}")
    print(f"dx={dx_med:+.3f} +/- {dx_sig:.3f} pix")
    print(f"dy={dy_med:+.3f} +/- {dy_sig:.3f} pix")
    print(f"AZ={az_med:+.6f} +/- {az_sig:.6f} deg")
    print(f"EL={el_med:+.6f} +/- {el_sig:.6f} deg")
    print(f"Saved velocity debug table: {debug_table_file}")
    print(f"Updated offsets file: {offsets_file}")


if __name__ == "__main__":
    main()