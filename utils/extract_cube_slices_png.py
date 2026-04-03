import argparse
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib

# Force non-interactive backend so this works in terminal sessions.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_PATTERN = re.compile(r"^run\s+(\d+)$", re.IGNORECASE)
LINE_PATTERN = re.compile(r"(CII|NII)", re.IGNORECASE)
MIXER_PATTERN = re.compile(r"_(\d+)", re.IGNORECASE)
MIXER_ORDER = {"0": 0, "2": 2, "3": 3, "5": 5, "6": 6, "8": 8}


def find_run_dir(base_dir: Path, run: str) -> Path:
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
    crval = float(header.get("CRVAL3", 0.0))
    cdelt = float(header.get("CDELT3", 1.0))
    crpix = float(header.get("CRPIX3", 1.0))
    cunit = str(header.get("CUNIT3", "")).strip().lower()

    axis = crval + ((np.arange(nchan) + 1.0) - crpix) * cdelt

    if "km/s" in cunit or "kms" in cunit:
        return axis

    # Some cubes are labeled m/s even when values are already in km/s.
    if "m/s" in cunit or "ms-1" in cunit:
        if np.nanmax(np.abs(axis)) > 2000.0:
            return axis / 1000.0
        return axis

    # Unknown unit: infer from magnitude.
    if np.nanmax(np.abs(axis)) > 2000.0:
        return axis / 1000.0
    return axis


def line_from_filename(file_path: Path) -> str | None:
    match = LINE_PATTERN.search(file_path.name)
    if not match:
        return None
    return match.group(1).upper()

def mixer_from_filename(file_path: Path) -> str | None:
    match = MIXER_PATTERN.search(file_path.name)
    if not match:
        return None
    return match.group(1)


def scales_for_line(line: str) -> tuple[float, float]:
    if line == "CII":
        return -2.0, 6.0
    if line == "NII":
        return -1.0, 2.0
    raise ValueError(f"Unknown line type: {line}")


def velocity_label(value: float) -> str:
    return f"{value:+06.1f}".replace("+", "p").replace("-", "m")


def spatial_plot_metadata(header: fits.Header, frame: np.ndarray) -> tuple[list[float], str, str]:
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


def extract_slice_frame(cube_path: Path, target_vel: float) -> tuple[str, str, float, np.ndarray, float, float, fits.Header]:
    with fits.open(cube_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None or data.ndim < 3:
        raise ValueError(f"Expected 3D cube in {cube_path}")

    # Support both (chan, y, x) and (1, chan, y, x)
    if data.ndim == 4:
        data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError(f"Unsupported cube shape {data.shape} in {cube_path}")

    nchan = data.shape[0]
    vel_axis = spectral_axis_kms(header, nchan)

    line = line_from_filename(cube_path)
    if line is None:
        raise ValueError(f"Could not determine line (CII/NII) from filename {cube_path.name}")

    mixer = mixer_from_filename(cube_path) or "unknown"
    idx = int(np.argmin(np.abs(vel_axis - target_vel)))
    actual_vel = float(vel_axis[idx])
    frame = np.array(data[idx, :, :], dtype=float)

    return line, mixer, actual_vel, frame, *scales_for_line(line), header


def save_velocity_slice_png(cube_path: Path, velocities: list[float]) -> list[Path]:
    line = line_from_filename(cube_path)
    if line is None:
        raise ValueError(f"Could not determine line (CII/NII) from filename {cube_path.name}")

    vmin, vmax = scales_for_line(line)
    created = []

    for target_vel in velocities:
        _, _, actual_vel, frame, _, _, header = extract_slice_frame(cube_path, target_vel)
        extent, xlabel, ylabel = spatial_plot_metadata(header, frame)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        image = ax.imshow(frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
        cb = fig.colorbar(image, ax=ax, shrink=0.9)
        cb.set_label("Intensity")
        ax.set_title(
            f"{cube_path.name} | target v={target_vel:.1f} km/s | channel v={actual_vel:.2f} km/s"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        safe_vel = velocity_label(target_vel)
        out_name = f"{line_from_filename(cube_path)}_{mixer_from_filename(cube_path)}_slice_v{safe_vel}kms.png"
        slice_dir = cube_path.parent / f"slice_{safe_vel}kms"
        slice_dir.mkdir(parents=True, exist_ok=True)
        out_path = slice_dir / out_name
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        created.append(out_path)

    return created


def save_velocity_compare_png(cubes: list[Path], target_vel: float, run_dir: Path) -> list[Path]:
    entries: list[tuple[str, str, float, np.ndarray, float, float, fits.Header]] = []

    for cube_path in cubes:
        try:
            line, mixer, actual_vel, frame, vmin, vmax, header = extract_slice_frame(cube_path, target_vel)
        except Exception:
            continue
        entries.append((line, mixer, actual_vel, frame, vmin, vmax, header))

    if not entries:
        return []

    entries.sort(key=lambda item: (0 if item[0] == "CII" else 1, MIXER_ORDER.get(item[1], 999), item[1]))

    safe_vel = velocity_label(target_vel)
    slice_dir = run_dir / f"slice_{safe_vel}kms"
    slice_dir.mkdir(parents=True, exist_ok=True)

    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), dpi=120, squeeze=False)
    # Reserve space at right for row-specific colorbars.
    fig.subplots_adjust(right=0.90, wspace=0.25, hspace=0.30)

    cii_image = None
    nii_image = None
    for ax in axes.ravel():
        ax.axis("off")

    row_map = {"CII": 0, "NII": 1}
    row_offsets = {"CII": 0, "NII": 0}

    for line, mixer, actual_vel, frame, vmin, vmax, header in entries:
        row = row_map.get(line, 1)
        col = row_offsets[line]
        row_offsets[line] += 1
        if col >= ncols:
            continue

        ax = axes[row, col]
        ax.axis("on")
        extent, xlabel, ylabel = spatial_plot_metadata(header, frame)
        image = ax.imshow(frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
        if line == "CII":
            cii_image = image
        elif line == "NII":
            nii_image = image
        ax.set_title(f"{line} mixer {mixer}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if cii_image is not None:
        cax_cii = fig.add_axes([0.915, 0.56, 0.015, 0.32])
        fig.colorbar(cii_image, cax=cax_cii, label="CII Intensity")
    if nii_image is not None:
        cax_nii = fig.add_axes([0.915, 0.12, 0.015, 0.32])
        fig.colorbar(nii_image, cax=cax_nii, label="NII Intensity")
        
    fig.suptitle(f"Velocity slice comparison at target v={target_vel:.1f} km/s", fontsize=16)
    out_path = run_dir / "Compare" / f"Compare_slice_v{safe_vel}kms.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return [out_path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract velocity slices from GUSTOgridder cubes and save PNG previews in the run directory."
        )
    )
    parser.add_argument("--source", default="G337", help="Source under Data/level2")
    parser.add_argument(
        "--run",
        default="latest",
        help='Run number (for "run N") or "latest". Ignored if --run-dir is provided.',
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit run directory path. Example: Data/level2/G337/run 2",
    )
    parser.add_argument(
        "--velocities",
        nargs="+",
        type=float,
        required=True,
        help="Velocity values in km/s to extract, e.g. --velocities -120 -90 -60",
    )
    parser.add_argument(
        "--pattern",
        default="*.fits",
        help="Glob pattern for FITS cubes inside run directory.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        base_dir = repo_root / "Data" / "level2" / args.source
        run_dir = find_run_dir(base_dir, str(args.run))

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    cubes = sorted(run_dir.glob(args.pattern))
    if not cubes:
        raise FileNotFoundError(f"No FITS files found in {run_dir} matching {args.pattern}")

    print(f"Using run directory: {run_dir}")
    print(f"Velocities (km/s): {args.velocities}")

    total_png = 0
    compare_png = 0

    for cube in cubes:
        try:
            outputs = save_velocity_slice_png(cube, args.velocities)
            total_png += len(outputs)
            print(f"{cube.name}: wrote {len(outputs)} PNG slices")
        except Exception as exc:
            print(f"Skipping {cube.name}: {exc}")

    for target_vel in args.velocities:
        try:
            outputs = save_velocity_compare_png(cubes, target_vel, run_dir)
            compare_png += len(outputs)
            print(f"velocity {target_vel:.1f} km/s: wrote {len(outputs)} comparison PNGs")
        except Exception as exc:
            print(f"Skipping comparison for {target_vel:.1f} km/s: {exc}")

    print(f"Done. Wrote {total_png} slice PNGs and {compare_png} comparison PNGs in {run_dir}")


if __name__ == "__main__":
    main()
