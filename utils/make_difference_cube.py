import argparse
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy.io import fits


def shift2d_no_wrap(image: np.ndarray, shift_x: int, shift_y: int, fill_value: float = np.nan) -> np.ndarray:
    """Shift a 2D image without wraparound."""
    ny, nx = image.shape
    out = np.full((ny, nx), fill_value, dtype=float)

    src_x0 = max(0, -shift_x)
    src_x1 = min(nx, nx - shift_x)
    src_y0 = max(0, -shift_y)
    src_y1 = min(ny, ny - shift_y)

    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

    return out


def shift3d_no_wrap(cube: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """Shift every channel in a cube by the same integer offset."""
    shifted = np.empty_like(cube, dtype=float)
    for chan in range(cube.shape[0]):
        shifted[chan] = shift2d_no_wrap(cube[chan], shift_x, shift_y)
    return shifted


def build_difference_cube(reference_cube: np.ndarray, target_cube: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """Return the residual cube after shifting target to match reference."""
    shifted_target_cube = shift3d_no_wrap(target_cube, shift_x, shift_y)
    return shifted_target_cube - reference_cube


def write_difference_cube(
    reference_cube: np.ndarray,
    target_cube: np.ndarray,
    shift_x: int,
    shift_y: int,
    output_path: Path,
    header: fits.Header | None = None,
) -> Path:
    """Build and write a residual cube to a FITS file."""
    residual_cube = build_difference_cube(reference_cube, target_cube, shift_x, shift_y)
    fits.PrimaryHDU(residual_cube, header=header).writeto(output_path, overwrite=True)
    return output_path


def load_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Load a FITS cube from disk."""
    with fits.open(path) as hdul:
        primary = hdul[0]
        data = np.squeeze(cast(Any, primary).data)
        header = cast(fits.Header, cast(Any, primary).header)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube in {path}, got shape={data.shape}")
    return np.array(data, dtype=float), header


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

RUN_PATTERN = re.compile(r"^run\s+(\d+)$", re.IGNORECASE)


def find_latest_run_dir(source_dir: Path) -> Path:
    """Return the highest-numbered ``run N`` directory under *source_dir*."""
    runs: list[tuple[int, Path]] = []
    for child in source_dir.iterdir():
        if child.is_dir():
            m = RUN_PATTERN.match(child.name)
            if m:
                runs.append((int(m.group(1)), child))
    if not runs:
        raise FileNotFoundError(f"No run directories found in {source_dir}")
    return sorted(runs, key=lambda item: item[0])[-1][1]


DEFAULT_LINE_TARGETS: dict[str, tuple[int, list[int]]] = {
    "CII": (8, [5]),
    "NII": (3, [2, 6]),
}


def discover_pairs(
    run_dir: Path,
    line_targets: dict[str, tuple[int, list[int]]] | None = None,
) -> list[dict[str, object]]:
    """Find all reference → matched cube pairs in *run_dir*.

    Returns a list of dicts with keys: line, reference_mixer, target_mixer,
    reference_path, target_path, output_name.
    """
    if line_targets is None:
        line_targets = DEFAULT_LINE_TARGETS

    # Index cubes by (line, mixer) → path
    cubes: dict[tuple[str, int], Path] = {}
    for fpath in sorted(run_dir.glob("*.fits")):
        name = fpath.name
        parts = name.replace(".fits", "").split("_")
        if len(parts) < 3:
            continue
        line = parts[1].upper()
        if line not in ("CII", "NII"):
            continue
        try:
            mixer = int(parts[2])
        except ValueError:
            continue
        if mixer == 0:
            continue
        cubes[(line, mixer)] = fpath

    pairs: list[dict[str, object]] = []
    for line, (ref_mx, tgt_mxs) in line_targets.items():
        ref_path = cubes.get((line, ref_mx))
        if ref_path is None:
            continue
        for tgt_mx in tgt_mxs:
            tgt_path = cubes.get((line, tgt_mx))
            if tgt_path is None:
                continue
            output_name = f"{run_dir.parent.name}_{line}_{tgt_mx}_minus_M{ref_mx}_manual.fits"
            pairs.append({
                "line": line,
                "reference_mixer": ref_mx,
                "target_mixer": tgt_mx,
                "reference_path": ref_path,
                "target_path": tgt_path,
                "output_name": output_name,
            })

    return pairs


def auto_main(source: str, data_root: str, shift_x: int, shift_y: int) -> None:
    """Discover the latest run directory for *source* and generate all
    difference cubes automatically."""
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = (repo_root / data_root / source).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    run_dir = find_latest_run_dir(source_dir)
    compare_dir = run_dir / "Compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(run_dir)
    if not pairs:
        print(f"No reference → matched cube pairs found in {run_dir}")
        return

    print(f"Run dir:  {run_dir}")
    print(f"Source:   {source}")
    print()

    for pair in pairs:
        ref_path = Path(str(pair["reference_path"]))
        tgt_path = Path(str(pair["target_path"]))
        output_path = compare_dir / str(pair["output_name"])

        ref_cube, header = load_cube(ref_path)
        tgt_cube, _ = load_cube(tgt_path)

        if ref_cube.shape != tgt_cube.shape:
            print(f"  [{pair['line']} M{pair['target_mixer']} vs M{pair['reference_mixer']}] "
                  f"SKIP: shape mismatch ({ref_cube.shape} vs {tgt_cube.shape})")
            continue

        write_difference_cube(
            reference_cube=ref_cube,
            target_cube=tgt_cube,
            shift_x=shift_x,
            shift_y=shift_y,
            output_path=output_path,
            header=header,
        )
        print(f"  [{pair['line']} M{pair['target_mixer']} vs M{pair['reference_mixer']}] "
              f"→ {output_path.name}")

    print(f"\nWrote {len(pairs)} difference cube(s) to {compare_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for manual difference-cube creation."""
    parser = argparse.ArgumentParser(
        description="Create difference cubes from two FITS cubes."
    )
    # Auto-discovery mode
    parser.add_argument(
        "--source",
        default=None,
        help="Source name (e.g. G337). When given, auto-discovers the latest "
             "run directory and generates all reference→matched difference cubes.",
    )
    parser.add_argument(
        "--data-root",
        default="Data/level2",
        help="Root data directory for Level-2 cubes (default: Data/level2)",
    )
    # Manual mode (positional, required only when --source is not given)
    parser.add_argument(
        "reference", nargs="?", default=None,
        help="Reference FITS cube (manual mode)",
    )
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Target FITS cube to shift and subtract (manual mode)",
    )
    parser.add_argument(
        "--shift-x", type=int, default=0,
        help="Pixel shift along x to apply to target (default: 0)",
    )
    parser.add_argument(
        "--shift-y", type=int, default=0,
        help="Pixel shift along y to apply to target (default: 0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output residual FITS path (manual mode)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ----- Auto-discovery mode -----
    if args.source:
        auto_main(
            source=args.source,
            data_root=args.data_root,
            shift_x=args.shift_x,
            shift_y=args.shift_y,
        )
        return

    # ----- Manual mode -----
    if args.reference is None or args.target is None:
        parser.error("Either --source or both positional arguments (reference target) are required")
    if args.output is None:
        parser.error("--output is required in manual mode")

    reference_path = Path(args.reference).resolve()
    target_path = Path(args.target).resolve()
    output_path = Path(args.output).resolve()

    reference_cube, header = load_cube(reference_path)
    target_cube, _ = load_cube(target_path)

    if reference_cube.shape != target_cube.shape:
        raise ValueError(
            f"Cube shape mismatch: reference={reference_cube.shape} target={target_cube.shape}"
        )

    write_difference_cube(
        reference_cube=reference_cube,
        target_cube=target_cube,
        shift_x=int(args.shift_x),
        shift_y=int(args.shift_y),
        output_path=output_path,
        header=header,
    )
    print(f"Wrote difference cube: {output_path}")


if __name__ == "__main__":
    main()
