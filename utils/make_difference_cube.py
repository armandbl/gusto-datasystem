import argparse
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


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for manual difference-cube creation."""
    parser = argparse.ArgumentParser(description="Create a difference cube from two FITS cubes")
    parser.add_argument("reference", help="Reference FITS cube")
    parser.add_argument("target", help="Target FITS cube to shift and subtract")
    parser.add_argument("--shift-x", type=int, required=True, help="Pixel shift along x to apply to target")
    parser.add_argument("--shift-y", type=int, required=True, help="Pixel shift along y to apply to target")
    parser.add_argument("--output", required=True, help="Output residual FITS path")
    return parser


def main() -> None:
    """Run the manual difference-cube CLI."""
    parser = build_parser()
    args = parser.parse_args()

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