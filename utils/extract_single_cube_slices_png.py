import argparse
from pathlib import Path

from extract_cube_slices_png import save_velocity_slice_png


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for single-cube slice extraction."""
    parser = argparse.ArgumentParser(
        description="Extract velocity slice PNGs from one FITS cube."
    )
    parser.add_argument(
        "cube",
        help="Path to the FITS cube to slice",
    )
    parser.add_argument(
        "--velocities",
        nargs="+",
        type=float,
        required=True,
        help="Velocity values in m/s to extract, e.g. --velocities -160 -120 -80",
    )
    return parser


def main() -> None:
    """Run the single-cube slice extraction CLI."""
    parser = build_parser()
    args = parser.parse_args()

    cube_path = Path(args.cube).resolve()
    if not cube_path.exists():
        raise FileNotFoundError(f"Cube does not exist: {cube_path}")

    outputs = save_velocity_slice_png(cube_path, args.velocities)
    for output_path in outputs:
        print(f"Wrote slice PNG: {output_path}")


if __name__ == "__main__":
    main()