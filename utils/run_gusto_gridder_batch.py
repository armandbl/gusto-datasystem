import argparse
import re
import subprocess
from pathlib import Path


def next_nth_run_dir(base_dir: Path) -> Path:
    """Create the next '<n>th run' directory under base_dir."""
    max_n = 0
    pattern = re.compile(r"^run (\d+)$", re.IGNORECASE)

    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if match := pattern.match(child.name):
            max_n = max(max_n, int(match.group(1)))

    run_dir = base_dir / f"run {max_n + 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_command(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GUSTOgridder for fixed mixer/line sequence with shared WCS reference file."
    )
    parser.add_argument("--source", default="G337", help="Source name passed to GUSTOgridder -s")
    parser.add_argument("--vmin", type=float, default=-160.0, help="Minimum velocity for -l")
    parser.add_argument("--vmax", type=float, default=0.0, help="Maximum velocity for -l")
    parser.add_argument("--beam", type=float, default=1.0, help="Beam FWHM in arcmin for -Beam")
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable to use. Defaults to current interpreter.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    gridder = script_dir / "GUSTOgridder.py"

    if not gridder.exists():
        raise FileNotFoundError(f"Cannot find gridder script: {gridder}")

    base_output_dir = repo_root / "Data" / "level2" / args.source
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = next_nth_run_dir(base_output_dir)

    py_exe = args.python or "python"

    # GUSTOgridder concatenates dir_write + ofile, so use a leading slash to place
    # outputs under Data/level2/<source>/<n>th run/.
    first_output = run_dir / f"{args.source}_CII_8_reference.fits"
    first_ofile = f"\\{run_dir.name}\\{first_output.name}"

    base_args = [
        "-s",
        args.source,
        "-l",
        str(args.vmin),
        str(args.vmax),
        "-Beam",
        str(args.beam),
    ]

    first_cmd = [
        py_exe,
        str(gridder),
        "-b",
        "CII",
        *base_args,
        "-x",
        "8",
        "-o",
        first_ofile,
    ]
    run_command(first_cmd, cwd=script_dir)

    jobs = [
        ("CII", "6"),
        ("CII", "0"),
        ("NII", "2"),
        ("NII", "3"),
        ("NII", "6"),
        ("NII", "0"),
    ]

    for line, mixer in jobs:
        out_name = f"{args.source}_{line}_{mixer}_matched.fits"
        out_path = run_dir / out_name
        ofile = f"\\{run_dir.name}\\{out_name}"

        cmd = [ py_exe, str(gridder), "-b", line, *base_args, "-f", str(first_output), "-x", mixer, "-o", ofile ]
        run_command(cmd, cwd=script_dir)

    print(f"All runs complete. Output folder: {run_dir}")


if __name__ == "__main__":
    main()
