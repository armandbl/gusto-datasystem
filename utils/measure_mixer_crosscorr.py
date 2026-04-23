import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from astropy.io import fits
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
    dx_pix: float
    dy_pix: float
    az_deg: float
    el_deg: float
    mean_abs_pixel_residual: float
    frame_mean_abs_pixel_residual: float
    contributes_to_offsets: bool
    weight: float
    normalized_weight: float
    status: str
    correlation_png: str
    residual_cube: str


def find_latest_run_dir(source_dir: Path, run_selector: str) -> Path:
    if run_selector.lower() != "latest":
        run_dir = source_dir / f"run {run_selector}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    runs: list[tuple[int, Path]] = []
    for child in source_dir.iterdir():
        if not child.is_dir():
            continue
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
    fits_files = sorted(run_dir.glob("*.fits"))
    candidates: list[Path] = []
    for file_path in fits_files:
        p_line, p_mixer = parse_line_and_mixer_from_name(file_path)
        if p_line == line and p_mixer == mixer:
            candidates.append(file_path)

    if not candidates:
        raise FileNotFoundError(
            f"No cube found in {run_dir} for line={line} mixer={mixer}"
        )

    if len(candidates) == 1:
        return candidates[0]

    # Prefer naming used by current alignment workflow.
    if mixer == 8:
        refs = [p for p in candidates if "reference" in p.name.lower()]
        if refs:
            return sorted(refs)[-1]
    else:
        matched = [p for p in candidates if "matched" in p.name.lower()]
        if matched:
            return sorted(matched)[-1]

    return sorted(candidates, key=lambda p: len(p.name))[0]


def load_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        primary = hdul[0]
        data = np.squeeze(cast(Any, primary).data)
        header = cast(fits.Header, cast(Any, primary).header)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube in {path}, got shape={data.shape}")
    return np.array(data, dtype=float), header


def build_integrated_map(cube: np.ndarray) -> np.ndarray:
    # Integrate all velocity channels for first-pass robust registration.
    return np.nanmean(cube, axis=0)


def prep_map(image: np.ndarray) -> np.ndarray:
    out = np.array(image, dtype=float)
    median = np.nanmedian(out)
    if np.isfinite(median):
        out = out - median
    out[~np.isfinite(out)] = 0.0
    return out


def mean_absolute_residual_from_cube(cube: np.ndarray) -> tuple[float, float, np.ndarray]:
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube for residual scoring, got shape={cube.shape}")

    frame_mean_abs_values: list[float] = []
    finite_values = cube[np.isfinite(cube)]
    if finite_values.size == 0:
        raise ValueError("Residual cube contains no finite values to score")

    for frame in cube:
        finite = frame[np.isfinite(frame)]
        if finite.size == 0:
            continue
        frame_mean_abs_values.append(float(np.mean(np.abs(finite))))

    if not frame_mean_abs_values:
        raise ValueError("Residual cube contains no finite values to score")

    frame_mean_array = np.array(frame_mean_abs_values, dtype=float)
    mean_abs_pixel_residual = float(np.mean(np.abs(finite_values)))
    frame_mean_abs_pixel_residual = float(np.mean(frame_mean_array))
    return mean_abs_pixel_residual, frame_mean_abs_pixel_residual, frame_mean_array


def header_float(header: fits.Header, key: str, default: float = 0.0) -> float:
    value = header.get(key, default)
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float(default)


def measure_shift_integer(reference_map: np.ndarray, target_map: np.ndarray) -> tuple[int, int, np.ndarray]:
    # FFT-based cross-correlation scales much better than direct correlate2d on large maps.
    ref = prep_map(reference_map)
    tgt = prep_map(target_map)
    corr = fftconvolve(ref, tgt[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    center_y, center_x = (s // 2 for s in corr.shape)

    # This lag is the shift to apply to target to align it with reference.
    lag_y = int(peak_y - center_y)
    lag_x = int(peak_x - center_x)
    return lag_x, lag_y, corr


def shift2d_no_wrap(image: np.ndarray, shift_x: int, shift_y: int, fill_value: float = np.nan) -> np.ndarray:
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
    shifted = np.empty_like(cube, dtype=float)
    for chan in range(cube.shape[0]):
        shifted[chan] = shift2d_no_wrap(cube[chan], shift_x, shift_y)
    return shifted


def update_offsets_file(offsets_file: Path, pix_label: str, az_deg: float, el_deg: float) -> None:
    lines = offsets_file.read_text(encoding="utf-8").splitlines()
    new_line = f"{pix_label}\t{az_deg:.6f}\t{el_deg:.6f}\tAS_MEASURED"

    as_measured_idx = None
    theory_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or stripped.upper().startswith("PIX"):
            continue
        cols = stripped.split()
        if len(cols) < 4:
            continue
        if cols[0] != pix_label:
            continue
        comment = cols[3].upper()
        if comment == "THEORY":
            theory_idx = i
        if comment == "AS_MEASURED":
            as_measured_idx = i

    if as_measured_idx is not None:
        lines[as_measured_idx] = new_line
    elif theory_idx is not None:
        lines.insert(theory_idx + 1, new_line)
    else:
        while lines and not lines[-1].strip():
            lines.pop()
        lines.append("")
        lines.append(new_line)

    offsets_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_correlation_png(corr: np.ndarray, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("X lag index")
    ax.set_ylabel("Y lag index")
    plt.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def pix_label_from_line_mixer(line: str, mixer: int) -> str:
    band = "B2" if line.upper() == "CII" else "B1"
    return f"{band}M{mixer}"


def list_sources(data_root: Path) -> list[str]:
    sources = [child.name for child in data_root.iterdir() if child.is_dir()]
    return sorted(sources)


def compare_dir_for_run(run_dir: Path) -> Path:
    compare_dir = run_dir / "Compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    return compare_dir


def score_residual_fits(residual_fits_path: Path) -> tuple[float, float, np.ndarray]:
    cube, _header = load_cube(residual_fits_path)
    mean_abs_pixel_residual, frame_mean_abs_pixel_residual, frame_means = mean_absolute_residual_from_cube(cube)
    return mean_abs_pixel_residual, frame_mean_abs_pixel_residual, frame_means


def process_job(
    repo_root: Path,
    data_root: Path,
    job: Job,
) -> JobResult:
    source_dir = data_root / job.source
    run_dir = find_latest_run_dir(source_dir, job.run)

    reference_cube_path = select_cube(run_dir, job.line, job.target_mixer)
    target_cube_path = select_cube(run_dir, job.line, job.mixer)

    ref_cube, ref_header = load_cube(reference_cube_path)
    tgt_cube, _ = load_cube(target_cube_path)

    if ref_cube.shape[1:] != tgt_cube.shape[1:]:
        raise ValueError(
            f"Spatial shape mismatch: ref={ref_cube.shape[1:]} target={tgt_cube.shape[1:]}"
        )

    nchan = min(ref_cube.shape[0], tgt_cube.shape[0])
    ref_cube = ref_cube[:nchan]
    tgt_cube = tgt_cube[:nchan]

    map_ref = build_integrated_map(ref_cube)
    map_tgt = build_integrated_map(tgt_cube)

    lag_x, lag_y, corr = measure_shift_integer(map_ref, map_tgt)

    # Convert lag (shift target->ref) to target-minus-reference convention.
    dx_pix = -float(lag_x)
    dy_pix = -float(lag_y)

    cdelt1 = header_float(ref_header, "CDELT1", 0.0)
    cdelt2 = header_float(ref_header, "CDELT2", 0.0)
    az_deg = dx_pix * cdelt1
    el_deg = dy_pix * cdelt2

    compare_dir = compare_dir_for_run(run_dir)

    corr_png = compare_dir / f"crosscorr_{job.source}_{job.line}_M{job.target_mixer}_vs_M{job.mixer}.png"
    save_correlation_png(
        corr,
        corr_png,
        title=f"{job.source} {job.line} M{job.target_mixer} vs M{job.mixer} | lag=({lag_x},{lag_y})",
    )

    shifted_target_cube = shift3d_no_wrap(tgt_cube, lag_x, lag_y)
    residual_cube = shifted_target_cube - ref_cube
    residual_path = compare_dir / f"residual_{job.source}_{job.line}_M{job.mixer}_minus_M{job.target_mixer}.fits"
    fits.PrimaryHDU(residual_cube, header=ref_header).writeto(residual_path, overwrite=True)

    mean_abs_pixel_residual, frame_mean_abs_pixel_residual, _frame_means = score_residual_fits(residual_path)

    return JobResult(
        source=job.source,
        run_dir=str(run_dir),
        line=job.line,
        target_mixer=job.target_mixer,
        mixer=job.mixer,
        reference_cube=reference_cube_path.name,
        target_cube=target_cube_path.name,
        dx_pix=dx_pix,
        dy_pix=dy_pix,
        az_deg=az_deg,
        el_deg=el_deg,
        mean_abs_pixel_residual=mean_abs_pixel_residual,
        frame_mean_abs_pixel_residual=frame_mean_abs_pixel_residual,
        contributes_to_offsets=False,
        weight=0.0,
        normalized_weight=0.0,
        status="OK",
        correlation_png=str(corr_png),
        residual_cube=str(residual_path),
    )


def parse_line_target_config(config: dict[str, object], line: str) -> tuple[int, list[int]]:
    defaults = {
        "CII": (8, [5]),
        "NII": (2, [3, 6]),
    }
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

            mixers: list[int]
            if "mixers" in entry:
                mixers = [int(m) for m in entry["mixers"]]
            elif "mixer" in entry:
                mixers = [int(entry["mixer"])]
            else:
                raise ValueError(f"Job must provide 'mixer' or 'mixers': {entry}")

            for mixer in mixers:
                if mixer == 0:
                    continue
                if mixer == target_mixer:
                    continue
                jobs.append(
                    Job(
                        source=source,
                        line=line,
                        target_mixer=target_mixer,
                        mixer=mixer,
                        run=run,
                    )
                )

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
                jobs.append(
                    Job(
                        source=source,
                        line=line,
                        target_mixer=target_mixer,
                        mixer=mixer,
                        run=default_run,
                    )
                )

    if not jobs:
        raise ValueError("No valid jobs generated from config and available sources")
    return jobs


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "source",
        "run_dir",
        "line",
        "target_mixer",
        "mixer",
        "reference_cube",
        "target_cube",
        "dx_pix",
        "dy_pix",
        "az_deg",
        "el_deg",
        "mean_abs_pixel_residual",
        "frame_mean_abs_pixel_residual",
        "contributes_to_offsets",
        "weight",
        "normalized_weight",
        "status",
        "correlation_png",
        "residual_cube",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score residual FITS cubes by mean absolute pixel residual and select the best source per mixer"
    )
    parser.add_argument(
        "--config",
        help="Path to JSON config file for batch processing",
    )
    parser.add_argument(
        "--score-fits",
        default=None,
        help="Score a single residual FITS cube and print its mean residual",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.score_fits:
        residual_path = Path(args.score_fits).resolve()
        mean_abs_pixel_residual, frame_mean_abs_pixel_residual, frame_means = score_residual_fits(residual_path)
        print(f"file={residual_path}")
        print(f"mean_abs_pixel_residual={mean_abs_pixel_residual:.6f}")
        print(f"frame_mean_abs_pixel_residual={frame_mean_abs_pixel_residual:.6f}")
        print(f"frame_count={len(frame_means)}")
        return

    if not args.config:
        raise SystemExit("--config is required unless --score-fits is provided")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    data_root_cfg = str(config.get("data_root", "Data/level2"))
    data_root = (repo_root / data_root_cfg).resolve() if not Path(data_root_cfg).is_absolute() else Path(data_root_cfg)

    offsets_cfg = str(config.get("offsets_file", "src/GUSTO_Pipeline/calib/offsets.txt"))
    offsets_file = (repo_root / offsets_cfg).resolve() if not Path(offsets_cfg).is_absolute() else Path(offsets_cfg)

    default_run = str(config.get("run", "latest"))
    weight_power = float(config.get("weight_power", 1.0))
    weight_epsilon = float(config.get("weight_epsilon", 1e-6))

    csv_out_cfg = str(config.get("csv_output", "crosscorr_offsets_results.csv"))
    csv_out = (repo_root / csv_out_cfg).resolve() if not Path(csv_out_cfg).is_absolute() else Path(csv_out_cfg)

    jobs = parse_jobs(config, default_run=default_run, data_root=data_root)

    results: list[JobResult] = []
    for job in jobs:
        try:
            row = process_job(
                repo_root=repo_root,
                data_root=data_root,
                job=job,
            )
            print(
                f"[{job.source} {job.line} M{job.mixer}] "
                f"dx={row.dx_pix:+.1f} pix dy={row.dy_pix:+.1f} pix "
                f"AZ={row.az_deg:+.6f} EL={row.el_deg:+.6f} "
                f"mean_abs={row.mean_abs_pixel_residual:.6f}"
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
                    dx_pix=float("nan"),
                    dy_pix=float("nan"),
                    az_deg=float("nan"),
                    el_deg=float("nan"),
                    mean_abs_pixel_residual=float("inf"),
                    frame_mean_abs_pixel_residual=float("nan"),
                    contributes_to_offsets=False,
                    weight=0.0,
                    normalized_weight=0.0,
                    status=f"ERROR: {exc}",
                    correlation_png="",
                    residual_cube="",
                )
            )
            print(f"[{job.source} {job.line} M{job.mixer}] ERROR: {exc}")

    by_mixer: dict[str, list[JobResult]] = {}
    for row in results:
        if row.status.startswith("ERROR"):
            continue
        pix_label = pix_label_from_line_mixer(row.line, row.mixer)
        by_mixer.setdefault(pix_label, []).append(row)

    for pix_label, rows_for_mixer in by_mixer.items():
        weights: list[float] = []
        valid_rows: list[JobResult] = []
        for row in rows_for_mixer:
            score = row.mean_abs_pixel_residual
            if not np.isfinite(score):
                continue
            weight = 1.0 / ((score + weight_epsilon) ** weight_power)
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            row.weight = float(weight)
            valid_rows.append(row)
            weights.append(float(weight))

        if not valid_rows:
            continue

        w = np.array(weights, dtype=float)
        wsum = float(np.sum(w))
        if wsum <= 0.0 or not np.isfinite(wsum):
            continue

        az_vals = np.array([row.az_deg for row in valid_rows], dtype=float)
        el_vals = np.array([row.el_deg for row in valid_rows], dtype=float)
        az_weighted = float(np.sum(az_vals * w) / wsum)
        el_weighted = float(np.sum(el_vals * w) / wsum)

        for row in valid_rows:
            row.contributes_to_offsets = True
            row.normalized_weight = float(row.weight / wsum)

        update_offsets_file(offsets_file, pix_label, az_weighted, el_weighted)
        print(
            f"[{pix_label}] weighted update from {len(valid_rows)} sources "
            f"AZ={az_weighted:+.6f} EL={el_weighted:+.6f}"
        )

    write_csv([row.__dict__ for row in results], csv_out)
    print(f"Saved CSV results: {csv_out}")
    print(f"Offsets file: {offsets_file}")


if __name__ == "__main__":
    main()
