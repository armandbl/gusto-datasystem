#!/usr/bin/env python3
"""Compare cross-correlation alignment before and after offset correction.

For a given source and set of mixer pairs, measures cross-correlation
shifts on the cubes in a run directory and generates:

1. **Annotated cross-correlation maps** — PNG + FITS showing the full 2-D
   correlation surface with centre crosshair and correlation peak (red X)
   marked, plus an offset arrow and info box with pixel/angular shifts.
2. **Before/after comparison figure** — when ``--compare-with`` is passed,
   a 2×N grid (BEFORE row / AFTER row, one column per mixer pair) is saved.

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
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure utils/ is importable regardless of CWD
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from viz_helpers import find_latest_run_dir  # canonical implementation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LINE_TARGETS: dict[str, dict[str, object]] = {
    "CII": {"target_mixer": 8, "mixers": [5]},
    "NII": {"target_mixer": 3, "mixers": [2, 6]},
}

# ---------------------------------------------------------------------------
# Shared helpers — imported from measure_mixer_crosscorr instead of duplicated
# ---------------------------------------------------------------------------

from measure_mixer_crosscorr import (  # type: ignore[import-not-found]  # noqa: E402
    _cube_center_galactic,
    build_moment0_map,
    fmt_uncertainty,
    get_observer_metadata,
    header_float,
    load_cube,
    measure_shift_integer,
    measure_shift_with_uncertainty,
    pixel_offset_to_azel,
    propagate_pixel_uncertainty_to_azel,
    select_cube,
)


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
    """Draw a red X at the peak position."""
    defaults = dict(
        markersize=14, markeredgewidth=2.5, marker="x", color="red", zorder=10
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
    """Draw an arrow from centre to peak."""
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


def plot_correlation_map(
    corr: np.ndarray,
    out_path: Path,
    source: str,
    line: str,
    ref_mixer: int,
    tgt_mixer: int,
    label: str,
    lag_x: float,
    lag_y: float,
    dx_pix: float,
    dy_pix: float,
    peak_value: float,
    dlon_deg: float | None = None,
    dlat_deg: float | None = None,
    az_deg: float | None = None,
    el_deg: float | None = None,
    sigma_x_pix: float | None = None,
    sigma_y_pix: float | None = None,
    sigma_az_deg: float | None = None,
    sigma_el_deg: float | None = None,
) -> None:
    """Save a highly annotated cross-correlation map.

    Marks:
    - The zero-lag centre (white dashed crosshair)
    - The correlation peak (red X) with 1-σ error ellipse
    - An arrow from centre → peak
    - Info box with peak value, offsets, and uncertainties
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

    # Uncertainty ellipse (1-σ) if available
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

    # Offset arrow
    _add_offset_arrow(ax, cx, cy, peak_x, peak_y, dx_pix, dy_pix)

    # Info box
    lines = [
        f"Peak value: {peak_value:.3g}",
        f"Pixel shift: ({dx_pix:+.1f}, {dy_pix:+.1f}) pix",
        f"Offset magnitude: {np.sqrt(dx_pix**2 + dy_pix**2):.1f} pix",
    ]
    if (sigma_x_pix is not None and sigma_y_pix is not None
            and np.isfinite(sigma_x_pix)):
        lines.append(f"σ_pix: ({fmt_uncertainty(sigma_x_pix)}, {fmt_uncertainty(sigma_y_pix)}) pix")
    if dlon_deg is not None and dlat_deg is not None:
        lines.append(f"Galactic: ({dlon_deg:+.4f}, {dlat_deg:+.4f})°")
    if az_deg is not None and el_deg is not None:
        lines.append(f"AZ/EL: ({az_deg:+.5f}, {el_deg:+.5f})°")
    if (sigma_az_deg is not None and sigma_el_deg is not None
            and np.isfinite(sigma_az_deg)):
        lines.append(f"σ_AZ/EL: ({fmt_uncertainty(sigma_az_deg)}, {fmt_uncertainty(sigma_el_deg)})°")

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

    ax.set_xlim(cx - 75, cx + 75)
    ax.set_ylim(cy - 75, cy + 75)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_combined_before_after(
    pairs: list[dict[str, object]],
    out_path: Path,
) -> None:
    """Create a 2×N before/after comparison figure.

    Layout:
        rows:    BEFORE (top), AFTER (bottom)
        columns: one per mixer pair
    """
    n = len(pairs)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(8 * n, 14), dpi=150)
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, pair in enumerate(pairs):
        corr_before = pair["corr_before"]
        corr_after = pair["corr_after"]
        res_before = pair["res_before"]
        res_after = pair["res_after"]
        pair_label = pair["label"]

        cy_c, cx_c = (s // 2 for s in corr_before.shape)

        for row, (corr, res, subtitle) in enumerate([
            (corr_before, res_before, "BEFORE"),
            (corr_after, res_after, "AFTER"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(corr, origin="lower", cmap="viridis", aspect="auto")
            ax.set_title(f"{pair_label}  [{subtitle}]", fontsize=11, fontweight="bold")
            ax.set_xlabel("X lag index")
            ax.set_ylabel("Y lag index")
            plt.colorbar(im, ax=ax, label="Correlation", shrink=0.85)

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
            sigma_x = res.get("sigma_x_pix")
            sigma_y = res.get("sigma_y_pix")
            sigma_az = res.get("sigma_az_deg")
            sigma_el = res.get("sigma_el_deg")

            info = [
                f"({dx:+.1f}, {dy:+.1f}) pix",
                f"|offset| = {offset_mag:.1f} pix",
            ]
            if sigma_x is not None and np.isfinite(float(sigma_x)):
                info.append(f"σ: ({fmt_uncertainty(float(sigma_x))}, {fmt_uncertainty(float(sigma_y))}) pix")
            if dlon is not None and dlat is not None:
                info.append(f"Gal: ({float(dlon):+.4f}, {float(dlat):+.4f})°")
            az = res.get("az_deg")
            el = res.get("el_deg")
            if az is not None and el is not None:
                info.append(f"AZ/EL: ({float(az):+.5f}, {float(el):+.5f})°")
            if sigma_az is not None and np.isfinite(float(sigma_az)):
                info.append(f"σ AZ/EL: ({fmt_uncertainty(float(sigma_az))}, {fmt_uncertainty(float(sigma_el))})°")

            ax.text(
                0.02, 0.98, "\n".join(info),
                transform=ax.transAxes, fontsize=8, fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                zorder=13,
            )

            ax.set_xlim(cx_c - 75, cx_c + 75)
            ax.set_ylim(cy_c - 75, cy_c + 75)

    fig.suptitle("Alignment Before / After", fontsize=14, fontweight="bold", y=0.998)
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

    # Cross-correlation (integer argmax for peak_val)
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

    # Coordinate conversion — use subcube spatial centre, not CRVAL
    ref_glon, ref_glat = _cube_center_galactic(ref_header, ref_cube.shape)
    obs = get_observer_metadata(config, data_root, source, line,
                                ref_glon=ref_glon, ref_glat=ref_glat)
    az_deg, el_deg, coord_method = pixel_offset_to_azel(
        dx_pix, dy_pix, ref_header, obs
    )

    # Propagate pixel uncertainty to AZ/EL
    sigma_az_deg, sigma_el_deg = propagate_pixel_uncertainty_to_azel(
        sigma_x_pix, sigma_y_pix, cov_xy_pix,
        ref_header, obs, coord_method,
        dx_pix=dx_pix, dy_pix=dy_pix,
    )

    # --- Systematic uncertainty floor ---
    # Optional: add a configurable floor (arcsec) for unmodeled systematics.
    # Default 0.0 — only set when an empirical upper bound is measured.
    _sys_floor_arcsec = float(config.get("systematic_floor_arcsec", 0.0))
    _sys_floor_deg = _sys_floor_arcsec / 3600.0
    if _sys_floor_deg > 0:
        if np.isfinite(sigma_az_deg):
            sigma_az_deg = float(np.sqrt(sigma_az_deg ** 2 + _sys_floor_deg ** 2))
        if np.isfinite(sigma_el_deg):
            sigma_el_deg = float(np.sqrt(sigma_el_deg ** 2 + _sys_floor_deg ** 2))

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
        sigma_x_pix=sigma_x_pix,
        sigma_y_pix=sigma_y_pix,
        sigma_az_deg=sigma_az_deg,
        sigma_el_deg=sigma_el_deg,
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
        "lag_x": lag_x_int,
        "lag_y": lag_y_int,
        "dx_pix": dx_pix,
        "dy_pix": dy_pix,
        "peak_value": peak_val,
        "dlon_deg": dlon_deg,
        "dlat_deg": dlat_deg,
        "az_deg": az_deg,
        "el_deg": el_deg,
        "coord_method": coord_method,
        "sigma_x_pix": sigma_x_pix,
        "sigma_y_pix": sigma_y_pix,
        "sigma_az_deg": sigma_az_deg,
        "sigma_el_deg": sigma_el_deg,
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
            saz = res.get("sigma_az_deg", float("nan"))
            sel = res.get("sigma_el_deg", float("nan"))
            print(
                f"    dx={res['dx_pix']:+.1f} pix  "
                f"dy={res['dy_pix']:+.1f} pix  "
                f"AZ={res['az_deg']:+.6f}±{fmt_uncertainty(float(saz))}°  "
                f"EL={res['el_deg']:+.6f}±{fmt_uncertainty(float(sel))}°"
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

    # Collect all matching pairs
    pairs: list[dict[str, object]] = []
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

        corr_before_path = Path(str(b_res.get("corr_fits", "")))
        corr_after_path = Path(str(a_res.get("corr_fits", "")))

        if not corr_before_path.exists() or not corr_after_path.exists():
            print(f"  Missing correlation FITS for {pair_tag}")
            continue

        with fits.open(corr_before_path) as hdul:
            corr_before = np.array(hdul[0].data, dtype=float)
        with fits.open(corr_after_path) as hdul:
            corr_after = np.array(hdul[0].data, dtype=float)

        pairs.append({
            "corr_before": corr_before,
            "corr_after": corr_after,
            "res_before": b_res,
            "res_after": a_res,
            "label": f"{source} {line} M{ref_m} vs M{tgt_m}",
        })

    if pairs:
        comp_png = comparison_dir / "comparison_before_after.png"
        print(f"  Generating combined comparison ({len(pairs)} pairs): {comp_png.name}")
        plot_combined_before_after(pairs, comp_png)

        # ── Save before/after comparison table ─────────────────────────
        import csv as _csv
        comp_csv = comparison_dir / "comparison_table.csv"
        _fields = [
            "pair", "source", "line", "ref_mixer", "tgt_mixer",
            "dx_pix_before", "dy_pix_before", "dx_pix_after", "dy_pix_after",
            "az_before_deg", "el_before_deg", "az_after_deg", "el_after_deg",
            "sigma_az_before", "sigma_el_before", "sigma_az_after", "sigma_el_after",
            "offset_mag_before_pix", "offset_mag_after_pix",
            "improvement_pix",
        ]
        with comp_csv.open("w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=_fields)
            w.writeheader()
            for p in pairs:
                b = p["res_before"]
                a = p["res_after"]
                dx_b, dy_b = float(b.get("dx_pix", 0)), float(b.get("dy_pix", 0))
                dx_a, dy_a = float(a.get("dx_pix", 0)), float(a.get("dy_pix", 0))
                mag_b = np.sqrt(dx_b**2 + dy_b**2)
                mag_a = np.sqrt(dx_a**2 + dy_a**2)
                w.writerow({
                    "pair": p["label"],
                    "source": b.get("source", ""),
                    "line": b.get("line", ""),
                    "ref_mixer": b.get("ref_mixer", ""),
                    "tgt_mixer": b.get("tgt_mixer", ""),
                    "dx_pix_before": f"{dx_b:.3f}",
                    "dy_pix_before": f"{dy_b:.3f}",
                    "dx_pix_after": f"{dx_a:.3f}",
                    "dy_pix_after": f"{dy_a:.3f}",
                    "az_before_deg": f"{float(b.get('az_deg', 0)):.6f}",
                    "el_before_deg": f"{float(b.get('el_deg', 0)):.6f}",
                    "az_after_deg": f"{float(a.get('az_deg', 0)):.6f}",
                    "el_after_deg": f"{float(a.get('el_deg', 0)):.6f}",
                    "sigma_az_before": f"{float(b.get('sigma_az_deg', 0)):.6e}",
                    "sigma_el_before": f"{float(b.get('sigma_el_deg', 0)):.6e}",
                    "sigma_az_after": f"{float(a.get('sigma_az_deg', 0)):.6e}",
                    "sigma_el_after": f"{float(a.get('sigma_el_deg', 0)):.6e}",
                    "offset_mag_before_pix": f"{mag_b:.3f}",
                    "offset_mag_after_pix": f"{mag_a:.3f}",
                    "improvement_pix": f"{mag_b - mag_a:.3f}",
                })
        print(f"  Comparison table saved: {comp_csv.name}")

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
        saz = r.get("sigma_az_deg", float("nan"))
        sel = r.get("sigma_el_deg", float("nan"))
        print(
            f"  {r['source']:6s} {r['line']:3s} "
            f"M{r['ref_mixer']} vs M{r['tgt_mixer']}:  "
            f"dx={r['dx_pix']:+.1f}  dy={r['dy_pix']:+.1f} pix  "
            f"|offset|={np.sqrt(float(r['dx_pix'])**2 + float(r['dy_pix'])**2):.1f} pix  "
            f"AZ={r['az_deg']:+.6f}±{fmt_uncertainty(float(saz))}°  "
            f"EL={r['el_deg']:+.6f}±{fmt_uncertainty(float(sel))}°"
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
