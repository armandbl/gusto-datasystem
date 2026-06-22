#!/usr/bin/env python3
"""Apply summed delta offsets to a calibration table, writing AS_MEASURED rows."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Offset history tracking  (keeps a running log of every AS_MEASURED update)
# ---------------------------------------------------------------------------

TRACKING_FIELDS = [
    "timestamp",
    "mixer",
    "old_az_deg",
    "old_el_deg",
    "new_az_deg",
    "new_el_deg",
    "delta_az_deg",
    "delta_el_deg",
    "source_deltas_csv",
]


def append_offset_history(
    csv_path: Path,
    mixer: str,
    old_az: float,
    old_el: float,
    new_az: float,
    new_el: float,
    source_deltas_csv: str,
) -> None:
    """Append a single offset-change record to the tracking CSV."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mixer": mixer,
            "old_az_deg": f"{old_az:.6f}",
            "old_el_deg": f"{old_el:.6f}",
            "new_az_deg": f"{new_az:.6f}",
            "new_el_deg": f"{new_el:.6f}",
            "delta_az_deg": f"{new_az - old_az:.6f}",
            "delta_el_deg": f"{new_el - old_el:.6f}",
            "source_deltas_csv": source_deltas_csv,
        })


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def read_deltas(path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Read delta CSV, returning ``{pix_label: (daz, del_, sigma_az, sigma_el)}``.

    Uncertainty columns are optional — defaults to NaN if missing.
    """
    deltas: dict[str, tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pix_label = str(row.get("pix_label", "")).strip()
            if not pix_label:
                continue
            az_value = row.get("mean_az_deg")
            el_value = row.get("mean_el_deg")
            if az_value in (None, ""):
                az_value = row.get("sum_az_deg", 0.0)
            if el_value in (None, ""):
                el_value = row.get("sum_el_deg", 0.0)

            daz = float(az_value or 0.0)
            del_ = float(el_value or 0.0)

            # Uncertainty columns (added by measure_mixer_crosscorr uncertain branch)
            saz = row.get("sigma_az_deg")
            sel = row.get("sigma_el_deg")
            sigma_az = float(saz) if saz not in (None, "") else float("nan")
            sigma_el = float(sel) if sel not in (None, "") else float("nan")

            deltas[pix_label] = (daz, del_, sigma_az, sigma_el)
    return deltas


# Priority order matching getMixerOffsets() in L10_pointing.py.
# AS_MEASURED > FIDUCIAL > THEORY
_OFFSET_TYPE_PRIORITY = ("AS_MEASURED", "FIDUCIAL", "THEORY")

# Band → anchor mixer (hardcoded to match the zero-referencing anchors in
# L10_pointing.py:getMixerOffsets).  Band 1 = NII, Band 2 = CII.
_BAND_ANCHORS: dict[int, str] = {1: "B1M3", 2: "B2M8"}


def _band_from_pix_label(pix_label: str) -> int:
    """Extract band number from a pix label like ``B2M5`` → 2."""
    # pix_label must start with 'B' followed by a digit.
    if len(pix_label) >= 2 and pix_label[0] == "B" and pix_label[1].isdigit():
        return int(pix_label[1])
    raise ValueError(f"Cannot determine band from pix_label: {pix_label!r}")


def _get_current_effective_offsets(
    lines: list[str],
) -> dict[str, tuple[float, float]]:
    """Determine the currently-effective (az, el) for each mixer.

    Uses the same priority order as ``getMixerOffsets()`` in
    ``L10_pointing.py``: for each mixer label the highest-priority
    entry type wins; within a type the *last* occurrence wins.
    """
    # Collect every data entry  (pix_label → list of (az, el, type))
    raw: dict[str, list[tuple[float, float, str]]] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or stripped.upper().startswith("PIX"):
            continue
        cols = stripped.split()
        if len(cols) < 4:
            continue
        pix_label = cols[0]
        try:
            az = float(cols[1])
            el = float(cols[2])
        except ValueError:
            continue
        etype = cols[3].upper()
        raw.setdefault(pix_label, []).append((az, el, etype))

    # Walk priorities; last-in-type wins (matching getMixerOffsets logic)
    current: dict[str, tuple[float, float]] = {}
    for pix_label, entries in raw.items():
        for ptype in _OFFSET_TYPE_PRIORITY:
            matches = [(az, el) for az, el, et in entries if et == ptype]
            if matches:
                current[pix_label] = matches[-1]
                break

    return current


def apply_deltas_to_offsets(
    offsets_file: Path,
    deltas: dict[str, tuple[float, float, float, float]],
    history_csv: Path | None = None,
    source_deltas_label: str = "",
) -> str:
    """Build a new offsets table with ``AS_MEASURED`` entries.

    **Critical convention**:

    The cross-correlation measures the RESIDUAL emission shift after
    current offsets are applied.  The new absolute offset is the old
    offset plus the residual::

        new_az = old_target_az + daz

    This works for both first measurement (old = THEORY) and iterative
    updates (old = previous AS_MEASURED).

    After L10 zero-referencing the correction adjusts by exactly daz::

        effective_new = (old + daz) − anchor = effective_old + daz

    The function also:
    * Strips old ``AS_MEASURED`` lines.
    * Inserts a single ``AS_MEASURED`` line after the corresponding
      ``THEORY`` / ``FIDUCIAL`` baseline.
    * Tracks uncertainties in the history CSV (not in the offsets file).
    """
    lines = offsets_file.read_text(encoding="utf-8").splitlines()
    current_offsets = _get_current_effective_offsets(lines)

    # Compute new offsets; uncertainties are tracked in history log only
    computed: dict[str, tuple[float, float]] = {}
    computed_unc: dict[str, tuple[float, float]] = {}
    for pix_label, (daz, del_, sigma_daz, sigma_del) in deltas.items():
        band = _band_from_pix_label(pix_label)
        anchor_label = _BAND_ANCHORS[band]

        anchor = current_offsets.get(anchor_label)
        if anchor is None:
            print(
                f"  WARNING: anchor {anchor_label} not found in offsets file "
                f"— skipping delta for {pix_label}"
            )
            continue

        # The cross-correlation measures the RESIDUAL emission shift
        # after current offsets are applied.
        old_target = current_offsets.get(pix_label)
        if old_target is None:
            old_target = (anchor[0], anchor[1])
        new_az = old_target[0] + daz
        new_el = old_target[1] + del_
        computed[pix_label] = (new_az, new_el)

        # Uncertainty (tracked in history log, not offsets file)
        dsa = sigma_daz if np.isfinite(sigma_daz) else 0.0
        dse = sigma_del if np.isfinite(sigma_del) else 0.0
        computed_unc[pix_label] = (dsa, dse)

    # --- Append offset-change history --------------------------------------
    if history_csv is not None:
        for pix_label, (new_az, new_el) in computed.items():
            old = current_offsets.get(pix_label)
            sa, se = computed_unc.get(pix_label, (float("nan"), float("nan")))
            if old is not None:
                append_offset_history(
                    history_csv,
                    mixer=pix_label,
                    old_az=old[0],
                    old_el=old[1],
                    new_az=new_az,
                    new_el=new_el,
                    source_deltas_csv=source_deltas_label,
                )
            else:
                append_offset_history(
                    history_csv,
                    mixer=pix_label,
                    old_az=0.0,
                    old_el=0.0,
                    new_az=new_az,
                    new_el=new_el,
                    source_deltas_csv=source_deltas_label,
                )

    # --- Build the new offsets table (4-column, no UNC) --------------------
    output_lines: list[str] = []
    emitted: set[str] = set()

    for line in lines:
        stripped = line.strip()

        # Always keep blank / structural lines.
        if not stripped or stripped.startswith("[") or stripped.upper().startswith("PIX"):
            output_lines.append(line)
            continue

        cols = stripped.split()
        if len(cols) < 4:
            output_lines.append(line)
            continue

        pix_label = cols[0]
        etype = cols[3].upper()

        # Drop old measured / applied rows — they will be replaced.
        if etype == "AS_MEASURED":
            continue

        # Preserve THEORY / FIDUCIAL (and any other unexpected types).
        output_lines.append(line)

        # Only attach an AS_MEASURED row to THEORY / FIDUCIAL lines,
        # and only once per mixer.
        if etype not in {"THEORY", "FIDUCIAL"}:
            continue
        if pix_label in emitted:
            continue

        entry = computed.get(pix_label)
        if entry is None:
            continue

        new_az, new_el = entry
        output_lines.append(
            f"{pix_label}\t{new_az:.6f}\t{new_el:.6f}\tAS_MEASURED"
        )
        emitted.add(pix_label)

    return "\n".join(output_lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply summed delta offsets to a calibration file")
    parser.add_argument("--offsets", required=True, help="Base offsets text file")
    parser.add_argument("--deltas", required=True, help="Delta CSV file from measure_mixer_crosscorr")
    parser.add_argument("--output", required=True, help="Output offsets file")
    parser.add_argument(
        "--history-csv",
        default="Perso/offset_history_log.csv",
        help="Path to the offset-change tracking CSV (default: Perso/offset_history_log.csv)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    offsets_path = Path(args.offsets).resolve()
    deltas_path = Path(args.deltas).resolve()
    output_path = Path(args.output).resolve()

    # Resolve history CSV path relative to the repo root
    _utils_dir = Path(__file__).resolve().parent
    history_csv = (_utils_dir.parent / args.history_csv).resolve()
    history_csv.parent.mkdir(parents=True, exist_ok=True)

    deltas = read_deltas(deltas_path)
    output_text = apply_deltas_to_offsets(
        offsets_path,
        deltas,
        history_csv=history_csv,
        source_deltas_label=deltas_path.name,
    )
    output_path.write_text(output_text, encoding="utf-8")

    print(f"Wrote adjusted offsets file: {output_path}")
    print(f"Offset change history appended to: {history_csv}")


if __name__ == "__main__":
    main()
