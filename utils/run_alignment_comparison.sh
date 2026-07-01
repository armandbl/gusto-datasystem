#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full before/after alignment comparison for G337 obj1
#
# Runs the complete workflow:
#   1.  Create theory-only offsets file
#   2.  Run pipeline (L10 → gridder → diff cubes) with theory offsets
#   3.  Find the new run, extract obj1 subcubes
#   4.  Measure cross-correlation deltas (measure_mixer_crosscorr.py)
#   5.  Save annotated cross-correlation maps (theory)
#   6.  Apply measured deltas → corrected offsets file
#   7.  Re-run pipeline with corrected offsets
#   8.  Find the new run, extract obj1 subcubes
#   9.  Save annotated cross-correlation maps (measured) + comparisons
#
# Usage:
#   bash Perso/run_alignment_comparison.sh
#
# The script activates the GUSTO venv automatically.
# ---------------------------------------------------------------------------
set -euo pipefail

# ------------------------------------------------------------------
# Configuration — edit these if needed
# ------------------------------------------------------------------
SOURCE="G337"
SCANID_START=5024
SCANID_END=7279

OBJ1_L=-23.2
OBJ1_B=-0.0
OBJ1_VMIN=-85
OBJ1_VMAX=-60
OBJ1_RADIUS_ARCMIN=20

GRIDDER_VMIN=-160
GRIDDER_VMAX=0
GRIDDER_BEAM=1.0
GRIDDER_KERNEL="gauss"
GRIDDER_JOBS=4          # 4 workers = memory-safe, near-peak speed (benchmark: 3.0×)

SLICES_VELOCITIES=(-70 -38 -40 -100 -20 -120 -74 -64 -122)

OFFSETS_FILE="src/GUSTO_Pipeline/calib/offsets.txt"
OFFSETS_THEORY="src/GUSTO_Pipeline/calib/offsets_${SOURCE}_theory.txt"
OFFSETS_CORRECTED="src/GUSTO_Pipeline/calib/offsets_${SOURCE}_corrected.txt"
CONFIG_JSON="utils/measure_mixer_crosscorr_config_obj.json"

ALIGN_DIR="Data/level2/${SOURCE}/alignment_comparison"
THEORY_DIR="${ALIGN_DIR}/theory"
MEASURED_DIR="${ALIGN_DIR}/measured"
COMPARISON_DIR="${ALIGN_DIR}/comparison_before_after"

# Resolve repo root from the location of this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the GUSTO venv (script must be self-contained)
# shellcheck disable=SC1090
VENV_ACTIVATE="$HOME/.venvs/gusto-datasystem/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
else
    echo "ERROR: GUSTO venv not found at $VENV_ACTIVATE"
    echo "Run: source Perso/gusto_workon.sh"
    exit 1
fi

# ------------------------------------------------------------------
# Helper: run a Python script from utils/ with correct PYTHONPATH
# ------------------------------------------------------------------
run_utils_py() {
    PYTHONPATH="utils:${PYTHONPATH:-}" python "utils/$1" "${@:2}"
}

# ------------------------------------------------------------------
# Helper: list only numbered "run N" directories for a source
# ------------------------------------------------------------------
list_numbered_runs() {
    local source_dir="$1"
    ls -d "$source_dir"/run\ [0-9]*/ 2>/dev/null | sort -V || true
}

# ------------------------------------------------------------------
# Helper: find the latest numbered "run N" directory
# ------------------------------------------------------------------
latest_run_dir() {
    local source_dir="$1"
    local path
    path=$(list_numbered_runs "$source_dir" | tail -1)
    basename "$path" 2>/dev/null || true
}

# ------------------------------------------------------------------
# Helper: extract obj1 subcubes for all mixers from a run dir
# ------------------------------------------------------------------
extract_obj1_subcubes() {
    local run_dir="$1"    # e.g. Data/level2/G337/run 27
    local out_dir="$2"    # e.g. Data/level2/G337/run obj1_theory

    mkdir -p "$out_dir"

    echo "  Extracting obj1 subcubes from $(basename "$run_dir") → $(basename "$out_dir")"

    # The mixers we need: CII 8 (ref), CII 5 (tgt), CII 0 (all),
    #                     NII 3 (ref), NII 2, 6 (tgt), NII 0 (all)
    local -a cubes_to_extract=()

    for f in "$run_dir"/*CII*8*.fits "$run_dir"/*CII*5*.fits "$run_dir"/*CII*0*.fits; do
        [[ -f "$f" ]] && cubes_to_extract+=("$f")
    done
    for f in "$run_dir"/*NII*3*.fits "$run_dir"/*NII*2*.fits "$run_dir"/*NII*6*.fits "$run_dir"/*NII*0*.fits; do
        [[ -f "$f" ]] && cubes_to_extract+=("$f")
    done

    # Deduplicate
    local -A seen
    local -a unique_cubes
    for f in "${cubes_to_extract[@]}"; do
        local bn
        bn="$(basename "$f")"
        if [[ -z "${seen[$bn]:-}" ]]; then
            seen["$bn"]=1
            unique_cubes+=("$f")
        fi
    done

    for cube in "${unique_cubes[@]}"; do
        local base
        base="$(basename "$cube" .fits)"
        echo "    → ${base}_obj_1.fits"
        run_utils_py make_subcube.py \
            --input "$cube" \
            --output "${out_dir}/${base}_obj_1.fits" \
            --vmin "$OBJ1_VMIN" --vmax "$OBJ1_VMAX" \
            --l "$OBJ1_L" --b "$OBJ1_B" \
            --radius-arcmin "$OBJ1_RADIUS_ARCMIN" \
            > /dev/null
    done
    local n_cubes
    n_cubes=$(ls "$out_dir"/*.fits 2>/dev/null | wc -l) || true
    echo "  Done: ${n_cubes// /} subcubes written"
}

# ------------------------------------------------------------------
# Helper: save gondola telemetry from Level-1 headers to a txt file
# in the latest run directory.
# ------------------------------------------------------------------
save_telemetry() {
    local source="$1"
    local run_dir
    run_dir="Data/level2/${source}/$(latest_run_dir "Data/level2/${source}")"
    local out_file="${run_dir}/telemetry.txt"

    echo "  Saving telemetry to ${out_file} ..."
    python << TELEOF
import sys; sys.path.insert(0, 'utils')
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
import numpy as np

l1_dir = Path('Data/level1') / '${source}'
for line in ['CII', 'NII']:
    files = sorted(l1_dir.glob(f'{line}_*_L10.fits'))
    if not files:
        continue
    lats, lons, alts, utimes = [], [], [], []
    for f in files:
        try:
            with fits.open(f) as hdul:
                hdr = hdul[0].header
                lats.append(float(hdr['GON_LAT']))
                lons.append(float(hdr['GON_LON']))
                alts.append(float(hdr['GON_ALT']))
                data = hdul[1].data
                osel = data['scan_type'] == 'OTF'
                if np.any(osel):
                    utimes.append(float(np.median(data['UNIXTIME'][osel])))
        except Exception:
            continue
    if utimes:
        obs_time = Time(float(np.median(utimes)), format='unix').iso
        with open('${out_file}', 'w') as fh:
            fh.write(f'observer_lat = {float(np.median(lats)):.6f} deg\n')
            fh.write(f'observer_lon = {float(np.median(lons)):.6f} deg\n')
            fh.write(f'observer_alt = {float(np.median(alts)):.1f} m\n')
            fh.write(f'obs_time_utc = {obs_time}\n')
            fh.write(f'source = ${source}\n')
            fh.write(f'n_l1_files = {len(files)}\n')
            fh.write(f'line = {line}\n')
        print(f'  Telemetry ({line}): lat={float(np.median(lats)):.4f}, lon={float(np.median(lons)):.4f}, alt={float(np.median(alts)):.0f}m')
        break
TELEOF
}

# ------------------------------------------------------------------
# Helper: build comparison CSV from results.json
# ------------------------------------------------------------------
build_comparison_csv() {
    local results_json="$1"
    local output_csv="$2"

    echo "  Building comparison CSV → ${output_csv}"
    python << CSVEOF
import json, csv
from pathlib import Path

results = json.loads(Path('${results_json}').read_text())
with open('${output_csv}', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Source and Mixer', 'xPixels', 'yPixels', 'AZ', 'EL'])
    for r in results:
        band = '1' if r['line'] == 'NII' else '2'
        label = f"{r['source']} B{band}M{r['tgt_mixer']}"
        w.writerow([
            label,
            f"{r['dx_pix']:.1f} $\pm$ {r['sigma_x_pix']:.2f}",
            f"{r['dy_pix']:.1f} $\pm$ {r['sigma_y_pix']:.2f}",
            f"{r['az_deg']:.6f} $\pm$ {r['sigma_az_deg']:.6f}",
            f"{r['el_deg']:.6f} $\pm$ {r['sigma_el_deg']:.6f}",
        ])
print(f'  Wrote {len(results)} rows to ${output_csv}')
CSVEOF
}

# ------------------------------------------------------------------
# Helper: run the pipeline steps for a given offsets file
# ------------------------------------------------------------------
run_pipeline() {
    local offsets_file="$1"
    local label="$2"

    echo ""
    echo "===== Running pipeline [$label] ====="

    echo "  [L10] runGUSTO --source $SOURCE -s $SCANID_START $SCANID_END -l 1.0 with ${offsets_file} ..."
    runGUSTO -c src/GUSTO_Pipeline/config.gusto \
        --source "$SOURCE" \
        -s "$SCANID_START" "$SCANID_END" \
        --offsets-file "$offsets_file" \
        -e \
        -l 1.0

    echo "  [Gridder] run_gusto_gridder_batch.py -s $SOURCE -j $GRIDDER_JOBS -k $GRIDDER_KERNEL ..."
    run_utils_py run_gusto_gridder_batch.py \
        -s "$SOURCE" \
        --vmin "$GRIDDER_VMIN" --vmax "$GRIDDER_VMAX" \
        --beam "$GRIDDER_BEAM" \
        --kernel "$GRIDDER_KERNEL" \
        -j "$GRIDDER_JOBS"

    echo "  [Telemetry] Saving gondola metadata ..."
    save_telemetry "$SOURCE"

    echo "  [Diff] make_difference_cube.py --source $SOURCE ..."
    run_utils_py make_difference_cube.py --source "$SOURCE"
}

# ------------------------------------------------------------------
# Helper: measure cross-correlation + save annotated maps
# ------------------------------------------------------------------
measure_crosscorr() {
    local run_label="$1"      # e.g. obj1_theory
    local state_label="$2"    # e.g. theory
    local output_dir="$3"
    local compare_with="${4:-}"  # optional path to previous results

    # Create output dir first — realpath fails on non-existent paths
    mkdir -p "$output_dir"

    echo ""
    echo "===== Cross-correlation [$state_label] ====="
    if [[ -n "$compare_with" ]]; then
        run_utils_py compare_alignment_before_after.py \
            --config "$CONFIG_JSON" \
            --run "$run_label" \
            --label "$state_label" \
            --output-dir "$(realpath -- "$output_dir")" \
            --compare-with "$(realpath -- "$compare_with")"
    else
        run_utils_py compare_alignment_before_after.py \
            --config "$CONFIG_JSON" \
            --run "$run_label" \
            --label "$state_label" \
            --output-dir "$(realpath -- "$output_dir")"
    fi
}

# ==================================================================
# MAIN
# ==================================================================

echo "============================================================"
echo "GUSTO Alignment Before/After Comparison"
echo "Source: $SOURCE  |  Obj1: (l=$OBJ1_L, b=$OBJ1_B)"
echo "============================================================"

# ---- Step 1: Create theory-only offsets file --------------------
echo ""
echo "===== Step 1: Creating theory-only offsets ====="
grep -vE 'AS_MEASURED|DELTA_APPLIED' "$OFFSETS_FILE" > "$OFFSETS_THEORY"
echo "Theory-only offsets written to: $OFFSETS_THEORY"

# ---- Step 2: Run pipeline with theory offsets -------------------
BEFORE_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"

run_pipeline "$OFFSETS_THEORY" "theory"

# ---- Step 3: Find the new run and extract obj1 subcubes ---------
AFTER_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"
THEORY_RUN="$(comm -13 <(echo "$BEFORE_RUNS") <(echo "$AFTER_RUNS") | head -1 || true)"

if [[ -z "$THEORY_RUN" ]]; then
    THEORY_RUN="Data/level2/${SOURCE}/$(latest_run_dir "Data/level2/${SOURCE}")"
    echo "WARNING: Could not determine new run, using latest: $THEORY_RUN"
else
    THEORY_RUN="${THEORY_RUN%/}"
    echo "Theory pipeline produced: $THEORY_RUN"
fi

OBJ1_THEORY_DIR="Data/level2/${SOURCE}/run obj1_theory"
extract_obj1_subcubes "$THEORY_RUN" "$OBJ1_THEORY_DIR"

# ---- Step 4: Measure deltas from theory cubes -------------------
echo ""
echo "===== Step 4: Measuring cross-correlation deltas [theory] ====="
THEORY_DELTAS="Data/level2/crosscorr_deltas_${SOURCE}_theory.csv"

# Create a temp config pointing to the theory subcubes
CONFIG_THEORY="${CONFIG_JSON%.json}_theory.json"
python -c "
import json
cfg = json.load(open('$CONFIG_JSON'))
cfg['run'] = 'obj1_theory'
cfg['delta_output'] = '$THEORY_DELTAS'
json.dump(cfg, open('$CONFIG_THEORY', 'w'), indent=2)
print('Temp config written:', '$CONFIG_THEORY')
"

run_utils_py measure_mixer_crosscorr.py --config "$CONFIG_THEORY"

rm -f "$CONFIG_THEORY"

if [[ -f "$THEORY_DELTAS" ]]; then
    echo "Theory deltas saved to: $THEORY_DELTAS"
else
    echo "ERROR: Delta CSV not produced at $THEORY_DELTAS"
    exit 1
fi

# ---- Step 5: Moment-0 comparison grid (theory) ------------------
echo ""
echo "===== Step 5: Moment-0 comparison grid [theory] ====="
run_utils_py make_moment0_grid.py \
    --source "$SOURCE" \
    --moment0-dir "${OBJ1_THEORY_DIR}/Compare/moment0" \
    --output "${THEORY_DIR}/moment0_grid.png"

# ---- Step 6: Generate annotated cross-correlation maps (theory) --
measure_crosscorr "obj1_theory" "theory" "$THEORY_DIR"

# ---- Step 7: Apply deltas to create corrected offsets -----------
echo ""
echo "===== Step 6: Applying measured deltas ====="
run_utils_py apply_offset_deltas.py \
    --offsets "$OFFSETS_THEORY" \
    --deltas "$THEORY_DELTAS" \
    --output "$OFFSETS_CORRECTED"
echo "Corrected offsets written to: $OFFSETS_CORRECTED"

# ---- Step 7: Run pipeline with corrected offsets -----------------
BEFORE_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"

run_pipeline "$OFFSETS_CORRECTED" "corrected"

# ---- Step 8: Find the new run and extract obj1 subcubes ----------
AFTER_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"
CORRECTED_RUN="$(comm -13 <(echo "$BEFORE_RUNS") <(echo "$AFTER_RUNS") | head -1 || true)"

if [[ -z "$CORRECTED_RUN" ]]; then
    CORRECTED_RUN="Data/level2/${SOURCE}/$(latest_run_dir "Data/level2/${SOURCE}")"
    echo "WARNING: Could not determine new run, using latest: $CORRECTED_RUN"
else
    CORRECTED_RUN="${CORRECTED_RUN%/}"
    echo "Corrected pipeline produced: $CORRECTED_RUN"
fi

OBJ1_MEASURED_DIR="Data/level2/${SOURCE}/run obj1_measured"
extract_obj1_subcubes "$CORRECTED_RUN" "$OBJ1_MEASURED_DIR"

# ---- Step 9: Cross-correlation (measured) + comparison -----------
measure_crosscorr "obj1_measured" "measured" "$MEASURED_DIR" "$THEORY_DIR"

# ---- Step 10: Build comparison CSV --------------------------------
echo ""
echo "===== Step 10: Building comparison CSV ====="
build_comparison_csv "${THEORY_DIR}/results.json" "${COMPARISON_DIR}/comparison_table.csv"

# ---- Step 11: Cube slices -----------------------------------------
echo ""
echo "===== Step 11: Cube slices ====="
echo "  Running extract_cube_slices_png.py on corrected run ($CORRECTED_RUN) ..."
run_utils_py extract_cube_slices_png.py \
    --run-dir "$CORRECTED_RUN" \
    --velocities "${SLICES_VELOCITIES[@]}"

# ---- Summary ------------------------------------------------------
echo ""
echo "============================================================"
echo "DONE — Alignment comparison complete"
echo "============================================================"
echo ""
echo "Theory results:     $THEORY_DIR"
echo "Measured results:   $MEASURED_DIR"
echo "Comparison figures: $COMPARISON_DIR"
echo ""
echo "Output files per state:"
echo "  - crosscorr_G337_{LINE}_M{REF}_vs_M{TGT}.png  (annotated map)"
echo "  - crosscorr_G337_{LINE}_M{REF}_vs_M{TGT}.fits (correlation surface)"
echo "  - results.json"
echo ""
if [[ -d "$COMPARISON_DIR" ]]; then
    echo "Comparison figures:"
    ls -1 "$COMPARISON_DIR"/*.png 2>/dev/null || echo "  (none)"
    echo ""
    echo "Comparison CSV:"
    ls -1 "$COMPARISON_DIR"/comparison_table.csv 2>/dev/null || echo "  (not found)"
fi
