#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full before/after alignment comparison for G348 (obj4 subcubes)
# ---------------------------------------------------------------------------
set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SOURCE="G348"
SCANID_START=19480
SCANID_END=24283

# obj4 subcube parameters
OBJ4_L=-11.4
OBJ4_B=-0.6
OBJ4_VMIN=-24
OBJ4_VMAX=-14
OBJ4_RADIUS_ARCMIN=20

GRIDDER_VMIN=-30
GRIDDER_VMAX=10
GRIDDER_BEAM=1.0
GRIDDER_KERNEL="gauss"
GRIDDER_JOBS=4

OFFSETS_FILE="src/GUSTO_Pipeline/calib/offsets.txt"
OFFSETS_THEORY="src/GUSTO_Pipeline/calib/offsets_${SOURCE}_theory.txt"
OFFSETS_CORRECTED="src/GUSTO_Pipeline/calib/offsets_${SOURCE}_corrected.txt"
CONFIG_JSON="utils/measure_mixer_crosscorr_config_g348.json"

ALIGN_DIR="Data/level2/${SOURCE}/alignment_comparison"
THEORY_DIR="${ALIGN_DIR}/theory"
MEASURED_DIR="${ALIGN_DIR}/measured"
COMPARISON_DIR="${ALIGN_DIR}/comparison_before_after"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv
VENV_ACTIVATE="$HOME/.venvs/gusto-datasystem/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
else
    echo "ERROR: GUSTO venv not found at $VENV_ACTIVATE"
    exit 1
fi

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
run_utils_py() {
    PYTHONPATH="utils:${PYTHONPATH:-}" python "utils/$1" "${@:2}"
}

list_numbered_runs() {
    ls -d "$1"/run\ [0-9]*/ 2>/dev/null | sort -V || true
}

latest_run_dir() {
    local source_dir="$1"
    local path
    path=$(list_numbered_runs "$1" | tail -1)
    basename "$path" 2>/dev/null || true
}

extract_obj4_subcubes() {
    local run_dir="$1"    # e.g. Data/level2/G348/run 5
    local out_dir="$2"    # e.g. Data/level2/G348/run obj4_theory

    mkdir -p "$out_dir"

    echo "  Extracting obj4 subcubes from $(basename "$run_dir") -> $(basename "$out_dir")"

    local -a cubes=()
    for pat in "CII_8_reference" "CII_5_matched" "CII_0_matched" \
               "NII_3_matched" "NII_2_matched" "NII_6_matched" "NII_0_matched"; do
        for f in "$run_dir"/G348_"$pat".fits; do
            [[ -f "$f" ]] && cubes+=("$f")
        done
    done

    for cube in "${cubes[@]}"; do
        local base
        base="$(basename "$cube" .fits)"
        echo "    -> ${base}_obj_4.fits"
        run_utils_py make_subcube.py \
            --input "$cube" \
            --output "${out_dir}/${base}_obj_4.fits" \
            --vmin "$OBJ4_VMIN" --vmax "$OBJ4_VMAX" \
            --l "$OBJ4_L" --b "$OBJ4_B" \
            --radius-arcmin "$OBJ4_RADIUS_ARCMIN" \
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

run_pipeline() {
    local offsets_file="$1"
    local label="$2"

    echo ""
    echo "===== Running pipeline [$label] ====="

    echo "  [L10] runGUSTO -b 1 2 --source $SOURCE -s $SCANID_START $SCANID_END ..."
    runGUSTO -c src/GUSTO_Pipeline/config.gusto \
        -b 1 2 \
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

measure_crosscorr() {
    local run_label="$1"
    local state_label="$2"
    local output_dir="$3"
    local compare_with="${4:-}"

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
echo "GUSTO Alignment Before/After Comparison — G348 obj4"
echo "obj4: l=$OBJ4_L, b=$OBJ4_B, v=[$OBJ4_VMIN,$OBJ4_VMAX], r=${OBJ4_RADIUS_ARCMIN}'"
echo "============================================================"

# ---- Step 1: Create G348 config ---------------------------------
cat > "$CONFIG_JSON" << 'EOF'
{
  "data_root": "Data/level2",
  "run": "obj4",
  "auto_detect_observer": true,
  "sources": ["G348"],
  "delta_output": "Data/level2/crosscorr_deltas_g348.csv",
  "line_targets": {
    "CII": {"target_mixer": 8, "mixers": [5]},
    "NII": {"target_mixer": 3, "mixers": [2, 6]}
  }
}
EOF
echo "G348 config written to: $CONFIG_JSON"

# ---- Step 2: Create theory-only offsets file --------------------
echo ""
echo "===== Step 2: Creating theory-only offsets ====="
grep -vE 'AS_MEASURED|DELTA_APPLIED' "$OFFSETS_FILE" > "$OFFSETS_THEORY"
echo "Theory-only offsets written to: $OFFSETS_THEORY"

# ---- Step 3: Run pipeline with theory offsets -------------------
BEFORE_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"

run_pipeline "$OFFSETS_THEORY" "theory"

AFTER_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"
THEORY_RUN="$(comm -13 <(echo "$BEFORE_RUNS") <(echo "$AFTER_RUNS") | head -1 || true)"

if [[ -z "$THEORY_RUN" ]]; then
    THEORY_RUN="Data/level2/${SOURCE}/$(latest_run_dir "Data/level2/${SOURCE}")"
    echo "WARNING: Could not determine new run, using latest: $THEORY_RUN"
else
    THEORY_RUN="${THEORY_RUN%/}"
    echo "Theory pipeline produced: $THEORY_RUN"
fi

# ---- Step 4: Extract obj4 theory subcubes ------------------------
OBJ4_THEORY_DIR="Data/level2/${SOURCE}/run obj4_theory"
extract_obj4_subcubes "$THEORY_RUN" "$OBJ4_THEORY_DIR"

# ---- Step 5: Measure cross-corr on theory subcubes --------------
echo ""
echo "===== Step 5: Measuring cross-correlation deltas [theory] ====="
THEORY_DELTAS="Data/level2/crosscorr_deltas_g348_theory.csv"

CONFIG_THEORY="${CONFIG_JSON%.json}_theory.json"
python -c "
import json
cfg = json.load(open('$CONFIG_JSON'))
cfg['run'] = 'obj4_theory'
cfg['delta_output'] = '$THEORY_DELTAS'
json.dump(cfg, open('$CONFIG_THEORY', 'w'), indent=2)
"

run_utils_py measure_mixer_crosscorr.py --config "$CONFIG_THEORY"
rm -f "$CONFIG_THEORY"

if [[ -f "$THEORY_DELTAS" ]]; then
    echo "Theory deltas saved to: $THEORY_DELTAS"
else
    echo "ERROR: Delta CSV not produced at $THEORY_DELTAS"
    exit 1
fi

# ---- Step 6: Moment-0 comparison grid (theory) ------------------
echo ""
echo "===== Step 6: Moment-0 comparison grid [theory] ====="
run_utils_py make_moment0_grid.py \
    --source "$SOURCE" \
    --moment0-dir "${OBJ4_THEORY_DIR}/Compare/moment0" \
    --output "${THEORY_DIR}/moment0_grid.png"

# ---- Step 7: Generate annotated maps (theory) -------------------
measure_crosscorr "obj4_theory" "theory" "$THEORY_DIR"

# ---- Step 8: Apply deltas → corrected offsets -------------------
echo ""
echo "===== Step 7: Applying measured deltas ====="
run_utils_py apply_offset_deltas.py \
    --offsets "$OFFSETS_THEORY" \
    --deltas "$THEORY_DELTAS" \
    --output "$OFFSETS_CORRECTED"
echo "Corrected offsets written to: $OFFSETS_CORRECTED"

# ---- Step 8: Run pipeline with corrected offsets -----------------
BEFORE_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"

run_pipeline "$OFFSETS_CORRECTED" "corrected"

AFTER_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"
CORRECTED_RUN="$(comm -13 <(echo "$BEFORE_RUNS") <(echo "$AFTER_RUNS") | head -1 || true)"

if [[ -z "$CORRECTED_RUN" ]]; then
    CORRECTED_RUN="Data/level2/${SOURCE}/$(latest_run_dir "Data/level2/${SOURCE}")"
    echo "WARNING: Could not determine new run, using latest: $CORRECTED_RUN"
else
    CORRECTED_RUN="${CORRECTED_RUN%/}"
    echo "Corrected pipeline produced: $CORRECTED_RUN"
fi

# ---- Step 9: Extract obj4 measured subcubes ---------------------
OBJ4_MEASURED_DIR="Data/level2/${SOURCE}/run obj4_measured"
extract_obj4_subcubes "$CORRECTED_RUN" "$OBJ4_MEASURED_DIR"

# ---- Step 10: Cross-correlation (measured) + comparison ----------
measure_crosscorr "obj4_measured" "measured" "$MEASURED_DIR" "$THEORY_DIR"

# ---- Step 11: Build comparison CSV --------------------------------
echo ""
echo "===== Step 11: Building comparison CSV ====="
build_comparison_csv "${THEORY_DIR}/results.json" "${COMPARISON_DIR}/comparison_table.csv"

# ---- Step 12: Cube slices + verify -------------------------------
echo ""
echo "===== Step 12: Cube slices + verification ====="
run_utils_py extract_cube_slices_png.py --source "$SOURCE" --velocities -13 -6 -9 -18 -20 -25
run_utils_py verify_alignment.py \
    --source "$SOURCE" \
    --data-root Data/level2 \
    --config "$CONFIG_JSON"

# ---- Summary ------------------------------------------------------
echo ""
echo "============================================================"
echo "DONE — G348 obj4 Alignment comparison complete"
echo "============================================================"
echo ""
echo "Theory run:         $THEORY_RUN"
echo "Corrected run:      $CORRECTED_RUN"
echo "Theory subcubes:    $OBJ4_THEORY_DIR"
echo "Measured subcubes:  $OBJ4_MEASURED_DIR"
echo "Theory results:     $THEORY_DIR"
echo "Measured results:   $MEASURED_DIR"
echo "Comparison figures: $COMPARISON_DIR"
echo ""
if [[ -d "$COMPARISON_DIR" ]]; then
    echo "Comparison figures:"
    ls -1 "$COMPARISON_DIR"/*.png 2>/dev/null || echo "  (none)"
    echo ""
    echo "Comparison CSV:"
    ls -1 "$COMPARISON_DIR"/comparison_table.csv 2>/dev/null || echo "  (not found)"
fi
