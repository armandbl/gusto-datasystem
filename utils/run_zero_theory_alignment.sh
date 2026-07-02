#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Measure mixer offsets starting from zero THEORY offsets.
# Runs the full before/after cycle: zero theory → measure → correct → verify.
#
# Usage:
#   bash utils/run_zero_theory_alignment.sh G337 && bash utils/run_zero_theory_alignment.sh G348
#
# All outputs go under alignment_zero/ — no existing offsets or results touched.
# ---------------------------------------------------------------------------
set -euo pipefail

SOURCE="${1:?Usage: $0 <G337|G348>}"

# ------------------------------------------------------------------
# Source-specific parameters
# ------------------------------------------------------------------
case "$SOURCE" in
    G337)
        SCANID_START=5024
        SCANID_END=7279
        OBJ_L=-23.2
        OBJ_B=-0.0
        OBJ_VMIN=-85
        OBJ_VMAX=-60
        OBJ_RADIUS_ARCMIN=20
        OBJ_NUM=1
        GRIDDER_VMIN=-160
        GRIDDER_VMAX=0
        # CII 8=ref, 5=tgt; NII 3=ref, 2,6=tgt
        ;;
    G348)
        SCANID_START=19480
        SCANID_END=24283
        OBJ_L=-11.4
        OBJ_B=-0.6
        OBJ_VMIN=-24
        OBJ_VMAX=-14
        OBJ_RADIUS_ARCMIN=20
        OBJ_NUM=4
        GRIDDER_VMIN=-30
        GRIDDER_VMAX=10
        ;;
    *)
        echo "ERROR: unknown source '$SOURCE'. Use G337 or G348."
        exit 1
        ;;
esac

GRIDDER_BEAM=1.0
GRIDDER_KERNEL="gauss"
GRIDDER_JOBS=4

# All outputs go under alignment_zero/ — nothing overwrites alignment_comparison/
OFFSETS_ZERO_THEORY="src/GUSTO_Pipeline/calib/offsets_${SOURCE}_zero_theory.txt"
OFFSETS_ZERO_CORRECTED="src/GUSTO_Pipeline/calib/offsets_${SOURCE}_zero_corrected.txt"
CONFIG_JSON="utils/measure_mixer_crosscorr_config_${SOURCE,,}_zero.json"

ALIGN_DIR="Data/level2/${SOURCE}/alignment_zero"
THEORY_DIR="${ALIGN_DIR}/theory"
MEASURED_DIR="${ALIGN_DIR}/measured"
COMPARISON_DIR="${ALIGN_DIR}/comparison_before_after"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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
    path=$(list_numbered_runs "$1" | tail -1)
    basename "$path" 2>/dev/null || true
}

extract_obj_subcubes() {
    local run_dir="$1"
    local out_dir="$2"

    mkdir -p "$out_dir"
    echo "  Extracting obj${OBJ_NUM} subcubes from $(basename "$run_dir") -> $(basename "$out_dir")"

    # Collect unique cube files for the mixers we need
    local -A seen
    local -a unique_cubes
    for f in "$run_dir"/*.fits; do
        [[ -f "$f" ]] || continue
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
        echo "    -> ${base}_obj_${OBJ_NUM}.fits"
        run_utils_py make_subcube.py \
            --input "$cube" \
            --output "${out_dir}/${base}_obj_${OBJ_NUM}.fits" \
            --vmin "$OBJ_VMIN" --vmax "$OBJ_VMAX" \
            --l "$OBJ_L" --b "$OBJ_B" \
            --radius-arcmin "$OBJ_RADIUS_ARCMIN" \
            > /dev/null
    done
    local n_cubes
    n_cubes=$(ls "$out_dir"/*.fits 2>/dev/null | wc -l) || true
    echo "  Done: ${n_cubes// /} subcubes written"
}

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

    echo "  [Gridder] run_gusto_gridder_batch.py -s $SOURCE -j $GRIDDER_JOBS ..."
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

build_comparison_csv() {
    local results_json="$1"
    local output_csv="$2"

    echo "  Building comparison CSV -> ${output_csv}"
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

# ==================================================================
# MAIN
# ==================================================================

echo "============================================================"
echo "GUSTO Zero-Theory Alignment — $SOURCE obj${OBJ_NUM}"
echo "obj${OBJ_NUM}: l=$OBJ_L, b=$OBJ_B, v=[$OBJ_VMIN,$OBJ_VMAX], r=${OBJ_RADIUS_ARCMIN}'"
echo "Scan range: $SCANID_START–$SCANID_END"
echo "============================================================"

# ---- Step 1: Create zero-theory offsets file ---------------------
echo ""
echo "===== Step 1: Creating zero-theory offsets file ====="
cat > "$OFFSETS_ZERO_THEORY" << EOF
[offsets]
PIX	AZ		EL		COMMENT
B2M8	0.000000	0.000000	FIDUCIAL
B2M7	0.000000	0.000000	THEORY
B2M6	0.000000	0.000000	THEORY
B2M5	0.000000	0.000000	THEORY
B2M4	0.000000	0.000000	THEORY
B2M3	0.000000	0.000000	THEORY
B2M2	0.000000	0.000000	THEORY
B2M1	0.000000	0.000000	THEORY
B1M1	0.000000	0.000000	THEORY
B1M2	0.000000	0.000000	THEORY
B1M3	0.000000	0.000000	THEORY
B1M4	0.000000	0.000000	THEORY
B1M5	0.000000	0.000000	THEORY
B1M6	0.000000	0.000000	THEORY
B1M7	0.000000	0.000000	THEORY
B1M8	0.000000	0.000000	THEORY
EOF
echo "Zero-theory offsets written to: $OFFSETS_ZERO_THEORY"

# ---- Step 2: Write crosscorr config for this source --------------
cat > "$CONFIG_JSON" << EOF
{
  "data_root": "Data/level2",
  "offsets_file": "$OFFSETS_ZERO_THEORY",
  "run": "obj${OBJ_NUM}_zero",
  "auto_detect_observer": true,
  "sources": ["$SOURCE"],
  "delta_output": "Data/level2/crosscorr_deltas_${SOURCE}_zero.csv",
  "line_targets": {
    "CII": {"target_mixer": 8, "mixers": [5]},
    "NII": {"target_mixer": 3, "mixers": [2, 6]}
  }
}
EOF
echo "Crosscorr config written to: $CONFIG_JSON"

# ---- Step 3: Run pipeline with zero-theory offsets ----------------
BEFORE_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"

run_pipeline "$OFFSETS_ZERO_THEORY" "zero-theory"

AFTER_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"
ZERO_RUN="$(comm -13 <(echo "$BEFORE_RUNS") <(echo "$AFTER_RUNS") | head -1 || true)"

if [[ -z "$ZERO_RUN" ]]; then
    ZERO_RUN="Data/level2/${SOURCE}/$(latest_run_dir "Data/level2/${SOURCE}")"
    echo "WARNING: Could not determine new run, using latest: $ZERO_RUN"
else
    ZERO_RUN="${ZERO_RUN%/}"
    echo "Zero-theory pipeline produced: $ZERO_RUN"
fi

# ---- Step 4: Extract subcubes from zero-theory run ----------------
OBJ_ZERO_DIR="Data/level2/${SOURCE}/run obj${OBJ_NUM}_zero"
extract_obj_subcubes "$ZERO_RUN" "$OBJ_ZERO_DIR"

# ---- Step 5: Measure cross-correlation deltas --------------------
echo ""
echo "===== Step 5: Measuring cross-correlation deltas [zero-theory] ====="
ZERO_DELTAS="Data/level2/crosscorr_deltas_${SOURCE}_zero.csv"

CONFIG_ZERO="${CONFIG_JSON%.json}_run.json"
python -c "
import json
cfg = json.load(open('$CONFIG_JSON'))
cfg['run'] = 'obj${OBJ_NUM}_zero'
cfg['delta_output'] = '$ZERO_DELTAS'
cfg['offsets_file'] = '$OFFSETS_ZERO_THEORY'
json.dump(cfg, open('$CONFIG_ZERO', 'w'), indent=2)
"

run_utils_py measure_mixer_crosscorr.py --config "$CONFIG_ZERO"
rm -f "$CONFIG_ZERO"

if [[ ! -f "$ZERO_DELTAS" ]]; then
    echo "ERROR: Delta CSV not produced at $ZERO_DELTAS"
    exit 1
fi
echo "Zero-theory deltas saved to: $ZERO_DELTAS"

# ---- Step 6: Moment-0 comparison grid (theory) --------------------
echo ""
echo "===== Step 6: Moment-0 comparison grid [zero-theory] ====="
run_utils_py make_moment0_grid.py \
    --source "$SOURCE" \
    --moment0-dir "${OBJ_ZERO_DIR}/Compare/moment0" \
    --output "${THEORY_DIR}/moment0_grid.png"

# ---- Step 7: Generate annotated cross-correlation maps (theory) ---
measure_crosscorr "obj${OBJ_NUM}_zero" "zero_theory" "$THEORY_DIR"

# ---- Step 8: Apply deltas to create corrected offsets -------------
echo ""
echo "===== Step 7: Applying measured deltas -> corrected offsets ====="
run_utils_py apply_offset_deltas.py \
    --offsets "$OFFSETS_ZERO_THEORY" \
    --deltas "$ZERO_DELTAS" \
    --output "$OFFSETS_ZERO_CORRECTED"
echo "Corrected offsets written to: $OFFSETS_ZERO_CORRECTED"

# ---- Step 8: Run pipeline with corrected offsets ------------------
BEFORE_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"

run_pipeline "$OFFSETS_ZERO_CORRECTED" "corrected"

AFTER_RUNS="$(list_numbered_runs "Data/level2/${SOURCE}")"
CORRECTED_RUN="$(comm -13 <(echo "$BEFORE_RUNS") <(echo "$AFTER_RUNS") | head -1 || true)"

if [[ -z "$CORRECTED_RUN" ]]; then
    CORRECTED_RUN="Data/level2/${SOURCE}/$(latest_run_dir "Data/level2/${SOURCE}")"
    echo "WARNING: Could not determine new run, using latest: $CORRECTED_RUN"
else
    CORRECTED_RUN="${CORRECTED_RUN%/}"
    echo "Corrected pipeline produced: $CORRECTED_RUN"
fi

# ---- Step 9: Extract subcubes from corrected run ------------------
OBJ_CORRECTED_DIR="Data/level2/${SOURCE}/run obj${OBJ_NUM}_zero_corrected"
extract_obj_subcubes "$CORRECTED_RUN" "$OBJ_CORRECTED_DIR"

# ---- Step 10: Cross-correlation (corrected) + comparison ----------
# Create a temp config pointing to the corrected subcubes
CONFIG_CORRECTED="${CONFIG_JSON%.json}_corrected.json"
python -c "
import json
cfg = json.load(open('$CONFIG_JSON'))
cfg['run'] = 'obj${OBJ_NUM}_zero_corrected'
cfg['delta_output'] = 'Data/level2/crosscorr_deltas_${SOURCE}_zero_corrected.csv'
cfg['offsets_file'] = '$OFFSETS_ZERO_CORRECTED'
json.dump(cfg, open('$CONFIG_CORRECTED', 'w'), indent=2)
"
# Re-measure on corrected subcubes to get residual deltas
run_utils_py measure_mixer_crosscorr.py --config "$CONFIG_CORRECTED"
rm -f "$CONFIG_CORRECTED"

# ---- Step 10: Moment-0 comparison grid (corrected) ----------------
echo ""
echo "===== Step 10: Moment-0 comparison grid [corrected] ====="
run_utils_py make_moment0_grid.py \
    --source "$SOURCE" \
    --moment0-dir "${OBJ_CORRECTED_DIR}/Compare/moment0" \
    --output "${MEASURED_DIR}/moment0_grid.png"

measure_crosscorr "obj${OBJ_NUM}_zero_corrected" "corrected" "$MEASURED_DIR" "$THEORY_DIR"

# ---- Step 11: Build comparison CSV --------------------------------
echo ""
echo "===== Step 11: Building comparison CSV ====="
mkdir -p "$COMPARISON_DIR"
build_comparison_csv "${THEORY_DIR}/results.json" "${COMPARISON_DIR}/comparison_table.csv"

# ---- Summary ------------------------------------------------------
echo ""
echo "============================================================"
echo "DONE — $SOURCE zero-theory alignment comparison complete"
echo "============================================================"
echo ""
echo "Zero-theory run:       $ZERO_RUN"
echo "Corrected run:         $CORRECTED_RUN"
echo ""
echo "Offsets files (new):"
echo "  Theory (zero):       $OFFSETS_ZERO_THEORY"
echo "  Corrected:           $OFFSETS_ZERO_CORRECTED"
echo ""
echo "Results:"
echo "  Theory maps:         $THEORY_DIR"
echo "  Measured maps:       $MEASURED_DIR"
echo "  Comparison figures:  $COMPARISON_DIR"
echo "  Delta CSV (zero):    $ZERO_DELTAS"
echo ""
if [[ -d "$COMPARISON_DIR" ]]; then
    echo "Comparison figures:"
    ls -1 "$COMPARISON_DIR"/*.png 2>/dev/null || echo "  (none)"
    echo ""
    echo "Comparison CSV:"
    ls -1 "$COMPARISON_DIR"/comparison_table.csv 2>/dev/null || echo "  (not found)"
fi
