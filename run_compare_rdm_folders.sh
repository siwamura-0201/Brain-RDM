#!/usr/bin/env bash
# Run compare_rdm_folders.py using a preset from config/.
# Usage: ./run_compare_rdm_folders.sh [preset_name]
# Defaults to the "things_vs_weights_fmri1" preset, i.e.
# config/things_vs_weights_fmri1.conf. To add a preset, drop a new
# <name>.conf file into config/ (same FOLDER1/SUBSTRING1/FOLDER2/
# SUBSTRING2/STATISTICS/OUTPUT_DIR/N_PERM variables) and run with that name.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PRESET="${1:-things_vs_weights_fmri1}"
CONFIG="config/${PRESET}.conf"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

source "$CONFIG"

read -ra STATS <<< "$STATISTICS"
args=(uv run python compare_rdm_folders.py "$FOLDER1" "$SUBSTRING1" "$FOLDER2" "$SUBSTRING2" --statistic "${STATS[@]}")
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    args+=(-o "$OUTPUT_DIR")
fi
if [[ -n "${N_PERM:-}" ]]; then
    args+=(--n-perm "$N_PERM")
fi

"${args[@]}"
