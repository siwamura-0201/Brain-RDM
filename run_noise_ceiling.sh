#!/usr/bin/env bash
# Run run_noise_ceiling.py using a preset from config/.
# Usage: ./run_noise_ceiling.sh [preset_name] [stage]
# Defaults to the "noise_ceiling_things" preset and stage "all".
#
# Requires the brain RDMs for all three subjects. crossnobis additionally
# needs the trial-level pattern caches, since it cross-validates across the
# repeats that get_brainRDM-things.py averages away:
#   for s in 01 02 03; do
#     uv run python extract_patterns.py "$s" --file-type test --strip-suffix
#     uv run python get_brainRDM-crossnobis.py "$s" --file-type test
#   done
#
# Stage "diagnostics" is the retired split-half path; it is never part of
# "all". See the section header in noise_ceiling.py for why it is kept.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PRESET="${1:-noise_ceiling_things}"
STAGE="${2:-all}"
CONFIG="config/${PRESET}.conf"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

source "$CONFIG"

read -ra SUBS <<< "$SUBJECTS"
read -ra MTRS <<< "$METRICS"
read -ra CMPS <<< "$COMPARATORS"

args=(uv run python run_noise_ceiling.py
      --stage "$STAGE"
      --subjects "${SUBS[@]}"
      --file-type "$FILE_TYPE"
      --metric "${MTRS[@]}"
      --comparator "${CMPS[@]}"
      --model-folder "$MODEL_FOLDER"
      --model-substring "$MODEL_SUBSTRING")
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    args+=(-o "$OUTPUT_DIR")
fi
if [[ -n "${N_PERM:-}" ]]; then
    args+=(--n-perm "$N_PERM")
fi
if [[ -n "${N_BOOT:-}" ]]; then
    args+=(--n-boot "$N_BOOT")
fi

"${args[@]}"
