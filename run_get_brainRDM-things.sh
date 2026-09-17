#!/usr/bin/env bash
# Generate THINGS-fMRI RDMs for one subject, across every registered metric.
# Usage: ./run_get_brainRDM-things.sh [subject] [file_type]
# Defaults to subject 01, file_type test.
#
# All metrics are requested in a single get_brainRDM-things.py invocation,
# so each ROI's (large) voxel data and the h5 file itself are each loaded
# only once for the whole run, not once per metric.
#
# --strip-suffix is added automatically for file_type "test" (one stimulus
# per category, so stripping the THINGS "_<code>" suffix is safe and lets
# the RDM be compared against category-keyed RDMs, e.g. in model_output).
# It is NOT added for "training", where it would error (many images share
# a category there).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SUBJECT="${1:-01}"
FILE_TYPE="${2:-test}"
METRICS=(pearson correlation euclidean_centered)

STRIP_SUFFIX_FLAG=()
if [[ "$FILE_TYPE" == "test" ]]; then
    STRIP_SUFFIX_FLAG=(--strip-suffix)
fi

uv run python get_brainRDM-things.py "$SUBJECT" \
    --file-types "$FILE_TYPE" \
    --metric "${METRICS[@]}" \
    "${STRIP_SUFFIX_FLAG[@]}"
