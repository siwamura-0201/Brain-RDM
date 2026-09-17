#!/bin/bash
# Compare the SPoSE-embedding model RDM against every RDM under output/god/,
# printing the Pearson r and saving a fit plot per comparison.
set -euo pipefail

MODEL_RDM="model_output/spose_embedding_test_50_predicted_from_test_50_pearson_rdm.csv"
GOD_DIR="output/god"
OUT_DIR="output/compare_rdms"

mkdir -p "$OUT_DIR"

for rdm in "$GOD_DIR"/*.csv; do
    name=$(basename "$rdm" .csv)
    echo "=== $name ==="
    uv run python compare_rdms.py "$MODEL_RDM" "$rdm" \
        -o "$OUT_DIR/${name}_vs_spose_fit.png" \
        --xlabel "predicted" \
        --ylabel "measured"
done
