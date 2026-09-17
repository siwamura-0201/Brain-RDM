import argparse
from pathlib import Path

import bdpy

from rdm_metrics import METRICS
from rdm_output import save_rdm_outputs
from things_io import (
    AVERAGE_REPEATS,
    FILE_TYPES,
    ROIS,
    h5_path,
    load_patterns,
    sanitize_roi_name,
    strip_stimulus_suffixes,
)

OUTPUT_DIR = Path("output/things")


def main(subject, roi, roi_label, file_type, metrics, key_order_file, strip_suffix, bdata):
    # Loading patterns (which reads the whole ROI's voxel data out of the
    # bdpy dataset) is the expensive part, so it's done once per ROI here
    # and reused for every requested metric, rather than once per metric.
    patterns, stimuli = load_patterns(
        bdata,
        roi,
        average_repeats=file_type in AVERAGE_REPEATS,
        key_order_file=key_order_file,
    )
    if strip_suffix:
        stimuli = strip_stimulus_suffixes(stimuli)

    roi_label = sanitize_roi_name(roi_label)
    for metric in metrics:
        matrix = METRICS[metric](patterns)
        out_prefix = OUTPUT_DIR / f"sub-{subject}_{file_type}_{roi_label}_{metric}_rdm"
        save_rdm_outputs(matrix, stimuli, out_prefix, metric)
        print(f"Saved: {out_prefix}.csv / .png ({len(stimuli)} stimuli)")


# Usage
# uv run python get_brainRDM-things.py 01 --file-types test
# uv run python get_brainRDM-things.py 01 --file-types test --metric pearson correlation
# uv run python get_brainRDM-things.py <subject_num> --file-types <file_types> [--key-order <path>]
#
# Note: THINGS BData files are much larger than the GOD dataset's (the
# training file alone is ~15GB on disk and has 8640 unique stimuli, so its
# RDM is 8640x8640). Loading a file with bdpy reads the whole thing into
# memory, and that load is done once per file type (shared across all ROIs
# below) rather than once per ROI -- and each ROI's voxel data is in turn
# loaded once and reused across every requested --metric, rather than once
# per metric. Pass multiple --metric values in one run instead of invoking
# this script separately per metric.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", help="e.g. 01")
    parser.add_argument(
        "--file-types", nargs="+", choices=FILE_TYPES, required=True
    )
    parser.add_argument(
        "--metric", nargs="+", choices=list(METRICS), default=["pearson"],
        dest="metrics",
        help=(
            "One or more metrics to compute. Each ROI's voxel data is "
            "loaded once and reused across all requested metrics."
        ),
    )
    parser.add_argument(
        "--key-order",
        type=Path,
        default=None,
        help=(
            "Optional path to a txt file listing stimulus names in the "
            "desired row/column order (one per line). If omitted, stimuli "
            "are ordered ascending (alphabetically) by name."
        ),
    )
    parser.add_argument(
        "--strip-suffix",
        action="store_true",
        help=(
            "Strip the THINGS '_<code>' suffix (e.g. 'alligator_14n' -> "
            "'alligator') from stimulus labels, so the RDM can be compared "
            "against RDMs keyed by category alone. Only safe for file "
            "types with one stimulus per category (e.g. test); raises an "
            "error otherwise (e.g. training)."
        ),
    )
    args = parser.parse_args()

    for file_type in args.file_types:
        path = h5_path(args.subject, file_type)
        print(f"Loading {path} ...")
        bdata = bdpy.BData(path)
        for roi, roi_label in ROIS.items():
            main(
                args.subject, roi, roi_label, file_type, args.metrics,
                args.key_order, args.strip_suffix, bdata,
            )
