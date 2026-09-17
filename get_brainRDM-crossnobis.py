import argparse
from pathlib import Path

from extract_patterns import cache_path, load_cached_patterns
from rdm_metrics import TRIAL_METRICS
from rdm_output import save_rdm_outputs
from things_io import ROIS, sanitize_roi_name

OUTPUT_DIR = Path("output/things")


def main(subject, roi_label, file_type, metric):
    path = cache_path(subject, file_type, roi_label)
    if not path.exists():
        raise SystemExit(
            f"missing pattern cache {path}; run "
            f"`uv run python extract_patterns.py {subject} "
            f"--file-type {file_type} --strip-suffix` first"
        )
    patterns, stimuli, sessions = load_cached_patterns(path)

    matrix = TRIAL_METRICS[metric](patterns)
    out_prefix = (
        OUTPUT_DIR
        / f"sub-{subject}_{file_type}_{sanitize_roi_name(roi_label)}_{metric}_rdm"
    )
    save_rdm_outputs(matrix, stimuli, out_prefix, metric)
    print(
        f"Saved: {out_prefix}.csv / .png "
        f"({len(stimuli)} stimuli, {len(sessions)} sessions as CV folds)"
    )


# Usage
# uv run python get_brainRDM-crossnobis.py 01 --file-type test
#
# Builds the cross-validated Mahalanobis (crossnobis) RDM, which needs the
# individual repeats and so cannot come from get_brainRDM-things.py -- that
# script averages the repeats before building its RDM. The input is the
# trial-level cache from extract_patterns.py, so the 2GB BData files are
# not touched here.
#
# Most of the run time is the noise precision matrix (about 3 min for LOC's
# 2700 voxels, seconds for the smaller ROIs); the RDM itself takes under a
# second because the 66 session pairs are cheap once the precision is in
# hand.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build crossnobis RDMs for one subject from the trial-level "
            "pattern cache, one per ROI."
        )
    )
    parser.add_argument("subject", help="e.g. 01")
    parser.add_argument("--file-type", default="test")
    parser.add_argument(
        "--metric", default="crossnobis", choices=list(TRIAL_METRICS)
    )
    args = parser.parse_args()

    for roi_label in ROIS.values():
        main(args.subject, roi_label, args.file_type, args.metric)
