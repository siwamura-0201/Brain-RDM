import argparse
from pathlib import Path

import bdpy
import numpy as np

from things_io import (
    FILE_TYPES,
    ROIS,
    h5_path,
    load_trial_patterns,
    sanitize_roi_name,
    strip_stimulus_suffixes,
)

OUTPUT_DIR = Path("output/things/patterns")


def save_patterns(patterns, stimuli, sessions, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        patterns=patterns,
        stimuli=np.array(stimuli),
        sessions=sessions,
    )


def load_cached_patterns(path):
    """Read one cache file back as (patterns, stimuli, sessions)."""
    with np.load(path, allow_pickle=False) as data:
        return (
            data["patterns"],
            [str(s) for s in data["stimuli"]],
            data["sessions"],
        )


def cache_path(subject, file_type, roi_label):
    return OUTPUT_DIR / f"sub-{subject}_{file_type}_{sanitize_roi_name(roi_label)}.npz"


def main(subject, roi, roi_label, file_type, key_order_file, strip_suffix, bdata):
    patterns, stimuli, sessions = load_trial_patterns(bdata, roi, key_order_file)
    if strip_suffix:
        stimuli = strip_stimulus_suffixes(stimuli)

    out_path = cache_path(subject, file_type, roi_label)
    save_patterns(patterns, stimuli, sessions, out_path)
    print(
        f"Saved: {out_path} {patterns.shape} "
        f"({len(sessions)} repeats x {len(stimuli)} stimuli x "
        f"{patterns.shape[2]} voxels)"
    )


# Usage
# uv run python extract_patterns.py 01 --file-type test --strip-suffix
#
# Caches trial-level (not repeat-averaged) ROI patterns as
# output/things/patterns/sub-<subject>_<file_type>_<ROI>.npz, arranged as
# (repeats, stimuli, voxels). This is what split-half reliability needs and
# what get_brainRDM-things.py throws away: it averages the repeats before
# building the RDM.
#
# Patterns are kept in float64 -- the dtype bdpy hands back -- so that
# patterns.mean(axis=0) reproduces get_brainRDM-things.py's averaged matrix
# exactly rather than approximately. The caches are small anyway (~32MB per
# subject across all four ROIs), and once written the whole noise-ceiling
# pipeline runs off them without touching the 2GB h5 files again.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Cache trial-level ROI voxel patterns from a THINGS-fMRI BData "
            "file, arranged as (repeats, stimuli, voxels)."
        )
    )
    parser.add_argument("subject", help="e.g. 01")
    parser.add_argument("--file-type", choices=FILE_TYPES, default="test")
    parser.add_argument(
        "--key-order",
        type=Path,
        default=None,
        help=(
            "Optional path to a txt file listing stimulus names in the "
            "desired order (one per line). If omitted, stimuli are ordered "
            "ascending (alphabetically) by name -- matching "
            "get_brainRDM-things.py's default."
        ),
    )
    parser.add_argument(
        "--strip-suffix",
        action="store_true",
        help=(
            "Strip the THINGS '_<code>' suffix from stimulus labels, so the "
            "cache's labels match the RDM csv files in output/things/."
        ),
    )
    args = parser.parse_args()

    path = h5_path(args.subject, args.file_type)
    print(f"Loading {path} ...")
    bdata = bdpy.BData(path)
    for roi, roi_label in ROIS.items():
        main(
            args.subject, roi, roi_label, args.file_type,
            args.key_order, args.strip_suffix, bdata,
        )
