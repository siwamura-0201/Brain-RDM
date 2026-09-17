"""Reading THINGS-fMRI BData: ROI selection, stimulus labelling, ordering.

Split out of get_brainRDM-things.py for the same reason as rdm_metrics.py:
extract_patterns.py needs the *identical* ROI masks and stimulus ordering,
and a second copy of this logic drifting out of step would silently
misalign every RDM built downstream.
"""

import re
from pathlib import Path

import numpy as np

# All ROIs available in the THINGS-fMRI BData (see data/THINGS/README.md).
# Keys are the exact metadata names bdpy needs to look up the ROI (passed
# to select_roi() -> bdata.get_metadata()); values are nicknames used for
# output filenames and plot labels, freely chosen. This split means a key
# like "LO1 (prf)" -- whose space and parentheses bdpy's select() query
# parser would misread as grouping syntax, see select_roi() below -- can
# have a filesystem-safe nickname ("LO1_prf") without any sanitizing logic.

# ROIS = {
#     "V1": "V1", "V2": "V2", "V3": "V3", "hV4": "hV4",
#     "VO1": "VO1", "VO2": "VO2",
#     "LO1 (prf)": "LO1_prf", "LO2 (prf)": "LO2_prf",
#     "TO1": "TO1", "TO2": "TO2", "V3b": "V3b", "V3a": "V3a",
#     "lEBA": "lEBA", "rEBA": "rEBA", "lFFA": "lFFA", "rFFA": "rFFA",
#     "lOFA": "lOFA", "rOFA": "rOFA", "lSTS": "lSTS", "rSTS": "rSTS",
#     "lPPA": "lPPA", "rPPA": "rPPA", "lRSC": "lRSC", "rRSC": "rRSC",
#     "lTOS": "lTOS", "rTOS": "rTOS", "lLOC": "lLOC", "rLOC": "rLOC",
# }
ROIS = {
    "lLOC + rLOC": "LOC",
    "lFFA + rFFA": "FFA",
    "lPPA + rPPA": "PPA",
    "V1 + V2 + V3": "Early_VC",
}
DATA_DIR = Path("data/THINGS")

# THINGS-fMRI files, keyed by the suffix used in "sub-{subject}_{suffix}.h5".
# Test stimuli are repeated across runs (12x, to reduce noise via
# averaging); training stimuli are shown only once each.
FILE_TYPES = ["training", "test"]
AVERAGE_REPEATS = {"test"}


def h5_path(subject, file_type):
    return DATA_DIR / f"sub-{subject}_{file_type}.h5"


def select_roi(bdata, roi):
    """Select voxel columns for `roi`.

    This delegates to bdpy's own bdata.get(), whose query parser already
    treats "+" as a union (OR) of the named ROIs -- e.g. "V1 + V2 + V3"
    selects voxels belonging to any of the three, which is how ROIS above
    builds combined ROIs like "Early_VC". The one thing that parser can't
    handle is "(" / ")" in a name (e.g. "LO1 (prf)"), which it misreads as
    grouping syntax; for those, each "+"-separated component is instead
    looked up directly by exact metadata-key match.
    """
    if "(" in roi or ")" in roi:
        mask = None
        for name in (part.strip() for part in roi.split("+")):
            values = bdata.get_metadata(name)
            if values is None:
                raise KeyError(f"ROI not found in metadata: {name}")
            mask = (values == 1) if mask is None else mask | (values == 1)
        return bdata.dataset[:, mask]

    try:
        return bdata.get(roi)
    except RuntimeError as e:
        raise KeyError(f"ROI not found in metadata: {roi}") from e


def sanitize_roi_name(roi):
    return roi.replace(" ", "_").replace("(", "").replace(")", "")


# THINGS stimulus_name values are "<category>_<2-digit code><letter>", e.g.
# "alligator_14n". Other RDM sources (e.g. model_output) key their rows by
# category alone ("alligator"), so stripping this suffix is what lets those
# RDMs be compared against a THINGS-derived one (see compare_rdm_folders.py).
STIMULUS_SUFFIX_RE = re.compile(r"_\d{2}[a-z]$")


def strip_stimulus_suffix(stimulus_name):
    return STIMULUS_SUFFIX_RE.sub("", stimulus_name)


def strip_stimulus_suffixes(stimuli):
    """Strip the THINGS suffix from every stimulus label.

    Only safe when it doesn't collapse distinct stimuli onto the same
    label -- e.g. the "training" file type has ~12 images per category, so
    stripping would map many different stimuli to the same category name.
    """
    stripped = [strip_stimulus_suffix(s) for s in stimuli]
    if len(set(stripped)) != len(stripped):
        raise ValueError(
            "--strip-suffix would map multiple distinct stimuli to the "
            "same label (e.g. several images share a category). This is "
            "expected for the training file type; use it with test only."
        )
    return stripped


def collapse_by_stimulus(patterns, stimuli, average_repeats):
    """Reduce trial-level patterns to one pattern per unique stimulus."""
    if average_repeats:
        return {
            s: patterns[stimuli == s].mean(axis=0) for s in set(stimuli)
        }
    return dict(zip(stimuli, patterns))


def read_key_order_file(key_order_file):
    with open(key_order_file, encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_stimulus_order(available_stimuli, key_order_file):
    if key_order_file is None:
        print(
            "--key-order not specified: using ascending (alphabetical) "
            "stimulus order."
        )
        return sorted(available_stimuli)

    ordered = [s for s in read_key_order_file(key_order_file) if s in available_stimuli]
    missing = available_stimuli - set(ordered)
    if missing:
        raise ValueError(
            f"{len(missing)} stimuli present in the data are missing from "
            f"key-order file {key_order_file}: {sorted(missing)[:10]}"
        )
    return ordered


def load_patterns(bdata, roi, average_repeats, key_order_file):
    patterns = select_roi(bdata, roi)
    stimuli = np.array(bdata.get_label("stimulus_name"))

    pattern_by_stimulus = collapse_by_stimulus(patterns, stimuli, average_repeats)
    order = resolve_stimulus_order(set(pattern_by_stimulus), key_order_file)
    return np.stack([pattern_by_stimulus[s] for s in order]), order


def load_trial_patterns(bdata, roi, key_order_file):
    """Trial-level patterns arranged as (repeats, stimuli, voxels).

    In the THINGS test set each stimulus is shown exactly once per session,
    so "repeat" and "session" are the same thing and the trials form a
    complete stimulus x session grid. That is what makes a clean split-half
    possible, and it is asserted here rather than assumed: a missing or
    duplicated (stimulus, session) cell would quietly turn the split into
    an unbalanced one.

    Repeats come out ordered by session number and stimuli in the same
    order load_patterns() uses, so that patterns.mean(axis=0) reproduces
    load_patterns()'s averaged matrix exactly -- see
    tests/test_noise_ceiling.py::test_cache_matches_averaged_patterns.
    """
    patterns = select_roi(bdata, roi)
    stimuli = np.array(bdata.get_label("stimulus_name"))
    sessions = np.asarray(bdata.select("session")).ravel().astype(int)

    order = resolve_stimulus_order(set(stimuli), key_order_file)
    session_ids = sorted(set(sessions.tolist()))

    row_of = {}
    for i, key in enumerate(zip(stimuli, sessions)):
        if key in row_of:
            raise ValueError(
                f"stimulus {key[0]!r} appears more than once in session "
                f"{key[1]}: the trials do not form a stimulus x session grid"
            )
        row_of[key] = i

    missing = [
        (s, sess)
        for sess in session_ids
        for s in order
        if (s, sess) not in row_of
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} (stimulus, session) cells are missing, e.g. "
            f"{missing[:5]}: the trials do not form a stimulus x session grid"
        )

    stacked = np.stack([
        np.stack([patterns[row_of[(s, sess)]] for s in order])
        for sess in session_ids
    ])
    return stacked, order, np.array(session_ids)
