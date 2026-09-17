import argparse
from pathlib import Path

import bdpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROIS = ["ROI_LOC", "ROI_FFA", "ROI_PPA", "Early_VC"]
DATA_DIR = Path("data/god_raw")
OUTPUT_DIR = Path("output/god")

# GOD dataset files, keyed by the suffix used in "Subject{n}_{suffix}.h5".
# ImageNetTest and Imagery repeat the same stimuli across runs (to reduce
# noise via averaging); ImageNetTraining shows each stimulus only once.
FILE_TYPES = ["ImageNetTraining", "ImageNetTest", "Imagery"]
AVERAGE_REPEATS = {"ImageNetTest", "Imagery"}

# Registry of RDM cell metrics, so new ones can be added without touching
# the rest of the pipeline.
METRICS = {}


def register_metric(name):
    def decorator(fn):
        METRICS[name] = fn
        return fn

    return decorator


@register_metric("pearson")
def pearson_similarity(patterns):
    return np.corrcoef(patterns)


def h5_path(subject, file_type):
    return DATA_DIR / f"Subject{subject}_{file_type}.h5"


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
    patterns = bdata.select(roi)
    stimuli = np.array(bdata.get_label("stimulus_name"))

    pattern_by_stimulus = collapse_by_stimulus(patterns, stimuli, average_repeats)
    order = resolve_stimulus_order(set(pattern_by_stimulus), key_order_file)
    return np.stack([pattern_by_stimulus[s] for s in order]), order


def save_outputs(matrix, stimuli, out_prefix, metric):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(matrix, index=stimuli, columns=stimuli)
    df.to_csv(out_prefix.with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(out_prefix.name)
    if len(stimuli) <= 60:
        ax.set_xticks(range(len(stimuli)))
        ax.set_xticklabels(stimuli, rotation=90, fontsize=6)
        ax.set_yticks(range(len(stimuli)))
        ax.set_yticklabels(stimuli, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main(subject, roi, file_type, metric, key_order_file):
    bdata = bdpy.BData(h5_path(subject, file_type))
    patterns, stimuli = load_patterns(
        bdata,
        roi,
        average_repeats=file_type in AVERAGE_REPEATS,
        key_order_file=key_order_file,
    )
    matrix = METRICS[metric](patterns)

    out_prefix = OUTPUT_DIR / f"Subject{subject}_{file_type}_{roi}_{metric}_rdm"
    save_outputs(matrix, stimuli, out_prefix, metric)
    print(f"Saved: {out_prefix}.csv / .png ({len(stimuli)} stimuli)")

# Usage
# uv run python get_brainRDM-god.py 1 --file-types ImageNetTest Imagery
# uv run python get_brainRDM-god.py <subject_number> --file-types <file_types> [--key-order <path>]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", help="e.g. 1")
    parser.add_argument(
        "--file-types", nargs="+", choices=FILE_TYPES, required=True
    )
    parser.add_argument("--metric", choices=list(METRICS), default="pearson")
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
    args = parser.parse_args()

    for file_type in args.file_types:
        for roi in ROIS:
            main(args.subject, roi, file_type, args.metric, args.key_order)
