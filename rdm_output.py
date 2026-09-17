"""Writing an RDM out as csv plus a preview figure.

Split out of get_brainRDM-things.py so the crossnobis driver, which builds
its RDMs from the trial-level cache rather than from a BData file, writes
byte-identical output through the same code path.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdm_metrics import SIMILARITY_METRICS


def save_rdm_outputs(matrix, stimuli, out_prefix, metric):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(matrix, index=stimuli, columns=stimuli)
    df.to_csv(out_prefix.with_suffix(".csv"))

    if metric in SIMILARITY_METRICS:
        vmin, vmax, cmap = -1, 1, "RdBu_r"
    elif np.nanmin(matrix) < 0:
        # crossnobis is unbiased and crosses zero where the true distance is
        # small, so a colormap anchored at 0 would hide half its range.
        limit = np.nanmax(np.abs(matrix))
        vmin, vmax, cmap = -limit, limit, "RdBu_r"
    else:
        vmin, vmax, cmap = 0, None, "viridis"

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
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
