import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_rdm(path):
    return pd.read_csv(path, index_col=0)


def check_axis_order(df, path):
    """An RDM's own row order must match its column order."""
    if list(df.index) != list(df.columns):
        raise ValueError(
            f"{path}: row order does not match column order.\n"
            f"  rows:    {list(df.index)}\n"
            f"  columns: {list(df.columns)}"
        )


def check_matching_order(df1, path1, df2, path2):
    """The two RDMs must share the same stimulus order on both axes."""
    if list(df1.index) != list(df2.index):
        raise ValueError(
            f"Stimulus order mismatch between {path1} and {path2}.\n"
            f"  {path1}: {list(df1.index)}\n"
            f"  {path2}: {list(df2.index)}"
        )


def upper_triangle(df):
    iu = np.triu_indices(len(df), k=1)
    return df.to_numpy()[iu]


def plot_fit(x, y, r, xlabel, ylabel, title, out_path):
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=12, color="#4daf4a", alpha=0.6, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.06, 0.9, f"r = {r:.2f}",
        transform=ax.transAxes, fontsize=14, style="italic",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(rdm1_path, rdm2_path, output, xlabel, ylabel, title):
    df1 = load_rdm(rdm1_path)
    df2 = load_rdm(rdm2_path)

    check_axis_order(df1, rdm1_path)
    check_axis_order(df2, rdm2_path)
    check_matching_order(df1, rdm1_path, df2, rdm2_path)

    x = upper_triangle(df1)
    y = upper_triangle(df2)
    r = np.corrcoef(x, y)[0, 1]
    print(f"Pearson r = {r:.4f} (n = {len(x)} pairs)")

    xlabel = xlabel or rdm1_path.stem
    ylabel = ylabel or rdm2_path.stem
    out_path = output or Path(f"{rdm1_path.stem}_vs_{rdm2_path.stem}_fit.png")
    plot_fit(x, y, r, xlabel, ylabel, title, out_path)
    print(f"Saved: {out_path}")


# Usage
# uv run python compare_rdms.py rdm1.csv rdm2.csv -o fit.png
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Pearson correlation between two RDMs (upper "
            "triangle, excluding the diagonal) and plot the fit as a "
            "measured-vs-predicted scatter."
        )
    )
    parser.add_argument("rdm1", type=Path, help="Path to the first RDM csv")
    parser.add_argument("rdm2", type=Path, help="Path to the second RDM csv")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--xlabel", default=None, help="Defaults to rdm1's filename stem"
    )
    parser.add_argument(
        "--ylabel", default=None, help="Defaults to rdm2's filename stem"
    )
    parser.add_argument("--title", default="Model fit")
    args = parser.parse_args()

    main(args.rdm1, args.rdm2, args.output, args.xlabel, args.ylabel, args.title)
