import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from compare_rdms import check_axis_order, load_rdm

# Registry of comparison statistics, so new ones can be added without
# touching the rest of the pipeline. `direction` says which tail of the
# permutation null counts as evidence for a match ("higher" = a larger
# value is a better match); `permutable` is False for statistics that a
# permutation leaves unchanged, whose null would collapse to a point.
STATISTICS = {}

RESULT_COLUMNS = [
    "file1", "file2", "statistic", "value", "n_shared",
    "null_mean", "null_sd", "p_perm", "z", "n_perm", "seed",
]


def register_statistic(name, direction="higher", permutable=True):
    def decorator(fn):
        fn.direction = direction
        fn.permutable = permutable
        STATISTICS[name] = fn
        return fn

    return decorator


@register_statistic("pearson")
def pearson_stat(x, y):
    return np.corrcoef(x, y)[0, 1]


@register_statistic("spearman")
def spearman_stat(x, y):
    return spearmanr(x, y).correlation


@register_statistic("kendall")
def kendall_stat(x, y):
    return kendalltau(x, y).correlation


# L1 (ratio-level) statistics: like Pearson but without centering, so a
# perfect match requires D_B = alpha * D_M through the origin rather than
# just a linear relationship. x is treated as D_B, y as D_M.
@register_statistic("l1_r0")
def l1_r0(x, y):
    """Uncentered correlation (cosine similarity) between x and y."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


@register_statistic("l1_eps", direction="lower")
def l1_eps(x, y):
    """Residual ratio of the origin-through regression x = alpha * y.

    Smaller is a better match, so its permutation p-value counts the
    lower tail.
    """
    return 1 - l1_r0(x, y) ** 2


@register_statistic("l1_alpha", permutable=False)
def l1_alpha(x, y):
    """Direction-symmetric scale factor: sign(r0) * ||x|| / ||y||.

    A permutation only reshuffles x's entries, leaving ||x|| untouched, so
    the permutation null is a single point and carries no information --
    its null columns are reported as NaN instead.
    """
    return np.sign(l1_r0(x, y)) * np.linalg.norm(x) / np.linalg.norm(y)


def find_matching_files(folder, substring):
    matches = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix == ".csv" and substring in p.name
    )
    if not matches:
        raise SystemExit(
            f"No .csv files containing '{substring}' found in {folder}"
        )
    return matches


def align_common_stimuli(df1, df2, path1, path2, min_shared):
    common_set = set(df1.index) & set(df2.index)
    if len(common_set) < min_shared:
        raise ValueError(
            f"Only {len(common_set)} shared stimuli between {path1.name} "
            f"and {path2.name} (need >= {min_shared})"
        )

    # Both RDMs must list their shared stimuli in the same order -- unlike
    # the min-shared check above, a mismatch here isn't something to just
    # skip and move on from (it usually means one file used a different
    # stimulus ordering than intended), so it stops the whole run.
    order1 = [s for s in df1.index if s in common_set]
    order2 = [s for s in df2.index if s in common_set]
    if order1 != order2:
        raise SystemExit(
            f"Shared-stimulus order mismatch between {path1} and {path2}.\n"
            f"  {path1}: {order1}\n"
            f"  {path2}: {order2}"
        )

    return df1.loc[order1, order1], df2.loc[order1, order1], order1


def permuted_upper(mat, perm, iu):
    """Upper triangle of `mat` after relabelling stimuli by `perm`.

    The permutation is applied to rows and columns together, so the result
    is the RDM of the same stimuli under a shuffled labelling. Permuting
    the upper-triangle vector directly would instead scramble which pair
    of stimuli each cell belongs to, destroying the RDM's geometry.
    """
    return mat[np.ix_(perm, perm)][iu]


def pair_seed(path1, path2):
    """Derive a seed from the pair's filenames alone.

    Keeps each pair's permutations reproducible regardless of loop order,
    how many pairs are run, or whether they are run at all.
    """
    digest = hashlib.blake2b(
        f"{path1.name}|{path2.name}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def summarize(name, fn, obs, null, n_shared, n_perm, seed):
    row = {
        "statistic": name,
        "value": obs,
        "n_shared": n_shared,
        "null_mean": np.nan,
        "null_sd": np.nan,
        "p_perm": np.nan,
        "z": np.nan,
        "n_perm": n_perm,
        "seed": seed,
    }
    if null is None:
        return row

    hit = (null >= obs) if fn.direction == "higher" else (null <= obs)
    null_mean = null.mean()
    null_sd = null.std(ddof=1)
    row.update({
        "null_mean": null_mean,
        "null_sd": null_sd,
        "p_perm": (1 + hit.sum()) / (1 + n_perm),
        "z": (obs - null_mean) / null_sd,
    })
    return row


def compare_pair(path1, path2, statistics, min_shared, n_perm):
    df1 = load_rdm(path1)
    df2 = load_rdm(path2)
    check_axis_order(df1, path1)
    check_axis_order(df2, path2)

    df1, df2, common = align_common_stimuli(df1, df2, path1, path2, min_shared)
    a, b = df1.to_numpy(), df2.to_numpy()
    iu = np.triu_indices(len(common), k=1)
    x, y = a[iu], b[iu]

    seed = pair_seed(path1, path2)
    rng = np.random.default_rng(seed)
    # Drawn once per pair and shared by every statistic: with a separate
    # draw per statistic, differences between statistics (the L1/L2/L3
    # ladder) would carry permutation noise on top of the real difference.
    perms = [rng.permutation(len(common)) for _ in range(n_perm)]

    rows = []
    for name in statistics:
        fn = STATISTICS[name]
        obs = fn(x, y)
        null = None
        if fn.permutable and n_perm:
            null = np.array([fn(permuted_upper(a, p, iu), y) for p in perms])
        rows.append(summarize(name, fn, obs, null, len(common), n_perm, seed))
    return rows


def build_long_results(files1, files2, statistics, min_shared, n_perm):
    rows = []
    for p1 in files1:
        for p2 in files2:
            try:
                pair_rows = compare_pair(p1, p2, statistics, min_shared, n_perm)
            except ValueError as e:
                print(f"Skipped {p1.name} vs {p2.name}: {e}")
                continue
            for row in pair_rows:
                print(
                    f"{p1.name} vs {p2.name}: {row['statistic']} = "
                    f"{row['value']:.4f} (n = {row['n_shared']}, "
                    f"p_perm = {row['p_perm']:.4f}, z = {row['z']:.2f})"
                )
            rows.extend(
                {"file1": p1.name, "file2": p2.name, **row} for row in pair_rows
            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def output_path_for(output_dir, folder1, substring1, folder2, substring2):
    filename = f"{folder1.name}_{substring1}_vs_{folder2.name}_{substring2}.csv"
    return (output_dir or Path(".")) / filename


def main(folder1, substring1, folder2, substring2, statistics, output_dir,
         min_shared, n_perm):
    files1 = find_matching_files(folder1, substring1)
    files2 = find_matching_files(folder2, substring2)

    results = build_long_results(files1, files2, statistics, min_shared, n_perm)

    out_path = output_path_for(
        output_dir, folder1, substring1, folder2, substring2
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(results)} rows)")


# Usage
# uv run python compare_rdm_folders.py output/things sub-01_test model_output test_50 \
#     --statistic pearson spearman kendall -o results --n-perm 1000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare RDM csv files between two folders. Files in folder1 "
            "whose filename contains substring1 are matched against every "
            "file in folder2 whose filename contains substring2, and each "
            "pair is compared on their shared stimuli using the chosen "
            "statistics (upper triangle, excluding the diagonal). Each "
            "observed value is tested against a permutation null that "
            "relabels the folder1 RDM's stimuli. Results are written as "
            "one long-format csv: a row per (file1, file2, statistic)."
        )
    )
    parser.add_argument("folder1", type=Path)
    parser.add_argument(
        "substring1", help="Only folder1 filenames containing this are compared"
    )
    parser.add_argument("folder2", type=Path)
    parser.add_argument(
        "substring2", help="Only folder2 filenames containing this are compared"
    )
    parser.add_argument(
        "--statistic", nargs="+", choices=list(STATISTICS), default=["pearson"],
        dest="statistics",
        help=(
            "One or more statistics to compute. All of them share one "
            "output csv, one row per (file1, file2, statistic), and for a "
            "given pair they share the same draw of permutations."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, dest="output_dir",
        help=(
            "Directory to save results in (default: current directory). "
            "The filename is always auto-generated, i.e. "
            "'{folder1}_{substring1}_vs_{folder2}_{substring2}.csv'."
        ),
    )
    parser.add_argument(
        "--min-shared", type=int, default=3,
        help="Minimum number of shared stimuli required to compare a pair",
    )
    parser.add_argument(
        "--n-perm", type=int, default=1000, dest="n_perm",
        help=(
            "Number of permutations for the null distribution "
            "(default: 1000). Pass 0 to skip the null entirely, leaving "
            "the null_mean/null_sd/p_perm/z columns empty."
        ),
    )
    args = parser.parse_args()

    main(
        args.folder1, args.substring1, args.folder2, args.substring2,
        args.statistics, args.output_dir, args.min_shared, args.n_perm,
    )
