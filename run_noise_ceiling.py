import argparse
import hashlib
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import noise_ceiling as nc
from compare_rdm_folders import (
    align_common_stimuli,
    find_matching_files,
    pair_seed,
    permuted_upper,
)
from compare_rdms import check_axis_order, load_rdm
from extract_patterns import cache_path, load_cached_patterns
from things_io import ROIS, h5_path, sanitize_roi_name

SUBJECTS = ["01", "02", "03"]

# Distance metrics only: "pearson" is the sign flip of "correlation" and a
# ceiling computed on it would differ only in sign (noise_ceiling.py rejects
# it outright). "mahalanobis" is gone -- its RDM was constant by
# construction -- and "crossnobis" is its corrected replacement.
BRAIN_METRICS = ["correlation", "euclidean_centered", "crossnobis"]

# One comparator per rung, in rsatoolbox's names. See noise_ceiling.LEVELS
# for why kendall and rho-a are not here.
COMPARATORS = ["cosine", "corr", "spearman"]

RDM_DIR = Path("output/things")
OUTPUT_DIR = Path("results/noise_ceiling")

# Retired: the split-half stage below is kept runnable but is not part of
# `--stage all` any more. The ceiling is Nili's bound alone.
#
# `crossnobis` is absent on purpose. Splitting the repeats and averaging
# each half collapses the trial dimension it cross-validates over, so the
# split-half machinery cannot build it; doing it properly would need a
# separate noise precision per half. Since nothing derives a ceiling from
# this stage any more, that work was not done.
SPLIT_MODES_BY_METRIC = {
    "correlation": ["all", "contiguous", "alternating"],
    "euclidean_centered": ["all", "contiguous", "alternating"],
}

DEGENERATE_CV = 1e-9


def brain_rdm_path(subject, file_type, roi_label, metric):
    return RDM_DIR / f"sub-{subject}_{file_type}_{sanitize_roi_name(roi_label)}_{metric}_rdm.csv"


def coefficient_of_variation(vector):
    mean = vector.mean()
    return np.nan if mean == 0 else float(vector.std(ddof=1) / mean)


def voxel_ceiling_reference(subject, file_type, roi_keys):
    """THINGS' own voxelwise noise ceilings, averaged over each ROI.

    These ship inside the BData as voxel metadata (nc_testset,
    splithalf_corrected), so the figshare download the spec mentions isn't
    needed. They are explained variance per voxel, not a ceiling on an RDM,
    and are reported only as a reference: a high voxel ceiling next to a low
    RDM ceiling says the individual voxels are stable but the geometry they
    form is not.

    Read with h5py rather than bdpy so that only the metadata block is
    touched, not the 2 GB of voxel data.
    """
    rows = []
    with h5py.File(h5_path(subject, file_type), "r") as f:
        keys = [k.decode() for k in f["metadata/key"][:]]
        values = f["metadata/value"]

        def row(name):
            return values[keys.index(name), :]

        for roi_key, roi_label in roi_keys.items():
            mask = None
            for part in (p.strip() for p in roi_key.split("+")):
                selected = row(part) == 1
                mask = selected if mask is None else (mask | selected)
            entry = {"subject": subject, "roi": sanitize_roi_name(roi_label),
                     "n_voxels": int(mask.sum())}
            for name in ("nc_testset", "nc_singletrial", "splithalf_corrected"):
                if name in keys:
                    entry[f"voxel_{name}_mean"] = float(np.nanmean(row(name)[mask]))
            rows.append(entry)
    return rows


# --------------------------------------------------------------------------
# Nili ceilings and achievement rates
# --------------------------------------------------------------------------

def load_aligned(brain_paths, model_path, min_shared):
    """Model RDM and every subject's RDM on one shared stimulus order."""
    model_df = load_rdm(model_path)
    check_axis_order(model_df, model_path)

    aligned = {}
    order = None
    for subject, path in brain_paths.items():
        brain_df = load_rdm(path)
        check_axis_order(brain_df, path)
        brain_df, model_aligned, common = align_common_stimuli(
            brain_df, model_df, path, model_path, min_shared
        )
        if order is not None and common != order:
            raise SystemExit(
                f"subjects disagree on the shared stimulus order "
                f"({path} vs the others); a Nili bound over differently "
                f"ordered RDMs would be meaningless"
            )
        order, aligned[subject] = common, brain_df.to_numpy()
    return aligned, model_aligned.to_numpy(), order


def draw_permutations(n, n_perm, seed):
    """The permutations the null is built from, in the order they are drawn.

    Kept separate so the batched and reference nulls below can be held to the
    *same* draws, not merely to the same distribution. One permutation per
    draw, shared across subjects, so that the group mean score is nulled as a
    group rather than washed out by averaging three independent shuffles.
    """
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(n) for _ in range(n_perm)], dtype=np.int32)


def permutation_null_reference(score_fn, brain_matrices, n_perm, seed):
    """The obvious null: score every permutation one at a time.

    Retained as the definition that `permutation_null` is tested against
    (tests/test_permutation_null.py). It is ~6x slower over the full run, so
    nothing in the pipeline calls it.
    """
    if not n_perm:
        return None
    n = len(next(iter(brain_matrices)))
    iu = np.triu_indices(n, k=1)
    return np.array([
        score_fn([permuted_upper(m, perm, iu) for m in brain_matrices])
        for perm in draw_permutations(n, n_perm, seed)
    ])


def _comparator_vector(u, comparator):
    """What rsatoolbox's compare_<comparator> does to a vector before cosine.

    `compare_cosine` takes the vectors raw, `compare_correlation` centres
    them, and `compare_spearman` ranks and then centres them. Everything
    after that is the same cosine, so pulling this out lets the whole null be
    one matrix product.
    """
    if comparator == "spearman":
        u = rankdata(u)
    if comparator in ("corr", "spearman"):
        u = u - u.mean()
    return u


def permutation_null(u_model, brain_matrices, comparator, n_perm, seed,
                     chunk=4096):
    """Null from relabelling the stimuli of every brain RDM together.

    Same quantity as `permutation_null_reference`, computed as a handful of
    matrix products instead of `n_perm` calls into rsatoolbox. Three facts
    make that possible, all of them consequences of one observation:
    **permuting a symmetric RDM's rows and columns together only reorders the
    values of its upper triangle -- it is a bijection on unordered pairs, so
    the multiset of dissimilarities is untouched.** Therefore

      * the ranks Spearman needs are the ranks of the *unpermuted* RDM,
        reordered the same way, so `rankdata` runs once per RDM instead of
        once per permutation (this alone is ~20x on Spearman, which was
        re-ranking 4950 values through `apply_along_axis` every draw);
      * the mean subtracted by `corr`/`spearman` is the same for every
        permutation, so it is a scalar rather than a per-draw reduction;
      * the norm in the denominator of the cosine is likewise constant.

    What is left varying is the numerator, and that is one gather plus one
    dot product per draw -- i.e. a single (n_perm, n_cells) @ (n_cells,)
    product, done in chunks to bound peak memory.

    The draws come from `draw_permutations`, so this returns the *same*
    null as the reference for the same seed, not merely one from the same
    distribution.
    """
    if not n_perm:
        return None
    brain_matrices = list(brain_matrices)
    nc.check_comparator(comparator)
    n = len(brain_matrices[0])
    iu = np.triu_indices(n, k=1)
    perms = draw_permutations(n, n_perm, seed)

    u_m = _comparator_vector(np.asarray(u_model, dtype=float), comparator)
    norm_m = float(np.sqrt(u_m @ u_m))

    total = np.zeros(n_perm)
    for matrix in brain_matrices:
        u_b = np.asarray(matrix, dtype=float)[iu]
        if comparator == "spearman":
            u_b = rankdata(u_b)
        shift = float(u_b.mean()) if comparator in ("corr", "spearman") else 0.0
        norm_b = float(np.sqrt(((u_b - shift) ** 2).sum()))
        if norm_b == 0.0 or norm_m == 0.0:
            continue                      # rsatoolbox returns 0 for a zero vector

        # Gather source: the (possibly rank-transformed) RDM as a full
        # symmetric matrix, so a permutation is a pair of index arrays.
        gather = np.zeros((n, n))
        gather[iu] = u_b
        gather += gather.T

        denominator = norm_b * norm_m
        for start in range(0, n_perm, chunk):
            block = perms[start:start + chunk]
            permuted = gather[block[:, iu[0]], block[:, iu[1]]]
            if shift:
                permuted -= shift
            total[start:start + len(block)] += (permuted @ u_m) / denominator
    return total / len(brain_matrices)


def summarize_null(observed, null, n_perm, seed):
    if null is None:
        return {"perm_p": np.nan, "perm_z": np.nan, "n_perm": n_perm, "perm_seed": seed}
    null_sd = null.std(ddof=1)
    return {
        "perm_p": float((1 + (null >= observed).sum()) / (1 + n_perm)),
        "perm_z": float((observed - null.mean()) / null_sd) if null_sd else np.nan,
        "n_perm": n_perm,
        "perm_seed": seed,
    }


class CeilingCache:
    """Nili bounds keyed by what they actually depend on.

    The bounds are a property of (ROI, brain metric, comparator, stimulus
    set) and not of the model RDM, so computing them once and reusing them
    across the four model RDMs cuts the bootstrap -- by far the most
    expensive part -- to a quarter. Reuse *across* ROI, metric or comparator
    would be the mistake; the key makes that impossible.
    """

    def __init__(self, n_boot, seed_base):
        self._cache = {}
        self.n_boot = n_boot
        self.seed_base = seed_base

    def get(self, roi, metric, comparator, stack, order):
        key = (roi, metric, comparator,
               hashlib.blake2b("|".join(order).encode(), digest_size=8).hexdigest())
        if key not in self._cache:
            seed = int(key[3], 16) % 2**32
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bounds = nc.nili_ceiling(stack, comparator)
                boot = nc.nili_ceiling_bootstrap(
                    stack, comparator, n_boot=self.n_boot, seed=seed
                )
            self._cache[key] = (bounds, boot)
        return self._cache[key]


def run_ceilings(subjects, file_type, metrics, comparators, model_folder,
                 model_substring, min_shared, n_perm, n_boot):
    model_paths = find_matching_files(model_folder, model_substring)
    cache = CeilingCache(n_boot, 0)
    rows = []

    for roi_key, roi_label in ROIS.items():
        roi = sanitize_roi_name(roi_label)
        for metric in metrics:
            brain_paths = {
                s: brain_rdm_path(s, file_type, roi_label, metric) for s in subjects
            }
            missing = [p for p in brain_paths.values() if not p.exists()]
            if missing:
                print(f"Skipped {roi} / {metric}: missing {missing[0]}")
                continue

            for model_path in model_paths:
                aligned, model_matrix, order = load_aligned(
                    brain_paths, model_path, min_shared
                )
                stack = np.stack([aligned[s] for s in subjects])
                u_model = nc.upper_triangle(model_matrix)
                seed = pair_seed(Path(f"{roi}_{metric}"), model_path)

                for comparator in comparators:
                    bounds, boot = cache.get(roi, metric, comparator, stack, order)
                    rows.extend(_ceiling_rows(
                        subjects, roi, metric, model_path, comparator, stack,
                        model_matrix, u_model, order, bounds, boot, n_perm, seed,
                    ))
                print(f"{roi:9s} {metric:18s} vs {model_path.name}: done")
    return pd.DataFrame(rows)


def _ceiling_rows(subjects, roi, metric, model_path, comparator, stack,
                  model_matrix, u_model, order, bounds, boot, n_perm, seed):
    level = nc.LEVELS[comparator]

    # L1 compares through the origin, where two unrelated non-negative RDMs
    # already agree closely; the achievement rate is measured as the gain
    # above that baseline rather than from zero. The brain-side coefficient
    # of variation differs between subjects, so each subject gets its own
    # baseline and the group row gets their mean.
    baselines = [None] * len(stack)
    baseline_info = [{}] * len(stack)
    if level == "L1":
        baseline_info = [
            nc.l1_baseline(nc.upper_triangle(rdm), u_model) for rdm in stack
        ]
        baselines = [info["baseline"] for info in baseline_info]
    group_baseline = None if level != "L1" else float(np.mean(baselines))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scored = nc.model_score(model_matrix, stack, comparator)
        group_null = permutation_null(u_model, stack, comparator, n_perm, seed)

    common = {
        "roi": roi,
        "brain_metric": metric,
        "model_rdm": model_path.name,
        "level": level,
        "comparator": comparator,
        "n_stimuli": len(order),
        "n_cells": len(u_model),
        "nili_upper": bounds["upper"],
        "nili_lower": bounds["lower"],
        "nili_upper_ci_lo": boot["upper_ci"][0],
        "nili_upper_ci_hi": boot["upper_ci"][1],
        "nili_lower_ci_lo": boot["lower_ci"][0],
        "nili_lower_ci_hi": boot["lower_ci"][1],
        "nili_crossed": bounds["crossed"],
        "n_subjects": bounds["n_subjects"],
        "bootstrap": boot["bootstrap"],
        "n_boot": boot["n_boot"],
        "boot_seed": boot["boot_seed"],
        "l1_cv_model": baseline_info[0].get("cv_model", np.nan),
    }

    def achievement_row(score, baseline):
        low = nc.achievement(score, bounds["lower"], baseline)
        high = nc.achievement(score, bounds["upper"], baseline)
        return {
            "achievement_nili_lower": low["achievement"],
            "achievement_nili_upper": high["achievement"],
            "degenerate": low["degenerate"],
        }

    # The group row first: its score is the mean of the per-subject scores
    # below it, which is the only shape the Nili bounds apply to.
    rows = [{
        **common,
        "subject": "+".join(f"F{int(s)}" for s in subjects),
        "row_type": "group_mean",
        "score_obs": scored["score"],
        "score_per_subject": ";".join(f"{v:.6f}" for v in scored["score_per_subject"]),
        **summarize_null(scored["score"], group_null, n_perm, seed),
        "nili_upper_subject": np.nan,
        "nili_lower_subject": np.nan,
        "l1_baseline": np.nan if group_baseline is None else group_baseline,
        "l1_cv_brain": np.nan,
        **achievement_row(scored["score"], group_baseline),
    }]

    for i, subject in enumerate(subjects):
        observed = float(scored["score_per_subject"][i])
        null = permutation_null(
            u_model, [stack[i]], comparator, n_perm, seed + i + 1)
        rows.append({
            **common,
            "subject": f"sub-{subject}",
            "row_type": "subject",
            "score_obs": observed,
            "score_per_subject": "",
            **summarize_null(observed, null, n_perm, seed + i + 1),
            # This subject's own contribution to each bound. Dividing its
            # score by these gives an individualised achievement rate; the
            # columns above use the group bound instead, so that subject and
            # group rows sit on the same scale.
            "nili_upper_subject": bounds["upper_per_subject"][i],
            "nili_lower_subject": bounds["lower_per_subject"][i],
            "l1_baseline": np.nan if baselines[i] is None else baselines[i],
            "l1_cv_brain": baseline_info[i].get("cv_brain", np.nan),
            **achievement_row(observed, baselines[i]),
        })
    return rows


# --------------------------------------------------------------------------
# Retired: within-subject split-half diagnostics
# --------------------------------------------------------------------------

def run_diagnostics(subjects, file_type, metrics, comparators):
    """Split-half reliability per subject x ROI x metric x comparator.

    Retired from the pipeline -- `--stage all` no longer runs it and no
    ceiling is derived from it. Kept runnable because it is the evidence for
    the `mahalanobis` degeneracy and for the absence of session drift; see
    the section header in noise_ceiling.py.
    """
    rows = []
    for subject in subjects:
        for roi_key, roi_label in ROIS.items():
            roi = sanitize_roi_name(roi_label)
            patterns, stimuli, sessions = load_cached_patterns(
                cache_path(subject, file_type, roi_label)
            )
            for metric in metrics:
                if metric not in SPLIT_MODES_BY_METRIC:
                    print(f"Skipped {roi} / {metric}: not supported by the "
                          f"retired split-half path")
                    continue
                build = nc.check_distance_metric(metric)
                rdm_cv = coefficient_of_variation(
                    nc.upper_triangle(build(patterns.mean(axis=0)))
                )

                for split_mode in SPLIT_MODES_BY_METRIC[metric]:
                    results = nc.split_half_reliability_multi(
                        patterns, metric, comparators, split_mode
                    )
                    for comparator, result in results.items():
                        rows.append({
                            "subject": subject, "roi": roi,
                            "brain_metric": metric, "comparator": comparator,
                            "level": nc.LEVELS[comparator],
                            "split_mode": split_mode,
                            "n_repeats": len(sessions),
                            "n_splits": result["n_splits"],
                            "n_stimuli": len(stimuli),
                            "n_voxels": patterns.shape[2],
                            "rdm_cv": rdm_cv,
                            "degenerate_rdm": bool(rdm_cv < DEGENERATE_CV),
                            "r_hh": result["r_hh"], "r_sb": result["r_sb"],
                            "r_sb_lo": result["ci"][0], "r_sb_hi": result["ci"][1],
                        })
                    print(
                        f"sub-{subject} {roi:9s} {metric:18s} {split_mode:11s} "
                        + "  ".join(
                            f"{c}: r_sb={results[c]['r_sb']:.3f}" for c in comparators
                        )
                    )
    return pd.DataFrame(rows)


# Usage
#   uv run python run_noise_ceiling.py
#   uv run python run_noise_ceiling.py --stage diagnostics   # retired path
#
# The ceiling is Nili's between-subject bound, computed through rsatoolbox.
# Every cell is reported twice: once per subject, and once as the mean over
# subjects -- the latter being the score shape the bounds actually apply to.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Nili noise ceilings for THINGS brain RDMs and the achievement "
            "rates they imply, per subject and averaged over subjects. "
            "Ceilings are recomputed for every ROI x metric x comparator "
            "combination and never reused across them."
        )
    )
    parser.add_argument(
        "--stage", choices=["ceilings", "reference", "diagnostics", "all"],
        default="all",
        help=(
            "'all' = reference + ceilings. 'diagnostics' is the retired "
            "split-half path and is never included in 'all'."
        ),
    )
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    parser.add_argument("--file-type", default="test")
    parser.add_argument("--metric", nargs="+", default=BRAIN_METRICS,
                        dest="metrics", choices=BRAIN_METRICS)
    parser.add_argument("--comparator", nargs="+", default=COMPARATORS,
                        dest="comparators", choices=COMPARATORS)
    parser.add_argument("--model-folder", type=Path, default=Path("model_output"))
    parser.add_argument("--model-substring", default="weights_sorted_fMRI1_test_100")
    parser.add_argument("--min-shared", type=int, default=3)
    parser.add_argument("--n-perm", type=int, default=1000, dest="n_perm")
    parser.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUTPUT_DIR,
                        dest="output_dir")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("reference", "all"):
        reference = pd.DataFrame([
            row for s in args.subjects
            for row in voxel_ceiling_reference(s, args.file_type, ROIS)
        ])
        path = args.output_dir / "voxel_ceiling_reference.csv"
        reference.to_csv(path, index=False)
        print(f"Saved: {path} ({len(reference)} rows)")

    if args.stage in ("ceilings", "all"):
        ceilings = run_ceilings(
            args.subjects, args.file_type, args.metrics, args.comparators,
            args.model_folder, args.model_substring, args.min_shared,
            args.n_perm, args.n_boot,
        )
        path = args.output_dir / "ceilings.csv"
        ceilings.to_csv(path, index=False)
        print(f"Saved: {path} ({len(ceilings)} rows)")

    if args.stage == "diagnostics":
        diagnostics = run_diagnostics(
            args.subjects, args.file_type, args.metrics, args.comparators
        )
        path = args.output_dir / "diagnostics.csv"
        diagnostics.to_csv(path, index=False)
        print(f"Saved: {path} ({len(diagnostics)} rows)")
