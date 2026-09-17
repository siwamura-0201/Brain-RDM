"""Noise ceilings for brain RDMs.

The ceiling is Nili's between-subject bound: how much of the geometry the
three subjects share could any model explain? It is computed through
rsatoolbox (`pool_rdm` + `compare` + `sets_leave_one_out_rdm`) rather than
by hand, and `tests/test_rsatoolbox_parity.py` holds it to
`rsatoolbox.inference.boot_noise_ceiling`.

The within-subject split-half path below it is **retired**: it is kept
because the reliability numbers it produced are still the evidence for the
`mahalanobis` degeneracy and for the absence of session drift, but the
pipeline no longer runs it and no ceiling is derived from it any more. See
the section header for what that means for `attenuation_ceiling`.

Three things in here are easy to get wrong and are therefore enforced
rather than documented:

  * A model's score must be the mean of its per-subject correlations, not
    its correlation with the averaged RDM. The latter has an upper bound of
    1 and cannot be compared against these ceilings at all. model_score()
    only accepts a stack of per-subject RDMs, so the wrong form has no way
    to be expressed.
  * Subject RDMs must be normalized before they are averaged, differently
    per comparator. `pool_rdm` owns that step -- centre-and-scale for
    `corr`, ranks for `spearman`, RMS normalization for `cosine`. Do not
    re-implement it here; an earlier version did, and the two can drift.
  * The normalization happens *before* the leave-one-out average, not
    after. Averaging raw RDMs and transforming afterwards gives different,
    wrong bounds.

Comparator names are rsatoolbox's, not the ones `compare_rdm_folders.py`
uses for its own statistics registry:

    L1  cosine     uncentered correlation (what `l1_r0` computes, to 1e-15)
    L2  corr       Pearson
    L3  spearman   Spearman

`kendall` / `tau-b` are deliberately absent. `pool_rdm` warns and truncates
their ceiling at the averaged-rank RDM -- the MATLAB toolbox's iterative
optimization was never ported -- so the bound comes out too low and the
achievement rate correspondingly too high. `rho-a` is absent for the
opposite reason: with no tied dissimilarities anywhere in this dataset it
is numerically identical to `spearman` (2.8e-17), so it would only
duplicate a column.
"""

import itertools
import warnings

import numpy as np
from rsatoolbox.inference.crossvalsets import sets_leave_one_out_rdm
from rsatoolbox.rdm import RDMs, compare
from rsatoolbox.util.inference_util import pool_rdm

from rdm_metrics import METRICS, SIMILARITY_METRICS, TRIAL_METRICS

# Comparators a ceiling is defined for, and which rung of the L1-L4 ladder
# each one measures.
LEVELS = {
    "cosine": "L1",
    "corr": "L2",
    "spearman": "L3",
}

# Comparators whose values are correlation coefficients, and so are averaged
# through Fisher's z *in the retired split-half code only*. Nili's bounds
# use a plain arithmetic mean, matching rsatoolbox and the original
# formulation; that is what `nili_ceiling` and `model_score` do.
FISHER_COMPARATORS = {"corr", "spearman", "cosine"}

# Keeps arctanh finite when a comparator returns exactly +/-1.
_FISHER_CLIP = 1 - 1e-12


def check_comparator(comparator):
    if comparator not in LEVELS:
        raise ValueError(
            f"no ceiling is defined for comparator {comparator!r}; "
            f"expected one of {sorted(LEVELS)}"
        )
    return comparator


def check_distance_metric(metric):
    """Reject similarity-valued RDMs.

    "pearson" and "correlation" are the same RDM up to a sign flip, and a
    ceiling computed on one is the sign flip of the other's. Letting both
    through would mean achievement rates whose sign depends on which of two
    equivalent files was passed in.

    The test is on the metric *name*, never on whether the values are
    non-negative: crossnobis is unbiased and crosses zero wherever the true
    distance is small, so a non-negativity test would reject it.
    """
    if metric in SIMILARITY_METRICS:
        raise ValueError(
            f"{metric!r} is a similarity, not a distance. Ceilings are "
            f"computed on distance RDMs only -- use 'correlation' (1 - r) "
            f"instead of 'pearson'."
        )
    known = dict(METRICS, **TRIAL_METRICS)
    if metric not in known:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(known)}")
    return known[metric]


def upper_triangle(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return matrix[np.triu_indices(len(matrix), k=1)]


def as_rdms(vectors):
    """Wrap one or more dissimilarity vectors as an rsatoolbox RDMs object."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[None]
    return RDMs(dissimilarities=vectors)


def compare_vectors(vec1, vec2, comparator):
    """One comparator value between two dissimilarity vectors."""
    return float(compare(as_rdms(vec1), as_rdms(vec2), comparator))


def fisher_mean(values, comparator):
    values = np.asarray(values, dtype=float)
    if comparator not in FISHER_COMPARATORS:
        return float(values.mean())
    if np.any(np.abs(values) >= _FISHER_CLIP):
        warnings.warn(
            f"{comparator}: some values reached |r| = 1 and were clipped "
            f"before Fisher-z averaging",
            stacklevel=2,
        )
    return float(np.tanh(np.arctanh(np.clip(values, -_FISHER_CLIP, _FISHER_CLIP)).mean()))


def spearman_brown(r_hh):
    """Reliability of the full-length measurement from its half-length one.

    Always the n = 2 form here: the THINGS test grid has an even number of
    sessions, so every split is exactly half against half.
    """
    r_hh = np.asarray(r_hh, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r_hh == -1, np.nan, 2 * r_hh / (1 + r_hh))


# --------------------------------------------------------------------------
# Nili between-subject bounds
# --------------------------------------------------------------------------

def _as_rdm_stack(rdms, name):
    rdms = np.asarray(rdms, dtype=float)
    if rdms.ndim != 3 or rdms.shape[1] != rdms.shape[2]:
        raise ValueError(
            f"{name} must be a stack of per-subject RDMs with shape "
            f"(n_subjects, n_stimuli, n_stimuli); got {rdms.shape}. "
            f"If you meant to pass a single averaged RDM: don't -- a model "
            f"scored against the group-average RDM has an upper bound of 1 "
            f"and cannot be compared against these ceilings."
        )
    return rdms


def _stack_to_rdms(stack):
    return as_rdms(np.stack([upper_triangle(r) for r in stack]))


def _bounds_from_rdms(rsa_rdms, comparator):
    """Per-subject upper and lower bounds, in subject order.

    Mirrors rsatoolbox's boot_noise_ceiling, but keeps the per-subject
    values instead of collapsing straight to their means: three subjects is
    too few for the mean alone to be worth reporting.

    sets_leave_one_out_rdm groups by the 'index' rdm_descriptor and takes
    np.unique of it, so entry i is subject i as long as that descriptor is
    the default 0..S-1 -- asserted here rather than assumed.
    """
    index = np.asarray(rsa_rdms.rdm_descriptors["index"])
    assert list(index) == list(range(len(index))), (
        "RDMs must carry the default 0..S-1 'index' descriptor for "
        "leave-one-out entries to line up with subjects"
    )

    pooled_all = pool_rdm(rsa_rdms, method=comparator)
    _, test_set, ceil_set = sets_leave_one_out_rdm(rsa_rdms)

    upper, lower = [], []
    for i in range(len(ceil_set)):
        held_out = test_set[i][0]
        pooled_rest = pool_rdm(ceil_set[i][0], method=comparator)
        upper.append(float(np.mean(compare(pooled_all, held_out, comparator))))
        lower.append(float(np.mean(compare(pooled_rest, held_out, comparator))))
    return np.array(upper), np.array(lower)


def nili_ceiling(rdms, comparator):
    """Upper and lower bounds on how well any model could match this group.

    Upper: each subject against the average that includes them (optimistic,
    since the average is fitted to that subject too). Lower: each subject
    against the average of the others (pessimistic). The true ceiling lies
    between.

    These are bounds on the correlation itself, so an observed r is
    compared against them directly. Do not put them through
    attenuation_ceiling() -- that square root belongs to the reliability
    route and would correct a second time.
    """
    check_comparator(comparator)
    rdms = _as_rdm_stack(rdms, "rdms")
    n_subjects = len(rdms)
    if n_subjects < 2:
        raise ValueError(f"need at least 2 subjects for Nili bounds, got {n_subjects}")

    upper_per, lower_per = _bounds_from_rdms(_stack_to_rdms(rdms), comparator)
    upper, lower = float(upper_per.mean()), float(lower_per.mean())

    # Tolerance, so that identical subjects (where both bounds are 1 up to
    # rounding) don't trip the warning that is meant for a real crossing.
    crossed = lower > upper + 1e-9
    if crossed:
        warnings.warn(
            f"{comparator}: lower bound ({lower:.4f}) exceeds upper bound "
            f"({upper:.4f}). With {n_subjects} subjects the bounds are "
            f"unstable and can cross; the values are reported unrounded.",
            stacklevel=2,
        )
    return {
        "upper": upper,
        "lower": lower,
        "upper_per_subject": upper_per,
        "lower_per_subject": lower_per,
        "n_subjects": n_subjects,
        "crossed": crossed,
    }


def nili_ceiling_bootstrap(rdms, comparator, n_boot=1000, seed=0,
                           percentiles=(2.5, 97.5)):
    """Percentile intervals for the Nili bounds, resampling *stimuli*.

    Resampling subjects is what the earlier version did and it cannot work
    here: drawing 3 subjects with replacement admits only 5 distinct bound
    values, and every draw that repeats a subject pushes the bounds towards
    1, so the interval came out as [point estimate, 1.0] by construction.
    With 100 stimuli the pattern dimension has enough entries to say
    something.

    Duplicate draws of the same stimulus leave structurally-zero cells;
    rsatoolbox's subsample_pattern marks those NaN and pool_rdm/compare are
    NaN-aware, so they drop out rather than inflating agreement.
    """
    check_comparator(comparator)
    rdms = _as_rdm_stack(rdms, "rdms")
    rsa_rdms = _stack_to_rdms(rdms)
    n_stimuli = rdms.shape[1]

    rng = np.random.default_rng(seed)
    uppers, lowers = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            drawn = rng.integers(0, n_stimuli, size=n_stimuli)
            resampled = rsa_rdms.subsample_pattern("index", drawn)
            upper_per, lower_per = _bounds_from_rdms(resampled, comparator)
            uppers.append(upper_per.mean())
            lowers.append(lower_per.mean())
    return {
        "upper_ci": tuple(np.percentile(uppers, percentiles)),
        "lower_ci": tuple(np.percentile(lowers, percentiles)),
        "n_boot": n_boot,
        "boot_seed": seed,
        "bootstrap": "pattern",
    }


def model_score(model_rdm, subject_rdms, comparator):
    """Mean over subjects of the model's correlation with each subject.

    This is the only score shape the Nili bounds apply to. The alternative
    -- correlating the model with the averaged subject RDM -- is not a
    weaker version of it but a different quantity with a different ceiling
    (1), and mixing the two is what produced the retracted ceiling in
    Khaligh-Razavi & Kriegeskorte (2014).
    """
    check_comparator(comparator)
    subject_rdms = _as_rdm_stack(subject_rdms, "subject_rdms")
    u_model = upper_triangle(model_rdm)
    if len(u_model) != subject_rdms.shape[1] * (subject_rdms.shape[1] - 1) // 2:
        raise ValueError("model_rdm and subject_rdms cover different numbers of stimuli")

    per_subject = np.array(
        compare(as_rdms(u_model), _stack_to_rdms(subject_rdms), comparator)
    ).ravel()
    return {
        "score": float(per_subject.mean()),
        "score_per_subject": per_subject,
        "n_subjects": len(per_subject),
    }

# --------------------------------------------------------------------------
# Within-subject split-half reliability -- RETIRED
#
# The pipeline no longer calls any of this and no ceiling is derived from
# it. It stays because its output is still the evidence for two things that
# would otherwise have to be re-derived: that the removed `mahalanobis` RDM
# was degenerate rather than merely noisy, and that the 12 sessions show no
# drift (contiguous and alternating splits land within 0.021 of the
# 462-split mean). results/noise_ceiling/diagnostics.csv is that output.
#
# Do not wire it back into the ceiling path without re-reading section 4.3
# of the spec: the reliability route needs the sqrt(r_SB) conversion, which
# Nili's bounds must not get.
# --------------------------------------------------------------------------

def balanced_splits(n_repeats, mode="all"):
    """Ways of cutting the repeats into two equal halves.

    "all" enumerates every balanced split exactly once. Because each split
    and its complement give the same pair of halves, one repeat (index 0) is
    held fixed in the first half, which selects one member of each
    complementary pair: C(n-1, n/2-1) splits, i.e. 462 for the 12 THINGS
    sessions. That is few enough to enumerate exhaustively, so there is no
    random sampling here and no seed to fix.

    "contiguous" (first half of the sessions against the second) and
    "alternating" (odd against even) are single diagnostic splits: every
    repeat in this dataset is a different session, so session effects cannot
    be avoided, only measured. If "contiguous" comes out markedly below the
    other two, the sessions have drifted and Spearman-Brown's assumption
    that the halves are parallel measurements no longer holds.
    """
    if n_repeats % 2:
        raise ValueError(
            f"balanced splits need an even number of repeats, got {n_repeats}"
        )
    index = np.arange(n_repeats)
    half = n_repeats // 2

    if mode == "contiguous":
        return [(index[:half], index[half:])]
    if mode == "alternating":
        return [(index[0::2], index[1::2])]
    if mode == "all":
        return [
            (np.array((0,) + rest), np.setdiff1d(index, (0,) + rest))
            for rest in itertools.combinations(range(1, n_repeats), half - 1)
        ]
    raise ValueError(
        f"unknown split mode {mode!r}; expected 'all', 'contiguous' or 'alternating'"
    )


def split_half_rdm_pairs(patterns, metric, split_mode="all"):
    """Dissimilarity vectors for each half of each balanced split.

    Returns a list of (upper_A, upper_B). Building these is the expensive
    part -- prohibitively so for `mahalanobis`, whose pseudo-inverse is
    O(voxels^3) -- so they are built once here and reused across every
    comparator rather than rebuilt per comparator.
    """
    build_rdm = check_distance_metric(metric)

    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim != 3:
        raise ValueError(
            f"patterns must have shape (repeats, stimuli, voxels); got {patterns.shape}"
        )

    return [
        (
            upper_triangle(build_rdm(patterns[a].mean(axis=0))),
            upper_triangle(build_rdm(patterns[b].mean(axis=0))),
        )
        for a, b in balanced_splits(len(patterns), split_mode)
    ]


def split_half_reliability_multi(patterns, metric, comparators, split_mode="all",
                                 percentiles=(2.5, 97.5)):
    """split_half_reliability() for several comparators at once.

    Same results, but the RDM halves are built once instead of once per
    comparator.
    """
    for comparator in comparators:
        check_comparator(comparator)
    pairs = split_half_rdm_pairs(patterns, metric, split_mode)
    return {
        comparator: _summarize_split_half(pairs, comparator, split_mode, percentiles)
        for comparator in comparators
    }


def split_half_reliability(patterns, metric, comparator, split_mode="all",
                           percentiles=(2.5, 97.5)):
    """How well one subject's RDM reproduces itself across repeats.

    `patterns` is (repeats, stimuli, voxels). Each half's repeats are
    averaged first and the RDM is built from that average, so the two RDMs
    share no trials.
    """
    check_comparator(comparator)
    pairs = split_half_rdm_pairs(patterns, metric, split_mode)
    return _summarize_split_half(pairs, comparator, split_mode, percentiles)


def _summarize_split_half(pairs, comparator, split_mode, percentiles):
    check_comparator(comparator)
    with warnings.catch_warnings():
        # A constant RDM (see the mahalanobis note in run_noise_ceiling.py)
        # makes the correlation 0/0; NaN is the right answer and is reported
        # as such rather than as a numpy warning per split.
        warnings.simplefilter("ignore")
        r_per_split = np.array(
            [compare_vectors(a, b, comparator) for a, b in pairs]
        )

    r_hh = fisher_mean(r_per_split, comparator) if np.all(np.isfinite(r_per_split)) else np.nan
    r_sb_per_split = spearman_brown(r_per_split)
    # The splits are not independent samples (they reuse the same repeats),
    # so this interval describes the spread across splits, not sampling
    # error in the underlying reliability. Reported as such.
    ci = (
        tuple(np.percentile(r_sb_per_split, percentiles))
        if len(pairs) > 1 and np.all(np.isfinite(r_sb_per_split))
        else (np.nan, np.nan)
    )
    return {
        "r_hh": r_hh,
        "r_sb": float(spearman_brown(r_hh)),
        "r_hh_per_split": r_per_split,
        "r_sb_per_split": r_sb_per_split,
        "ci": ci,
        "n_splits": len(pairs),
        "split_mode": split_mode,
    }


# --------------------------------------------------------------------------
# From reliability to a ceiling (RETIRED), and from a ceiling to an
# achievement rate (still in use, now against the Nili bounds)
# --------------------------------------------------------------------------

def attenuation_ceiling(r_sb_brain, r_sb_model=None):
    """Highest correlation a perfect model could reach, given the noise.

    The square root is the whole point: split-half reliability bounds the
    *explained variance*, so a bound on r is its square root. Comparing an
    observed r directly against r_sb understates the ceiling and flatters
    the model.

    Never apply this to Nili's bounds -- those are already bounds on the
    correlation, and taking their square root corrects a second time.

    RETIRED along with split_half_reliability: nothing in the pipeline
    calls this any more, because the ceiling is now Nili's bound.
    """
    def _sqrt(value):
        value = float(value)
        return float(np.sqrt(value)) if value > 0 else np.nan

    one_sided = _sqrt(r_sb_brain)
    two_sided = (
        np.nan if r_sb_model is None
        else _sqrt(float(r_sb_brain) * float(r_sb_model))
    )
    return {
        "ceiling_one_sided": one_sided,
        "ceiling_two_sided": two_sided,
        "r_sb_brain": float(r_sb_brain),
        "r_sb_model": np.nan if r_sb_model is None else float(r_sb_model),
    }


def l1_baseline(u_brain, u_model):
    """Uncentered correlation two unrelated non-negative vectors already share.

    L1 compares distances through the origin, so any two RDMs with small
    coefficients of variation agree at r0 close to 1 before any shared
    structure is involved. The coefficients are taken straight from the data
    -- never back-derived from an eps value, which would make the baseline a
    function of the very quantity it is meant to calibrate.
    """
    def _cv(u):
        u = np.asarray(u, dtype=float)
        mean = u.mean()
        return np.nan if mean == 0 else u.std(ddof=1) / mean

    cv_brain, cv_model = _cv(u_brain), _cv(u_model)
    baseline = (1 / np.sqrt(1 + cv_brain**2)) * (1 / np.sqrt(1 + cv_model**2))
    return {"baseline": float(baseline), "cv_brain": float(cv_brain), "cv_model": float(cv_model)}


def achievement(score, ceiling, baseline=None, degenerate_eps=1e-6):
    """Observed score as a fraction of the ceiling.

    Never clipped to 1. When the ceiling is low the ratio can legitimately
    exceed 1, and hiding that hides the fact that the denominator is the
    unreliable part of the estimate.

    With a baseline (L1 only) the rate is measured as the gain above what
    two unrelated RDMs already share. If the ceiling leaves no room above
    that baseline the rate is NaN and `degenerate` is set: not a failure,
    but the correct report that this rung is not testable here.
    """
    numerator = score if baseline is None else score - baseline
    denominator = ceiling if baseline is None else ceiling - baseline
    if not np.isfinite(denominator) or abs(denominator) < degenerate_eps:
        return {"achievement": np.nan, "degenerate": True}
    return {"achievement": float(numerator / denominator), "degenerate": False}
