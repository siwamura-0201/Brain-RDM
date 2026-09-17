"""Acceptance tests for the noise-ceiling implementation.

Tests 5, 6, 8 and 13 are the point of this file: they pin down the four
mistakes that make a ceiling silently wrong rather than obviously broken
(skipping the pre-average normalization, dropping the square root, scoring
against the averaged RDM, and mixing similarity RDMs in with distances).
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import noise_ceiling as nc
from extract_patterns import cache_path, load_cached_patterns
from rdm_metrics import METRICS, TRIAL_METRICS

ROOT = Path(__file__).resolve().parent.parent


def symmetric_rdm(rng, n=20):
    """A random matrix with an RDM's shape: symmetric, zero diagonal."""
    a = rng.normal(size=(n, n))
    a = np.abs(a + a.T)
    np.fill_diagonal(a, 0)
    return a


def patterns_from_signal(rng, signal, n_repeats, noise):
    """Repeats of one subject: a fixed signal plus independent noise."""
    return signal + noise * rng.normal(size=(n_repeats,) + signal.shape)


# --- 1-2: Nili bounds at the two extremes ---------------------------------

def test_identical_subjects_give_bounds_of_one():
    rdm = symmetric_rdm(np.random.default_rng(0))
    for comparator in nc.LEVELS:
        result = nc.nili_ceiling(np.stack([rdm] * 3), comparator)
        assert result["upper"] == pytest.approx(1.0)
        assert result["lower"] == pytest.approx(1.0)


def test_independent_subjects_give_bounds_near_zero():
    rng = np.random.default_rng(1)
    rdms = np.stack([symmetric_rdm(rng, n=60) for _ in range(8)])
    result = nc.nili_ceiling(rdms, "corr")
    assert abs(result["lower"]) < 0.2
    assert result["lower"] < result["upper"]


# --- 3-4: split-half reliability against known noise ----------------------

def test_noiseless_data_is_perfectly_reliable():
    rng = np.random.default_rng(2)
    signal = rng.normal(size=(10, 30))
    patterns = np.repeat(signal[None], 6, axis=0)
    result = nc.split_half_reliability(patterns, "correlation", "corr")
    assert result["r_hh"] == pytest.approx(1.0)
    assert result["r_sb"] == pytest.approx(1.0)


def test_reliability_decreases_with_noise():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(30, 40))
    values = [
        nc.split_half_reliability(
            patterns_from_signal(rng, signal, 6, noise), "correlation", "corr"
        )["r_sb"]
        for noise in (0.25, 0.5, 1.0, 2.0, 4.0)
    ]
    assert values == sorted(values, reverse=True)


# --- 5: the normalization that makes the bounds valid ---------------------

def test_comparator_changes_the_nili_bounds():
    """If pearson and spearman give the same bounds, the per-comparator
    normalization before averaging was skipped."""
    rng = np.random.default_rng(4)
    rdms = np.stack([symmetric_rdm(rng, n=40) ** 3 for _ in range(4)])
    pearson = nc.nili_ceiling(rdms, "corr")
    spearman = nc.nili_ceiling(rdms, "spearman")
    assert pearson["upper"] != pytest.approx(spearman["upper"])
    assert pearson["lower"] != pytest.approx(spearman["lower"])


# --- 6: the square root that makes the ceiling comparable to r ------------

def test_ceiling_is_the_square_root_of_reliability():
    for r_sb in (0.04, 0.25, 0.36, 0.81):
        result = nc.attenuation_ceiling(r_sb, r_sb_model=1.0)
        assert result["ceiling_one_sided"] == pytest.approx(np.sqrt(r_sb))
        # With the SPoSE side treated as noise-free (r_MM = 1) the two
        # positions coincide; they must not silently differ.
        assert result["ceiling_two_sided"] == pytest.approx(np.sqrt(r_sb))


def test_nili_bounds_are_not_square_rooted():
    """Nili's bounds are bounds on r already; attenuation_ceiling must not
    be reachable from the Nili path."""
    rng = np.random.default_rng(5)
    rdms = np.stack([symmetric_rdm(rng, n=30) for _ in range(4)])
    result = nc.nili_ceiling(rdms, "corr")
    assert set(result) == {
        "upper", "lower", "upper_per_subject", "lower_per_subject",
        "n_subjects", "crossed",
    }


# --- 7: the two ceilings must order the right way around ------------------

def test_split_half_ceiling_exceeds_nili_lower_bound():
    """Within-subject reliability excludes between-subject variance, so the
    ceiling it implies must sit above the leave-one-subject-out bound.

    Against Nili's *upper* bound the comparison is not one-sided and must
    not be asserted: each subject there is correlated with an average it
    contributes 1/S of, which inflates the bound -- badly so at S = 3. On
    the real data the split-half ceiling lands above nili_lower in every
    cell but above nili_upper in only about 40% of them, which is the
    expected pattern and not a bug.
    """
    rng = np.random.default_rng(6)
    shared = rng.normal(size=(25, 50))
    subjects = [shared + 0.8 * rng.normal(size=shared.shape) for _ in range(3)]
    patterns = [patterns_from_signal(rng, s, 6, 0.3) for s in subjects]

    rdms = np.stack([METRICS["correlation"](p.mean(axis=0)) for p in patterns])
    nili = nc.nili_ceiling(rdms, "corr")
    split = nc.split_half_reliability(patterns[0], "correlation", "corr")
    ceiling = nc.attenuation_ceiling(split["r_sb"], 1.0)["ceiling_one_sided"]

    assert ceiling > nili["lower"]


# --- 8: the score shape the ceilings apply to -----------------------------

def test_model_score_rejects_an_averaged_rdm():
    rng = np.random.default_rng(7)
    rdms = np.stack([symmetric_rdm(rng, n=20) for _ in range(3)])
    model = symmetric_rdm(rng, n=20)

    with pytest.raises(ValueError, match="per-subject"):
        nc.model_score(model, rdms.mean(axis=0), "corr")

    result = nc.model_score(model, rdms, "corr")
    assert result["score"] == pytest.approx(result["score_per_subject"].mean())


# --- 9: reproducibility ---------------------------------------------------

def test_results_are_reproducible():
    rng = np.random.default_rng(8)
    signal = rng.normal(size=(20, 30))
    patterns = patterns_from_signal(rng, signal, 6, 1.0)
    rdms = np.stack([symmetric_rdm(np.random.default_rng(i), n=20) for i in range(3)])

    first = nc.split_half_reliability(patterns, "correlation", "spearman")
    second = nc.split_half_reliability(patterns, "correlation", "spearman")
    assert first["r_sb"] == second["r_sb"]
    np.testing.assert_array_equal(first["r_hh_per_split"], second["r_hh_per_split"])

    boot_a = nc.nili_ceiling_bootstrap(rdms, "corr", n_boot=50, seed=0)
    boot_b = nc.nili_ceiling_bootstrap(rdms, "corr", n_boot=50, seed=0)
    assert boot_a["upper_ci"] == boot_b["upper_ci"]


# --- 10: L1 degeneracy ----------------------------------------------------

def test_l1_is_degenerate_when_dissimilarities_barely_vary():
    rng = np.random.default_rng(9)
    # Coefficient of variation ~1e-6: every pair is at almost the same
    # distance, so r0 sits at 1 whatever the structure is.
    u_brain = 1.0 + 1e-6 * rng.normal(size=4950)
    u_model = 1.0 + 1e-6 * rng.normal(size=4950)

    baseline = nc.l1_baseline(u_brain, u_model)
    assert baseline["baseline"] == pytest.approx(1.0, abs=1e-9)

    result = nc.achievement(0.999999, ceiling=1.0, baseline=baseline["baseline"])
    assert result["degenerate"] is True
    assert np.isnan(result["achievement"])


def test_achievement_is_not_clipped():
    result = nc.achievement(0.4, ceiling=0.25)
    assert result["achievement"] == pytest.approx(1.6)


# --- 11: the split enumeration -------------------------------------------

def test_balanced_splits_enumerate_each_split_once():
    n = 12
    splits = nc.balanced_splits(n, "all")
    assert len(splits) == 462

    seen = set()
    for a, b in splits:
        assert len(a) == len(b) == n // 2
        assert sorted(np.concatenate([a, b])) == list(range(n))
        key = frozenset([frozenset(a.tolist()), frozenset(b.tolist())])
        assert key not in seen, "a split and its complement were both counted"
        seen.add(key)
    assert len(seen) == len(list(itertools.combinations(range(n), n // 2))) // 2


def test_balanced_splits_reject_odd_repeat_counts():
    with pytest.raises(ValueError, match="even number of repeats"):
        nc.balanced_splits(11, "all")


# --- 12: the cache really is the same data the RDMs were built from -------

@pytest.mark.parametrize("metric", ["correlation", "euclidean_centered"])
def test_cache_matches_averaged_patterns(metric):
    path = ROOT / cache_path("01", "test", "LOC")
    rdm_path = ROOT / "output/things/sub-01_test_LOC_{}_rdm.csv".format(metric)
    if not path.exists() or not rdm_path.exists():
        pytest.skip("run extract_patterns.py / get_brainRDM-things.py first")

    patterns, stimuli, _ = load_cached_patterns(path)
    expected = pd.read_csv(rdm_path, index_col=0)

    assert stimuli == list(expected.index)
    # Not bit-exact only because the RDM went through a csv round trip.
    np.testing.assert_allclose(
        METRICS[metric](patterns.mean(axis=0)), expected.to_numpy(), rtol=1e-9, atol=1e-12
    )


# --- 13: distances only ---------------------------------------------------

def test_similarity_metrics_are_rejected():
    rng = np.random.default_rng(10)
    patterns = rng.normal(size=(4, 10, 20))
    with pytest.raises(ValueError, match="similarity"):
        nc.split_half_reliability(patterns, "pearson", "corr")


def test_unknown_comparator_is_rejected():
    rng = np.random.default_rng(11)
    rdms = np.stack([symmetric_rdm(rng, n=10) for _ in range(3)])
    with pytest.raises(ValueError, match="no ceiling is defined"):
        nc.nili_ceiling(rdms, "kendall")


# --- crossnobis and the pattern bootstrap --------------------------------

def test_crossnobis_can_go_negative_and_survives_a_csv_round_trip(tmp_path):
    """crossnobis is unbiased, so it crosses zero where the true distance is
    small. Nothing downstream may assume non-negative dissimilarities."""
    rng = np.random.default_rng(20)
    # No stimulus structure at all: every true distance is zero, so roughly
    # half the estimates must come out below it.
    patterns = rng.normal(size=(6, 12, 40))
    matrix = TRIAL_METRICS["crossnobis"](patterns)

    upper = nc.upper_triangle(matrix)
    assert (upper < 0).any(), "crossnobis on pure noise produced no negative cells"
    np.testing.assert_allclose(np.diag(matrix), 0, atol=1e-12)

    path = tmp_path / "rdm.csv"
    labels = [f"s{i:02d}" for i in range(len(matrix))]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)
    np.testing.assert_allclose(
        pd.read_csv(path, index_col=0).to_numpy(), matrix, rtol=1e-12
    )


def test_crossnobis_is_accepted_as_a_distance():
    """The similarity guard is name-based, so a negative-valued distance
    metric must pass it."""
    assert nc.check_distance_metric("crossnobis") is TRIAL_METRICS["crossnobis"]


def test_bootstrap_resamples_stimuli_not_subjects():
    """At S = 3 a subject bootstrap can only take a handful of values and
    every repeated draw pushes the bounds to 1. The pattern bootstrap has
    100 entries to work with, so its interval is informative."""
    rng = np.random.default_rng(21)
    shared = rng.normal(size=(40, 60))
    rdms = np.stack([
        METRICS["correlation"](shared + 0.9 * rng.normal(size=shared.shape))
        for _ in range(3)
    ])
    result = nc.nili_ceiling_bootstrap(rdms, "corr", n_boot=100, seed=0)
    assert result["bootstrap"] == "pattern"

    lo, hi = result["lower_ci"]
    point = nc.nili_ceiling(rdms, "corr")["lower"]
    assert lo < point < hi, "the interval should straddle the point estimate"
    assert hi < 1.0, "a pattern bootstrap must not pile up at 1 the way the subject one did"
