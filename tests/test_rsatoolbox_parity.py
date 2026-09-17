"""Holds the local implementation to rsatoolbox, so drift is noticed.

Everything here compares our code against the upstream it was ported from.
If one of these fails after an rsatoolbox upgrade, the local code is the
thing to re-read -- not the tolerance.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from rsatoolbox.inference import boot_noise_ceiling
from rsatoolbox.rdm import RDMs, compare

import noise_ceiling as nc
from compare_rdm_folders import STATISTICS
from compare_rdms import check_axis_order, load_rdm
from rdm_metrics import METRICS

ROOT = Path(__file__).resolve().parent.parent
ROIS = ["Early_VC", "LOC", "FFA", "PPA"]
METRIC_NAMES = ["correlation", "euclidean_centered"]
COMPARATORS = ["cosine", "corr", "spearman"]


def subject_stack(roi, metric):
    paths = [
        ROOT / f"output/things/sub-{s}_test_{roi}_{metric}_rdm.csv"
        for s in ("01", "02", "03")
    ]
    if not all(p.exists() for p in paths):
        pytest.skip("run get_brainRDM-things.py for all three subjects first")
    frames = [load_rdm(p) for p in paths]
    for frame, path in zip(frames, paths):
        check_axis_order(frame, path)
    assert list(frames[0].index) == list(frames[1].index) == list(frames[2].index)
    return np.stack([f.to_numpy() for f in frames])


@pytest.mark.parametrize("roi", ROIS)
@pytest.mark.parametrize("metric", METRIC_NAMES)
@pytest.mark.parametrize("comparator", COMPARATORS)
def test_nili_ceiling_matches_boot_noise_ceiling(roi, metric, comparator):
    """Our bounds are rsatoolbox's, with the per-subject values kept.

    boot_noise_ceiling returns (lower, upper) -- the opposite order to
    nili_ceiling's keys, which is exactly the kind of thing this test is
    here to catch.
    """
    stack = subject_stack(roi, metric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = nc.nili_ceiling(stack, comparator)
        lower, upper = boot_noise_ceiling(nc._stack_to_rdms(stack), method=comparator)

    assert ours["upper"] == pytest.approx(upper, rel=1e-9, abs=1e-12)
    assert ours["lower"] == pytest.approx(lower, rel=1e-9, abs=1e-12)
    assert ours["upper"] == pytest.approx(ours["upper_per_subject"].mean())
    assert ours["lower"] == pytest.approx(ours["lower_per_subject"].mean())


def test_l1_r0_is_rsatoolbox_cosine():
    """The L1 rung's statistic already exists upstream under another name.

    `l1_r0` is uncentered correlation, which is cosine similarity, which is
    rsatoolbox's 'cosine'. That identity is what gives L1 a ceiling at all:
    pool_rdm supports 'cosine' (RMS-normalise, then average), so the L1
    bound is as well-founded as L2's.
    """
    rng = np.random.default_rng(0)
    for _ in range(5):
        x, y = np.abs(rng.normal(size=4950)), np.abs(rng.normal(size=4950))
        assert STATISTICS["l1_r0"](x, y) == pytest.approx(
            nc.compare_vectors(x, y, "cosine"), rel=1e-12, abs=1e-14
        )


def test_correlation_metric_matches_numpy():
    """Switching `correlation` to rsatoolbox must not move any number."""
    rng = np.random.default_rng(1)
    patterns = rng.normal(size=(100, 300))
    np.testing.assert_allclose(
        METRICS["correlation"](patterns), 1 - np.corrcoef(patterns),
        rtol=0, atol=1e-12,
    )


def test_spearman_and_rho_a_agree_without_ties():
    """Why `rho-a` is not in LEVELS.

    With no tied dissimilarities the two are the same number, so offering
    both would duplicate a column rather than add a robustness check. This
    test records the fact; if a future RDM does carry ties it will fail and
    the choice can be revisited.
    """
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(1, 4950)), rng.normal(size=(1, 4950))
    spearman = float(compare(RDMs(x), RDMs(y), "spearman"))
    rho_a = float(compare(RDMs(x), RDMs(y), "rho-a"))
    assert spearman == pytest.approx(rho_a, abs=1e-12)


def test_rho_a_penalises_ties():
    """...and why the two are not the same statistic in general."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 4950))
    y = np.round(rng.normal(size=(1, 4950)), 1)  # plenty of ties
    assert abs(float(compare(RDMs(x), RDMs(y), "rho-a"))) < abs(
        float(compare(RDMs(x), RDMs(y), "spearman"))
    )


def test_kendall_ceiling_is_truncated_upstream():
    """The reason `kendall` carries no ceiling here.

    pool_rdm stops at the averaged-rank RDM for tau and says so. The
    MATLAB toolbox's iterative optimisation was never ported, so the bound
    comes out too low and the achievement rate too high.
    """
    from rsatoolbox.util.inference_util import pool_rdm

    rng = np.random.default_rng(4)
    rdms = RDMs(rng.normal(size=(3, 4950)))
    with pytest.warns(UserWarning, match="averaged ranks"):
        pool_rdm(rdms, method="tau-b")
    assert "kendall" not in nc.LEVELS and "tau-b" not in nc.LEVELS
