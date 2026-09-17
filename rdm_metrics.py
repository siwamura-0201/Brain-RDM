"""RDM cell metrics, shared by every script that turns voxel patterns into an RDM.

Split out of get_brainRDM-things.py so that modules which cannot import
that file (its name contains a hyphen) -- noise_ceiling.py in particular --
compute their RDMs with exactly the same functions, rather than a second
copy that could drift out of step.

Two registries, because two kinds of metric need different inputs:

  METRICS        (stimuli, voxels)          -- one pattern per stimulus
  TRIAL_METRICS  (repeats, stimuli, voxels) -- the individual repeats, which
                                               crossnobis needs in order to
                                               cross-validate across them

`mahalanobis` used to live here and was removed: estimating the voxel
covariance from the same 100 stimulus patterns whose distances it then
measured made the covariance rank-deficient, and its pseudo-inverse
collapsed every off-diagonal cell onto the constant sqrt(2(M-1)). The RDM
carried no information at all. `crossnobis` is the corrected estimator --
noise covariance from trial-level residuals with shrinkage, cross-validated
across sessions.
"""

import numpy as np
from rsatoolbox.data import Dataset, prec_from_residuals
from rsatoolbox.rdm import calc_rdm
from scipy.spatial.distance import pdist, squareform

METRICS = {}
TRIAL_METRICS = {}

# Metrics that are similarities bounded in [-1, 1] (plotted with a
# diverging colormap on a fixed range). Everything else is treated as a
# distance and plotted on an auto-scaled range starting at 0.
#
# This is a list of names on purpose, not a test on the values: crossnobis
# is an unbiased estimator and is *expected* to go negative where the true
# distance is near zero, so "has negative cells" cannot be used to tell a
# similarity from a distance.
SIMILARITY_METRICS = {"pearson"}


def register_metric(name):
    def decorator(fn):
        METRICS[name] = fn
        return fn

    return decorator


def register_trial_metric(name):
    def decorator(fn):
        TRIAL_METRICS[name] = fn
        return fn

    return decorator


@register_metric("pearson")
def pearson_similarity(patterns):
    return np.corrcoef(patterns)


@register_metric("correlation")
def correlation_distance(patterns):
    """Correlation distance: 1 - Pearson correlation between patterns.

    Delegates to rsatoolbox, which uses the same 1 - r convention; the two
    agree to 3e-16 (tests/test_rsatoolbox_parity.py).
    """
    return _calc_rdm_matrix(patterns, method="correlation")


@register_metric("euclidean_centered")
def mean_removed_euclidean_distance(patterns):
    """Euclidean distance after removing each pattern's own mean activation
    (mean removal only -- no variance normalization, unlike z-scoring).

    Kept as a local implementation rather than rsatoolbox's
    `euclidean, remove_mean=True`: that one returns the *squared* distance
    divided by the number of channels. Rank-based comparators would not
    notice, but every Pearson (L2) value would change.
    """
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    return squareform(pdist(centered, metric="euclidean"))


@register_trial_metric("crossnobis")
def crossnobis_distance(patterns):
    """Cross-validated Mahalanobis distance from trial-level patterns.

    `patterns` is (repeats, stimuli, voxels) -- the cache
    extract_patterns.py writes. In the THINGS test set one repeat is one
    session, so sessions are the cross-validation folds.

    The noise precision comes from the trial-level residuals (each trial
    minus its own stimulus mean, so 1200 - 100 = 1100 degrees of freedom)
    with rsatoolbox's diagonal shrinkage. That is the estimate the removed
    `mahalanobis` never had: it read the covariance off the stimulus means
    themselves, which is exactly where the signal lives.

    rsatoolbox's calc_rdm sorts its output by the stimulus descriptor, so
    the descriptor is a zero-padded index rather than the stimulus name --
    sorting it alphabetically then reproduces the caller's order, whatever
    that order was, and the assert below holds it to that.
    """
    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim != 3:
        raise ValueError(
            f"crossnobis needs (repeats, stimuli, voxels); got {patterns.shape}"
        )
    n_repeats, n_stimuli, _ = patterns.shape
    if n_repeats < 2:
        raise ValueError("crossnobis needs at least 2 repeats to cross-validate")

    keys = np.array([f"{i:06d}" for i in range(n_stimuli)])
    residuals = (patterns - patterns.mean(axis=0, keepdims=True)).reshape(
        n_repeats * n_stimuli, -1
    )
    precision = prec_from_residuals(
        residuals, dof=n_repeats * n_stimuli - n_stimuli, method="shrinkage_diag"
    )

    dataset = Dataset(
        measurements=patterns.reshape(n_repeats * n_stimuli, -1),
        obs_descriptors={
            "stimulus": np.tile(keys, n_repeats),
            "session": np.repeat(np.arange(n_repeats), n_stimuli),
        },
    )
    rdm = calc_rdm(
        dataset, method="crossnobis", descriptor="stimulus",
        cv_descriptor="session", noise=precision,
    )
    assert list(rdm.pattern_descriptors["stimulus"]) == list(keys), (
        "calc_rdm reordered the stimuli; the zero-padded keys should have "
        "made its alphabetical sort a no-op"
    )
    return rdm.get_matrices()[0]


def _calc_rdm_matrix(patterns, method):
    patterns = np.asarray(patterns, dtype=float)
    keys = np.array([f"{i:06d}" for i in range(len(patterns))])
    dataset = Dataset(measurements=patterns, obs_descriptors={"stimulus": keys})
    rdm = calc_rdm(dataset, method=method, descriptor="stimulus")
    assert list(rdm.pattern_descriptors["stimulus"]) == list(keys)
    return rdm.get_matrices()[0]
