"""`permutation_null` が参照実装と同じ帰無分布を返すことを固定する。

二段構えで見る。

  1. **基本ロジック** -- バッチ化が乗っている不変量そのものを直接テストする。
     「RDM の行と列を同時に置換しても上三角の値の多重集合は変わらない」が崩れ
     れば、順位・平均・ノルムを置換の外に出す最適化はすべて無効になる。結果だけ
     を突き合わせると、たまたま一致したのか不変量が成り立っているのかが区別で
     きない。
  2. **結果の一致** -- 同じ seed で参照実装と要素ごとに一致すること。分布が同じ
     ではなく、同じ引き（draw）に対して同じ値であることを要求する。

参照実装 `permutation_null_reference` は速度のためにパイプラインからは外して
あるが、この定義に対して縛るために残してある。消さないこと。
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import noise_ceiling as nc
import run_noise_ceiling as rnc
from compare_rdm_folders import permuted_upper
from compare_rdms import load_rdm

COMPARATORS = ["cosine", "corr", "spearman"]
RDM_DIR = Path(__file__).resolve().parents[1] / "output" / "things"
MODEL_DIR = Path(__file__).resolve().parents[1] / "model_output"


def symmetric(rng, n, values=None):
    m = rng.random((n, n)) if values is None else values
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m


@pytest.fixture(scope="module")
def real_stack():
    """実データ 3 被験者ぶんと、それに合わせたモデル RDM。"""
    mats = [load_rdm(RDM_DIR / f"sub-{s}_test_Early_VC_crossnobis_rdm.csv")
            for s in ("01", "02", "03")]
    model = load_rdm(MODEL_DIR / "weights_sorted_fMRI1_test_100_choice_probability_rdm.csv")
    common = sorted(set(mats[0].index) & set(model.index))
    stack = [m.loc[common, common].to_numpy() for m in mats]
    mm = model.loc[common, common].to_numpy()
    return stack, mm[np.triu_indices(len(common), 1)]


def reference(u_model, stack, comparator, n_perm, seed):
    """パイプラインが以前使っていた形。置換ごとに rsatoolbox を呼ぶ。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rnc.permutation_null_reference(
            lambda uppers: float(np.mean([
                nc.compare_vectors(u_model, u, comparator) for u in uppers
            ])),
            stack, n_perm, seed,
        )


# --------------------------------------------------------------- 1. 基本ロジック


@pytest.mark.parametrize("n", [7, 40])
def test_permutation_preserves_the_multiset_of_dissimilarities(n):
    """行と列を同時に置換しても上三角の値の集合は変わらない。

    バッチ化の全体がこの一点に乗っている。ここが崩れたら、順位・平均・ノルムを
    置換の外に出すことは一切できない。
    """
    rng = np.random.default_rng(0)
    m = symmetric(rng, n)
    iu = np.triu_indices(n, k=1)
    base = m[iu]
    for _ in range(20):
        perm = rng.permutation(n)
        assert np.array_equal(np.sort(permuted_upper(m, perm, iu)), np.sort(base))


@pytest.mark.parametrize("n", [7, 40])
def test_ranks_commute_with_permutation(n):
    """順位付けは置換と可換。だから rankdata は RDM ごとに 1 回でよい。

    `compare_spearman` は置換ごとに 4950 個を順位付けし直していた。これが最適化
    前の最大の費用で、この可換性がその削除の根拠になっている。
    """
    rng = np.random.default_rng(1)
    for values in (None, rng.integers(0, 3, size=(n, n)).astype(float)):   # 同値あり
        m = symmetric(rng, n, None if values is None else values)
        iu = np.triu_indices(n, k=1)
        ranked = np.zeros((n, n))
        ranked[iu] = scipy.stats.rankdata(m[iu])
        ranked += ranked.T
        for _ in range(20):
            perm = rng.permutation(n)
            assert np.array_equal(
                permuted_upper(ranked, perm, iu),
                scipy.stats.rankdata(permuted_upper(m, perm, iu)),
            )


@pytest.mark.parametrize("n", [7, 40])
def test_mean_and_norm_are_permutation_invariant(n):
    """平均とノルムが定数だから、余弦の分母を置換の外に出せる。"""
    rng = np.random.default_rng(2)
    m = symmetric(rng, n)
    iu = np.triu_indices(n, k=1)
    base = m[iu]
    for _ in range(20):
        v = permuted_upper(m, rng.permutation(n), iu)
        assert v.mean() == pytest.approx(base.mean(), abs=1e-15)
        assert np.linalg.norm(v - v.mean()) == pytest.approx(
            np.linalg.norm(base - base.mean()), abs=1e-13)


@pytest.mark.parametrize("comparator", COMPARATORS)
def test_comparator_vector_matches_rsatoolbox(comparator):
    """`_comparator_vector` は rsatoolbox の前処理と同じでなければならない。

    前処理を取り違えると（たとえば cosine を中心化してしまうと）、帰無分布は
    それらしい値のまま静かに別の量になる。
    """
    rng = np.random.default_rng(3)
    n = 30
    iu = np.triu_indices(n, k=1)
    a, b = symmetric(rng, n)[iu], symmetric(rng, n)[iu]
    va, vb = rnc._comparator_vector(a, comparator), rnc._comparator_vector(b, comparator)
    manual = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    assert manual == pytest.approx(
        nc.compare_vectors(a, b, comparator), abs=1e-12)


def test_draws_are_the_same_sequence_not_just_the_same_distribution():
    """seed から引かれる置換列そのものが、以前の実装と同じ順序であること。"""
    n, n_perm, seed = 12, 25, 987
    rng = np.random.default_rng(seed)
    expected = [rng.permutation(n) for _ in range(n_perm)]     # 旧実装と同じ引き方
    got = rnc.draw_permutations(n, n_perm, seed)
    assert len(got) == n_perm
    for a, b in zip(got, expected):
        assert np.array_equal(a, b)


# --------------------------------------------------------------- 2. 結果の一致


@pytest.mark.parametrize("comparator", COMPARATORS)
@pytest.mark.parametrize("n_subjects", [1, 3], ids=["subject", "group"])
def test_matches_reference_on_real_data(real_stack, comparator, n_subjects):
    stack, u_model = real_stack
    stack = stack[:n_subjects]
    fast = rnc.permutation_null(u_model, stack, comparator, 300, 4242)
    slow = reference(u_model, stack, comparator, 300, 4242)
    assert fast.shape == slow.shape
    assert np.abs(fast - slow).max() < 1e-12

    # p 値は帰無分布の順序統計量なので、そこまで一致していることを直接見る
    for q in (5, 50, 95):
        obs = float(np.percentile(slow, q))
        assert (1 + (fast >= obs).sum()) == (1 + (slow >= obs).sum())


@pytest.mark.parametrize("comparator", COMPARATORS)
def test_matches_reference_on_awkward_inputs(comparator):
    """同値・負値・定数ベクトルでも参照実装と一致すること。"""
    rng = np.random.default_rng(5)
    n = 25
    cases = {
        "ties": symmetric(rng, n, rng.integers(0, 4, size=(n, n)).astype(float)),
        "negative": symmetric(rng, n, rng.normal(size=(n, n))),
        "constant": symmetric(rng, n, np.ones((n, n))),
    }
    u_model = symmetric(rng, n)[np.triu_indices(n, 1)]
    for name, m in cases.items():
        fast = rnc.permutation_null(u_model, [m], comparator, 200, 7)
        slow = reference(u_model, [m], comparator, 200, 7)
        assert np.abs(fast - slow).max() < 1e-12, name


def test_zero_permutations_returns_none(real_stack):
    stack, u_model = real_stack
    assert rnc.permutation_null(u_model, stack, "corr", 0, 1) is None


def test_chunking_does_not_change_the_answer(real_stack):
    """チャンク幅は性能だけの調整で、値に触ってはいけない。"""
    stack, u_model = real_stack
    ref = rnc.permutation_null(u_model, stack, "spearman", 500, 3, chunk=4096)
    for chunk in (1, 7, 128, 100000):
        got = rnc.permutation_null(u_model, stack, "spearman", 500, 3, chunk=chunk)
        assert np.abs(got - ref).max() < 1e-15, chunk


def test_unknown_comparator_raises(real_stack):
    stack, u_model = real_stack
    with pytest.raises(ValueError):
        rnc.permutation_null(u_model, stack, "kendall", 10, 1)
