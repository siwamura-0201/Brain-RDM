"""仕様書 §9 の受け入れテスト。

中心は 5（符号正規化）と 6（帯の x 範囲）。それぞれ最も踏みやすい誤りを突く。
"""

import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_ceilings as pc
from plot_ceilings import (
    BRAIN_METRICS,
    COMPARATORS,
    MODEL_METRICS,
    ComparatorSpec,
    FigureConfig,
    MetricSpec,
    build_bars,
    is_licensed,
    load,
    parse_model_metric,
    plot_main_figure,
    sig_mark,
)

CSV = Path(__file__).resolve().parents[1] / "results" / "noise_ceiling" / "ceilings.csv"


@pytest.fixture(scope="module")
def df():
    return load(CSV)


@pytest.fixture(scope="module")
def n_perm(df):
    """置換回数は CSV が正。上流を回し直すとここが動く。"""
    return int(df["n_perm"].dropna().unique()[0])


@pytest.fixture
def registries():
    """レジストリを書き換えるテストのために、前後で元に戻す。"""
    saved = (dict(BRAIN_METRICS), dict(MODEL_METRICS), dict(COMPARATORS))
    yield
    for registry, backup in zip((BRAIN_METRICS, MODEL_METRICS, COMPARATORS), saved):
        registry.clear()
        registry.update(backup)


def _bars(df, **kw):
    return build_bars(df, FigureConfig(**kw))


# ------------------------------------------------------------------ 1 / 2 / 3


def test_1_adding_a_model_metric_adds_a_bar(df, registries):
    cfg = dict(comparator="spearman")
    before, _ = _bars(df, **cfg)
    w_before = plot_main_figure(df, FigureConfig(**cfg)).get_figwidth()

    MODEL_METRICS["euclidean_clone"] = MetricSpec("euclidean_clone", "distance", "L1")
    d = df.copy()
    extra = d[d.model_metric == "euclidean"].copy()
    extra["model_metric"] = "euclidean_clone"
    d = pd.concat([d, extra], ignore_index=True)

    after, meta = _bars(d, **cfg)
    assert len(meta["model_metrics"]) == 5
    assert len(after) == len(before) + len(before) // 4
    assert plot_main_figure(d, FigureConfig(**cfg)).get_figwidth() > w_before


def test_2_brain_metric_adds_a_block_and_a_band(df):
    """crossnobis は include=True で既にレジストリにある。落とすとブロックが減る。"""
    with_it, meta_with = _bars(df, comparator="spearman")
    without, meta_without = _bars(
        df, comparator="spearman", brain_metrics=("correlation", "euclidean_centered"))
    assert len(meta_with["brain_metrics"]) == len(meta_without["brain_metrics"]) + 1
    n_bands_with = len({(b.roi, b.brain_metric, b.lower, b.upper) for b in with_it})
    n_bands_without = len({(b.roi, b.brain_metric, b.lower, b.upper) for b in without})
    assert n_bands_with > n_bands_without


def test_3_comparator_registry_drives_the_figure_set(registries):
    base = FigureConfig()
    n_before = len(pc.figure_set(base))
    COMPARATORS["kendall"] = ComparatorSpec("Kendall τ", "L3")
    assert len(pc.figure_set(base)) == n_before + 1


# ------------------------------------------------------------------------ 4


def test_4_unlicensed_block_is_dropped_not_emptied(df, capsys):
    """corr (L2) は correlation (L3) に対して licensed でない。"""
    with pytest.raises(ValueError, match="licensed"):
        _bars(df, comparator="corr", brain_metrics=("correlation",), unlicensed="error")

    bars, meta = _bars(df, comparator="corr", brain_metrics=("correlation", "crossnobis"),
                       unlicensed="omit")
    assert "correlation" not in meta["brain_metrics"]      # ブロックごと落ちる
    assert meta["brain_metrics"] == ["crossnobis"]
    assert all(b.brain_metric == "crossnobis" for b in bars)
    assert "licensing" in capsys.readouterr().err


def test_4b_unknown_metric_raises_instead_of_being_dropped(df):
    with pytest.raises(ValueError, match="レジストリに無い"):
        _bars(df, comparator="spearman", brain_metrics=("no_such_metric",))


# ------------------------------------------------------------------------ 5


@pytest.mark.parametrize("model", ["dot_product", "choice_probability"])
def test_5_similarity_model_metric_bars_point_up(df, model):
    """類似度型 × 距離型は符号を反転して初めて「上が一致」になる。

    一致している ROI で棒が正に出ること、そして棒が生の値の符号反転そのもので
    あることの両方を見る。Early_VC は生の値が正（= 反一致）なので、反転後に負の
    棒になるのが正しい。ここを「全て正」で書くと、この図の所見そのものを誤りと
    判定してしまう。
    """
    assert MODEL_METRICS[model].kind == "similarity"
    bars, _ = _bars(df, comparator="spearman", model_metrics=(model,))
    raw = df[(df.row_type == "group_mean") & (df.comparator == "spearman")
             & (df.model_metric == model)].set_index(["roi", "brain_metric"])

    assert all(b.sign == -1 for b in bars)
    for b in bars:
        assert np.isclose(b.value, -raw.loc[(b.roi, b.brain_metric), "score_obs"])

    # LOC / PPA は 3 つの脳側計量すべてで一致が明確に出ている ROI。FFA の
    # correlation ブロックは生の値がほぼ 0 で符号が安定しないので外す。
    agree = [b for b in bars if b.roi in ("LOC", "PPA")]
    assert len(agree) == 6 and all(b.value > 0 for b in agree), "符号正規化が効いていない"


@pytest.mark.parametrize("model", ["cosine", "euclidean"])
def test_5a_distance_model_metric_is_not_flipped(df, model):
    assert MODEL_METRICS[model].kind == "distance"
    bars, _ = _bars(df, comparator="spearman", model_metrics=(model,))
    raw = df[(df.row_type == "group_mean") & (df.comparator == "spearman")
             & (df.model_metric == model)].set_index(["roi", "brain_metric"])
    assert all(b.sign == 1 for b in bars)
    for b in bars:
        assert np.isclose(b.value, raw.loc[(b.roi, b.brain_metric), "score_obs"])
    agree = [b for b in bars if b.roi in ("LOC", "PPA")]
    assert len(agree) == 6 and all(b.value > 0 for b in agree)


def test_5b_bands_are_not_sign_flipped(df):
    """帯は脳側 RDM どうしの比較。符号を一括で掛けてはいけない。"""
    flipped, _ = _bars(df, comparator="spearman", model_metrics=("dot_product",))
    kept, _ = _bars(df, comparator="spearman", model_metrics=("cosine",))
    for a, b in zip(flipped, kept):
        assert (a.roi, a.brain_metric) == (b.roi, b.brain_metric)
        assert np.isclose(a.lower, b.lower) and np.isclose(a.upper, b.upper)
        assert a.lower > 0 and a.upper > 0


def test_5c_transform_hits_bar_and_band_alike(df):
    """L1 (cosine) は l1_baseline を引く。棒だけに当てると図が破綻する。"""
    assert COMPARATORS["cosine"].transform == "baseline"
    bars, _ = _bars(df, comparator="cosine", unlicensed="mark")
    row = df[(df.row_type == "group_mean") & (df.comparator == "cosine")].iloc[0]
    b = next(x for x in bars if (x.roi, x.brain_metric, x.model_metric)
             == (row.roi, row.brain_metric, row.model_metric))
    sign = 1 if BRAIN_METRICS[row.brain_metric].kind == MODEL_METRICS[row.model_metric].kind else -1
    assert np.isclose(b.value, sign * (row.score_obs - row.l1_baseline))
    assert np.isclose(b.lower, row.nili_lower - row.l1_baseline)
    assert np.isclose(b.upper, row.nili_upper - row.l1_baseline)


def test_5d_missing_transform_column_raises(df, registries):
    COMPARATORS["spearman"] = replace(COMPARATORS["spearman"], transform="chance")
    with pytest.raises(ValueError, match="chance_level"):
        _bars(df, comparator="spearman")


# ------------------------------------------------------------------------ 6


def test_6_bands_stay_inside_their_block(df):
    """axhspan を使っていたら帯が軸幅いっぱいに伸びる。"""
    from matplotlib.patches import Rectangle

    cfg = FigureConfig(comparator="spearman")
    fig = plot_main_figure(df, cfg)
    ax = fig.axes[0]
    x0_ax, x1_ax = ax.get_xlim()
    _, meta = build_bars(df, cfg)
    n_bars, n_block = len(meta["model_metrics"]), len(meta["brain_metrics"])
    block_span = n_bars + pc.BLOCK_GAP_UNITS
    max_block_w = n_bars - 1 + 1.0 + 2 * pc.BLOCK_PAD

    bands = [p for p in ax.patches
             if isinstance(p, Rectangle) and p.get_facecolor()[:3]
             == matplotlib.colors.to_rgb(pc.BAND_FILL)]
    assert len(bands) == n_block
    for p in bands:
        assert p.get_width() <= max_block_w + 1e-9
        assert p.get_width() < (x1_ax - x0_ax) / 2, "軸幅いっぱい = axhspan"
        assert x0_ax <= p.get_x() and p.get_x() + p.get_width() <= x1_ax


def test_6b_pairwise_baseline_splits_the_band_per_bar(df):
    """L1 の基準値はモデル側にも依存するので、帯はブロック 1 本にまとめられない。"""
    bars, meta = _bars(df, comparator="cosine", unlicensed="mark")
    n_bars = len(meta["model_metrics"])
    block = [b for b in bars if b.roi == "LOC" and b.brain_metric == "correlation"]
    patches = pc._band_patches(block, 0, n_bars,
                               {(b.brain_metric, b.model_metric): i
                                for i, b in enumerate(block)})
    assert len(patches) == len(block) > 1
    assert all(w < 1.0 for _, w, _, _ in patches)


# ------------------------------------------------------------------------ 7


def test_7_significance_ladder():
    cfg = FigureConfig(sig_operator="le")
    assert [sig_mark(p, cfg) for p in (0.0005, 0.001, 0.008, 0.03, 0.2)] == \
        ["***", "***", "**", "*", "n.s."]
    lt = replace(cfg, sig_operator="lt")
    assert [sig_mark(p, lt) for p in (0.0005, 0.001, 0.008, 0.03, 0.2)] == \
        ["***", "**", "**", "*", "n.s."]


def test_7b_descending_sig_levels_raise(df):
    with pytest.raises(ValueError, match="昇順"):
        _bars(df, comparator="spearman",
              sig_levels=((0.05, "*"), (0.01, "**"), (0.001, "***")))


def test_7c_negative_bar_marks_sit_above_zero(df):
    """負の棒のアスタリスクを棒の下端に置くと軸外に出る。"""
    cfg = FigureConfig(comparator="spearman")
    fig = plot_main_figure(df, cfg)
    bars, _ = build_bars(df, cfg)
    assert any(b.value < 0 for b in bars), "前提が崩れている: 負の棒が無い"
    ymin, _ = fig.axes[0].get_ylim()
    marks = {"***", "**", "*", cfg.ns_label}
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text() in marks:
                assert t.get_position()[1] > 0, "0 線より下に置かれている"
                assert t.get_position()[1] > ymin


def test_7d_resolution_warning(df, capsys):
    """分解能の下限が最厳閾値以上なら警告する。判定は CSV の n_perm から。"""
    coarse = df.copy()
    coarse["n_perm"] = 100                     # 下限 = 1/101 ≈ 0.0099 > 0.001
    _bars(coarse, comparator="spearman")
    assert "分解能の下限" in capsys.readouterr().err

    _bars(df, comparator="spearman")           # 実データ
    err = capsys.readouterr().err
    assert ("分解能の下限" in err) == (pc.p_resolution(
        int(df["n_perm"].dropna().unique()[0])) >= 0.001)


def test_7d2_csv_n_perm_overrides_the_config(df, capsys, n_perm):
    """設定を正にすると、古い CSV を新しい既定で描いたとき脚注だけが嘘になる。"""
    _, meta = _bars(df, comparator="spearman", n_permutations=n_perm + 12345)
    assert meta["n_permutations"] == n_perm
    assert "食い違う" in capsys.readouterr().err

    ragged = df.copy()
    ragged.loc[ragged.index[ragged.comparator == "spearman"][0], "n_perm"] = 7
    with pytest.raises(ValueError, match="n_perm が揃っていない"):
        _bars(ragged, comparator="spearman")


def test_7e_mark_keeps_the_unlicensed_block(df):
    bars, meta = _bars(df, comparator="corr", unlicensed="mark")
    assert "correlation" in meta["brain_metrics"]
    corr_block = [b for b in bars if b.brain_metric == "correlation"]
    assert corr_block and all(not b.licensed for b in corr_block)
    assert meta["has_unlicensed"]


def test_7f_p_is_flipped_only_for_sign_flipped_pairs(df):
    bars, meta = _bars(df, comparator="spearman")
    floor = pc.p_resolution(meta["n_permutations"])
    for b in bars:
        expect = 1.0 - b.p_raw + floor if b.sign < 0 else b.p_raw
        assert np.isclose(b.p_eff, min(1.0, max(floor, expect)))
    off, _ = _bars(df, comparator="spearman", flip_p_for_similarity=False)
    assert all(b.p_eff == b.p_raw for b in off)


def test_7g_complement_p_never_returns_zero(df, n_perm):
    """素朴な 1-p は perm_p = 1.0 の行で 0 という有り得ない p を出す。"""
    assert pc.complement_p(1.0, 1000) == pytest.approx(pc.p_resolution(1000))
    assert pc.complement_p(0.5, 1000) == pytest.approx(0.5 + pc.p_resolution(1000))
    bars, _ = _bars(df, comparator="spearman")
    assert any(b.p_raw == 1.0 for b in bars), "前提が崩れている: perm_p = 1.0 の行が無い"
    # CSV 中の p は桁を落として書かれているので下限の比較には許容誤差を置く
    assert all(b.p_eff >= pc.p_resolution(n_perm) - 1e-12 for b in bars)


def test_7h_p_row_is_decimal_with_a_floor_abbreviation():
    """p 値行は小数表記。0.001 以下は "<0.001"、端数は四捨五入。"""
    cfg = FigureConfig()
    assert cfg.p_row_fmt == "decimal"
    f = lambda p: pc._fmt_p(p, cfg, 10000)

    assert f(0.0235) == "0.024"
    assert f(0.5) == "0.500"
    assert f(1.0) == "1.000"
    assert "e" not in f(0.0011), "指数表記が残っている"

    # 下限以下は略記。境界は「以下」
    for p in (9.999e-05, 0.0005, 0.001):
        assert f(p) == "<0.001", p
    assert f(0.0010999) == "0.001"

    # 切り捨てではなく四捨五入（切り捨てなら 0.001 になり実際より小さく見える）
    assert f(0.0015) == "0.002"
    assert f(0.0019) == "0.002"
    assert f(0.9994) == "0.999"

    trim = replace(cfg, p_row_fmt="trim")
    assert pc._fmt_p(0.5, trim, 10000) == ".500"
    assert pc._fmt_p(0.0005, trim, 10000) == "<.001"
    assert pc._fmt_p(1.0, trim, 10000) == "1.000"

    assert pc._fmt_p(0.0005, replace(cfg, p_row_fmt="sci"), 10000) == "5.0e-04"
    assert pc._fmt_p(float("nan"), cfg, 10000) == "—"


def test_7i_p_row_settings_are_configurable(df):
    cfg = replace(FigureConfig(), p_min_display=0.01, p_decimals=2)
    assert pc._fmt_p(0.004, cfg, 10000) == "<0.01"
    assert pc._fmt_p(0.0449, cfg, 10000) == "0.04"
    assert pc._fmt_p(0.045, cfg, 10000) == "0.05"
    with pytest.raises(ValueError, match="p_decimals"):
        _bars(df, comparator="spearman", p_decimals=0)
    with pytest.raises(ValueError, match="p_row_fmt"):
        _bars(df, comparator="spearman", p_row_fmt="exponent")


def test_7j_figure_p_row_has_no_scientific_notation(df):
    fig = plot_main_figure(df, FigureConfig(comparator="spearman"))
    rendered = [t.get_text() for ax in fig.axes for t in ax.texts]
    p_cells = [t for t in rendered if t.startswith(("0.", "<0.", "1.0"))]
    assert len(p_cells) >= 40, "p 値行が見つからない"
    assert not any("e-" in t or "e+" in t for t in p_cells)
    assert "<0.001" in p_cells


# ------------------------------------------------------------------------ 8


def test_8_bar_order_is_identical_across_panels(df):
    bars, meta = _bars(df, comparator="spearman")
    orders = {}
    for b in bars:
        orders.setdefault(b.roi, []).append((b.brain_metric, b.model_metric))
    assert len(set(map(tuple, orders.values()))) == 1
    assert len(orders) == len(meta["rois"])


# ------------------------------------------------------------------------ 9


def test_9_missing_row_leaves_a_gap_not_a_zero_bar(df):
    drop = (df.roi == "LOC") & (df.brain_metric == "crossnobis") \
        & (df.model_metric == "cosine")
    bars, _ = _bars(df[~drop], comparator="spearman")
    hit = [b for b in bars if (b.roi, b.brain_metric, b.model_metric)
           == ("LOC", "crossnobis", "cosine")]
    assert hit == []
    assert not any(b.value == 0.0 for b in bars)


# ----------------------------------------------------------------------- 10


def test_10_png_is_byte_identical_across_runs(df, tmp_path):
    out = []
    for i in range(2):
        fig = plot_main_figure(df, FigureConfig(comparator="spearman"))
        pc.save(fig, str(tmp_path / f"run{i}"))
        matplotlib.pyplot.close(fig)
        out.append((tmp_path / f"run{i}.png").read_bytes())
    assert out[0] == out[1]


def test_10b_footnotes_go_to_the_pdf_only(df, tmp_path):
    """脚注は PDF だけ。PNG は注記が図の外に付く場所に貼るので図から外す。"""
    from PIL import Image

    fig = plot_main_figure(df, FigureConfig(comparator="spearman"))
    assert fig._ceiling_footnote.get_text().strip(), "脚注が空"
    pc.save(fig, str(tmp_path / "with"))
    h_without = Image.open(tmp_path / "with.png").height

    # 脚注を見せたまま書いた PNG は、そのぶん背が高くなる
    fig._ceiling_footnote.set_visible(True)
    fig.savefig(tmp_path / "forced.png", dpi=300, bbox_inches="tight")
    h_with = Image.open(tmp_path / "forced.png").height
    assert h_with > h_without, "PNG から脚注が外れていない"

    # 保存後も図には脚注が残っている（2 回目の save でも PDF に載る）
    assert fig._ceiling_footnote.get_visible()
    matplotlib.pyplot.close(fig)


def test_10c_annotation_is_off_by_default_but_still_works(df):
    assert FigureConfig().annotate == {}
    plain = plot_main_figure(df, FigureConfig(comparator="spearman"))
    marked = plot_main_figure(
        df, FigureConfig(comparator="spearman", annotate={"Early_VC": "しるし"}))
    texts = lambda f: [t.get_text() for ax in f.axes for t in ax.texts]
    assert "しるし" not in texts(plain)
    assert "しるし" in texts(marked)
    assert marked.get_figheight() > plain.get_figheight(), "注釈のぶん上が広がる"


# ----------------------------------------------------------------------- 11


def test_11_inconsistent_band_across_subjects_raises(df, tmp_path):
    bad = df.copy()
    idx = bad.index[(bad.roi == "LOC") & (bad.subject == "sub-02")][0]
    bad.loc[idx, "nili_lower"] = bad.loc[idx, "nili_lower"] + 0.01
    path = tmp_path / "bad.csv"
    bad.drop(columns=["model_metric"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="nili_lower"):
        load(path)


# ----------------------------------------------------------------------- 12


def test_12_english_figure_has_no_japanese(df):
    fig = plot_main_figure(df, FigureConfig(comparator="spearman", lang="en"))
    # 句読点 (U+3000-303F) と全角英数 (U+FF00-FFEF) も含める。「、」は仮名でも
    # 漢字でもないので、範囲を狭く取ると英語版に残ったまま素通りする。
    cjk = re.compile("[　-〿぀-ヿ一-鿿＀-￯]")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    texts += [t.get_text() for t in fig.texts]
    texts += [ax.get_ylabel() for ax in fig.axes]
    for lg in fig.legends:
        texts += [t.get_text() for t in lg.get_texts()] + [lg.get_title().get_text()]
    offenders = [t for t in texts if cjk.search(t)]
    assert offenders == []


# -------------------------------------------------------------------- その他


def test_model_metric_parsed_from_filename():
    assert parse_model_metric("weights_sorted_fMRI1_test_100_dot_product_rdm.csv") \
        == "dot_product"
    assert parse_model_metric("weights_sorted_fMRI1_test_100_cosine_rdm.csv") == "cosine"
    with pytest.raises(ValueError):
        parse_model_metric("something_else.txt")


def test_licensing_is_the_weaker_of_the_two_sides():
    assert is_licensed("euclidean_centered", "euclidean", "cosine")      # L1 × L1
    assert not is_licensed("correlation", "euclidean", "cosine")         # L3 が天井
    assert not is_licensed("euclidean_centered", "cosine", "corr")       # モデル側が L3
    assert is_licensed("correlation", "cosine", "spearman")              # L3 まで落とせば可


def test_excluded_metrics_are_reported(df, capsys):
    _bars(df, comparator="spearman")
    err = capsys.readouterr().err
    assert "[除外] brain_metric=mahalanobis" in err
    assert "[除外] brain_metric=pearson" in err
