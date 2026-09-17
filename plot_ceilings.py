"""主図「一致度と Nili 上下界」の作図。

`results/noise_ceiling/ceilings.csv` を読み、ROI をパネル、脳側計量をブロック、
モデル側計量を棒として、Nili 上下界の帯とともに描く。

この図は計量と比較統計量が後から増えることを前提にしている。棒の本数・パネル
数・図幅・色はすべて下のレジストリと `FigureConfig` から導出され、関数本体に
計量名の分岐は一つも無い。計量を足すときに触るのはレジストリの 1 エントリだけ。

帯の粒度はデータから決まる。変換を持たない比較統計量では Nili 界は
(roi, brain_metric, comparator) ごとなので帯はブロック 1 本に描かれ、L1 のよう
に基準値がモデル側にも依存する比較統計量では帯が棒ごとに分かれる。どちらも
`_band_patches` が同じ規則で扱うので、比較統計量を足すときに描画側は無変更。
"""

from __future__ import annotations

import argparse
import decimal
import operator
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# ---------------------------------------------------------------- レジストリ

# 段の梯子。小さいほど強い主張。
LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}


@dataclass(frozen=True)
class MetricSpec:
    label: str            # 表示名
    kind: str             # "distance" | "similarity"
    max_level: str        # この計量が licensed な最強の段
    include: bool = True  # 図に出すか
    exclude_reason: str = ""

    def __post_init__(self):
        if self.kind not in ("distance", "similarity"):
            raise ValueError(f"unknown kind {self.kind!r} for {self.label!r}")
        if self.max_level not in LEVEL_RANK:
            raise ValueError(f"unknown level {self.max_level!r} for {self.label!r}")


BRAIN_METRICS: dict[str, MetricSpec] = {
    "correlation":        MetricSpec("correlation",        "distance", "L3"),
    "euclidean_centered": MetricSpec("euclidean_centered", "distance", "L1"),
    "crossnobis":         MetricSpec("crossnobis",         "distance", "L1"),
    "mahalanobis":        MetricSpec("mahalanobis", "distance", "L1",
                                     include=False,
                                     exclude_reason="定数行列（擬似逆行列による縮退）"),
    "pearson":            MetricSpec("pearson", "similarity", "L3",
                                     include=False,
                                     exclude_reason="correlation の符号反転で冗長"),
}

MODEL_METRICS: dict[str, MetricSpec] = {
    "cosine":             MetricSpec("cosine",        "distance",   "L3"),
    "euclidean":          MetricSpec("euclidean",     "distance",   "L1"),
    "dot_product":        MetricSpec("dot_product",   "similarity", "L3"),
    "choice_probability": MetricSpec("p(i,j)",        "similarity", "L3"),
}


@dataclass(frozen=True)
class ComparatorSpec:
    label: str
    level: str
    transform: str = "none"   # "none" | "baseline" | "chance"
    note: str = ""

    def __post_init__(self):
        if self.level not in LEVEL_RANK:
            raise ValueError(f"unknown level {self.level!r} for {self.label!r}")
        if self.transform not in TRANSFORM_COLUMNS:
            raise ValueError(f"unknown transform {self.transform!r} for {self.label!r}")


# 変換が要求する CSV 列。無ければ例外を投げる（ゼロで代用しない）。
TRANSFORM_COLUMNS: dict[str, tuple[str, ...]] = {
    "none": (),
    "baseline": ("l1_baseline",),
    "chance": ("chance_level", "k"),
}

# 比較統計量名は rsatoolbox / 上流パイプラインのもの。`cosine` は非中心化相関で、
# 引き継ぎドキュメントの `l1_r0` と同じ量（1e-15 まで一致）。
COMPARATORS: dict[str, ComparatorSpec] = {
    "cosine":   ComparatorSpec("L1 上積み", "L1", transform="baseline",
                               note="l1_baseline (r0) を引いた値を描く"),
    "corr":     ComparatorSpec("Pearson r", "L2"),
    "spearman": ComparatorSpec("Spearman ρ", "L3"),
}

# ------------------------------------------------------------------- 文言

LABELS: dict[str, dict[str, str]] = {
    "ja": {
        "ylabel_plain": "{comp}（脳 RDM とモデル RDM の一致度）",
        "ylabel_transformed": "{comp}（{note}）",
        "legend_title": "モデル側計量",
        "band_lower": "Nili 下界",
        "band_upper": "Nili 上界",
        "subject_dots": "被験者 (n={n})",
        "unlicensed": "licensed でない",
        "p_row_label": "p =",
        "ns": "n.s.",
        "list_sep": "、",
        "foot_sig": "有意性: {ladder}（置換 {n_perm} 回、分解能の下限 {res}）",
        "foot_resolution": "置換回数が {n_perm} のため p の分解能の下限は {res}。"
                           "最も厳しい記号は「下限に張り付いた」ことしか意味しない。",
        "foot_signflip": "符号正規化: 脳側とモデル側で計量の型（距離／類似度）が"
                         "異なる組は符号を反転し、常に上を一致とした。",
        "foot_pflip": "上流の p は符号正規化前のスコアに対する片側上側裾なので、"
                      "符号を反転した組では暫定的に 1−p を用いた両側扱いとした"
                      "（恒久的には上流で符号正規化後に取り直すこと）。",
        "foot_p_row": "p 値行は小数表記で、{cut} 以下は「<{cut}」と略した"
                      "（小数第 {dec} 位で四捨五入）。",
        "foot_band": "帯は Nili の上下界（下界 = 実線、上界 = 破線）。脳側 RDM"
                     "どうしの比較なので符号は反転していない。",
        "foot_unlicensed": "‡ = licensed でない。ペアの天井は両側の弱いほうで"
                           "決まり、それ以上に弱い比較統計量しか使えない。",
        "foot_bars": "8 本の棒は競合モデルではなく、同じ主張を別手順で計算した"
                     "ものである。",
        "foot_errorbar": "誤差棒は描いていない（S=3 ではブートストラップ区間が"
                         "構成上 [点推定値, 1.0] になり情報を持たない）。"
                         "白丸は被験者ごとの値。",
    },
    "en": {
        "ylabel_plain": "{comp} (brain-model RDM agreement)",
        "ylabel_transformed": "{comp} ({note})",
        "legend_title": "Model metric",
        "band_lower": "Nili lower bound",
        "band_upper": "Nili upper bound",
        "subject_dots": "subjects (n={n})",
        "unlicensed": "not licensed",
        "p_row_label": "p =",
        "ns": "n.s.",
        "list_sep": ", ",
        "foot_sig": "Significance: {ladder} ({n_perm} permutations, resolution floor {res})",
        "foot_resolution": "With {n_perm} permutations the p-value resolution floor is {res}; "
                           "the strictest mark only means the value is pinned at that floor.",
        "foot_signflip": "Sign normalisation: pairs whose brain-side and model-side metrics "
                         "differ in kind (distance vs similarity) are sign-flipped, so up is "
                         "always agreement.",
        "foot_pflip": "Upstream p-values are one-sided upper-tail on the raw score, so "
                      "sign-flipped pairs provisionally use 1-p (a two-sided reading); "
                      "these should be recomputed upstream on sign-normalised scores.",
        "foot_p_row": "The p row is decimal, with values at or below {cut} shown as "
                      "\"<{cut}\" (rounded to {dec} decimal places).",
        "foot_band": "Bands are Nili's bounds (lower = solid, upper = dashed). They compare "
                     "brain RDMs with each other and are therefore never sign-flipped.",
        "foot_unlicensed": "‡ = not licensed. A pair's ceiling is set by the weaker of its two "
                           "sides, and only comparators at or below that rung may be used.",
        "foot_bars": "The bars are not competing models: they are the same claim computed by "
                     "different procedures.",
        "foot_errorbar": "No error bars are drawn (with S=3 the bootstrap interval is "
                         "[point estimate, 1.0] by construction and carries no information). "
                         "Open circles are per-subject values.",
    },
}

# ------------------------------------------------------------------- 設定

DEFAULT_ROIS = ("Early_VC", "LOC", "FFA", "PPA")

# レイアウト定数（棒 1 本 = x 単位の 1）
BLOCK_GAP_UNITS = 1.0
BLOCK_PAD = 0.12
BAND_FILL = "#7f9fbf"
BAND_EDGE = "#3d5a73"
UNLIC_COLOR = "#9a9a9a"
UNLIC_HATCH = "///"


@dataclass
class FigureConfig:
    comparator: str = "spearman"
    rois: tuple[str, ...] = DEFAULT_ROIS
    brain_metrics: tuple[str, ...] | None = None   # None ならレジストリの include=True 全部
    model_metrics: tuple[str, ...] | None = None
    unlicensed: str = "omit"                       # "omit" | "mark" | "error"
    show_subject_dots: bool = True
    alpha: float = 0.05
    # 有意性表記
    sig_levels: tuple[tuple[float, str], ...] = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
    # 置換 10000 回では下限が 1/10001 ≈ 1e-4 なので p < 0.001 が実質的な主張に
    # なる。1000 回のままの CSV を描くときだけ "le" に落とすこと（§7.3）。
    sig_operator: str = "lt"                       # "le" (p <= th) | "lt" (p < th)
    ns_label: str = "n.s."
    show_p_row: bool = True
    p_row_fmt: str = "decimal"                     # "decimal" | "trim" | "sci"
    p_min_display: float = 0.001                   # これ以下は "<0.001" に略す
    p_decimals: int = 3                            # 小数第 3 位まで（四捨五入）
    n_permutations: int = 10000                    # CSV の n_perm が優先。食い違えば警告
    flip_p_for_similarity: bool = True             # 符号反転した組で 1-p を使う（暫定・§5.2）
    lang: str = "ja"                               # "ja" | "en"
    annotate: dict[str, str] = field(default_factory=dict)
    # レイアウト（インチ）
    bar_w: float = 0.26
    bar_gap: float = 0.09
    block_gap: float = 0.40
    panel_gap: float = 0.34
    plot_h: float = 3.4
    margin_left: float = 1.15
    margin_right: float = 1.75
    margin_top: float = 0.95
    margin_bottom: float = 2.05
    # 棒が詰まりすぎたら p 値行を自動で落とす閾値（1 図あたりの棒の本数）
    p_row_max_bars: int = 64


def _L(cfg: FigureConfig, key: str, **kw) -> str:
    """文言はここからだけ引く。関数内にリテラルを書かないこと。"""
    try:
        table = LABELS[cfg.lang]
    except KeyError:
        raise ValueError(f"unknown lang {cfg.lang!r}; expected one of {sorted(LABELS)}")
    return table[key].format(**kw)


# 日本語を持つフォントの候補。環境に無いものを rcParams に残すと findfont が
# 一文字ごとに警告を出すので、実在するものだけに絞る。
CJK_FONT_CANDIDATES = (
    "Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic",
    "IPAGothic", "TakaoGothic", "VL Gothic", "Yu Gothic", "MS Gothic",
)


def setup_fonts() -> None:
    """日本語が豆腐にならないようにする。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    found = [f for f in CJK_FONT_CANDIDATES if f in installed]
    if not found:
        print("[警告] 日本語フォントが見つからない。lang='ja' では文字が豆腐になる。"
              f"候補: {list(CJK_FONT_CANDIDATES)}", file=sys.stderr)
    matplotlib.rcParams["font.family"] = found + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.hashsalt"] = "brain-rdm-ceilings"


# ------------------------------------------------------------------- 入力

MODEL_RDM_PATTERN = re.compile(r"^.*?_(?P<metric>[a-z_]+)_rdm\.csv$")

REQUIRED_COLUMNS = (
    "roi", "brain_metric", "model_rdm", "comparator", "row_type",
    "subject", "score_obs", "perm_p", "nili_lower", "nili_upper",
)


def parse_model_metric(model_rdm: str) -> str:
    """モデル RDM のファイル名からモデル側計量名を取り出す。

    レジストリのキーを長いものから順に照合する。`dot_product` と `product` の
    ような接尾辞の衝突を避けるため、正規表現の貪欲さには頼らない。
    """
    stem = str(model_rdm)
    for key in sorted(MODEL_METRICS, key=len, reverse=True):
        if re.search(rf"(?:^|[_/]){re.escape(key)}_rdm\.csv$", stem):
            return key
    m = MODEL_RDM_PATTERN.match(stem)
    if m:
        return m.group("metric")
    raise ValueError(f"モデル側計量をファイル名から決められない: {model_rdm!r}")


def load(csv_path: str | Path) -> pd.DataFrame:
    """CSV を読み、未知の列は残したまま `model_metric` を足して返す。"""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: 必須列が無い: {missing}")
    df = df.copy()
    df["model_metric"] = df["model_rdm"].map(parse_model_metric)
    _validate_bands(df)
    return df


def _validate_bands(df: pd.DataFrame) -> None:
    """Nili 界が (roi, brain_metric, comparator) で一意であることを検証する。

    被験者行に同じ値が繰り返し入っている前提が崩れていたら上流のバグなので、
    黙って平均を取らずに例外を投げる。
    """
    keys = ["roi", "brain_metric", "comparator"]
    for col in ("nili_lower", "nili_upper"):
        spread = df.groupby(keys)[col].agg(lambda s: s.max() - s.min())
        bad = spread[spread > 1e-9]
        if len(bad):
            raise ValueError(
                f"{col} が (roi, brain_metric, comparator) 内で一致しない:\n"
                f"{bad.to_string()}"
            )


# ----------------------------------------------------------------- 前処理


@dataclass
class Bar:
    """棒 1 本分。描画側はこれしか見ない。"""
    roi: str
    brain_metric: str
    model_metric: str
    value: float                 # 符号正規化・変換後の点推定値
    lower: float                 # 変換後の Nili 下界
    upper: float                 # 変換後の Nili 上界
    p_raw: float
    p_eff: float
    sign: int
    licensed: bool
    subject_pts: tuple[float, ...]
    n_subjects: int


def _resolve_metrics(requested, registry, side: str) -> tuple[str, ...]:
    if requested is None:
        return tuple(k for k, spec in registry.items() if spec.include)
    unknown = [m for m in requested if m not in registry]
    if unknown:
        raise ValueError(f"{side} にレジストリに無い計量が指定された: {unknown}")
    return tuple(requested)


def _report_exclusions(registry, side: str) -> None:
    width = max((len(k) for k in registry), default=0)
    for key, spec in registry.items():
        if not spec.include:
            print(f"[除外] {side}={key.ljust(width)} : {spec.exclude_reason}", file=sys.stderr)


def is_licensed(brain: str, model: str, comparator: str) -> bool:
    """段の梯子は鎖。ペアの天井は両側の弱いほう（rank の大きいほう）で決まる。"""
    ceiling_rank = max(LEVEL_RANK[BRAIN_METRICS[brain].max_level],
                       LEVEL_RANK[MODEL_METRICS[model].max_level])
    return LEVEL_RANK[COMPARATORS[comparator].level] >= ceiling_rank


def sig_mark(p: float, cfg: FigureConfig) -> str:
    cmp_ = operator.le if cfg.sig_operator == "le" else operator.lt
    for th, mark in cfg.sig_levels:          # 厳しい順（閾値の昇順）
        if np.isfinite(p) and cmp_(p, th):
            return mark
    return cfg.ns_label


def _check_config(cfg: FigureConfig) -> None:
    if cfg.comparator not in COMPARATORS:
        raise ValueError(f"未知の比較統計量 {cfg.comparator!r}; "
                         f"レジストリにあるのは {sorted(COMPARATORS)}")
    if cfg.unlicensed not in ("omit", "mark", "error"):
        raise ValueError(f"未知の unlicensed 設定 {cfg.unlicensed!r}")
    if cfg.sig_operator not in ("le", "lt"):
        raise ValueError(f"未知の sig_operator {cfg.sig_operator!r}")
    if cfg.p_row_fmt not in ("decimal", "trim", "sci"):
        raise ValueError(f"未知の p_row_fmt {cfg.p_row_fmt!r}")
    if cfg.p_decimals < 1:
        raise ValueError(f"p_decimals は 1 以上: {cfg.p_decimals}")
    ths = [th for th, _ in cfg.sig_levels]
    if ths != sorted(ths):
        raise ValueError(
            f"sig_levels は閾値の昇順（厳しい順）で書くこと: {cfg.sig_levels}。"
            "降順だと全セルが最も緩い記号に落ちる。"
        )


def _n_permutations(sel: pd.DataFrame, cfg: FigureConfig) -> int:
    """図に書く置換回数。CSV の n_perm を正とする。

    設定値を正にすると、古い CSV を新しい既定で描いたときに脚注だけが嘘になる。
    分解能の下限も有意性記号もここから決まるので、食い違いは黙って通さない。
    """
    n_perm = cfg.n_permutations
    if "n_perm" in sel.columns:
        found = sorted(set(sel["n_perm"].dropna().astype(int)))
        if len(found) > 1:
            raise ValueError(f"CSV 内で n_perm が揃っていない: {found}")
        if found:
            if found[0] != cfg.n_permutations:
                print(f"[警告] CSV の n_perm={found[0]} が設定の "
                      f"n_permutations={cfg.n_permutations} と 食い違う。"
                      "CSV の値を採る。", file=sys.stderr)
            n_perm = found[0]

    floor = p_resolution(n_perm)
    ths = [th for th, _ in cfg.sig_levels]
    if ths and ths[0] <= floor:
        tail = ("sig_operator='le' なので下限に張り付いた値だけが拾われる。"
                if cfg.sig_operator == "le"
                else "sig_operator='lt' ではこの記号は一つも出ない。")
        print(
            f"[警告] 置換 {n_perm} 回では p の分解能の下限が {floor:.2g} で、"
            f"最も厳しい閾値 {ths[0]:g} はその下限以下。" + tail,
            file=sys.stderr,
        )
    return n_perm


def p_resolution(n_perm: int) -> float:
    """置換検定 p の分解能の下限。上流は (r+1)/(n+1) で計算している。"""
    return 1.0 / (n_perm + 1)


def complement_p(p: float, n_perm: int) -> float:
    """片側上側裾の p から、下側裾の p を離散的に取り直す。

    素朴に `1 - p` とすると、p = 1.0 の行で 0 という有り得ない p 値が出る。
    上流が p = (r+1)/(n+1) で数えているので、下側は (n-r+1)/(n+1) = 1 - p +
    1/(n+1) になり、下限はちょうど分解能の下限に止まる。

    これはあくまで暫定処置で、正しくは上流で符号正規化後のスコアに対して p を
    取り直すこと（仕様書 §5.2）。
    """
    floor = p_resolution(n_perm)
    return float(min(1.0, max(floor, 1.0 - p + floor)))


def build_bars(df: pd.DataFrame, cfg: FigureConfig) -> tuple[list[Bar], dict]:
    """CSV から棒のリストを作る。ここが符号正規化・変換・licensing の唯一の場所。"""
    _check_config(cfg)
    _report_exclusions(BRAIN_METRICS, "brain_metric")
    _report_exclusions(MODEL_METRICS, "model_metric")

    brain_metrics = _resolve_metrics(cfg.brain_metrics, BRAIN_METRICS, "brain_metrics")
    model_metrics = _resolve_metrics(cfg.model_metrics, MODEL_METRICS, "model_metrics")
    cspec = COMPARATORS[cfg.comparator]

    need = TRANSFORM_COLUMNS[cspec.transform]
    absent = [c for c in need if c not in df.columns]
    if absent:
        raise ValueError(
            f"比較統計量 {cfg.comparator!r} は transform={cspec.transform!r} なので "
            f"列 {absent} が要る。ゼロで代用しないこと。"
        )

    sel = df[df["comparator"] == cfg.comparator]
    if sel.empty:
        raise ValueError(f"comparator={cfg.comparator!r} の行が CSV に無い")
    n_perm = _n_permutations(sel, cfg)
    if "level" in sel.columns:
        levels = set(sel["level"].dropna().unique())
        if levels and levels != {cspec.level}:
            raise ValueError(
                f"CSV の level {sorted(levels)} がレジストリの "
                f"{cfg.comparator}={cspec.level} と食い違う"
            )

    bars: list[Bar] = []
    unlicensed_pairs: set[tuple[str, str]] = set()
    for roi in cfg.rois:
        for brain in brain_metrics:
            for model in model_metrics:
                rows = sel[(sel["roi"] == roi)
                           & (sel["brain_metric"] == brain)
                           & (sel["model_metric"] == model)]
                if rows.empty:
                    continue                      # 0 の棒は描かない。位置を空ける
                licensed = is_licensed(brain, model, cfg.comparator)
                if not licensed:
                    unlicensed_pairs.add((brain, model))
                    if cfg.unlicensed == "error":
                        raise ValueError(
                            f"licensed でない組: brain={brain} ({BRAIN_METRICS[brain].max_level}) × "
                            f"model={model} ({MODEL_METRICS[model].max_level}) に対して "
                            f"comparator={cfg.comparator} ({cspec.level})"
                        )
                    if cfg.unlicensed == "omit":
                        continue
                bars.append(_make_bar(rows, roi, brain, model, cfg, cspec, licensed))

    for brain, model in sorted(unlicensed_pairs):
        verb = "灰色 + ‡ で描く" if cfg.unlicensed == "mark" else "棒を落とした"
        print(
            f"[licensing] {cfg.comparator} ({cspec.level}) は brain={brain}"
            f"({BRAIN_METRICS[brain].max_level}) × model={model}"
            f"({MODEL_METRICS[model].max_level}) に対して licensed でない → {verb}",
            file=sys.stderr,
        )

    if not bars:
        raise ValueError(f"comparator={cfg.comparator!r} で描ける棒が一本も残らなかった")

    meta = {
        "brain_metrics": [b for b in brain_metrics
                          if any(x.brain_metric == b for x in bars)],
        "model_metrics": [m for m in model_metrics
                          if any(x.model_metric == m for x in bars)],
        "rois": [r for r in cfg.rois if any(x.roi == r for x in bars)],
        "has_unlicensed": any(not x.licensed for x in bars),
        "has_signflip": any(x.sign < 0 for x in bars),
        "n_subjects": max(x.n_subjects for x in bars),
        "n_permutations": n_perm,
        "comparator": cspec,
    }
    ns = {x.n_subjects for x in bars}
    if len(ns) > 1:
        print(f"[警告] 被験者数が棒ごとに揃っていない: {sorted(ns)}", file=sys.stderr)
    return bars, meta


def _make_bar(rows, roi, brain, model, cfg, cspec, licensed) -> Bar:
    group = rows[rows["row_type"] == "group_mean"]
    subs = rows[rows["row_type"] == "subject"]
    if len(group) != 1:
        raise ValueError(
            f"group_mean 行が {len(group)} 本ある: roi={roi} brain={brain} model={model}"
        )
    g = group.iloc[0]

    # 距離どうし・類似度どうしなら正が一致、混ざれば負が一致。正規化後は常に上が一致。
    sign = 1 if BRAIN_METRICS[brain].kind == MODEL_METRICS[model].kind else -1

    # 変換は棒にも帯にも同じものを当てる。片方だけに当てると図が破綻する。
    shift = _transform_shift(g, cspec)
    value = sign * (float(g["score_obs"]) - shift)
    # Nili 界は脳側 RDM どうしの比較なので符号は反転しない。
    lower = float(g["nili_lower"]) - shift
    upper = float(g["nili_upper"]) - shift

    p_raw = float(g["perm_p"])
    n_perm = int(g["n_perm"]) if "n_perm" in g.index and np.isfinite(g["n_perm"]) \
        else cfg.n_permutations
    p_eff = complement_p(p_raw, n_perm) \
        if (sign < 0 and cfg.flip_p_for_similarity) else p_raw

    pts = tuple(sign * (float(r["score_obs"]) - _transform_shift(r, cspec))
                for _, r in subs.iterrows())
    if pts and not np.isclose(np.mean(pts), value, atol=1e-6):
        print(f"[警告] 被験者平均 {np.mean(pts):.6f} が group_mean の "
              f"{value:.6f} と一致しない: roi={roi} brain={brain} model={model}",
              file=sys.stderr)
    return Bar(roi, brain, model, value, lower, upper, p_raw, p_eff,
               sign, licensed, pts, len(pts))


def _transform_shift(row, cspec: ComparatorSpec) -> float:
    """棒と帯の両方から引く量。`transform` が "none" なら 0。"""
    if cspec.transform == "none":
        return 0.0
    if cspec.transform == "baseline":
        return float(row["l1_baseline"])
    if cspec.transform == "chance":
        return float(row["chance_level"])
    raise ValueError(f"未知の transform {cspec.transform!r}")


# --------------------------------------------------------------- レイアウト


def _colors(model_metrics: list[str]):
    n = len(model_metrics)
    if n <= 5:
        cmap = plt.get_cmap("Blues")
        return {m: cmap(0.45 + 0.35 * i / max(1, n - 1)) for i, m in enumerate(model_metrics)}
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return {m: cmap(i % cmap.N) for i, m in enumerate(model_metrics)}


def _band_patches(block_bars: list[Bar], block_i: int, n_bars: int, xs: dict):
    """帯を描く矩形の (x0, width, lower, upper) を返す。

    ブロック内の全ての棒が同じ界を持つなら、ブロック 1 本にまとめる。基準値が
    モデル側にも依存する比較統計量（L1）では界が棒ごとに違うので、棒ごとに分か
    れる。`axhspan` は軸幅いっぱいに広がるので使わない。
    """
    bounds = {(round(b.lower, 12), round(b.upper, 12)) for b in block_bars}
    block_span = n_bars + BLOCK_GAP_UNITS
    if len(bounds) == 1:
        lower, upper = next(iter(bounds))
        x0 = block_i * block_span - 0.5 - BLOCK_PAD
        w = n_bars - 1 + 1.0 + 2 * BLOCK_PAD
        return [(x0, w, lower, upper)]
    return [(xs[(b.brain_metric, b.model_metric)] - 0.5 + 0.06,
             1.0 - 0.12, b.lower, b.upper) for b in block_bars]


def _round_half_up(value: float, decimals: int) -> str:
    """表示用の丸め。統計値の報告は切り捨てではなく四捨五入が慣例。

    切り捨てにすると 0.0019 が 0.001 になり、実際より小さい p として読まれる。
    Python の書式指定は二進表現に対する偶数丸めなので、十進で見たときの
    四捨五入とごく稀に食い違う。ここは表示だけの話なので、人が読む十進表記の
    ほうに合わせる。
    """
    quantum = decimal.Decimal(1).scaleb(-decimals)
    d = decimal.Decimal(repr(float(value))).quantize(
        quantum, rounding=decimal.ROUND_HALF_UP)
    return f"{d:f}"


def _fmt_p(p: float, cfg: FigureConfig, n_perm: int) -> str:
    """p 値行 1 セルの文字列。

    既定は小数表記で、`p_min_display` 以下は "<0.001" に略す。置換検定の分解能
    の下限（1/(n+1)）より細かい値は意味を持たないので、そこまで書いても読み手を
    誤らせるだけという判断。
    """
    if not np.isfinite(p):
        return "—"
    if cfg.p_row_fmt == "sci":
        floor = p_resolution(n_perm)
        return f"<{floor:.0e}" if p <= floor else f"{p:.1e}"

    cut = _round_half_up(cfg.p_min_display, cfg.p_decimals)
    text = f"<{cut}" if p <= cfg.p_min_display else _round_half_up(p, cfg.p_decimals)
    if cfg.p_row_fmt == "trim":
        text = text.replace("0.", ".", 1)
    return text


# ------------------------------------------------------------------- 描画


def plot_main_figure(df: pd.DataFrame, cfg: FigureConfig) -> matplotlib.figure.Figure:
    setup_fonts()
    bars, meta = build_bars(df, cfg)
    rois = meta["rois"]
    brain_metrics = meta["brain_metrics"]
    model_metrics = meta["model_metrics"]
    cspec = meta["comparator"]

    n_bars = len(model_metrics)
    n_block = len(brain_metrics)
    block_span = n_bars + BLOCK_GAP_UNITS
    bar_w_data = cfg.bar_w / (cfg.bar_w + cfg.bar_gap)
    colors = _colors(model_metrics)

    # x 座標は棒の本数から導く。ハードコードしないこと。
    xs = {(b, m): bi * block_span + mi
          for bi, b in enumerate(brain_metrics)
          for mi, m in enumerate(model_metrics)}

    panel_w = n_block * n_bars * cfg.bar_w + (n_block - 1) * cfg.block_gap
    fig_w = cfg.margin_left + len(rois) * panel_w + (len(rois) - 1) * cfg.panel_gap + cfg.margin_right
    # 注釈を出すときだけ、矢印と文字のぶん上を広げる
    margin_top = cfg.margin_top + (0.40 if any(r in cfg.annotate for r in rois) else 0.0)
    fig_h = margin_top + cfg.plot_h + cfg.margin_bottom
    fig, axes = plt.subplots(
        1, len(rois), sharey=True, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [panel_w] * len(rois),
                     "wspace": cfg.panel_gap / max(panel_w, 1e-9)},
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(
        left=cfg.margin_left / fig_w,
        right=1 - cfg.margin_right / fig_w,
        top=1 - margin_top / fig_h,
        bottom=cfg.margin_bottom / fig_h,
    )

    values = [b.value for b in bars]
    pts = [p for b in bars for p in b.subject_pts] if cfg.show_subject_dots else []
    ymin = min([0.0] + values + pts) - 0.05
    ymax = max([b.upper for b in bars] + values + pts) + 0.08

    show_p_row = cfg.show_p_row
    if show_p_row and len(bars) > cfg.p_row_max_bars:
        print(f"[警告] 棒が {len(bars)} 本あり p 値行が重なるので show_p_row を落とした",
              file=sys.stderr)
        show_p_row = False

    # p 値行は縦書きなので、その下に置くブロック名の位置は p の文字列長から決める。
    # 固定値にすると p_row_fmt を "trim" から "sci" に変えた時点で重なる。
    P_ROW_FS = 7.0
    Y_P_ROW = -0.055
    if show_p_row:
        longest = max(len(_fmt_p(b.p_eff, cfg, meta["n_permutations"])) for b in bars)
        p_row_h = longest * P_ROW_FS * 0.62 / (cfg.plot_h * 72)   # 軸の高さに対する比
        Y_BLOCK = Y_P_ROW - p_row_h - 0.035
    else:
        Y_BLOCK = -0.06

    for ax, roi in zip(axes, rois):
        blend = blended_transform_factory(ax.transData, ax.transAxes)
        roi_bars = [b for b in bars if b.roi == roi]

        for bi, brain in enumerate(brain_metrics):
            block_bars = [b for b in roi_bars if b.brain_metric == brain]
            if not block_bars:
                continue
            for x0, w, lower, upper in _band_patches(block_bars, bi, n_bars, xs):
                ax.add_patch(Rectangle((x0, lower), w, upper - lower, facecolor=BAND_FILL,
                                       alpha=0.25, edgecolor="none", zorder=1))
                ax.hlines(lower, x0, x0 + w, color=BAND_EDGE, lw=1.6, zorder=2)
                ax.hlines(upper, x0, x0 + w, color=BAND_EDGE, lw=1.0,
                          ls=(0, (4, 3)), zorder=2)

            # ブロックが丸ごと licensed でないなら、ブロック全体に網掛けを重ねる
            if all(not b.licensed for b in block_bars):
                x0 = bi * block_span - 0.5 - BLOCK_PAD
                w = n_bars - 1 + 1.0 + 2 * BLOCK_PAD
                ax.add_patch(Rectangle((x0, 0), w, 1, transform=blend, facecolor="none",
                                       edgecolor=UNLIC_COLOR, hatch=UNLIC_HATCH,
                                       alpha=0.15, lw=0.0, zorder=0.5))

        ax.axhline(0, color="#333", lw=1.4, zorder=3)

        for b in roi_bars:
            x = xs[(b.brain_metric, b.model_metric)]
            face = colors[b.model_metric] if b.licensed else UNLIC_COLOR
            ax.bar(x, b.value, width=bar_w_data, facecolor=face, edgecolor="none",
                   hatch=None if b.licensed else UNLIC_HATCH, zorder=4)
            if not b.licensed:
                ax.plot([x], [b.value], marker="x", ms=6, mew=1.4,
                        color="#5a5a5a", zorder=6, ls="none")
            if cfg.show_subject_dots and b.subject_pts:
                offs = np.linspace(-0.18, 0.18, len(b.subject_pts))
                ax.scatter(x + offs, b.subject_pts, s=12, facecolor="white",
                           edgecolor="#333", linewidth=0.7, zorder=5)

            # 負の棒は 0 線を基準にする（棒の下端に置くと軸外に出る）
            mark = sig_mark(b.p_eff, cfg)
            pad = 0.018 * (ymax - ymin)
            top = max([b.value] + list(b.subject_pts), default=b.value) if cfg.show_subject_dots \
                else b.value
            y_anchor = max(top, 0.0)
            ax.text(x, y_anchor + pad, mark, ha="center", va="bottom",
                    fontsize=11 if mark != cfg.ns_label else 8,
                    fontweight="bold" if mark != cfg.ns_label else "normal",
                    color="#1a1a1a" if mark != cfg.ns_label else "#888", zorder=7)

            if show_p_row:
                ax.text(x, Y_P_ROW, _fmt_p(b.p_eff, cfg, meta["n_permutations"]),
                        transform=blend, ha="center",
                        va="top", fontsize=P_ROW_FS, color="#444", rotation=90)

        for bi, brain in enumerate(brain_metrics):
            block_bars = [b for b in roi_bars if b.brain_metric == brain]
            if not block_bars:
                continue
            unlic = all(not b.licensed for b in block_bars)
            label = BRAIN_METRICS[brain].label.replace("_", "_\n")
            if unlic:
                label += " ‡"
            ax.text(bi * block_span + (n_bars - 1) / 2, Y_BLOCK, label, transform=blend,
                    ha="center", va="top", fontsize=8.5,
                    color="#8a6d00" if unlic else "#222")

        ax.set_xlim(-0.8, n_block * block_span - BLOCK_GAP_UNITS - 0.2)
        ax.set_xticks([])
        ax.set_title(roi, fontweight="bold", fontsize=11, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5)

        if roi in cfg.annotate:
            # 矢印は棒の先端を指す（帯ではなく「一致がどこにあるか」が主張なので）。
            # 文字は ROI 名の上に置く。パネル内に置くと帯と重なる。
            tops = [max([b.value] + list(b.subject_pts)) for b in roi_bars]
            # 矢印の x はブロック間の隙間に置く。ブロックの中心に落とすと棒や
            # 有意性記号の上を通る。
            center = n_block * block_span / 2 - 0.5
            gaps = [((i - 1) * block_span + n_bars - 1 + i * block_span) / 2
                    for i in range(1, n_block)]
            x_arrow = min(gaps, key=lambda g: abs(g - center)) if gaps else center
            ax.annotate(
                cfg.annotate[roi],
                xy=(x_arrow, max(tops, default=0.0) + 0.05 * (ymax - ymin)),
                xytext=(0.5, 1.13), textcoords="axes fraction",
                ha="center", va="bottom", fontsize=8.5, color="#a33",
                arrowprops=dict(arrowstyle="->", color="#a33", lw=1.0,
                                connectionstyle="arc3,rad=0.0",
                                shrinkA=2, shrinkB=2),
            )

    axes[0].set_ylim(ymin, ymax)
    ylab = (_L(cfg, "ylabel_transformed", comp=cspec.label, note=cspec.note)
            if cspec.transform != "none"
            else _L(cfg, "ylabel_plain", comp=cspec.label))
    axes[0].set_ylabel(ylab, fontsize=9.5)
    if show_p_row:
        axes[0].text(-0.02, Y_P_ROW, _L(cfg, "p_row_label"), transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=7.5, color="#444")

    _legend(fig, cfg, meta, colors)
    # 脚注は `save()` が PDF にだけ載せる。Text を図に持たせておき、PNG を書く
    # ときに外す。
    fig._ceiling_footnote = _footnotes(fig, cfg, meta, show_p_row)
    return fig


def _legend(fig, cfg: FigureConfig, meta: dict, colors) -> None:
    handles = [Patch(facecolor=colors[m], label=MODEL_METRICS[m].label)
               for m in meta["model_metrics"]]
    handles += [
        Patch(facecolor=BAND_FILL, alpha=0.25, edgecolor=BAND_EDGE,
              label=f'{_L(cfg, "band_lower")} / {_L(cfg, "band_upper")}'),
    ]
    if cfg.show_subject_dots:
        handles.append(Line2D([], [], marker="o", ls="none", mfc="white", mec="#333",
                              ms=4, label=_L(cfg, "subject_dots", n=meta["n_subjects"])))
    if meta["has_unlicensed"]:
        handles.append(Patch(facecolor=UNLIC_COLOR, hatch=UNLIC_HATCH,
                             label=_L(cfg, "unlicensed") + " (‡)"))
    fig.legend(handles=handles, title=_L(cfg, "legend_title"), loc="upper right",
               bbox_to_anchor=(0.995, 0.97), frameon=False, fontsize=8.5,
               title_fontsize=9, borderaxespad=0.0)


def _footnotes(fig, cfg: FigureConfig, meta: dict, show_p_row: bool):
    n_perm = meta["n_permutations"]
    floor = p_resolution(n_perm)
    op = "≤" if cfg.sig_operator == "le" else "<"
    sep = _L(cfg, "list_sep")
    ladder = sep.join([f"{mark} p {op} {th:g}" for th, mark in cfg.sig_levels]
                      + [cfg.ns_label])
    lines = [
        _L(cfg, "foot_sig", ladder=ladder, n_perm=n_perm, res=f"{floor:.2g}"),
        _L(cfg, "foot_resolution", n_perm=n_perm, res=f"{floor:.2g}"),
        _L(cfg, "foot_band"),
        _L(cfg, "foot_bars"),
        _L(cfg, "foot_errorbar"),
    ]
    if show_p_row and cfg.p_row_fmt != "sci":
        lines.insert(2, _L(cfg, "foot_p_row",
                           cut=_round_half_up(cfg.p_min_display, cfg.p_decimals),
                           dec=cfg.p_decimals))
    if meta["has_signflip"]:
        lines.insert(2, _L(cfg, "foot_signflip"))
        if cfg.flip_p_for_similarity:
            lines.insert(3, _L(cfg, "foot_pflip"))
    if meta["has_unlicensed"]:
        lines.append(_L(cfg, "foot_unlicensed"))
    return fig.text(0.008, 0.008, "\n".join(lines), ha="left", va="bottom",
                    fontsize=7.0, color="#555", linespacing=1.5)


def save(fig, stem: str) -> None:
    """PDF と PNG（300 dpi）を出力。埋め込みタイムスタンプを消して決定的にする。

    脚注は PDF にだけ載せる。PNG はスライドや Notion に貼る用で、そこでは注記が
    図の外に付くため、図の中に持たせると二重になる。PDF は単体で配るものなので
    脚注が要る。`bbox_inches="tight"` が余白を詰めるので、PNG は脚注のぶんだけ
    背が低くなる。
    """
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    footnote = getattr(fig, "_ceiling_footnote", None)

    if footnote is not None:
        footnote.set_visible(True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})

    if footnote is not None:
        footnote.set_visible(False)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", metadata={"Software": None})
    if footnote is not None:
        footnote.set_visible(True)


# -------------------------------------------------------------------- main


def figure_set(base: FigureConfig) -> list[tuple[str, FigureConfig]]:
    """比較統計量ごとに 1 枚。レジストリに 1 エントリ足せば図が 1 枚増える。"""
    # 図に出る全ペアの天井のうち最も弱い段。これ以上に弱い比較統計量なら全ブロック
    # が licensed になる。
    strongest = max(
        [LEVEL_RANK[BRAIN_METRICS[b].max_level]
         for b in _resolve_metrics(base.brain_metrics, BRAIN_METRICS, "brain_metrics")]
        + [LEVEL_RANK[MODEL_METRICS[m].max_level]
           for m in _resolve_metrics(base.model_metrics, MODEL_METRICS, "model_metrics")]
    )
    out = []
    for name, spec in COMPARATORS.items():
        # 全ブロックが licensed になる段は主図で、警告が出たらレジストリか上流
        # データの誤りなので error。それ以外は補足図で mark にする。omit だと
        # ブロックが丸ごと消えて「先行研究の標準構成がどこにいたか」という補足図
        # の目的が失われる。
        full = LEVEL_RANK[spec.level] >= strongest
        kind = "main" if full else "supp"
        out.append((f"fig_{kind}_{name}",
                    replace(base, comparator=name,
                            unlicensed="error" if full else "mark")))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="results/noise_ceiling/ceilings.csv")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--lang", default="ja", choices=sorted(LABELS))
    ap.add_argument("--comparator", default=None, choices=sorted(COMPARATORS),
                    help="指定するとその比較統計量だけを描く")
    args = ap.parse_args(argv)

    df = load(args.csv)
    # `annotate` は空。矢印つきの注釈は §7.8 の仕組みとして残してあるが、既定では
    # 描かない（所見は図そのものとキャプションで示す）。
    base = FigureConfig(lang=args.lang)
    figs = figure_set(base)
    if args.comparator:
        figs = [(s, c) for s, c in figs if c.comparator == args.comparator]

    for stem, cfg in figs:
        print(f"\n=== {stem} (comparator={cfg.comparator}, unlicensed={cfg.unlicensed}) ===",
              file=sys.stderr)
        fig = plot_main_figure(df, cfg)
        path = str(Path(args.outdir) / f"{stem}_{args.lang}")
        save(fig, path)
        plt.close(fig)
        print(f"  -> {path}.pdf / {path}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
