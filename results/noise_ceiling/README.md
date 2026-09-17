# ノイズ天井の結果（rsatoolbox 移行後・2026-09-16）

生成：`./run_noise_ceiling.sh`
前提：3 名分の脳 RDM（`get_brainRDM-things.py` ＋ `get_brainRDM-crossnobis.py`）

| ファイル | 内容 |
|---|---|
| `ceilings.csv` | Nili 界と達成率。576 行 |
| `voxel_ceiling_reference.csv` | THINGS 公式のボクセル単位天井の ROI 内平均（参照用） |
| `diagnostics.csv` | **退役**。split-half 信頼性。天井は導かない（§6） |
| `_before_rsatoolbox/` | 移行前の `ceilings.csv` / `diagnostics.csv` |

---

## 0. 移行で変わったこと

| | 移行前 | 移行後 |
|---|---|---|
| 天井 | Nili 界 ＋ split-half（√r_SB） | **Nili 界のみ** |
| 実装 | 自前 | `rsatoolbox` の `pool_rdm` / `compare` / `sets_leave_one_out_rdm` |
| 脳側計量 | correlation / euclidean_centered / **mahalanobis** | correlation / euclidean_centered / **crossnobis** |
| comparator | l1_r0 / pearson / spearman / kendall | **cosine(L1) / corr(L2) / spearman(L3)** |
| 区間推定 | 被験者ブートストラップ | **刺激ブートストラップ** |
| 出力単位 | Nili 行＋split-half 行 | **被験者3行＋被験者平均1行**（`row_type`） |

自前の `nili_ceiling` は `rsatoolbox.inference.boot_noise_ceiling` と
**24 セルすべてでビット一致**（差 0.00e+00）。`tests/test_rsatoolbox_parity.py` が
この一致を継続的に検証する。

---

## 1. `ceilings.csv` の読み方

1 行 = ROI × 脳計量 × モデル RDM × comparator × 行種別。
4 × 3 × 4 × 3 × 4 = 576 行。

**`row_type` が 2 種類ある。**

| `row_type` | 行数 | `subject` | 内容 |
|---|---|---|---|
| `group_mean` | 144 | `F1+F2+F3` | 各被験者との相関の**平均**。Nili 界が適用できるのはこの形だけ |
| `subject` | 432 | `sub-01` 等 | 被験者個別の相関 |

`group_mean` の `score_obs` は同じセルの `subject` 3 行の平均に一致する
（検証済み、差 3.3e-16）。集団平均 RDM との相関**ではない** — そちらは
上界が 1 になり Nili 界と比較できない（仕様書 §2.3）。

### 主な列

| 列 | 内容 |
|---|---|
| `level` / `comparator` | `L1`/`cosine`、`L2`/`corr`、`L3`/`spearman` |
| `score_obs` | 観測された一致度 |
| `perm_p` / `perm_z` | 刺激ラベル置換検定（10000 回）。`group_mean` 行は 3 名に同一置換を当てる。分解能の下限は 1/10001 ≈ 1.0e-4 |
| **`nili_lower`** / `nili_upper` | 下界＝自分を除く 2 名平均との一致、上界＝自分を含む 3 名平均との一致 |
| `nili_*_ci_*` | **刺激**ブートストラップ（1000 回）のパーセンタイル区間。平均幅 0.138 |
| `nili_lower_subject` / `nili_upper_subject` | `subject` 行のみ。その被験者自身の界への寄与 |
| **`achievement_nili_lower`** / `achievement_nili_upper` | `score_obs` ÷ 各界。**1 でクリップしない** |
| `l1_baseline` / `l1_cv_*` | L1 行のみ。無相関でも到達する基準値 |
| `bootstrap` / `n_boot` / `boot_seed` | `pattern`（刺激側）であることの記録 |

`subject` 行の達成率は**群の界**（`nili_lower`）で割っている。被験者間で
比較できるようにするため。その被験者自身の界で割りたい場合は
`score_obs / nili_lower_subject` を計算すること。

---

## 2. 主要な結果

### 2.1 crossnobis は mahalanobis と違って機能する

旧 `mahalanobis` は全 ROI で RDM が定数行列に縮退し、相関が構造的にゼロだった。
`crossnobis` は置換検定で有意な相関を出す。

L2（`corr`）× SPoSE euclidean、被験者平均：

| ROI | 脳計量 | スコア | perm_p | Nili 下界 [CI] | 達成率 |
|---|---|---|---|---|---|
| LOC | `crossnobis` | 0.109 | 0.001 | 0.263 [0.18, 0.35] | **0.41** |
| LOC | `euclidean_centered` | 0.105 | 0.001 | 0.390 [0.29, 0.49] | 0.27 |
| LOC | `correlation` | 0.079 | 0.001 | 0.358 [0.24, 0.47] | 0.22 |
| FFA | `crossnobis` | 0.096 | 0.001 | 0.190 [0.09, 0.28] | **0.50** |
| FFA | `euclidean_centered` | 0.112 | 0.001 | 0.353 [0.23, 0.48] | 0.32 |
| Early_VC | `crossnobis` | −0.011 | 0.629 | 0.572 [0.47, 0.66] | −0.02 |

**crossnobis は達成率を押し上げるが、それは分子が増えたからではなく分母が下がったから。**
crossnobis の Nili 下界は correlation より一貫して低い（平均 0.271 対 0.509）。
ボクセル集合もノイズ構造も被験者ごとに違うため、白色化が被験者間の
RDM 一致を下げている。**生スコアと達成率を必ず併記すること。**

### 2.2 Early_VC の結論は計量を変えても動かない

Early_VC は 3 計量すべてで Nili 界が最も高く（下界 0.57–0.81）、
モデルとの相関だけがゼロ（perm_p = 0.40–0.88）。
「測れていない」という説明は crossnobis に変えても使えない。**対応が無い。**

### 2.3 PPA の crossnobis は分母がゼロ付近で達成率が意味を持たない

| ROI | 計量 | comparator | Nili 下界 | CI |
|---|---|---|---|---|
| PPA | `crossnobis` | `corr` | 0.057 | **[−0.028, 0.136]** |
| PPA | `crossnobis` | `spearman` | 0.049 | **[−0.034, 0.126]** |

区間がゼロを跨いでいる。ここでの達成率 0.83（sub-01 では 1.41）は
分母が小さいことの産物であって、モデルが当たっている証拠ではない。
仕様書 §4.5 がクリップを禁じているのはこの状態を見えるようにするため。

全 576 行のうち達成率が 1 を超えるのは 10 行。すべてこの型。

### 2.4 有意セルの達成率

被験者平均 144 セル中、置換検定で有意なのは 46 セル。

| 段 | 有意セル数 | 達成率の範囲 |
|---|---|---|
| L1 (`cosine`) | 16 | −0.36 – 1.52 |
| L2 (`corr`) | 16 | 0.16 – 1.14 |
| L3 (`spearman`) | 14 | 0.16 – 1.20 |

L2 と L3 はほぼ同じ範囲に収まる。段の選択で結論は変わらない。
L1 の幅が広いのは基準値を引いた後の残りが小さく、比が不安定になるため。

### 2.5 刺激ブートストラップで区間が意味を持つようになった

被験者ブートストラップは S=3 では相異なる値を 5 通りしか取れず、
区間が構成上 `[点推定値, 1.0]` にしかならなかった。
刺激側（100 個）で取り直した結果、下界の区間幅は平均 0.138 になった。

---

## 3. 退役させたもの（削除していない）

| | 場所 | 理由 |
|---|---|---|
| split-half 信頼性 | `noise_ceiling.py` の該当節 | 天井は Nili に一本化。`mahalanobis` の縮退とセッションドリフト不在の根拠として保持 |
| `attenuation_ceiling`（√r_SB） | 同上 | Nili 界に平方根を当ててはいけない（二重補正）。呼び出し元が無くなった |
| `--stage diagnostics` | `run_noise_ceiling.py` | 動くが `--stage all` には含めない。`crossnobis` は非対応 |
| `mahalanobis` RDM | `output/things/_deprecated/` | 定数行列で情報を持たない |

`diagnostics.csv` は移行前のまま置いてある。上の退役に伴い再生成していない。

---

## 4. 不採用にした comparator

| | 理由 |
|---|---|
| `kendall` / `tau-b` | Python 版 `pool_rdm` は τ の天井を平均順位 RDM で打ち切り、MATLAB 版にあった反復最適化を移植していない。界が過小評価＝達成率が甘く出る方向 |
| `rho-a` | 実データに同順位が 1 つも無いため Spearman と完全一致（差 2.8e-17）。列が増えるだけで情報が増えない |

**この結果、L3 段内の頑健性チェックは存在しない。** L3 は `spearman` 1 本。

---

## 5. 既知の制約

- **$r_{MM} = 1$ 固定**は変更していない。SPoSE を決定論的モデルとみなす立場
- **L4（k-NN）未実装**。統計量本体が `compare_rdm_folders.py` に無い
- **`euclidean_centered` は自前実装のまま**。rsatoolbox の `euclidean` は
  二乗距離をチャネル数で割ったもので、L2 の値が全て動くため乗り換えていない
- **crossnobis の精度行列は全 12 セッションから 1 つ推定**。split-half を
  廃止したので半分ごとの推定は不要になった
