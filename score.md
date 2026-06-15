# Condensate HEAD route scorecard

この文書は、凝縮あり HEAD route の trial ごとの変更内容、評価指標、score、差分を保存するための scorecard である。FastChem4 は比較対象であり、ExoGibbs の constructor input や exact replay target にはしない。

数値 score の機械可読版は `score.json` に置く。この Markdown は、人間が trial の意味、判断、次の調査方針を読むための companion document とする。

## 指標の考え方

今後の改善では、単一の `converged` count だけでは不十分である。少なくとも次を分けて見る。

| metric | 目的 | 改善方向 |
|---|---|---|
| public convergence / budget gate | public API として返してよいかを確認する。 | `converged` を維持し、budget gate reject を増やさない。 |
| ExoGibbs Gibbs objective `G/RT` | ExoGibbs-native objective 上で実装変更が物理的に悪化していないかを見る。 | 同じ条件の before/after で下がる、少なくとも上がらない。 |
| inactive positive condensate driving | support 外に「まだ凝縮したがっている種」を残していないかを見る。 | 最大値と count を下げる。 |
| FastChem4 gas abundance difference | trace gas を含む public output の大ぽかを検出する。 | max `|log10 gas ratio|` を下げる。 |
| active condensate Jaccard | FastChem4 active condensate list との近さを見る。ただし exact replay 目標ではない。 | mean/min を上げる。 |
| FastChem4-scaled Gibbs comparison | FastChem4 state を ExoGibbs budget gauge に scale したとき、ExoGibbs-native `G/RT` でどちらが低いかを見る。 | ExoGibbs-lower count を増やし、max `|dG/RT|` を下げる。 |
| route/support churn | 変更が局所的か、広範囲に route を作り替えていないかを見る。 | 意図した rows だけが変わる。 |

優先順位は、まず public convergence と budget consistency を壊さないこと、次に ExoGibbs-native `G/RT` と inactive driving を改善すること、その後 FastChem4 gas/Jaccard 差を詰めることとする。

## Baseline: HEAD route v1.4

v1.4 は public full condensate budget consistency を主目的にした固定版である。

確認済み baseline:

| metric | score |
|---|---:|
| curated full-profile rows | 99 |
| public converged | 99 |
| public budget gate rejects | 0 |
| route counts | 82 primary, 17 gas-only, 0 native fallback |
| bookkeeping/gate mismatch rows | 0 |

v1.4 の主な残課題は、public budget は閉じている一方で、support-cap retry が小さい support で早く promoted route を返し、support 外に大きな inactive positive condensate driving を残す rows があることだった。

### v1.4 inactive driving hotspots

| family / layer | v1.4 max inactive driving |
|---|---:|
| `solar_water_condensation` layer 7 | 3505 |
| `lowT_strong_condensation_budget_stress` layer 7 | 2130 |
| `carbon_rich_CaS_MgS_AlN_window` layer 7 | 1022 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` layer 5 | 821 |
| `SiO_s_condensate_window` layer 8 | 787 |

### v1.4 FastChem4 comparison highlights

| family | Jaccard mean/min | max `|log10 gas ratio|` |
|---|---:|---:|
| `solar_water_condensation` | 0.130 / 0.102 | 163 |
| `lowT_strong_condensation_budget_stress` | 0.131 / 0.102 | 105 |
| `carbon_rich_CaS_MgS_AlN_window` | 0.156 / 0.124 | 35.7 |
| `SiO_s_condensate_window` | 0.160 / 0.121 | 32.5 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` | 0.154 / 0.118 | 39.4 |
| `solar_highT_no_condensate_gas_regression` | 0.972 / 0.500 | 0.565 |

## Trial v1.5.1: support-closure retry gate

### 内容

v1.5.1 は、support-cap retry と support-growth staging retry の候補を採用する前に、候補の ExoGibbs-native gas state から inactive positive condensate driving を再評価する trial である。

Provenance:

| item | value |
|---|---|
| source tree | working tree after HEAD route v1.5.1 implementation |
| exact-v1.4 rerun | no |
| v1.4-like comparison | current code with `enable_support_closure_retry_gate=False` |
| comparison artifacts | `volatiles_artifacts/fastchem4_head_route_v13_support_free_comparison_summary.md`, `volatiles_artifacts/head_route_v13_fastchem4_deep_comparison_summary.md` |
| artifact commit policy | `volatiles_artifacts/` and `volatiles_code/` remain scratch-only and are not commit targets |
| machine-readable score | `score.json` |

変更内容:

- `support_cap_retry` / `support_growth_staging_retry` の promoted candidate に support-closure gate を追加。
- `max_positive_inactive_driving <= 5.0e2` の候補だけ採用。
- gate fail の promoted candidate は採用せず、次の cap/staging 候補へ進む。
- retry candidate の例外は outer loop 全体を落とさず、failed attempt として diagnostics に記録。
- FastChem4 active list、runtime trace、public values は gate に使わない。

実装上の default option:

```text
enable_support_closure_retry_gate = True
support_closure_max_positive_inactive_driving = 5.0e2
```

### Public convergence score

| metric | v1.4 | v1.5.1 | delta |
|---|---:|---:|---:|
| curated full-profile rows | 99 | 99 | 0 |
| public converged | 99 | 99 | 0 |
| public not_converged | 0 | 0 | 0 |
| public budget gate rejects | 0 | 0 | 0 |
| primary route | 82 | 82 | 0 |
| gas-only route | 17 | 17 | 0 |
| native fallback route | 0 | 0 | 0 |
| support-changed rows | 0 | 13 | +13 |
| support-increased rows | 0 | 13 | +13 |

### ExoGibbs-native Gibbs before/after

同じ現行コードで、support-closure gate 有効を v1.5.1、無効を v1.4-like として再計算した。これは exact v1.4 checkout の再実行ではなく、support-closure gate 以外の v1.5.1 周辺変更は現行コードに残ったままの比較である。

| metric | score |
|---|---:|
| compared layers | 99 |
| finite comparisons | 99 |
| v1.5.1 lower `G/RT` | 13 |
| v1.5.1 higher `G/RT` | 0 |
| unchanged | 86 |
| sum `G_v1.5.1 - G_v1.4_like` | -0.0021467 |
| min `G_v1.5.1 - G_v1.4_like` | -7.1287e-4 |
| max `G_v1.5.1 - G_v1.4_like` | 0 |

Largest `G/RT` decreases:

| family / layer | support v1.4-like -> v1.5.1 | `G_v1.5.1 - G_v1.4_like` |
|---|---:|---:|
| `carbon_rich_CaS_MgS_AlN_window` layer 2 | 14 -> 48 | -7.1287e-4 |
| `solar_water_condensation` layer 7 | 34 -> 142 | -5.2964e-4 |
| `solar_water_condensation` layer 5 | 34 -> 147 | -2.5305e-4 |
| `solar_water_condensation` layer 0 | 34 -> 128 | -1.7918e-4 |
| `lowT_strong_condensation_budget_stress` layer 7 | 34 -> 134 | -1.7311e-4 |
| `lowT_strong_condensation_budget_stress` layer 2 | 34 -> 80 | -7.8807e-5 |

Interpretation: v1.5.1 does not raise the ExoGibbs-native Gibbs objective on any curated layer. The rows changed by the support-closure gate move to larger supports and lower `G/RT`.

### Targeted inactive-driving hotspot score

| family / layer | v1.4 max inactive driving | v1.5.1 max inactive driving | delta | improvement |
|---|---:|---:|---:|---:|
| `solar_water_condensation` layer 7 | 3505 | 0 | -3505 | full closure |
| `lowT_strong_condensation_budget_stress` layer 7 | 2130 | 19.7 | -2110.3 | 108x lower |
| `carbon_rich_CaS_MgS_AlN_window` layer 7 | 1022 | 487 | -535 | 2.1x lower |
| `solar_metal_sulfide_or_Fe_Ni_S_region` layer 5 | 821 | 191 | -630 | 4.3x lower |
| `SiO_s_condensate_window` layer 8 | 787 | 111 | -676 | 7.1x lower |

Interpretation: the trial directly improves the intended failure mode. It does not prove all support closure issues are solved, because water layer 8 and layer 2 still show large inactive driving in the deep comparison.

This table is a targeted hotspot score, not the global maximum over all 99 layers. In the v1.5.1 deep comparison, `solar_water_condensation` layer 8 still has max inactive driving about 2579 and layer 2 about 1662.

### FastChem4 gas comparison

| family | v1.4 max `|log10 gas ratio|` | v1.5.1 max `|log10 gas ratio|` | delta |
|---|---:|---:|---:|
| `solar_water_condensation` | 163 | 121 | -42 |
| `lowT_strong_condensation_budget_stress` | 105 | 30.4 | -74.6 |
| `carbon_rich_CaS_MgS_AlN_window` | 35.7 | 25.8 | -9.9 |
| `SiO_s_condensate_window` | 32.5 | 32.5 | 0 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` | 39.4 | 39.4 | 0 |
| `solar_highT_no_condensate_gas_regression` | 0.565 | 0.565 | 0 |

Interpretation: gas abundance outliers improve for water, lowT, and carbon-rich CaS/MgS/AlN, but not for all families.

### Active condensate Jaccard

| family | v1.4 mean/min | v1.5.1 mean/min | delta mean |
|---|---:|---:|---:|
| `solar_water_condensation` | 0.130 / 0.102 | 0.142 / 0.126 | +0.012 |
| `lowT_strong_condensation_budget_stress` | 0.131 / 0.102 | 0.142 / 0.103 | +0.011 |
| `carbon_rich_CaS_MgS_AlN_window` | 0.156 / 0.124 | 0.147 / 0.124 | -0.009 |
| `SiO_s_condensate_window` | 0.160 / 0.121 | 0.164 / 0.121 | +0.004 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` | 0.154 / 0.118 | 0.157 / 0.118 | +0.003 |
| `solar_highT_no_condensate_gas_regression` | 0.972 / 0.500 | 0.972 / 0.500 | 0 |

Interpretation: v1.5.1 is not primarily a FastChem4 active-list matching trial. Jaccard is mostly flat and remains low for condensation-heavy families.

### FastChem4-scaled Gibbs comparison after v1.5.1

FastChem4 state is scaled to the ExoGibbs element-budget gauge and both states are evaluated with the same ExoGibbs-native ideal gas + condensate objective.

| metric | score |
|---|---:|
| compared layers | 99 |
| ExoGibbs state lower `G/RT` | 17 |
| FastChem4-scaled state lower `G/RT` | 82 |
| max `|dG/RT|` | 9.8443e-4 |

Interpretation: v1.5.1 lowers `G/RT` relative to the v1.4-like support path where it changes rows, but FastChem4-scaled states still have lower ExoGibbs-native `G/RT` on most layers. The remaining difference is small in absolute `G/RT` but systematic.

## Current verdict

v1.5.1 is a good local improvement:

- It preserves public convergence and budget-gate behavior.
- It never raises ExoGibbs-native `G/RT` in the 99-layer gate on/off comparison.
- It strongly reduces the intended inactive-driving hotspots.
- It reduces several large FastChem4 gas abundance outliers.

It is not yet a complete FastChem4 agreement improvement:

- Active condensate Jaccard remains low for most condensation families.
- Some water layers still have large inactive driving in deep comparison.
- FastChem4-scaled states still have lower ExoGibbs-native `G/RT` in 82/99 layers.

## Next trial candidates

### Candidate A: residual support closure after accepted result

Goal: reduce the remaining large inactive driving rows, especially `solar_water_condensation` layer 8 and layer 2.

Required score checks:

- public convergence remains 99/99;
- no `G/RT` increases relative to v1.5.1;
- max inactive driving decreases on remaining hotspots;
- water max `|log10 gas ratio|` decreases below 121;
- support size does not explode without corresponding `G/RT` decrease.

### Candidate B: amount-floor complementarity repair

Goal: rows where support species are technically present but at near-zero amounts should not leave large positive driving because the amount floor effectively removes them.

Required score checks:

- top inactive driving rows before/after;
- active count under floors `0`, `1e-100`, `1e-50`, `1e-20`;
- `G/RT` before/after;
- budget residual and full-budget gate acceptance.

### Candidate C: FastChem4 gas trace outlier audit

Goal: identify whether remaining gas outliers are caused by support closure, amount floors, gas thermochemistry differences, or objective gauge/scaling.

Required score checks:

- top gas `|log10 ratio|` rows by species/family/layer;
- whether the outlier species involves elements in remaining inactive condensates;
- ExoGibbs-native driving of condensates containing those elements;
- `G/RT` and relative budget residual for the same rows.
