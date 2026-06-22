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

## Score 記録ルール

今後の trial では、全体 summary だけでなく case-by-case delta を
`score.md` と `score.json` の両方に保存する。`volatiles_artifacts/` は
scratch artifact なので、後から上書きされても scorecard だけで「どの
curated case がどう変わったか」を追跡できる状態にする。

各 trial の case-by-case delta には、少なくとも次を family ごとに残す。

| field | 内容 |
|---|---|
| rows / status counts / route counts | public route surface が case 単位で変わったか。 |
| max `|dG/RT|` before/after/delta | FastChem4-scaled comparison の Gibbs gap がどの case で改善・悪化したか。 |
| Exo temperature-valid max inactive driving before/after/delta | support closure が case 単位で閉じたか。 |
| temperature-valid inactive count / rows above thresholds | max だけでなく残存数と threshold exceedance を追う。 |
| active support count summary | support が過剰に膨らんでいないか。 |
| top changed layers | case 内の代表 layer と、その layer の before/after metrics。 |

`score.json` では各 trial に `case_by_case_delta` block を置く。過去 trial で
before 側の case-by-case artifact が保存されていない場合は、推測で埋めず、
`case_by_case_delta_available: false` と理由を明記する。

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

### v1.5.1 extended tracking metrics

The following metrics are added before the next implementation trial so that
future changes can be judged against a fixed v1.5.1 score surface. These values
come from existing scratch artifacts and are not a new implementation run.

Provenance:

| artifact | used for |
|---|---|
| `volatiles_artifacts/head_route_v13_fastchem4_deep_comparison.json` | global inactive-driving, residual component, support-size, and FastChem4-scaled Gibbs distributions |
| `volatiles_artifacts/fastchem4_head_route_v13_support_free_comparison.json` | amount-floor active-count / Jaccard sensitivity and gas outlier rows |
| `volatiles_artifacts/inactive_driving_support_closure_audit.json` | support-closure retry hotspot variant behavior |

Global ExoGibbs inactive-driving distribution over 99 layers:

| metric | value |
|---|---:|
| max positive inactive driving min / p50 / p90 / p95 / max | 0 / 203 / 762 / 769 / 2579 |
| mean max positive inactive driving | 306 |
| positive inactive count min / p50 / p90 / p95 / max | 0 / 5 / 42.8 / 68.2 / 80 |
| rows with max driving > 0 / >10 / >100 / >500 / >1000 | 86 / 81 / 68 / 21 / 2 |

Top remaining inactive-driving rows:

| family / layer | max inactive driving | inactive count | active count | `dG/RT` Exo-FC | Exo rel budget L2 |
|---|---:|---:|---:|---:|---:|
| `solar_water_condensation` layer 8 | 2579 | 68 | 69 | 2.8823e-4 | 1.4855e-5 |
| `solar_water_condensation` layer 2 | 1662 | 78 | 58 | 1.4896e-4 | 7.5537e-5 |
| `solar_highT_no_condensate_gas_regression` layer 17 | 771.7 | 5 | 1 | 3.6738e-5 | 1.1797e-5 |
| `solar_highT_no_condensate_gas_regression` layer 16 | 770.9 | 5 | 0 | -9.6847e-11 | 1.9355e-8 |
| `solar_highT_no_condensate_gas_regression` layer 15 | 769.8 | 5 | 0 | 2.2983e-10 | 1.3503e-4 |

Residual and support-size distribution over the same 99 layers:

| metric | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|
| Exo relative budget L2 | 7.694e-5 | 5.682e-4 | 8.211e-4 | 1.055e-3 |
| Exo relative budget max | 5.909e-5 | 4.820e-4 | 7.651e-4 | 9.676e-4 |
| Exo gas stationarity L2 | 33.0 | 323 | 419 | 864 |
| Exo gas stationarity max | 17.9 | 112 | 274 | 501 |
| Exo active condensate count | 48 | 147 | 156 | 184 |
| `|dG/RT Exo-FC|` | 2.671e-6 | 7.384e-5 | 4.745e-4 | 9.844e-4 |

Amount-floor sensitivity from the support-free comparison artifact:

| active floor | Exo active p50 / p95 / max | FastChem4 active p50 / max | intersection p50 / max | Jaccard mean / min / max |
|---|---:|---:|---:|---:|
| 0 | 48 / 155 / 184 | 15 / 21 | 10 / 21 | 0.3357 / 0.1033 / 1.0 |
| 1e-100 | 48 / 155 / 184 | 15 / 21 | 9 / 21 | 0.3360 / 0.1033 / 1.0 |
| 1e-50 | 48 / 155 / 184 | 15 / 21 | 9 / 21 | 0.3360 / 0.1033 / 1.0 |
| 1e-20 | 48 / 155 / 184 | 15 / 21 | 9 / 21 | 0.3375 / 0.1033 / 1.0 |

Largest gas-ratio attribution rows above gas abundance floor `1e-12`:

| family / layer | species | ExoGibbs gas x | FastChem4 gas x | log10 ratio |
|---|---|---:|---:|---:|
| `solar_water_condensation` layer 8 | `O2Ti1` | 2.447e-8 | 8.128e-130 | 121 |
| `solar_water_condensation` layer 8 | `Cl2O1Ti1` | 2.661e-8 | 2.985e-117 | 109 |
| `solar_water_condensation` layer 8 | `Cl3Ti1` | 9.500e-8 | 3.539e-110 | 102 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` layer 2 | `Al2O1` | 1.443e-6 | 5.496e-46 | 39.4 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` layer 2 | `Al2O2` | 1.089e-10 | 5.773e-50 | 39.3 |
| `SiO_s_condensate_window` layer 4 | `Al2O1` | 1.373e-6 | 4.406e-39 | 32.5 |
| `SiO_s_condensate_window` layer 4 | `Al2O2` | 1.345e-10 | 5.059e-43 | 32.4 |

Support-closure hotspot variants show that removing the early cap retry is not
uniformly better. For `carbon_rich_CaS_MgS_AlN_window` layer 7, no-cap/staged
variants reduce max inactive driving from 487 to 5.09 with support 121, but for
`solar_metal_sulfide_or_Fe_Ni_S_region` layer 5 the same no-cap/staged family
worsens max inactive driving from 191 to 822 with support 110. This makes
support-size increase alone an unsafe objective.

## Trial v1.6: actual-support outer-loop growth

### 内容

v1.6 は、support outer loop の support 更新規則を修正する trial である。v1.5.1 では、
outer loop が「solver に渡した support」を次 round の existing support として扱っていた。
そのため、solver が実際には support から落とした species も existing 扱いになり、final gas state
で再び activity-positive になっても再追加されなかった。

v1.6 では、non-fallback accepted result から support を grow するとき、次 round の existing support
を `last_result.condensate_support_indices`、つまり solver が実際に保持した support に変更する。
これにより、落とされた species がまだ ExoGibbs-native activity-positive なら、既存の
activity-driven support outer loop の通常規則で再追加される。

Provenance:

| item | value |
|---|---|
| source tree | working tree after actual-support outer-loop growth implementation |
| baseline | v1.5.1 score surface recorded above |
| comparison artifacts | `volatiles_artifacts/head_route_budget_consistency_audit.json`, `volatiles_artifacts/fastchem4_head_route_v13_support_free_comparison.json`, `volatiles_artifacts/head_route_v13_fastchem4_deep_comparison.json` |
| artifact commit policy | `volatiles_artifacts/` and `volatiles_code/` remain scratch-only and are not commit targets |

この変更は新しい retry policy や固定の追加個数を導入しない。既存の
`max_support_outer_iterations`、`max_support_add_per_round`、activity-driven support ordering、
budget seed をそのまま使う。FastChem4 active list、runtime trace、public values は constructor input
に使わない。

API exposure:

```text
result.head_route_version = "v1.6"
result.head_route_name = "head_route_v1_6_actual_support_outer_loop_growth"
```

`return_diagnostics=True` の場合は、diagnostics にも同じ `head_route_version` と
`head_route_name` を残す。

### Public convergence score

| metric | v1.5.1 | v1.6 | delta |
|---|---:|---:|---:|
| curated full-profile rows | 99 | 99 | 0 |
| public converged | 99 | 99 | 0 |
| public not_converged | 0 | 0 | 0 |
| public budget gate rejects | 0 | 0 | 0 |
| bookkeeping/gate mismatch rows | 0 | 0 | 0 |
| primary route | 82 | 82 | 0 |
| gas-only route | 17 | 17 | 0 |
| native fallback route | 0 | 0 | 0 |

### Inactive-driving before/after

| metric | v1.5.1 | v1.6 | delta |
|---|---:|---:|---:|
| global max positive inactive driving | 2579 | 771.7 | -1807 |
| global mean max positive inactive driving | 306 | 247 | -59 |
| rows with max driving > 500 | 21 | 18 | -3 |
| rows with max driving > 1000 | 2 | 0 | -2 |
| `solar_water_condensation` layer 8 max driving | 2579 | 22.4 | -2557 |
| `solar_water_condensation` layer 8 active count | 69 | 162 | +93 |
| `solar_water_condensation` layer 8 positive inactive count | 68 | 1 | -67 |
| `solar_water_condensation` layer 2 max driving | 1662 | 4.62 | -1657 |
| `solar_water_condensation` layer 2 active count | 58 | 129 | +71 |

Targeted water layer 8 diagnostics show that the ordinary support outer loop now
re-grows from actual solver support: the default run adds 64, then 64, then 34
support species and terminates with `no_inactive_positive_support`. No residual
support closure retry is used.

Global ExoGibbs inactive-driving distribution over 99 layers after v1.6:

| metric | value |
|---|---:|
| max positive inactive driving min / p50 / p90 / p95 / max | 0 / 185 / 758 / 766 / 771.7 |
| mean max positive inactive driving | 247 |
| positive inactive count min / p50 / p90 / p95 / max | 0 / 5 / 32.2 / 39.3 / 80 |
| rows with max driving > 0 / >10 / >100 / >500 / >1000 | 86 / 80 / 62 / 18 / 0 |

Top remaining inactive-driving rows:

| family / layer | max inactive driving | inactive count | active count | `dG/RT` Exo-FC | Exo rel budget L2 |
|---|---:|---:|---:|---:|---:|
| `solar_highT_no_condensate_gas_regression` layer 17 | 771.7 | 5 | 1 | 3.6738e-5 | 1.1797e-5 |
| `solar_highT_no_condensate_gas_regression` layer 16 | 770.9 | 5 | 0 | -9.6847e-11 | 1.9355e-8 |
| `solar_highT_no_condensate_gas_regression` layer 15 | 769.8 | 5 | 0 | 2.2983e-10 | 1.3503e-4 |
| `solar_highT_no_condensate_gas_regression` layer 14 | 768.6 | 5 | 0 | 2.8640e-10 | 2.1460e-8 |
| `solar_highT_no_condensate_gas_regression` layer 13 | 767.5 | 5 | 0 | 1.5592e-10 | 2.8595e-5 |

### FastChem4 comparison after v1.6

The support-free FastChem4 comparison improves several large gas outliers,
especially the water family.

| family | max `|log10 gas ratio|` | Jaccard mean/min at floor 0 | Jaccard mean/min at floor 1e-50 |
|---|---:|---:|---:|
| `solar_water_condensation` | 23.3 | 0.139 / 0.126 | 0.139 / 0.126 |
| `lowT_strong_condensation_budget_stress` | 30.4 | 0.142 / 0.103 | 0.142 / 0.103 |
| `carbon_rich_CaS_MgS_AlN_window` | 25.8 | 0.147 / 0.124 | 0.147 / 0.124 |
| `SiO_s_condensate_window` | 32.5 | 0.163 / 0.121 | 0.163 / 0.121 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` | 25.0 | 0.154 / 0.118 | 0.154 / 0.118 |
| `solar_highT_no_condensate_gas_regression` | 0.565 | 0.972 / 0.500 | 0.972 / 0.500 |

FastChem4-scaled Gibbs comparison:

| metric | v1.5.1 | v1.6 | delta |
|---|---:|---:|---:|
| compared layers | 99 | 99 | 0 |
| ExoGibbs state lower `G/RT` | 17 | 19 | +2 |
| FastChem4-scaled state lower `G/RT` | 82 | 80 | -2 |
| max `|dG/RT|` | 9.844e-4 | 7.952e-4 | -1.892e-4 |

Interpretation: v1.6 is a more principled support-closure improvement than the
rejected residual retry approach. It fixes both water inactive-driving hotspots
through the existing support outer loop, reduces the water gas-ratio outlier,
and slightly improves the FastChem4-scaled Gibbs comparison while preserving
public convergence and route counts.

## Current verdict

v1.7 keeps the v1.6 algorithmic surface and adds a validity-aware diagnostic
readout for inactive condensate driving. The solver route score remains the
v1.6 score:

- It preserves public convergence and budget-gate behavior.
- It removes the remaining `>1000` inactive-driving rows without adding a new ad hoc retry.
- It fixes the water layer 8 and layer 2 support-closure hotspots using the normal outer loop.
- It reduces the water FastChem4 gas-ratio outlier from 121 to 23.3.
- It improves the FastChem4-scaled Gibbs score from 17/82 to 19/80 and lowers max `|dG/RT|`.
- It separates temperature-invalid highT gas-only diagnostic artifacts from
  temperature-valid inactive-driving support misses.

It is not yet a complete support-closure or FastChem4 agreement improvement:

- Active condensate Jaccard remains low for most condensation families.
- all-condensate inactive-driving diagnostics still show highT gas-only rows,
  but the v1.7 validity-aware audit below shows these are temperature-invalid
  condensates rather than temperature-valid support misses.
- FastChem4-scaled states still have lower ExoGibbs-native `G/RT` in 80/99 layers.

## Trial v1.7: validity-aware inactive-driving diagnostics

v1.7 checks whether the remaining highT gas-only inactive-driving rows are real
missing condensates or diagnostic artifacts near the no-condensate boundary.
The implementation adds a diagnostic-only
`evaluate_inactive_condensate_driving()` helper that reports both the legacy
all-condensate inactive-driving metric and a temperature-valid subset using the
same `temperature_validity_upper` metadata as HEAD support selection.

API exposure:

```text
result.head_route_version = "v1.7"
result.head_route_name = "head_route_v1_7_validity_aware_inactive_driving_diagnostics"
diagnostics["inactive_condensate_driving"] = {
    "all_condensates": ...,
    "temperature_valid_condensates": ...,
}
```

This is a diagnostics-only HEAD route update. It does not change route
selection, support growth, solver acceptance, gas-only API defaults, or
FastChem4 constructor inputs.

Artifacts:

| artifact | purpose |
|---|---|
| `volatiles_artifacts/highT_gas_only_head_route_audit.json` | direct gas-only solver vs HEAD empty-support route comparison |
| `volatiles_artifacts/head_route_v13_fastchem4_deep_comparison.json` | full-profile FastChem4 output comparison with validity-aware inactive-driving fields |

highT gas-only route comparison over `solar_highT_no_condensate_gas_regression`:

| metric | value |
|---|---:|
| rows | 18 |
| HEAD route counts | 17 gas-only, 1 primary |
| HEAD status counts | 18 converged |
| strict gas-only budget retry rows | 2 |
| max all-condensate inactive driving, HEAD | 771.7 |
| max all-condensate inactive driving, direct gas-only default | 772.0 |
| max temperature-valid inactive driving, HEAD | 0 |
| rows with final temperature-valid inactive support | 0 |

Interpretation: the highT hotspot is not evidence that the HEAD route is
missing temperature-valid condensates. The largest all-condensate inactive
driving species are `H2S(s,l)`, `CH4(s,l)`, `CO2(s,l)`, and `N2(s,l)`, all of
which are above their condensate temperature-validity range at 2200 K. The
direct gas-only solver and the HEAD empty-support route show the same
all-condensate driving surface; HEAD support selection correctly excludes those
species through the standard temperature-validity gate.

Full-profile FastChem4 comparison after adding v1.7 validity-aware diagnostics:

| metric | value |
|---|---:|
| rows | 99 |
| public status | 99 converged |
| route counts | 82 primary, 17 gas-only |
| Exo all-condensate max inactive driving | 771.7 |
| Exo temperature-valid max inactive driving | 487.1 |
| Exo rows with temperature-valid inactive driving >500 | 0 |
| Exo rows with temperature-valid inactive driving >1000 | 0 |
| FastChem4-scaled temperature-valid max inactive driving | 21.6 |
| ExoGibbs lower `G/RT` vs FastChem4-scaled | 19/99 |
| FastChem4-scaled lower `G/RT` | 80/99 |
| max `|dG/RT|` | 7.952e-4 |

The next real support-closure targets are no longer the highT gas-only rows.
By temperature-valid inactive-driving, the largest remaining ExoGibbs rows are
in `carbon_rich_CaS_MgS_AlN_window`, `lowT_strong_condensation_budget_stress`,
and `SiO_s_condensate_window`, with top species such as `Cr23C6(s)` and
`Ti4O7(s,l)`. Those should be investigated as support/amount/complementarity
issues, not as gas-only boundary problems.

## Trial v1.8: best support-closure retry candidate

v1.8 changes fallback-only retry adoption from "first acceptable retry" to
"best inactive-closure retry". Support-cap retry and staged support-growth
retry now evaluate all configured candidates and select the converged candidate
with the lowest `(positive_inactive_count, max_positive_inactive_driving,
support_count)` score. The existing max-driving threshold remains a feasibility
filter. The positive-inactive count is recorded in the gate and may be used as
an explicit hard tolerance, but the default uses it as the primary ranking
metric rather than as a convergence-breaking hard reject.

Motivation: in v1.7, `SiO_s_condensate_window` layer 4 accepted the cap-34 retry
because its max inactive driving was 470.8, below the 500 gate. That candidate
still left 77 temperature-valid inactive condensates and produced the largest
FastChem4-scaled Gibbs gap. The cap-128 candidate converges, closes
temperature-valid inactive driving, and is selected by v1.8.

Target row score:

| metric | v1.7 cap-34 candidate | v1.8 selected cap-128 candidate |
|---|---:|---:|
| row | `SiO_s_condensate_window` layer 4 | same |
| support count | 5 | 72 |
| temperature-valid max inactive driving | 470.8 | 0 |
| temperature-valid inactive count | 77 | 0 |
| `dG/RT` Exo-FC | 7.952e-4 | 5.905e-6 |

Full-profile FastChem4 comparison after v1.8:

| metric | v1.7 | v1.8 |
|---|---:|---:|
| rows | 99 | 99 |
| public status | 99 converged | 99 converged |
| route counts | 82 primary, 17 gas-only | 82 primary, 17 gas-only |
| Exo temperature-valid max inactive driving | 487.1 | 224.3 |
| Exo rows with temperature-valid inactive driving >500 | 0 | 0 |
| ExoGibbs lower `G/RT` vs FastChem4-scaled | 19/99 | 19/99 |
| FastChem4-scaled lower `G/RT` | 80/99 | 80/99 |
| max `|dG/RT|` | 7.952e-4 | 9.973e-5 |

The largest remaining Gibbs gap after v1.8 is
`solar_metal_sulfide_or_Fe_Ni_S_region` layer 5 with
`dG/RT Exo-FC = 9.972649e-5`. The largest remaining temperature-valid inactive
driving is in `carbon_rich_CaS_MgS_AlN_window` with max 224.3. This keeps the
next investigation focused on real condensation-family support/amount closure,
not on highT gas-only artifacts.

## Trial v1.9: cross-retry best support-closure candidate

v1.9 generalizes the v1.8 retry selection policy across retry families. In
v1.8, support-cap retry could already select the best support-closure candidate
within its configured cap sequence, but an acceptable support-cap candidate
prevented staged support-growth retry from being compared. v1.9 evaluates both
support-cap retry and staged support-growth retry candidates when the default
support-free path falls back, then selects the converged candidate with the
lowest `(positive_inactive_count, max_positive_inactive_driving, support_count)`
score across both retry kinds.

Motivation: after v1.8, the largest Gibbs gap was
`solar_metal_sulfide_or_Fe_Ni_S_region` layer 5. The support-cap candidate was
acceptable under the max-driving gate, but it still left temperature-valid
inactive condensates. The staged support-growth candidate closed the inactive
support set and reduced the FastChem4-scaled Gibbs gap by about two orders of
magnitude without changing public convergence or route counts.

Target row score:

| metric | v1.8 selected cap-128 candidate | v1.9 selected staged candidate |
|---|---:|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 5 | same |
| retry kind | `support_cap_retry` | `support_growth_staging_retry` |
| retry parameter | cap 128 | `add_per_round=32` |
| support count | 73 | 95 |
| temperature-valid max inactive driving | 60.55 | 0 |
| temperature-valid inactive count | 12 | 0 |
| `dG/RT` Exo-FC | 9.972649e-5 | 1.193677e-6 |

Full-profile FastChem4 comparison after v1.9:

| metric | v1.8 | v1.9 |
|---|---:|---:|
| rows | 99 | 99 |
| public status | 99 converged | 99 converged |
| route counts | 82 primary, 17 gas-only | 82 primary, 17 gas-only |
| Exo temperature-valid max inactive driving | 224.3 | 194.8 |
| Exo rows with temperature-valid inactive driving >0 | 16 | 11 |
| Exo rows with temperature-valid inactive driving >500 | 0 | 0 |
| ExoGibbs lower `G/RT` vs FastChem4-scaled | 19/99 | 19/99 |
| FastChem4-scaled lower `G/RT` | 80/99 | 80/99 |
| max `|dG/RT|` | 9.973e-5 | 5.936e-5 |

Largest case-by-case improvements:

| family / layer | max `|dG/RT|` v1.8 | max `|dG/RT|` v1.9 | temperature-valid inactive v1.8 | temperature-valid inactive v1.9 | support v1.8 -> v1.9 |
|---|---:|---:|---:|---:|---:|
| `solar_metal_sulfide_or_Fe_Ni_S_region` layer 5 | 9.972649e-5 | 1.193677e-6 | 60.5 / 12 | 0 / 0 | 73 -> 95 |
| `carbon_rich_CaS_MgS_AlN_window` layer 3 | 2.240725e-5 | 7.135055e-6 | 224.3 / 40 | 0 / 0 | 48 -> 82 |
| `carbon_rich_graphite_window` layer 7 | 5.502966e-5 | 5.396749e-5 | 33.2 / 17 | 0 / 0 | 48 -> 66 |
| `near_phase_boundary_support_sensitivity` layer 8 | 3.755240e-5 | 3.683722e-5 | 14.0 / 18 | 0 / 0 | 34 -> 39 |
| `solar_silicate_first_condensation` layer 8 | 3.130483e-5 | 3.059256e-5 | 23.6 / 12 | 0 / 0 | 48 -> 53 |

The largest remaining Gibbs gap after v1.9 is
`carbon_rich_CaS_MgS_AlN_window` layer 2 with
`dG/RT Exo-FC = 5.935862e-5`; its temperature-valid inactive driving is already
0 / 0, so the next Gibbs-energy investigation should separate amount/budget
surface differences from support-closure misses. A small regression is recorded
for `carbon_rich_graphite_window` layer 4
(`|dG/RT|` 4.207600e-5 -> 4.264412e-5) with unchanged temperature-valid
inactive driving 0 / 0.

The v1.9 score record includes a `case_by_case_delta` block in `score.json`.
Future trials should keep this before/after block so family-level regressions
and improvements are visible even when the global max improves.

## Trial v1.10: scalar step-control retry candidate

v1.10 integrates the PD-IPM scalar fraction-to-boundary step-control policy into
the HEAD route without replacing the normal primary default. The default primary
continuation still starts with `component_clip`; when a support-free row falls
back to `native_budget_seed_fallback_budget_tradeoff`, the route now evaluates
one additional fresh API retry family with
`head_route_primary_step_control_policy="scalar_fraction_to_boundary"`.

The scalar retry is compared with support-cap retry and staged support-growth
retry by the same v1.9 closure score:
`(positive_inactive_count, max_positive_inactive_driving, support_count)`.

Target row score:

| metric | v1.9 staged candidate | v1.10 scalar candidate |
|---|---:|---:|
| row | `solar_water_condensation` layer 8 | same |
| retry kind | `support_growth_staging_retry` | `scalar_step_control_retry` |
| support count | 162 | 143 |
| temperature-valid max inactive driving | 0 | 0 |
| temperature-valid inactive count | 0 | 0 |
| public status | converged | converged |

Curated end-to-end validation after v1.10:

| metric | result |
|---|---:|
| curated end-to-end tests | 8 passed |
| warnings | 26 |
| runtime | 751.17 s |

This confirms that the scalar policy can be integrated as a retry candidate while
preserving the curated public convergence surface. It also increases runtime on
fallback-heavy rows because an additional fresh API retry family is evaluated.

Full-profile FastChem4 comparison after v1.10:

| metric | v1.9 | v1.10 |
|---|---:|---:|
| rows | 99 | 99 |
| public status | 99 converged | 99 converged |
| route counts | 82 primary, 17 gas-only | 82 primary, 17 gas-only |
| Exo temperature-valid max inactive driving | 194.8 | 194.8 |
| Exo rows with temperature-valid inactive driving >0 | 11 | 10 |
| Exo rows with temperature-valid inactive driving >500 | 0 | 0 |
| ExoGibbs lower `G/RT` vs FastChem4-scaled | 19/99 | 19/99 |
| FastChem4-scaled lower `G/RT` | 80/99 | 80/99 |
| max `|dG/RT|` | 5.936e-5 | 5.936e-5 |

The largest remaining Gibbs gap after v1.10 remains
`carbon_rich_CaS_MgS_AlN_window` layer 2 with
`dG/RT Exo-FC = 5.935862e-5` and temperature-valid inactive driving 0 / 0. The
largest remaining temperature-valid inactive driving is
`solar_water_condensation` layer 0 with max 194.826 / 11 species.

## Diagnostic trial: scalar primary default and centering decomposition

The scalar fraction-to-boundary policy was first tested as a public primary
default candidate after v1.10. Under the old production-safe acceptance rule it
was not promoted: explicit-support lifecycle rows improved, but support-free
curated regression rows lost existing v1.10 repairs when scalar replaced
`component_clip` as the normal primary default.

Original primary-default switch validation under the v1.10 contract:

| check | result |
|---|---:|
| `pytest -q tests/unittests` | 338 passed, 1 skipped |
| `pytest -q tests/endtoend/curated_cases` with scalar primary default | 6 failed, 2 passed |
| failure mode | support-free water regressions returned to `not_converged`; several fixed status/route contracts changed |
| original decision | keep v1.10 public default as `component_clip`; keep scalar as retry candidate |

PD-IPM step-control lifecycle comparison was rerun from fresh ExoGibbs
restricted payloads:

| variant | route converged | primary converged | stopped reasons | median final residual |
|---|---:|---:|---|---:|
| `component_clip_default` | 0 / 14 | 0 / 14 | `no_p_armijo_trial`: 13, `current_barrier_not_centered`: 1 | 3.725e1 |
| `scalar_fraction_to_boundary` | 12 / 14 | 12 / 14 | `final_barrier_centered`: 12, `current_barrier_not_centered`: 2 | 4.386e-5 |

The two remaining scalar `current_barrier_not_centered` rows were decomposed
with relaxed center tolerance and filter/merit variants:

| variant | route converged | primary converged | stopped reasons | median last center ratio |
|---|---:|---:|---|---:|
| `scalar_baseline` | 0 / 2 | 0 / 2 | `current_barrier_not_centered`: 2 | 2.249 |
| `scalar_relaxed_center` | 0 / 2 | 0 / 2 | `current_barrier_not_centered`: 2 | 1.538 |
| `scalar_filter_current_scale` | 0 / 2 | 0 / 2 | `current_barrier_not_centered`: 2 | 2.249 |
| `scalar_ipopt_persistent_filter` | 1 / 2 | 1 / 2 | `final_barrier_centered`: 1, `no_p_armijo_trial`: 1 | 0.674 |
| `scalar_ipopt_relaxed_center` | 1 / 2 | 1 / 2 | `final_barrier_centered`: 1, `no_p_armijo_trial`: 1 | 0.569 |

Interpretation:

- Relaxing the barrier center gate alone is insufficient.
- The graphite row is repaired by the IPOPT persistent-filter style variant.
- The heavy/Ti/Zr row moves from `current_barrier_not_centered` to
  `no_p_armijo_trial`, so the remaining blocker is filter/merit trial
  acceptance rather than only the barrier update threshold.

Artifacts:

- `volatiles_artifacts/pdipm_step_control_curated_comparison.json`
- `volatiles_artifacts/pdipm_step_control_curated_comparison.md`
- `volatiles_artifacts/scalar_centering_failure_decomposition.json`
- `volatiles_artifacts/scalar_centering_failure_decomposition.md`

FastChem4 full-profile output comparison was rerun after this diagnostic change
with the public default restored to v1.10. That comparison matched the v1.10
surface: 99/99 public converged, 82 primary + 17 gas-only, 19/99
ExoGibbs-lower vs 80/99 FastChem4-scaled-lower, max `|dG/RT|` 5.935862e-5,
and no temperature-valid inactive row above 500.

## Trial v1.11: PD-IPM scalar primary baseline

v1.11 changes the policy decision. The goal is no longer to hide scalar-primary
regressions behind the v1.10 component-clipped default. Instead, the public
HEAD route now uses `scalar_fraction_to_boundary` as the primary step-control
policy so PD-IPM/R-GIE is the main route again. The known metric regressions are
accepted as solver blockers.

Implementation:

| item | value |
|---|---|
| `CONDENSATE_HEAD_ROUTE_VERSION` | `v1.11` |
| `CONDENSATE_HEAD_ROUTE_NAME` | `head_route_v1_11_pdipm_scalar_primary` |
| public primary step control | `scalar_fraction_to_boundary` |
| v1.10 role | production-safe comparison baseline |
| FastChem4 constructor inputs | not used |

Curated end-to-end validation after updating the expected blocker surface:

| check | result |
|---|---:|
| `pytest -q tests/endtoend/curated_cases` | 8 passed |
| warnings | 21 |
| runtime | 1026.41 s |
| expected support-free blockers | `solar_water_condensation` layers 0 and 7 |
| explicit-support blocker/caveat | graphite T1300 is hard reject; heavy/Ti/Zr remains caveat |

Full-profile FastChem4 comparison after v1.11:

| metric | v1.10 production-safe | v1.11 PD-IPM-first |
|---|---:|---:|
| rows | 99 | 99 |
| public status | 99 converged | 95 converged, 1 caveat, 3 not converged |
| route counts | 82 primary, 17 gas-only | 78 primary, 4 fallback, 17 gas-only |
| ExoGibbs lower `G/RT` vs FastChem4-scaled | 19/99 | 32/99 |
| FastChem4-scaled lower `G/RT` | 80/99 | 67/99 |
| max `|dG/RT|` | 5.936e-5 | 3.297e3 |
| max temperature-valid inactive driving | 194.8 | 343.2 |
| temperature-valid inactive rows >0 | 10 | 16 |
| temperature-valid inactive rows >500 | 0 | 0 |

The largest v1.11 regression is `solar_water_condensation` layer 0:
`dG/RT Exo-FC = -3297.0902626575994`, with relative budget L2
`1e+07 / 4.35e-07` for ExoGibbs/FastChem4-scaled states. This row is now the
primary PD-IPM-first blocker rather than a reason to restore the v1.10
component-clipped default.

### v1.11 blocker audit: Ipopt-oriented scalar-step limiter

The first blocker audit keeps the v1.11 PD-IPM-first policy and asks why the
largest water row fails under scalar fraction-to-boundary.  This was run from
fresh ExoGibbs API inputs; FastChem4 output remains comparison-only and is not
used as constructor input.

Implementation additions:

| item | purpose |
|---|---|
| fraction-to-boundary blocker report | records the limiting variable group, local support index, species name, raw alpha, safety alpha, and top active-support blockers |
| support species propagation | carries condensate species names from public API setup into lifecycle continuation diagnostics |
| Ipopt-oriented interpretation | treats tiny scalar alpha as an initial-point/dual/line-search blocker, not as a reason to restore component-wise clipping |

Target result for `solar_water_condensation` layer 0 with scalar primary and
support cap 128:

| metric | value |
|---|---:|
| public status | `not_converged` |
| selected route | `native_budget_seed_fallback_budget_tradeoff` |
| stop reason | `no_p_armijo_trial` |
| scalar alpha | `2.2303859960980942e-08` |
| limiting species | `FeS2(s)` |
| limiting variable | `r` |
| limiting raw alpha | `2.241593965927733e-08` |
| blocker count | 128 |

Top scalar-step blockers:

| rank | species | group | direction | safety alpha |
|---:|---|---|---:|---:|
| 1 | `FeS2(s)` | `r` | -44611112.235311896 | 2.2303859960980942e-08 |
| 2 | `FeS(s,l)` | `rho` | -39866972.657306716 | 2.4958002418516698e-08 |
| 3 | `MgS(s)` | `r` | -23917560.19974084 | 4.16012332232274e-08 |
| 4 | `MgO(s,l)` | `r` | -19682405.697240468 | 5.055276348355638e-08 |
| 5 | `Fe2O3(s)` | `r` | -12417390.506961612 | 8.012955696626994e-08 |

Verdict: the blocker is consistent with a poor active-support interior point
and dual state.  The next metric-moving work should implement Ipopt-like
primal/dual initialization and then revisit filter/restoration.  Adding another
support retry or restoring component-wise clipping would move away from the
PD-IPM-first objective.

## Trial v1.12: Ipopt-style dual push initialization

v1.12 keeps the v1.11 PD-IPM-first scalar primary path and repairs the largest
blocker by changing the active-support dual initialization, not by clipping the
Newton direction.  When `rho` is inferred from `epsilon`, the dual
`eta = exp(rho)` is pushed to a floor of `0.1`, matching the Ipopt idea that
multipliers should be initialized away from pathological boundaries.

Implementation:

| item | value |
|---|---|
| `CONDENSATE_HEAD_ROUTE_VERSION` | `v1.12` |
| `CONDENSATE_HEAD_ROUTE_NAME` | `head_route_v1_12_ipopt_dual_push_primary` |
| public primary step control | `scalar_fraction_to_boundary` |
| public dual initialization | `ipopt_push_floor` |
| dual push floor | `0.1` |
| FastChem4 constructor inputs | not used |

Curated end-to-end validation:

| check | result |
|---|---:|
| `pytest -q tests/endtoend/curated_cases` | 8 passed |
| warnings | 0 |
| former support-free blockers | `solar_water_condensation` layers 0 and 7 now converge |
| water layer 5 targeted audit | converged, tier 1, inactive closure `0 / 0` |

Full-profile FastChem4 comparison after v1.12:

| metric | v1.10 production-safe | v1.11 PD-IPM-first | v1.12 dual-push repair |
|---|---:|---:|---:|
| rows | 99 | 99 | 99 |
| public status | 99 converged | 95 converged, 1 caveat, 3 not converged | 99 converged |
| route counts | 82 primary, 17 gas-only | 78 primary, 4 fallback, 17 gas-only | 82 primary, 17 gas-only |
| ExoGibbs lower `G/RT` vs FastChem4-scaled | 19/99 | 32/99 | 38/99 |
| FastChem4-scaled lower `G/RT` | 80/99 | 67/99 | 61/99 |
| max `|dG/RT|` | 5.936e-5 | 3.297e3 | 9.632e-2 |
| max temperature-valid inactive driving | 194.8 | 343.2 | 1012.9 |
| temperature-valid inactive rows >0 | 10 | 16 | 3 |
| temperature-valid inactive rows >500 | 0 | 0 | 1 |

Largest remaining differences after v1.12:

| rank | row | `dG/RT Exo-FC` | rel budget L2 | valid inactive |
|---:|---|---:|---:|---:|
| 1 | `carbon_rich_graphite_window` layer 7 | -9.632193e-2 | 7.63e-6 | 0 / 0 |
| 2 | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | 4.692784e-3 | 5.86e-5 | 96 / 1012.945 |
| 3 | `carbon_rich_graphite_window` layer 3 | -5.578552e-4 | 2.27e-6 | 0 / 0 |

Verdict: v1.12 restores 99/99 public convergence while keeping PD-IPM scalar
step control as the main line.  The next work should target the remaining
support-closure / Gibbs-gap rows under the same PD-IPM-first contract.

## Trial v1.13: severe support-closure staged growth

v1.13 keeps the v1.12 PD-IPM scalar primary and dual push defaults.  The failed
attempt to promote persistent h-type filter/restoration directly into the
primary default regressed explicit-support and water rows, so it was not kept as
the public default.  Instead, v1.13 uses the PD-IPM final inactive-driving
diagnostic to trigger staged support-growth retry only for severe
temperature-valid support-closure failures.

Implementation:

| item | value |
|---|---|
| `CONDENSATE_HEAD_ROUTE_VERSION` | `v1.13` |
| `CONDENSATE_HEAD_ROUTE_NAME` | `head_route_v1_13_pdipm_staged_support_growth` |
| public primary step control | `scalar_fraction_to_boundary` |
| public dual initialization | `ipopt_push_floor` |
| severe closure retry trigger | temperature-valid max driving `>= 1000` and count `>= 50` |
| retry selection | best ExoGibbs-native support-closure score |
| FastChem4 constructor inputs | not used |

Validation:

| check | result |
|---|---:|
| target unit/API/PD-IPM tests | 79 passed, 1 warning |
| `pytest -q tests/endtoend/curated_cases` | 8 passed |
| target FastChem4 comparison | completed |
| full-profile FastChem4 comparison | attempted, stopped due runtime |

Runtime diagnostics:

| probe | wall time | result |
|---|---:|---|
| target solve, `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | 89.05 s | converged, support 67, inactive `0 / 0` |
| target metric evaluation after solve | 0.0011 s | ExoGibbs-native metrics only |
| `pytest -q tests/endtoend/curated_cases` | 400.55 s | 8 passed |
| `pytest -q tests/unittests` | 34.76 s | 341 passed, 1 skipped |
| `SiO_s_condensate_window` layer 0 gas solve + activity support report | 0.47 s | support selection is not the bottleneck |
| `SiO_s_condensate_window` layer 0 full solve probe | >180 s before interrupt | initial support 70; large active-support solve dominates |
| `SiO_s_condensate_window` layer 0 with `max_inner_iterations=5` | >90 s before interrupt | runtime is not explained only by inner iteration count |

Target row comparison:

| metric | v1.12 | v1.13 |
|---|---:|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | same |
| status | converged | converged |
| support count | 14 | 67 |
| temperature-valid inactive closure | `96 / 1012.945` | `0 / 0` |
| `dG/RT Exo-FC` | `4.692784e-3` | `2.210839e-6` |
| relative budget L2 | `5.86e-5` | `7.07e-4` |

Verdict: v1.13 repairs the largest v1.12 support-closure hotspot while keeping
the PD-IPM scalar primary line.  The full-profile comparison needs a longer
runtime window or a faster campaign driver; the target FastChem4 output
comparison was completed and did not use FastChem4 values as constructor input.

## Trial v1.14: PD-IPM core mainline

v1.14 changes the public primary continuation default to `pdipm_core`.  This is
a solver-development baseline: it makes the PD-IPM core the main line even when
the public metric surface worsens.

Implementation:

| item | value |
|---|---|
| `CONDENSATE_HEAD_ROUTE_VERSION` | `v1.14` |
| `CONDENSATE_HEAD_ROUTE_NAME` | `head_route_v1_14_pdipm_core_mainline` |
| primary continuation mode | `pdipm_core` |
| primary direction | `algorithm_v11_reduced` |
| step control | `scalar_fraction_to_boundary` |
| step diagnostics | primal / dual / combined fraction-to-boundary alphas |
| acceptance sequence | persistent h-type filter, soft restoration, dedicated restoration |
| HEAD route responsibility | active-set orchestration, support expansion, final gates |
| FastChem4 constructor inputs | not used |

Validation:

| check | result |
|---|---:|
| target API / PD-IPM unit tests | 80 passed, 7 warnings |
| `pytest -q tests/unittests` | 342 passed, 1 skipped, 8 warnings |
| explicit-support curated surface | 1 passed, 7 warnings |
| support-free midlayer curated test | interrupted after 330.18 s |
| full curated e2e | interrupted after 505.78 s |
| target FastChem4 comparison | completed |
| full-profile FastChem4 comparison | not run; current runtime is too high |

Runtime diagnostics:

| probe | wall time | result |
|---|---:|---|
| target solve, `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | 172.27 s | converged, support 52, inactive `4 / 23.120` |
| explicit-support curated surface | 20.44 s | 14 rows passed against v1.14 blocker surface |
| support-free midlayer test | >330.18 s before interrupt | runtime blocker |
| full curated e2e attempt | >505.78 s before interrupt | 1 failed before expectation update, 1 passed, then interrupted |
| `pytest -q tests/unittests` | 33.22 s | 342 passed, 1 skipped |

Explicit-support surface delta:

| metric | v1.13/v1.12-style expectation | v1.14 PD-IPM core |
|---|---:|---:|
| explicit support rows | 14 | 14 |
| converged | 12 | 8 |
| converged with caveat | 1 | 4 |
| not converged | 1 | 2 |
| notable improvement | graphite explicit-support row not converged | graphite explicit-support row converged |
| notable regressions | silicate, metal-sulfide P0.1, lowT P0.1, near-boundary rows tighter | blocker/caveat surface updated |

Target row comparison:

| metric | v1.13 | v1.14 |
|---|---:|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | same |
| status | converged | converged |
| support count | 67 | 52 |
| temperature-valid inactive closure | `0 / 0` | `4 / 23.120` |
| `dG/RT Exo-FC` | `2.210839e-6` | `2.195275e-6` |
| solve wall time | 89.05 s | 172.27 s |

Verdict: v1.14 succeeds at making the PD-IPM core path explicit and measurable,
but it is not a production metric win.  It exposes a new blocker surface and a
serious runtime blocker.  The next repair should reduce support-free runtime
and repair v1.14 blockers inside the PD-IPM/filter/restoration contract, not by
returning the main line to component-wise clipping.

## Trial v1.15: PD-IPM core tiny-step restoration

v1.15 keeps the v1.14 `pdipm_core` primary line and adds Ipopt-style tiny-step
handling.  If a primary reduced PD-IPM step has `alpha_primal <= 1.0e-8`, the
core mode records the tiny step and routes the iteration to restoration instead
of accepting a nearly zero primary update or returning to component-wise
clipping.

Implementation:

| item | value |
|---|---|
| `CONDENSATE_HEAD_ROUTE_VERSION` | `v1.15` |
| `CONDENSATE_HEAD_ROUTE_NAME` | `head_route_v1_15_pdipm_core_tiny_step_restoration` |
| primary continuation mode | `pdipm_core` |
| tiny-step alpha threshold | `1.0e-8` |
| consecutive tiny-step limit | `1` |
| tiny-step action | switch to restoration |
| FastChem4 constructor inputs | not used |

Validation:

| check | result |
|---|---:|
| py_compile targeted files | passed |
| target API / PD-IPM unit tests | 47 passed, 7 warnings |
| `pytest -q tests/unittests` | 342 passed, 1 skipped, 8 warnings |
| explicit-support curated surface | 1 passed, 7 warnings |
| support-free midlayer curated test | timed out at 240 s |
| target FastChem4 comparison | completed |
| full-profile FastChem4 comparison | not run; current runtime is too high |

Runtime diagnostics:

| probe | wall time | result |
|---|---:|---|
| target solve, `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | 186.35 s | converged, support 52, inactive `4 / 23.120` |
| explicit-support curated surface | 20.21 s | 14 rows passed against v1.15 blocker surface |
| support-free midlayer test | >240 s before timeout | runtime blocker remains |
| `pytest -q tests/unittests` | 34.45 s | 342 passed, 1 skipped |

Explicit-support surface delta:

| metric | v1.14 | v1.15 |
|---|---:|---:|
| explicit support rows | 14 | 14 |
| converged | 8 | 8 |
| converged with caveat | 4 | 4 |
| not converged | 2 | 2 |
| observed status delta | baseline | none |

Target row comparison:

| metric | v1.14 | v1.15 |
|---|---:|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | same |
| status | converged | converged |
| support count | 52 | 52 |
| temperature-valid inactive closure | `4 / 23.120` | `4 / 23.120` |
| `dG/RT Exo-FC` | `2.195275e-6` | `2.195275e-6` |
| solve wall time | 172.27 s | 186.35 s |

Verdict: v1.15 is a structural PD-IPM-mainline improvement, not a score win.
It imports an Ipopt-like tiny-step/restoration transition into `pdipm_core`, but
the visible blocker surface and target metrics are unchanged and runtime is
slightly worse on the target row.  The next repair should focus on
active-support solve cost and restoration effectiveness inside the PD-IPM
contract.

## Trial v1.16: PD-IPM core fast diagnostics

v1.16 keeps the v1.15 `pdipm_core` solver behavior and removes runtime blockers
from diagnostics and report serialization.  The reduced PD-IPM direction,
scalar fraction-to-boundary step control, dual push floor, filter/restoration
sequence, and tiny-step restoration policy are unchanged.

Implementation:

| item | value |
|---|---|
| `CONDENSATE_HEAD_ROUTE_VERSION` | `v1.16` |
| `CONDENSATE_HEAD_ROUTE_NAME` | `head_route_v1_16_pdipm_core_fast_diagnostics` |
| primary continuation mode | `pdipm_core` |
| solver math change | none |
| diagnostics change | manual report dictionaries instead of deep `dataclasses.asdict()` |
| inactive-driving diagnostics | JAX arrays are copied once and summarized with NumPy |
| support signature export | support/formula arrays are copied once before Python loops |
| FastChem4 constructor inputs | not used |

Validation:

| check | result |
|---|---:|
| py_compile targeted files | passed |
| `pytest -q tests/unittests` | 342 passed, 1 skipped, 8 warnings |
| explicit-support curated surface | 1 passed, 7 warnings, 10.32 s |
| support-free midlayer curated test | 1 passed, 8 warnings, 221.74 s |
| full curated e2e | 8 passed, 37 warnings, 352.32 s |
| target FastChem4 comparison | completed |
| full-profile FastChem4 comparison | completed, 99 layers |

Runtime diagnostics:

| probe | v1.15 | v1.16 |
|---|---:|---:|
| target solve, `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | 186.35 s | 76.61 s |
| support-free midlayer curated test | timed out at 240 s | 221.74 s |
| explicit-support curated surface | 20.21 s | 10.32 s |
| `solar_metal_sulfide_or_Fe_Ni_S_region` midlayer probe | 173.59 s | 73.17 s |
| `carbon_rich_CaS_MgS_AlN_window` midlayer probe | 128.30 s | 53.57 s |
| `SiO_s_condensate_window` midlayer probe | 80.53 s | 35.93 s |
| `near_phase_boundary_support_sensitivity` midlayer probe | 96.44 s | 45.73 s |

Explicit-support surface delta:

| metric | v1.15 | v1.16 |
|---|---:|---:|
| explicit support rows | 14 | 14 |
| converged | 8 | 8 |
| converged with caveat | 4 | 4 |
| not converged | 2 | 2 |
| observed status delta | baseline | none |

Support-free midlayer surface:

| metric | v1.15 | v1.16 |
|---|---:|---:|
| rows | 10 | 10 |
| converged | not completed before timeout | 9 |
| converged with caveat | not completed before timeout | 0 |
| not converged | not completed before timeout | 1 |
| remaining blocker | runtime timeout | `complex_heavy_element_or_boron_titanium_zirconium_case` |

Target row comparison:

| metric | v1.15 | v1.16 |
|---|---:|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 | same |
| status | converged | converged |
| support count | 52 | 52 |
| temperature-valid inactive closure | `4 / 23.120` | `4 / 23.120` |
| `dG/RT Exo-FC` | `2.195275e-6` | `2.195275e-6` |
| relative budget L2 | not recorded in v1.15 target table | `2.783856e-4` |
| solve wall time | 186.35 s | 76.61 s |

Full-profile FastChem4 comparison:

| metric | v1.16 |
|---|---:|
| artifact | `volatiles_artifacts/head_route_v16_fastchem4_deep_comparison.json` |
| layers | 99 |
| public status counts | `91 converged / 8 not_converged` |
| route counts | `79 primary / 17 gas-only / 3 native fallback` |
| ExoGibbs lower `G/RT` rows | 39 |
| FastChem4-scaled lower `G/RT` rows | 60 |
| max `abs(dG/RT Exo-FC)` | `9.767664e2` |
| max temperature-valid inactive driving | `9.645081e2` |
| temperature-valid rows > 500 | 2 |
| temperature-valid rows > 1000 | 0 |

Verdict: v1.16 is a runtime repair that preserves the PD-IPM mainline.  It does
not claim a new chemical metric improvement; instead it removes unnecessary
deep-copy and JAX diagnostic overhead that made v1.15 impractical to validate.
The next PD-IPM work should now return to the remaining blocker surface:
`complex_heavy_element_or_boron_titanium_zirconium_case`, support-free water
layers 7/8, graphite layers 0/1, and the remaining explicit-support
caveat/not-converged rows.

## Trial v1.17: PD-IPM full-budget restoration

v1.17 keeps the v1.16 `pdipm_core` mainline and adds final feasibility
restoration before the full-condensate budget gate:

- gas log-amount restoration for trace gas budget residuals;
- bounded least-squares active-support amount restoration for accepted primary
  states that still miss the full-budget gate;
- no FastChem4 public/runtime/trace values are used as constructor inputs.

Implementation:

| item | value |
|---|---|
| `CONDENSATE_HEAD_ROUTE_VERSION` | `v1.17` |
| `CONDENSATE_HEAD_ROUTE_NAME` | `head_route_v1_17_pdipm_core_full_budget_restoration` |
| primary continuation | unchanged from v1.16 `pdipm_core` |
| new restoration | gas log amount + active-support bounded LSQ amount |
| FastChem4 constructor inputs | not used |

Remaining-blocker probe:

| row | v1.16 status | v1.17 status |
|---|---|---|
| `carbon_rich_graphite_window` layer 0 | `not_converged` | `converged` |
| `carbon_rich_graphite_window` layer 1 | `not_converged` | `converged` |
| `carbon_rich_graphite_window` layer 2 | `not_converged` | `converged` |
| `complex_heavy_element_or_boron_titanium_zirconium_case` layer 4 | `not_converged` | `converged` |
| `near_phase_boundary_support_sensitivity` layer 7 | `not_converged` | `converged` |
| `solar_water_condensation` layer 5 | `not_converged` | `not_converged` |
| `solar_water_condensation` layer 7 | `not_converged` | `converged_with_caveat` |
| `solar_water_condensation` layer 8 | `not_converged` | `not_converged` |

Validation so far:

| check | result |
|---|---:|
| py_compile targeted files | passed |
| `pytest -q tests/unittests/api/condensate_equilibrium_test.py` | 38 passed, 5 warnings |
| `pytest -q tests/unittests` | 343 passed, 1 skipped, 8 warnings |
| `pytest -q tests/endtoend/curated_cases` | 8 passed, 43 warnings |
| remaining-blocker probe | completed |
| fixed rows in remaining-blocker probe | 6 / 8 |
| target FastChem4 comparison | completed |
| `python -m json.tool score.json` | passed |
| `git diff --check` | passed |

Target row comparison:

| metric | v1.17 |
|---|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 |
| status | converged |
| support count | 52 |
| temperature-valid inactive closure | `4 / 23.120` |
| `dG/RT Exo-FC` | `2.195275e-6` |
| relative budget L2 | `2.783856e-4` |
| solve wall time | 74.72 s |

Verdict: v1.17 repairs all remaining blocker rows that were failing only at the
final full-budget restoration layer.  The remaining water layers 5/8 are not
fixed by active-support restoration; they still fall to native seed fallback
with residual inactive driving, so the next PD-IPM work should focus on the
water low-temperature support-boundary / phase-selection path rather than
loosening final gates.

## v1.18 score update

HEAD route v1.18 keeps the PD-IPM-first v1.17 route and connects support
growth to the finite lifecycle final state when a native fallback result has no
restricted solver payload.  This is a support-boundary repair: the next support
round is driven by ExoGibbs PD-IPM continuation state, not FastChem4 replay or
constructor inputs.  The public fallback state also compares the original seed
fallback against the lifecycle final-state fallback with the full-budget gate:
an accepted seed is kept, while a failed seed may be replaced by a better
lifecycle final-state candidate.

Implementation summary:

| item | value |
|---|---|
| `CONDENSATE_HEAD_ROUTE_VERSION` | `v1.18` |
| `CONDENSATE_HEAD_ROUTE_NAME` | `head_route_v1_18_pdipm_lifecycle_support_growth` |
| new default option | `enable_lifecycle_final_state_support_growth=True` |
| support-growth state source | lifecycle `continuation_report.final_state` |
| fallback public-state selection | keep accepted seed; otherwise allow better lifecycle final state |
| native fallback joint restoration | gas log amount + active condensate amount feasibility restoration |
| water8 full-budget residual improvement | F max relative residual `3.593e7` -> accepted gate, max residual `4.998e-7` |
| FastChem4 constructor inputs | not used |

Remaining-blocker probe delta:

| row | v1.17 status / inactive | v1.18 status / inactive | support delta |
|---|---|---|---:|
| `carbon_rich_graphite_window` layer 0 | `converged`, `0 / 0.0` | `converged`, `0 / 0.0` | 5 -> 5 |
| `carbon_rich_graphite_window` layer 1 | `converged`, `0 / 0.0` | `converged`, `0 / 0.0` | 8 -> 8 |
| `carbon_rich_graphite_window` layer 2 | `converged`, `0 / 0.0` | `converged`, `0 / 0.0` | 19 -> 19 |
| `complex_heavy_element_or_boron_titanium_zirconium_case` layer 4 | `converged`, `0 / 0.0` | `converged`, `0 / 0.0` | 72 -> 72 |
| `near_phase_boundary_support_sensitivity` layer 7 | `converged`, `0 / 0.0` | `converged`, `0 / 0.0` | 54 -> 54 |
| `solar_water_condensation` layer 5 | `not_converged`, `5 / 53.575` | `converged_with_caveat`, `0 / 0.0` | 166 -> 197 |
| `solar_water_condensation` layer 7 | `converged_with_caveat`, `0 / 0.0` | `converged_with_caveat`, `0 / 0.0` | 199 -> 184 |
| `solar_water_condensation` layer 8 | `not_converged`, `9 / 729.347` | `converged_with_caveat`, `0 / 0.0` | 189 -> 208 |

Validation so far:

| check | result |
|---|---:|
| py_compile targeted files | passed |
| `pytest -q tests/unittests/api/condensate_equilibrium_test.py` | 43 passed, 5 warnings |
| `pytest -q tests/endtoend/curated_cases/test_fresh_api_curated_cases.py -k water_low_temperature` | 1 passed, 20 warnings, 90.99 s |
| `pytest -q tests/endtoend/curated_cases` | 8 passed, 46 warnings, 397.22 s |
| `pytest -q tests/unittests` | 347 passed, 1 skipped, 8 warnings, 73.71 s |
| remaining-blocker probe | completed |
| remaining not-converged rows in probe | 0 / 8 |
| target FastChem4 comparison | completed |
| `python -m json.tool score.json` | passed |
| `git diff --check` | passed |

Caveat route breakdown after adding `caveat_route_breakdown` diagnostics:

| row | caveat reason | primary stop | PD-IPM retry attempts | restoration path | fallback source |
|---|---|---|---|---|---|
| `solar_water_condensation` layer 5 | native budget tradeoff after primary lifecycle not accepted | `no_p_armijo_trial`; restoration count 38 | soft restoration rejected; Ipopt h-type rejected | gas-log polish accepted | selected warm-start seed |
| `solar_water_condensation` layer 7 | native budget tradeoff after primary lifecycle not accepted | `current_barrier_not_centered`; restoration count 32 | soft restoration rejected; Ipopt h-type rejected | joint gas+active restoration accepted, max residual `2.489e0 -> 9.831e-5` | lifecycle final state |
| `solar_water_condensation` layer 8 | native budget tradeoff after primary lifecycle not accepted | `no_p_armijo_trial`; restoration count 17 | soft restoration rejected; Ipopt h-type rejected | joint gas+active restoration accepted, max residual `1.000e-3 -> 4.998e-7` | selected warm-start seed |

Targeted validation for the diagnostic addition:

| check | result |
|---|---:|
| py_compile targeted API files | passed |
| `pytest -q tests/unittests/api/condensate_equilibrium_test.py` | 43 passed, 5 warnings |
| `pytest -q tests/endtoend/curated_cases/test_fresh_api_curated_cases.py -k water_low_temperature` | 1 passed, 7 deselected, 20 warnings, 90.99 s |
| `JAX_ENABLE_X64=1 python volatiles_code/probe_v16_remaining_blockers.py` | completed, 0 / 8 not_converged |

Follow-up experiment: automatically enabling the soft-restoration retry for these
primary stop reasons was tested and rejected for the default route because it
did not improve water5/7/8 status and raised the water low-temperature e2e
runtime to 197.52 s.  The default route keeps the existing explicit retry
behavior and records retry rejection in `caveat_route_breakdown`.

Target row comparison:

| metric | v1.18 |
|---|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 |
| status | converged |
| support count | 52 |
| temperature-valid inactive closure | `4 / 23.120` |
| `dG/RT Exo-FC` | `2.195275e-6` |
| relative budget L2 | `2.783856e-4` |
| solve wall time | 131.92 s |

Verdict: v1.18 repairs the water low-temperature support closure path and the
water8 full-budget feasibility blocker without FastChem4 replay.  Water layer 5
remains caveat converged, water layer 7 remains caveat converged, and water
layer 8 now moves from `not_converged` to `converged_with_caveat` with inactive
closure `0 / 0.0` and full-budget gate accepted.  The repair is a native
fallback-only joint feasibility restoration over gas log amounts and active
condensate amounts, not a species- or element-specific F patch.

## Trial v1.18-filter-selection: IpOpt-style filter selection repair

This trial keeps HEAD route version `v1.18` and fixes the line-search/filter
selection layer.  The previous diagnostic name `no_p_armijo_trial` was too
coarse: in `ipopt_persistent_h_type` mode it mixed actual P-Armijo failures
with h-filter, persistent-filter, and restoration rejection.  The trial adds:

- `selected_acceptance_source` for accepted inner steps;
- `line_search_failure_summary` for rejected inner steps;
- preserved stopped reasons such as `ipopt_h_filter_rejected`,
  `ipopt_persistent_filter_rejected`, and `no_acceptable_ipopt_filter_trial`;
- an IpOpt-style f-type entrance: a P-Armijo candidate can be accepted through
  persistent filter memory when protected feasibility components pass.

No FastChem4 public/runtime/trace values are used as ExoGibbs constructor
inputs.

Remaining-blocker probe delta:

| row | before | after | note |
|---|---|---|---|
| `solar_water_condensation` layer 5 | caveat | `converged`, tier 1 | primary route restored, support 151 |
| `solar_water_condensation` layer 7 | caveat | `converged`, tier 1 | primary route restored, support 119 |
| `solar_water_condensation` layer 8 | caveat | caveat | final barrier not centered; not an Armijo naming issue |

Curated layer score:

| metric | value |
|---|---:|
| artifact | `volatiles_artifacts/head_route_layer_evaluation.json` |
| rows | 99 |
| converged | 97 |
| exceptions | 0 |
| status counts | `86 converged / 11 converged_with_caveat / 2 not_converged` |
| route counts | `70 primary / 18 gas-only / 11 native fallback` |
| tier-1 rows | 70 |

Full-profile FastChem4 comparison:

| metric | value |
|---|---:|
| artifact | `volatiles_artifacts/head_route_v16_fastchem4_deep_comparison.json` |
| finite rows | 99 |
| ExoGibbs lower `G/RT` rows | 46 |
| FastChem4-scaled lower `G/RT` rows | 53 |
| max `abs(dG/RT Exo-FC)` | `1.724112e-1` |
| max temperature-valid inactive driving | `9.645081e2` |
| temperature-valid rows > 500 | 1 |
| temperature-valid rows > 1000 | 0 |

Target FastChem4 comparison:

| metric | value |
|---|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 |
| status | converged |
| support count | 67 |
| temperature-valid inactive closure | `0 / 0.0` |
| `dG/RT Exo-FC` | `-1.543382e-5` |
| relative budget L2 | `9.722846e-4` |
| solve wall time | 29.04 s |

Validation:

| check | result |
|---|---:|
| py_compile targeted files | passed |
| `pytest -q tests/unittests/optimize/condensate_algorithm_v11_callsite_test.py` | 11 passed, 2 warnings |
| `pytest -q tests/unittests/api/condensate_equilibrium_test.py` | 43 passed, 18 warnings |
| `pytest -q tests/endtoend/curated_cases` | 8 passed, 30 warnings |
| remaining-blocker probe | completed, 7 / 8 tier-1, 1 caveat |
| curated layer score | completed |
| full-profile FastChem4 comparison | completed |
| target FastChem4 comparison | completed |

Verdict: the f-type persistent-filter entrance fixes a real globalization gap
without weakening the protected feasibility guards.  Water5/water7 return to
tier-1 primary convergence.  Water8 remains a true PD-IPM centering problem:
the current amount-frame components are small, but complementarity/raw-frame
centering and inactive driving remain, and filter/restoration correctly reject
trials that would damage the physical feasibility components.

## Next trial candidates

### Candidate A: highT gas-only boundary closure

Status: investigated. The highT gas-only rows are diagnostic artifacts under
the legacy all-condensate metric; temperature-valid inactive-driving is zero.

Goal: keep the validity-aware metric in future score runs so highT
temperature-invalid condensates do not dominate the support-closure queue.

Required score checks:

- public convergence remains 99/99;
- gas-only route count does not turn into unstable condensate routes;
- highT max inactive driving and gas abundance changes are reported separately;
- ExoGibbs-native `G/RT` does not worsen.

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

## Trial v1.18-lifecycle-support-closure: final-state support closure retry

This trial keeps HEAD route version `v1.18` and adds a support-closure retry
from the PD-IPM lifecycle final state when the activity-driven support outer
loop stops at `max_support_outer_iterations_reached`.  The retry grows the
support from the ExoGibbs-native final gas/condensate state, not from
FastChem4, and is accepted only through the existing common retry conditions:
route promoted, result converged, and support-closure gate accepted.

Implementation:

- added `lifecycle_final_state_support_closure_retry` as a selectable retry
  candidate in `_run_activity_driven_support_outer_loop`;
- added remaining inactive-positive support from the lifecycle final state and
  re-ran the normal explicit-support public API route;
- kept selection under `best_support_closure_score_across_retry_kinds`;
- added a unit regression for the max-outer-iteration closure retry;
- updated the water8 end-to-end expectation from caveat repair to tier-1
  convergence.

Case-by-case delta:

| row | before | after | note |
|---|---|---|---|
| `solar_water_condensation` layer 8 | `converged_with_caveat`, support 87, valid inactive `17 / 430.584` | `converged`, tier 1, support 152, valid inactive `0 / 0.0` | lifecycle final-state closure retry adds the 6 remaining positive inactive species and promotes the route |
| `solar_water_condensation` layer 5 | `converged`, tier 1, support 151 | unchanged | blocker probe remains tier 1 |
| `solar_water_condensation` layer 7 | `converged`, tier 1, support 119 | `converged`, tier 1, support 110 in current fresh run | still tier 1 with valid inactive closure |

Curated layer score:

| metric | value |
|---|---:|
| artifact | `volatiles_artifacts/head_route_layer_evaluation.json` |
| rows | 99 |
| converged | 96 |
| exceptions | 0 |
| status counts | `86 converged / 10 converged_with_caveat / 3 not_converged` |
| route counts | `71 primary / 18 gas-only / 10 native fallback` |
| tier-1 rows | 70 |
| water family | `9 / 9 converged` |

The 3 not-converged rows in this score script are two highT gas-only
full-budget rows and one explicit-support `complex_heavy...` demo row.  The
same complex layer is converged when run support-free through the fresh public
API with the curated profile options; this trial targeted the support-free
outer-loop closure path.

Full-profile FastChem4 comparison:

| metric | value |
|---|---:|
| artifact | `volatiles_artifacts/head_route_v16_fastchem4_deep_comparison.json` |
| finite rows | 99 |
| ExoGibbs lower `G/RT` rows | 39 |
| FastChem4-scaled lower `G/RT` rows | 60 |
| max `abs(dG/RT Exo-FC)` | `1.730409e-1` |
| max temperature-valid inactive driving | `9.645081e2` |
| temperature-valid rows > 500 | 1 |
| temperature-valid rows > 1000 | 0 |
| water family Exo lower `G/RT` rows | 8 / 9 |
| water family valid inactive driving | max 0, count 0 on all layers |

Target FastChem4 comparison:

| metric | value |
|---|---:|
| row | `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 |
| status | converged |
| support count | 67 |
| temperature-valid inactive closure | `0 / 0.0` |
| `dG/RT Exo-FC` | `+2.233035e-6` |
| relative budget L2 | `6.354365e-4` |
| solve wall time | 34.47 s |

Validation:

| check | result |
|---|---:|
| py_compile targeted files | passed |
| `pytest -q tests/unittests/api/condensate_equilibrium_test.py` | 44 passed, 19 warnings |
| `pytest -q tests/unittests/optimize/condensate_algorithm_v11_callsite_test.py tests/unittests/optimize/condensate_algorithm_v11_direction_test.py` | 16 passed, 2 warnings |
| `pytest -q tests/endtoend/curated_cases` | 8 passed, 32 warnings |
| remaining-blocker probe | completed, water5/water7/water8 all tier 1 |
| curated layer score | completed |
| full-profile FastChem4 comparison | completed |
| target FastChem4 comparison | completed |

Verdict: water8 is no longer a PD-IPM centering caveat in the support-free
HEAD route.  The remaining issue is not the water support closure but the
broader score surface: highT gas-only budget rows and an explicit-support
complex demo row still need separate treatment.

## Trial v1.18-explicit-support-closure: explicit payload closure retry

This trial keeps HEAD route version `v1.18` and repairs the explicit-support
curated `complex_heavy...` layer-4 row.  The failure mode was not a broken
PD-IPM Newton solve: the public explicit support payload contained only the
four Ti-bearing seed condensates, while the ExoGibbs-native final state needed
additional inactive-positive support to satisfy the full condensate element
budget gate.  The retry grows support from the ExoGibbs final gas state and is
accepted only when the promoted route is converged, the support-closure gate is
accepted, and the full-budget gate is accepted.

Implementation:

- added an explicit-support lifecycle closure retry at the public API exit;
- trigger is limited to `selection_mode == explicit_support_payload` rows that
  are not converged, fail support closure, or fail the full-budget gate;
- retry support is selected from ExoGibbs-native inactive-positive driving;
- recursive retry disables support outer-loop and retry recursion guards;
- no FastChem4 public/runtime/trace values are used as constructor inputs;
- added an end-to-end regression for the failing complex layer-4 row.

Case-by-case delta:

| row | before | after | note |
|---|---|---|---|
| `complex_heavy_element_or_boron_titanium_zirconium_case` layer 4 | `not_converged`, support 4, full-budget max relative residual `0.953226` on Ca, valid inactive `62 / 243.508` | `converged`, tier 1, support 10, full-budget max relative residual `4.880842e-4` on F, valid inactive `49 / 244.312` | explicit support closure retry selected; support closure is accepted under the current max-driving gate |

Curated layer score:

| metric | value |
|---|---:|
| artifact | `volatiles_artifacts/head_route_layer_evaluation.json` |
| rows | 99 |
| converged | 97 |
| exceptions | 0 |
| status counts | `87 converged / 10 converged_with_caveat / 2 not_converged` |
| route counts | `71 primary / 18 gas-only / 10 native fallback` |
| tier-1 rows | 71 |
| complex family | `9 / 9 converged` |
| remaining not-converged rows | 2 highT gas-only full-budget rows |

Full-profile FastChem4 comparison:

| metric | value |
|---|---:|
| artifact | `volatiles_artifacts/head_route_v16_fastchem4_deep_comparison.json` |
| finite rows | 99 |
| ExoGibbs lower `G/RT` rows | 40 |
| FastChem4-scaled lower `G/RT` rows | 59 |
| max `abs(dG/RT Exo-FC)` | `1.730409e-1` |
| max temperature-valid inactive driving | `9.645081e2` |
| temperature-valid rows > 500 | 1 |
| temperature-valid rows > 1000 | 0 |
| complex layer 4 `dG/RT Exo-FC` | `+2.617036e-4` |
| complex layer 4 relative budget L2 | `8.322506e-7` |

Validation:

| check | result |
|---|---:|
| `python -m py_compile src/exogibbs/api/condensate_equilibrium.py tests/endtoend/curated_cases/test_fresh_api_curated_cases.py` | passed |
| `pytest -q tests/endtoend/curated_cases/test_fresh_api_curated_cases.py -k "complex_heavy_midlayer_explicit_support_closure"` | 1 passed, 8 deselected |
| `pytest -q tests/endtoend/curated_cases` | 9 passed, 32 warnings |
| `python volatiles_code/evaluate_curated_head_route_layers.py` | completed |
| `python volatiles_code/deep_compare_head_route_v13_fastchem4.py` | completed |
| `git diff --check` | passed |
| `python -m json.tool score.json` | passed |

Verdict: the explicit-support complex row is repaired without turning the
volatile demo artifacts or FastChem4 traces into constructor inputs.  The
remaining public score blockers are now the two highT gas-only full-budget rows.

## Trial v1.18-explicit-empty-support-strict-gas: highT gas-only gate repair

This trial keeps HEAD route version `v1.18` and repairs the last two highT
gas-only rows in the curated score.  The issue was not condensate support
closure or a PD-IPM Newton failure: the support-free API path already had an
empty-support strict gas retry, but the curated demo passes
`support_indices=()` explicitly.  That explicit-empty payload path built a
gas-only result directly and missed the same strict gas retry, so layers 14
and 16 failed the full condensate budget gate even though the stricter gas
solve could satisfy it.

Implementation:

- wired the existing `empty_support_strict_gas_retry` into the explicit
  empty-support public API branch;
- kept the retry limited to gas-only results that fail the full-budget gate;
- reused the existing native gas `equilibrium(..., epsilon_crit=1.0e-12)`
  retry and empty-support result builder;
- did not add condensates, support growth, volatile artifacts, or FastChem4
  constructor inputs;
- added an end-to-end regression for highT layer 14 with
  `support_indices=()` and `support_amounts_init=()`.

Case-by-case delta:

| row | before | after | note |
|---|---|---|---|
| `solar_highT_no_condensate_gas_regression` layer 14 | `not_converged`, explicit empty support, initial full-budget max relative residual `2.409032e-3` on F | `converged`, gas-only, retry full-budget max relative residual `2.125767e-8` on Ge, valid inactive `0 / 0.0` | explicit-empty path now uses strict gas retry |
| `solar_highT_no_condensate_gas_regression` layer 16 | `not_converged`, explicit empty support, initial full-budget max relative residual `1.368671e-3` on F | `converged`, gas-only, retry full-budget max relative residual `1.917696e-8` on Ge, valid inactive `0 / 0.0` | explicit-empty path now uses strict gas retry |

Curated layer score:

| metric | value |
|---|---:|
| artifact | `volatiles_artifacts/head_route_layer_evaluation.json` |
| rows | 99 |
| converged | 99 |
| exceptions | 0 |
| status counts | `89 converged / 10 converged_with_caveat` |
| route counts | `71 primary / 18 gas-only / 10 native fallback` |
| tier counts | `71 tier-1 / 18 runtime empty support / 10 tier-2 fallback` |
| highT family | `18 / 18 converged` |
| remaining not-converged rows | 0 |

Full-profile FastChem4 comparison:

| metric | value |
|---|---:|
| artifact | `volatiles_artifacts/head_route_v16_fastchem4_deep_comparison.json` |
| finite rows | 99 |
| ExoGibbs lower `G/RT` rows | 40 |
| FastChem4-scaled lower `G/RT` rows | 59 |
| max `abs(dG/RT Exo-FC)` | `1.730409e-1` |
| max temperature-valid inactive driving | `9.645081e2` |
| temperature-valid rows > 500 | 1 |
| temperature-valid rows > 1000 | 0 |
| highT ExoGibbs lower `G/RT` rows | 12 / 18 |
| highT max relative budget L2 | `2.494606e-4` |
| highT layer 14 `dG/RT Exo-FC` | `+2.863967e-10` |
| highT layer 16 `dG/RT Exo-FC` | `-9.684742e-11` |

Validation:

| check | result |
|---|---:|
| `python -m py_compile src/exogibbs/api/condensate_equilibrium.py tests/endtoend/curated_cases/test_fresh_api_curated_cases.py` | passed |
| `pytest -q tests/endtoend/curated_cases/test_fresh_api_curated_cases.py -k "highT_explicit_empty_support"` | 1 passed, 9 deselected |
| `pytest -q tests/endtoend/curated_cases` | 10 passed, 32 warnings |
| `pytest -q tests/unittests/api/condensate_equilibrium_test.py` | 44 passed, 19 warnings |
| `python volatiles_code/evaluate_curated_head_route_layers.py` | completed |
| `python volatiles_code/deep_compare_head_route_v13_fastchem4.py` | completed |
| `git diff --check` | passed |

Verdict: the curated fresh API score is now fully converged at 99/99.  The
highT rows remain true gas-only rows with no temperature-valid inactive
condensate support; the repair is a public API routing consistency fix for
explicit empty support payloads.
