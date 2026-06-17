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
