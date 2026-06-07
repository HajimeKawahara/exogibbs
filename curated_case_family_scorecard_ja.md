# FastChem4 curated case family 星取表

この表は、FastChem4 lane で定義した 10 curated case families（選定済み10ケース群）について、現時点でどこまで確認できているかを整理したものです。

最終更新: FC4-M4377 refresh policy route-selector integration。M4351 では実行済み solver rows 10本が 10本すべて accepted になった。M4352-M4354 では未実行だった 4 family に corrected representative row（修正済み代表 row）を 1本ずつ追加し、4 rows すべて solver 入口まで到達したが、当初は `no_p_armijo_trial` で blocked だった。M4355-M4356 では、P-Armijo 候補は存在するが gas residual（気相残差）を悪化させるため strict guard（厳密な安全柵）が止めており、guard を外すと全 4 rows でさらに悪化することを確認した。M4370-M4373 で原因を methane-rich gas frame（メタン支配の気相境界）と hardened gas refresh floor sensitivity（強化気相 refresh の floor 感度）に分解し、M4374 で adaptive floor candidates（floor 候補列）を使うことで corrected frontier 4 rows は 4/4 accepted になった。M4375 では、T500 の default depleted refresh budget tradeoff（凝縮後 budget とのトレードオフを許す refresh）と frontier adaptive floor repair が、同じ reusable gas-boundary refresh policy（再利用可能な気相境界 refresh policy）で表現できることを確認した。M4376/M4377 では、この policy を実 callsite 近傍の candidate generation（候補生成）と route selector（経路選択器）に接続した。

| # | case family | 目的 | 現状 |
|---:|---|---|---|
| 1 | `solar_highT_no_condensate_gas_regression` | 高温・凝縮なしの気相確認 | M030 で public-output cartography（FastChem4公開出力の地図作り）済み。M4331 では `gas_regression_public_output_only_no_condensate_route` と分類。凝縮 route（凝縮 solver 経路）の対象ではない。 |
| 2 | `solar_silicate_first_condensation` | silicate（ケイ酸塩）初期凝縮 | M030 実行済み。M4351 HEAD route v2 では T1400/T1500 ともに `m4309_promoted_high_start_callsite_policy` で accepted。T1500 は M4306 では `current_barrier_not_centered` だったが、M4309 route では centered。 |
| 3 | `solar_water_condensation` | water condensation（水凝縮） | M030 実行済み。M4350 HEAD route では T300 rows 2本が `m4310_full_promoted_policy_route` で accepted。 |
| 4 | `solar_metal_sulfide_or_Fe_Ni_S_region` | Fe/Ni/S, sulfide（金属・硫化物） | M030 実行済み。M4350 HEAD route では T700 rows 2本が `fastchem4_style_electron_refresh_route` で accepted。これは M4344 で source convention（source の符号・意味の規約）を RGIE 側に合わせた修正が効いている。 |
| 5 | `carbon_rich_graphite_window` | graphite（炭素凝縮） | M4354 で corrected row `carbon_rich_graphite_window__T1300_P1_corrected` を追加。`graphite` alias を `C(s)` に直し、T1300 で hvector が favorable になるよう修正した。M4374 adaptive floor frontier repair では `1e-300` floor のまま accepted。 |
| 6 | `carbon_rich_CaS_MgS_AlN_window` | C/O=2 系の CaS/MgS/AlN | M4354 で corrected row `carbon_rich_CaS_MgS_AlN_window__T700_P1_corrected` を追加。`CaS(s)`, `MgS(s)`, `AlN(s)` を target-guided support（狙った凝縮候補を明示的に入れる方針）として使用。M4374 adaptive floor frontier repair では default `1e-300` floor では gas refresh が失敗したが、`1e-200` floor で accepted。 |
| 7 | `SiO_s_condensate_window` | SiO(s) 凝縮 | M4354 で corrected row `SiO_s_condensate_window__T900_P0p1_corrected` を追加。custom placeholder と言わず、solar target-guided first pass として `SiO(s)` を直接 support に入れた。M4374 adaptive floor frontier repair では `1e-300` floor のまま accepted。 |
| 8 | `lowT_strong_condensation_budget_stress` | 低温・強凝縮・budget stress（元素収支が厳しい） | M030 実行済み。M4350 HEAD route では T500 rows 2本が `adaptive_refresh_selector_default_depleted_refresh_budget_tradeoff` で accepted。ただしこれは strict budget closure ではない。source-convention-safe electron refresh は gauge（評価基準）を整合させるが、T500 では coupled gas gap（気相・凝縮相を同時更新しないと残るずれ）を露出するため、現時点では default refresh の budget tradeoff を採用する。 |
| 9 | `near_phase_boundary_support_sensitivity` | 相境界付近の support sensitivity（凝縮種候補の敏感性） | M030 実行済み。M4351 HEAD route v2 では T1490/T1510 が `m4309_promoted_high_start_callsite_policy` で accepted。 |
| 10 | `complex_heavy_element_or_boron_titanium_zirconium_case` | B/Ti/Zr など複雑 minor species | M4354 で corrected row `complex_heavy_element_or_boron_titanium_zirconium_case__T1100_P1_corrected` を追加。contract に `ZrO2(s)` と `BN(s)` がないため、first pass は利用可能な titanium proxy targets（`TiO2(s,l)`, `TiC(s,l)`, `TiN(s,l)`）で実行。M4374 adaptive floor frontier repair では `1e-300` floor のまま accepted。 |

## 到達度の整理

### 1. corrected row で solver 到達済みだった frontier は M4374 で accepted

以下の4 family は curated cases（選定ケース群）に入っており、M4354 で corrected representative row を 1本ずつ追加して現在の HEAD route を試した。当初は accepted には至らなかったが、M4374 の adaptive floor frontier repair で 4 rows すべて accepted になった。

- `carbon_rich_graphite_window`: `C(s)` support で accepted。選択 floor は `1e-300`。
- `carbon_rich_CaS_MgS_AlN_window`: `CaS(s)`, `MgS(s)`, `AlN(s)` support で accepted。選択 floor は `1e-200`。
- `SiO_s_condensate_window`: `SiO(s)` support で accepted。選択 floor は `1e-300`。
- `complex_heavy_element_or_boron_titanium_zirconium_case`: titanium proxy support で accepted。選択 floor は `1e-300`。

M4354 の corrected 4 rows の集計は以下である。

```text
corrected_case_count: 4
all_targets_favorable_count: 4
restricted_callsite_called_count: 4
restricted_callsite_success_count: 0
head_route_attempted_count: 4
head_route_accepted_count: 0
```

M4355/M4356 の分解では、4 rows すべてで P-Armijo は P merit（バリア付き目的関数）を下げる候補を見つけていた。ただし、その候補は raw/amount-weighted gas residual（気相残差）を悪化させるため、residual nonworsening guard（残差悪化防止の安全柵）に拒否されていた。guard を外すと、center ratio（中心条件からのずれ）と amount-weighted gas residual が全ケースで悪化した。

```text
p_armijo_candidate_rejected_by_residual_nonworsening_guard: 4
strict_guard_best_variant: 4
no_guard_accepted_rows: 0
```

M4370-M4373 では、この blocker は support の失敗ではなく methane-rich gas frame（メタン支配の気相境界）と hardened gas refresh floor sensitivity（強化気相 refresh の floor 感度）に分解された。M4374 では同じ floor 候補列を4 rows に適用し、次の結果になった。

```text
accepted_count: 4
remaining_blocked_count: 0
adaptive_floor_needed_count: 1
selected_floor_values:
  carbon_rich_graphite_window__T1300_P1_corrected: 1e-300
  carbon_rich_CaS_MgS_AlN_window__T700_P1_corrected: 1e-200
  SiO_s_condensate_window__T900_P0p1_corrected: 1e-300
  complex_heavy_element_or_boron_titanium_zirconium_case__T1100_P1_corrected: 1e-300
```

### 2. public-output までは見たが、solver route は未完成

現時点で、この分類に残る主要 case はない。

T1500 は FastChem4 public-output（公開出力）側では対象に入っており、M4306 callsite replay（solver 呼び出し境界の再実行）でも実行できた。ただし M4306 では `current_barrier_not_centered` で止まっていた。M4351 では、後続の M4309 promoted high-start route を HEAD route v2 として採用し、T1500 は accepted に更新された。

T700 は M4344/M4350 でこの分類から外れた。未解決だった blocker は support boundary そのものではなく、electron refresh に渡す source convention と RGIE 側の gas stationarity source の不一致だった。

### 3. solver-callsite で前進している

以下の family は solver-callsite（solver を呼ぶ境界）で前進している。

- `solar_water_condensation`
- `lowT_strong_condensation_budget_stress`
- `near_phase_boundary_support_sensitivity`
- `solar_silicate_first_condensation`
- `solar_metal_sulfide_or_Fe_Ni_S_region`

`solar_silicate_first_condensation` は、M4350 時点では一部未解決だったが、M4351 で T1500 も accepted に更新された。

`solar_metal_sulfide_or_Fe_Ni_S_region` の T700 は、M4344 時点で次の状態まで進んだ。

```text
classification: T700_GAS_STATIONARITY_GAP_LAMBDA_GAUGE_COMPATIBLE
max_lambda_only_residual_l2: 2.082445943218344e-11
sentinel_count: 0
qtot_gap: 0.0
```

さらに、M4342 route の再実行では T700 の両 rows が final barrier へ到達した。このため T700 は、現時点では HEAD route の単一 policy の中で source-convention-safe electron refresh が主要に効く accepted case と扱う。

M4345 では、T700 で効いた source-convention-safe electron refresh を T500 に広げられるか確認した。結果は次である。

```text
decision: LOWT_ELECTRON_REFRESH_GAUGE_GENERALIZES_BUT_ROUTE_PARTIAL
lambda_gauge_compatible_count: 2 / 2
final_barrier_count: 0 / 2
```

つまり、source convention の修正は T500 でも正しく働くが、それだけでは solver policy 全体は完成しない。HEAD route は一つなので、T500 と T700 に別 route を割り当てるのではなく、同じ統合 policy の中で、どの repair / guard / continuation が主要に効くかを観測している。

M4346 でさらに分解した結果、T500/T700 の本質的な違いは lambda gauge ではなかった。両方とも source-convention-safe electron refresh 後の lambda gauge は整合している。違いは次である。

```text
T500 max neutral log budget residual after electron refresh: 5.663824094349078
T700 max neutral log budget residual after electron refresh: 1.403943628019988e-11

T500 max stage1 budget residual: 0.7335650159979301
T700 max stage1 budget residual: 0.0014846412451118764

T500 max final epsilon gap: 13.815510557964274
T700 max final epsilon gap: 0.000993164980670258
```

つまり、T500 は gauge は合うが、electron refresh 後にも budget 側の歪みが大きく残り、barrier を進める段階で止まる。T700 は budget が十分小さいため、同じ統合 policy の中で final barrier まで進める。

M4375 では、T500 で使っている budget-tradeoff refresh と、frontier 4 rows で使う adaptive floor repair を別々の例外扱いにせず、同じ refresh candidate selector（refresh 候補選択器）に通した。結果は次である。

```text
row_count: 6
accepted_count: 6
selected_policy_counts:
  default_depleted_refresh_budget_tradeoff: 2
  adaptive_floor_frontier_repair: 4
```

つまり、T500 と corrected frontier 4 rows は、どちらも「気相境界 refresh 候補を複数用意し、final-barrier / budget / amount-weighted gas の条件を満たす最初の候補を採用する」という同じ形に整理できる。

M4376/M4377 では、frontier 4 rows について、この整理を artifact 上だけではなく、実 callsite 近傍へ進めた。`src/exogibbs/diagnostics/condensate_gas_boundary_refresh_callsite.py` が floor 候補ごとの refresh init 候補を生成し、その候補を continuation に通したあと、`condensate_gas_boundary_refresh_policy.py` と `condensate_algorithm_v11_route_selector.py` で採用する。

```text
M4376 callsite-generated frontier accepted: 4 / 4
M4377 route-selector refresh-policy accepted: 4 / 4
CaS/MgS/AlN selected floor: 1e-200
other frontier selected floor: 1e-300
```

## FC4-M4351 HEAD route v2 星取表

| family | HEAD route status | executable rows | accepted rows | blocked rows | selected route |
|---|---:|---:|---:|---:|---|
| `solar_highT_no_condensate_gas_regression` | gas-only | 0 | 0 | 0 | 凝縮 route 対象外 |
| `solar_silicate_first_condensation` | accepted | 2 | 2 | 0 | `m4309_promoted_high_start_callsite_policy` |
| `solar_water_condensation` | accepted | 2 | 2 | 0 | `m4310_full_promoted_policy_route` |
| `solar_metal_sulfide_or_Fe_Ni_S_region` | accepted | 2 | 2 | 0 | `fastchem4_style_electron_refresh_route` |
| `carbon_rich_graphite_window` | accepted | 1 | 1 | 0 | M4374 adaptive floor frontier repair |
| `carbon_rich_CaS_MgS_AlN_window` | accepted with adaptive floor | 1 | 1 | 0 | M4374 adaptive floor frontier repair, selected floor `1e-200` |
| `SiO_s_condensate_window` | accepted | 1 | 1 | 0 | M4374 adaptive floor frontier repair |
| `lowT_strong_condensation_budget_stress` | accepted with caveat | 2 | 2 | 0 | `adaptive_refresh_selector_default_depleted_refresh_budget_tradeoff` |
| `near_phase_boundary_support_sensitivity` | accepted | 2 | 2 | 0 | `m4309_promoted_high_start_callsite_policy` |
| `complex_heavy_element_or_boron_titanium_zirconium_case` | accepted | 1 | 1 | 0 | M4374 adaptive floor frontier repair |

M4351 までの既存実行済み solver rows の合計は 10本で、そのうち 10本が accepted、blocked は 0本である。M4354 で corrected rows を含めると、solver に入った rows は 14本、そのうち accepted は 10本、blocked は 4本だった。M4374 では corrected frontier 4 rows も accepted になったため、solver に入った代表 rows 14本のうち 14本が accepted、blocked は 0本である。M4352 であった empty support 問題は、M4354 の corrected first pass では解消した。

## FC4-M4380 accepted / convergence evidence の明確化

M4380 では、HEAD route accepted（経路選択器が採用した）と convergence evidence（収束証拠）を分けて確認した。

```text
HEAD route accepted rows: 14 / 14
final-barrier evidenced rows: 14 / 14
uniform post-solver residual components available: 8 / 14
```

ここで final-barrier evidenced（最終 barrier 到達の証拠あり）とは、少なくとも1つの source artifact で `converged_at_final_barrier == true`、`final_barrier_centered == true`、または同等の final-barrier centered status が確認できる、という意味である。

注意点として、14/14 は「同じ residual table で全ケースの post-solver residual（solver 後の残差）が完全に揃った」という意味ではない。M4380 時点では、T500、T700、frontier 4 rows など 8/14 rows は component-level residual（成分別残差）まで同じ整理表に入っている。一方、near boundary、silicate、water の一部 rows は final barrier 到達 flag はあるが、同じ形式の component table はまだ未統一である。

したがって現在の正確な表現は、「HEAD route は 14 representative rows すべてで final barrier 到達の証拠を持つ。ただし、次は14 rows全体の uniform post-solver residual table を作る必要がある」である。

## FC4-M4381 uniform post-solver residual table

M4381 では、M4380 で未統一だった post-solver residual（solver 後の残差）を 14 rows 全体で同じ形式に整理した。

```text
uniform post-solver residual rows: 14 / 14
final-barrier evidenced rows: 14 / 14
tight residual components: 10
accepted budget-tradeoff components: 2
barrier-centered with raw-gas caveat: 2
```

分類の意味は以下である。

- tight residual components（十分小さい残差成分）: budget（元素収支）と amount-weighted gas（存在量で重み付けした気相残差）が小さい。
- accepted budget-tradeoff components（budget tradeoff 付き受理）: T500 の2 rows。final barrier には到達しているが、strict budget closure（厳密な元素収支閉鎖）ではなく、現時点では budget と gas の tradeoff を許している。
- barrier-centered with raw-gas caveat（barrier 中心化はできたが raw gas に注意）: T700 metal sulfide の2 rows。complementarity（内点法の中心化）は非常に小さく、budget も小さいが、raw gas residual（そのままの気相残差）は大きい。評価は amount-weighted gas を主に見る必要がある。

これで「14 representative rows は accepted か？」だけでなく、「どの残差分類で accepted なのか」まで見えるようになった。次の自然な作業は、この3分類を HEAD route acceptance tiers（HEAD route の受理階層）として明文化し、どの階層を production-adjacent opt-in に進めてよいかを決めることである。

## FC4-M4382 HEAD route acceptance tiers

M4382 では、M4381 の3分類を acceptance tiers（受理階層）として固定した。

```text
tier_1_tight_residual_production_adjacent_candidate: 10
tier_2_budget_tradeoff_experimental_only: 2
tier_3_raw_gas_caveat_diagnostic_only: 2
```

各 tier の扱いは以下である。

- tier 1: production-adjacent opt-in hardening（明示 opt-in の production 近傍 hardening）へ進めてよい候補。ただし default-on production wiring（標準経路への接続）はまだ不可。
- tier 2: T500 の budget tradeoff rows。実験としては accepted だが、strict budget closure（厳密な元素収支閉鎖）または coupled-gas closure（気相・凝縮相を同時に閉じる方針）が示されるまで production acceptance gate には進めない。
- tier 3: T700 metal sulfide の raw-gas caveat rows。barrier と budget は良いが、raw gas residual の評価基準が未解決なので diagnostic-only（診断限定）に留める。

したがって次に進めるべき中心は、tier 1 の 10 rows に限定した production-adjacent opt-in hardening である。T500/T700 は成功扱いから外すのではなく、caveat 付き accepted として保持し、別途 strict closure / residual frame の解決対象にする。

## FC4-M4383 tier 1 opt-in hardening scope

M4383 では、tier 1 の10 rowsだけを production-adjacent opt-in hardening（明示 opt-in の production 近傍 hardening）対象として切り出す helper を追加した。

```text
tier 1 hardening candidates: 10
caveat rows kept outside hardening scope: 4
gas-only default path protected: true
condensate legacy default preserved as acceptance target: false
```

ここで重要なのは、normal default path unchanged（標準経路を変えない）の意味を分けたことである。凝縮なしの gas-only default path（気相のみ標準経路）と既存 API 互換性は守る。一方で、凝縮ありの legacy default behavior（従来の凝縮あり挙動）はまともに動いていないため、保存すべき acceptance target（受理目標）とは扱わない。したがって、凝縮ありでは explicit opt-in の新 route を前に出してよい。ただし、default-on production wiring（標準で有効化）、production return signature change（返り値形式変更）、preset/default wiring（preset や default への接続）はまだ禁止する。

## FC4-M4384 tier 1 explicit opt-in callsite replay

M4384 では、M4383 で切り出した tier 1 の10 rowsを explicit opt-in callsite surface（明示 opt-in の入口）に実際に通した。

```text
source rows: 14
admitted tier 1 rows: 10
rejected caveat rows: 4
normal import guard: passed
```

admitted（入口通過）したのは、tight residual components（十分小さい残差成分）を持つ tier 1 の10 rowsだけである。T500 lowT の2 rowsは budget-tradeoff experimental only（元素収支 tradeoff 付きの実験限定）なので入口から外した。T700 metal sulfide の2 rowsは raw-gas caveat diagnostic only（そのままの気相残差に注意が残る診断限定）なので入口から外した。

この replay は solver call（solver 実行）ではない。目的は、production-adjacent opt-in に進める候補だけを、default-off（標準では無効）かつ explicit opt-in（明示指定が必要）な callsite boundary で選別できることを確認することである。

## FC4-M4385 tier 1 callsite payload readiness

M4385 では、M4384 で入口を通った tier 1 の10 rowsについて、restricted solver dry-run（制限付き solver の試走）へ渡す payload（入力束）を再構成できるか確認した。

```text
payload-ready rows: 10 / 10
unique cases: 9
recorded seed payload rows: 6
frontier selected-floor payload rows: 4
solver called: false
```

ここで unique cases が9なのは、water T300 が2 rows として記録されているためである。10 rows すべてで `support_indices`（使う凝縮種の番号）と `support_amounts_init`（初期凝縮量）が同じ長さで、有限かつ正値であることを確認した。

重要なのは、これはまだ solver の収束確認ではないことである。M4385 は、tier 1 だけを次の restricted solver dry-run へ安全に渡せる形にできる、という入力境界の確認である。

## FC4-M4386 tier 1 restricted solver dry-run evidence gate

M4386 では、M4385 で再構成した payload と、M4381 の uniform post-solver residual table（統一形式の solver 後残差表）を照合した。

```text
gate-passing rows: 10 / 10
max budget residual: 5.472953834043542e-07
max amount-weighted gas residual: 3.3417976036603513e-06
solver rerun performed: false
```

つまり、tier 1 の10 rowsは、入力 payload が作れるだけでなく、既存の post-solver evidence（solver 後の証拠）でも tight residual components（十分小さい残差成分）を持つことが確認された。

ただし、M4386 は solver を新しく走らせたわけではない。既存の solver evidence と今回の explicit opt-in payload gate が同じ rows を指しているかを確認した段階である。次は、この gate を実際の explicit restricted solver callsite smoke（明示 opt-in 制限付き solver の小規模実行）へ進める。

## FC4-M4387 tier 1 actual solver rerun confirmation

M4387 では、M4386 の確認を実際の solver rerun（solver の再実行）で裏取りした。対象は production-adjacent opt-in hardening 候補である tier 1 の10 rowsだけである。

```text
fresh solver rerun rows: 10 / 10
fresh solver rerun successes: 10 / 10
source artifacts: M4309 4 rows, M4310 2 rows, M4374 4 rows
max budget residual: 5.472953834043542e-07
max amount-weighted gas residual: 3.3417976036603513e-06
decision: TIER1_ACTUAL_SOLVER_RERUN_CONFIRMS_10_ROWS
```

つまり、M4384 では「入口で tier 1 だけを選べる」、M4385 では「solver に渡す payload を作れる」、M4386 では「既存 post-solver evidence と整合する」、M4387 では「実際に source replay を走らせ直しても tier 1 の10 rows が通る」ことを確認した。

注意点として、M4387 は T500 の budget-tradeoff rows と T700 metal sulfide の raw-gas caveat rows を production-adjacent opt-in tier 1 に昇格したわけではない。この4 rowsは HEAD route 上では accepted evidence を持つが、release-ready experimental surface（公開してよい実験 API 面）へ進めるには別の hardening が必要である。

## 現時点の短いまとめ

- 成功寄り: T300 water, T500 lowT, metal sulfide T700, near phase boundary T1490/T1510, silicate T1400/T1500, M4374 frontier 4 rows
- 未解決: corrected frontier 4 rows の blocked は M4374 で解消。M4375 で adaptive floor policy と T500 budget-tradeoff refresh は reusable policy として統合できることを確認した。M4376/M4377 で frontier rows については実 callsite 近傍の candidate generation と route selector 接続まで完了した。M4379 で T500 live payload でも accepted path が再確認され、M4380 で 14/14 rows の final-barrier evidence が揃った。M4381 で 14 rows 全体の post-solver residual table も揃った。M4382 で tier 1 の 10 rows を production-adjacent opt-in hardening 候補、T500/T700 の 4 rows を caveat 付き experimental/diagnostic として分類した。M4383 では tier 1 だけを opt-in hardening 対象として切り出し、凝縮あり legacy default は保存対象にしない方針を固定した。M4384 では tier 1 の10 rowsだけが explicit opt-in callsite surface を通り、T500/T700 の4 caveat rows は production-adjacent opt-in 入口から外れることを確認した。M4385 では、この tier 1 の10 rowsすべてで restricted solver dry-run 用 payload を再構成できることを確認した。M4386 では、これら10 rowsすべてが既存 post-solver evidence と整合し、restricted solver dry-run evidence gate を通ることを確認した。M4387 では、source replay を実際に走らせ直し、tier 1 の10 rowsすべてが fresh rerun でも final barrier centered になることを確認した。
- 凝縮 route 対象外: highT no-condensate gas regression
- support input: M4354 corrected first pass で empty support は解消。ただし target-guided support を正式 policy に昇格するかは未決。

次に広げる自然な順序は、未実行4 family の route input を作ることである。T500 は現時点では accepted だが、budget tradeoff 付きなので、後で strict closure 化する余地が残る。

## HEAD route v1.1 update#1: water intermediate layers

fresh API から作る `solar_water_condensation` profile では、T345/P0.003162、T330/P0.01、T315/P0.031623、T300/P0.1 の中間4層で v1 が `not_converged` になっていた。調査では restricted support solver は成功しており、H2O condensate amount も正値だった。一方で primary continuation は `no_p_armijo_trial` で止まり、route selector が `support_boundary_construction_required_before_selector` を返していた。

v1.1 では、fresh API runtime で lifecycle selector が accepted しない場合でも、finite warm-start candidate があるなら `native_budget_seed_fallback_budget_tradeoff` を許可する。この fallback は `metric_status` や saved primary/refresh summary が外から注入されていない場合だけ有効で、results artifact や FastChem4 trace/public/runtime values は constructor input に使わない。

v1.1 の layer-based demo 評価は次の通りである。

```text
profiles: 10
layers: 99
converged: 99
exceptions: 0
status_counts:
  converged: 18
  converged_with_caveat: 81
route_counts:
  head_v1_empty_positive_support_gas_only: 18
  native_budget_seed_fallback_budget_tradeoff: 81
```

これは水 profile を切らさないための caveat 付き改善であり、`no_p_armijo_trial` を primary continuation 側で解消したわけではない。次の改善候補は、水中間層の primary continuation を centered route へ近づけ、v1.1 fallback 依存を減らすことである。
