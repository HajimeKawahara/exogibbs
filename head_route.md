# HEAD route 定義

この文書は、ExoGibbs の凝縮あり計算で現在の基準経路として扱う **HEAD route** を定義する。HEAD route の内容を変更した場合は、この文書を更新する。

現在の実装版は **HEAD route v1.1** である。v1.1 は v1 の public API surface を保ったまま、fresh API runtime で lifecycle が accepted にならない場合の native seed fallback gate を拡張する。

## 一言でいうと

現在の HEAD route は、凝縮あり equilibrium（平衡計算）を次の順序で進める経路である。

1. ExoGibbs-native thermochemistry（ExoGibbs 内の熱化学データ）と elemental budget（元素予算）だけから、positive support（ゼロでない凝縮種候補）を作る。
2. 小さい正の condensate amount seed（凝縮量初期値）を入れる。
3. 凝縮で使う元素を気相 budget から差し引き、depleted gas refresh（凝縮後の元素予算で気相を作り直す処理）を行う。
4. restricted support solver（選んだ凝縮種だけを動かす制限付き solver）に入れる。
5. 失敗または不十分な場合は、HEAD route lifecycle（外側制御）で primary continuation、fallback、electron refresh、frontier refresh を試す。
   v1.1 では、restricted support solver が成功しても lifecycle selector が accepted しない fresh API runtime layer について、finite warm-start candidate が残っていれば native seed fallback を許可する。
6. 最後に selected route（採用経路）と acceptance tier（成功品質ランク）を返す。

これは FastChem4 exact replay（FastChem4 の分岐を完全再現すること）ではない。FastChem4 public/runtime/trace values（公開出力・実行時出力・内部 trace 値）は、ExoGibbs の constructor input（初期値や構成入力）として使わない。

## 現在の実装入口

凝縮ありの公開入口は以下である。

- `src/exogibbs/api/condensate_equilibrium.py`
  - `CondensateChemicalSetup`
  - `CondensateEquilibriumOptions`
  - `CondensateEquilibriumResult`
  - `build_condensate_chemical_setup`
  - `condensate_equilibrium`

通常の gas-only API（凝縮なし API）は変更しない。HEAD route は凝縮あり専用入口から使う。

## 主要な src 実装

現在の HEAD route は、主に `src/exogibbs/condensates/` に焼き直されている。

| module | 役割 |
|---|---|
| `src/exogibbs/api/condensate_equilibrium.py` | 凝縮あり public API shell。setup/options/result を定義し、HEAD route を呼ぶ。 |
| `src/exogibbs/condensates/positive_support_initializer.py` | native thermochemistry から positive support と seed payload を作る。 |
| `src/exogibbs/condensates/head_route_warm_start.py` | baseline seed と depleted gas refresh seed の候補を作る。 |
| `src/exogibbs/condensates/depleted_gas_refresh.py` | 凝縮量を差し引いた depleted budget で gas log-density boundary を作り直す。 |
| `src/exogibbs/condensates/support_boundary.py` | solver/warm-start output を continuation に渡す native boundary として整理する。 |
| `src/exogibbs/condensates/continuation_input.py` | algorithm-v1.1 continuation に必要な q/r/lambda/source/budget frame を作る。 |
| `src/exogibbs/condensates/head_route_lifecycle.py` | HEAD route の primary/fallback/refresh/selector を統合する facade。 |
| `src/exogibbs/condensates/head_route_selector.py` | primary、fallback、refresh policy のどれを採用するか決める。 |
| `src/exogibbs/condensates/electron_refresh.py` | source-convention-safe electron refresh の gauge 整合を確認する。 |
| `src/exogibbs/condensates/frontier_refresh.py` | frontier cases 用の refresh candidate metrics から採用候補を選ぶ。 |
| `src/exogibbs/condensates/center_primary_fallback.py` | primary が止まった場合の center-primary fallback を選ぶ。 |
| `src/exogibbs/condensates/head_route_acceptance.py` | HEAD route accepted rows を tier 1/2/3 に分類する。 |
| `src/exogibbs/condensates/head_route_standard_gate.py` | public API 側の `converged` / `converged_with_caveat` / `not_converged` を決める。 |

下位 solver callsite は以下を使う。

- `src/exogibbs/optimize/minimize_cond.py`
- `src/exogibbs/optimize/condensate_algorithm_v11_callsite.py`

`src/exogibbs/diagnostics/` に残る古い探索 helper は、現在の public HEAD route の主経路ではない。新しい文書では `src/exogibbs/condensates/` 側を正本として扱う。

## API から見た実行順序

`condensate_equilibrium()` は概ね次の順序で動く。

### 1. setup validation

`CondensateChemicalSetup` が検証される。

- gas setup と condensate setup の element order が一致すること。
- gas formula matrix と condensate formula matrix の element row 数が一致すること。
- species order と formula matrix column 数が一致すること。

### 2. positive support selection

ユーザーが `support_indices` を渡さない場合、`build_positive_support_initializer_report()` が呼ばれる。

ここでは、FastChem4 output から support を取らない。使うのは以下だけである。

- `formula_matrix_cond`
- `element_inventory_target`
- `condensate_species_order`
- `hvector_cond`

初期 policy は少数 support、主に top candidate を使う。seed amount は `seed_fraction <= 1e-3`、`max_seed_amount <= 1e-3` の安全 envelope で作る。

### 3. empty support boundary

positive support が空の場合、凝縮 solver に入らず gas-only equilibrium に戻す。

これは「凝縮ありが失敗した」という意味ではなく、現在の native support policy では凝縮候補がない boundary として扱う。

### 4. warm-start candidate generation

non-empty support の場合、`build_condensate_head_route_warm_start_report()` が候補を作る。

候補は少なくとも次を含む。

- `baseline_positive_support_seed`
- `depleted_gas_refresh_native_gas_solver`

`depleted_gas_refresh` では、凝縮 seed が使う元素量 `Ac @ m` を全体 budget `b` から引き、残りの budget で気相 log-density を解く。

### 5. restricted support solver attempt

各 warm-start candidate を `solve_restricted_support_condensate_layer()` に渡す。

最初に成功した candidate を採用する。成功しない場合でも、finite な warm-start state があるときは lifecycle continuation に渡して、HEAD route 側で再評価する。

### 6. HEAD route lifecycle

`run_condensate_head_route_lifecycle()` が以下を行う。

- support boundary を作る。
- continuation input を作る。
- primary continuation を走らせる、または外から渡された primary summary を使う。
- 必要なら center-primary fallback を評価する。
- 必要なら source-convention-safe electron refresh を評価する。
- 必要なら frontier refresh policy を評価する。
- route selector に渡して accepted / not accepted を決める。

### 7. v1.1 native seed fallback gate

HEAD route v1 では、restricted support solver が失敗した場合だけ native seed fallback に進んでいた。水凝縮 profile の中間層では、restricted solver は成功する一方で、primary continuation が `no_p_armijo_trial` で止まり、route selector が `support_boundary_construction_required_before_selector` を返すことがあった。この場合、良い runtime boundary はあるが standard gate へ進めず `not_converged` になっていた。

HEAD route v1.1 では、次の条件をすべて満たす場合にも `native_budget_seed_fallback_budget_tradeoff` を返す。

- lifecycle result が accepted / converged ではない。
- `enable_native_seed_fallback=True`。
- `metric_status`、`head_route_primary_summary`、`head_route_refresh_policy_summary` が外から注入されていない。
- finite な warm-start candidate がある。

この fallback は FastChem4 trace/public/runtime value を constructor input に使わない。返り値は `converged_with_caveat` で、acceptance tier は `tier_2_budget_tradeoff_experimental_only` である。diagnostics には、restricted solver が成功していたかどうかを `restricted_solver_success` として残す。

### 8. standard gate and result

最後に `build_condensate_equilibrium_result_from_solver_payload()` が public result を作る。

返す主なものは以下である。

- `gas_ln_n`
- `gas_n`
- `gas_x`
- `gas_ntot`
- `condensate_amounts`
- `condensate_support_indices`
- `condensate_support_names`
- `acceptance_tier`
- `selected_route`
- `status`
- `converged`
- `diagnostics`

## HEAD route の構成要素

HEAD route は単一の Newton step ではない。複数の lifecycle component を統合した route である。

### primary route

primary route は algorithm-v1.1 continuation を正面から試す経路である。次の条件を満たすと primary accepted と扱う。

- `row_status == "centered"`、または
- `converged_at_final_barrier == true`

現在の route selector では、primary が centered なら `m4310_full_promoted_policy_route` として accepted する。

### promoted high-start route

T1400/T1500 や near-boundary cases で有効だった route。barrier schedule と初期 state を改善し、以前 `current_barrier_not_centered` で止まっていた cases を final barrier centered まで進めた。

実装上は primary continuation policy の一部として扱う。

### center-primary fallback

primary が centered しない場合に、center ratio と budget guard を見て fallback candidate を採用する経路である。

route selector 上の selected route は `m4326_center_primary_budget_guard_fallback` になる。

現時点の default allowed family は `lowT_strong_condensation_budget_stress` である。

### source-convention-safe electron refresh

T700 metal/sulfide cases で重要だった repair。

問題は electron policy そのものではなく、gas stationarity source の convention が refresh 側と RGIE 評価側でずれていたことだった。修正後は次の frame が一致する。

```text
q + gas_stationarity_source ~= A_g.T @ lambda
```

ここで `q` は gas log-density、`lambda` は element potential（元素ポテンシャルに対応する dual 変数）である。

source-convention-safe electron refresh は、case-specific shortcut ではなく、共通の gauge repair として扱う。

### reusable gas-boundary refresh policy

T500 の budget-tradeoff refresh と frontier 4 rows の adaptive floor repair は、同じ形で整理された。

考え方は次の通りである。

1. 複数の gas-boundary refresh candidate を作る。
2. 各 candidate を continuation に通す。
3. final barrier、budget、amount-weighted gas の条件を満たす最初の candidate を採用する。

frontier 4 rows では hardened gas log-density refresh に floor 候補を順に試す。

```text
1e-300, 1e-200, 1e-50, 1e-20
```

CaS/MgS/AlN T700 だけ `1e-200` が必要で、他の frontier rows は `1e-300` で accepted になった。

## acceptance tiers

HEAD route の accepted は単一品質ではない。現在は 3 tier に分ける。

| tier | metric status | 意味 | 現在の扱い |
|---|---|---|---|
| tier 1 | `tight_residual_components` | budget と amount-weighted gas が小さい。最も良い成功。 | `converged` |
| tier 2 | `accepted_budget_tradeoff_components` | final barrier には到達するが、budget tradeoff caveat がある。 | `converged_with_caveat` |
| tier 3 | `barrier_centered_with_raw_gas_caveat` | barrier と budget は良いが raw gas residual frame に caveat がある。 | `converged_with_caveat` |

`allow_caveat_tiers=False` の場合、tier 2/3 は public result 上では `not_converged` 扱いにできる。

## 14 representative rows と fresh profile の現状

HEAD route は、14 representative rows で fresh API regression を通る。v1.1 では results artifact に依存せず、public `condensate_equilibrium()` から 14 rows を実行する。

| group | rows | status |
|---|---:|---|
| tier 1 tight residual | 10 | production-adjacent candidate |
| tier 2 budget tradeoff | 2 | caveat 付き accepted |
| tier 3 raw-gas caveat | 2 | caveat 付き accepted |

10 curated case families との関係は以下である。

| family | representative case | current HEAD route behavior |
|---|---|---|
| `solar_highT_no_condensate_gas_regression` | high T gas-only | 凝縮 route 対象外。gas-only regression。 |
| `solar_silicate_first_condensation` | T1400/T1500 | promoted high-start route で accepted。 |
| `solar_water_condensation` | T300 | v1.1 native seed fallback gate で fresh API profile の中間層も caveat accepted。 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` | T700 | source-convention-safe electron refresh で accepted。ただし raw-gas caveat tier。 |
| `carbon_rich_graphite_window` | T1300 corrected | adaptive floor frontier repair で accepted。 |
| `carbon_rich_CaS_MgS_AlN_window` | T700 corrected | adaptive floor frontier repair で accepted。selected floor は `1e-200`。 |
| `SiO_s_condensate_window` | T900/P0.1 corrected | adaptive floor frontier repair で accepted。 |
| `lowT_strong_condensation_budget_stress` | T500 | depleted gas refresh budget tradeoff で accepted。ただし budget tradeoff tier。 |
| `near_phase_boundary_support_sensitivity` | T1490/T1510 | promoted high-start route で accepted。 |
| `complex_heavy_element_or_boron_titanium_zirconium_case` | T1100 corrected | titanium proxy support で accepted。 |

## 成功と caveat の意味

`converged` は、HEAD route standard gate 上で tight residual tier に入ったという意味である。

`converged_with_caveat` は、HEAD route が runtime result を返せるが、次のような注意が残るという意味である。

- v1.1 native seed fallback: lifecycle selector が accepted しないため、native gas equilibrium と budget-preserving condensate seed を caveat 付きで返す。
- T500: strict budget closure ではなく budget tradeoff を許している。
- T700 metal/sulfide: amount-weighted gas では良いが raw gas residual frame に注意がある。

したがって、14/14 rows は「同じ品質で完全成功」ではない。正確には、10 rows は tight、4 rows は caveat 付きである。

## 守るべき境界

HEAD route は凝縮あり初版標準経路として進めてよいが、以下は守る。

- gas-only `equilibrium()` の挙動を変えない。
- production return signature を変えない。
- presets/defaults wiring を追加しない。
- FastChem4 public/runtime/trace values を constructor input にしない。
- FastChem4 exact branch replay を acceptance target にしない。
- case、species、element を落として成功扱いにしない。
- caveat tier を tight success と混同しない。

## 現在の未解決

次に整理すべき課題は以下である。

1. T500 の budget tradeoff を strict budget closure または coupled-gas closure に近づける。
2. T700 raw-gas caveat を、raw gas residual と amount-weighted gas residual の frame 分解で解消できるか確認する。
3. 水凝縮中間層の `no_p_armijo_trial` を primary continuation 側で解消し、v1.1 fallback 依存を減らす。
4. target-guided support を、broad grid で破綻しない support policy に一般化する。
5. `condensate_equilibrium_profile()` を profile/layer 計算へ接続する。
6. docs と examples で、HEAD route の public API 使用例を整備する。

## 更新ルール

HEAD route を変更した場合は、少なくとも次を更新する。

1. `head_route.md`
2. `curated_case_family_scorecard_ja.md`
3. 関連する `src/exogibbs/condensates/*`
4. 関連する API tests
5. 必要なら docs/examples

HEAD route の変更は、既存 accepted rows を壊さないこと、または壊した場合に理由と新しい acceptance policy を明示することを条件にする。
