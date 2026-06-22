# HEAD route 定義

この文書は、ExoGibbs の凝縮あり計算で現在の基準経路として扱う **HEAD route** を定義する。HEAD route の内容を変更した場合は、この文書を更新する。

現在の実装版は **HEAD route v1.18** である。v1.18 は v1.17 の `pdipm_core` primary continuation、Ipopt-style tiny-step handling、full-condensate budget feasibility restoration を維持しつつ、native fallback が restricted solver payload を持たない場合でも、有限な PD-IPM lifecycle final state から support growth を続行する。fallback public state も seed candidate と lifecycle final-state candidate を full-budget gate で比較し、seed が既に accepted の場合は未受理 final state で上書きしない。さらに native fallback に限り、full-budget gate 直前に gas log amount と active condensate amount を joint に動かす feasibility restoration を試し、元素 budget residual が改善または accepted する場合だけ反映する。explicit support payload が狭すぎる場合は、ExoGibbs-native inactive driving から support を広げる `explicit_support_closure_retry` を一度だけ試す。explicit empty support payload は support-free empty support と同じ `empty_support_strict_gas_retry` を使って gas-only full-budget gate を修復する。v1.18 の public diagnostics は `caveat_route_breakdown` を持ち、`converged_with_caveat` が primary stop、fallback source、final restoration、PD-IPM retry rejection のどこに由来するかを分解して記録する。soft restoration を primary stop で自動起動する default route は、water5/7/8 の status を改善せず runtime を悪化させたため採用しない。core mode は reduced PD-IPM direction、scalar fraction-to-boundary、primal/dual 別 step-length diagnostics、persistent filter、soft/dedicated restoration を本線 sequence として扱う。tiny な primary step は Newton direction を component-wise に壊さず、restoration phase に渡す。HEAD route は active-set orchestration と final gates を担当し、FastChem4 output は比較対象としてのみ使い、constructor input には使わない。現在の curated fresh API score は `99 / 99 converged` である。

## 一言でいうと

現在の HEAD route は、凝縮あり equilibrium（平衡計算）を次の順序で進める経路である。

1. ExoGibbs-native thermochemistry（ExoGibbs 内の熱化学データ）と elemental budget（元素予算）だけから、positive support（ゼロでない凝縮種候補）を作る。
2. 小さい正の condensate amount seed（凝縮量初期値）を入れる。
3. 凝縮で使う元素を気相 budget から差し引き、depleted gas refresh（凝縮後の元素予算で気相を作り直す処理）を行う。
4. restricted support solver（選んだ凝縮種だけを動かす制限付き solver）に入れる。
5. 失敗または不十分な場合は、HEAD route lifecycle（外側制御）で primary continuation、fallback、electron refresh、frontier refresh を試す。
   v1.2 では、restricted support solver が成功しても lifecycle selector が accepted しない fresh API runtime layer について、finite warm-start candidate が残っていれば native seed fallback を許可する。
   v1.3 では、support-free outer loop 内に center-gate retry、residual-worsening retry、soft-restoration retry、Ipopt-style persistent h-type retry、support-cap retry を追加し、native seed fallback へ落ちる runtime layer を減らす。
6. v1.4 では、返却直前に gas と full condensate vector から元素 budget を再構成し、element-wise relative residual が許容値を超える accepted row を relative joint budget-correction retry、budget-preserving seed retry、empty-support strict gas retry で直す。retry 後も許容値を超える row だけを `not_converged` に降格する。
7. v1.5 では、support-cap retry と support-growth staging retry の候補を採用する前に、候補の gas state から inactive condensate driving を再評価し、positive inactive driving が許容値を超える候補を採用しない。
8. v1.6 では、support outer loop の次 round を作るとき、前 round で試した support ではなく、accepted result の実 support indices を既存 support として使う。実 support から落ちた species がまだ positive inactive なら、通常の support-growth で再追加する。
9. v1.7 では、`return_diagnostics=True` の result に `inactive_condensate_driving` report を追加し、all-condensate と temperature-valid の inactive-driving summary を分ける。
10. v1.8 では、support-cap retry と staged support-growth retry の候補を全て評価し、converged かつ support-closure gate を通る候補の中から `(positive_inactive_count, max_positive_inactive_driving, support_count)` が最小の候補を採用する。
11. v1.9 では、support-cap retry family が候補を持つ場合でも staged support-growth retry family を評価し、両 family を横断して同じ support-closure score で最良候補を選ぶ。
12. v1.10 では、通常の component clipping primary が fallback になった support-free row に対して、同じ support-free path を scalar fraction-to-boundary step control で再評価する retry family を追加し、support-cap / staged support-growth retry と同じ support-closure score で比較する。
13. v1.11 では、PD-IPM-first 方針として scalar fraction-to-boundary を primary default に昇格する。v1.10 は production-safe baseline として残し、v1.11 は solver 本体の blocker を減らすための baseline として扱う。
14. v1.12 では、active support の dual `eta = exp(rho)` に `0.1` の push floor を入れ、`rho = epsilon - r` だけでは小さすぎる dual を interior に戻す。これは Ipopt の multiplier initialization に対応する修復で、FastChem4 値は使わない。
15. v1.13 では、収束済み primary result でも temperature-valid inactive driving が severe (`max >= 1000`, `count >= 50`) に残る場合、既存の staged support-growth retry を support-closure score で選ぶ。`solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 は support 14 / inactive `96 / 1012.9` から support 67 / inactive `0 / 0` に改善する。
16. v1.14 では、primary continuation default を `pdipm_core` mode に切り替える。core mode は alternate direction candidates を本線から外し、reduced PD-IPM direction と Ipopt-style filter/restoration sequence を標準にする。explicit-support surface は metric 悪化を blocker として記録し、support-free midlayer runtime blocker も score に残す。
17. v1.15 では、Ipopt の line-search 周辺制御にならって `pdipm_core` に tiny-step detection を追加する。`alpha_primal <= 1.0e-8` の primary step は通常受理ではなく restoration に渡し、report に tiny-step count と理由を残す。
18. v1.16 では、solver 数式は変えず、`dataclasses.asdict()` による deep copy と JAX 配列の診断 gather を hot path から外す。support-free midlayer regression は timeout ではなく完走する。
19. v1.17 では、accepted primary/fallback state の full-condensate budget gate 直前に gas log-amount restoration と active-support bounded least-squares amount restoration を入れる。これは PD-IPM が作った active support 内の feasibility restoration であり、FastChem4 trace/replay は使わない。
20. v1.18 では、fallback result に restricted solver payload がなくても lifecycle `final_state` が有限なら、その PD-IPM final state を使って次の support-growth round を作る。public fallback state は seed と lifecycle final state を full-budget gate で比較し、accepted seed を未受理 final state で壊さない。native fallback の final gate 直前には、Ipopt-style restoration 方針に沿って gas log amount と active condensate amount の joint feasibility restoration を試す。
21. v1.18 固定前の final cleanup では、explicit support payload が狭すぎる `complex_heavy...` layer 4 を `explicit_support_closure_retry` で修復し、explicit empty support payload の highT gas-only rows にも既存の `empty_support_strict_gas_retry` を配線する。どちらも FastChem4 trace/replay を constructor input に使わない。
22. 最後に selected route（採用経路）と acceptance tier（成功品質ランク）を返す。

これは FastChem4 exact replay（FastChem4 の分岐を完全再現すること）ではない。FastChem4 public/runtime/trace values（公開出力・実行時出力・内部 trace 値）は、ExoGibbs の constructor input（初期値や構成入力）として使わない。

## 次の solver 改善方針

HEAD route v1.4-v1.10 の改善は、public convergence、budget consistency、support closure を守るために retry / gate / diagnostics を厚くしてきた。v1.11 では、metric 悪化を許容して PD-IPM を主役に戻す方針へ切り替えた。v1.12 では、その blocker を support retry ではなく dual initialization として修復した。v1.13 では、PD-IPM が収束した active support の KKT は良いが inactive closure が severe に悪い行を、PD-IPM 診断に基づく support expansion として扱う。v1.14 では、route-level retry をさらに足すのではなく、primary continuation 自体を `pdipm_core` mode に切り替えた。v1.15 では、Ipopt-style tiny-step handling を core mode に入れ、tiny step を component-wise clip ではなく restoration へ渡す。v1.16 では、この PD-IPM 本線を維持したまま runtime blocker になっていた診断 serialization を削った。v1.17 では、PD-IPM が作った active support の final feasibility restoration を強め、trace 元素の gas log amount と active condensate amount を ExoGibbs-native full-budget gate に合わせて修復する。v1.18 では、support-boundary growth の情報源も raw fallback seed ではなく PD-IPM lifecycle final state に寄せる。v1.18 固定時点では、狭すぎる explicit support payload と explicit empty support payload の API 経路差も、既存の ExoGibbs-native retry/gate 体系にそろえて修復済みである。

v1.13 以降の score では、accuracy / convergence だけでなく runtime diagnostics も保存する。最低限、curated end-to-end wall time、代表 target solve wall time、target metric-evaluation wall time、full-profile comparison の完走または中断状態を `score.md` / `score.json` に残す。大 support row で数分級の solve が出た場合は、support selection time と active-support solve time を分けて記録する。

次に凝縮 solver を改善する場合は、新しい support retry や acceptance fallback を追加する前に、`documents/ipm_audit.md` の PD-IPM 再整備方針に従う。優先順位は、active-support fixed PD-IPM の residual/Jacobian/step acceptance を検証し、`rho` / `eta` / complementarity の表現を一本化し、log-space update と step-length control を一貫させることである。FastChem4/fastchem3 replay を constructor input に使わず、ExoGibbs-native KKT residual と finite-difference Jacobian audit を先に固定する。

2026-06-19 の PD-IPM/R-GIE audit では、component-wise clipping が GIE Newton 方向の結合を壊しうることを確認し、代わりに scalar fraction-to-boundary step control を候補として実装した。explicit-support curated lifecycle 比較では、既定の component clipping が 0/14 converged だったのに対し、scalar fraction-to-boundary は 12/14 converged まで改善した。v1.10 ではこの policy を retry candidate として導入したが、2026-06-20 の方針変更で v1.11 は scalar を primary default に昇格する。詳細と volatile artifact path は `documents/ipm_audit.md` に残す。

## HEAD route v1.11 固定内容

HEAD route v1.11 は、v1.10 の scalar retry を public primary default に昇格した PD-IPM-first baseline である。

| item | fixed choice | rationale |
|---|---|---|
| route version | `v1.11` | production-safe v1.10 ではなく PD-IPM-first baseline として区別する |
| route name | `head_route_v1_11_pdipm_scalar_primary` | scalar fraction-to-boundary primary を明示する |
| primary step control | `scalar_fraction_to_boundary` | coupled R-GIE/Newton direction を成分別 clip で壊さず、scalar step length で境界を守る |
| legacy component clip | opt-in / comparison path | production-safe v1.10 の比較対象として残す |
| blocker policy | expected and tracked | public metric 悪化を solver 改善 queue として扱う |

v1.11 curated end-to-end surface:

| metric | result |
|---|---:|
| curated e2e tests | 8 passed |
| expected support-free blockers | `solar_water_condensation` layers 0 and 7 |
| explicit-support blocker | `carbon_rich_graphite_window__T1300_P1_corrected`; heavy/Ti/Zr remains caveat, not hard reject |
| water low-T layer 8 | primary scalar route succeeds with 143 support; no scalar retry selection needed |

v1.11 full-profile FastChem4 comparison:

| metric | v1.10 production-safe | v1.11 PD-IPM-first |
|---|---:|---:|
| rows | 99 | 99 |
| public status | 99 converged | 95 converged, 1 caveat, 3 not converged |
| route counts | 82 primary, 17 gas-only | 78 primary, 4 fallback, 17 gas-only |
| ExoGibbs lower `G/RT` | 19/99 | 32/99 |
| FastChem4-scaled lower `G/RT` | 80/99 | 67/99 |
| max `\|dG/RT\|` | 5.936e-5 | 3.297e3 |
| max temperature-valid inactive driving | 194.8 | 343.2 |
| temperature-valid inactive rows >500 | 0 | 0 |

The largest v1.11 regression is `solar_water_condensation` layer 0: `dG/RT Exo-FC = -3297.0902626575994` with very large relative budget residual. This is now the primary PD-IPM-first blocker.

2026-06-20 の Ipopt-oriented blocker audit では、この row は support retry の不足ではなく、active support 内の primal/dual interior point の悪さで scalar fraction-to-boundary step が `2.2303859960980942e-08` まで潰れる問題として分解された。limiting species は `FeS2(s)` の `r` update で、次の修復対象は component-wise clip への復帰ではなく、Ipopt 的な initial point / dual initialization / filter-restoration の整備である。詳細は `documents/ipm_audit.md` に残す。

## HEAD route v1.12 固定内容

HEAD route v1.12 は、v1.11 の PD-IPM-first scalar primary に Ipopt-style dual push initialization を統合した修正版である。

| item | fixed choice | rationale |
|---|---|---|
| route version | `v1.12` | v1.11 scalar primary baseline の blocker 修復として区別する |
| route name | `head_route_v1_12_ipopt_dual_push_primary` | dual initialization repair を明示する |
| primary step control | `scalar_fraction_to_boundary` | coupled R-GIE/Newton direction を component-wise clip で壊さない |
| dual initialization | `ipopt_push_floor` | active-support dual を Newton 前に interior へ押し戻す |
| dual push floor | `0.1` | water layer 0/5/8 を tier 1 に戻し、full-profile public convergence を 99/99 に戻す |
| FastChem4 constructor inputs | not used | fresh ExoGibbs API path のみで再現する |

v1.12 full-profile FastChem4 comparison:

| metric | v1.11 PD-IPM-first | v1.12 dual-push repair |
|---|---:|---:|
| rows | 99 | 99 |
| public status | 95 converged, 1 caveat, 3 not converged | 99 converged |
| route counts | 78 primary, 4 fallback, 17 gas-only | 82 primary, 17 gas-only |
| ExoGibbs lower `G/RT` | 32/99 | 38/99 |
| FastChem4-scaled lower `G/RT` | 67/99 | 61/99 |
| max `\|dG/RT\|` | 3.297e3 | 9.632e-2 |
| max temperature-valid inactive driving | 343.2 | 1012.9 |
| temperature-valid inactive rows >0 | 16 | 3 |
| temperature-valid inactive rows >500 | 0 | 1 |

The previous largest v1.11 blocker, `solar_water_condensation` layer 0, now converges as tier 1 with inactive closure `0 / 0`. `solar_water_condensation` layer 5 also returns to tier 1 under the `0.1` dual floor. The remaining largest FastChem4-scaled Gibbs gap is `carbon_rich_graphite_window` layer 7 (`dG/RT Exo-FC = -0.09632193272911493`), and the largest temperature-valid inactive driving is now `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 (`1012.9452767757098`, 96 species). These are next solver/support-closure targets, not reasons to restore component clipping.

## HEAD route v1.10 production-safe baseline

HEAD route v1.10 は、v1.9 の retry family 横断 support-closure selection を、PD-IPM step-control candidate まで広げる固定版である。通常の primary continuation は `component_clip` のまま走る。通常 path が `native_budget_seed_fallback_budget_tradeoff` になり、caller が support count を明示していない support-free row だけ、同じ fresh API path を `head_route_primary_step_control_policy="scalar_fraction_to_boundary"` で一度再実行する。この scalar retry は support-cap retry / staged support-growth retry と同じ候補 pool に入り、同じ `(positive_inactive_count, max_positive_inactive_driving, support_count)` score で採否を決める。

v1.10 は production-safe baseline として残す。2026-06-20 の検証で scalar primary default は curated support-free water rows を `not_converged` に戻すことが分かったが、v1.11 ではこれを blocker として受け入れ、solver 本体の改善対象にする。

| promoted item | 目的 | 適用範囲 |
|---|---|---|
| scalar step-control retry family | component-wise clipping で primary が fallback になる row に、Newton 方向を壊さず scalar fraction-to-boundary で進む lifecycle candidate を追加する。 | support-free fallback-only retry |
| cross-family closure selection extension | support-cap / staged support-growth / scalar step-control retry を同じ support-closure score で比較し、retry kind ではなく ExoGibbs-native closure で採用する。 | support-free fallback-only retry |
| primary default preservation | 通常 primary の `component_clip` default は維持し、scalar policy は fallback-only candidate としてだけ自動評価する。 | public HEAD route default |

v1.10 audit では、curated end-to-end tests は `8 passed` を維持する。`solar_water_condensation` layer 8 の support-growth regression row は、v1.9 の staged support-growth candidate では 162 support / inactive closure 0/0 だったが、v1.10 では scalar step-control retry が 143 support / inactive closure 0/0 で同じ support-closure score の先頭 2 要素を満たし、support count が小さいため採用される。これは FastChem4 exact replay ではなく、ExoGibbs-native support closure score による candidate selection の更新である。

## HEAD route v1.9 固定内容

HEAD route v1.9 は、v1.8 の support-closure score を retry family 横断の selection policy に拡張する固定版である。v1.8 では support-cap retry 内で best closure candidate を採用できたが、support-cap retry family に採用可能候補があると staged support-growth retry を比較対象にしないため、より closure が良い staged candidate を見逃す row が残った。v1.9 は support-cap retry と staged support-growth retry の候補を一つの pool として扱い、同じ `(positive_inactive_count, max_positive_inactive_driving, support_count)` score で最良 candidate を採用する。

| promoted item | 目的 | 適用範囲 |
|---|---|---|
| cross-retry support-closure selection | support-cap retry と staged support-growth retry の両方を configured candidate pool として評価し、retry kind によらず closure score が最良の converged candidate を採用する。 | support-free fallback-only retry |
| common retry selection diagnostics | 採用された旧互換 key に加えて `support_closure_retry_selection` を残し、candidate count、selected retry kind、cap/staging の全 attempts、selected score を記録する。 | retry diagnostics |
| support-cap/staging ordering separation | 個々の retry family 内の順序は diagnostics と再現性のため維持し、採用判断だけを cross-family score へ集約する。 | support-free fallback-only retry |

v1.9 audit では、v1.8 の最大 Gibbs 差だった `solar_metal_sulfide_or_Fe_Ni_S_region` layer 5 が cap-128 support-cap retry ではなく staged support-growth retry (`add_per_round=32`) を採用する。temperature-valid inactive driving は 60.55 / 12 種から 0 / 0 種へ下がり、同 row の FastChem4-scaled comparison は `dG/RT Exo-FC = 9.972649e-5` から `1.193677e-6` へ改善する。99-layer full-profile comparison では public status は `99 converged`、route counts は `82 primary, 17 gas-only` を維持し、max `|dG/RT|` は `5.935862e-5` へ下がる。

v1.9 は FastChem4 exact replay ではない。FastChem4-scaled state の方が ExoGibbs-native `G/RT` で低い row はまだ 80/99 あり、今回の改善は ExoGibbs-native support closure の selection policy 更新である。

## HEAD route v1.8 固定内容

HEAD route v1.8 は、v1.7 の public diagnostics と highT gas-only 解釈を維持したまま、fallback-only retry の採用規則を強化する固定版である。v1.7 では support-cap retry が最初に `max_positive_inactive_driving <= 500` を満たした候補を採用したため、`SiO_s_condensate_window` layer 4 で max driving は 470 と閾値内でも temperature-valid inactive condensates が 77 種残る candidate を採用していた。v1.8 では retry candidate を最初の合格で止めず、全 configured cap / staging candidate を比較して closure score が最も良い converged candidate を採用する。

| promoted item | 目的 | 適用範囲 |
|---|---|---|
| best support-closure retry candidate selection | support-cap retry / staged support-growth retry で、最初の合格候補ではなく inactive count、max driving、support count の lexicographic score が最良の converged candidate を採用する。 | support-free fallback-only retry |
| inactive-count-aware gate diagnostics | gate report に `positive_inactive_count_tolerance` と count/max それぞれの accepted flag を残す。default では count を hard reject には使わず、candidate ranking に使う。 | retry diagnostics |
| configured cap retry coverage | current broad support より大きい configured cap も retry candidate として評価し、large cap で solver が実 support を落として closure する path を見逃さない。 | support-cap retry |

v1.8 audit では、v1.7 の最大 Gibbs 差だった `SiO_s_condensate_window` layer 4 が cap 34 ではなく cap 128 retry を採用し、temperature-valid inactive driving は 470.8 / 77 種から 0 / 0 種へ下がる。同 row の FastChem4-scaled comparison は `dG/RT Exo-FC = 7.952e-4` から `5.905e-6` へ改善する。99-layer full-profile comparison では public status は `99 converged`、route counts は `82 primary, 17 gas-only` を維持し、max `|dG/RT|` は `9.973e-5` になる。

## HEAD route v1.7 固定内容

HEAD route v1.7 は、v1.6 の route selection、support outer loop、solver acceptance を変えない diagnostics 更新である。主目的は、highT gas-only boundary で温度範囲外の condensate が legacy all-condensate inactive-driving metric を支配し、実際の support miss のように見える問題を切り分けることである。

| promoted item | 目的 | 適用範囲 |
|---|---|---|
| validity-aware inactive-driving diagnostics | `return_diagnostics=True` の result に `inactive_condensate_driving` report を追加し、all-condensate summary と temperature-valid summary を併記する。 | public condensate API diagnostics |
| support-selection validity alignment | temperature-valid subset は HEAD support selection と同じ `temperature_validity_upper` metadata を使う。 | diagnostics only |
| gas-only route comparability | empty positive support route と通常の gas-only equilibrium solver を、同じ validity-aware inactive-driving 指標で比較できるようにする。 | diagnostics only |

この更新は solver route、support update、public convergence、budget gate を変更しない。FastChem4 output は比較対象としてのみ使い、constructor input には使わない。

v1.7 audit では、`solar_highT_no_condensate_gas_regression` の legacy all-condensate inactive driving は HEAD empty-support route と direct gas-only solver の両方で約 772 と一致する一方、temperature-valid inactive driving は HEAD 側で 0 である。したがって highT gas-only hotspot は温度範囲外 condensate を legacy metric が数えた diagnostic artifact であり、温度有効な condensate support miss ではない。

## HEAD route v1.6 固定内容

HEAD route v1.6 は、v1.5.1 の score surface に残った水凝縮の support-closure hotspot を、後処理 retry ではなく support outer loop の更新規則で直す固定版である。FastChem4 output は比較対象として使い、constructor input には使わない。

| promoted item | 目的 | 適用範囲 |
|---|---|---|
| actual accepted support growth | support outer loop の次 round で、non-fallback result の `condensate_support_indices` を existing support として使う。前 round で solver に渡したが実際には落ちた species は、既存扱いにしない。 | support-free activity-driven outer loop |
| dropped-support reactivation | solver が落とした species が final gas state / restricted dual state から再び positive inactive と判定される場合、既存の support-growth 規則で再追加する。 | support-free activity-driven outer loop |
| existing retry policy reuse | 固定の residual retry、固定の追加数、専用 seed floor は導入せず、既存の `max_support_outer_iterations`、`max_support_add_per_round`、budget seed、support ordering をそのまま使う。 | support-free activity-driven outer loop |

v1.6 では新しい public option は追加しない。fallback payload から grow する場合は、従来どおり payload の support indices を使う。non-fallback accepted result から grow する場合だけ、accepted result の実 support indices と full condensate amount vector を warm-start の基準にする。

v1.6 の fresh API support-free route selection は v1.5 と同じ public status 面を維持する。

| surface | gas-only route | primary route | native seed fallback route | exception | public converged | public not_converged |
|---|---:|---:|---:|---:|---:|---:|
| 10 curated full-profile families | 17 | 82 | 0 | 0 | 99 | 0 |

v1.6 audit では、v1.5.1 で最大だった `solar_water_condensation` layer 8 の positive inactive driving が約 2579 から約 22.4 に下がり、active condensate count は 69 から 162 に増える。`solar_water_condensation` layer 2 も約 1662 から約 4.62 に下がる。global max positive inactive driving は約 2579 から約 771.7 に下がり、`>1000` rows は 2 から 0 になる。FastChem4-scaled Gibbs comparison は ExoGibbs lower 17 / FastChem4-scaled lower 82 から ExoGibbs lower 19 / FastChem4-scaled lower 80 に小さく改善する。したがって v1.6 は support-closure と water gas outlier を改善するが、FastChem4 exact replay ではない。

## HEAD route v1.5 固定内容

HEAD route v1.5 は、trial v1.5.1 の結果を昇格し、v1.4 の full-budget gate と public convergence 修復を維持したまま、support-free retry の採用条件を強化する固定版である。v1.4 の FastChem4 比較では、public budget residual は閉じていた一方、support-cap retry が小さい support で早く promoted route を返す row に、大きな inactive positive condensate driving が残ることが分かった。v1.5 はこの大ぽかを ExoGibbs-native state だけで検出する。

| promoted item | 目的 | 適用範囲 |
|---|---|---|
| support-closure retry gate | `support_cap_retry` と `support_growth_staging_retry` の候補 result に対し、候補の `gas_ln_n` と採用済み support indices から activity-driven support report を再評価する。support 外に残る positive inactive driving の最大値が `support_closure_max_positive_inactive_driving` を超える候補は、non-fallback route でも採用しない。 | support-free fallback-only retry |
| retry exception isolation | retry 候補が非有限 amount などで例外になっても、outer loop 全体を落とさず、その候補を diagnostics の failed attempt として記録して次の cap/staging 候補へ進む。 | support-cap retry / support-growth staging retry |

既定値は `enable_support_closure_retry_gate=True`、`support_closure_max_positive_inactive_driving=5.0e2` である。この gate は FastChem4 active list や FastChem4 runtime trace を使わない。採用された retry diagnostics には `support_closure_accepted` と `retry_support_closure_gate` を残し、各 attempt にも `support_closure_gate` を残す。

v1.5 の fresh API support-free route selection は v1.4 と同じ public status 面を維持する。ただし、早い小cap retryが大きな inactive driving を残す場合は、より大きい cap または staged support-growth retry へ進む。

| surface | gas-only route | primary route | native seed fallback route | exception | public converged | public not_converged |
|---|---:|---:|---:|---:|---:|---:|
| 10 curated midlayers | 1 | 9 | 0 | 0 | 10 | 0 |
| 10 curated full-profile families | 17 | 82 | 0 | 0 | 99 | 0 |

v1.5 audit では、v1.4 で最も大きかった inactive driving rows が次のように変わる。`solar_water_condensation` layer 7 は small-cap retry ではなく staged support-growth retry を採用し、positive inactive driving は 0 になる。`lowT_strong_condensation_budget_stress` layer 7 は staged retry で最大 driving 約 19.7 まで下がる。`carbon_rich_CaS_MgS_AlN_window` layer 7、`solar_metal_sulfide_or_Fe_Ni_S_region` layer 5、`SiO_s_condensate_window` layer 8 は cap 80 retry が closure gate を通り、最大 driving は許容値内に収まる。

## HEAD route v1.4 固定内容

HEAD route v1.4 は、v1.3 の route selection と retry sequence を維持したまま、public result の受理判定を強化する固定版である。

| promoted item | 目的 | 適用範囲 |
|---|---|---|
| full condensate element budget gate | final gas amount と full condensate vector から `A_gas n + A_cond m - b` を再計算し、元素ごとの相対 residual 最大値が `full_condensate_budget_relative_tolerance` を超える accepted/caveat row を `not_converged` に降格する。 | public `condensate_equilibrium()` result |
| electron row exclusion | `e-` は化学元素 budget gate の対象から外し、diagnostics の ignored element として残す。 | full condensate element budget gate |
| external condensate budget in RGIE/PD-IPM residuals | thermo-invalid filtering などで Newton/Jacobian 変数から外れた凝縮種の現在量を `A_cond,E m_E` として元素保存 residual、merit、direction builder に加え、active subproblem が full condensate budget と同じ収支を解くようにする。 | algorithm-v1.1 RGIE/PD-IPM callsites |
| lifecycle final-state public result wiring | restricted solver 成功後の public result に solver 初期状態ではなく lifecycle continuation の final state を反映し、support-growth / support filtering 後の lifecycle support indices と合わせて full condensate budget gate が実際の accepted state を評価するようにする。 | public `condensate_equilibrium()` result |
| relative joint budget-correction retry | full budget gate が落ちた accepted row では、active condensate 量だけで帳尻を合わせず、gas amount と active condensate amount を同じ linearized budget system で動かす。`budget_row_scaling_policy="relative_target"` と `relative_budget_max` filter component により、trace element の budget residual が絶対量の大きい元素に埋もれないようにする。 | algorithm-v1.1 continuation retry |
| budget-preserving seed retry | support-free outer loop が `max_density` seed で native seed fallback に落ち、public full-budget gate も通らない場合、shared budget fraction を守る `budget_preserving_fraction` seed で同じ fresh API path を一度だけ再実行する。 | support-free fallback-only retry |
| empty-support strict gas retry | positive condensate support が空の gas-only row で v1.4 full-budget gate が落ちた場合、`epsilon_crit=1.0e-12` の gas-only solve を一度だけ試す。 | support-free empty support route |
| budget-correction retry final-state start | full budget gate で落ちた accepted lifecycle row の retry は、restricted solver output へ戻らず、lifecycle continuation の final `ln_nk` / `ln_mk` / support indices から開始する。diagnostics には `retry_start_state="lifecycle_final_state"` を残す。 | algorithm-v1.1 continuation retry |
| condensate capacity cap in continuation trials | 各 active condensate amount を、その凝縮種が単独で使える元素 budget 上限 `min_i b_i / A_cond,i,k` 以下に抑え、小さい budget の元素を含む extra support が過剰量を持つことを防ぐ。 | algorithm-v1.1 continuation trial / accepted state update |
| final support amount polish | lifecycle final state が accepted した後、gas state を固定し、support amount だけを相対 budget LS と capacity cap で微修正する補助 polish。full budget gate が accepted した場合だけ public support amounts に反映する。 | public `condensate_equilibrium()` result |

既定値は `enable_full_condensate_budget_residual_gate=True`、`full_condensate_budget_relative_tolerance=1.0e-3` である。gate が reject した場合、`acceptance_tier` は `full_condensate_element_budget_residual_failed` になり、diagnostics に `full_condensate_budget_residual_gate`、`pre_full_condensate_budget_gate_status`、`pre_full_condensate_budget_gate_acceptance_tier` を残す。

v1.4 の fresh API support-free route selection は v1.3 の primary route を基本にする。ただし public status は full budget gate 後の値になる。fallback seed 過剰と empty-support gas 精度の guarded retry 後、現在の full-profile audit では public `not_converged` は残らない。

| surface | gas-only route | primary route | native seed fallback route | exception | public converged | public not_converged |
|---|---:|---:|---:|---:|---:|---:|
| 10 curated midlayers | 1 | 9 | 0 | 0 | 10 | 0 |
| 10 curated full-profile families | 17 | 82 | 0 | 0 | 99 | 0 |

v1.4 固定後の次課題は、`condensate_equilibrium_profile()` を support-free API path に接続すること、public API examples/docs を整備することである。2026-06-13 の maintenance update では、`ref/tce_v1.3.tex` に外部凝縮種 budget の式を追加し、RGIE/PD-IPM の residual/merit/direction/callsite に `external_condensate_budget` を通した。さらに lifecycle continuation final state を public result に反映し、support-growth / support filtering 後の lifecycle support indices で final-state amounts を展開するように直した。その後、小さい budget の元素を含む active condensate が過剰量を持たないように、continuation capacity cap と final support amount polish を追加し、full-budget gate fail accepted row には lifecycle final state から relative joint budget-correction retry を適用するように固定した。さらに remaining rejects の分解結果に基づき、native seed fallback の過剰 seed には budget-preserving seed retry、empty-support gas-only row には strict gas retry を追加した。現在の 99-layer audit では public status count は `99 converged / 0 not_converged` である。

## HEAD route v1.3 固定内容

HEAD route v1.3 は、v1.2 の fresh API runtime path に次の guarded retry と数値安定化を追加した固定版である。

| promoted item | 目的 | 適用範囲 |
|---|---|---|
| stable L2 residual norms | large finite condensate activity residual の norm 計算 overflow を避ける。 | restricted solver / continuation residual reports |
| center-gate retry | `current_barrier_not_centered` で止まる support-free row を、緩い center tolerance で再評価する。 | support-free fallback-only retry |
| residual-worsening retry | p-Armijo/filter candidate が residual nonworsening guard で止まる row を、guarded tolerance 付きで再評価する。 | support-free fallback-only retry |
| support-cap retry | broad support の一括選択が悪い場合に、小さい support cap から fresh API path を再実行する。 | support-free fallback-only retry |
| soft-restoration retry | 通常 retry 後も accepted しない row に、component-weighted restoration を一度試す。 | support-free fallback-only retry |
| Ipopt-style persistent h-type retry | objective descent ではなく feasibility violation 改善を採用する h-type step を一度試す。 | support-free fallback-only retry |
| support-growth warm-start amount floor | support growth 時に 0/非有限 amount を finite seed へ戻し、warm-start candidate 全滅を避ける。 | support-free outer loop |
| staged support-growth retry | final support set 自体は解けるが、一括投入の初期化が悪い row を段階投入で再実行する。 | support-free fallback-only retry |

v1.3 default fresh API support-free route selection evidence は次の通りである。

| surface | gas-only | primary | native seed fallback | exception |
|---|---:|---:|---:|---:|
| 10 curated midlayers | 1 | 9 | 0 | 0 |
| 10 curated full-profile families | 17 | 82 | 0 | 0 |

最後に残っていた `solar_water_condensation` layer `7` は、`H2O(s,l)` がちょうど activity-positive へ入る support transition row だった。default one-shot support growth では 154 support species を初回 round で一括投入して fallback になったが、同じ 154 species support でも staged support-growth retry では primary route に到達する。したがって v1.3 は default one-shot path を先に試し、fallback の場合だけ `support_growth_staging_retry_add_per_rounds=(64, 32, 16, 8)` を試す。

この v1.3 evidence は route selection の記録であり、v1.4 以降の public `converged` count とは一致しない。

## 現在の実装入口

凝縮ありの公開入口は以下である。

- `src/exogibbs/api/condensate_equilibrium.py`
  - `CondensateChemicalSetup`
  - `CondensateEquilibriumOptions`
  - `CondensateEquilibriumResult`
  - `build_condensate_chemical_setup`
  - `condensate_equilibrium`

`condensate_equilibrium()` が返す `CondensateEquilibriumResult` から、現在の固定版は
`result.head_route_version == "v1.18"` および
`result.head_route_name == "head_route_v1_18_pdipm_lifecycle_support_growth"` として読み出せる。
`return_diagnostics=True` の場合は diagnostics にも同じ `head_route_version` と
`head_route_name` を残し、さらに `inactive_condensate_driving` report を残す。

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

### 2. activity-driven support selection

ユーザーが `support_indices` を渡さない場合、HEAD route v1.4 は native gas equilibrium から
activity-driven support outer loop を開始する。

ここでは、FastChem4 runtime output から support を取らない。使うのは以下だけである。

- `formula_matrix_cond`
- `element_inventory_target`
- `condensate_species_order`
- `hvector_cond`
- native gas state から作る element potential
- condensate preset metadata の temperature validity upper bound

activity driving が正で、元素 budget capacity を持ち、temperature validity 内にある凝縮種だけを
activity-positive 候補として追跡する。`max_activity_support_count=None` の場合、この候補追跡は
個数で切らない。これは FastChem4 の active condensate list に近い診断境界であり、FastChem4
runtime output を constructor input として使うものではない。

restricted solver に実際に渡す support は `max_positive_support_count` で別に制御する。
`max_positive_support_count=None` の場合は positive 候補を個数で制限せず solver に渡せるが、
現時点では large support set が restricted solver で収束する保証はない。bounded support の
優先順位は capacity を主に使い、明らかな trace-species dominance を避ける。

restricted solver step では `restricted_reduced_coupling_mode` で reduced-coupling 方向を選べる。
HEAD route v1.4 の default は v1.3 から引き続き `pdipm_rgie_v11_activity_correction` である。
`candidate_selected_active_plus_near_jacobian` は FastChem4 の
active condensate list / near-active Jacobian 縮約に近い実験 mode である。この mode は
ExoGibbs native state から activity proxy を作り、active と near-active を solver 内部で分ける。
`candidate_selected_active_plus_near_jacobian_with_rem_inventory` は、near-active でない support
凝縮相を Newton/Jacobian 変数から外し、その現在量を `b_eff = b - A_cond m_rem` として元素収支から
差し引く opt-in 実験 mode である。これは FastChem4 の condensates_rem の inventory subtraction
に対応する入口であり、rem 量には correctValues 型の解析更新も適用する。ExoGibbs の RGIE reduced
elimination では condensed variables がもともと `pi` 方向から解析更新されるため、この mode は
`current` reduced direction と数値的に近くなる。水凝縮・低温強凝縮・silicate の sampled curated
run では例外なく実行できたが、route 改善はまだ確認されていない。

`pdipm_rgie_v11_activity_correction` は、FastChem4 の condensed solver が持つ
`activity_correction` 変数に対応する明示 `rho/eta` state を持つ opt-in 実験 mode である。
初期値は `rho0 = 0, eta0 = 1`、すなわち FastChem4 の新規 active condensate に対する
`activity_correction = 1` に対応する。barrier/tau は FastChem4 の
`condTau * reference_element_budget` gauge に直し、reference element は
`element_inventory_target / stoichiometric_coefficient` が最小の元素として ExoGibbs native
budget から決める。入力は ExoGibbs native gas/condensate thermochemistry と runtime budget
から構成し、FastChem4 public/runtime/trace values は constructor input に使わない。
初期 element-potential carrier は tce の有効気相 source
`h_gas + log(P/Pref) - ln_ntot` を使い、`A_g.T pi ~= ln_n + source` の least-squares で作る。
旧 p-IPM/RGIE residual helper から recovered `pi` を流用しない。
midlayer 10 curated family では fresh API 経由で例外なく走り、FastChem4 first condensed
Newton step の amount 比較は `max_density + current` と同等かやや良い。v1.3 以降ではこの mode を
既定 mode として採用する。

support が非空の場合、restricted solver/lifecycle/fallback を既存の明示 support path で実行し、
結果 gas state から inactive condensate driving を再評価する。正の inactive support が残れば
`max_support_add_per_round` の範囲で solver support を追加し、`max_support_outer_iterations`
まで繰り返す。`max_support_add_per_round=None` の場合、残りの positive 候補を round 内で
まとめて追加できる。
seed amount は `seed_fraction <= 1e-3`、`max_seed_amount <= 1e-3` の安全 envelope で作る。

ユーザーが `support_indices` を明示した場合、このouter loopは使わず、指定supportを固定してHEAD route
v1.4のrestricted support pathを実行する。

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

### 7. v1.2 native seed fallback gate

HEAD route v1 では、restricted support solver が失敗した場合だけ native seed fallback に進んでいた。水凝縮 profile の中間層では、restricted solver は成功する一方で、primary continuation が `no_p_armijo_trial` で止まり、route selector が `support_boundary_construction_required_before_selector` を返すことがあった。この場合、良い runtime boundary はあるが standard gate へ進めず `not_converged` になっていた。

HEAD route v1.2 では、次の条件をすべて満たす場合にも `native_budget_seed_fallback_budget_tradeoff` を返す。

- lifecycle result が accepted / converged ではない。
- `enable_native_seed_fallback=True`。
- `metric_status`、`head_route_primary_summary`、`head_route_refresh_policy_summary` が外から注入されていない。
- finite な warm-start candidate がある。

この fallback は FastChem4 trace/public/runtime value を constructor input に使わない。返り値は `converged_with_caveat` で、acceptance tier は `tier_2_budget_tradeoff_experimental_only` である。diagnostics には、restricted solver が成功していたかどうかを `restricted_solver_success` として残す。

### 8. v1.3 support-free retry gates

HEAD route v1.3 では、support-free activity-driven outer loop 内に次の retry を追加する。いずれも明示 support rows や opt-in fixed-support experiments には自動適用しない。

- `center-gate retry`: restricted solver が成功し、primary continuation が `current_barrier_not_centered` で止まった場合だけ、center tolerance を `1.0e11` に緩めて lifecycle を一度再評価する。
- `residual-worsening retry`: primary continuation が `no_p_armijo_trial` で止まり、p-Armijo/filter candidate が residual nonworsening guard で止まっている場合だけ、`residual_worsening_tolerance=2.0e-2` で lifecycle を一度再評価する。必要なら center-gate retry と連結する。
- `soft-restoration retry`: residual-worsening retry などの通常 retry 後も lifecycle が accepted しない場合だけ、既存の native soft restoration fallback を有効化し、center tolerance `1.0e11` と組み合わせて lifecycle を一度再評価する。restoration は budget、total density、amount-weighted gas、amount-weighted condensate の component weights で guarded に選ぶ。
- `Ipopt-style persistent h-type retry`: soft-restoration retry 後も lifecycle が accepted しない場合だけ、Ipopt の filter line-search に対応する persistent h-type feasibility step を一度再評価する。theta は budget、total density、amount-weighted gas、amount-weighted condensate、complementarity の component weights から作り、budget と total density を protected components として守る。これは objective descent ではなく constraint violation の改善を採用する retry である。
- `support-cap retry`: 通常の support-free path が `native_budget_seed_fallback_budget_tradeoff` になり、caller が `max_positive_support_count` を明示しておらず、選ばれた support が retry cap を超える場合だけ、fresh API path を `support_cap_retry_counts=(34, 48, 80, 128)` の小さい cap から順に再実行する。retry が non-fallback route を返した場合だけ採用する。旧互換として `support_cap_retry_counts=None` の場合は `support_cap_retry_count` の単一 cap を使う。
- `support-growth staging retry`: 通常 path が fallback になり、caller が `max_support_add_per_round` を明示していない場合だけ、support を一括投入せず `support_growth_staging_retry_add_per_rounds=(64, 32, 16, 8)` の順に段階投入して fresh API path を再実行する。v1.9 では support-cap retry に採用可能候補があっても staged retry を評価し、両 retry family の non-fallback candidates から support-closure score が最良の候補を採用する。
- `scalar step-control retry`: 通常 path が fallback になり、caller が support count を明示していない場合だけ、primary step-control policy を `scalar_fraction_to_boundary` にして fresh API path を一度再実行する。v1.10 ではこの retry も support-cap / staged support-growth retry と同じ support-closure score で比較する。
- `support-growth warm-start amount floor`: support-free outer loop が前 round の solver/result condensate amount を次 round の warm-start seed として引き継ぐとき、0 以下または非有限の amount は `min_seed_amount` に戻す。これにより solver 側で 0 に落ちた support species が warm-start finite check を全候補失敗にすることを防ぐ。

これらの retry は FastChem4 trace/public/runtime value を constructor input に使わない。採用された retry は diagnostics に `head_route_center_gate_retry`、`head_route_residual_worsening_retry`、`head_route_soft_restoration_retry`、`head_route_ipopt_h_type_retry`、`support_cap_retry`、`support_growth_staging_retry`、または `scalar_step_control_retry` として記録する。v1.10 の support-cap / staged support-growth / scalar step-control selection では、横断 selection summary を `support_closure_retry_selection` にも記録する。

### 9. standard gate and result

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

## support-free profile と fixed-support rows の現状

HEAD route v1.5 の本来の default API regression は、`support_indices` を渡さずに
public `condensate_equilibrium()` を呼ぶ support-free 経路である。この経路では
activity-driven support selection、`max_density` seed、solver-output-driven support growth が
有効になり、必要な場合だけ v1.3 support-free retry gates、v1.5 support-closure retry gate、v1.4 full-budget gate / budget-correction
retry が発火する。

10 curated case families の中間層では、results artifact に依存せず fresh API から次の挙動を確認する。

| group | rows | status |
|---|---:|---|
| empty gas-only boundary | 1 | `converged` |
| primary promoted route accepted by full-budget gate | 9 | `converged` / tier 1 |
| primary promoted route rejected by full-budget gate | 0 | なし |
| primary promoted route not accepted by current PD-IPM surface | 0 | なし |
| native seed fallback | 0 | なし |

10 curated support-select demo families の全 profile layer について、v1.18 固定時点の fresh API score は `99 / 99 converged` である。v1.17 までに graphite layers、near-boundary layer、water layer 7 などの support-free blocker は戻り、v1.18 では water low-temperature support-boundary surface、explicit support `complex_heavy...` layer 4、explicit empty support highT gas-only layers 14/16 も修復した。

v1.18 の water 修復では、`native_budget_seed_fallback_budget_tradeoff` が restricted solver payload を持たない場合でも、PD-IPM lifecycle の `continuation_report.final_state` が有限なら、その final state から inactive-driving support growth を続行する。public fallback state は seed fallback と lifecycle final-state fallback を full-budget gate で比較し、accepted seed は保持し、seed も gate failed の場合だけ改善する final state を採用する。さらに native fallback の full-budget gate 直前に joint gas+active-condensate feasibility restoration を入れる。これにより water family は `9 / 9 converged` に戻った。

v1.18 の final cleanup では、fixed explicit support rows も public API 経路として整理した。`complex_heavy_element_or_boron_titanium_zirconium_case` layer 4 は、caller-provided support が Ti 系 4 種に狭すぎたため full-budget gate を落としていた。これは FastChem4 replay ではなく、ExoGibbs-native inactive driving から support を広げる `explicit_support_closure_retry` で tier 1 に戻す。`solar_highT_no_condensate_gas_regression` layers 14/16 は凝縮候補が温度有効ではない true gas-only rows であり、問題は explicit empty support path に support-free path と同じ `empty_support_strict_gas_retry` が配線されていなかったことだった。explicit empty support payload でも strict gas retry を使うことで full-budget gate を通過する。

一方、14 representative rows は v1.1 由来の fixed-support regression である。これらは
`support_indices` と `support_amounts_init` を明示して restricted-support path を固定するため、
v1.5 の support-free default support selection を検証するものではない。現在の 14 rows は
explicit support path と full-budget gate の回帰であり、accepted rows と full-budget gate reject rows
の両方を含む。

10 curated case families の support-free 中間層との関係は以下である。

| family | midlayer behavior |
|---|---|
| `solar_highT_no_condensate_gas_regression` | 凝縮候補なし。empty gas-only boundary。 |
| `solar_silicate_first_condensation` | center-gate retry で primary promoted route に到達し、v1.4 full-budget gate を通過。 |
| `solar_water_condensation` | solver-output-driven support growth で primary promoted route に到達し、v1.4 full-budget gate を通過。 |
| `solar_metal_sulfide_or_Fe_Ni_S_region` | primary promoted route に到達し、v1.4 full-budget gate を通過。 |
| `carbon_rich_graphite_window` | stable L2 residual norm と capacity cap により primary promoted route に到達し、v1.4 full-budget gate を通過。 |
| `carbon_rich_CaS_MgS_AlN_window` | primary promoted route に到達し、v1.4 full-budget gate を通過。 |
| `SiO_s_condensate_window` | support-cap retry で primary promoted route に到達し、v1.4 relative joint budget-correction retry 後に full-budget gate を通過。 |
| `lowT_strong_condensation_budget_stress` | residual-worsening retry と center-gate retry で primary promoted route に到達し、v1.4 full-budget gate を通過。 |
| `near_phase_boundary_support_sensitivity` | primary promoted route に到達し、v1.4 full-budget gate を通過。 |
| `complex_heavy_element_or_boron_titanium_zirconium_case` | support-free では primary promoted route に到達し、fixed explicit support layer 4 は `explicit_support_closure_retry` で full-budget gate を通過。 |

## 成功と caveat の意味

`converged` は、HEAD route standard gate 上で tight residual tier に入ったという意味である。

`converged_with_caveat` は、HEAD route が runtime result を返せるが、次のような注意が残るという意味である。

- v1.2 native seed fallback: lifecycle selector が accepted しないため、native gas equilibrium または有限な warm-start boundary と condensate seed を caveat 付きで返す。
- T500: strict budget closure ではなく budget tradeoff を許している。
- T700 metal/sulfide: amount-weighted gas では良いが raw gas residual frame に注意がある。

したがって、14 fixed-support rows は「同じ品質で完全成功」ではない。正確には、現在の
fresh API では tier 1、caveat、empty gas-only boundary を分けて見る。support-free 10-family 中間層では、
9 rows が tight、0 row が caveat、1 row が empty gas-only boundary、0 row が current PD-IPM surface 上の
`not_converged`、0 row が full-budget gate reject である。全 99-layer curated score では
`89 converged / 10 converged_with_caveat / 0 not_converged` である。

## 守るべき境界

HEAD route は凝縮あり初版標準経路として進めてよいが、以下は守る。

- gas-only `equilibrium()` の挙動を変えない。
- 既存の production result fields の意味を変えない。
- gas-only API の defaults を変更しない。凝縮あり API の fixed default は HEAD route v1.18 として扱う。
- 現在の凝縮あり API の fixed default は `result.head_route_version == "v1.18"` として公開する。
- v1.10 production-safe surface は比較 baseline として残すが、現在の本線には戻さない。
- FastChem4 public/runtime/trace values を constructor input にしない。
- FastChem4 exact branch replay を acceptance target にしない。
- case、species、element を落として成功扱いにしない。
- caveat tier を tight success と混同しない。

## 現在の未解決

次に整理すべき課題は以下である。

1. support-closure retry gate の tolerance `5.0e2` と、support-cap retry sequence `(34, 48, 80, 128)` / support-growth staging retry sequence `(64, 32, 16, 8)` が curated full-profile 以外の broad grid でも妥当か監査する。
2. `converged_with_caveat` 10 rows を、PD-IPM/filter/restoration の中で tier 1 に寄せられるか調べる。
3. `condensate_equilibrium_profile()` を profile/layer 計算へ接続する。
4. docs と examples で、HEAD route の public API 使用例を整備する。

## 更新ルール

HEAD route を変更した場合は、少なくとも次を更新する。

1. `head_route.md`
2. `curated_case_family_scorecard_ja.md`
3. 関連する `src/exogibbs/condensates/*`
4. 関連する API tests
5. 必要なら docs/examples

HEAD route の変更は、既存 accepted rows を壊さないこと、または壊した場合に理由と新しい acceptance policy を明示することを条件にする。
