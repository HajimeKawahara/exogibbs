# Condensate legacy cleanup refactoring strategy

## 状態

| 項目 | 値 |
|---|---|
| 監査日 | 2026-07-25 |
| worktree | `/home/kawahara/exogibbs-condensate-cleanup` |
| branch | `cleanup/condensate-legacy` |
| 監査対象 base | `c09d327` |
| production promotion | `e68902e` は base に含まれる |
| 現在の段階 | reachability / compatibility / evidence の監査のみ |
| この段階でのコード削除 | なし |

この文書は、fixed-support solver v2 の production 昇格後に
`src/exogibbs/optimize/pipm_rgie_cond.py` と
`src/exogibbs/optimize/minimize_cond.py` を分解・整理するための判断記録である。

先に結論を示す。

1. public default の `head_v2` route から、2つの巨大 legacy module への
   import/call 経路はない。
2. ただし両 module には、explicit `head_v1`、direct optimize API、
   experimental profile API、test、archive replay のいずれかから到達する
   コードが残っている。
3. 最初の実装は一括削除ではなく、import provenance と互換性 contract の
   固定にする。
4. 最初の構造変更は compatibility facade を維持した move-only extraction
   にする。
5. 削除は、direct-import compatibility の扱いを決めた後、
   repository 内 zero-call-site carrier から段階的に行う。

## 1. authoritative contract

今回の判断では、次の文書を上から順に優先する。

1. `documents/fixed_support_solver_v2_production_migration.md`
2. `documents/fixed_support_solver_v2_validation.md`
3. `documents/fixed_support_solver_v2_design.md`、特に sections 18, 19, 22
4. `documents/condensate_profile.rst`
5. `benchmarks/fixed_support_v2/README.md`

cleanup 中も次の production contract を維持する。

- default route は `head_v2`
- route version は `v2.0`
- production preset は `validated_2026_07`
- support lifecycle は fixed-support solver の外側
- `head_v2` から `head_v1` への automatic fallback は存在しない
- `head_v1` は明示的な backward-compatibility route
- rollback は solver fallback ではなく release artifact 単位

`documents/fixed_support_solver_v2_validation.md:1-204` は promotion 前の
frozen validation record である。そこにある「未昇格」「次に production
budget を確認する」といった記述を現在状態として扱わない。
現在状態は同文書の production follow-up `:206-217` と production
migration 文書で判断する。

migration/validation 文書に記録された source hash と artifact hash は
historical evidence である。mutable source の refactoring により現在の
source hash が変わることは許容されるが、記録済みの過去の値は書き換えない。

## 2. 監査方法と限界

### 2.1 reachability root

production runtime、compatibility、historical evidence を混同しないため、
次の root を分けて調べた。

| root | 意味 |
|---|---|
| P0 | default options の public `condensate_equilibrium` |
| P1 | default options の public `condensate_equilibrium_profile` |
| C0 | explicit `route="head_v1"` の上記 public API |
| C1 | `optimize/minimize_cond.py` の direct API |
| C2 | `api/condensate_equilibrium.py` の experimental prepared-plan API |
| A0 | frozen validation data と archive benchmark runner |
| T0 | unit/end-to-end test と fixture |
| D0 | Sphinx toctree、tutorial、design record、生成図 |

### 2.2 確認した evidence

- Python import と function-local import
- AST による top-level symbol と line span
- `rg` による repository-wide symbol/call-site 検索
- `__all__`、lazy export、direct module import
- route/method/support mode/environment-variable literal
- unit/end-to-end test
- benchmark runner、dynamic `importlib` load、C-shell wrapper
- active Sphinx toctree と tutorial import
- `MANIFEST.in` と package-data 設定
- frozen artifact 宣言と SHA-256
- selected git history と symbol 導入履歴
- この worktree の `src` を優先した fresh-process import smoke

### 2.3 監査の限界

この文書の「未参照」は、監査対象 repository 内で未参照という意味である。
third-party user の direct import までは証明できない。特に次が重要である。

- `minimize_cond.py` は `__all__` で36名を明示的に公開している。
- `pipm_rgie_cond.py` には `__all__` がなく、非 underscore 名は
  star import の対象になり得る。
- caller は historical module path から直接 import できる。
- caller-supplied callback 内の動的 import は静的監査では観測できない。

したがって repository-internal zero-call-site は強い削除候補ではあるが、
それだけで public compatibility を保証したとはみなさない。

## 3. reachability 分類

| class | 現在の内容 | 方針 |
|---|---|---|
| 1. production `head_v2` reachable | API route dispatch、v2 policy/profile、`fixed_support_v2/*` | 保護し、regression gate を置く |
| 2. explicit `head_v1` compatibility | restricted layer、legacy profile batch、old RGIE mode、warm-start/depleted-gas helper | compatibility boundary の内側へ隔離 |
| 3. frozen validation/archive evidence | frozen v1 summary、matrix、historical report、archive runner | byte/path を保存し replay dependency を隔離 |
| 4. runtime-unreachable/obsolete experiment | zero-call-site carrier、default-off/test-only adapter | compatibility 確認後に wave 単位で削除 |
| 5. historical documentation only | superseded audit/ablation/design input/旧 route 図 | historical と明示または archive |

### 3.1 production `head_v2`

one-layer default route は次の経路を通る。

```text
condensate_equilibrium
  -> CondensateEquilibriumOptions(route="head_v2")
  -> _run_head_v2_profile
  -> fixed_support_v2_production_policy
  -> fixed_support_v2_profile.run_prepared_profile_v2
  -> optimize/fixed_support_v2/*
```

profile default route も同じ `_run_head_v2_profile` に入る。

根拠:

- `src/exogibbs/api/condensate_equilibrium.py:37`:
  `CondensateRoute = Literal["head_v1", "head_v2"]`
- `:90-92`: v2 route name/version
- `:138-139`: default `head_v2` / `validated_2026_07`
- `:5672-5686`: one-layer public API の v2 early dispatch
- `:10779-10798`: profile public API の v2 dispatch
- `:5163-5363`: v2 policy/profile の import と実行
- `_head_v2_initial_state` `:4969-5045`: v1 solver に依存しない初期状態
- `src/exogibbs/optimize/fixed_support_v2_profile.py`: `fixed_support_v2`
  implementation のみを使用

この worktree を `PYTHONPATH` の先頭に固定した fresh process では、
次を確認した。

```text
exogibbs.__file__ = .../exogibbs-condensate-cleanup/src/exogibbs/__init__.py
CondensateEquilibriumOptions().route = head_v2
exogibbs.optimize.minimize_cond in sys.modules = False
exogibbs.optimize.pipm_rgie_cond in sys.modules = False
```

`benchmarks/fixed_support_v2/production_profile_gpu_gate.py:54-60` の
production source set は API、v2 policy、v2 profile implementation を含むが、
`minimize_cond.py` と `pipm_rgie_cond.py` は含まない。

route dispatch は validated literal であり、v2 failure から v1 を呼ぶ branch はない。

#### experimental API の例外

「default options なら legacy に到達しない」という結論は、上記2つの
public route function に対するものである。export 済み experimental helper
すべてには適用できない。

`prepare_experimental_profile_fixed_support_batch_plan` は
`options.route` を route gate とせず legacy bucket を構築する。
`run_experimental_profile_fixed_support_v2_batch_plan` も prepare 段階では
legacy bucket representation を使い、execution のみ v2 runner へ渡す。
これらは default options を受け取っても compatibility/experimental class
として扱う。

### 3.2 explicit `head_v1`

| entry | call/import path | giant module への影響 |
|---|---|---|
| one-layer, default v1 coupling | API local import `condensate_equilibrium.py:5769-5772` → `solve_restricted_support_condensate_layer`; dispatch `minimize_cond.py:9331-9341` | `minimize_cond` を call、`pipm_rgie_cond` を eager import |
| one-layer, alternative old coupling | `_minimize_gibbs_cond_legacy` `minimize_cond.py:1770-1947` | `pipm_rgie_cond` の raw solver を call |
| profile `auto` + complete fixed-support payload | API legacy batch path `condensate_equilibrium.py:10816-10840` | `minimize_cond.py:2021-8854` の v1 batch を call |
| direct optimize API | `minimize_cond.__all__` の names | 選択した call に応じ legacy/semismooth/diagnostic へ到達 |
| public experimental plan API | API module exports `condensate_equilibrium.py:10912-10947` | legacy bucket type/preparation helper を使用 |

explicit `head_v1` の default restricted coupling は
`pdipm_rgie_v11_activity_correction` であり、数値 solver 本体は別の小さい
`pdipm_rgie_cond.py` にある。しかし `minimize_cond.py:68-81` が raw
solver/diagnostics を `pipm_rgie_cond.py` から一括 import するため、
通常の v1 adapter を import するだけで旧 module 17,121行全体を load する。

default v1 layer が通常経路で使うのは主に次の小範囲である。

- `_recompute_pi_for_residual`
- `summarize_rgie_inactive_driving`
- low-level no-seed input の
  `build_rgie_condensate_init_from_policy`

この eager import が最初に切るべき dependency seam である。
standard v1 compatibility route は historical raw diagnostics を load
する必要がない。

### 3.3 frozen validation/archive evidence

frozen baseline の宣言元は
`benchmarks/fixed_support_v2/fixed_support_v2_gpu_matrix.json` である。
宣言値と実 file の SHA-256 が一致することを確認した。

| path | SHA-256 |
|---|---|
| `benchmarks/fixed_support_v2/data/frozen_v1_baseline/selected_case_summary.json` | `9e3b28b7c856ca8bb5bbf4aebdbe216e1392294ff611f5350477d07e0484f299` |
| `benchmarks/fixed_support_v2/data/frozen_v1_baseline/selected_case_summary.md` | `ae41c6ec1edb2e6d3957045d53b2081fcc8a97d0c0e74d044b20f222ee37ddf1` |

この path と byte content は保存する。matrix 自身が、これらの hash は
historical evidence を freeze するものであり、mutable v1 source の hash
ではないと明記している。

`fixed_support_v2_unbiased_gpu_experiment.py` は archive comparison のため、
現在も v1 prepared-plan implementation を import/execute する。
同 runner の current-source integrity report には `minimize_cond.py` が入るが、
その実行時生成 hash は frozen historical source contract ではない。

archive dependency はまとまりとして扱う。
`benchmarks/fixed_support_v2/support_atlas_sweep.py` は
`importlib.util.spec_from_file_location` で
`benchmarks/fastchem4/fastchem4_vmap_cold_rescue_compare.py` を動的 load する。
通常の import 検索では見えにくいため、後者だけを削除すると replay を壊す。

### 3.4 historical documentation

次は設計・調査記録として有用だが、現 production behavior の仕様ではない。

- `documents/amount_space_restoration_design.md`
- `documents/audit_condensates.md`
- `documents/ipm_audit.md`
- `documents/fixed_support_pdipm_static_audit.md`
- `documents/pdipm_math_contract_audit.md`
- `documents/ipopt_exogibbs_mathnote.md`
- `documents/solver_component_ablation.md`

historical label を付けて保持するか、相互参照を壊さず archive bundle へ移す。
過去の experiment を記載しているという理由だけで runtime code を残さない。
一方で、過去の結論を新 implementation の説明へ無言で書き換えない。

## 4. giant file の構成

### 4.1 `pipm_rgie_cond.py`

exact replay deletion wave 後は5,745行。

| lines | 主な役割 | reachability/方針 |
|---|---|---|
| 39-1358 | carrier、inventory/coupling/support/direction helper | 主に direct/test/experimental |
| 1359-2054 | residual、lambda、gas-limiter diagnostic | alternate v1/direct diagnostic |
| 2057-3569 | backend/direction/trajectory comparison | direct/test/diagnostic |
| 3572-5204 | legacy update loop/raw solver core | alternate v1/direct API |
| 5207-5413 | raw solver wrappers/diagnostics | alternate v1/direct API |
| 5416-5745 | trace API | direct/test/diagnostic |

旧 `_build_kl_gas_phase_calculate_replay_results` と
`_build_reduced_solver_exact_input_bundle` を中心とした default-off exact replay
cluster は削除済みである。shared reduced-system primitive は
`legacy_condensate/rgie_reduced_system.py` にあり、旧 import pathから再exportする。

### 4.2 `minimize_cond.py`

exact replay deletion wave 後は10,214行。

| lines | 主な役割 | reachability/方針 |
|---|---|---|
| 122-1424 | legacy type/config、profile setup、gas/RGIE helper | public/cross-module compatibility |
| 1427-8069 | v1 fixed-support batch/continuation/bucket API | explicit v1 profile/archive replay |
| 8072-8856 | standard restricted v1 layer/metrics | explicit v1 compatibility |
| 8859-9392 | semismooth/active-set experiment | explicit support mode/direct export |
| 9395-10214 | compatibility profile API/trace/`__all__` | public/direct compatibility |

v1 fixed-support batch は約6,643行、module 全体の約65%である。
core は60引数を持ち、result は111個の positional value に unpack される。
これは v2 が focused typed component で置き換えた historical all-in-one kernel
の形そのものである。

production v2 dependency ではないが、explicit v1 profile と archive runner
から到達するため、現時点では削除不可である。

## 5. public/configuration surface

### 5.1 export

- `src/exogibbs/api/__init__.py` は condensate API を lazy export する。
- `src/exogibbs/api/condensate_equilibrium.py:10912-10947` は production API
  と legacy experimental prepared-plan API を同じ `__all__` に置く。
- `src/exogibbs/optimize/minimize_cond.py:10178-10214` は35名を公開する。
- `src/exogibbs/optimize/pipm_rgie_cond.py` には `__all__` がない。
- `src/exogibbs/optimize/__init__.py` は空で、追加の package-level contract
  を定義しない。

`minimize_cond.py` は自身を backward-compatible import path と説明している。
別途 public deprecation/removal を決めるまでは、この historical path を
thin facade として維持する。

### 5.2 lazy export collision

`src/exogibbs/api/__init__.py` には既存の import-order-dependent collision がある。
package は function `condensate_equilibrium` を lazy export する意図だが、
同名 submodule から別名を import すると、その submodule が package attribute
として設定される。

fresh process で次を実行すると、

```python
from exogibbs.api import CondensateEquilibriumOptions, condensate_equilibrium
```

現在の `condensate_equilibrium` は意図された function ではなく module になる。
`exogibbs.api.condensate_equilibrium` submodule からの direct import は function
を返す。

これは今回作った問題ではなく、既存 contract の曖昧さである。
public type や lazy import を移動する前に import-order matrix test を追加し、
意図する package-level behavior を明示的に決める。
legacy extraction の副作用として暗黙に fix したり、逆に bug を無条件に contract
として固定したりしない。

### 5.3 options と environment variable

`CondensateEquilibriumOptions` は production v2 field と多数の v1/experimental
field を混在させている。dataclass の分割は public API change になるため、
最初は facade と validation behavior を維持し、内部で route ごとの typed
configuration へ必要 field だけを変換する。

legacy batch は47個の `EXOGIBBS_FIXED_SUPPORT_BATCH_*` environment variable
を読む。production v2 は1つも読まず、environment variable を kernel config
へ変換してはならない。

監査結果:

- 24個は parser 外の test/benchmark/doc/example 参照なし
- 14個は test のみ
- archive runner が使うのは小さな subset

まず parser を明示的な v1/archive config boundary へ移す。
その後 archive runner を typed config に置換し、未使用 variable を別 change
で削除する。

## 6. test/doc/benchmark/package data

### 6.1 test

現在の test は次の異なる責務を混在させている。

- production v2 route/policy/profile contract
- explicit v1 compatibility
- archived v1/v2 validation integrity
- low-level experimental diagnostics

削除前に test をこの責務で分類する。

- `tests/unittests/condensates/fixed_support_v2_policy_test.py`
- `tests/unittests/optimize/fixed_support_v2_*_test.py`
- `tests/unittests/benchmarks/fixed_support_v2_production_profile_test.py`

上記は production v2 evidence として保持する。
explicit v1 test は route choice を必ず明示する。
`minimize_cond_diagnostics_test.py` と PIPM/RGIE audit test は production
acceptance ではなく experimental evidence として扱う。
archive hash/integrity test は mutable legacy solver test から独立させる。

`tests/endtoend/curated_cases` は unit-test CI command の対象外である。
README/expected route は v1.1-v1.5 を前提とする一方、複数の call は
`route="head_v1"` を指定せず、現在は v2 を実行する。

次のどちらかを選ぶまで deletion evidence として使わない。

1. explicit `head_v1` compatibility suite に固定する。
2. v2 expectation へ書き換える。

### 6.2 active documentation

live Sphinx surface には次の不整合がある。

- `documents/index.rst` は `condensate_profile.rst` を含む。
- 同ページの route image は旧 v1 route のまま。
- `documents/index.rst` は `ipynb/h2ocond.rst` も含む。
- H2O tutorial は `pipm_rgie_cond.minimize_gibbs_cond_core` を直接 call する。
- `documents/presets/fastchem4.rst` は default `auto` を legacy
  batch/fallback または hot scan と説明している。
- `documents/index.rst` は存在しない `exogibbs/index.rst` を参照する。
- `documents/exogibbs/exogibbs.rst` は存在しない generated RST を参照する。

production route 図と prose を先に更新する。
H2O tutorial は public v2 API へ変更するか active toctree から外してから、
raw core dependency を削除する。

`documents/conf.py` は `os.path.abspath("~/exogibbs")` を挿入する。
`abspath` は `~` を展開せず、この worktree の `src` も明示しない。
documentation verification も benchmark と同様に old-worktree import の影響を
受け得る。

### 6.3 example

curated support-selection example は default v1 route と説明しつつ、常に
explicit v1 option を渡しているわけではない。各 example を次のどちらかへ分類する。

- current public v2 example
- 明示的に命名した explicit v1 compatibility example

説明と実行 route が異なる example は残さない。

### 6.4 benchmark runner provenance

fixed-support v2 の全 C-shell wrapper は repository root へ移動後、
bare `python` を実行する。この worktree の `src` を `PYTHONPATH` 先頭へ
固定していない。

さらに Python runner は `ROOT` を計算する前に `exogibbs` を import する。
現在の system `PYTHONPATH` には旧 worktree が含まれるため、異なる source の
結果をこの branch の結果として報告する危険がある。

これは refactoring 前に解消すべき gate である。

- wrapper は resolved repository-relative `src` を Python 起動前に設定する。
- Python entry point は `exogibbs.__file__` を検証・記録する。
- import 済み package が resolved repository root 外なら fail-fast する。

`benchmarks/fastchem4` の旧 C-shell script の一部は
`/home/kawahara/exogibbs` を hard-code し、`volatiles_artifacts` へ出力する。
この clean worktree では実行せず、historical として archive するか書き換える。
旧 worktree から artifact をコピーしない。

### 6.5 package data

`MANIFEST.in` は chemistry dataset を含むが、legacy solver source や frozen
benchmark artifact は含まない。frozen baseline は repository archive であり
wheel runtime data ではないため、manifest に追加する必要はない。

FastChem4 runtime chemistry data は production-reachable であり削除対象外とする。
condensate solver 以外の未参照 dataset は、scope を広げず別の data-cleanup
audit で扱う。

## 7. deletion candidate register

この監査で登録した候補と、その後の実施状況を以下に示す。

### 7.1 Wave A: 高信頼 repository-internal orphan

repository-wide search で、definition、自身の helper 内 construction、
docstring 以外の参照が見つからなかった。

| file/lines | symbol | compatibility 上の注意 |
|---|---|---|
| `pipm_rgie_cond.py:39-163` | `FixedHighGainSourceStateCarrierRow`, `build_fixed_high_gain_source_state_lifecycle_carrier` | explicit export ではないが module に `__all__` がない |
| `pipm_rgie_cond.py:258-280` | `build_inventory_capped_rgie_startup_ln_mk` | public-looking direct-import name |
| `pipm_rgie_cond.py:8175-8254` | `build_fastchem_exact_total_element_density_convention_carrier` | docstring 自身が production call なしと明記 |
| `pipm_rgie_cond.py:8257-8332` | `build_condensate_density_budget_cap_total_element_density_carrier` | default-off diagnostic carrier |
| `minimize_cond.py:226-320` | `emit_correctvalues_condensation_diagnostic_record` | `minimize_cond.__all__` 外 |
| `minimize_cond.py:323-544` | `build_case_keyed_correctvalues_condensation_source_state_carrier` | `minimize_cond.__all__` 外 |
| `minimize_cond.py:547-764` | `build_case_keyed_reduced_slot_solve_state_source_carrier` | `minimize_cond.__all__` 外 |

実施状況:

- `minimize_cond.py` の3 carrier は、最初の deletion wave で削除済み。
- exact replay cluster 内の2つの density carrier は、direct-import policy を
  記録した exact replay deletion wave で削除済み。
- `pipm_rgie_cond.py` のその他の候補は追加確認まで保留中。
- 上表の line number は監査対象 base `c09d327` での位置を示す。

Wave A 削除前の条件:

1. non-exported legacy diagnostic の direct import は supported compatibility
   contract ではない、と判断記録を残す。
2. public-looking module name を public と扱う policy なら deprecation/release
   note を追加する。
3. 削除対象 carrier 専用の test/reference だけを同時に整理する。
4. historical conclusion は runtime code ではなく document または git history
   で保存する。

### 7.2 Wave B: test-only/default-off adapter

強い cleanup 候補だが、test または diagnostic entry point が残る。

- `_solve_pdipm_rgie_v11_activity_correction_profile_buckets`
  (`minimize_cond.py:8666-8854`): source caller は0、unit test からのみ直接 call
- raw update/direction/lambda-cap experiment
  (`pipm_rgie_cond.py:1102-1276`)
- support audit helper (`pipm_rgie_cond.py:1432-1501`)
- exact replay/input-bundle cluster (`pipm_rgie_cond.py:1561-12280`)
- selected direction/trajectory diagnostic/trace

実施状況:

- private one-shot profile adapter
  `_solve_pdipm_rgie_v11_activity_correction_profile_buckets` は削除済み。
- 同じ3-layer/budget contract を検証する prepared `prepare/run` pair と test
  は保持している。
- exact replay/input-bundle cluster は削除済み。
- raw update/support/selected direction diagnostic 候補は未着手。

exact replay/input-bundle cluster は Phase 1 の lazy boundary と shared RGIE
primitive extraction 完了後に再監査した。再監査結果は次の通りである。

- production `head_v2` と default explicit `head_v1` から未到達
- 通常の raw v1 solve では activation flag/context が無効
- repository 内の有効化 caller は diagnostic unit test のみ
- benchmark、example、active document、archive runner に executable consumer なし
- frozen validation artifact の path/hash は cluster source に依存しない

したがって、historical diagnostics module への移動は行わず deletion とする。
historical evidence owner はこの strategy、既存 audit/design document、git history、
および frozen report/artifact であり、10,000行超の executable reconstruction
code 自体は evidence として保持しない。

この判断は `pipm_rgie_cond.py` の暗黙 star-import surface を縮小する。ただし、
private exact-bundle builder と repository consumer を持たない density carrier は
supported compatibility API ではないと明示的に扱う。raw v1 solver、reduced solve、
direction/residual diagnostic、および explicit `head_v1` route は保持する。

### 7.3 Wave C: public compatibility 判断が必要

repository call count が少ないだけでは削除しない。

- `compute_sk_feasible_epsilon_floor`: `minimize_cond.__all__` にある。
- semismooth aliases/diagnostics: export 済みで
  `smoothed_semismooth_outer` から到達する。
- `minimize_cond.__all__` の raw solver/trace name
- direct-import policy 決定前の `pipm_rgie_cond.py` 非 underscore function
- API module が export する experimental prepared-plan function

通常の deprecation window、または明示した breaking-release decision を使う。

### 7.4 現段階で削除しないもの

- `fixed_support_v2` production implementation
- `solve_restricted_support_condensate_layer`
- default `head_v1` v11 activity-correction adapter
- explicit v1 profile と archive replay が使う間の v1 profile batch
- active tutorial/alternate v1 mode が使う間の raw RGIE solver
- frozen baseline file、宣言 path、記録 hash
- `fastchem4_vmap_cold_rescue_compare.py` などの dynamic archive dependency
- superseded という理由だけでの historical design/validation record

## 8. target module boundary

具体名は実装時に調整できるが、dependency は次の形を目標にする。

```text
api/condensate_equilibrium.py
  |-- production head_v2 --> optimize/fixed_support_v2_profile.py
  |                         optimize/fixed_support_v2/*
  |
  `-- explicit head_v1 ---> optimize/legacy_condensate/head_v1_layer.py

optimize/legacy_condensate/
  types.py
  rgie_reduced_system.py
  rgie_solver.py
  rgie_diagnostics.py
  semismooth.py

optimize/fixed_support_v1/
  config.py
  batch.py
  profile.py

optimize/minimize_cond.py       # backward-compatible facade
optimize/pipm_rgie_cond.py      # backward-compatible facade
```

dependency rule:

1. `fixed_support_v2` は `legacy_condensate`、`fixed_support_v1`、
   `minimize_cond`、`pipm_rgie_cond` を import しない。
2. public API は route validation 後に route implementation を lazy import する。
3. default explicit v1 layer は小さな RGIE helper を import できるが、
   historical diagnostics/v1 profile batch を import しない。
4. v1 profile batch は v1 helper に依存できるが、production v2 は legacy
   bucket representation に依存しない。
5. archive runner は production facade を偶然経由せず explicit v1/archive
   adapter を import する。
6. historical diagnostic module は solver primitive に依存できるが、
   solver primitive から historical reconstruction code へ逆依存しない。
7. compatibility policy が removal を許可するまで、`minimize_cond.py` と
   `pipm_rgie_cond.py` は historical import path/name/signature を維持する。

既存 top-level `fixed_support_*` module も v1 batch implementation の一部だが、
最初から全て移すと change が大きすぎる。まず giant batch を package boundary
の内側へ隔離し、import/numerical parity の確立後に helper module を統合する。

## 9. execution plan

### Phase 0: trustworthy gate の確立

これを次の最初の実装とする。

1. benchmark/Sphinx import provenance を修正し、旧 worktree を使えなくする。
2. default `head_v2` が `minimize_cond`/`pipm_rgie_cond` を import しないことを
   fresh subprocess で確認する test を追加する。
3. route name/version/preset/no-v1-fallback を assertion 化する。
4. standard one-layer と complete-payload profile の explicit `head_v1`
   contract fixture を作る。
5. API lazy export の import-order matrix を追加し、意図する
   `condensate_equilibrium` package attribute behavior を決める。
6. supported `minimize_cond.__all__` と private diagnostics を分けて記録する。
7. test/example を production v2、explicit v1、archive、historical diagnostic
   に分類する。

Phase 0 では numerical algorithm を変更しない。

### Phase 1: eager giant-module edge の除去

1. default v1 layer に必要な small reduced-system/residual helper を抽出する。
2. `minimize_cond.py` の raw solver/diagnostic import を、実際の
   compatibility/diagnostic entry point 内の local import に変える。
3. historical module では compatibility re-export を維持する。
4. default explicit `head_v1` が historical diagnostics 全体を load しないこと、
   alternate mode は引き続き動くことを確認する。

この段階は functionality を削除せず、有効な dependency boundary を作る。

実施状況:

- module-level `minimize_cond -> pipm_rgie_cond` import は除去済み。
- historical raw alias は monkeypatch compatibility を保つ lazy proxy にした。
- plain `minimize_cond` import は `pipm_rgie_cond` を load しない。
- default explicit `head_v1` の実行時 helper call は、まだ
  `pipm_rgie_cond` を load する。small helper extraction は次の作業。

### Phase 2: move-only decomposition

1 change につき1つの coherent unit を移す。

1. shared legacy type
2. standard `head_v1` layer
3. v1 fixed-support batch/config/profile
4. raw RGIE solver
5. semismooth experiment
6. exact replay/input-bundle と diagnostic trace
   （exact replay deletion 完了、残る trace は別判断）

各 move で次を守る。

- old module を compatibility facade として残す。
- name/default/signature/result structure/dtype/exception behavior を維持する。
- solver tuning や tuple/schema redesign を同時に行わない。
- import cycle check と該当 numerical parity test を実行する。

60引数/111-result の v1 batch を typed internal structure で包む作業は、
focused module への分離後に行う。external facade は維持する。

### Phase 3: Wave A orphan の削除

direct-import policy を記録した後、Wave A のみを小さい change で削除する。
その carrier 専用の test/stale reference だけを整理し、module move と同じ
change に混ぜない。

### Phase 4: obsolete experiment の retire

Wave B/C の各 cluster を次のいずれかへ分類する。

- supported compatibility API
- archive-only executable evidence
- frozen document/artifact evidence
- deletion

選択後、code/test/doc/environment switch を coherent unit として整理する。
consumer を削除した test-only runtime adapter や orphan option を残さない。

### Phase 5: active doc/example/benchmark の修正

1. production route diagram を再生成する。
2. active tutorial を v2 public API へ変更する。
3. historical v1 visual/audit を明示的に archive する。
4. stale curated end-to-end test/example を分類する。
5. hard-coded old-worktree benchmark path を置換する。
6. frozen validation artifact は変更しない。

### Phase 6: optional `head_v1` sunset

`head_v1` removal は別の product/versioning decision とする。実行する場合:

1. deprecation を告知する。
2. experimental v1 API の export を停止する。
3. 必要な archive replay implementation を runtime package import から外す。
4. v1 implementation と compatibility facade を削除する。
5. historical artifact/migration record は保存する。
6. 移行中も v2-to-v1 fallback を追加しない。

## 10. verification gate

ExoGibbs を import する全 command で、この worktree を先頭に固定する。

```bash
env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -c \
  "import exogibbs; print(exogibbs.__file__)"
```

期待値:

```text
/home/kawahara/exogibbs-condensate-cleanup/src/exogibbs/__init__.py
```

### change 種別ごとの minimum gate

| change | minimum gate |
|---|---|
| documentation/audit のみ | import provenance、frozen hash、`git diff --check` |
| facade/import move | fresh-process import matrix、`__all__` contract、targeted v1/v2 test |
| v1 batch move | explicit v1 one-layer/profile、archive preflight、numerical parity |
| diagnostic deletion | zero-call-site search、direct-import policy、専用 test/doc cleanup |
| API/v2 source change | production route/policy/profile test、必要に応じ production-profile gate |
| refactor wave 完了時 | full `pytest tests/unittests` |

推奨 CPU check:

```bash
env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q \
  tests/unittests/condensates/fixed_support_v2_policy_test.py \
  tests/unittests/benchmarks/fixed_support_v2_production_profile_test.py

env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q \
  tests/unittests/api/condensate_equilibrium_test.py \
  tests/unittests/api/condensate_equilibrium_profile_test.py

env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q \
  tests/unittests/optimize/minimize_cond_api_test.py \
  tests/unittests/optimize/minimize_cond_diagnostics_test.py

env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest tests/unittests
```

documentation-only audit では GPU production validation の再実行は不要である。
public v2 route、v2 solver、lifecycle behavior、production policy/gate を変えた場合は、
change に比例した validation を行う。pure legacy move でも v2 import graph と
source behavior が不変であることは確認する。

### stop condition

次のいずれかが発生したら、その refactor wave を止める。

- default v2 が legacy giant module を import する。
- route/version/preset/lifecycle owner が変わる。
- v2 failure が v1 に到達する。
- current worktree 以外の package を import する。
- exported compatibility name が意図せず消える。
- archive runner の dynamic dependency が消える。
- frozen artifact の path/byte/recorded hash が変わる。
- active document の説明 route と実行 route が異なる。

## 11. 推奨する次の change

この監査後の最初の pull request は、小さい contract/provenance change とする。

1. benchmark/documentation の import root を worktree-relative にして fail-fast
   させる。
2. default-v2 legacy-non-import subprocess test を追加する。
3. explicit route/version/preset/no-fallback assertion を追加する。
4. API lazy-export import-order behavior を characterization する。
5. existing test を production v2、explicit v1、archive、historical diagnostic
   に分類する。

Phase 1 の move-only helper extraction と local import により
`minimize_cond -> pipm_rgie_cond` の eager diagnostic edge は切断済みである。
再監査で test-only/default-off と確定した10,340行の exact diagnostic builder、
専用 carrier/context plumbing/test の coherent deletion は完了した。
次の change は、5,745行に縮小した `pipm_rgie_cond.py` の残る
orphan/experimental diagnostic を symbol単位で再監査するか、raw v1 solverを
focused legacy moduleへ分離する。約6,643行の v1 batch は explicit v1
profile/archiveから到達するため、引き続き削除対象にしない。

## 12. 実施記録

### 2026-07-25: audit baseline

- commit: `72dedd1`
- production/v1/archive/historical surface を監査した。
- この strategy を削除前の判断基準として追加した。
- runtime code の変更・削除は行っていない。

### 2026-07-25: Phase 0 import provenance/contract gate

- commit: `acdb707`
- fixed-support v2 の全 C-shell wrapper で checkout 内 `src` を
  `PYTHONPATH` に固定した。
- production/archive/support-atlas runner で `exogibbs.__file__` を検証し、
  repository 外からの import を fail-fast にした。
- Sphinx configuration も repository-relative `src` を使うようにした。
- default `head_v2` が legacy module を importせず、v2 failure を v1 へ
  fallback せず伝播する fresh-process regression test を追加した。
- benchmark provenance test を追加した。

検証:

- targeted tests: `13 passed`
- full unit tests: `512 passed, 22 warnings`
- C-shell syntax check: 6 runners passed
- 3 Python runners: intentionally invalid `PYTHONPATH` からの `--help` passed
- Sphinx configuration: `py_compile` passed
- Sphinx runtime import: `.venv` に `sphinx_rtd_theme` がないため未実施

### 2026-07-25: first deletion wave

- commit: `08e8859`
- commit subject: `condensates: remove unused diagnostic carriers`
- `minimize_cond.py` から default-off/repository-internal zero-call-site の
  carrier 3群を削除した。
  - `emit_correctvalues_condensation_diagnostic_record`
  - `build_case_keyed_correctvalues_condensation_source_state_carrier`
  - `build_case_keyed_reduced_slot_solve_state_source_carrier`
- `minimize_cond.__all__`、production v2、explicit v1 solver、archive runner
  の symbol/call path は変更していない。
- file size: `11,067 -> 10,526` lines (`-541`)

検証:

- removed symbol repository search: 参照0
- `py_compile`: passed
- targeted optimize/API tests: `76 passed`
- full unit tests: `512 passed, 22 warnings`

### 2026-07-25: second deletion wave

- commit subject: `condensates: remove obsolete profile bucket adapter`
- source caller がなく、test からのみ到達していた private one-shot adapter
  `_solve_pdipm_rgie_v11_activity_correction_profile_buckets` を削除した。
- adapter 専用の重複 test を削除した。
- current API/archive path が使用する
  `_prepare_pdipm_rgie_v11_activity_correction_profile_buckets` と
  `_run_pdipm_rgie_v11_activity_correction_prepared_profile_buckets`、および
  その3-layer/budget contract test は保持した。
- runtime source: `-191` lines
- duplicate test: `-76` lines
- `minimize_cond.py`: `10,526 -> 10,335` lines

検証:

- removed source/test symbol repository search: 参照0
- `py_compile`: passed
- targeted optimize/API/archive tests: `48 passed`
- full unit tests: `511 passed, 22 warnings`

### 2026-07-26: Phase 1 lazy PIPM/RGIE boundary

- commit: `b24b489`
- commit subject: `condensates: lazy-load legacy PIPM diagnostics`
- `minimize_cond.py` の module-level `pipm_rgie_cond` import を除去した。
- raw solver、diagnostic、legacy helper name は lazy proxy として保持し、
  existing internal monkeypatch contract と `minimize_cond.__all__` を維持した。
- fresh-process import regression を追加し、plain `minimize_cond` import 後も
  `exogibbs.optimize.pipm_rgie_cond` が `sys.modules` にないことを確認した。
- diagnostic/alternate legacy call は、対応する proxy が実行された時点でのみ
  raw module を load する。

検証:

- `py_compile`: passed
- targeted optimize/API tests: `73 passed`
- full unit tests: `512 passed, 22 warnings`

### 2026-07-26: Phase 1 legacy RGIE primitive extraction

- commit subject: `condensates: extract legacy RGIE support primitives`
- `legacy_condensate/rgie_helpers.py` に、default `head_v1` support workflow が
  使用する startup と inactive-driving summary を分離した。
  - `build_rgie_condensate_init_from_policy`
  - `summarize_rgie_inactive_driving`
- `legacy_condensate/rgie_reduced_system.py` に reduced-system assembly/solve と
  post-solve residual 用 `pi` 再計算を分離した。
  - `_assemble_reduced_system_terms`
  - `_regularize_q_block`
  - `solve_reduced_gibbs_iteration_equations_cond`
  - `_solve_reduced_gibbs_iteration_equations_cond_with_metrics`
  - `_recompute_pi_for_residual`
- `pipm_rgie_cond.py` は上記 symbol を import/re-export するため、既存の
  explicit compatibility import path は維持される。
- `minimize_cond.py` の standard support helper は小モジュールを直接使用する。
  raw solver/alternate diagnostic proxy は引き続き opt-in call 時だけ
  `pipm_rgie_cond.py` を load する。
- fresh-process regression は startup、reduced residual、inactive-driving
  summary の各 helper を実行し、その後も raw module が未 load であることを
  確認する。
- file size:
  - `minimize_cond.py`: `10,415 -> 10,406` lines
  - `pipm_rgie_cond.py`: `17,121 -> 16,668` lines
  - 新規 focused package: `497` lines

この wave は symbol の ownership を移動したもので、production `head_v2`、
route version、preset、lifecycle owner、no-fallback contract は変更していない。
raw v1 solver 本体と archive replay dependency の削除は行っていない。

検証:

- worktree import provenance: passed
- `py_compile`: passed
- `git diff --check`: passed
- move-only function AST parity: `7/7`
- legacy import alias identity: `4/4`
- targeted legacy/API/production-route tests: `86 passed`
- full unit tests: `512 passed, 22 warnings`
- frozen v1 summary SHA-256: 記録済み2件とも一致

### 2026-07-26: exact replay diagnostic deletion

- commit subject: `condensates: remove obsolete exact replay diagnostics`
- default-off/test-only の exact replay/input-bundle feature を削除した。
- `pipm_rgie_cond.py` から次の cluster 固有 symbol 10件を削除した。
  - `_build_kl_gas_phase_calculate_replay_results`
  - `_build_reduced_solver_exact_input_bundle`
  - `_build_element_slot_gas_density_ntot_normalization_carrier`
  - `build_fastchem_exact_total_element_density_convention_carrier`
  - `build_condensate_density_budget_cap_total_element_density_carrier`
  - `_normalize_exact_input_bundle_context`
  - `_diagnostic_json_array`
  - `_diagnostic_source_state_hash`
  - `_build_ln_nk_producer_trace`
  - `_with_lnnk_source_state_trace`
- `minimize_cond.py` から exact replay 専用 source-trace adapter 2件を削除し、
  `build_lnnk_constructor_source_trace` を `__all__` から除外した。
- exact bundle activation/context parameter、emitter metadata、trace record
  injectionをbackend comparison/raw trace wrapperから除去した。
- exact replay専用test 6件を削除した。一般backend比較、structured wrapper、
  gas source trace、raw v1 solver testは保持した。

file size:

- `pipm_rgie_cond.py`: `16,668 -> 5,745` lines (`-10,923`)
- `minimize_cond.py`: `10,406 -> 10,214` lines (`-192`)
- `minimize_cond_diagnostics_test.py`: `1,711 -> 1,111` lines (`-600`)

この deletion は implicit `pipm_rgie_cond` star-import surface と
`minimize_cond.__all__` を縮小する意図した compatibility cleanup である。
production `head_v2`、explicit `head_v1` route、raw v1 solver、v1 profile batch、
archive runner、shared reduced-system primitiveは変更していない。

検証:

- removed runtime/test/benchmark/example/document symbol search: 参照0
- worktree import provenance: passed
- `py_compile`: passed
- `git diff --check`: passed
- preserved top-level function/class AST parity:
  `pipm_rgie_cond 62/62`、`minimize_cond 71/71`
- `minimize_cond.__all__` delta: exact-source adapter 1名のみ
- targeted legacy/API/production-route tests: `80 passed`
- full unit tests: `506 passed, 22 warnings`
- frozen v1 summary SHA-256: 記録済み2件とも一致
