# ExoGibbs 凝縮相アルゴリズム設計案

## 1. 目的

本設計案は、ExoGibbs における気相・凝縮相同時平衡計算のための native アルゴリズムを記述する。ここでの目的は、FastChem4 の内部 branch を再現することではなく、Gibbs 自由エネルギー最小化を、元素収支制約、凝縮相の不等式制約、および interior-point 型の相補性条件として定式化することである。

FastChem4 は比較用 reference oracle として用いるが、ExoGibbs の solver は FastChem4 の branch replay ではなく、ExoGibbs 固有の RGIE/PIPM/PD-IPM 系として設計する。

## 2. 変数

気相 species 数を `N_g`、凝縮相 species 数を `N_c`、元素数を `N_e` とする。

主変数は次である。

```text
q_i = ln n_i        i = 1, ..., N_g
r_j = ln m_j        j = 1, ..., N_c
qtot = ln ntot
lambda_a            a = 1, ..., N_e
rho_j               j = 1, ..., N_c
```

ここで、

- `n_i` は気相 species `i` の量
- `m_j` は凝縮相 species `j` の量
- `ntot` は気相総量
- `lambda` は元素収支に対応する Lagrange multiplier、すなわち element potential
- `rho_j` は凝縮相の不等式制約または barrier/complementarity を扱うための dual-like variable

指数変換により、

```text
n_i = exp(q_i)
m_j = exp(r_j)
ntot = exp(qtot)
```

とする。

## 3. 化学ポテンシャル項

気相の無次元化学ポテンシャル項を

```text
g_i(q, qtot; T, P)
  = h_i(T) + q_i - qtot + ln(P / P0)
```

と定義する。

ここで `h_i(T)` は標準化学ポテンシャルを `RT` で割った量に対応する。

凝縮相の無次元化学ポテンシャル項を

```text
c_j(T)
```

と書く。実装上は `hvector_cond[j]` に対応する。

### 3.1 FastChem 型 mass-action constants と RGIE source の区別

FastChem 型の気相 log-density boundary では、気相 species `i` について

```text
q_i = K_i(T, P) + A_g[:, i]^T y
```

の形で `q_i = ln n_i` を復元する。ここで `K_i` は FastChem-frame の mass-action constants に対応し、`y` は元素側の log-density / potential-like variable である。

一方、RGIE の気相 stationarity は

```text
A_g[:, i]^T lambda - g_i = 0
```

であり、実装では

```text
g_i = q_i + gas_stationarity_source_i
```

という形で評価している。したがって FastChem-frame の log-density boundary を RGIE の stationarity frame に接続する場合は、

```text
gas_stationarity_source_i = -K_i(T, P)
q_i + gas_stationarity_source_i = A_g[:, i]^T lambda
```

となるように source convention を合わせる必要がある。

ここで `hvector` と `K_i` を混同してはならない。`hvector` は Gibbs/RGIE 側の thermochemical source であり、FastChem-frame の mass-action constants と同一とは限らない。electron refresh や depleted gas refresh に `hvector` を mass-action constants として渡すと、charge/budget が閉じても `q + gas_stationarity_source` が `A_g.T @ lambda` の frame から外れ、気相 stationarity が壊れて見える。

## 4. 制約

気相 stoichiometry matrix を `A_g`、凝縮相 stoichiometry matrix を `A_c`、元素 budget を `b` とする。

元素収支は

```text
A_g n + A_c m = b
```

である。

気相総量制約は

```text
sum_i n_i = ntot
```

である。

凝縮相量は非負であり、

```text
m_j >= 0
```

を満たす。

## 5. KKT 条件

### 5.1 気相 stationarity

気相 species `i` に対する potential-frame stationarity は

```text
A_g[:, i]^T lambda - g_i = 0
```

である。

ただし ExoGibbs の主変数は `q_i = ln n_i` であるため、solver-stage で評価する log-variable-frame stationarity は

```text
n_i (A_g[:, i]^T lambda - g_i) = 0
```

である。

この区別は重要である。`n_i` が極小の species では、potential-frame residual が大きくても、log-variable-frame residual は KKT 条件上小さい場合がある。

### 5.2 凝縮相 stationarity と complementarity

凝縮相 species `j` に対する driving term を

```text
d_j = A_c[:, j]^T lambda - c_j(T)
```

と定義する。

凝縮相の interior-point 型 complementarity residual は、barrier parameter `mu > 0` を用いて

```text
R_c,j = m_j d_j + mu
```

と置く。

ここで `mu = exp(epsilon)` として実装することができる。

中心経路上では

```text
m_j d_j + mu = 0
```

を満たす方向へ進む。

符号規約は `c_j(T)` の定義に依存するため、実装では `d_j = A_c[:, j]^T lambda - hvector_cond[j]` を基準にし、artifact で明示する。

### 5.3 元素収支 residual

元素収支 residual は

```text
R_b = A_g n + A_c m - b
```

である。

### 5.4 気相総量 residual

気相総量 residual は

```text
R_t = sum_i n_i - ntot
```

である。

## 6. Residual vector

solver-stage residual は次の block から構成する。

```text
R_g,i = n_i (A_g[:, i]^T lambda - g_i)
R_c,j = m_j (A_c[:, j]^T lambda - c_j) + mu
R_b   = A_g n + A_c m - b
R_t   = sum_i n_i - ntot
```

すなわち、

```text
R(q, r, qtot, lambda; mu)
  = [R_g, R_c, R_b, R_t]
```

である。

この residual を小さくすることが、ExoGibbs native condensate solver の基本目標である。

## 7. Potential-frame と log-variable-frame の使い分け

ExoGibbs では、以下を明確に分ける。

```text
potential-frame gas residual:
  A_g.T lambda - g

log-variable-frame gas residual:
  n * (A_g.T lambda - g)

potential-frame condensate driving:
  A_c.T lambda - c

log-variable / complementarity condensate residual:
  m * (A_c.T lambda - c) + mu
```

potential-frame は、凝縮候補選択、activity 診断、support expansion の参考に使う。

log-variable-frame は、solver-stage KKT residual、収束判定、merit function に使う。

この分離により、極小量 species の potential-frame residual が solver の主失敗要因として誤分類されることを避ける。

## 8. Support set

凝縮相全体を常に full dense に解くのではなく、候補 support set `S` を用いる。

```text
S subset {1, ..., N_c}
```

support 内の凝縮相量は明示的に `m_j > 0` として扱い、support 外の凝縮相は `m_j = 0` として扱う。

support selection には potential-frame driving を使ってよい。

```text
d_j = A_c[:, j]^T lambda - c_j
```

ただし、support selection の指標と solver residual の指標は分ける。support selection で大きい `d_j` が見えても、それをそのまま KKT residual として扱わない。

## 9. 外側反復

外側反復は、support と気相状態の整合を更新する。

典型的な外側反復は次の流れである。

```text
1. 現在の support S を固定する
2. 凝縮相量 m を用いて depleted budget を作る
3. depleted budget に対して気相 log-density state を更新する
4. lambda を気相・凝縮相を含む residual frame に合わせて更新する
5. support 外の potential-frame driving を評価する
6. 必要なら support を追加または保持する
7. 内側 solver に進む
```

FastChem4 でも、凝縮相選択、気相再計算、安定性評価、必要に応じた coupled solve という lifecycle が存在する。ExoGibbs ではこれを branch replay としてではなく、KKT residual を明示した外側反復として設計する。

### 9.1 electron refresh と lambda gauge consistency

電荷行は通常の元素 budget ではなく、zero-target charge constraint である。

```text
z_g^T n = 0
```

低温・強凝縮ケースでは、この行が気相 log-density boundary と強く結びつき、通常の相対元素収支と同じ扱いでは診断を誤ることがある。

M4342-M4344 の T700 診断で分かった重要点は、electron refresh の成否は、charge/budget closure だけでは判定できないということである。refresh 後の state は、同じ source convention で

```text
q + gas_stationarity_source = A_g.T @ lambda
```

を満たす必要がある。

M4343 時点では、electron refresh に `_gas_hvector(contract)` を mass-action constants として渡していたため、RGIE 側の `_mass_action_constants(contract, T)` に基づく stationarity frame と一致していなかった。その結果、lambda-only residual が巨大になり、electron policy 自体が失敗したように見えていた。

M4344 で refresh 側に RGIE 評価と同じ mass-action constants を渡すよう修正したところ、T700 の gas stationarity gap は次まで下がった。

```text
max_lambda_only_residual_l2 = 2.082445943218344e-11
sentinel_count = 0
qtot_gap = 0.0
```

このため、electron refresh は次の条件を満たすときだけ HEAD route の一部として扱う。

1. charge/budget が閉じる。
2. mass-action constants と gas stationarity source の convention が一致している。
3. `q + gas_stationarity_source` が `A_g.T @ lambda` frame に射影できる。
4. sentinel thermochemistry が混入していない。

この設計は FastChem4 branch replay ではない。FastChem4 の「凝縮更新後に気相を再計算し、安定性を再評価する」lifecycle を、ExoGibbs native の lambda gauge consistency 条件として再定式化したものである。

## 10. 内側反復

内側反復では、support `S` を固定し、次の変数を同時に更新する。

```text
q_S, r_S, qtot, lambda
```

ここで `q_S` は全気相 species の log amount、`r_S` は support 内凝縮相の log amount である。

内側反復の目標は、

```text
R_g -> 0
R_c -> 0
R_b -> 0
R_t -> 0
```

である。

更新方向は Newton 型または reduced Newton 型で求める。更新幅は trust region、line search、または component-wise merit によって制御する。

## 11. Barrier continuation

現在の実験では、`mu = exp(epsilon)` が residual floor を作っている可能性が高い。

したがって、production-adjacent solver では固定 `mu` ではなく continuation を使う。

```text
mu_0 > mu_1 > ... > mu_final
```

各 `mu_l` で内側反復を行い、十分に center path に近づいたら次の小さい `mu` へ進む。

受理条件の例は次である。

```text
||R_b||_inf does not increase beyond tolerance
||R_g||_inf does not increase beyond tolerance
||R_c||_inf decreases or tracks mu
all q, r, lambda remain finite
support remains valid
```

## 12. Merit function

候補更新の受理には、単一の L2 residual ではなく、成分分解された merit を使う。

候補としては、

```text
M = w_b ||R_b||_inf
  + w_t |R_t|
  + w_g ||R_g||_inf
  + w_c ||R_c||_inf
  + w_i max_positive_inactive_driving
```

を用いる。

ただし、重み `w_*` は固定の magic number として production に入れるのではなく、diagnostic campaign で意味を確認してから設計する。

重要なのは、budget closure だけを改善して stationarity を破壊しないこと、また stationarity だけを改善して budget を破壊しないことである。

## 13. 収束判定

収束判定は次を同時に満たすことを要求する。

```text
||A_g n + A_c m - b||_inf < tol_budget
|sum(n) - ntot| < tol_total
||n * (A_g.T lambda - g)||_inf < tol_gas
||m * (A_c.T lambda - c) + mu||_inf < tol_cond
max_positive_inactive_driving < tol_inactive
```

`mu` が有限の場合、`tol_cond` は `mu` に応じて設定する。

最終解では `mu` を十分小さくし、barrier floor が物理的 residual を支配しないようにする。

## 14. FastChem4 との比較

FastChem4 との比較は、以下の順で行う。

1. species order、stoichiometry、thermochemistry contract の一致確認
2. 気相-only runtime regression
3. public condensate output comparison
4. ExoGibbs native KKT residual comparison
5. 必要な場合のみ temporary trace による mismatch localization

FastChem4 trace 値を ExoGibbs constructor input にしてはならない。

FastChem4 の branch label、LM damping、fallback timing は、ExoGibbs solver の production acceptance には使わない。

## 15. 現在までの実験からの設計判断

現在までの実験から、次が分かっている。

1. 中温から高温の C1 系では、budget residual は十分小さい領域まで到達できる。
2. 未スケールの気相 stationarity residual は巨大に見えるが、log-variable-frame では支配的ではない場合がある。
3. その領域では、residual floor は `mu = exp(epsilon)` に対応する active complementarity が支配している。
4. 一方、低温の強凝縮ケースでは、凝縮相 update 以前に gas-only boundary が大きく over-inventory になる場合がある。
5. T700 では、support boundary そのものよりも、electron refresh と RGIE gas stationarity source の convention 整合が支配的だった。
6. source convention を合わせた electron-policy refresh により、T700 は final barrier まで到達した。
7. したがって低温側の主設計課題は、barrier continuation だけではなく、気相 log-density boundary、row scaling、charge/no-ion 分岐、そして lambda gauge consistency を同時に安定化することである。

この判断により、アルゴリズム設計は温度・状態に応じて次の二つの診断経路へ分岐する。

```text
中温・高温 C1 経路:
  support-stable barrier continuation
    + log-variable-frame KKT residual
    + budget-safe amount update
    + inactive driving guard

低温 gas-boundary 経路:
  log element-density gas refresh
    + selected-element row scaling
    + zero-target charge row handling
    + source-convention-safe electron refresh
    + lambda gauge consistency check
    + no-ion diagnostic split
    + support lifecycle reassessment after gas repair
```

## 16. 次に検証すべきアルゴリズム部品

次に検証すべき部品は以下である。順番としては、低温ケースでは 16.1 から 16.4 を先に確認し、その後に 16.5 以降へ戻る。

### 16.1 low-temperature gas-boundary refresh

低温ケースでは、まず気相だけで

```text
A_g n = b
sum_i n_i = ntot
```

がどの frame で破れているかを確認する。

FastChem4 の低温改善は、熱力学データ更新だけでなく、log element density、selected-element Newton、row scaling、update cap に依存している。ExoGibbs ではこれを branch replay として使わず、次の native diagnostic として切り出す。

```text
q_i = ln n_i
b_g(q) = A_g exp(q)
R_b^gas = b_g(q) - b
```

元素行ごとに

```text
s_a = max(sum_i |A_g[a, i]| n_i, |b_a|, tiny)
R_scaled,a = R_b^gas,a / s_a
```

を評価する。

### 16.2 charge/no-ion boundary split

電荷行は通常の正の元素 budget ではなく、zero-target equality constraint である。したがって

```text
z_g^T n = 0
```

を通常の相対元素誤差と同じに扱わない。低温ではイオンが物理的に不要な場合が多いため、診断として charged species を除いた no-ion proxy を評価する。

```text
n_neutral = n restricted to z_g = 0
R_b^neutral = A_g n_neutral - b
```

もし `||R_b^neutral||` が `||R_b||` とほぼ同じなら、主因は electron/ion ではなく気相 log-density gauge または row scaling にある。

### 16.3 selected-element row scaling

FastChem4 の multidimensional Newton accelerator は、未収束元素を選び、その行をスケールして同時に解く。ExoGibbs では次を診断 prototype とする。

```text
U = { a | |R_scaled,a| > tol_row }
J_U = A_g[U, :] diag(n) A_g[U, :]^T
rhs_U = -R_b^gas[U]
J_scaled,U Δlambda_U = rhs_scaled,U
```

ここで update は `q` に戻して

```text
Δq = A_g[U, :]^T Δlambda_U
```

と評価する。ただし production solver へ入れる前に、total-density gauge と stationarity frame との整合を確認する。

### 16.4 support lifecycle after gas repair

support expansion は gas boundary repair の前に行うと誤った driving を拾う可能性がある。したがって低温では

```text
1. gas boundary refresh
2. lambda / element potential refresh
3. inactive condensate driving evaluation
4. support refresh
5. fixed-support PD-IPM
```

の順に進める。

### 16.5 epsilon sweep

`mu = exp(epsilon)` を段階的に小さくし、`R_c` の floor が `mu` に追従するか確認する。これは gas boundary が十分安定しているケース、または gas refresh 後の低温ケースで行う。

### 16.6 complementarity row decomposition

各凝縮相について、

```text
R_c,j = m_j d_j + mu
```

を分解し、どの凝縮相が floor を作っているかを見る。

### 16.7 amount update policy

`m_j` の更新が budget を壊さず、かつ `R_c,j` を下げるかを確認する。

### 16.8 support-stable centering

support を固定したまま center path に近づける反復を作る。

### 16.9 support refresh

support-stable centering 後に inactive driving を評価し、必要な凝縮種だけを追加する。

## 17. sentinel thermochemistry と初収束到達点

最近の診断で、support 内に混入していた sentinel thermochemistry が、固定 support PD-IPM/RGIE の凝縮相 stationarity を支配していたことが確認された。

ここでいう sentinel thermochemistry とは、凝縮相の標準化学ポテンシャル `c_j(T)` が実際の物理値ではなく、実装上の巨大値として入っている状態である。実験ではおおむね

```text
|c_j(T)| >= 1e10
```

を sentinel とみなし、これはその温度・条件で通常の fixed-support solver に入れるべき凝縮相ではないと扱う。

sentinel が support に入ると、凝縮相 residual

```text
R_c,j = m_j (A_c[:, j]^T lambda - c_j) + mu
```

の `c_j` 項が巨大になり、実際には物理的な stationarity gap ではなく、データ境界値によって residual が支配される。この状態で Newton 方向や PD-IPM 方向を評価すると、更新方向の良し悪しではなく sentinel 混入の影響を見てしまう。

したがって fixed-support solver に入る前に、thermochemistry-valid support filter を通す。

```text
S_valid = { j in S | c_j(T) is finite and |c_j(T)| < c_sentinel }
```

この filter は production default ではなく、explicit opt-in diagnostic callsite policy として扱う。FastChem4 の trace 値や public output 値を constructor input にするものではなく、ExoGibbs が持つ thermochemistry table の有効性境界を確認するための filter である。

この境界を入れた結果、support-positive C1 cases の 4 ケースで次が確認された。

```text
case count: 4
accepted-step cases: 4
converged cases: 2
nonconverged cases: 2
```

収束したケースは次である。

```text
solar_silicate_first_condensation__T1500_P1
near_phase_boundary_support_sensitivity__T1490_P1
```

この 2 ケースでは、短い multistep diagnostic loop の中で、次の residual が収束閾値まで下がった。

```text
||R_b||       budget residual
||R_t||       total-density residual
||R_g||       gas log-variable-frame stationarity
||R_c||       condensate complementarity / stationarity
```

これは ExoGibbs native RGIE/PD-IPM 系で、実ケースを収束まで持っていける経路が初めて明確に見えた到達点である。ただし、これはまだ production solver の通常経路ではない。あくまで explicit opt-in diagnostic callsite-near loop での収束であり、production solver 完成を意味しない。

未収束の 2 ケースは同じ原因ではなかった。

```text
solar_silicate_first_condensation__T1400_P0p1:
  initial budget residual がすでに極小であるため、
  absolute budget gate が相対的な budget 悪化を許す。
  単純な per-step relative budget guard では、各ステップの悪化は抑えられても
  累積 drift を止めきれなかった。
  初期 budget を anchor とする relative budget guard により、
  final budget ratio は約 1.0606 から約 1.0081 へ改善し、
  T1400 の budget drift は抑制された。
  ただし収束そのものは未達であり、残る主因は gas / condensate stationarity 側である。

near_phase_boundary_support_sensitivity__T1510_P1:
  budget は大きく改善するが、
  condensate stationarity が収束閾値より上に残る。
  次は condensate stationarity 向けの反復延長または acceptance retuning が必要である。
```

この結果から、次の設計判断を採用する。

1. sentinel thermochemistry を含む凝縮相は fixed-support PD-IPM/RGIE の active support に入れない。
2. sentinel 除去は単なる数値掃除ではなく、収束可能性に直結する support validity condition である。
3. ただし sentinel 除去だけを solver convergence mechanism と見なしてはならない。
4. 収束しないケースでは、budget gate と condensate stationarity gate を分けて扱う。
5. tiny-budget case では absolute residual だけでなく、初期または best-so-far budget を anchor とする relative budget worsening を監視する。
6. phase-boundary case では、condensate stationarity を下げるための iteration policy または merit retuning が必要である。

この到達点により、当面の内側反復は次の形へ更新する。

```text
fixed-support inner diagnostic loop:
  1. receive candidate support S
  2. remove sentinel thermochemistry entries from S
  3. solve explicit opt-in thermo-valid PD-IPM/RGIE step
  4. accept by component-aware merit
  5. apply anchored relative budget guard for tiny-budget states
  6. continue or retune when condensate stationarity remains dominant
```

重要なのは、これは FastChem4 branch replay ではないという点である。FastChem4 から得た教訓は、invalid thermochemistry support を coupled solver に入れないという lifecycle 上の知見であり、ExoGibbs ではこれを KKT residual と support validity の言葉で再定式化する。

## 18. 論文 draft での主張候補

論文 draft では、次の点を ExoGibbs の設計上の特徴として記述できる。

1. FastChem4 branch replay ではなく、KKT residual に基づく native formulation を採用する。
2. potential-frame residual と log-variable-frame residual を分離する。
3. 極小量 species の potential residual を solver failure と誤分類しない。
4. 凝縮相 support selection と solver convergence metric を分離する。
5. barrier continuation により凝縮相 complementarity floor を制御する。
6. FastChem4 は public-output reference oracle として用い、内部 trace は mismatch localization に限定する。
7. thermochemistry-valid support を fixed-support solver の前提条件として扱い、sentinel 値に支配された residual を物理的 stationarity gap と混同しない。
8. 初期 budget が極小の状態では absolute budget gate だけでなく anchored relative budget worsening を管理する。

## 19. 現時点の要約

現在の ExoGibbs 凝縮相 solver 設計は、次の形に収束しつつある。

```text
外側反復:
  support refresh
  depleted-budget gas refresh
  inactive driving audit

内側反復:
  thermochemistry-valid support filter
  fixed-support KKT solve
  log-variable-frame stationarity
  budget closure
  complementarity centering
  barrier continuation
  anchored relative budget guard for tiny-budget states
```

ただし低温の強凝縮ケースでは、上の内側反復へ入る前に gas-only boundary がすでに over-inventory になる。したがって次の作業は、FastChem4 から得た知見を ExoGibbs native diagnostic として使い、次を確認することである。

```text
low-temperature gas-boundary refresh:
  log element-density frame
  selected-element row scaling
  zero-target charge row handling
  source-convention-safe electron refresh
  lambda gauge consistency
  no-ion proxy split
  support lifecycle after gas repair
```

この段階では FastChem4 の branch や trace 値を ExoGibbs constructor input にしない。FastChem4 は、低温で何を分解すべきかを教える reference oracle であり、production solver の直接 replay target ではない。

加えて、中温から相境界付近の C1 cases では、sentinel thermochemistry を除いた explicit diagnostic callsite-near loop により、4 ケース中 2 ケースで初めて収束が確認された。このため、RGIE/PD-IPM 系そのものは引き続き有望である。T1400 については anchored relative budget guard により budget drift が抑制されたため、次の焦点は solver family を変更することではなく、T1400 と T1510 に残る gas / condensate stationarity の支配成分を切り分けて retuning することである。

M4344 時点では、T700 metal sulfide case も source-convention-safe electron refresh により final barrier へ到達した。これは、低温・電荷行を含むケースでも、FastChem4 の branch を replay せずに、ExoGibbs native の lambda gauge consistency と depleted gas refresh を整合させれば進められることを示している。ただし、これは HEAD route から分岐した別 route ではない。単一 HEAD policy の中で、electron refresh が主要な gauge repair として効いた観測結果である。

M4345 では、同じ source-convention-safe electron refresh を T500 low-temperature strong-condensation case に適用した。結果は次である。

```text
decision = LOWT_ELECTRON_REFRESH_GAUGE_GENERALIZES_BUT_ROUTE_PARTIAL
lambda_gauge_compatible_count = 2 / 2
final_barrier_count = 0 / 2
```

この結果から、electron refresh は単独 route ではなく、統合 policy 内の gauge repair として扱う必要がある。

```text
source-convention-safe electron refresh as gauge repair:
  q + gas_stationarity_source を A_g.T @ lambda frame に合わせる。
  T500/T700 の両方で有効。

integrated HEAD policy after gauge repair:
  budget guard, center-primary restoration, barrier continuation と組み合わせる。
  どの部品が主要に効くかは case の観測結果として記録する。
```

したがって、electron refresh を一般的な低温 route として昇格するのではなく、HEAD route の共通 gauge repair として保持する。設計上は、source convention check、budget guard、center-primary restoration、barrier continuation を一つの policy として組み合わせる。M4345 の T500/T700 比較は、別 route を選ぶための分岐表ではなく、同じ policy の中でどの補助条件が主要に効くかを示す診断結果である。

M4346 では、この違いをさらに分解した。結論は次である。

```text
common finding:
  source_convention_safe_electron_refresh_preserves_lambda_gauge

primary difference:
  t500_has_depleted_budget_and_barrier_advancement_gap
```

T500/T700 の違いは lambda gauge ではない。どちらも source-convention-safe electron refresh 後に `q + gas_stationarity_source` は `A_g.T @ lambda` frame に乗る。差は、T500 では refresh 後にも neutral depleted-budget residual と stage1 budget residual が大きく残る点である。

```text
T500 max neutral log budget residual after electron refresh: 5.663824094349078
T700 max neutral log budget residual after electron refresh: 1.403943628019988e-11

T500 max stage1 budget residual: 0.7335650159979301
T700 max stage1 budget residual: 0.0014846412451118764
```

この差が barrier advancement に直接現れる。

```text
T500 max final epsilon gap: 13.815510557964274
T700 max final epsilon gap: 0.000993164980670258
```

したがって、次の設計課題は electron refresh のさらなる分岐ではなく、post-gauge state に対する budget guard と barrier advancement の接続である。言い換えると、HEAD policy は次のように見る。

```text
1. source convention を合わせて lambda gauge を修復する
2. depleted-budget residual を評価する
3. budget が大きく残る場合は center-primary budget guard を先に強める
4. budget が小さい場合は barrier continuation を進める
```

## HEAD route v1.1 runtime fallback boundary

fresh API profile 実行では、水凝縮の中間層で restricted support solver が成功しても、HEAD route lifecycle の primary continuation が `no_p_armijo_trial` で止まり、route selector が accepted に進まない場合がある。この状態は「凝縮量や boundary が作れない」のではなく、「standard route selector が centered evidence を得られない」失敗である。

HEAD route v1.1 では、この場合の runtime boundary を次のように定義する。

```text
if restricted support candidate is finite
and no saved metric/primary/refresh evidence is injected
and lifecycle route_result is not accepted:
  return native_budget_seed_fallback_budget_tradeoff
```

この fallback は fresh API の継続性を守るための caveat tier であり、FastChem4 trace/public/runtime values を constructor input に使わない。結果は `converged_with_caveat` として返し、diagnostics に lifecycle が accepted しなかったことと、restricted solver が成功していたかどうかを残す。

v1.1 は 10 curated demo profiles / 99 layers で例外なしに完走した。水凝縮 profile の中間4層は `not_converged` から `native_budget_seed_fallback_budget_tradeoff` の caveat accepted に移った。ただし、primary continuation の `no_p_armijo_trial` 自体は未解決であり、次の solver 改善対象として残す。
