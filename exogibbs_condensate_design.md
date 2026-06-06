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
4. 一方、低温の強凝縮ケースでは、凝縮相 update 以前に gas-only boundary が大きく over-inventory になる。
5. 低温ケースでは、support expansion だけでは改善せず、PD-IPM は `no_p_armijo_trial` で停止しやすい。
6. したがって低温側の主設計課題は、barrier continuation ではなく、まず気相 log-density boundary、row scaling、charge/no-ion 分岐を安定化することである。

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

## 17. 論文 draft での主張候補

論文 draft では、次の点を ExoGibbs の設計上の特徴として記述できる。

1. FastChem4 branch replay ではなく、KKT residual に基づく native formulation を採用する。
2. potential-frame residual と log-variable-frame residual を分離する。
3. 極小量 species の potential residual を solver failure と誤分類しない。
4. 凝縮相 support selection と solver convergence metric を分離する。
5. barrier continuation により凝縮相 complementarity floor を制御する。
6. FastChem4 は public-output reference oracle として用い、内部 trace は mismatch localization に限定する。

## 18. 現時点の要約

現在の ExoGibbs 凝縮相 solver 設計は、次の形に収束しつつある。

```text
外側反復:
  support refresh
  depleted-budget gas refresh
  inactive driving audit

内側反復:
  fixed-support KKT solve
  log-variable-frame stationarity
  budget closure
  complementarity centering
  barrier continuation
```

ただし低温の強凝縮ケースでは、上の内側反復へ入る前に gas-only boundary がすでに over-inventory になる。したがって次の作業は、FastChem4 から得た知見を ExoGibbs native diagnostic として使い、次を確認することである。

```text
low-temperature gas-boundary refresh:
  log element-density frame
  selected-element row scaling
  zero-target charge row handling
  no-ion proxy split
  support lifecycle after gas repair
```

この段階では FastChem4 の branch や trace 値を ExoGibbs constructor input にしない。FastChem4 は、低温で何を分解すべきかを教える reference oracle であり、production solver の直接 replay target ではない。
