import jax.numpy as jnp

CEA_SIZE = 18.420681  # = -ln(1e-8)
LN_X_CAP = 9.2103404  # = -ln(1e-4)


def stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot, size=CEA_SIZE):
    """Compute a heuristic safe step size lambda for CEA gas phase minimization."""
    # λ1: ensure |Δln n|<=0.4, |Δln n_k|<=2
    cap_ntot = 5.0 * jnp.abs(delta_ln_ntot)  # 1/0.4
    cap_sp = jnp.max(jnp.abs(delta_ln_nk))
    denom1 = jnp.maximum(jnp.maximum(cap_ntot, cap_sp), 1e-300)
    lam1 = 2.0 / denom1

    # maintain x_k<=1e-4 if increasing when x_k<=1e-8
    ln_xk = ln_nk - ln_ntot
    small = (ln_xk <= -size) & (delta_ln_nk >= 0.0)
    denom2 = delta_ln_nk - delta_ln_ntot
    safe = small & (denom2 > 0.0)
    cand = (-LN_X_CAP - ln_xk) / denom2  # (-ln 1e-4 - ln xk)/(Δln nk - Δln n)
    lam2 = jnp.where(jnp.any(safe), jnp.min(jnp.where(safe, cand, jnp.inf)), jnp.inf)

    lam = jnp.minimum(1.0, jnp.minimum(lam1, lam2))
    # Do not force a minimum step; allow very small values when needed.
    lam = jnp.clip(lam, 0.0, 1.0)
    return lam


def stepsize_cond_heurstic(delta_ln_mk):
    cap_m = jnp.max(jnp.abs(delta_ln_mk))
    denom = jnp.maximum(cap_m, 1e-300)
    lam1_cond = 2.0 / denom  # |Δ ln m| <~ 2
    return lam1_cond


S_MAX = 1e6  # want to be sk <= 1e6
LOG_S_MAX = jnp.log(S_MAX)


def stepsize_sk(delta_ln_mk, ln_mk, epsilon, log_s_max=LOG_S_MAX):
    inc = delta_ln_mk > 0.0
    denom = 2.0 * delta_ln_mk
    num = log_s_max + epsilon - 2.0 * ln_mk  # 2 ln m_new - eps <= log_s_max
    safe = inc & (denom > 0.0)
    cand = num / denom
    # take a minimum of lambda along each Δ direction
    lam2_cond = jnp.where(
        jnp.any(safe), jnp.min(jnp.where(safe, cand, jnp.inf)), jnp.inf
    )
    return lam2_cond
