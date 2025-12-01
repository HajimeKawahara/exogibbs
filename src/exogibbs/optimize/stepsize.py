import jax.numpy as jnp

CEA_SIZE = 18.420681        # = -ln(1e-8)
LN_X_CAP = 9.2103404        # = -ln(1e-4)

def lambda_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot, size=CEA_SIZE):
    """Compute a safe step size lambda for CEA gas phase minimization.
    """
    # λ1: ensure |Δln n|<=0.4, |Δln n_k|<=2
    cap_ntot = 5.0 * jnp.abs(delta_ln_ntot)           # 1/0.4
    cap_sp   = jnp.max(jnp.abs(delta_ln_nk))
    denom1   = jnp.maximum(jnp.maximum(cap_ntot, cap_sp), 1e-300)
    lam1     = 2.0 / denom1

    # maintain x_k<=1e-4 if increasing when x_k<=1e-8
    ln_xk  = ln_nk - ln_ntot
    small  = (ln_xk <= -size) & (delta_ln_nk >= 0.0)
    denom2 = delta_ln_nk - delta_ln_ntot
    safe   = small & (denom2 > 0.0)
    cand   = ( -LN_X_CAP - ln_xk ) / denom2           # (-ln 1e-4 - ln xk)/(Δln nk - Δln n)
    lam2   = jnp.where(jnp.any(safe), jnp.min(jnp.where(safe, cand, jnp.inf)), jnp.inf)

    lam = jnp.minimum(1.0, jnp.minimum(lam1, lam2))
    # safe guard
    lam = jnp.clip(lam, 1e-6, 1.0)
    return lam
