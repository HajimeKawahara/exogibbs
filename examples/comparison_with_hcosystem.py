"""
Validation of Gibbs Minimization Against Analytical HCO System
============================================================

This example demonstrates and validates the ExoGibbs thermochemical equilibrium
solver against the analytical solution for the hydrogen dissociation equilibrium:

    CO + 3H₂ ⇌ CH₄ + H₂O

The HCO system provides analytical solutions that can be used to verify
the numerical accuracy of the Gibbs energy minimization algorithm and its
automatic differentiation capabilities.

Key validations performed:
- Single-point equilibrium composition
- Temperature derivatives (∂ln n/∂T)
- Pressure derivatives (∂ln n/∂ln P)
- Vectorized computation over temperature range
- Volume mixing ratio (VMR) calculations
"""
#FROM HERE
from exogibbs.optimize.minimize import minimize_gibbs_core
from exogibbs.optimize.minimize import minimize_gibbs
from exogibbs.test.analytic_hcosystem import HCOSystem
from exogibbs.optimize.core import compute_ln_normalized_pressure
import numpy as np
from jax import jacrev
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

##############################################################################
# Setup Test System and Parameters
# ---------------------------------
# We initialize the analytical H system and define the thermochemical
# equilibrium problem parameters.

# Initialize the analytic HCO system
hcosystem = HCOSystem()

# Define stoichiometric constraint matrix: [H atoms per species]
# Species order: [H₂, CO, CH₄, H₂O]
# Elements order: [H, C, O]
formula_matrix = jnp.array(
    [[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [4.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
).T

# Thermodynamic conditions
temperature = 1500.0  # K
P = 1.5  # bar
Pref = 1.0  # bar, reference pressure
ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

# Initial guess for log number densities
ln_nk = jnp.array([0.0, 0.0, 0.0, 0.0])  # log(n_H₂), log(n_CO), log(n_CH₄), log(n_H₂O)
ln_ntot = 0.0  # log(total number density)


def hvector_func(temperature):
    """Chemical potential function h(T) = μ°(T)/RT for [H₂, CO, CH₄, H₂O]"""
    return hcosystem.hv_hco(temperature)


# Element abundance constraint: total H nuclei = 1.0
bH = 0.5 
bC = 0.2
bO = 0.3
b_element_vector = jnp.array([bH, bC, bO])  # H, C, O

# Convergence criteria
epsilon_crit = 1e-11
max_iter = 1000

##############################################################################
# Single-Point Equilibrium Validation
# ------------------------------------
# First, we solve for equilibrium at a single temperature and pressure point
# using both the core and main minimize_gibbs functions.

# Run Gibbs minimization using core function (returns iteration count)
ln_nk_result, ln_ntot_result, counter = minimize_gibbs_core(
    temperature,
    ln_normalized_pressure,
    b_element_vector,
    ln_nk,
    ln_ntot,
    formula_matrix,
    hvector_func,
    epsilon_crit=epsilon_crit,
    max_iter=max_iter,
)

print(f"Convergence: {counter} iterations")
print(
    f"Log number densities: ln(n_H2)={ln_nk_result[0]:.6f}, ln(n_CO)={ln_nk_result[1]:.6f}, ln(n_CH4)={ln_nk_result[2]:.6f}, ln(n_H2O)={ln_nk_result[3]:.6f}"
)
from exogibbs.test.analytic_hcosystem import function_equilibrium

hco_system = HCOSystem()    
k = hco_system.equilibrium_constant(temperature, P/Pref)
x_CO = jnp.exp(ln_nk_result[1]) / bC  # n_CO / bC
res = function_equilibrium(x_CO, k, bC, bH, bO)
assert jnp.abs(res) < epsilon_crit*10.0
#TO HERE
exit()
# Run using main minimize_gibbs function (auto-differentiable version)
ln_nk_result = minimize_gibbs(
    temperature,
    ln_normalized_pressure,
    b_element_vector,
    ln_nk,
    ln_ntot,
    formula_matrix,
    hvector_func,
    epsilon_crit=epsilon_crit,
    max_iter=max_iter,
)

##############################################################################
# Temperature Derivative Validation
# ----------------------------------
# Test automatic differentiation for temperature derivatives ∂ln(n)/∂T
# against the analytical H system solution.

# Compute temperature derivative using JAX automatic differentiation
dln_dT = jacrev(
    lambda temperature_in: minimize_gibbs(
        temperature_in,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector_func,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )
)(temperature)
print(f"Numerical dln_dT: {dln_dT}")


# Compare with analytical solution
k = hcosystem.compute_k(ln_normalized_pressure, temperature)
refH = hcosystem.ln_nH_dT(jnp.array([temperature]), ln_normalized_pressure)[0]
refH2 = hcosystem.ln_nH2_dT(jnp.array([temperature]), ln_normalized_pressure)[0]
print(f"Analytical dln_dT: H={refH:.6f}, H₂={refH2:.6f}")

# Validate numerical accuracy
diff = refH - dln_dT[0]
diff2 = refH2 - dln_dT[1]
print(f"Temperature derivative errors: H={diff:.2e}, H₂={diff2:.2e}")


##############################################################################
# Pressure Derivative Validation
# --------------------------------
# Test automatic differentiation for pressure derivatives ∂ln(n)/∂ln(P)
# against the analytical H system solution.

# Compute pressure derivative using JAX automatic differentiation
dln_dlogp = jacrev(
    lambda ln_normalized_pressure: minimize_gibbs(
        temperature,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector_func,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )
)(ln_normalized_pressure)
print(f"Numerical dln_dlogp: {dln_dlogp}")

# Compare with analytical solution
refH = hcosystem.ln_nH_dlogp(jnp.array([temperature]), ln_normalized_pressure)[0]
refH2 = hcosystem.ln_nH2_dlogp(jnp.array([temperature]), ln_normalized_pressure)[0]
print(f"Analytical dln_dlogp: H={refH:.6f}, H₂={refH2:.6f}")

# Validate numerical accuracy
diff = refH - dln_dlogp[0]
diff2 = refH2 - dln_dlogp[1]
print(f"Pressure derivative errors: H={diff:.2e}, H₂={diff2:.2e}")

##############################################################################
# Vectorized Temperature Range Analysis
# --------------------------------------
# Demonstrate vectorized computation over a wide temperature range to
# validate equilibrium solutions and derivatives across different conditions.

from jax import vmap, jit

# Define temperature range for comprehensive analysis
Tarr = jnp.linspace(300.0, 6000.0, 300)  # 300K to 6000K
print(f"Temperature range: {Tarr[0]:.0f}K to {Tarr[-1]:.0f}K ({len(Tarr)} points)")

# Initial conditions for vectorized computation
ln_nk_init = jnp.array([0.0, 0.0])
ln_ntot_init = 0.0

# Vectorize minimize_gibbs over temperature axis
vmap_minimize_gibbs = vmap(
    minimize_gibbs, in_axes=(0, None, None, None, None, None, None, None, None)
)

# Compute equilibrium compositions across temperature range
ln_nk_arr = vmap_minimize_gibbs(
    Tarr,
    ln_normalized_pressure,
    b_element_vector,
    ln_nk_init,
    ln_ntot_init,
    formula_matrix,
    hvector_func,
    epsilon_crit,
    max_iter,
)

# Vectorize temperature derivatives
vmap_minimize_gibbs_dT = vmap(
    jacrev(minimize_gibbs),
    in_axes=(0, None, None, None, None, None, None, None, None),
)

# Compute temperature derivatives across temperature range
dln_dT_arr = vmap_minimize_gibbs_dT(
    Tarr,
    ln_normalized_pressure,
    b_element_vector,
    ln_nk_init,
    ln_ntot_init,
    formula_matrix,
    hvector_func,
    epsilon_crit,
    max_iter,
)

##############################################################################
# Comparison with Analytical Solutions
# -------------------------------------
# Compute analytical reference solutions and compare numerical accuracy
# across the entire temperature range.

# Compute analytical equilibrium constants
karr = vmap(hcosystem.compute_k, in_axes=(None, 0))(ln_normalized_pressure, Tarr)

# Convert log number densities to volume mixing ratios (VMRs)
n_H = jnp.exp(ln_nk_arr[:, 0])
n_H2 = jnp.exp(ln_nk_arr[:, 1])
ntot = n_H + n_H2
vmrH = n_H / ntot
vmrH2 = n_H2 / ntot

# Compare VMRs with analytical solutions
diffH = vmrH - vmap(hcosystem.vmr_h)(karr)
diffH2 = vmrH2 - vmap(hcosystem.vmr_h2)(karr)

# Compare temperature derivatives with analytical solutions
diff_dT_H = dln_dT_arr[:, 0] - hcosystem.ln_nH_dT(Tarr, ln_normalized_pressure)
diff_dT_H2 = dln_dT_arr[:, 1] - hcosystem.ln_nH2_dT(Tarr, ln_normalized_pressure)

# Report maximum errors across temperature range
print(
    f"Maximum VMR errors: H={jnp.max(jnp.abs(diffH)):.2e}, H₂={jnp.max(jnp.abs(diffH2)):.2e}"
)
print(
    f"Maximum dln_dT errors: H={jnp.max(jnp.abs(diff_dT_H)):.2e}, H₂={jnp.max(jnp.abs(diff_dT_H2)):.2e}"
)

##############################################################################
# Visualization of Results
# -------------------------
# Create comprehensive plots showing equilibrium compositions and temperature
# derivatives across the temperature range, comparing numerical and analytical solutions.
print("vis 1")
import matplotlib.pyplot as plt

# Create three-panel figure
fig = plt.figure(figsize=(10, 12))

# Panel 1: Volume mixing ratios (linear scale)
ax1 = fig.add_subplot(311)
plt.plot(Tarr, vmrH, label="H (numerical)", alpha=0.7, linewidth=2)
plt.plot(Tarr, vmrH2, label="H₂ (numerical)", alpha=0.7, linewidth=2)
plt.plot(
    Tarr, vmap(hcosystem.vmr_h)(karr), ls="--", label="H (analytical)", linewidth=2
)
plt.plot(
    Tarr, vmap(hcosystem.vmr_h2)(karr), ls="--", label="H₂ (analytical)", linewidth=2
)
plt.ylabel("Volume Mixing Ratio")
plt.title("Hydrogen Dissociation Equilibrium: 2H ⇌ H₂")
plt.legend()
plt.grid(True, alpha=0.3)

# Panel 2: Volume mixing ratios (log scale)
ax2 = fig.add_subplot(312)
plt.plot(Tarr, vmrH, label="H (numerical)", alpha=0.7, linewidth=2)
plt.plot(Tarr, vmrH2, label="H₂ (numerical)", alpha=0.7, linewidth=2)
plt.plot(
    Tarr, vmap(hcosystem.vmr_h)(karr), ls="--", label="H (analytical)", linewidth=2
)
plt.plot(
    Tarr, vmap(hcosystem.vmr_h2)(karr), ls="--", label="H₂ (analytical)", linewidth=2
)
plt.yscale("log")
plt.ylabel("Volume Mixing Ratio (log scale)")
plt.legend()
plt.grid(True, alpha=0.3)

# Panel 3: Temperature derivatives (log scale)
ax3 = fig.add_subplot(313)
plt.plot(Tarr, jnp.abs(dln_dT_arr[:, 0]), label="H (numerical)", alpha=0.7, linewidth=2)
plt.plot(
    Tarr, jnp.abs(dln_dT_arr[:, 1]), label="H₂ (numerical)", alpha=0.7, linewidth=2
)
plt.plot(
    Tarr,
    jnp.abs(hcosystem.ln_nH_dT(Tarr, ln_normalized_pressure)),
    ls="--",
    label="H (analytical)",
    linewidth=2,
)
plt.plot(
    Tarr,
    jnp.abs(hcosystem.ln_nH2_dT(Tarr, ln_normalized_pressure)),
    ls="--",
    label="H₂ (analytical)",
    linewidth=2,
)
plt.yscale("log")
plt.ylabel("|∂ln(n)/∂T| (K⁻¹)")
plt.xlabel("Temperature (K)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gibbs_minimization.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nVisualization saved as 'gibbs_minimization.png'")

##############################################################################
# Vectorized Pressure Range Analysis
# -----------------------------------
# Demonstrate vectorized computation over a pressure range to validate
# equilibrium solutions and pressure derivatives across different conditions.
print("vis 2")

# Define pressure range for comprehensive analysis
Parr = jnp.logspace(-3, 3, 200)  # 0.001 to 1000 bar (log scale)
ln_normalized_pressure_arr = jnp.log(Parr / Pref)
print(f"Pressure range: {Parr[0]:.3f} to {Parr[-1]:.0f} bar ({len(Parr)} points)")

# Vectorize minimize_gibbs over pressure axis
vmap_minimize_gibbs_pressure = vmap(
    minimize_gibbs, in_axes=(None, 0, None, None, None, None, None, None, None)
)

# Compute equilibrium compositions across pressure range
ln_nk_arr_pressure = vmap_minimize_gibbs_pressure(
    temperature,
    ln_normalized_pressure_arr,
    b_element_vector,
    ln_nk_init,
    ln_ntot_init,
    formula_matrix,
    hvector_func,
    epsilon_crit,
    max_iter,
)

# Vectorize pressure derivatives
vmap_minimize_gibbs_dlogp = vmap(
    jacrev(
        minimize_gibbs, argnums=1
    ),  # derivative w.r.t. ln_normalized_pressure (arg 1)
    in_axes=(None, 0, None, None, None, None, None, None, None),
)

# Compute pressure derivatives across pressure range
dln_dlogp_arr = vmap_minimize_gibbs_dlogp(
    temperature,
    ln_normalized_pressure_arr,
    b_element_vector,
    ln_nk_init,
    ln_ntot_init,
    formula_matrix,
    hvector_func,
    epsilon_crit,
    max_iter,
)

# Compute analytical equilibrium constants for pressure range
karr_pressure = vmap(hcosystem.compute_k, in_axes=(0, None))(
    ln_normalized_pressure_arr, temperature
)

# Convert log number densities to volume mixing ratios (VMRs)
n_H_pressure = jnp.exp(ln_nk_arr_pressure[:, 0])
n_H2_pressure = jnp.exp(ln_nk_arr_pressure[:, 1])
ntot_pressure = n_H_pressure + n_H2_pressure
vmrH_pressure = n_H_pressure / ntot_pressure
vmrH2_pressure = n_H2_pressure / ntot_pressure

# Compare VMRs with analytical solutions
diffH_pressure = vmrH_pressure - vmap(hcosystem.vmr_h)(karr_pressure)
diffH2_pressure = vmrH2_pressure - vmap(hcosystem.vmr_h2)(karr_pressure)

# Compare pressure derivatives with analytical solutions
temp_array = jnp.full_like(ln_normalized_pressure_arr, temperature)
diff_dlogp_H = dln_dlogp_arr[:, 0] - hcosystem.ln_nH_dlogp(
    temp_array, ln_normalized_pressure_arr
)
diff_dlogp_H2 = dln_dlogp_arr[:, 1] - hcosystem.ln_nH2_dlogp(
    temp_array, ln_normalized_pressure_arr
)

# Report maximum errors across pressure range
print(
    f"Maximum VMR errors: H={jnp.max(jnp.abs(diffH_pressure)):.2e}, H₂={jnp.max(jnp.abs(diffH2_pressure)):.2e}"
)
print(
    f"Maximum dln_dlogp errors: H={jnp.max(jnp.abs(diff_dlogp_H)):.2e}, H₂={jnp.max(jnp.abs(diff_dlogp_H2)):.2e}"
)

##############################################################################
# Pressure Visualization
# -----------------------
# Create comprehensive plots showing equilibrium compositions and pressure
# derivatives across the pressure range.

# Create three-panel figure for pressure analysis
fig_pressure = plt.figure(figsize=(10, 12))

# Panel 1: Volume mixing ratios vs pressure (linear scale)
ax1_p = fig_pressure.add_subplot(311)
plt.plot(Parr, vmrH_pressure, label="H (numerical)", alpha=0.7, linewidth=2)
plt.plot(Parr, vmrH2_pressure, label="H₂ (numerical)", alpha=0.7, linewidth=2)
plt.plot(
    Parr,
    vmap(hcosystem.vmr_h)(karr_pressure),
    ls="--",
    label="H (analytical)",
    linewidth=2,
)
plt.plot(
    Parr,
    vmap(hcosystem.vmr_h2)(karr_pressure),
    ls="--",
    label="H₂ (analytical)",
    linewidth=2,
)
plt.xscale("log")
plt.ylabel("Volume Mixing Ratio")
plt.title(f"Hydrogen Dissociation Equilibrium vs Pressure at T={temperature:.0f}K")
plt.legend()
plt.grid(True, alpha=0.3)

# Panel 2: Volume mixing ratios vs pressure (log scale)
ax2_p = fig_pressure.add_subplot(312)
plt.plot(Parr, vmrH_pressure, label="H (numerical)", alpha=0.7, linewidth=2)
plt.plot(Parr, vmrH2_pressure, label="H₂ (numerical)", alpha=0.7, linewidth=2)
plt.plot(
    Parr,
    vmap(hcosystem.vmr_h)(karr_pressure),
    ls="--",
    label="H (analytical)",
    linewidth=2,
)
plt.plot(
    Parr,
    vmap(hcosystem.vmr_h2)(karr_pressure),
    ls="--",
    label="H₂ (analytical)",
    linewidth=2,
)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Volume Mixing Ratio (log scale)")
plt.legend()
plt.grid(True, alpha=0.3)

# Panel 3: Pressure derivatives (log scale)
ax3_p = fig_pressure.add_subplot(313)
plt.plot(
    Parr, jnp.abs(dln_dlogp_arr[:, 0]), label="H (numerical)", alpha=0.7, linewidth=2
)
plt.plot(
    Parr, jnp.abs(dln_dlogp_arr[:, 1]), label="H₂ (numerical)", alpha=0.7, linewidth=2
)
plt.plot(
    Parr,
    jnp.abs(hcosystem.ln_nH_dlogp(temp_array, ln_normalized_pressure_arr)),
    ls="--",
    label="H (analytical)",
    linewidth=2,
)
plt.plot(
    Parr,
    jnp.abs(hcosystem.ln_nH2_dlogp(temp_array, ln_normalized_pressure_arr)),
    ls="--",
    label="H₂ (analytical)",
    linewidth=2,
)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("|∂ln(n)/∂ln(P)|")
plt.xlabel("Pressure (bar)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gibbs_minimization_pressure.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPressure analysis visualization saved as 'gibbs_minimization_pressure.png'")
