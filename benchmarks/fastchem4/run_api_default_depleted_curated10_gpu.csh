#!/bin/csh -f

cd /home/kawahara/exogibbs

setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cuda,cpu
setenv XLA_PYTHON_CLIENT_PREALLOCATE false

set OUTDIR = volatiles_artifacts/fastchem4_major_species_investigation
mkdir -p $OUTDIR

set ALL_FAMILIES = ( \
  solar_highT_no_condensate_gas_regression \
  solar_silicate_first_condensation \
  solar_water_condensation \
  solar_metal_sulfide_or_Fe_Ni_S_region \
  carbon_rich_graphite_window \
  carbon_rich_CaS_MgS_AlN_window \
  SiO_s_condensate_window \
  lowT_strong_condensation_budget_stress \
  near_phase_boundary_support_sensitivity \
  complex_heavy_element_or_boron_titanium_zirconium_case \
)

echo "== GPU =="
date
nvidia-smi

echo "== curated10: API default depleted-gas fixed-support init vs FastChem4 =="
date
python volatiles_code/fastchem4_vmap_cold_rescue_compare.py \
  --families $ALL_FAMILIES \
  --exogibbs-method auto \
  --fastchem-condensation equilibrium \
  --no-explicit-gas-init \
  --gas-floor-sweep 1e-300 1e-30 1e-20 \
  --major-gas-thresholds 1e-8 1e-6 1e-4 \
  --output-dir $OUTDIR \
  --output-prefix gpu_curated10_api_default_depleted_fastchem4_screen

echo "== curated10: explicit full-budget gas init reference vs FastChem4 =="
date
python volatiles_code/fastchem4_vmap_cold_rescue_compare.py \
  --families $ALL_FAMILIES \
  --exogibbs-method auto \
  --fastchem-condensation equilibrium \
  --gas-floor-sweep 1e-300 1e-30 1e-20 \
  --major-gas-thresholds 1e-8 1e-6 1e-4 \
  --output-dir $OUTDIR \
  --output-prefix gpu_curated10_explicit_full_budget_fastchem4_screen

echo "== outputs =="
date
ls -lh \
  $OUTDIR/gpu_curated10_api_default_depleted_fastchem4_screen.md \
  $OUTDIR/gpu_curated10_api_default_depleted_fastchem4_screen.json \
  $OUTDIR/gpu_curated10_explicit_full_budget_fastchem4_screen.md \
  $OUTDIR/gpu_curated10_explicit_full_budget_fastchem4_screen.json

echo "== done =="
date
