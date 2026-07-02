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
python benchmarks/fastchem4/fastchem4_vmap_cold_rescue_compare.py \
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
python benchmarks/fastchem4/fastchem4_vmap_cold_rescue_compare.py \
  --families $ALL_FAMILIES \
  --exogibbs-method auto \
  --fastchem-condensation equilibrium \
  --gas-floor-sweep 1e-300 1e-30 1e-20 \
  --major-gas-thresholds 1e-8 1e-6 1e-4 \
  --output-dir $OUTDIR \
  --output-prefix gpu_curated10_explicit_full_budget_fastchem4_screen

echo "== curated10: API default depleted-gas with native activity support expansion vs FastChem4 =="
date
python benchmarks/fastchem4/fastchem4_vmap_cold_rescue_compare.py \
  --families $ALL_FAMILIES \
  --exogibbs-method auto \
  --fastchem-condensation equilibrium \
  --no-explicit-gas-init \
  --enable-native-activity-support-expansion \
  --native-activity-support-topk 24 \
  --native-activity-max-support-count 32 \
  --native-activity-threshold 0.0 \
  --gas-floor-sweep 1e-300 1e-30 1e-20 \
  --major-gas-thresholds 1e-8 1e-6 1e-4 \
  --output-dir $OUTDIR \
  --output-prefix gpu_curated10_api_default_depleted_native_activity_expanded_fastchem4_screen

foreach SEED_FRACTION ( 0.5 0.7 0.8 )
  echo "== curated10: API native activity support expansion, seed_fraction=$SEED_FRACTION =="
  date
  python benchmarks/fastchem4/fastchem4_vmap_cold_rescue_compare.py \
    --families $ALL_FAMILIES \
    --exogibbs-method auto \
    --fastchem-condensation equilibrium \
    --no-explicit-gas-init \
    --enable-native-activity-support-expansion \
    --native-activity-support-topk 24 \
    --native-activity-max-support-count 32 \
    --native-activity-threshold 0.0 \
    --seed-fraction $SEED_FRACTION \
    --max-seed-amount 1.0 \
    --gas-floor-sweep 1e-300 1e-30 1e-20 \
    --major-gas-thresholds 1e-8 1e-6 1e-4 \
    --output-dir $OUTDIR \
    --output-prefix gpu_curated10_api_default_depleted_native_activity_expanded_seed${SEED_FRACTION}_fastchem4_screen
end

echo "== outputs =="
date
ls -lh \
  $OUTDIR/gpu_curated10_api_default_depleted_fastchem4_screen.md \
  $OUTDIR/gpu_curated10_api_default_depleted_fastchem4_screen.json \
  $OUTDIR/gpu_curated10_explicit_full_budget_fastchem4_screen.md \
  $OUTDIR/gpu_curated10_explicit_full_budget_fastchem4_screen.json \
  $OUTDIR/gpu_curated10_api_default_depleted_native_activity_expanded_fastchem4_screen.md \
  $OUTDIR/gpu_curated10_api_default_depleted_native_activity_expanded_fastchem4_screen.json \
  $OUTDIR/gpu_curated10_api_default_depleted_native_activity_expanded_seed0.5_fastchem4_screen.md \
  $OUTDIR/gpu_curated10_api_default_depleted_native_activity_expanded_seed0.5_fastchem4_screen.json \
  $OUTDIR/gpu_curated10_api_default_depleted_native_activity_expanded_seed0.7_fastchem4_screen.md \
  $OUTDIR/gpu_curated10_api_default_depleted_native_activity_expanded_seed0.7_fastchem4_screen.json \
  $OUTDIR/gpu_curated10_api_default_depleted_native_activity_expanded_seed0.8_fastchem4_screen.md \
  $OUTDIR/gpu_curated10_api_default_depleted_native_activity_expanded_seed0.8_fastchem4_screen.json

echo "== done =="
date
