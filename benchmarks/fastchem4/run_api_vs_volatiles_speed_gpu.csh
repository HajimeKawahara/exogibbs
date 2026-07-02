#!/bin/csh -f

cd /home/kawahara/exogibbs

setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cuda,cpu
setenv XLA_PYTHON_CLIENT_PREALLOCATE false

set OUTDIR = volatiles_artifacts/fastchem4_major_species_investigation
mkdir -p $OUTDIR

set COND_FAMILIES = ( \
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

set CASES = ()
foreach FAMILY ( $COND_FAMILIES )
  foreach LAYER ( 0 1 2 3 4 5 6 7 8 )
    set CASES = ( $CASES ${FAMILY}:${LAYER} )
  end
end

set SEEDS = ( \
  1e-6 \
  1e-5 \
  1e-4 \
  1e-3 \
  1e-2 \
  3e-2 \
  1e-1 \
  2e-1 \
  3e-1 \
  4e-1 \
  4.5e-1 \
  5e-1 \
  6e-1 \
  7e-1 \
  8e-1 \
)

echo "== GPU =="
date
nvidia-smi

echo "== speed: API native activity support expansion, curated9/81 layers =="
date
/usr/bin/time \
  -f "elapsed_seconds %e\nuser_seconds %U\nsystem_seconds %S\nmax_rss_kb %M" \
  -o $OUTDIR/gpu_speed_api_native_activity_expanded_curated9.time \
  python benchmarks/fastchem4/fastchem4_vmap_cold_rescue_compare.py \
    --families $COND_FAMILIES \
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
    --output-prefix gpu_speed_api_native_activity_expanded_curated9

echo "== speed: volatiles seed/support sweep, curated9/81 layers =="
date
/usr/bin/time \
  -f "elapsed_seconds %e\nuser_seconds %U\nsystem_seconds %S\nmax_rss_kb %M" \
  -o $OUTDIR/gpu_speed_volatiles_seed_partition_curated9_depleted.time \
  python benchmarks/fastchem4/condensate_seed_partition_sweep.py \
    --cases $CASES \
    --support-modes curated curated_plus_fastchem_active \
    --seed-fractions $SEEDS \
    --include-fastchem-scaled-init \
    --depleted-gas-init \
    --max-inner-iterations 150 \
    --output-dir $OUTDIR \
    --output-prefix gpu_speed_volatiles_seed_partition_curated9_depleted

echo "== timing outputs =="
date
echo "-- API --"
cat $OUTDIR/gpu_speed_api_native_activity_expanded_curated9.time
echo "-- volatiles sweep --"
cat $OUTDIR/gpu_speed_volatiles_seed_partition_curated9_depleted.time

echo "== result outputs =="
ls -lh \
  $OUTDIR/gpu_speed_api_native_activity_expanded_curated9.md \
  $OUTDIR/gpu_speed_api_native_activity_expanded_curated9.json \
  $OUTDIR/gpu_speed_api_native_activity_expanded_curated9.time \
  $OUTDIR/gpu_speed_volatiles_seed_partition_curated9_depleted.md \
  $OUTDIR/gpu_speed_volatiles_seed_partition_curated9_depleted.json \
  $OUTDIR/gpu_speed_volatiles_seed_partition_curated9_depleted.time

echo "== done =="
date
