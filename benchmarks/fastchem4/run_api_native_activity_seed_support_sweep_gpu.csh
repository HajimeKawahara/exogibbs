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

set SEED_FRACTIONS = ( \
  0.50 \
  0.60 \
  0.70 \
  0.80 \
  0.85 \
  0.90 \
  0.95 \
  1.00 \
)

set SUPPORT_TOPKS = ( \
  8 \
  12 \
  16 \
  24 \
  32 \
  48 \
)

echo "== GPU =="
date
nvidia-smi

echo "== API native activity support/seed sweep: 8 seed fractions x 6 top-k = 48 runs =="
date

foreach TOPK ( $SUPPORT_TOPKS )
  @ MAX_SUPPORT = $TOPK + 8
  foreach SEED_FRACTION ( $SEED_FRACTIONS )
    set SEED_LABEL = `echo $SEED_FRACTION | sed 's/\\.//g'`
    set PREFIX = gpu_api_native_activity_seed_support_sweep_top${TOPK}_seed${SEED_LABEL}
    echo "== run $PREFIX =="
    date
    python benchmarks/fastchem4/fastchem4_vmap_cold_rescue_compare.py \
      --families $COND_FAMILIES \
      --exogibbs-method auto \
      --fastchem-condensation equilibrium \
      --no-explicit-gas-init \
      --enable-native-activity-support-expansion \
      --native-activity-support-topk $TOPK \
      --native-activity-max-support-count $MAX_SUPPORT \
      --native-activity-threshold 0.0 \
      --seed-fraction $SEED_FRACTION \
      --max-seed-amount 1.0 \
      --gas-floor-sweep 1e-300 1e-30 1e-20 \
      --major-gas-thresholds 1e-8 1e-6 1e-4 \
      --output-dir $OUTDIR \
      --output-prefix $PREFIX
    if ( $status != 0 ) then
      echo "ERROR: $PREFIX failed"
      exit 1
    endif
  end
end

echo "== outputs =="
date
ls -lh $OUTDIR/gpu_api_native_activity_seed_support_sweep_top*_seed*.md
ls -lh $OUTDIR/gpu_api_native_activity_seed_support_sweep_top*_seed*.json

echo "== done =="
date
