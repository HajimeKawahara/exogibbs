#!/bin/csh -f

cd /home/kawahara/exogibbs || exit 1

set mode = "a100_smoke"
if ( $#argv >= 1 ) then
    set mode = "$argv[1]"
endif

setenv PYTHONPATH src:volatiles_code
setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cuda
setenv JAX_PLATFORM_NAME cuda

set stamp = `date +%Y%m%d_%H%M%S`
set iterations = 40
set warmup = 1
set repeat = 3
set families = "--families carbon_rich_graphite_window solar_silicate_first_condensation"
set budget_gate = "--disable-budget-gate"
set label = "a100_smoke"

if ( "$mode" == "a100_quick100" ) then
    set iterations = 100
    set warmup = 1
    set repeat = 3
    set families = "--families carbon_rich_graphite_window solar_silicate_first_condensation"
    set budget_gate = "--disable-budget-gate"
    set label = "a100_quick100"
endif

if ( "$mode" == "a100_broad100" ) then
    set iterations = 100
    set warmup = 1
    set repeat = 3
    set families = "--families carbon_rich_graphite_window solar_silicate_first_condensation solar_water_condensation solar_metal_sulfide_or_Fe_Ni_S_region"
    set budget_gate = "--disable-budget-gate"
    set label = "a100_broad100"
endif

set output = "volatiles_artifacts/condensate_profile_auto_rescue_cuda_${label}_${stamp}.json"

python volatiles_code/benchmark_condensate_profile_auto_rescue.py \
    --jax-platform cuda \
    --print-jax-devices \
    --iterations $iterations \
    --warmup $warmup \
    --repeat $repeat \
    $families \
    $budget_gate \
    --output $output
