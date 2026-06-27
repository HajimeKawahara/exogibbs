#!/bin/csh -f

cd /home/kawahara/exogibbs || exit 1

set mode = "default"
if ( $#argv >= 1 ) then
    set mode = "$argv[1]"
endif
set custom_batch_size = ""
if ( $#argv >= 2 ) then
    set custom_batch_size = "$argv[2]"
endif

setenv PYTHONPATH src:volatiles_code
setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cuda
setenv JAX_PLATFORM_NAME cuda

set stamp = `date +%Y%m%d_%H%M%S`
set iterations = 8
set warmup = 2
set repeat = 3
set families = ""
set baseline = ""
set budget_gate = ""
set block_output = "--block-output layers"
set prepared_plan = ""
set element_inventory_scale = ""
set element_inventory_batch_size = ""
set element_inventory_batch_mode = ""
set support_source = ""
set rho_initialization = ""
set lambda_initialization = ""
set residual_tolerance_multiplier = ""
set support_candidate_mode = ""
set support_candidate_prune_floors = ""
set support_candidate_neighbor_union = ""
set label = "default"

if ( "$mode" == "quick" ) then
    set iterations = 8
    set warmup = 0
    set repeat = 2
    set families = "--families solar_metal_sulfide_or_Fe_Ni_S_region solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set label = "quick"
endif

if ( "$mode" == "quick16" ) then
    set iterations = 16
    set warmup = 0
    set repeat = 2
    set families = "--families solar_metal_sulfide_or_Fe_Ni_S_region solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set label = "quick16"
endif

if ( "$mode" == "quick32" ) then
    set iterations = 32
    set warmup = 0
    set repeat = 2
    set families = "--families solar_metal_sulfide_or_Fe_Ni_S_region solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set label = "quick32"
endif

if ( "$mode" == "quick64" ) then
    set iterations = 64
    set warmup = 0
    set repeat = 2
    set families = "--families solar_metal_sulfide_or_Fe_Ni_S_region solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set label = "quick64"
endif

if ( "$mode" == "quick_diag" ) then
    set iterations = 8
    set warmup = 0
    set repeat = 2
    set families = "--families solar_metal_sulfide_or_Fe_Ni_S_region solar_water_condensation"
    set baseline = "--skip-baseline --return-diagnostics"
    set label = "quick_diag"
endif

if ( "$mode" == "all_prepared" ) then
    set iterations = 8
    set warmup = 1
    set repeat = 3
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set label = "all_prepared"
endif

if ( "$mode" == "all_prepared16" ) then
    set iterations = 16
    set warmup = 1
    set repeat = 3
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set label = "all_prepared16"
endif

if ( "$mode" == "all_prepared64" ) then
    set iterations = 64
    set warmup = 1
    set repeat = 3
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set label = "all_prepared64"
endif

if ( "$mode" == "all_prepared16_bscale" ) then
    set iterations = 16
    set warmup = 1
    set repeat = 3
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set label = "all_prepared16_bscale"
endif

if ( "$mode" == "all_prepared16_bmany4" ) then
    set iterations = 16
    set warmup = 1
    set repeat = 3
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set element_inventory_batch_size = "--element-inventory-batch-size 4"
    set label = "all_prepared16_bmany4"
endif

if ( "$mode" == "all_prepared16_bmany8" ) then
    set iterations = 16
    set warmup = 1
    set repeat = 3
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set element_inventory_batch_size = "--element-inventory-batch-size 8"
    set label = "all_prepared16_bmany8"
endif

if ( "$mode" == "all_prepared16_bmany16" ) then
    set iterations = 16
    set warmup = 1
    set repeat = 3
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set element_inventory_batch_size = "--element-inventory-batch-size 16"
    set label = "all_prepared16_bmany16"
endif

if ( "$mode" == "all_prepared16_bmany" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 all_prepared16_bmany N_EVAL"
        exit 2
    endif
    set iterations = 16
    set warmup = 1
    set repeat = 3
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set label = "all_prepared16_bmany${custom_batch_size}"
endif

if ( "$mode" == "activity_water16" || "$mode" == "activity_water100" || "$mode" == "a100_water100" ) then
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set families = "--families solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "activity_water100"
    if ( "$mode" == "a100_water100" ) then
        set label = "a100_water100"
    endif
endif

if ( "$mode" == "activity_quick16" || "$mode" == "activity_quick100" || "$mode" == "a100_quick100" ) then
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set families = "--families solar_metal_sulfide_or_Fe_Ni_S_region solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "activity_quick100"
    if ( "$mode" == "a100_quick100" ) then
        set label = "a100_quick100"
    endif
endif

if ( "$mode" == "activity_water16_bmany" || "$mode" == "activity_water100_bmany" || "$mode" == "a100_water100_bmany" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_water100_bmany N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set families = "--families solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "activity_water100_bmany${custom_batch_size}"
    if ( "$mode" == "a100_water100_bmany" ) then
        set label = "a100_water100_bmany${custom_batch_size}"
    endif
endif

if ( "$mode" == "a100_water100_brepeat" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_water100_brepeat N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set families = "--families solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set element_inventory_batch_mode = "--element-inventory-batch-mode repeat"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "a100_water100_brepeat${custom_batch_size}"
endif

if ( "$mode" == "activity_quick100_bmany" || "$mode" == "a100_quick100_bmany" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_quick100_bmany N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set families = "--families solar_metal_sulfide_or_Fe_Ni_S_region solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "activity_quick100_bmany${custom_batch_size}"
    if ( "$mode" == "a100_quick100_bmany" ) then
        set label = "a100_quick100_bmany${custom_batch_size}"
    endif
endif

if ( "$mode" == "a100_quick100_brepeat" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_quick100_brepeat N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set families = "--families solar_metal_sulfide_or_Fe_Ni_S_region solar_water_condensation"
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set element_inventory_batch_mode = "--element-inventory-batch-mode repeat"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "a100_quick100_brepeat${custom_batch_size}"
endif

if ( "$mode" == "a100_broad100" ) then
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "a100_broad100"
endif

if ( "$mode" == "a100_broad100_brepeat" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_broad100_brepeat N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set element_inventory_batch_mode = "--element-inventory-batch-mode repeat"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "a100_broad100_brepeat${custom_batch_size}"
endif

if ( "$mode" == "a100_broad100_bmany" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_broad100_bmany N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set label = "a100_broad100_bmany${custom_batch_size}"
endif

if ( "$mode" == "a100_broad100_brepeat_candidates" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_broad100_brepeat_candidates N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set element_inventory_batch_mode = "--element-inventory-batch-mode repeat"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set support_candidate_mode = "--support-candidate-mode current_prune_neighbor"
    set label = "a100_broad100_brepeat${custom_batch_size}_candidates"
endif

if ( "$mode" == "a100_broad100_brepeat_rescue_candidates" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_broad100_brepeat_rescue_candidates N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set element_inventory_batch_mode = "--element-inventory-batch-mode repeat"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set support_candidate_mode = "--support-candidate-mode fallback_rescue_prune_neighbor"
    set label = "a100_broad100_brepeat${custom_batch_size}_rescue_candidates"
endif

if ( "$mode" == "a100_broad100_brepeat_rescue_slim" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_broad100_brepeat_rescue_slim N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set element_inventory_batch_mode = "--element-inventory-batch-mode repeat"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set support_candidate_mode = "--support-candidate-mode fallback_rescue_prune_neighbor"
    set support_candidate_prune_floors = "--support-candidate-prune-floors 1e-5,1e-3"
    set support_candidate_neighbor_union = "--disable-support-candidate-neighbor-union"
    set label = "a100_broad100_brepeat${custom_batch_size}_rescue_slim"
endif

if ( "$mode" == "a100_broad100_bmany_candidates" ) then
    if ( "$custom_batch_size" == "" ) then
        echo "usage: $0 a100_broad100_bmany_candidates N_EVAL"
        exit 2
    endif
    set iterations = 100
    set warmup = 0
    set repeat = 2
    set baseline = "--skip-baseline"
    set budget_gate = "--disable-budget-gate"
    set block_output = "--block-output batched"
    set prepared_plan = "--prepared-plan"
    set element_inventory_scale = "--element-inventory-scale 1.001"
    set element_inventory_batch_size = "--element-inventory-batch-size $custom_batch_size"
    set support_source = "--support-source activity_outer"
    set rho_initialization = "--rho-initialization complementarity"
    set lambda_initialization = "--lambda-initialization best_residual"
    set residual_tolerance_multiplier = "--residual-tolerance-multiplier 1.0e9"
    set support_candidate_mode = "--support-candidate-mode current_prune_neighbor"
    set label = "a100_broad100_bmany${custom_batch_size}_candidates"
endif

set output = "volatiles_artifacts/pdipm_api_profile_fixed_support_batch_cuda_${label}_${stamp}.json"

python volatiles_code/benchmark_pdipm_api_profile_fixed_support_batch.py \
    --jax-platform cuda \
    --print-jax-devices \
    --iterations $iterations \
    --warmup $warmup \
    --repeat $repeat \
    $families \
    $baseline \
    $budget_gate \
    $block_output \
    $prepared_plan \
    $element_inventory_scale \
    $element_inventory_batch_size \
    $element_inventory_batch_mode \
    $support_source \
    $rho_initialization \
    $lambda_initialization \
    $residual_tolerance_multiplier \
    $support_candidate_mode \
    $support_candidate_prune_floors \
    $support_candidate_neighbor_union \
    --output $output
