# Selected Strict1000 Continuation Diagnostic

| label | family | variant | layer | success | n_iter | residual | stop | dominant | final centered | support |
|---|---|---|---:|---|---:|---:|---|---|---|---:|
| strict_success_good_gas | solar_highT_no_condensate_gas_regression | activity_capacity_tm1ep00_cap8 | 16 | True | 398 | 3.909e-06 | converged | condensate_stationarity | True | 2 |
| highT_good_gas_bad_kkt | solar_highT_no_condensate_gas_regression | activity_driving_top_cap96 | 13 | False | 842 | 0.009532 | no_accepted_trial | complementarity | False | 96 |
| highT_compact_plateau | solar_highT_no_condensate_gas_regression | capacity_top_cap32 | 13 | False | 1000 | 79 | max_iter_tiny_step | condensate_stationarity | False | 32 |
| silicate_good_gas_bad_kkt | solar_silicate_first_condensation | activity_capacity_t0ep00_cap48 | 6 | False | 157 | 3.009e-05 | no_accepted_trial | complementarity | False | 48 |
| silicate_bad_gas_bad_kkt | solar_silicate_first_condensation | capacity_top_cap48 | 5 | False | 1000 | 41.51 | max_iter_tiny_step | condensate_stationarity | False | 48 |
| strict_success_bad_gas | solar_silicate_first_condensation | curated_base | 1 | True | 9 | 7.404e-06 | converged | condensate_stationarity | True | 3 |
| water_early_stop | solar_water_condensation | activity_capacity_t0ep00_cap128 | 8 | False | 174 | 8.847e-11 | no_accepted_trial | budget | False | 128 |
| water_plateau | solar_water_condensation | activity_capacity_t0ep00_cap64 | 7 | True | 305 | 6.198e-07 | converged | condensate_stationarity | True | 64 |
