# FastChem <-> ExoGibbs Variable-Mapping Micro-Parity Audit

Decision: **continue reduced/full reconstruction alignment next**

Inferred mapping: `activity_correction~lambda`
maxDensity interpretation: `C: formulas are effectively equivalent after normalization for these traces`

## Layer 0 epsilon 0.0

- Dominant mismatch: reduced/full split or reconstruction mismatch
- Top mismatch stage: post_gas_only_activity_maxdensity_scan
- Mapping: activity_correction~lambda
- maxDensity: not observable for this trace

### A post gas-only activity/maxDensity scan

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Al(s) | 0 | -10 | 2.47e+15 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| AlClO(s) | 1 | -10 | 1.87e+14 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| AlCl3(s,l) | 2 | -10 | 6.24e+13 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| KAlCl4(s) | 3 | -10 | 4.68e+13 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| NaAlCl4(s) | 4 | -10 | 4.68e+13 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| K3AlCl6(s) | 5 | -10 | 3.12e+13 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| Na3AlCl6(s) | 6 | -10 | 3.12e+13 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| AlF3(s,l) | 7 | -10 | 7.68e+12 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |

### B post selectActiveCondensates reset

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|

### C post calculate() entry seeding

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|

### D post first correctValues / correctValuesFull update

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|

### E post final removal

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|

### Candidate Species Examples

- Al(s) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=None, lambda_EG=None
- AlClO(s) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=None, lambda_EG=None
- AlF3(s,l) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=None, lambda_EG=None
- Na3AlF6(s,l) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=None, lambda_EG=None

## Layer 45 epsilon -10.0

- Dominant mismatch: cap timing mismatch
- Top mismatch stage: after_first_correctValues_update
- Mapping: activity_correction~lambda
- maxDensity: C: formulas are effectively equivalent after normalization for these traces

### A post gas-only activity/maxDensity scan

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Al(s) | 0 | -1.09 | 1.22e+13 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| AlClO(s) | 1 | 38 | 9.27e+11 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | False | False |
| AlCl3(s,l) | 2 | -10 | 3.09e+11 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| KAlCl4(s) | 3 | -10 | 2.32e+11 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| NaAlCl4(s) | 4 | -10 | 2.32e+11 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| K3AlCl6(s) | 5 | -10 | 1.54e+11 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| Na3AlCl6(s) | 6 | -10 | 1.54e+11 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| AlF3(s,l) | 7 | -2.29 | 3.8e+10 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | False | False |

### B post selectActiveCondensates reset

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| AlClO(s) | 1 | 38 | 9.27e+11 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| K3AlF6(s) | 8 | 12 | 1.9e+10 | 0 | NA | 0 | NA | NA | NA | False | False | True | False |
| Na3AlF6(s,l) | 9 | 28.7 | 1.9e+10 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| AlN(s) | 10 | 52.4 | 1.22e+13 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| NaAlO2(s) | 11 | 107 | 7.53e+12 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| MgAl2O4(s,l) | 13 | 231 | 6.11e+12 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| Al2O3(s,l) | 14 | 171 | 6.11e+12 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| Al2SiO5(s) | 15 | 253 | 6.11e+12 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |

### C post calculate() entry seeding

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| MgCO3(s) | 23 | 22.7 | 1.61e+14 | 1.61e+14 | 8.225768719393136e-69 | 1 | 1.0080554518261695 | NA | NA | False | False | True | False |
| SiC(s) | 26 | 12.8 | 1.47e+14 | 1.47e+14 | 8.225768719393136e-69 | 1 | 1.0088003788821167 | NA | NA | False | False | True | False |
| Al4C3(s) | 30 | 2.62 | 3.05e+12 | 3.05e+12 | 2.741922906464379e-69 | 1 | 1.0093109384088403 | NA | NA | False | False | True | False |
| NaCO3(s,l) | 25 | 19.6 | 3.77e+12 | 3.77e+12 | 8.225768719393136e-69 | 1 | 1.0095676370113638 | NA | NA | False | False | True | False |
| Cr3C2(s) | 28 | 165 | 6.31e+11 | 6.31e+11 | 4.112884359696568e-69 | 1 | 0.9988848303209493 | NA | NA | False | False | True | False |
| Cr7C3(s) | 31 | 399 | 2.7e+11 | 2.7e+11 | 2.741922906464379e-69 | 1 | 0.9837721477467067 | NA | NA | False | False | True | False |
| Cr23C6(s) | 35 | 1.31e+03 | 8.23e+10 | 8.23e+10 | 1.3709614532321894e-69 | 1 | 0.9270078717162621 | NA | NA | False | False | True | False |
| TiC(s,l) | 27 | 52.2 | 4.24e+11 | 4.24e+11 | 8.225768719393136e-69 | 1 | 1.0061372061373384 | NA | NA | False | False | True | False |

### D post first correctValues / correctValuesFull update

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| NaCO3(s,l) | 25 | 19.6 | 3.77e+12 | 1.39e+12 | 8.225768719393136e-69 | 0.00674 | 149.83312233657983 | 0.3680778299381864 | 148.4131591025766 | False | True | False | False |
| MgCO3(s) | 23 | 22.7 | 1.61e+14 | 6.98e+13 | 8.225768719393136e-69 | 0.00674 | 0.006792224206543883 | 0.4335908294936599 | 148.4131591025766 | False | True | False | False |
| SiC(s) | 26 | 12.8 | 1.47e+14 | 5.16e+13 | 8.225768719393136e-69 | 0.00674 | 0.00679724348556504 | 0.35165590870713836 | 148.4131591025766 | False | True | False | False |
| Al4C3(s) | 30 | 2.62 | 3.05e+12 | 2.06e+10 | 2.741922906464379e-69 | 0.00674 | 0.006800683608595983 | 0.006737972291985239 | 148.4131591025766 | False | True | False | False |
| Cr3C2(s) | 28 | 165 | 6.31e+11 | 5.16e+11 | 4.112884359696568e-69 | 0.00674 | 0.01803893458166826 | 0.8180193664902954 | 148.4131591025766 | False | True | False | False |
| Cr7C3(s) | 31 | 399 | 2.7e+11 | 2.7e+11 | 2.741922906464379e-69 | 0.00674 | 0.006628604590693787 | 1.0 | 148.4131591025766 | True | True | False | False |
| Cr23C6(s) | 35 | 1.31e+03 | 8.23e+10 | 8.23e+10 | 1.3709614532321894e-69 | 0.00674 | 0.006246129907359193 | 1.0 | 148.4131591025766 | True | True | False | False |
| TiC(s,l) | 27 | 52.2 | 4.24e+11 | 2.02e+11 | 8.225768719393136e-69 | 0.00674 | 0.006779299168761315 | 0.4778467938230246 | 148.4131591025766 | False | True | False | False |

### E post final removal

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Ti4O7(s,l) | 158 | -10 | 1.06e+11 | 0 | 1.0251201298696579e-47 | 10 | 4.448684883725723e-05 | NA | NA | False | False | False | True |
| Ca2Al2SiO7(s) | 175 | -10 | 4.53e+12 | 0 | 1.0251201298696579e-47 | 10 | 4.452122299504934e-05 | NA | NA | False | False | False | True |
| CaAl2Si2O8(s) | 176 | -10 | 6.11e+12 | 0 | 8.969801136359506e-48 | 10 | 4.45384954789593e-05 | NA | NA | False | False | False | True |
| Ti3O5(s,l) | 156 | -10 | 1.41e+11 | 0 | 1.435168181817521e-47 | 10 | 4.480191704869817e-05 | NA | NA | False | False | False | True |
| Ca2SiO4(s) | 172 | -10 | 4.53e+12 | 0 | 1.7939602272719012e-47 | 10 | 4.5053382415435565e-05 | NA | NA | False | False | False | True |
| Ti2O3(s,l) | 152 | -10 | 2.12e+11 | 0 | 2.3919469696958683e-47 | 10 | 4.51111177518184e-05 | NA | NA | False | False | False | True |
| Si3N4(s) | 132 | -10 | 4.9e+13 | 0 | 2.0104171883206375e-47 | 10 | 4.527499659557567e-05 | NA | NA | False | False | False | True |
| V2O4(s,l) | 155 | -10 | 1.8e+10 | 0 | 1.7939602272719012e-47 | 10 | 4.528904539000556e-05 | NA | NA | False | False | False | True |

### Candidate Species Examples

- Al(s) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=None, lambda_EG=None
- AlClO(s) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=0.0, lambda_EG=0.0
- AlF3(s,l) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=0.0, lambda_EG=0.0
- Na3AlF6(s,l) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=0.0, lambda_EG=0.0
- AlClO(s) at post_selectActiveCondensates_reset: n_FC=0, n_EG=0.0, lambda_EG=0.0
- Na3AlF6(s,l) at post_selectActiveCondensates_reset: n_FC=0, n_EG=0.0, lambda_EG=0.0
- Al2O3(s,l) at post_selectActiveCondensates_reset: n_FC=0, n_EG=0.0, lambda_EG=0.0
- MgCO3(s) at post_calculate_entry_seeding: n_FC=1.61e+14, n_EG=8.225768719393136e-69, lambda_EG=1.0080554518261695
- Na3AlF6(s,l) at post_calculate_entry_seeding: n_FC=1.9e+10, n_EG=3.0439250642540553e-51, lambda_EG=1.006303723245405
- MgCO3(s) at after_first_correctValues_update: n_FC=6.98e+13, n_EG=8.225768719393136e-69, lambda_EG=0.006792224206543883
- Na3AlF6(s,l) at after_first_correctValues_update: n_FC=1.9e+10, n_EG=2.050980575213165e-53, lambda_EG=149.34871458353552
- NaOH(s,l) at after_first_correctValues_update: n_FC=2.96e+12, n_EG=7.175840909087605e-47, lambda_EG=149.33003870912376

## Layer 90 epsilon -5.0

- Dominant mismatch: cap timing mismatch
- Top mismatch stage: after_first_correctValues_update
- Mapping: activity_correction~lambda
- maxDensity: C: formulas are effectively equivalent after normalization for these traces

### A post gas-only activity/maxDensity scan

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Al(s) | 0 | 3.39 | 2.29e+10 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | False | False |
| AlClO(s) | 1 | 115 | 1.74e+09 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | False | False |
| AlCl3(s,l) | 2 | -10 | 5.8e+08 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| KAlCl4(s) | 3 | 1.88 | 4.35e+08 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| NaAlCl4(s) | 4 | -10 | 4.35e+08 | 0 | NA | 0 | NA | NA | NA | False | False | False | False |
| K3AlCl6(s) | 5 | 82.2 | 2.9e+08 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | False | False |
| Na3AlCl6(s) | 6 | 67 | 2.9e+08 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | False | False |
| AlF3(s,l) | 7 | 34.2 | 7.14e+07 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | False | False |

### B post selectActiveCondensates reset

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Al(s) | 0 | 3.39 | 2.29e+10 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| AlClO(s) | 1 | 115 | 1.74e+09 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| KAlCl4(s) | 3 | 1.88 | 4.35e+08 | 0 | NA | 0 | NA | NA | NA | False | False | True | False |
| K3AlCl6(s) | 5 | 82.2 | 2.9e+08 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| Na3AlCl6(s) | 6 | 67 | 2.9e+08 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| AlF3(s,l) | 7 | 34.2 | 7.14e+07 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| K3AlF6(s) | 8 | 92.2 | 3.57e+07 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |
| Na3AlF6(s,l) | 9 | 123 | 3.57e+07 | 0 | 0.0 | 0 | 0.0 | NA | NA | False | False | True | False |

### C post calculate() entry seeding

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| MgCO3(s) | 23 | 80.9 | 3.02e+11 | 3.02e+11 | 4.971466540502857e-145 | 1 | 1.0431392511868367 | NA | NA | False | False | True | False |
| SiC(s) | 26 | 33 | 2.76e+11 | 2.76e+11 | 4.971466540502857e-145 | 1 | 1.0531610020976663 | NA | NA | False | False | True | False |
| Al4C3(s) | 30 | 37.1 | 5.73e+09 | 5.73e+09 | 1.6571555135009524e-145 | 1 | 1.0481536996343124 | NA | NA | False | False | True | False |
| NaCO3(s,l) | 25 | 73.6 | 7.07e+09 | 7.07e+09 | 4.971466540502857e-145 | 1 | 1.0449038181037498 | NA | NA | False | False | True | False |
| Cr3C2(s) | 28 | 422 | 1.18e+09 | 1.18e+09 | 2.4857332702514286e-145 | 1 | 0.9839106438100564 | NA | NA | False | False | True | False |
| Cr7C3(s) | 31 | 1.01e+03 | 5.08e+08 | 5.08e+08 | 1.6571555135009524e-145 | 1 | 0.8867414298497768 | NA | NA | False | False | True | False |
| Cr23C6(s) | 35 | 3.31e+03 | 1.54e+08 | 1.54e+08 | 8.285777567504762e-146 | 1 | 0.5904158262395325 | NA | NA | False | False | True | False |
| TiC(s,l) | 27 | 119 | 7.95e+08 | 7.95e+08 | 4.971466540502857e-145 | 1 | 1.035973494456167 | NA | NA | False | False | True | False |

### D post first correctValues / correctValuesFull update

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| MgCO3(s) | 23 | 80.9 | 3.02e+11 | 2.02e+11 | 4.971466540502857e-145 | 0.00674 | 0.007028616987162607 | 0.6667592620150732 | 148.4131591025766 | False | True | False | False |
| SiC(s) | 26 | 33 | 2.76e+11 | 1.08e+11 | 4.971466540502857e-145 | 0.00674 | 0.007096143013637814 | 0.393112000696197 | 148.4131591025766 | False | True | False | False |
| Al4C3(s) | 30 | 37.1 | 5.73e+09 | 3.86e+07 | 1.6571555135009524e-145 | 0.00674 | 0.007062404075031345 | 0.006737946560133364 | 148.4131591025766 | False | True | False | False |
| NaCO3(s,l) | 25 | 73.6 | 7.07e+09 | 3.8e+09 | 4.971466540502857e-145 | 0.00674 | 0.007040506545525107 | 0.5368031968597019 | 148.4131591025766 | False | True | False | False |
| Cr3C2(s) | 28 | 422 | 1.18e+09 | 1.1e+09 | 2.4857332702514286e-145 | 0.00674 | 0.00662953776982822 | 0.9304966814719741 | 148.4131591025766 | False | True | False | False |
| Cr7C3(s) | 31 | 1.01e+03 | 5.08e+08 | 5.08e+08 | 1.6571555135009524e-145 | 0.00674 | 0.00597481675622106 | 1.0 | 148.4131591025766 | True | True | False | False |
| Cr23C6(s) | 35 | 3.31e+03 | 1.54e+08 | 1.54e+08 | 8.285777567504762e-146 | 0.00674 | 0.0039781905446232245 | 1.0 | 148.4131591025766 | True | True | False | False |
| TiC(s,l) | 27 | 119 | 7.95e+08 | 4.36e+08 | 4.971466540502857e-145 | 0.00674 | 0.006980334498103015 | 0.5484040092133453 | 148.4131591025766 | False | True | False | False |

### E post final removal

| condensate | idx | log_activity | maxDensity | n_FC | n_EG | actcorr_FC | lambda_EG | upd_FC | upd_EG | cap_FC | cap_EG | new | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Mg2TiO4(s,l) | 124 | -10 | 7.95e+08 | 0 | 9.007822426491182e-98 | 10 | 4.294323563735367e-05 | NA | NA | False | False | False | True |
| CaTiO3(s) | 168 | -10 | 7.95e+08 | 0 | 1.2010429901988243e-97 | 10 | 4.377942347590687e-05 | NA | NA | False | False | False | True |
| Si3N4(s) | 132 | -10 | 9.19e+10 | 0 | 3.550852268241e-96 | 10 | 4.414245800483459e-05 | NA | NA | False | False | False | True |
| Cr3C2(s) | 28 | -10 | 1.18e+09 | 0 | 0.0 | 10 | 4.4669474121537815e-05 | NA | NA | False | False | False | True |
| Mn(s,l) | 128 | -10 | 2.24e+09 | 0 | 1.492181975910265e-11 | 10 | 4.481387895131678e-05 | NA | NA | False | False | False | True |
| CaS(s) | 41 | -10 | 1.7e+10 | 0 | 2.1996259048895823e-57 | 10 | 4.517593477441722e-05 | NA | NA | False | False | False | True |
| TiN(s,l) | 129 | -10 | 7.95e+08 | 0 | 1.4203409072964e-95 | 10 | 4.545650907396432e-05 | NA | NA | False | False | False | True |
| Na2S(s,l) | 141 | -10 | 7.07e+09 | 0 | 2.1996259048895823e-57 | 10 | 4.5584169366882406e-05 | NA | NA | False | False | False | True |

### Candidate Species Examples

- Al(s) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=0.0, lambda_EG=0.0
- AlClO(s) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=0.0, lambda_EG=0.0
- AlF3(s,l) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=0.0, lambda_EG=0.0
- Na3AlF6(s,l) at post_gas_only_activity_maxdensity_scan: n_FC=0, n_EG=0.0, lambda_EG=0.0
- Al(s) at post_selectActiveCondensates_reset: n_FC=0, n_EG=0.0, lambda_EG=0.0
- AlClO(s) at post_selectActiveCondensates_reset: n_FC=0, n_EG=0.0, lambda_EG=0.0
- AlF3(s,l) at post_selectActiveCondensates_reset: n_FC=0, n_EG=0.0, lambda_EG=0.0
- Na3AlF6(s,l) at post_selectActiveCondensates_reset: n_FC=0, n_EG=0.0, lambda_EG=0.0
- MgCO3(s) at post_calculate_entry_seeding: n_FC=3.02e+11, n_EG=4.971466540502857e-145, lambda_EG=1.0431392511868367
- MgCO3(s) at after_first_correctValues_update: n_FC=2.02e+11, n_EG=4.971466540502857e-145, lambda_EG=0.007028616987162607

