# Complete-domain second-liquid bound at the saved central source

`central/` is a fresh independent property assessment of the accepted
16-layer H=1e24 mol source at 2173.15 K and 267.20283416157343 bar.
It does not solve planetary pressure again. The original input bytes,
native receipts, requests that failed, parameter identity, source hashes,
and every retained box bound are preserved.

| Quantity | Result |
| --- | --- |
| Complete element-supported liquid domain | All 12 BSE-supported components |
| Boxes evaluated | 4715 |
| Retained proof leaves | 2358 |
| Verified tangent-plane lower bound | **0 RT** |
| Formal second-liquid bound | Accepted, including every ideal dissolved-H2 fraction |
| Elapsed fixed-source assessment | 151.378 s |
| Native composition requests | 14 attempted; 13 finite, 1 unavailable |
| Maximum finite native potential discrepancy | 3.185e-9 RT |
| Actual process exit code | 0 |

The proof bounds the declared ExoEOS expression with outward interval
arithmetic; finite native agreement is separate evidence. It establishes
neither a uniform native binary error bound nor empirical BSE applicability.
Competing solids and the Fe-Si-O-H source alloy require separate assessments.

The original fixed-source attempt stopped at a native 98% vertex request
whose oxide inversion did not preserve the requested composition. Its
partial files and exit-1 log remain under `failed_original_vertex_probe/`.
The corrected runner preserves that unavailable request and tries the
separately declared 50% request. It does not clip a failed composition.

The model was fixed at Gibbs `237050a` and EOS `d797ab6`. The successful
execution used Gibbs `5ae43b9`, which changes only the runner's handling of
unavailable native probes. Each execution retains its own exact protocol.
The mathematical verifier and provider expressions did not change between
these executions.

Full unit tests completed with **1876 passed, 1 skipped, 1 failed** in
1898.97 s. The one failure identified the new runner's missing registration
in the all-example execution manifest. After registering its actual input,
native resources and output artifacts, all **36 targeted tests passed** in
2.90 s, including that manifest, global bounds and coupled callbacks. The
single full-suite skip is the unavailable setuptools-scm version module.
Both original and final logs are retained; the full suite is not relabeled
as a clean run.

See the [algorithm and API](../../../examples/metal_silicate/m2_liquid_global.md)
and [ExoEOS published provider PR](https://github.com/HajimeKawahara/exoeos/pull/35).
`manifest.json` pins archived files and records observed process exit codes.
