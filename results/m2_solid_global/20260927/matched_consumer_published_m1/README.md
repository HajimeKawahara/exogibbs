# Formal comparison using the fixed column consumer

After the finite-JANAF column-reference correction, the 35-gas source was
closed again with the same final consumer used for expanded scenarios.
This is a new saved root and fresh native phase assessment, kept separately
from the preceding closed root and its offline replay.

The source uses Inventory `4b5255f`, Gibbs `7cf41f1`, EOS `ee21fa1`;
T=2173.15 K and P=267.2028341602758 bar. Its accepted physical audit and
full hash are copied unchanged in both run directories. The proof code
is Gibbs `1af5821`, EOS `113ae4d`. Both actual processes exited 0.

All 20 competing solution phases have strictly positive global insertion
lower bounds; the minimum is orthopyroxene
**0.000370622565611766 RT/formula unit**. The full second-liquid/H2 bound
is **0 RT**, with 4719 nodes. Solid and liquid runs took 253.711 s and
151.469 s respectively. No proof runner solved a new pressure root.

| Phase | Lower bound (RT/formula unit) | Nodes |
| --- | ---: | ---: |
| hornblende | 14.97826319942905 | 1 |
| biotite | 13.35339662045244 | 1 |
| garnet | 1.140485730436018 | 3 |
| alkali-feldspar | 2.429315922798057 | 3 |
| kalsilite | 8.405848619328255 | 1 |
| leucite | 1.177473932506022 | 1 |
| alloy-solid | 0.2306027839422533 | 1 |
| alloy-liquid | 0.04536556585173135 | 1 |
| cummingtonite | 9.778603466507164 | 1 |
| olivine | 0.001562348662674533 | 9 |
| nepheline | 4.667491913622348 | 1 |
| clinoamphibole | 3.933737635185373 | 1 |
| orthoamphibole | 3.564008628189747 | 1 |
| melilite | 1.903911992506217 | 1 |
| ortho-oxide | 0.9416522933615805 | 3 |
| rhm-oxide | 0.09796099625620645 | 59 |
| plagioclase | 1.931935891315485 | 1 |
| clinopyroxene | 0.007455451042878329 | 515 |
| orthopyroxene | 0.000370622565611766 | 731 |
| spinel | 0.01028145072060386 | 313 |

Every domain and uncertainty distinction in the parent README still applies.
The native Fe/Ni alloy candidates do not represent the source Fe/Si/O/H alloy.
The 76-gas, 67-condensate and alternative-standard roots require separate
assessments; this 35-gas result is not a substitute for them.

`execution_receipt.json` records exact input/code hashes and actual outcomes;
`solids` and `liquid` retain complete native receipts and proof partitions.
