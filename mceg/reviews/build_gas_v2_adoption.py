"""Reproduce compact curated-gas-v2 inputs from pinned, selected candidate records."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "src/exogibbs/data/curated/gas_v2.json"
BASE = "src/exogibbs/data/curated/ch2_v1.json"
BASE_SHA256 = "a2d353156b957c38b6f3ff44af1bf735d60e58cf1089f99b7b9358e01ad81536"
LOGK = "src/exogibbs/data/FastChem4/logK/logK.dat"
LOGK_SHA256 = "eda9b9d7e62ccf7398036ee84349bd0a721b779acdc9d191cbf9fe1dd01f619c"
R = 8.31446261815324

SELECTIONS = [('tests/data/validation/c2_candidate/20261106/candidate.json',
  '30396a88f535b41cb84b24cea4a625b750c622eb41fc001b0acee796273b8934',
  'retained'),
 ('tests/data/validation/c2o_candidate/20261106/candidate.json',
  'b7927cb42f8f89dd74f5e93153036ff998ecf11041604210199aeb9db9e42131',
  'retained'),
 ('tests/data/validation/c4n2_candidate/20261106/candidate.json',
  'e9673e8799a9884ae279b37771bcc20c21ba79b598b8e30c42d79561632e40f3',
  'retained'),
 ('tests/data/validation/cis_h2n2_candidate/20261106/candidate.json',
  '022aca8794a50da7390828b796662de63a8edeb5e7e02e964b423c71e81415a1',
  'retained'),
 ('tests/data/validation/cnc_candidate/20261106/candidate.json',
  'b0c838354f837efdb281667b545517fb84ead7363dd0638830e454e7df0d6985',
  'retained'),
 ('tests/data/validation/cnn_candidate/20261106/candidate.json',
  '4a2c140189d3cfddabdf7b6ba58ff3d5295fa30b5b612f150cdccb7f9ef0f9f0',
  'retained'),
 ('tests/data/validation/hco_candidate/20261106/candidate.json',
  '2c251d42eaa4d4a7116f07fbe689bbdaccbac1efb640d555be5a4e0474589575',
  'retained'),
 ('tests/data/validation/hnco_candidate/20261106/candidate.json',
  'f14c5203e36f9d6c1e5e52e16bfcd7b425ac3d0fc3780cac417abc45ced2d26d',
  'retained'),
 ('tests/data/validation/ncn_candidate/20261106/candidate.json',
  '880edeade830b910fb84ce60253bc89f80235f436c7a6b7e5467dd2caa4aa77b',
  'retained'),
 ('tests/data/validation/nco_candidate/20261106/candidate.json',
  '823c82db7752e17b06a2026c74931f77fbbd5bc83389ccdd0fc8311bf66f791f',
  'joint_hnco_nco'),
 ('tests/data/validation/nh2_candidate/20261106/candidate.json',
  '3d486440d9f75f5e3c6f7493f3d7db84032d2cb7f4463721a8dabd27aab5aa02',
  'retained'),
 ('tests/data/validation/p4s3_candidate/20261106/candidate.json',
  '92e74fae09f0edf54d1f1129758fb0427558de5f33d1c17a572a02058505ac7f',
  'retained'),
 ('tests/data/validation/vn_candidate/20261106/candidate.json',
  '438d114c087cd4ca317ffcbc3161a194a834c1c74913730e1b65daff64bd58ad',
  'retained'),
 ('tests/data/validation/no_candidate/20261106/candidate.json',
  'd09ba2bb5dccb863a56ce5559bd32d83f4729d57a8dd6f042cc3f10d7b623081',
  'retained'),
 ('tests/data/validation/n2o_candidate/20261106/candidate.json',
  'b3653aa73b92de3baced51e65337fe9ddba1b11352c060ef2a1b238839293d21',
  'retained'),
 ('tests/data/validation/no2_candidate/20261106/candidate.json',
  '91fbc0b465ac2c3dbaf0668e9bfd692aecaeb3107e1df262d07231dbb95f27a1',
  'retained'),
 ('tests/data/validation/hno_candidate/20261106/candidate.json',
  'ac2b0b9571d73ff788e5747a19fdc953215ce36b3734613c6c1e898198644d35',
  'retained'),
 ('tests/data/validation/n2h4_candidate/20261106/candidate.json',
  '20fd1823894ac945c5a03cd71e93320a5bda3e9a3f6602b52aa520a4b2baf6df',
  'retained'),
 ('tests/data/validation/no3_candidate/20261106/candidate.json',
  'a80765e4ae719ad46f49db8dd2cbdc84371f815c6b973f842c811c52429e7593',
  'retained'),
 ('tests/data/validation/n2o4_candidate/20261106/candidate.json',
  '8a56a1b206dc2fbfa9fb975fa3d0c9072b89b1e73053ef3eff893c51481c12bb',
  'joint_no2_n2o4_atct'),
 ('tests/data/validation/cis_hno2_candidate/20261106/candidate.json',
  '3188d8624841c43a80559549eaf277e9f879071762fad38d9525f436034e3fab',
  'joint_cis_trans_atct'),
 ('tests/data/validation/n2o5_h298_candidate/20261106/candidate.json',
  '9d6493a1de48d5516b95da10a980e6f62feadfac21f1d159ca1680470e658ec4',
  'retained_h298'),
 ('tests/data/validation/c3o2_candidate/20261106/candidate.json',
  'dbbd974c3927101a7965c97068ac7d788d630af59b95304e8973b0bf55119213',
  'retained'),
 ('tests/data/validation/o3_candidate/20261106/candidate.json',
  '7c724f0f2a009edf0697fede0e961dc7c5925f607f4e6fbbfc1f4569d209f49e',
  'retained'),
 ('tests/data/validation/o2_candidate/20261106/candidate.json',
  '47e1541d97041b903441e8184cad479f277bf67328c4f91690be6362d3da8863',
  'evaluated_2025_atct'),
 ('tests/data/validation/na2_candidate/20261106/candidate.json',
  '13730be14309fa8a5a29f17e5686a477cb218c0f3706ab57b714b071349d84b1',
  'measured_D0_retained_thermal'),
 ('tests/data/validation/k2_candidate/20261106/candidate.json',
  '69433be9d56ffb400e3aadfa21aa82d274d5fd265e8d046ff5ff2a8fd825501f',
  'measured_D0_retained_thermal')]

ANCHORS = {'C1H1N1O1': {'base_D0_J_mol': 1743502.0,
              'basis': 'ATcT_1.228_balanced_D0',
              'formal_fit_intercept_Ra_J_mol': 1749568.8132269196,
              'reference_temperature_K': 0.0,
              'selected_D0_J_mol': 1760762.0},
 'C1H1O1': {'base_D0_J_mol': 1130859.0,
            'basis': 'ATcT_1.228_balanced_D0',
            'formal_fit_intercept_Ra_J_mol': 1133067.7349987559,
            'reference_temperature_K': 0.0,
            'selected_D0_J_mol': 1132863.0},
 'C1N1O1': {'base_D0_J_mol': 1269854.0,
            'basis': 'ATcT_1.228_balanced_D0',
            'formal_fit_intercept_Ra_J_mol': 1274854.6992437688,
            'reference_temperature_K': 0.0,
            'selected_D0_J_mol': 1301838.0},
 'C1N2_cnn': {'base_D0_J_mol': 1068973.0,
              'basis': 'ATcT_1.228_balanced_D0',
              'formal_fit_intercept_Ra_J_mol': 1072820.351884698,
              'reference_temperature_K': 0.0,
              'selected_D0_J_mol': 1074780.0},
 'C1N2_ncn': {'base_D0_J_mol': 1180558.0,
              'basis': 'ATcT_1.228_balanced_D0',
              'formal_fit_intercept_Ra_J_mol': 1185912.9108015168,
              'reference_temperature_K': 0.0,
              'selected_D0_J_mol': 1201350.0},
 'C2': {'base_D0_J_mol': 593109.0,
        'basis': 'ATcT_1.228_balanced_D0',
        'formal_fit_intercept_Ra_J_mol': 593225.5991623604,
        'reference_temperature_K': 0.0,
        'selected_D0_J_mol': 602800.0},
 'C2N1': {'base_D0_J_mol': 1341362.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 1347390.0815147804,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 1221195.0},
 'C2O1': {'base_D0_J_mol': 1386585.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 1391729.2782679491,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 1292503.0},
 'C3O2': {'base_D0_J_mol': 2723954.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 2735576.082221071,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 2722256.0},
 'C4N2': {'base_D0_J_mol': 3257782.0,
          'basis': 'NASA_header_balanced_D0_corrected_energy',
          'formal_fit_intercept_Ra_J_mol': 3271928.0801342097,
          'reference_temperature_K': 0.0,
          'scope': 'NASA physical D0 using Hf298=529.2kJ/mol; not NIST529.3 and not ATcT.',
          'selected_D0_J_mol': 3262140.965},
 'H1N1O1': {'base_D0_J_mol': 831145.0,
            'basis': 'ATcT_1.228_balanced_D0',
            'formal_fit_intercept_Ra_J_mol': 832987.4582849323,
            'reference_temperature_K': 0.0,
            'selected_D0_J_mol': 823532.0},
 'H1N1O2cis': {'base_D0_J_mol': 1250945.0,
               'basis': 'ATcT_1.228_balanced_D0',
               'formal_fit_intercept_Ra_J_mol': 1257691.4102120618,
               'reference_temperature_K': 0.0,
               'selected_D0_J_mol': 1251685.0},
 'H1N1O2trans': {'base_D0_J_mol': 1253110.0,
                 'basis': 'ATcT_1.228_balanced_D0',
                 'formal_fit_intercept_Ra_J_mol': 1259917.3737213963,
                 'reference_temperature_K': 0.0,
                 'selected_D0_J_mol': 1253295.0},
 'H2N1': {'base_D0_J_mol': 709647.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 709010.6791267293,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 713736.0},
 'H2N2': {'base_D0_J_mol': 1153717.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 1156584.5873278216,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 1144458.0},
 'H4N2': {'base_D0_J_mol': 1696355.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 1705006.4544209887,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 1693616.0},
 'K2': {'base_D0_J_mol': 52660.999999999985,
        'basis': 'Measured_ground_vibrational_D0',
        'formal_fit_intercept_Ra_J_mol': 54635.3242055327,
        'measurement': {'D0_cm_inverse': 4404.808,
                        'access': 'Complete article in author thesis inspected, section 4.1.4; center of '
                                  'gravity of atomic ground asymptote.',
                        'doi': '10.1063/1.2375120',
                        'isotopologue': '39K2',
                        'reference': 'v=0,J=0 to ground-atom asymptote; not De.',
                        'reported_uncertainty_cm_inverse': 0.004},
        'reference_temperature_K': 0.0,
        'reported_D0_uncertainty_J_mol': 0.04785062625547881,
        'selected_D0_J_mol': 52693.20533378577},
 'N1O1': {'base_D0_J_mol': 627835.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 627944.2665082532,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 626811.0},
 'N1O2': {'base_D0_J_mol': 928473.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 932462.1985488434,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 927415.0},
 'N1O3': {'base_D0_J_mol': 1133661.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 1144114.4291619926,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 1131744.0},
 'N1V1': {'base_D0_J_mol': 459848.0,
          'basis': 'neutral_VN_measured_D0',
          'formal_fit_intercept_Ra_J_mol': 461954.68514242646,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 482117.9075537555},
 'N2O1': {'base_D0_J_mol': 1102951.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 1107649.4444990016,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 1102122.0},
 'N2O4': {'base_D0_J_mol': 1910083.0,
          'basis': 'ATcT_1.228_balanced_D0',
          'formal_fit_intercept_Ra_J_mol': 1925865.0482706549,
          'reference_temperature_K': 0.0,
          'selected_D0_J_mol': 1908392.0},
 'N2O5': {'balanced_H298': {'ATcT_1_228_Hf298_J_mol': {'N': 472518, 'N2O5': 14860, 'O': 249229},
                            'ATcT_atomization_H298_J_mol': 2176321.0,
                            'JANAF_Hf298_J_mol': {'N': 472683, 'N2O5': 11297, 'O': 249173},
                            'JANAF_atomization_H298_J_mol': 2179934.0,
                            'delta_formation_GH_J_mol': 3613.0,
                            'physical_JANAF_D0_J_mol': None,
                            'reference_temperature_K': 298.15,
                            'scope': 'Balanced reaction H298 anchor, not D0; unchanged source thermal shape '
                                     'and fit residual.'},
          'base_D0_J_mol': None,
          'basis': 'ATcT_1.228_balanced_reaction_H298',
          'formal_fit_intercept_Ra_J_mol': 2174609.595825929,
          'reference_temperature_K': 298.15,
          'selected_D0_J_mol': None},
 'Na2': {'base_D0_J_mol': 70539.0,
         'basis': 'Measured_ground_vibrational_D0',
         'formal_fit_intercept_Ra_J_mol': 73631.51475459315,
         'measurement': {'D0_cm_inverse': 5942.688,
                         'access': 'Primary publisher abstract inspected; full original paper not obtained.',
                         'doi': '10.1103/PhysRevA.54.R1006',
                         'isotopologue': '23Na2',
                         'reference': 'v=0,J=0 to ground-atom asymptote; not De.',
                         'reported_uncertainty_cm_inverse': 0.0049},
         'reference_temperature_K': 0.0,
         'reported_D0_uncertainty_J_mol': 0.05861701716296154,
         'selected_D0_J_mol': 71090.33561022971},
 'O2': {'base_D0_J_mol': 493579.0,
        'basis': 'ATcT_1.228_balanced_D0',
        'formal_fit_intercept_Ra_J_mol': 494989.92505723226,
        'reference_temperature_K': 0.0,
        'same_version_hf0_kJ_mol': {'O': 246.843, 'O2': 0.0},
        'selected_D0_J_mol': 493686.0},
 'O3': {'base_D0_J_mol': 595021.0,
        'basis': 'ATcT_1.228_balanced_D0',
        'formal_fit_intercept_Ra_J_mol': 600766.6643808839,
        'reference_temperature_K': 0.0,
        'selected_D0_J_mol': 596120.0},
 'P4S3': {'base_D0_J_mol': None,
          'base_Hf298_J_mol': -151042.0,
          'basis': 'preferred_cycle_Hf298_at_retained_atomic_reference',
          'formal_fit_intercept_Ra_J_mol': 2248818.8629735513,
          'reference_temperature_K': 298.15,
          'selected_D0_J_mol': None,
          'selected_Hf298_J_mol': -94500.0,
          'source_doi': '10.1007/s11224-019-01352-7'}}

DECISIONS = {'C1H1N1O1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                           'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                           'completeness and full reaction covariance remain unquantified.',
              'energy_source': 'ATcT_1.228_balanced_D0',
              'name': 'HNCO',
              'reason': 'Recommend retained-thermal ATcT energy adoption, with HNCO/NCO bookkeeping reviewed '
                        'jointly before production selection. A low-temperature partition table is not '
                        'extrapolated into the hot candidate.',
              'source_group': 'molecular',
              'status': 'prefer_conditionally',
              'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C1H1O1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                         'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                         'completeness and full reaction covariance remain unquantified.',
            'energy_source': 'ATcT_1.228_balanced_D0',
            'name': 'HCO',
            'reason': 'Recommend the retained-thermal ATcT energy update. Retain NASA as a distinct '
                      'thermal/reference sensitivity, without declaring it the superior full-range thermal '
                      'function.',
            'source_group': 'molecular',
            'status': 'prefer_conditionally',
            'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C1N1O1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                         'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                         'completeness and full reaction covariance remain unquantified.',
            'energy_source': 'ATcT_1.228_balanced_D0',
            'name': 'NCO',
            'reason': 'Recommend the retained-thermal ATcT energy update, coordinating its production '
                      'selection with HNCO. Separate and joint fixed-network variants quantify coupled '
                      'response; they do not provide reaction covariance or validate an entire rovibronic '
                      'model.',
            'source_group': 'molecular',
            'status': 'prefer_conditionally',
            'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C1N2_cnn': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                           'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                           'completeness and full reaction covariance remain unquantified.',
              'energy_source': 'ATcT_1.228_balanced_D0',
              'name': 'CNN',
              'reason': 'A provisional retained-thermal ATcT energy update is scientifically supported as '
                        'one bounded improvement. Do not describe the retained thermal model as validated: a '
                        'source-resolved vibration/electronic update remains necessary for a stronger '
                        'full-function claim. NASA is a sensitivity, not an automatic replacement.',
              'source_group': 'molecular',
              'status': 'prefer_conditionally',
              'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C1N2_ncn': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                           'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                           'completeness and full reaction covariance remain unquantified.',
              'energy_source': 'ATcT_1.228_balanced_D0',
              'name': 'NCN',
              'reason': 'Recommend the current evaluated ATcT energy with retained thermals. Keep NASA as a '
                        'separate sensitivity and the2026 measurement as corroboration. Full excited-state '
                        'thermal accuracy remains unknown.',
              'source_group': 'molecular',
              'status': 'prefer_conditionally',
              'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C2': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                     'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                     'completeness and full reaction covariance remain unquantified.',
        'energy_source': 'ATcT_1.228_balanced_D0',
        'name': 'C2',
        'reason': 'Recommend retained thermals with the ATcT energy. The measured ground-referenced '
                  'dissociation supports the physical change independently of equilibrium fitting. A later '
                  'eight-state thermal candidate should make dissociation and isotope conventions explicit.',
        'source_group': 'molecular',
        'status': 'prefer_conditionally',
        'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C2N1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'CNC',
          'reason': 'Recommend the evaluated energy as a bounded correction, retaining the present thermal '
                    'function and its residual. The separate NASA result quantifies remaining thermal '
                    'sensitivity; a complete Renner-Teller thermal update is deferred and full physical '
                    'accuracy remains unknown.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C2O1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'C2O',
          'reason': 'Recommend the evaluated energy as a bounded improvement while retaining the literal '
                    'thermal function. The old electronic model remains a material inherited limitation; '
                    'NASA sensitivity does not resolve its physical completeness. A complete thermal update '
                    'is deferred, not silently invented.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C3O2': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'C3O2',
          'reason': 'Prefer the same-version ATcT physical energy correction as a scoped future update, '
                    'preserving the existing thermal model and fit residual. The source cycle supports G/H '
                    '+1698 J/mol; the NASA alternative changes the thermal/gauge construction and can '
                    'reverse the response, without demonstrated physical superiority. Negligible abundance '
                    'in this grid does not by itself validate or invalidate its energy source.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'C4N2': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'NASA_header_balanced_D0_corrected_energy',
          'name': 'C4N2',
          'reason': 'A bounded corrected-energy update using the explicitly selected NASA physical anchor is '
                    'supported by the documented combustion correction. Prefer retaining thermals for that '
                    'step; the second full NASA model is a sensitivity. This trace-species update has low '
                    'immediate priority on the tested grid.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'H1N1O1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                         'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                         'completeness and full reaction covariance remain unquantified.',
            'energy_source': 'ATcT_1.228_balanced_D0',
            'name': 'HNO',
            'reason': 'Recommend the retained-thermal ATcT energy correction; the NASA thermal alternative '
                      'is a quantified sensitivity without demonstrated physical superiority.',
            'source_group': 'molecular',
            'status': 'prefer_conditionally',
            'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'H1N1O2cis': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                            'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                            'completeness and full reaction covariance remain unquantified.',
               'energy_source': 'ATcT_1.228_balanced_D0',
               'name': 'cis-HNO2',
               'reason': 'Prefer the coordinated cis/trans ATcT energy candidate as the internally '
                         'consistent evaluated option. Keep the 2023 measured-gap hybrid and its '
                         'quoted-width edges as explicit alternatives; do not adopt a mixed-isomer thermal '
                         'function.',
               'source_group': 'molecular',
               'status': 'prefer_conditionally',
               'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'H1N1O2trans': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                              'intercept; atomic gauge and fit residual are retained. Source-specific '
                              'thermal completeness and full reaction covariance remain unquantified.',
                 'energy_source': 'ATcT_1.228_balanced_D0',
                 'name': 'trans-HNO2',
                 'reason': 'Prefer the coordinated cis/trans ATcT energy candidate as the internally '
                           'consistent evaluated option. Keep the 2023 measured-gap hybrid and its '
                           'quoted-width edges as explicit alternatives; do not adopt a mixed-isomer thermal '
                           'function.',
                 'source_group': 'molecular',
                 'status': 'prefer_conditionally',
                 'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'H2N1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'NH2',
          'reason': 'Recommend the retained-thermal ATcT energy update for a separately authorized bounded '
                    'adoption. Keep the NASA thermal candidate as a measured model contrast; full-range '
                    'superiority is not established. The2026 spin-convention issue does not block this '
                    'energy decision.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'H2N2': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'cis-H2N2',
          'reason': 'Recommend the exact-isomer retained-thermal ATcT energy change. Do not replace cis with '
                    'a NASA isomer-mixture function; a separate cis-resolved thermal model can be assessed '
                    'later.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'H4N2': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'N2H4',
          'reason': 'Prefer the retained-thermal ATcT energy update; do not adopt the substantially '
                    'different NASA thermal function until torsional and inversion treatments are '
                    'reconciled.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'K2': {'condition': '1500-4000 K,1bar standard pressure; atomic gauge and entire inherited residual '
                     'retained. G/H change by the stated constant; S/Cp remain unchanged. No added ZPE, H298 '
                     'offset, symmetry or nuclear-spin correction. The empirical R*a intercept is not the '
                     'physical JANAF D0. Keep both values separate. Measured anchor uncertainty is not '
                     'propagated as complete thermal or atmospheric uncertainty. No source inspected '
                     'establishes a complete corrected hot state inventory. Missing shallow a3Sigma-u, '
                     'rotational/near-dissociation and other electronic coverage remain unresolved. Gibbs '
                     'changes below the recorded optimizer diagnostic scale are not claims of an exact sign '
                     'or physical superiority. 39K2 principal-isotope D0 is used as a bounded central-energy '
                     'proxy for the inherited element-aggregate thermal species, not as a measured '
                     'natural-isotope mean. 2006 D0 is relative to atomic ground-asymptote center of '
                     'gravity; full article was inspected through author thesis. 2026 potential '
                     'D0=4404.80862651cm^-1 differs by0.007494723970012274J/mol, within the reported2006 '
                     'uncertainty. Its86 J=0 levels do not establish J>0/electronic/minor-isotope thermal '
                     'completeness.',
        'energy_source': 'Measured_ground_vibrational_D0',
        'name': 'K2',
        'reason': 'Prefer a direct measured ground-vibrational binding energy with an explicit ground-atom '
                  'asymptote, while preserving the literal thermal fit and its source residual.',
        'source_group': 'molecular',
        'status': 'prefer_conditionally',
        'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'N1O1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'NO',
          'reason': 'Recommend the ATcT energy update with the existing thermal function; no additional '
                    'thermal replacement is justified.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'N1O2': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'NO2',
          'reason': 'Prefer the retained-thermal ATcT energy update, with NASA thermals retained only as a '
                    'sensitivity; coordinate any adoption with the N2O4 energy cycle.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'N1O3': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'NO3',
          'reason': 'Recommend the retained-thermal ATcT energy correction. Do not turn the incomplete '
                    'modern vibronic levels into a hot thermal replacement.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'N1V1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'neutral_VN_measured_D0',
          'name': 'VN',
          'reason': 'Recommend the measured neutral-D0 update with retained thermals. The narrow threshold '
                    'width is separately propagated at selected conditions; no cation energy, well depth or '
                    'second zero-point correction is used.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'N2O1': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'N2O',
          'reason': 'Prefer the retained-thermal ATcT energy candidate for a bounded update; keep NASA plus '
                    'ATcT as a tested thermal sensitivity, not an established improvement.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'N2O4': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_D0',
          'name': 'N2O4',
          'reason': 'Recommend coordinated retained-thermal NO2 and N2O4 ATcT energy updates; the dimer-only '
                    'result is an isolation control.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'N2O5': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'ATcT_1.228_balanced_reaction_H298',
          'name': 'N2O5',
          'reason': 'Recommend the retained-thermal ATcT H298 energy correction; keep NASA plus ATcT as a '
                    'tested thermal sensitivity, with thermal superiority unresolved.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'Na2': {'condition': '1500-4000 K,1bar standard pressure; atomic gauge and entire inherited residual '
                      'retained. G/H change by the stated constant; S/Cp remain unchanged. No added ZPE, '
                      'H298 offset, symmetry or nuclear-spin correction. The empirical R*a intercept is not '
                      'the physical JANAF D0. Keep both values separate. Measured anchor uncertainty is not '
                      'propagated as complete thermal or atmospheric uncertainty. No source inspected '
                      'establishes a complete corrected hot state inventory. Missing shallow a3Sigma-u, '
                      'rotational/near-dissociation and other electronic coverage remain unresolved. Gibbs '
                      'changes below the recorded optimizer diagnostic scale are not claims of an exact sign '
                      'or physical superiority. 23Na2 is the naturally monoisotopic convention. The original '
                      'primary publisher abstract is inspected; full1996 original paper remains unavailable, '
                      'explicitly bounded rather than an indefinite adoption gate.',
         'energy_source': 'Measured_ground_vibrational_D0',
         'name': 'Na2',
         'reason': 'Prefer a direct measured ground-vibrational binding energy with an explicit ground-atom '
                   'asymptote, while preserving the literal thermal fit and its source residual.',
         'source_group': 'molecular',
         'status': 'prefer_conditionally',
         'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'O2': {'condition': '1500-4000 K only, despite original evaluation table extending to 5000 K. '
                     'Natural-isotope mass translation, authors Cp correction and '
                     'rotational/electronic/nuclear statistical conventions are already included. No extra '
                     'spin3, symmetry2, isotope mixing or mass term. The inherited residual is defined '
                     'against NASA molecular/atomic thermal proxies and physical JANAF D0. It is retained; '
                     'this is not a pure JANAF table replacement or a full thermochemical refit. ATcT '
                     'balanced rounded central values only; complete atom/molecule reaction covariance is '
                     'not propagated. The 15946-level manifold from2019 underlies2025 constants/uncertainty '
                     'reevaluation. Unbound-state uncertainty, scattering/continuum assignment, '
                     'isotope-network and atmospheric physical accuracy remain unknown. The fit uses inverse '
                     'expanded Cp uncertainty only as numerical weighting. Small fit error is not a new '
                     'physical covariance or confidence interval.',
        'energy_source': 'ATcT_1.228_balanced_D0',
        'name': 'O2',
        'reason': 'Prefer the evaluated source with explicit natural-isotope thermodynamic conventions and '
                  'uncertainty treatment, plus same-version balanced ATcT energy; preference is not based on '
                  'the small abundance response or closeness to NASA.',
        'source_group': 'molecular',
        'status': 'prefer_conditionally',
        'thermal_source': '2025 evaluated natural-isotope thermals; retained NASA-proxy fit residual'},
 'O3': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                     'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                     'completeness and full reaction covariance remain unquantified.',
        'energy_source': 'ATcT_1.228_balanced_D0',
        'name': 'O3',
        'reason': 'Prefer the same-version ATcT energy correction as a bounded future update: physical D0 '
                  'increases by 1099 J/mol, so G/H decrease by that amount. Retain the thermal fit until its '
                  'larger NASA difference has an independent physical resolution. This is an ideal-gas '
                  'equilibrium source decision, not an ozone photochemistry prediction.',
        'source_group': 'molecular',
        'status': 'prefer_conditionally',
        'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'},
 'P4S3': {'condition': '1500-4000 K,1bar standard pressure. Physical energy is separate from the fit '
                       'intercept; atomic gauge and fit residual are retained. Source-specific thermal '
                       'completeness and full reaction covariance remain unquantified.',
          'energy_source': 'preferred_cycle_Hf298_at_retained_atomic_reference',
          'name': 'P4S3',
          'reason': 'If updating this estimated source, prefer the2019 cycle-based H298 energy while '
                    'retaining explicit thermal limitations. This is scientifically supported but very low '
                    'priority for the tested hot gas. A tiny bulk response is not evidence of general '
                    'irrelevance; a full cage thermal model remains deferred.',
          'source_group': 'molecular',
          'status': 'prefer_conditionally',
          'thermal_source': 'Retained literal JANAF-derived FastChem reaction thermal fit'}}


def _checked(path: str, expected: str) -> bytes:
    raw = (ROOT / path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Curated v2 requires pinned candidate/base bytes: " + path)
    return raw


def build_data() -> dict:
    """Select reviewed variants once, retaining the literal base and O2 proxy residual."""
    base = json.loads(_checked(BASE, BASE_SHA256))
    if base["id"] != "curated-ch2-v1":
        raise ValueError("Unexpected curated v2 base")
    lines = _checked(LOGK, LOGK_SHA256).decode().splitlines()
    shifts, coefficients, references = {}, {}, {}
    source_hashes = {BASE: BASE_SHA256, LOGK: LOGK_SHA256}
    thermal = proxy = None
    for path, expected, selected in SELECTIONS:
        candidate = json.loads(_checked(path, expected))
        if (candidate["baseline_dataset"] != base["id"]
                or candidate["temperature_range_K"] != [1500, 4000]):
            raise ValueError("Candidate base or domain differs: " + path)
        variant = next(v for v in candidate["variants"] if v["name"] == selected)
        source_hashes[path] = expected
        for species, spec in variant["replacements"].items():
            if species == "O2":
                parameters = spec["parameters"]
                spec = parameters["retained"]
                thermal = {key: value for key, value in parameters["evaluated"].items()
                           if key not in ("fit_temperature_K", "fit_weights")}
                proxy = parameters["nasa_molecular_thermal_coefficients"]
            if spec["kind"] != "retained_energy":
                raise ValueError("Only selected retained-energy variants are adopted")
            shift = spec["delta_GH_J_mol"]
            literal = spec["coefficients"]
            line = next(i for i, value in enumerate(lines) if value.startswith(species + " "))
            if list(map(float, lines[line + 1].split())) != literal:
                raise ValueError("Candidate no longer retains the literal fit: " + species)
            if species in shifts and (shifts[species] != shift or coefficients[species] != literal):
                raise ValueError("Inconsistent overlapping joint selections: " + species)
            shifts[species], coefficients[species] = shift, literal
            references.setdefault(species, []).append({"path": path, "sha256": expected, "variant": selected})
    if len(shifts) != 28 or set(shifts) != set(ANCHORS) or set(shifts) != set(DECISIONS):
        raise ValueError("Curated v2 requires the exact 28 reviewed species")
    for species, shift in shifts.items():
        anchor = ANCHORS[species]
        if species == "P4S3":
            physical_shift = anchor["selected_Hf298_J_mol"] - anchor["base_Hf298_J_mol"]
        elif species == "N2O5":
            physical_shift = anchor["balanced_H298"]["delta_formation_GH_J_mol"]
        else:
            physical_shift = anchor["base_D0_J_mol"] - anchor["selected_D0_J_mol"]
        if abs(physical_shift - shift) > 1e-8:
            raise ValueError("Energy sign or anchor differs: " + species)
    if thermal is None or proxy is None or shifts["O2"] != -107:
        raise ValueError("The evaluated O2 thermal correction and single energy shift are required")
    names = sorted(shifts)
    return {
        "id": "curated-gas-v2", "base_dataset": base["id"], "base_data_sha256": BASE_SHA256,
        "temperature_range_K": [1500, 4000], "standard_pressure_Pa": 100000,
        "gas_constant_J_mol_K": R,
        "energy_shifts_J_mol": {name: shifts[name] for name in names},
        "decisions": {name: DECISIONS[name] for name in names},
        "energy_anchors": {name: ANCHORS[name] for name in names},
        "candidate_references": {name: references[name] for name in names},
        "thermal": {"species": "O2", **thermal},
        "nasa_molecular_thermal_coefficients": proxy,
        "thermal_scope": "Only O2 changes thermal shape: evaluated natural-isotope G-H0 minus the frozen NASA molecular thermal proxy. Preserve all inherited fit residuals and atomic thermals. No extra spin, symmetry, mass, isotope or ZPE correction.",
        "energy_scope": "Add each selected reaction G/H offset once, including O2=-107 J/mol. N2O5 uses balanced298.15K atomization enthalpies; P4S3 uses the preferred298.15K reaction cycle. Neither is relabeled as D0. Coupled HNCO/NCO,NO2/N2O4 and cis/trans-HNO2 selections are deduplicated.",
        "sources": {
            "adoption": "documents/mceg/curated_gas_v2_adoption.rst",
            "contract": "mceg/contracts/curated_gas_v2_adoption.yaml",
            "O2_evaluation": "https://doi.org/10.1063/5.0273661",
            "candidate_records": [path for path, _, _ in SELECTIONS],
        },
        "source_sha256": source_hashes,
        "interpretation": "Adopt27 reviewed candidate selections affecting28 unique molecular species on curated-ch2-v1, with one evaluated O2 thermal correction. All other thermal shapes, source-fit residuals, atomic potentials and network definitions are retained. Candidate sensitivities not selected here remain review controls. Scoped numerical support does not establish complete physical or atmospheric accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_data(), indent=2, allow_nan=False) + "\n")
