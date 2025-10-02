YKB4 Preset
===========

Overview
--------
The ykb4 preset bundles a ready-to-use ChemicalSetup based on JANAF-like
Gibbs tables. It exposes a device-friendly stoichiometric matrix and a
JAX-differentiable h(T) interpolator for equilibrium calculations.

Quick Start
-----------
.. code-block:: python

   from exogibbs.presets.ykb4 import chemsetup
   from exogibbs.api.equilibrium import equilibrium

   setup = chemsetup()
   T, P = 1500.0, 1.0  # K, bar
   b = setup.element_vector_reference  # or your own jnp.array([...])
   result = equilibrium(setup, T=T, P=P, b=b)
   print(result.x)  # mole fractions (K,)

Elements
--------
The following elements are included in this preset (including electrons as "e-"):

``C, H, He, K, N, Na, O, P, S, Ti, V, e-``

The reference elemental abundances (`element_vector_reference`) are taken from Asplund, Amarsi & Grevesse (2021).

Species (JANAF key)
-------------------
The species list corresponds to the ``JANAF`` column in ``molecule_names.csv``.

.. code-block:: text

   000: C1
   001: H2
   002: He1
   003: K1
   004: N2
   005: Na1
   006: O2
   007: Ti1
   008: V1
   009: P1
   010: S1
   011: e1-
   012: C1H1
   013: C1H1N1
   014: C1H1N1O1
   015: C1H1O1
   016: C1H2
   017: C1H2O1
   018: C1H3
   019: C1H4
   020: C1K1N1
   021: C1N1
   022: C1N1Na1
   023: C1N1O1
   024: C1N2(CNN)
   025: C1N2(NCN)
   026: C1O1
   027: C1O2
   028: C2
   029: C2H1
   030: C2H2
   031: C2H4
   032: C2H4O1
   033: C2K2N2
   034: C2N1
   035: C2N2
   036: C2N2Na2
   037: C2O1
   038: C3
   039: C3O2
   040: C4
   041: C4N2
   042: C5
   043: H1
   044: H1K1
   045: H1K1O1
   046: H1N1
   047: H1N1O1
   048: H1N1O2(cis)
   049: H1N1O2(trans)
   050: H1N1O3
   051: H1Na1
   052: H1Na1O1
   053: H1O1
   054: H1O2
   055: H2K2O2
   056: H2N1
   057: H2N2
   058: H2Na2O2
   059: H2O1
   060: H2O2
   061: H3N1
   062: H4N2
   063: K1O1
   064: K2
   065: N1
   066: N1O1
   067: N1O2
   068: N1O3
   069: N2O1
   070: N2O3
   071: N2O4
   072: N2O5
   073: N3
   074: Na1O1
   075: Na2
   076: O1
   077: O3
   078: N1V1
   079: O1Ti1
   080: O1V1
   081: O2Ti1
   082: O2V1
   083: C1H1P1
   084: C1O1S1
   085: C1P1
   086: C1S1
   087: C1S2
   088: H1P1
   089: H1S1
   090: H2O4S1
   091: H2P1
   092: H2S1
   093: H3P1
   094: K2O4S1
   095: N1P1
   096: N1S1
   097: Na2O4S1
   098: O1P1
   099: O1S1
   100: O1S2
   101: O2P1
   102: O2S1
   103: O3S1
   104: O6P4
   105: O10P4
   106: P1S1
   107: P2
   108: P4
   109: P4S3
   110: S2
   111: S3
   112: S4
   113: S5
   114: S6
   115: S7
   116: S8
   117: C1+
   118: C1-
   119: C1H1+
   120: C1H1O1+
   121: C1N1+
   122: C1N1-
   123: C1O2-
   124: C2-
   125: H1+
   126: H1-
   127: H1K1O1+
   128: H1Na1O1+
   129: H1O1+
   130: H1O1-
   131: H2+
   132: H2-
   133: H3O1+
   134: He1+
   135: K1+
   136: K1-
   137: K1O1-
   138: N1+
   139: N1-
   140: N1O1+
   141: N1O2-
   142: N2+
   143: N2-
   144: N2O1+
   145: Na1+
   146: Na1-
   147: Na1O1-
   148: O1+
   149: O1-
   150: O2+
   151: O2-
   152: P1+
   153: P1-
   154: S1+
   155: S1-
   156: Ti1+
   157: Ti1-
   158: V1+
   159: V1-
