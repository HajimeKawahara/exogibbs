"""Compatibility entry point for the current FastChem 4 gas comparison.

The historical script used PyFastChem 3 and retired ExoGibbs APIs.  The
readable implementation now lives in ``comparison_with_fastchem4_gas.py``.
"""

from comparison_with_fastchem4_gas import main


if __name__ == "__main__":
    main()
