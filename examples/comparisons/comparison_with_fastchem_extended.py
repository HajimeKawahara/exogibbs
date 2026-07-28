"""Compatibility entry point for the current FastChem 4 gas comparison.

This preserves the historical example path while delegating to the current
FastChem 4 standalone comparison in ``comparison_with_fastchem4_gas.py``.
"""

from comparison_with_fastchem4_gas import main


if __name__ == "__main__":
    main()
