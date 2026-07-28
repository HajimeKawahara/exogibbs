"""Compatibility entry point for the production condensate comparison.

The historical script did not complete an ExoGibbs condensate comparison.
The current implementation lives in
``comparison_with_fastchem4_condensates.py``.
"""

from comparison_with_fastchem4_condensates import main


if __name__ == "__main__":
    main()
