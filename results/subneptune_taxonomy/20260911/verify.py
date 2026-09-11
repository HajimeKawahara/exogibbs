"""Recompute archived chemistry, with optional fresh external MELTS evaluation."""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "examples" / "metal_silicate"))
from revalidate_archive import main


if __name__ == "__main__":
    main(HERE)
