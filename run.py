"""Direct launcher for Liquid Studio."""
import os
import sys

# Ensure parent directory of LiquidStudio is in sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from LiquidStudio.__main__ import main

if __name__ == "__main__":
    main()
