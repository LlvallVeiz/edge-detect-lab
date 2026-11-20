"""Entry point for ``python -m edge_detect_lab``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
