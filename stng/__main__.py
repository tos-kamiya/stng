import sys

from . import stng

__doc__ = """
Usage:
  python3 -m stng [--help] ...
"""

if len(sys.argv) <= 1:
    print(__doc__)
    sys.exit(0)

stng.main()

