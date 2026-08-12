"""Make ``scandar`` importable when the package has not been pip-installed.

Colab runtimes and fresh clones both hit this: ``python scripts/foo.py`` runs
without ``src/`` on the path. Importing this module first fixes that, and does
nothing when the package is already installed.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
