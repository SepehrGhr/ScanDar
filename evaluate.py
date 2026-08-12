#!/usr/bin/env python
"""Evaluation entry point — ``python evaluate.py --config configs/enhance.yaml``.

The brief asks for the evaluation step to live in ``evaluate.py``; it does, in
``src/scandar/evaluate.py``. This is the runnable shim from the repository root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scandar.evaluate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
