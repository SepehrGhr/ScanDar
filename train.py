#!/usr/bin/env python
"""Training entry point — ``python train.py --config configs/enhance.yaml``.

The brief asks for the training implementation to live in ``train.py``; it does,
in ``src/scandar/train.py``. This is the runnable shim so the command works from
the repository root without installing anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scandar.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
