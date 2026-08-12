"""ScanDar — turn a phone photo of a document into a clean scan.

Two networks, trained independently and chained at the end:

* an **enhancement** network, mapping a degraded rectified page to a clean scan;
* a **corner detector**, locating the four page corners in a raw photo, solved
  twice (direct coordinate regression and heatmap regression) so the experiments
  can decide which formulation wins.

Both are trained entirely on synthetic data: clean scans are perspective-warped
onto random backgrounds and degraded with an OpenCV-only pipeline, which hands us
pixel-perfect corner labels and perfectly aligned (degraded, clean) pairs for free.

Importing this package is deliberately cheap — it pulls in neither torch nor cv2.
"""

__version__ = "0.1.0"

__all__ = ["__version__", "paths", "DATA_ROOT", "OUT_ROOT", "REPO_ROOT"]


def __getattr__(name: str):
    """Expose the common path handles lazily, so ``import scandar`` stays light."""
    if name in ("paths", "DATA_ROOT", "OUT_ROOT", "REPO_ROOT"):
        from . import io

        return getattr(io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
