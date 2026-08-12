"""Data preparation: the scan cache and the split manifest.  *(Phase 0)*

Two jobs, both idempotent and both cheap to re-run:

**Cache the scans.** The originals are ~2480x3512 JPEGs. Re-reading and warping
those inside every ``__getitem__`` would make the CPU, not the GPU, the thing that
limits training, so each scan is downscaled once to 1600 px on the long side. The
cache is written as **PNG**: these images are the ground-truth targets, and
re-encoding them as JPEG would bake fresh compression artefacts into the very
thing the network is asked to reproduce.

**Write the split.** The brief is specific here: split by *source scan*, never by
generated sample, so two degraded versions of the same page can never end up on
opposite sides. 50 scans become 40 train / 5 validation / 5 test. Background
photos are split the same way — a surface the model trained on should not reappear
in validation or test — but by manifest rather than by moving files.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .io import imread_rgb, imwrite_rgb, list_images, natural_key, paths, write_json

BACKGROUND_HELDOUT_FRACTION = 0.25


def cache_scans(long_side: int = 1600, force: bool = False) -> dict:
    """Downscale every scan to *long_side* px and store it losslessly."""
    import cv2

    sources = list_images(paths.scans)
    if not sources:
        raise FileNotFoundError(
            f"no scans found in {paths.scans}. Drop the provided clean scans there first."
        )

    paths.scans_cache.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    for source in sources:
        target = paths.scans_cache / f"{source.stem}.png"
        if target.exists() and not force and target.stat().st_mtime >= source.stat().st_mtime:
            skipped += 1
            continue
        image = imread_rgb(source)
        height, width = image.shape[:2]
        scale = long_side / max(height, width)
        if scale < 1.0:
            size = (int(round(width * scale)), int(round(height * scale)))
            # INTER_AREA is the correct filter for downscaling: it averages the
            # pixels being merged instead of point-sampling them, which matters
            # for thin pen strokes.
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        imwrite_rgb(target, image)
        written += 1

    return {"long_side": long_side, "count": len(sources), "written": written, "skipped": skipped}


def make_splits(seed: int = 1234, n_val: int = 5, n_test: int = 5) -> dict:
    """Assign scans and backgrounds to train / val / test. Deterministic in *seed*."""
    scan_ids = [p.stem for p in list_images(paths.scans)]
    if len(scan_ids) < n_val + n_test + 1:
        raise ValueError(
            f"only {len(scan_ids)} scans found; need more than {n_val + n_test} to split"
        )

    # Scans are permuted before backgrounds, so adding background photos later and
    # re-running this script leaves the scan split untouched.
    rng = np.random.default_rng(seed)
    shuffled = [str(x) for x in rng.permutation(scan_ids)]
    test = _natural_sorted(shuffled[:n_test])
    val = _natural_sorted(shuffled[n_test : n_test + n_val])
    train = _natural_sorted(shuffled[n_test + n_val :])

    background_ids = [p.name for p in list_images(paths.backgrounds)]
    if background_ids:
        shuffled_bg = [str(x) for x in rng.permutation(background_ids)]
        n_heldout = max(1, round(BACKGROUND_HELDOUT_FRACTION * len(background_ids)))
        bg_heldout = _natural_sorted(shuffled_bg[:n_heldout])
        bg_train = _natural_sorted(shuffled_bg[n_heldout:])
    else:
        bg_heldout, bg_train = [], []

    return {
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "policy": (
            "Split by source scan, never by generated sample: two degraded versions of the "
            "same page must never land on opposite sides. Backgrounds are split too, so a "
            "surface seen in training does not reappear in validation or test."
        ),
        "scans": {"train": train, "val": val, "test": test},
        "backgrounds": {"train": bg_train, "heldout": bg_heldout},
        "counts": {
            "scans": {"train": len(train), "val": len(val), "test": len(test)},
            "backgrounds": {"train": len(bg_train), "heldout": len(bg_heldout)},
        },
    }


def load_splits() -> dict:
    """Read ``data/splits.json``, with a useful error if it has not been built."""
    from .io import read_json

    if not paths.splits.exists():
        raise FileNotFoundError(
            f"{paths.splits} not found — run `python scripts/prepare_data.py` first."
        )
    return read_json(paths.splits)


def run(seed: int = 1234, long_side: int = 1600, force: bool = False) -> dict:
    """Cache the scans, write the split manifest, print a summary."""
    print(f"data root : {paths.data}")

    cache_info = cache_scans(long_side=long_side, force=force)
    print(
        f"scan cache: {cache_info['count']} scans at {long_side}px long side "
        f"({cache_info['written']} written, {cache_info['skipped']} already current) "
        f"-> {_relative(paths.scans_cache)}"
    )

    splits = make_splits(seed=seed)
    write_json(paths.splits, splits)
    scans = splits["counts"]["scans"]
    backgrounds = splits["counts"]["backgrounds"]
    print(
        f"splits    : {scans['train']} train / {scans['val']} val / {scans['test']} test scans "
        f"(seed {seed}) -> {_relative(paths.splits)}"
    )
    if backgrounds["train"] or backgrounds["heldout"]:
        print(
            f"            {backgrounds['train']} train / {backgrounds['heldout']} held-out backgrounds"
        )
    else:
        print(
            f"            no backgrounds yet — drop 20-30 background-only photos into "
            f"{_relative(paths.backgrounds)} and re-run this script"
        )

    return {"cache": cache_info, "splits": splits}


def _natural_sorted(names: list[str]) -> list[str]:
    """Sort so ``2`` precedes ``10`` — the manifest is meant to be read by humans."""
    return sorted(names, key=natural_key)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(paths.repo))
    except ValueError:
        return str(path)
