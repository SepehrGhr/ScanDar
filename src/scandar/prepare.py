"""Data preparation: the scan cache, the split manifest and the frozen eval sets.

Three jobs, all idempotent and all cheap to re-run:

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

**Freeze the evaluation sets.** The dataset invents a fresh sample per
``__getitem__``, so an unfrozen validation curve would measure the dice as much as
the model. Validation and test are generated once from a fixed seed and written to
disk *(brief §2.3)*. There is a frozen *training* bucket too, which sounds odd
until you notice that with an infinite generator the model never sees the same
sample twice: "performance on the training set" can only mean performance on the
training *distribution* — the training scans and the training backgrounds — and
that is exactly what a fixed set of samples drawn from it measures.

Frozen sets are stored per task, under ``data/frozen/<task>/<split>/``. The two
tasks are deliberately generated from different worlds — the corner detector sees
tinted stock, distractor sheets and curled pages, the enhancement network must
not — so one shared set of photos cannot score both.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .io import imread_rgb, imwrite_rgb, list_images, natural_key, paths, write_json

BACKGROUND_HELDOUT_FRACTION = 0.25

#: All 40 training scans cached in a worker is roughly 220 MB. That is the right
#: trade on a 30 GB machine — decoding a 1600 px PNG costs more than everything
#: else in a sample put together — but it is the first knob to turn on a small one.
SCAN_CACHE_SIZE = 64

#: Frozen photos are stored as high-quality JPEG rather than PNG. They are already
#: the output of a quality 30-80 re-encode, so another pass at 96 changes almost
#: nothing — and four hundred PNGs of degraded photo texture would be well over a
#: gigabyte on a disk that does not have one to spare.
FROZEN_JPEG_QUALITY = 96


class ScanBank:
    """The cached clean scans belonging to one side of the split, by id."""

    def __init__(
        self,
        ids: list[str] | None = None,
        directory: Path | str | None = None,
        cache_size: int = SCAN_CACHE_SIZE,
    ) -> None:
        self.directory = Path(directory) if directory is not None else paths.scans_cache
        if ids is None:
            ids = [p.stem for p in list_images(self.directory)]
        self.ids = list(ids)
        self.cache_size = cache_size
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScanBank({len(self.ids)} scans from {self.directory})"

    def load(self, scan_id: str) -> np.ndarray:
        image = self._cache.get(scan_id)
        if image is None:
            image = imread_rgb(self.path_for(scan_id))
            if len(self._cache) < self.cache_size:
                self._cache[scan_id] = image
        return image

    def warm(self) -> int:
        """Decode every scan now. Returns how many are held.

        Worth calling in the parent process *before* a DataLoader forks its
        workers: decoding a 1600 px PNG costs more than compositing a whole
        sample, and a forked worker inherits this cache copy-on-write instead of
        rebuilding it. Without it every worker decodes the split again — and with
        ``persistent_workers=False``, once per epoch, for the whole run.
        """
        for scan_id in self.ids[: self.cache_size]:
            self.load(scan_id)
        return len(self._cache)

    def path_for(self, scan_id: str) -> Path:
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = self.directory / f"{scan_id}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"scan {scan_id!r} is not in {self.directory} — "
            "run `python scripts/prepare_data.py` to build the cache"
        )


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


# ---------------------------------------------------------------------------
# frozen evaluation sets
# ---------------------------------------------------------------------------
def freeze_split(
    config,
    split: str,
    count: int,
    seed: int = 1234,
    directory: Path | None = None,
    force: bool = False,
    task: str = "corner",
) -> dict:
    """Generate *count* samples of *task* for *split* once and write them to disk.

    Only the composited photo is stored; the rectified pair, the heatmaps and
    everything else are re-derived from it, so the evaluation set and the
    training pipeline cannot drift apart. What makes that safe is that the
    generator is a pure function of its key: ``rng_for("frozen", split, seed, i)``
    produces sample *i* identically on any machine, which the sanity checks
    verify by regenerating the set and comparing file hashes.

    *task* selects which generator options are in force, and it is part of the
    rng key: the two tasks must not be handed the same photos, because the corner
    options change what a page looks like in ways the enhancement target cannot
    account for.
    """
    # Deferred so the split and cache commands never pay for importing the
    # generator, and so this module stays free of an import cycle with it.
    from .seed import rng_for
    from .synth import build_sources

    directory = Path(directory) if directory is not None else paths.frozen_set(task, split)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not force:
        from .io import read_json

        existing = read_json(manifest_path)
        if (
            existing.get("count") == count
            and existing.get("seed") == seed
            and existing.get("task", "corner") == task
        ):
            return existing

    sources = build_sources(config, split, task=task)
    if len(sources.scans) == 0:
        raise FileNotFoundError(f"no cached scans for the {split!r} split")

    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("photo_*.jpg"):
        stale.unlink()

    entries = []
    for index in range(count):
        rng = rng_for("frozen", split, seed, index)
        sample = sources.compose(rng)
        name = f"photo_{index:04d}.jpg"
        imwrite_rgb(directory / name, sample.photo, quality=FROZEN_JPEG_QUALITY)
        entries.append(
            {
                "id": f"{split}_{index:04d}",
                "photo": name,
                "scan": sample.params["scan"],
                "corners": np.asarray(sample.corners, dtype=float).round(3).tolist(),
                "canvas": sample.params["canvas"],
                "params": sample.params,
            }
        )

    manifest = {
        "task": task,
        "split": split,
        "seed": seed,
        "count": count,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": (
            "Generated once with a fixed seed so that every epoch, and every model "
            "compared, is scored on identical images."
        ),
        # The recipe, stored alongside the result. A frozen set is only
        # reproducible from the config that produced it, and that is not
        # necessarily the default one: these buckets get re-frozen whenever the
        # generator changes, from whichever experiment motivated the change. With
        # the recipe recorded, the sanity check can regenerate a sample and
        # compare the bytes against what the set was *actually* frozen with,
        # instead of assuming a config and reporting a false alarm when the
        # assumption is wrong.
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "samples": entries,
    }
    write_json(manifest_path, manifest)
    return manifest


def freeze_eval_sets(
    config,
    seed: int | None = None,
    force: bool = False,
    tasks: list[str] | None = None,
) -> dict:
    """Freeze every bucket the config asks for, one set per task.

    The training bucket is frozen too. It is not the data the model trained on —
    with a generator that never repeats itself, no such data exists — but a fixed
    sample of the distribution it trained on, which is the only thing the brief's
    "Training" row can honestly mean *(brief §3.3)*.
    """
    data = config.get("data", {})
    seed = int(data.get("split_seed", 1234)) if seed is None else int(seed)
    tasks = list(tasks or data.get("frozen_tasks") or ["enhance", "corner"])
    counts = {
        "train": int(data.get("frozen_train_samples", 200)),
        "val": int(data.get("frozen_val_samples", 200)),
        "test": int(data.get("frozen_test_samples", 200)),
    }

    manifests: dict[str, dict] = {}
    for task in tasks:
        for split, count in counts.items():
            if count <= 0:
                continue
            manifest = freeze_split(config, split, count, seed=seed, force=force, task=task)
            manifests[f"{task}/{split}"] = manifest
            target = paths.frozen_set(task, split)
            print(f"frozen {task:<8} {split:<5}: {manifest['count']} samples -> {_relative(target)}")
    return manifests


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
