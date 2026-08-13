"""Dataset classes.  *(brief §2)*

``SyntheticEnhanceDataset``
    Composites a fresh sample per ``__getitem__``: 256x256 patches cut from pages
    rectified at 1024x1448 for training, whole pages for evaluation. A practically
    infinite training set that never touches disk.
``SyntheticCornerDataset``
    The raw synthetic photo plus its corner labels, both as normalised coordinates
    and as four Gaussian heatmaps, so approaches A and B train off one dataset.
``FrozenSyntheticDataset``
    Reads the train, validation and test samples that were generated once with a
    fixed seed. Freezing them is what makes the validation curve measure the model
    instead of the dice. For the enhancement task it serves whole rectified pages
    for the evaluation table and deterministic patches for the per-epoch
    validation curve.

Corners are normalised to [0, 1] by image width and height, which makes the
detection task resolution-independent, and are *always* rescaled together with
their image: a corner label that is not transformed with its image is a wrong
label. Both happen inside :func:`~scandar.geometry.resize_with_corners`, which is
one function precisely so that no caller can do half of it.

Images stop at [0, 1] and are not standardised by a mean and a standard deviation.
The restoration network's target *is* an image in [0, 1] behind a sigmoid, so its
input belongs in the same space; and with no pretrained weights in the project
there is no external normalisation to match. Saying so is worth more than
copying a ``transforms.Normalize`` line whose constants would mean nothing here.

The real-photo evaluation set is not wired up yet: it needs the Roboflow keypoint
export, which has not landed. Once it does, this module gains a fourth dataset
that rectifies each photo with its annotated corners for the enhancement network
and hands the raw photo with scaled corners to the detector — and it will never
run the degradation pipeline, because those photos arrive degraded by reality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import (
    denormalize_corners,
    homography,
    normalize_corners,
    rect_corners,
    resize_with_corners,
)
from .io import imread_rgb, read_json
from .prepare import ScanBank
from .seed import rng_for
from .synth import Sample, Sources, build_sources  # noqa: F401  (re-exported for callers)

# ---------------------------------------------------------------------------
# tensor plumbing
# ---------------------------------------------------------------------------
def to_tensor(image: np.ndarray) -> torch.Tensor:
    """RGB uint8 HWC -> float32 CHW in [0, 1], which is what the models consume."""
    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).to(torch.float32).div_(255.0)


def gaussian_heatmaps(corners_normalised, size, sigma: float) -> np.ndarray:
    """Four ``(height, width)`` maps, each a Gaussian blob on one corner *(brief §5)*.

    Peaks are placed at the sub-pixel position of the corner rather than at the
    nearest cell, so that a soft-argmax read back off the target reproduces the
    label instead of a quantised version of it.
    """
    width, height = int(size[0]), int(size[1])
    centres = denormalize_corners(corners_normalised, (width, height))
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]

    maps = np.empty((len(centres), height, width), dtype=np.float32)
    for index, (cx, cy) in enumerate(centres):
        squared = (xs - float(cx)) ** 2 + (ys - float(cy)) ** 2
        maps[index] = np.exp(-squared / (2.0 * sigma * sigma))
    return maps


# ---------------------------------------------------------------------------
# the synthetic datasets
# ---------------------------------------------------------------------------
class _SyntheticDataset(Dataset):
    """Shared machinery: a fresh composite per index, reproducible from its key.

    ``__getitem__`` derives its generator from ``(seed, task, split, epoch,
    index)`` and never touches the global random state. Two consequences the
    project depends on: an interrupted run resumes onto exactly the samples it
    would have seen, and a dataloader with eight workers cannot hand back the
    same "random" degradation eight times.
    """

    task = "synthetic"

    def __init__(
        self,
        sources: Sources,
        split: str,
        length: int,
        seed: int = 1234,
        epoch: int = 0,
    ) -> None:
        self.sources = sources
        self.split = split
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        """Move to a new stream of samples. Call it before iterating, per epoch."""
        self.epoch = int(epoch)

    def rng_for_index(self, index: int):
        return rng_for(self.seed, self.task, self.split, self.epoch, index)


class SyntheticEnhanceDataset(_SyntheticDataset):
    """(degraded input, clean target) pairs for the enhancement network *(brief §3)*.

    ``mode="patch"`` cuts one 256x256 crop out of the page rectified at
    1024x1448; ``mode="page"`` returns the whole rectified page. Patches are the
    training mode and the single most important architectural call in the
    project: a whole A4 page squeezed into 256x256 leaves a pen stroke under a
    pixel wide, and no loss function recovers what the resize threw away. The
    network is fully convolutional, so it trains on crops and infers on pages.

    **``patches_per_photo`` and why the loader must not shuffle.** Composing and
    degrading a 1152x1536 photo costs around 200 ms; cutting a 256x256 patch out
    of it costs two. Generating a whole photo per patch would leave the GPU idle
    98% of the time — fatal on a two-core Colab runtime. So consecutive indices
    are grouped: ``patches_per_photo`` of them share one composited photo, cut at
    different places on the page. Shuffling is pointless here anyway — every
    index invents a fresh random sample, so there is no fixed order to break —
    and with ``shuffle=False`` each worker receives a contiguous run of indices,
    which is exactly what makes the one-entry cache hit. Set it to 1 to get one
    photo per patch, at four to eight times the cost.
    """

    task = "enhance"

    def __init__(
        self,
        sources: Sources,
        split: str = "train",
        length: int = 6400,
        *,
        mode: str = "patch",
        patch_size: int = 256,
        rect_size=(1024, 1448),
        patches_per_photo: int = 4,
        patch_tries: int = 4,
        min_patch_std: float = 0.045,
        seed: int = 1234,
        epoch: int = 0,
    ) -> None:
        super().__init__(sources, split, length, seed=seed, epoch=epoch)
        if mode not in ("patch", "page"):
            raise ValueError(f"mode must be 'patch' or 'page', not {mode!r}")
        self.mode = mode
        self.patch_size = int(patch_size)
        self.rect_size = (int(rect_size[0]), int(rect_size[1]))
        self.patches_per_photo = max(1, int(patches_per_photo)) if mode == "patch" else 1
        self.patch_tries = int(patch_tries)
        self.min_patch_std = float(min_patch_std)
        self._cached: tuple[tuple, Sample] | None = None

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._cached = None

    def _photo_for(self, index: int) -> Sample:
        """The composited photo index *index* belongs to, reusing the last one."""
        group = index // self.patches_per_photo
        key = (self.seed, self.split, self.epoch, group)
        if self._cached is not None and self._cached[0] == key:
            return self._cached[1]
        sample = self.sources.compose(rng_for(self.task, *key))
        self._cached = (key, sample)
        return sample

    def __getitem__(self, index: int) -> dict:
        sample = self._photo_for(index)

        if self.mode == "patch":
            # The photo is shared across the group; where the patch lands is not.
            degraded, target, box = sample.random_patch(
                self.rng_for_index(index),
                self.patch_size,
                self.rect_size,
                tries=self.patch_tries,
                min_std=self.min_patch_std,
            )
        else:
            degraded, target = sample.rectify(self.rect_size)
            box = (0, 0, 0)

        return {
            "input": to_tensor(degraded),
            "target": to_tensor(target),
            "scan": sample.params["scan"],
            "box": torch.tensor(box, dtype=torch.int32),
            "index": index,
        }


class SyntheticCornerDataset(_SyntheticDataset):
    """The raw photo and its four corners, as coordinates *and* heatmaps *(brief §5)*.

    Both formulations the brief asks for are trained from this one dataset, on
    identical samples, so the comparison between direct regression and heatmap
    regression is a comparison of the models and nothing else.
    """

    task = "corner"

    def __init__(
        self,
        sources: Sources,
        split: str = "train",
        length: int = 6400,
        *,
        input_size: int = 256,
        heatmap_size: int = 128,
        heatmap_sigma: float = 3.0,
        seed: int = 1234,
        epoch: int = 0,
    ) -> None:
        super().__init__(sources, split, length, seed=seed, epoch=epoch)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)

    def __getitem__(self, index: int) -> dict:
        rng = self.rng_for_index(index)
        sample = self.sources.compose(rng)
        return corner_item(
            sample.photo,
            sample.corners,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            heatmap_sigma=self.heatmap_sigma,
            extra={"scan": sample.params["scan"], "index": index},
        )


def corner_item(
    photo: np.ndarray,
    corners: np.ndarray,
    input_size: int,
    heatmap_size: int,
    heatmap_sigma: float,
    extra: dict | None = None,
) -> dict:
    """Turn one photo and its corner labels into a batch entry.

    Shared by the synthetic dataset, the frozen sets and — once the annotations
    land — the real photos, so all three are preprocessed identically. The
    original size travels with the sample: predictions have to be mapped back to
    the photo they came from, and guessing that afterwards is how corners end up
    scaled by the wrong factor.
    """
    height, width = photo.shape[:2]
    resized, moved = resize_with_corners(photo, corners, (input_size, input_size))
    normalised = normalize_corners(moved, (input_size, input_size))

    item = {
        "image": to_tensor(resized),
        "corners": torch.from_numpy(normalised.astype(np.float32)),
        "heatmaps": torch.from_numpy(
            gaussian_heatmaps(normalised, (heatmap_size, heatmap_size), heatmap_sigma)
        ),
        "corners_px": torch.from_numpy(np.asarray(corners, dtype=np.float32)),
        "size": torch.tensor([width, height], dtype=torch.int32),
    }
    item.update(extra or {})
    return item


# ---------------------------------------------------------------------------
# the frozen evaluation sets
# ---------------------------------------------------------------------------
class FrozenSyntheticDataset(Dataset):
    """Evaluation samples generated once with a fixed seed and written to disk.

    Only the composited photo is stored. Everything else — the rectified input,
    the clean target, the heatmaps — is *derived* from it with the recorded
    corners and the cached scan, exactly as the on-the-fly datasets derive them.
    That keeps the frozen set to one file per sample, and it guarantees that the
    evaluation path and the training path cannot drift apart, because they are
    the same code.

    Frozen sets live under ``data/frozen/<task>/<split>/`` and are **generated per
    task**. The corner detector's samples deliberately include tinted page stock,
    a distractor sheet and pages that will not lie flat; scoring the enhancement
    network on those would ask it to invert a colour cast and unbend a page it was
    never trained to, and would put a hard ceiling on its measured PSNR that has
    nothing to do with how well it restores a document.

    ``mode`` applies to the enhancement task. ``"page"`` returns the whole
    rectified page, which is what the report's table is computed on. ``"patch"``
    cuts ``patches_per_page`` deterministic crops out of each page, which is what
    the per-epoch validation curve uses — the training loss is a patch loss, and
    plotting it against a full-page validation loss would be comparing two
    different quantities on one axis *(brief §3.2)*. The crop positions come from
    :func:`~scandar.seed.rng_for` keyed on the sample's own id, so they are as
    frozen as the photos are.
    """

    def __init__(
        self,
        directory: Path | str,
        task: str = "enhance",
        *,
        scans: ScanBank | None = None,
        rect_size=(1024, 1448),
        input_size: int = 256,
        heatmap_size: int = 128,
        heatmap_sigma: float = 3.0,
        mode: str = "page",
        patch_size: int = 256,
        patches_per_page: int = 2,
        patch_tries: int = 4,
        min_patch_std: float = 0.045,
    ) -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found — run `python scripts/freeze_eval_sets.py` first"
            )
        manifest = read_json(manifest_path)
        if manifest.get("task", "corner") != task:
            raise ValueError(
                f"{manifest_path} holds {manifest.get('task', 'corner')!r} samples, "
                f"not {task!r} — the two tasks have separate frozen sets"
            )
        if mode not in ("page", "patch"):
            raise ValueError(f"mode must be 'page' or 'patch', not {mode!r}")

        self.entries = manifest["samples"]
        self.manifest = manifest
        self.task = task
        self.scans = scans if scans is not None else ScanBank()
        self.rect_size = (int(rect_size[0]), int(rect_size[1]))
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)
        self.mode = mode
        self.patch_size = int(patch_size)
        self.patches_per_page = max(1, int(patches_per_page)) if mode == "patch" else 1
        self.patch_tries = int(patch_tries)
        self.min_patch_std = float(min_patch_std)
        self._cached: tuple[int, Sample] | None = None

    def __len__(self) -> int:
        return len(self.entries) * self.patches_per_page

    def sample_at(self, index: int) -> Sample:
        """Rebuild the :class:`~scandar.synth.Sample` behind entry *index*.

        The last page decoded is kept, because in patch mode consecutive indices
        ask for different crops of the same one and re-reading the JPEG each time
        would dominate the cost of validating.
        """
        if self._cached is not None and self._cached[0] == index:
            return self._cached[1]

        entry = self.entries[index]
        photo = imread_rgb(self.directory / entry["photo"])
        scan = self.scans.load(entry["scan"])
        corners = np.asarray(entry["corners"], dtype=np.float32)
        scan_height, scan_width = scan.shape[:2]
        sample = Sample(
            photo=photo,
            corners=corners,
            scan=scan,
            H=homography(rect_corners(scan_width, scan_height), corners),
            params={"scan": entry["scan"], "id": entry["id"]},
        )
        self._cached = (index, sample)
        return sample

    def __getitem__(self, index: int) -> dict:
        entry_index, patch_index = divmod(index, self.patches_per_page)
        sample = self.sample_at(entry_index)
        entry = self.entries[entry_index]

        if self.task == "enhance":
            if self.mode == "patch":
                degraded, target, box = sample.random_patch(
                    rng_for("frozen-patch", entry["id"], patch_index),
                    self.patch_size,
                    self.rect_size,
                    tries=self.patch_tries,
                    min_std=self.min_patch_std,
                )
            else:
                degraded, target = sample.rectify(self.rect_size)
                box = (0, 0, 0)
            return {
                "input": to_tensor(degraded),
                "target": to_tensor(target),
                "scan": entry["scan"],
                "id": entry["id"],
                "box": torch.tensor(box, dtype=torch.int32),
                "index": index,
            }

        return corner_item(
            sample.photo,
            sample.corners,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            heatmap_sigma=self.heatmap_sigma,
            extra={"scan": entry["scan"], "id": entry["id"], "index": index},
        )


def frozen_dataset(
    config,
    split: str,
    task: str = "enhance",
    mode: str = "page",
    scans: ScanBank | None = None,
) -> FrozenSyntheticDataset:
    """The frozen bucket for *task*/*split*, sized by the config.

    One place decides how a config becomes an evaluation set, so the trainer's
    validation set and the evaluator's test set cannot be built two subtly
    different ways.
    """
    from .io import paths

    data = config.get("data", {})
    return FrozenSyntheticDataset(
        paths.frozen_set(task, split),
        task=task,
        scans=scans,
        rect_size=tuple(data.get("rect_size", (1024, 1448))),
        input_size=int(data.get("corner_input", 256)),
        heatmap_size=int(data.get("heatmap_size", 128)),
        heatmap_sigma=float(data.get("heatmap_sigma", 3.0)),
        mode=mode,
        patch_size=int(data.get("patch_size", 256)),
        patches_per_page=int(data.get("frozen_patches_per_page", 2)),
        patch_tries=int(data.get("patch_tries", 4)),
        min_patch_std=float(data.get("min_patch_std", 0.045)),
    )
