"""Sanity checks.

A single command that answers "is anything quietly wrong?". Cheap to run, run
often — most of the expensive mistakes in a project like this are silent ones: a
corner label that was not rescaled with its image, a validation set that is
regenerated differently every epoch, a scan that ended up in two splits at once.

Checks come in two severities. An **error** means something is broken and whatever
runs next will produce nonsense. A **warning** means something is merely absent — the
background photos, the reference scans, the Roboflow export — which is expected
while those are still being collected. ``--strict`` promotes warnings to errors.

    python scripts/sanity_checks.py
    scandar sanity --strict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import list_images, paths

EXPECTED_SCANS = 50
EXPECTED_REAL_PHOTOS = 19

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Result:
    name: str
    status: str
    detail: str


def _image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the file header, without decoding the pixels."""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def check_environment() -> list[Result]:
    import sys

    results = [
        Result("python", PASS, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    ]

    try:
        import numpy

        results.append(Result("numpy", PASS, numpy.__version__))
    except ImportError as exc:
        results.append(Result("numpy", FAIL, str(exc)))

    try:
        import cv2

        results.append(Result("opencv", PASS, cv2.__version__))
    except ImportError as exc:
        results.append(Result("opencv", FAIL, str(exc)))

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 2**30
            results.append(
                Result("torch + cuda", PASS, f"{torch.__version__} | {name} ({memory:.1f} GiB)")
            )
        else:
            results.append(
                Result(
                    "torch + cuda",
                    WARN,
                    f"{torch.__version__} but CUDA is unavailable — training will fall back to CPU",
                )
            )
    except ImportError as exc:
        results.append(Result("torch", FAIL, str(exc)))

    return results


def check_package() -> list[Result]:
    import scandar

    from .config import load_config

    results = [Result("import scandar", PASS, f"v{scandar.__version__} from {paths.repo}")]

    config_path = paths.repo / "configs" / "base.yaml"
    try:
        config = load_config(config_path)
        rect = config.data.rect_size
        results.append(
            Result(
                "configs/base.yaml",
                PASS,
                f"seed {config.project.seed}, rect {rect[0]}x{rect[1]}, patch {config.data.patch_size}",
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface whatever went wrong
        results.append(Result("configs/base.yaml", FAIL, f"{type(exc).__name__}: {exc}"))

    return results


def check_scans() -> list[Result]:
    files = list_images(paths.scans)
    if not files:
        return [Result("clean scans", FAIL, f"none found in {paths.scans}")]

    status = PASS if len(files) == EXPECTED_SCANS else WARN
    sizes = [_image_size(p) for p in files]
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    portrait = all(h > w for w, h in sizes)
    detail = (
        f"{len(files)} scans, {min(widths)}-{max(widths)} x {min(heights)}-{max(heights)} px"
        f"{'' if portrait else ', WARNING: not all portrait'}"
    )
    if len(files) != EXPECTED_SCANS:
        detail += f" (expected {EXPECTED_SCANS})"
    return [Result("clean scans", status, detail)]


def check_real_photos() -> list[Result]:
    results = []

    photos = list_images(paths.real_photos)
    if photos:
        sizes = {_image_size(p) for p in photos}
        status = PASS if len(photos) == EXPECTED_REAL_PHOTOS else WARN
        summary = ", ".join(f"{w}x{h}" for w, h in sorted(sizes)[:3])
        results.append(Result("real photos", status, f"{len(photos)} photos ({summary}...)"))
    else:
        results.append(Result("real photos", FAIL, f"none found in {paths.real_photos}"))

    references = list_images(paths.real_reference)
    if not references:
        results.append(
            Result(
                "reference scans",
                WARN,
                f"none yet — capture one per photo with a scanning app into {_rel(paths.real_reference)}",
            )
        )
    else:
        photo_stems = {p.stem for p in photos}
        reference_stems = {p.stem for p in references}
        missing = photo_stems - reference_stems
        extra = reference_stems - photo_stems
        if missing or extra:
            problem = []
            if missing:
                problem.append(f"{len(missing)} photo(s) without a reference: {_sample(missing)}")
            if extra:
                problem.append(f"{len(extra)} reference(s) with no photo: {_sample(extra)}")
            results.append(Result("reference scans", WARN, "; ".join(problem)))
        else:
            results.append(Result("reference scans", PASS, f"{len(references)}, filenames all match"))

    annotations = [p for p in paths.real_annotations.glob("*.json")] if paths.real_annotations.is_dir() else []
    if annotations:
        results.append(
            Result("corner annotations", PASS, ", ".join(p.name for p in annotations))
        )
    else:
        results.append(
            Result(
                "corner annotations",
                WARN,
                f"no COCO keypoint export yet in {_rel(paths.real_annotations)}",
            )
        )

    transcripts = list(paths.real_transcripts.glob("*.txt")) if paths.real_transcripts.is_dir() else []
    if transcripts:
        results.append(Result("OCR transcripts", PASS, f"{len(transcripts)} document(s)"))
    else:
        results.append(
            Result(
                "OCR transcripts",
                WARN,
                "none yet — needed for character error rate on the printed documents",
            )
        )

    return results


def check_backgrounds() -> list[Result]:
    files = list_images(paths.backgrounds)
    if not files:
        return [
            Result(
                "backgrounds",
                WARN,
                f"none yet — 20-30 background-only photos go in {_rel(paths.backgrounds)}",
            )
        ]
    status = PASS if len(files) >= 15 else WARN
    detail = f"{len(files)} photos"
    if len(files) < 15:
        detail += " (fewer than 15 limits how varied the composites can be)"
    return [Result("backgrounds", status, detail)]


def check_scan_cache() -> list[Result]:
    sources = list_images(paths.scans)
    cached = list_images(paths.scans_cache)
    if not cached:
        return [
            Result("scan cache", WARN, "not built yet — run `python scripts/prepare_data.py`")
        ]

    missing = {p.stem for p in sources} - {p.stem for p in cached}
    if missing:
        return [
            Result("scan cache", FAIL, f"{len(missing)} scan(s) not cached: {_sample(missing)}")
        ]

    # The cache must not silently distort the page: aspect ratio has to survive.
    worst_name, worst_drift = None, 0.0
    long_sides = []
    for source in sources:
        source_w, source_h = _image_size(source)
        cache_w, cache_h = _image_size(paths.scans_cache / f"{source.stem}.png")
        long_sides.append(max(cache_w, cache_h))
        drift = abs((cache_w / cache_h) - (source_w / source_h)) / (source_w / source_h)
        if drift > worst_drift:
            worst_name, worst_drift = source.stem, drift

    if worst_drift > 0.01:
        return [
            Result(
                "scan cache",
                FAIL,
                f"aspect ratio drifted by {worst_drift:.2%} on scan {worst_name}",
            )
        ]

    unique = sorted(set(long_sides))
    return [
        Result(
            "scan cache",
            PASS,
            f"{len(cached)} PNGs, long side {unique[0]}"
            + (f"-{unique[-1]}" if len(unique) > 1 else "")
            + f" px, aspect preserved (max drift {worst_drift:.3%})",
        )
    ]


def check_splits() -> list[Result]:
    if not paths.splits.exists():
        return [Result("splits", WARN, "not built yet — run `python scripts/prepare_data.py`")]

    from .io import read_json

    splits = read_json(paths.splits)
    scans = splits.get("scans", {})
    train, val, test = set(scans.get("train", [])), set(scans.get("val", [])), set(scans.get("test", []))

    # The brief's central data rule: no source scan on two sides of the split.
    overlaps = []
    for a_name, a, b_name, b in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        shared = a & b
        if shared:
            overlaps.append(f"{a_name}/{b_name}: {_sample(shared)}")
    if overlaps:
        return [Result("splits", FAIL, "source scans appear in two splits — " + "; ".join(overlaps))]

    available = {p.stem for p in list_images(paths.scans)}
    covered = train | val | test
    if covered != available:
        detail = []
        if available - covered:
            detail.append(f"{len(available - covered)} scan(s) in no split: {_sample(available - covered)}")
        if covered - available:
            detail.append(f"{len(covered - available)} split entr(ies) with no file: {_sample(covered - available)}")
        return [Result("splits", FAIL, "; ".join(detail))]

    results = [
        Result(
            "splits",
            PASS,
            f"{len(train)}/{len(val)}/{len(test)} scans, disjoint and complete (seed {splits.get('seed')})",
        )
    ]

    backgrounds = splits.get("backgrounds", {})
    bg_train, bg_heldout = set(backgrounds.get("train", [])), set(backgrounds.get("heldout", []))
    if bg_train or bg_heldout:
        shared = bg_train & bg_heldout
        if shared:
            results.append(Result("background split", FAIL, f"overlap: {_sample(shared)}"))
        else:
            results.append(
                Result("background split", PASS, f"{len(bg_train)} train / {len(bg_heldout)} held out")
            )
    return results


# ---------------------------------------------------------------------------
# the synthetic generator
# ---------------------------------------------------------------------------
#: How far the rectified degraded input may sit from the clean target before the
#: pair stops being a fair supervision signal. Sub-pixel, because the brief warns
#: twice that misalignment punishes the model for errors it did not make.
MAX_ALIGNMENT_SHIFT_PX = 0.5

#: Mean Sobel magnitude of the degraded input over that of the clean target. The
#: brief warns against degradation so heavy it destroys the text; measured over
#: the shipped ranges this sits near 0.21 (0.30 mild, 0.18 hard), so a sample
#: below this floor means the pipeline has stopped leaving anything to restore.
MIN_DETAIL_RATIO = 0.04

GENERATOR_SAMPLES = 8


def _generator_sources(task: str = "enhance"):
    from .config import load_config
    from .synth import build_sources

    config = load_config(paths.repo / "configs" / "base.yaml")
    return build_sources(config, "train", task=task), config


def check_corner_ordering() -> list[Result]:
    """TL, TR, BR, BL, from any starting point and any winding."""
    import numpy as np

    from .geometry import order_corners

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        angle = rng.uniform(-np.pi / 5, np.pi / 5)
        half = rng.uniform(40, 300, size=2)
        quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * half
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        quad = quad @ rotation.T + rng.uniform(400, 800, size=2)
        quad += rng.uniform(-12, 12, size=(4, 2))

        canonical = order_corners(quad)
        for permutation in (rng.permutation(4), rng.permutation(4), [3, 2, 1, 0]):
            shuffled = order_corners(quad[list(permutation)])
            worst = max(worst, float(np.abs(shuffled - canonical).max()))

    if worst > 1e-4:
        return [Result("corner ordering", FAIL, f"permuting the input moved a corner by {worst:.3g}")]
    return [Result("corner ordering", PASS, "canonical under any permutation of 200 random quads")]


def check_generator() -> list[Result]:
    """Generate a handful of samples and audit what comes out."""
    import numpy as np

    from .geometry import normalize_corners, quad_problem, resize_with_corners
    from .seed import rng_for

    results: list[Result] = []
    try:
        sources, config = _generator_sources("enhance")
    except FileNotFoundError as exc:
        return [Result("synthetic generator", WARN, str(exc))]
    if len(sources.scans) == 0:
        return [Result("synthetic generator", WARN, "no cached scans to composite")]

    rect_size = tuple(config.data.rect_size)
    shifts, ratios, problems = [], [], []
    normalised_range = [1.0, 0.0]
    resize_drift = 0.0

    for index in range(GENERATOR_SAMPLES):
        sample = sources.compose(rng_for("sanity", index), keep_clean=True)
        canvas = sample.canvas_size

        problem = quad_problem(sample.corners, canvas)
        if problem is not None:
            problems.append(problem)

        degraded, target = sample.rectify(rect_size)
        aligned, _ = sample.rectify(rect_size, source=sample.clean_photo)
        shifts.append(alignment_shift(aligned, target))
        ratios.append(_detail(degraded) / max(_detail(target), 1e-6))

        # A resize must move the corners with the image: normalised coordinates
        # are the invariant, so they must come back unchanged.
        resized, moved = resize_with_corners(sample.photo, sample.corners, (256, 256))
        before = normalize_corners(sample.corners, canvas)
        after = normalize_corners(moved, (resized.shape[1], resized.shape[0]))
        resize_drift = max(resize_drift, float(np.abs(before - after).max()))
        normalised_range = [
            min(normalised_range[0], float(after.min())),
            max(normalised_range[1], float(after.max())),
        ]

    if problems:
        results.append(
            Result("page placement", FAIL, f"{len(problems)} invalid quad(s): {problems[0]}")
        )
    else:
        results.append(
            Result("page placement", PASS, f"{GENERATOR_SAMPLES} valid quads, ordered and in frame")
        )

    worst_shift = max(shifts)
    status = PASS if worst_shift <= MAX_ALIGNMENT_SHIFT_PX else FAIL
    results.append(
        Result(
            "rectification alignment",
            status,
            f"input vs target off by {worst_shift:.2f} px at worst "
            f"(mean {sum(shifts) / len(shifts):.2f}, limit {MAX_ALIGNMENT_SHIFT_PX})",
        )
    )

    worst_ratio = min(ratios)
    status = PASS if worst_ratio >= MIN_DETAIL_RATIO else FAIL
    results.append(
        Result(
            "degradation severity",
            status,
            f"{sum(ratios) / len(ratios):.0%} of the target's edge energy survives on average, "
            f"{worst_ratio:.0%} at worst (floor {MIN_DETAIL_RATIO:.0%})",
        )
    )

    low, high = normalised_range
    status = PASS if 0.0 <= low and high <= 1.0 else FAIL
    results.append(Result("normalised corners", status, f"in [{low:.3f}, {high:.3f}]"))

    status = PASS if resize_drift <= 1e-3 else FAIL
    results.append(
        Result(
            "resize consistency",
            status,
            f"normalised corners moved by {resize_drift:.1e} across a resize to 256x256",
        )
    )

    # Same key, same sample — the property the frozen sets and resuming rest on.
    first = sources.compose(rng_for("sanity", 0)).photo
    second = sources.compose(rng_for("sanity", 0)).photo
    other = sources.compose(rng_for("sanity", 1)).photo
    if not np.array_equal(first, second):
        results.append(Result("determinism", FAIL, "the same key produced two different photos"))
    elif np.array_equal(first, other):
        results.append(Result("determinism", FAIL, "two different keys produced the same photo"))
    else:
        results.append(Result("determinism", PASS, "same key same photo, different key different photo"))

    return results


def check_dataset_tensors() -> list[Result]:
    """What a training loop will actually receive."""
    import numpy as np

    try:
        from .datasets import SyntheticCornerDataset, SyntheticEnhanceDataset
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        return [Result("datasets", FAIL, str(exc))]

    try:
        sources, config = _generator_sources("corner")
    except FileNotFoundError as exc:
        return [Result("datasets", WARN, str(exc))]

    data = config.data
    corner = SyntheticCornerDataset(
        sources,
        "train",
        length=2,
        input_size=data.corner_input,
        heatmap_size=data.heatmap_size,
        heatmap_sigma=data.heatmap_sigma,
    )
    item = corner[0]
    image, corners, heatmaps = item["image"], item["corners"], item["heatmaps"]

    problems = []
    if tuple(image.shape) != (3, data.corner_input, data.corner_input):
        problems.append(f"image is {tuple(image.shape)}, expected CHW 3x{data.corner_input}²")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        problems.append(f"image is outside [0, 1]: [{image.min():.2f}, {image.max():.2f}]")
    if tuple(corners.shape) != (4, 2):
        problems.append(f"corners are {tuple(corners.shape)}, expected (4, 2)")
    if float(corners.min()) < 0.0 or float(corners.max()) > 1.0:
        problems.append("corners are not normalised into [0, 1]")
    if tuple(heatmaps.shape) != (4, data.heatmap_size, data.heatmap_size):
        problems.append(f"heatmaps are {tuple(heatmaps.shape)}")

    # Every heatmap must peak on its own corner, or approach B is training
    # against labels that point somewhere else.
    for index in range(4):
        peak = np.unravel_index(int(heatmaps[index].argmax()), heatmaps[index].shape)
        expected = (
            corners[index, 0].item() * data.heatmap_size - 0.5,
            corners[index, 1].item() * data.heatmap_size - 0.5,
        )
        if abs(peak[1] - expected[0]) > 1.0 or abs(peak[0] - expected[1]) > 1.0:
            problems.append(f"heatmap {index} peaks at {peak[::-1]}, corner is at {expected}")

    results = [
        Result("corner dataset", FAIL if problems else PASS, "; ".join(problems) if problems else
               f"image {tuple(image.shape)}, corners (4, 2) in [0, 1], "
               f"{data.heatmap_size}² heatmaps peaking on their corners")
    ]

    enhance = SyntheticEnhanceDataset(
        sources,
        "train",
        length=4,
        patch_size=data.patch_size,
        rect_size=tuple(data.rect_size),
        patches_per_photo=2,
    )
    pair = enhance[0]
    shape = (3, data.patch_size, data.patch_size)
    if tuple(pair["input"].shape) != shape or tuple(pair["target"].shape) != shape:
        results.append(
            Result("enhance dataset", FAIL, f"expected {shape}, got {tuple(pair['input'].shape)}")
        )
    else:
        shared = enhance[1]["scan"] == pair["scan"]
        results.append(
            Result(
                "enhance dataset",
                PASS,
                f"{shape} patch pairs, {enhance.patches_per_photo} per composited photo"
                + ("" if shared else " (WARNING: the photo cache did not hit)"),
            )
        )
    return results


def check_frozen_sets() -> list[Result]:
    """The frozen evaluation sets, and whether they still match their recipe."""
    from .io import read_json
    from .seed import rng_for
    from .synth import build_sources

    results = []
    for split in ("val", "test"):
        directory = paths.data / "frozen" / split
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            results.append(
                Result(
                    f"frozen {split}",
                    WARN,
                    "not built yet — run `python scripts/freeze_eval_sets.py`",
                )
            )
            continue

        manifest = read_json(manifest_path)
        entries = manifest["samples"]
        missing = [e["photo"] for e in entries if not (directory / e["photo"]).exists()]
        if missing:
            results.append(
                Result(f"frozen {split}", FAIL, f"{len(missing)} photo(s) missing: {_sample(missing)}")
            )
            continue

        # Regenerate a few samples from their keys and compare the encoded bytes.
        # Byte equality is the strongest form of this check and the cheapest: it
        # catches a change anywhere in the generator, including one that leaves the
        # corner labels untouched and only moves the pixels.
        try:
            import cv2

            from .config import load_config
            from .prepare import FROZEN_JPEG_QUALITY

            config = load_config(paths.repo / "configs" / "base.yaml")
            sources = build_sources(config, split, task="corner")
            mismatched = []
            for index in _spread(len(entries), 3):
                entry = entries[index]
                rebuilt = sources.compose(rng_for("frozen", split, manifest["seed"], index))
                ok, encoded = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(rebuilt.photo, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, FROZEN_JPEG_QUALITY],
                )
                if not ok or encoded.tobytes() != (directory / entry["photo"]).read_bytes():
                    mismatched.append(entry["id"])
        except Exception as exc:  # noqa: BLE001 - report rather than crash the report
            results.append(Result(f"frozen {split}", WARN, f"could not verify: {exc}"))
            continue

        if mismatched:
            results.append(
                Result(
                    f"frozen {split}",
                    FAIL,
                    f"{', '.join(mismatched)} no longer regenerate byte-identically — the "
                    "generator has changed since this set was frozen, so numbers measured on "
                    "it are not comparable with new ones. Re-freeze it with --force",
                )
            )
        else:
            results.append(
                Result(
                    f"frozen {split}",
                    PASS,
                    f"{len(entries)} samples, seed {manifest['seed']}, spot-checked "
                    "byte-identical on regeneration",
                )
            )
    return results


def _spread(count: int, wanted: int) -> list[int]:
    """A few indices spread across a range, always including the first and the last."""
    if count <= wanted:
        return list(range(count))
    step = (count - 1) / (wanted - 1)
    return sorted({int(round(index * step)) for index in range(wanted)})


def alignment_shift(image, reference) -> float:
    """Sub-pixel translation between two images, in pixels.

    Phase correlation measures the residual shift between the rectified input and
    the clean target directly, which is the thing that matters: an *intensity*
    difference between them is the whole point of the pair, but a *positional*
    one is a bug.

    ``cv2.phaseCorrelate`` is not exactly unbiased at every image shape — at some
    sizes it reports half a pixel for an image correlated against itself — so the
    estimator's own baseline is measured on the reference and subtracted. Without
    that correction this check reports a constant offset that has nothing to do
    with the generator.
    """
    import cv2
    import numpy as np

    first = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    second = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float64)
    (dx, dy), _ = cv2.phaseCorrelate(first, second)
    (bias_x, bias_y), _ = cv2.phaseCorrelate(second, second)
    return float(np.hypot(dx - bias_x, dy - bias_y))


def _detail(image) -> float:
    """Mean Sobel magnitude — how much edge energy an image still carries."""
    import cv2
    import numpy as np

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.abs(gradient_x).mean() + np.abs(gradient_y).mean())


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
CHECKS = (
    ("environment", check_environment),
    ("package", check_package),
    ("data: scans", check_scans),
    ("data: backgrounds", check_backgrounds),
    ("data: real (evaluation only)", check_real_photos),
    ("derived: scan cache", check_scan_cache),
    ("derived: splits", check_splits),
    ("geometry", check_corner_ordering),
    ("synthetic generator", check_generator),
    ("dataset contracts", check_dataset_tensors),
    ("derived: frozen evaluation sets", check_frozen_sets),
)

_SYMBOL = {PASS: "✓", WARN: "!", FAIL: "✗"}


def run(strict: bool = False) -> bool:
    """Run every check, print a report, return True if nothing failed."""
    results: list[Result] = []
    print(f"ScanDar sanity checks   (data root: {paths.data})\n")

    for group, check in CHECKS:
        print(f"  {group}")
        try:
            group_results = check()
        except Exception as exc:  # noqa: BLE001 - a crashing check is itself a failure
            group_results = [Result(group, FAIL, f"{type(exc).__name__}: {exc}")]
        for result in group_results:
            print(f"    {_SYMBOL[result.status]} {result.name:<22} {result.detail}")
        results.extend(group_results)
        print()

    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    print(
        f"  {len(results) - len(failures) - len(warnings)} passed, "
        f"{len(warnings)} warning(s), {len(failures)} failure(s)"
    )
    if warnings and not strict:
        print("  warnings are expected while data collection is still in progress")

    ok = not failures and (not warnings or not strict)
    print("\n  " + ("all good" if ok else "needs attention"))
    return ok


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(paths.repo))
    except ValueError:
        return str(path)


def _sample(items, limit: int = 4) -> str:
    ordered = sorted(items)[:limit]
    suffix = ", ..." if len(items) > limit else ""
    return ", ".join(str(i) for i in ordered) + suffix
