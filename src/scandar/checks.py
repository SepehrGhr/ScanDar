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
