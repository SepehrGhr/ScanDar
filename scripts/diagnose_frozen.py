#!/usr/bin/env python
"""Say *why* a frozen bucket no longer reproduces, rather than only that it does not.

    python scripts/diagnose_frozen.py                          # every bucket, the checked samples
    python scripts/diagnose_frozen.py --task corner --split val
    python scripts/diagnose_frozen.py --task enhance --split test --index 100 199

The sanity check compares encoded bytes, which is the right test and a useless
error message: a one-bit difference and a completely different photograph read
identically. This regenerates the same samples and reports, in order, the three
things that can actually be wrong — because they have different consequences and
different fixes:

**The sampling diverged.** The generator drew a different page placement, a
different background or a different degradation. The corners in the manifest and
the corners regenerated here will disagree, and the photo is a different
photograph. This is the serious one: it means the two machines are not running
the same generator, and it usually comes from a library whose sampling or
floating-point behaviour changed, tipping a rejection-sampling decision.

**The pixels differ slightly.** Same sample, same placement, but a filter rounds
differently — a different OpenCV, or a different BLAS under a warp. Reported as
the largest and mean absolute difference over the decoded image, and the share of
pixels involved. A max of 1-2 on a 0-255 scale is rounding; anything larger is a
stage behaving differently.

**Only the encoding differs.** The decoded pixels are identical and the JPEG bytes
are not, which means the same image was compressed by a different libjpeg. Numbers
measured on the stored file are unaffected, because everything reads the file.

Nothing here writes: it only reports. If it turns out the buckets really are
stale, `scripts/freeze_eval_sets.py --force` is the thing that rebuilds them —
and on a machine that is *not* the one they were frozen on, that is almost
certainly the wrong move.
"""

import argparse
import platform
import sys

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.io import imread_rgb, paths, read_json


def spread(count: int, wanted: int = 3) -> list[int]:
    """The same indices the sanity check spot-checks, so the two agree."""
    from scandar.checks import _spread

    return _spread(count, wanted)


def param_diff(stored, rebuilt, path: str = "", tolerance: float = 1e-3) -> list[str]:
    """Where two recorded parameter trees disagree, in `key.path: a vs b` form.

    Numbers are compared with a tolerance because the manifest rounds what it
    writes; anything past that tolerance is a different draw, not a different
    rounding.
    """
    if isinstance(stored, dict) and isinstance(rebuilt, dict):
        out = []
        for key in stored.keys() | rebuilt.keys():
            out += param_diff(stored.get(key), rebuilt.get(key), f"{path}.{key}" if path else key)
        return sorted(out)
    if isinstance(stored, (list, tuple)) and isinstance(rebuilt, (list, tuple)):
        if len(stored) != len(rebuilt):
            return [f"{path}: {len(stored)} values vs {len(rebuilt)}"]
        out = []
        for i, (a, b) in enumerate(zip(stored, rebuilt)):
            out += param_diff(a, b, f"{path}[{i}]")
        return out
    if isinstance(stored, (int, float)) and isinstance(rebuilt, (int, float)):
        gap = abs(float(stored) - float(rebuilt))
        return [] if gap <= tolerance else [f"{path}: {stored} vs {rebuilt}"]
    return [] if stored == rebuilt else [f"{path}: {stored!r} vs {rebuilt!r}"]


def diagnose_sample(sources, directory, entry, split, seed, index) -> dict:
    import cv2

    from scandar.prepare import FROZEN_JPEG_QUALITY
    from scandar.seed import rng_for

    rebuilt = sources.compose(rng_for("frozen", split, seed, index))

    stored_path = directory / entry["photo"]
    stored_bytes = stored_path.read_bytes()
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rebuilt.photo, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, FROZEN_JPEG_QUALITY],
    )
    bytes_match = bool(ok) and encoded.tobytes() == stored_bytes

    corners_stored = np.asarray(entry["corners"], dtype=float)
    corners_rebuilt = np.asarray(rebuilt.corners, dtype=float).round(3)
    corner_shift = float(np.abs(corners_stored - corners_rebuilt).max())

    # Decoded against decoded. Comparing the regenerated array to the stored file
    # directly would charge the difference for one JPEG round trip that the
    # stored side has already paid.
    stored_pixels = imread_rgb(stored_path).astype(np.int16)
    rebuilt_pixels = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    rebuilt_pixels = cv2.cvtColor(rebuilt_pixels, cv2.COLOR_BGR2RGB).astype(np.int16)

    if stored_pixels.shape != rebuilt_pixels.shape:
        return {
            "id": entry["id"],
            "verdict": "different canvas",
            "detail": f"{stored_pixels.shape} stored against {rebuilt_pixels.shape} regenerated",
            "bytes_match": bytes_match,
        }

    difference = np.abs(stored_pixels - rebuilt_pixels)
    max_diff = int(difference.max())
    mean_diff = float(difference.mean())
    share = float((difference > 0).mean())

    # The manifest stores every choice the generator made, so when two machines
    # disagree the first differing key names the stage that disagreed — which is
    # a far more useful answer than "the bytes differ".
    differing = param_diff(entry["params"], {**rebuilt.params, "scan": rebuilt.params.get("scan")})

    if differing:
        verdict = "sampling diverged"
        detail = f"first differences: {'; '.join(differing[:3])}"
    elif corner_shift > 0.01:
        verdict, detail = "sampling diverged", f"corners moved by up to {corner_shift:.2f} px"
    elif max_diff == 0:
        verdict = "encoding only" if not bytes_match else "identical"
        detail = "decoded pixels are identical"
    else:
        verdict = "pixels differ"
        detail = (
            f"max {max_diff}, mean {mean_diff:.4f}, {share * 100:.3f}% of pixels differ"
        )

    return {
        "id": entry["id"],
        "verdict": verdict,
        "detail": detail,
        "bytes_match": bytes_match,
    }


def diagnose(task: str, split: str, indices=None) -> list[dict]:
    from scandar.checks import _frozen_recipe
    from scandar.synth import build_sources

    directory = paths.frozen_set(task, split)
    manifest = read_json(directory / "manifest.json")
    entries = manifest["samples"]
    config, recipe = _frozen_recipe(manifest)
    if config is None:
        raise SystemExit(f"{task}/{split} records no recipe ({recipe}) — nothing to compare against")

    print(f"\n{task}/{split}  — {len(entries)} samples, frozen from {recipe}, seed {manifest['seed']}")
    sources = build_sources(config, split, task=task)
    chosen = list(indices) if indices else spread(len(entries))

    rows = []
    for index in chosen:
        row = diagnose_sample(sources, directory, entries[index], split, manifest["seed"], index)
        rows.append(row)
        mark = "ok " if row["bytes_match"] else "DIFF"
        print(f"  {mark} {row['id']:<14} {row['verdict']:<18} {row['detail']}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", nargs="*", default=["enhance", "corner"])
    parser.add_argument("--split", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--index", nargs="*", type=int, default=None,
                        help="default: the same samples the sanity check spot-checks")
    args = parser.parse_args()

    import cv2

    print("environment")
    print(f"  python {platform.python_version()} on {platform.platform()}")
    print(f"  opencv {cv2.__version__}   numpy {np.__version__}")
    try:
        import torch

        print(f"  torch  {torch.__version__}")
    except ImportError:
        pass
    print(f"  data   {paths.data}")

    rows = []
    for task in args.task:
        for split in args.split:
            if (paths.frozen_set(task, split) / "manifest.json").exists():
                rows += diagnose(task, split, args.index)

    verdicts = {row["verdict"] for row in rows if not row["bytes_match"]}
    print()
    if not verdicts:
        print("every sample checked reproduces byte-identically")
    elif verdicts <= {"encoding only"}:
        print(
            "only the JPEG encoding differs — the decoded pixels are identical, so every number "
            "measured on these files is unaffected. Do not re-freeze."
        )
    elif verdicts <= {"encoding only", "pixels differ"}:
        print(
            "the pixels differ. Read the max: 1-2 is rounding in a filter and harmless in "
            "practice; larger means a stage is behaving differently on this machine, and models "
            "trained here would not see quite the distribution the baselines saw."
        )
    else:
        print(
            "the sampling itself diverged — this machine's generator is drawing different "
            "samples, not encoding the same ones differently. Compare the environment block "
            "above against the machine the buckets were frozen on."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
