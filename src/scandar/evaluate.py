"""Evaluation.  *(brief §3.3 and §5)*

The brief names this file explicitly. It produces the numbers the report is built
on.

    python evaluate.py --checkpoint outputs/runs/enhance_baseline/best.pt
    python evaluate.py --config configs/enhance.yaml --limit 20

Built so far: the restoration table. PSNR and SSIM on the synthetic **training,
validation and test** buckets, on whole pages rectified at 1024x1448, with the
degraded-input baseline computed *first*. Each row answers a different question.
Training says how well the model fit what it was shown — a poor number there is a
capacity or optimisation problem, not a generalisation one. Validation is the
number the run was steered on, so it is optimistic by construction. Test is the
honest headline: source scans and background surfaces the model has never seen in
any form. A large training-to-test gap is overfitting; small gaps with poor
numbers everywhere are underfitting. And if the model's scores are not clearly
above the do-nothing baseline, it is not earning its parameters.

The "training" row is measured on a frozen sample of the training *distribution*,
not on data the model literally saw. With a generator that invents a fresh sample
every time it is asked, no sample is ever seen twice, so there is nothing else
the row could honestly mean — and it still answers the question it is there for,
because it holds the training scans and the training backgrounds.

Pages are restored through exactly the pipeline that runs at inference time —
overlapping tiles, cosine-blended *(brief §3.4)* — so the table scores what the
project actually ships rather than a more convenient variant of it.

Not built yet: the corner-detection numbers (localisation error, the
all-four-within-a-threshold success rate, quad IoU) and the OCR readability
comparison against the commercial scanning app. Both arrive with the tasks they
belong to; the OCR one also needs the reference scans and transcripts, which have
not been captured yet.

Tables are written to ``reports/tables/`` in both CSV and Markdown, so the report
never contains a hand-copied number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config, load_config, parse_override
from .datasets import frozen_dataset
from .device import amp_enabled, describe_device, get_device, recommended_workers
from .io import paths, write_json
from .metrics import MetricAccumulator, psnr, ssim_metric
from .model import clamp_image, load_model
from .pipelines import tiled_forward

__all__ = ["evaluate_enhancement", "main"]

SPLIT_LABELS = {"train": "Training", "val": "Validation", "test": "Test"}


def evaluate_enhancement(
    model,
    config: Config,
    splits=("train", "val", "test"),
    device=None,
    tile: int = 512,
    overlap: int = 192,
    limit: int | None = None,
    amp: bool | None = None,
    progress: bool = True,
) -> dict:
    """Score whole rectified pages, with the do-nothing baseline alongside.

    Returns ``{split: {"input": {...}, "model": {...}, "per_sample": [...]}}``.
    The baseline costs nothing extra — the degraded input is already in the batch
    — and computing it in the same loop, on the same images, is what makes the
    comparison exact rather than approximately fair.
    """
    device = device if device is not None else get_device()
    amp = amp_enabled(device) if amp is None else (amp and amp_enabled(device))
    workers = recommended_workers(config.train.get("num_workers", "auto"))
    model.eval().to(device)

    results: dict[str, dict] = {}
    for split in splits:
        dataset = frozen_dataset(config, split, task="enhance", mode="page")
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )

        baseline = MetricAccumulator()
        scored = MetricAccumulator()
        per_sample = []

        for index, batch in enumerate(loader):
            if limit is not None and index >= limit:
                break
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # The baseline first, as the brief asks — and on the very same pages.
            input_psnr = psnr(inputs, targets, reduction="none")
            input_ssim = ssim_metric(inputs, targets, reduction="none")
            baseline.add("psnr", input_psnr)
            baseline.add("ssim", input_ssim)

            with torch.no_grad():
                if tile and tile > 0:
                    restored = tiled_forward(
                        model, inputs.cpu(), tile=tile, overlap=overlap, device=device, amp=amp
                    ).to(device)
                else:
                    with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                        restored = model(inputs)
                    restored = clamp_image(restored.float())

            model_psnr = psnr(restored, targets, reduction="none")
            model_ssim = ssim_metric(restored, targets, reduction="none")
            scored.add("psnr", model_psnr)
            scored.add("ssim", model_ssim)

            per_sample.append(
                {
                    "id": batch["id"][0],
                    "scan": batch["scan"][0],
                    "input_psnr": round(float(input_psnr[0]), 4),
                    "input_ssim": round(float(input_ssim[0]), 4),
                    "psnr": round(float(model_psnr[0]), 4),
                    "ssim": round(float(model_ssim[0]), 4),
                }
            )
            if progress and (index + 1) % 25 == 0:
                print(
                    f"  {split:<5} {index + 1:>4}/{limit or len(dataset)}"
                    f"  psnr {scored.mean('psnr'):.2f} dB  (input {baseline.mean('psnr'):.2f})"
                )

        results[split] = {
            "n": len(per_sample),
            "input": baseline.summary(),
            "model": scored.summary(),
            "per_sample": per_sample,
        }
    return results


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def table_rows(results: dict) -> list[dict]:
    """Long-form rows: one per split per variant, which is what a CSV wants."""
    rows = []
    for split, entry in results.items():
        for variant, key in (("degraded input", "input"), ("enhanced", "model")):
            scores = entry[key]
            rows.append(
                {
                    "split": SPLIT_LABELS.get(split, split),
                    "variant": variant,
                    "n": entry["n"],
                    "psnr": round(scores.get("psnr", float("nan")), 3),
                    "psnr_std": round(scores.get("psnr_std", float("nan")), 3),
                    "ssim": round(scores.get("ssim", float("nan")), 4),
                    "ssim_std": round(scores.get("ssim_std", float("nan")), 4),
                }
            )
    return rows


def markdown_table(results: dict, title: str = "") -> str:
    """The brief's table, with the do-nothing baseline on the line above it."""
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        "| Split | PSNR (dB) | SSIM | baseline PSNR | baseline SSIM | gain |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, entry in results.items():
        model, baseline = entry["model"], entry["input"]
        gain = model.get("psnr", float("nan")) - baseline.get("psnr", float("nan"))
        lines.append(
            f"| {SPLIT_LABELS.get(split, split)} "
            f"| {model.get('psnr', float('nan')):.2f} ± {model.get('psnr_std', float('nan')):.2f} "
            f"| {model.get('ssim', float('nan')):.4f} ± {model.get('ssim_std', float('nan')):.4f} "
            f"| {baseline.get('psnr', float('nan')):.2f} "
            f"| {baseline.get('ssim', float('nan')):.4f} "
            f"| **{gain:+.2f} dB** |"
        )
    lines += [
        "",
        "Whole pages rectified at 1024x1448, restored in cosine-blended overlapping tiles — "
        "the same path the inference pipeline takes. The baseline columns are the degraded "
        "input measured against the same clean targets, before any enhancement.",
    ]
    return "\n".join(lines) + "\n"


def write_tables(results: dict, name: str, directory: Path | None = None) -> dict[str, Path]:
    """CSV, Markdown and per-sample scores, so no number is ever hand-copied."""
    import csv

    directory = Path(directory or paths.tables)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}

    rows = table_rows(results)
    csv_path = directory / f"{name}_restoration.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written["csv"] = csv_path

    markdown_path = directory / f"{name}_restoration.md"
    markdown_path.write_text(markdown_table(results, title=name), encoding="utf-8")
    written["markdown"] = markdown_path

    per_sample = [dict(row, split=split) for split, e in results.items() for row in e["per_sample"]]
    if per_sample:
        detail_path = directory / f"{name}_restoration_per_sample.csv"
        with open(detail_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
            writer.writeheader()
            writer.writerows(per_sample)
        written["per_sample"] = detail_path

    return written


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _find_checkpoint(args) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if not args.config:
        raise SystemExit("give either --checkpoint or --config")
    config = load_config(args.config, overrides=args.overrides)
    name = str(config.get("run", {}).get("name") or Path(config["_config_path"]).stem)
    candidate = paths.run_dir(name) / args.weights
    if not candidate.exists():
        raise SystemExit(f"no checkpoint at {candidate} — has {name} been trained yet?")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate.py", description="Score a trained ScanDar model."
    )
    parser.add_argument("--checkpoint", default=None, help="path to best.pt or last.pt")
    parser.add_argument("--config", default=None, help="find the checkpoint from a config instead")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")
    parser.add_argument("--weights", default="best.pt", help="which checkpoint in the run dir")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="score only the first N pages")
    parser.add_argument("--tile", type=int, default=512, help="tile size; 0 runs one whole-page pass")
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--out", default=None, help="table directory (default reports/tables)")
    parser.add_argument("--name", default=None, help="table name (default: the run name)")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    checkpoint_path = _find_checkpoint(args)
    device = get_device()
    model, config = load_model(checkpoint_path, device=device)
    # The checkpoint carries the config it trained with, which is the one that has
    # to be used to rebuild the model. Overrides still apply on top, so an
    # evaluation can be re-pointed at a different data root without retraining.
    for override in args.overrides:
        key, value = parse_override(override)
        config.set_path(key, value)

    name = args.name or checkpoint_path.parent.name
    print(f"checkpoint: {checkpoint_path}")
    print(f"device    : {describe_device(device)}")
    print(f"pages     : {'whole-page pass' if not args.tile else f'{args.tile}px tiles, {args.overlap}px overlap'}\n")

    results = evaluate_enhancement(
        model,
        config,
        splits=tuple(args.splits),
        device=device,
        tile=args.tile,
        overlap=args.overlap,
        limit=args.limit,
    )

    print("\n" + markdown_table(results))
    written = write_tables(results, name, directory=args.out)
    for kind, path in written.items():
        print(f"{kind:<10}: {path}")

    write_json(
        checkpoint_path.parent / "evaluation.json",
        {
            "checkpoint": str(checkpoint_path),
            "tile": args.tile,
            "overlap": args.overlap,
            "splits": {
                split: {"n": e["n"], "input": e["input"], "model": e["model"]}
                for split, e in results.items()
            },
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
