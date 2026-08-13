"""Training loop.  *(brief §3.2)*

The brief names this file explicitly. One config-driven trainer serves every
model in the project — the enhancement network now, both corner detectors when
they are built — because they differ in their data, loss and metrics, not in the
shape of the loop.

    python train.py --config configs/enhance.yaml
    python train.py --config configs/enhance.yaml --set train.epochs=20 run.name=quick

What it does, and why:

* **the device is chosen, never assumed.** The same command runs on the 6 GB
  laptop GPU and on whatever Colab hands out; ``batch_size`` and ``grad_accum``
  are separate knobs so the *effective* batch stays identical when the memory
  does not.
* **the validation set is frozen and scored every epoch**, giving the
  train-versus-validation curve the brief asks for — the graph that separates
  overfitting from underfitting. Validation is measured on patches, like the
  training loss, because two curves on one axis have to be the same quantity;
  the whole-page table is ``evaluate.py``'s job.
* **Adam with zero weight decay.** The first version of every model carries no
  explicit regularisation, so the later dropout study isolates dropout alone
  *(brief §3.1)*.
* **checkpoints carry the RNG state.** A Colab session dies without warning, and
  a run that resumes onto a different random stream than it would have seen is
  not the run it claims to be. ``last.pt`` holds model, optimiser, scaler, epoch
  and every generator's state; ``best.pt`` holds weights only, because that is
  all evaluation and the demo need and disk here is tight.
* **the config, the git commit and every epoch's metrics are written next to the
  checkpoint**, so any number in the report can be traced back to the run that
  produced it.

**A note on throughput.** Composing and degrading a synthetic photo costs far
more than a training step does, so this loop is fed by the CPU, not limited by
the GPU. ``patches_per_photo`` amortises one composited photo over several
patches, which is worth about four times the throughput and is the reason the
loader must not shuffle. The measured rate is printed at the start of every run
and logged per epoch — it is the number that decides how many epochs fit in an
evening.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .datasets import SyntheticEnhanceDataset, frozen_dataset
from .device import amp_enabled, describe_device, get_device, recommended_workers
from .io import paths, write_json
from .losses import build_loss
from .metrics import MetricAccumulator, psnr, ssim_metric
from .model import build_model, clamp_image, count_parameters
from .prepare import ScanBank, load_splits
from .seed import seed_everything, worker_init_fn
from .synth import build_sources

__all__ = ["train", "main"]


# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------
def git_sha(short: bool = True) -> str | None:
    """The commit the run was launched from, or None outside a git checkout.

    Written into the run directory so a number in the report can be traced back
    to the code that produced it, which is the whole reason for recording it.
    """
    command = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    try:
        result = subprocess.run(
            command, cwd=paths.repo, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def autocast_for(device, enabled: bool):
    """Mixed precision on the device we are actually on.

    ``torch.autocast("cuda")`` on a CPU-only machine warns on every call even
    when it is disabled, and a training log full of that is a log nobody reads.
    """
    return torch.autocast(device.type, enabled=enabled and device.type == "cuda")


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup into a cosine decay, evaluated per optimiser step.

    Per step rather than per epoch: an "epoch" here is an arbitrary number of
    generated samples, so a schedule that moves once per epoch would change shape
    the moment ``iters_per_epoch`` did, and two runs sized differently would stop
    being comparable.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def rng_state() -> dict:
    """Everything that has to come back for a resumed run to be the same run."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _byte_tensor(state) -> torch.Tensor:
    """A CPU uint8 tensor, whatever device or dtype the checkpoint came back on.

    ``torch.load`` maps every tensor in the file — including the saved RNG states
    — onto the training device, and ``set_rng_state`` insists on a CPU
    ByteTensor. Resuming crashed on exactly this before the coercion was added.
    """
    return torch.as_tensor(state).to(device="cpu", dtype=torch.uint8)


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_byte_tensor(s) for s in state["cuda"]])
        except (RuntimeError, ValueError, TypeError) as exc:
            # A run started on one GPU count and resumed on another. The CUDA
            # stream only drives dropout and initialisation, both of which are
            # already past by the time a run is resumed, so this is worth a note
            # rather than a crash.
            print(f"  note: could not restore the CUDA RNG state ({exc})")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_train_dataset(config: Config, task: str) -> SyntheticEnhanceDataset:
    """The infinite generator, sized so one epoch is ``iters_per_epoch`` steps."""
    if task != "enhance":
        raise NotImplementedError(
            f"task {task!r} is not implemented yet — only the enhancement network is built. "
            "The corner detectors come with the corner-detection task."
        )

    data = config.data
    train = config.train
    samples = int(train.iters_per_epoch) * int(train.batch_size) * int(train.get("grad_accum", 1))

    splits = load_splits()
    sources = build_sources(config, "train", task=task, splits=splits)

    # The brief's §3.2 option: many degradations of few scans, or few
    # degradations of many? Limiting the pool of source scans while keeping the
    # number of steps fixed is exactly that experiment, and it is one config key.
    limit = data.get("train_scan_limit")
    if limit:
        sources.scans = ScanBank(sources.scans.ids[: int(limit)], directory=sources.scans.directory)

    return SyntheticEnhanceDataset(
        sources,
        "train",
        length=samples,
        mode="patch",
        patch_size=int(data.patch_size),
        rect_size=tuple(data.rect_size),
        patches_per_photo=int(data.get("patches_per_photo", 4)),
        patch_tries=int(data.get("patch_tries", 4)),
        min_patch_std=float(data.get("min_patch_std", 0.045)),
        seed=int(config.project.seed),
    )


def warn_if_frozen_set_is_stale(config: Config, dataset) -> bool:
    """Shout if the frozen validation set was generated from different settings.

    There is one frozen set on disk and whichever config last froze it decides
    what is in it. Train with a config that changes the generator — a different
    canvas, a different page scale — and the validation curve silently measures a
    distribution the model is not being trained on, for the whole run. Nothing
    crashes and the numbers look plausible, which is what makes it worth an
    explicit check rather than a note in the documentation.
    """
    entries = getattr(dataset, "entries", None)
    if not entries:
        return True

    wanted = [int(v) for v in config.data.get("canvas", [])]
    frozen = [int(v) for v in (entries[0].get("canvas") or [])]
    # A landscape sample has its canvas transposed, so compare as a set of sides.
    if not wanted or not frozen or sorted(wanted) == sorted(frozen):
        return True

    print(
        f"\n  !! the frozen validation set was generated on a {frozen[0]}x{frozen[1]} canvas,\n"
        f"     but this run trains on {wanted[0]}x{wanted[1]}. The validation curve will be\n"
        f"     measuring a different distribution than the model is learning.\n"
        f"     Fix it before spending the GPU time:\n"
        f"       python scripts/freeze_eval_sets.py --config {config['_config_path']} --force\n"
    )
    return False


def make_loader(dataset, config: Config, workers: int, shuffle: bool, drop_last: bool) -> DataLoader:
    """A DataLoader with the two settings this project cannot change.

    ``shuffle=False`` is required by the patch amortisation: consecutive indices
    share one composited photo, and shuffling would throw that away for nothing —
    every index invents a fresh random sample, so there is no order to break.

    ``persistent_workers=False`` is required by ``set_epoch``. Workers are forked
    copies; moving the parent's dataset to a new epoch does not reach a worker
    that is already alive, so persistent workers would hand back the same epoch's
    samples for the whole run. Re-forking each epoch is the price of that being
    impossible rather than merely unlikely.
    """
    return DataLoader(
        dataset,
        batch_size=int(config.train.batch_size),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=worker_init_fn if workers > 0 else None,
        persistent_workers=False,
    )


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def evaluate_epoch(model, loader, criterion, device, use_amp: bool) -> MetricAccumulator:
    """Score the frozen validation set: the same loss, plus PSNR and SSIM."""
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with autocast_for(device, use_amp):
                outputs = model(inputs)
            loss, parts = criterion(outputs, targets)

            predictions = clamp_image(outputs.float())
            accumulator.add("loss", [float(loss)] * inputs.shape[0])
            for name, value in parts.items():
                accumulator.add(name, [value] * inputs.shape[0])
            accumulator.add("psnr", psnr(predictions, targets, reduction="none"))
            accumulator.add("ssim", ssim_metric(predictions, targets, reduction="none"))
    return accumulator


def train(config: Config, resume: str | None = None) -> Path:
    """Run the training described by *config*. Returns the run directory."""
    task = str(config.get("task", "enhance"))
    train_cfg = config.train
    log_cfg = config.get("log", {})

    seed_everything(int(config.project.seed), bool(config.project.get("deterministic", False)))
    device = get_device(str(train_cfg.get("device", "auto")))
    use_amp = bool(train_cfg.get("amp", True)) and amp_enabled(device)
    workers = recommended_workers(train_cfg.get("num_workers", "auto"))

    run_name = str(config.get("run", {}).get("name") or Path(config["_config_path"]).stem)
    run_dir = paths.run_dir(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- model, loss, optimiser -------------------------------------------
    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(train_cfg.epochs)
    iters_per_epoch = int(train_cfg.iters_per_epoch)
    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
    grad_clip = float(train_cfg.get("grad_clip", 0.0) or 0.0)
    total_steps = epochs * iters_per_epoch
    warmup_steps = int(float(train_cfg.get("warmup_epochs", 0)) * iters_per_epoch)
    base_lr, min_lr = float(train_cfg.lr), float(train_cfg.get("min_lr", 0.0))

    monitor = str(log_cfg.get("keep_best_on", "val_loss"))
    higher_is_better = str(log_cfg.get("best_mode", "auto")) == "max" or (
        str(log_cfg.get("best_mode", "auto")) == "auto"
        and not any(word in monitor for word in ("loss", "err"))
    )

    # --- data --------------------------------------------------------------
    print(f"run       : {run_name}  ->  {run_dir}")
    print(f"device    : {describe_device(device)}{'  + AMP' if use_amp else ''}")
    print(f"model     : {type(model).__name__}, {count_parameters(model):,} parameters")
    print(f"loss      : {criterion.extra_repr()}")

    train_dataset = build_train_dataset(config, task)
    train_dataset.sources.warm()
    scans = train_dataset.sources.scans

    val_dataset = frozen_dataset(config, "val", task=task, mode="patch")
    warn_if_frozen_set_is_stale(config, val_dataset)
    train_loader = make_loader(train_dataset, config, workers, shuffle=False, drop_last=True)
    val_loader = make_loader(val_dataset, config, workers, shuffle=False, drop_last=False)
    print(
        f"data      : {len(scans)} train scans, {len(train_dataset)} samples/epoch, "
        f"{len(val_dataset)} frozen val patches, {workers} worker(s)"
    )

    # --- resume ------------------------------------------------------------
    history: list[dict] = []
    start_epoch, global_step = 0, 0
    best_value = -math.inf if higher_is_better else math.inf

    checkpoint_path = _resume_path(train_cfg.get("resume", "auto") if resume is None else resume,
                                   run_dir)
    if checkpoint_path is not None:
        # Onto the CPU, not the device: the model and optimiser states are copied
        # into objects that already live on the device, and loading straight to
        # CUDA would need a second copy of the weights in GPU memory to do it.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        restore_rng_state(state.get("rng", {}))
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        history = list(state.get("history", []))
        best_value = float(state.get("best_value", best_value))
        print(f"resume    : {checkpoint_path.name} at epoch {start_epoch}, step {global_step}")
        if start_epoch >= epochs:
            print(f"\nnothing to do — the run already reached its {epochs} epochs")
            return run_dir

    config.save(run_dir / "config.yaml")
    write_json(
        run_dir / "run.json",
        {
            "run": run_name,
            "task": task,
            "git_sha": git_sha(),
            "device": str(describe_device(device)),
            "amp": use_amp,
            "workers": workers,
            "parameters": count_parameters(model),
            "loss": criterion.extra_repr(),
            "command": " ".join(sys.argv),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    # --- epochs ------------------------------------------------------------
    max_hours = train_cfg.get("max_hours")
    deadline = time.time() + float(max_hours) * 3600 if max_hours else None
    log_every = int(log_cfg.get("every_n_steps", 50))
    stopped_early = False

    print(f"\ntraining  : {epochs} epochs x {iters_per_epoch} steps, batch {config.train.batch_size}"
          f"{f' x {grad_accum} accumulated' if grad_accum > 1 else ''}\n")

    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running = MetricAccumulator()
        epoch_start = time.time()
        seen = 0
        step_in_epoch = 0
        micro = 0
        learning_rate = lr_at(global_step, total_steps, warmup_steps, base_lr, min_lr)

        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            seen += inputs.shape[0]

            if micro == 0:
                learning_rate = lr_at(global_step, total_steps, warmup_steps, base_lr, min_lr)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

            with autocast_for(device, use_amp):
                outputs = model(inputs)
            loss, parts = criterion(outputs, targets)
            scaler.scale(loss / grad_accum).backward()

            running.add("loss", float(loss))
            running.update({name: value for name, value in parts.items()})
            micro += 1

            if micro < grad_accum:
                continue
            micro = 0

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_in_epoch += 1

            if log_every and step_in_epoch % log_every == 0:
                rate = seen / max(1e-9, time.time() - epoch_start)
                print(
                    f"  epoch {epoch + 1:>3}/{epochs}  step {step_in_epoch:>4}/{iters_per_epoch}"
                    f"  loss {running.mean('loss'):.4f}  lr {learning_rate:.2e}"
                    f"  {rate:.1f} samples/s"
                )
            if step_in_epoch >= iters_per_epoch:
                break

        train_seconds = time.time() - epoch_start
        validation = evaluate_epoch(model, val_loader, criterion, device, use_amp)

        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "lr": learning_rate,
            "train_loss": running.mean("loss"),
            **{f"train_{name}": running.mean(name) for name in criterion.active},
            "val_loss": validation.mean("loss"),
            **{f"val_{name}": validation.mean(name) for name in criterion.active},
            "val_psnr": validation.mean("psnr"),
            "val_ssim": validation.mean("ssim"),
            "samples_per_s": seen / max(1e-9, train_seconds),
            "train_seconds": round(train_seconds, 1),
            "val_seconds": round(time.time() - epoch_start - train_seconds, 1),
        }
        history.append(row)
        _append_metrics(run_dir, row)

        current = row.get(monitor, row["val_loss"])
        improved = current > best_value if higher_is_better else current < best_value
        print(
            f"  epoch {epoch + 1:>3}/{epochs}  train {row['train_loss']:.4f}"
            f"  val {row['val_loss']:.4f}  psnr {row['val_psnr']:.2f} dB"
            f"  ssim {row['val_ssim']:.4f}  {row['samples_per_s']:.1f} samples/s"
            f"  [{train_seconds / 60:.1f} + {row['val_seconds'] / 60:.1f} min]"
            f"{'  *best*' if improved else ''}"
        )

        if improved:
            best_value = current
            _save_best(run_dir, model, config, epoch, row, monitor)

        if log_cfg.get("save_every_epoch", True) or epoch == epochs - 1:
            _save_last(run_dir, model, optimizer, scaler, config, epoch, global_step,
                       history, best_value, use_amp)

        if deadline is not None and time.time() > deadline:
            print(f"\nstopping at the {max_hours} hour guard — resume with the same command")
            stopped_early = True
            break

    _write_summary(run_dir, history, monitor, best_value, stopped_early)
    print(f"\nbest {monitor}: {best_value:.4f}     checkpoints and logs in {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# checkpoints and logs
# ---------------------------------------------------------------------------
def _resume_path(setting, run_dir: Path) -> Path | None:
    """``auto`` picks up ``last.pt`` if it is there; ``none`` never does."""
    if setting in (None, "none", False):
        return None
    if setting == "auto":
        candidate = run_dir / "last.pt"
        return candidate if candidate.exists() else None
    candidate = Path(setting)
    if not candidate.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {candidate}")
    return candidate


def _save_last(run_dir, model, optimizer, scaler, config, epoch, global_step, history,
               best_value, use_amp) -> Path:
    """The full state, written atomically so a killed process cannot corrupt it."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if use_amp else None,
        "config": config.to_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "history": history,
        "best_value": best_value,
        "rng": rng_state(),
    }
    return _atomic_save(payload, run_dir / "last.pt")


def _save_best(run_dir, model, config, epoch, metrics, monitor) -> Path:
    """Weights only.

    Everything needed to score or demo a model, and nothing else: the optimiser
    state in a full checkpoint is twice the size of the weights and is only ever
    read to continue training, which is what ``last.pt`` is for.
    """
    payload = {
        "model": model.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "monitor": monitor,
    }
    return _atomic_save(payload, run_dir / "best.pt")


def _atomic_save(payload: dict, path: Path) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _append_metrics(run_dir: Path, row: dict) -> None:
    """One row per epoch, in CSV for plotting and JSONL for everything else."""
    csv_path = run_dir / "metrics.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow({key: _round(value) for key, value in row.items()})
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: _round(value) for key, value in row.items()}) + "\n")


def _round(value):
    return round(value, 6) if isinstance(value, float) else value


def _write_summary(run_dir: Path, history: list[dict], monitor: str, best_value: float,
                   stopped_early: bool) -> None:
    if not history:
        return
    write_json(
        run_dir / "summary.json",
        {
            "epochs_completed": history[-1]["epoch"],
            "monitor": monitor,
            "best_value": best_value,
            "final": history[-1],
            "stopped_early": stopped_early,
            "mean_samples_per_s": round(
                sum(row["samples_per_s"] for row in history) / len(history), 2
            ),
        },
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train.py", description="Train a ScanDar model from a config file."
    )
    parser.add_argument("--config", required=True, help="e.g. configs/enhance.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")
    parser.add_argument("--resume", default=None, help="auto | none | <path to a checkpoint>")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    config = load_config(args.config, overrides=args.overrides)
    try:
        train(config, resume=args.resume)
    except KeyboardInterrupt:
        # The last completed epoch is already on disk; saying so is kinder than
        # a traceback that looks like the run was lost.
        print("\ninterrupted — the last completed epoch is in last.pt, resume with the same command")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
