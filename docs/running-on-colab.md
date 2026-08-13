# Running on Colab

Training moves between the development laptop (RTX 3060, 16 cores) and Colab depending on what is
free, and nothing in the codebase knows which one it is on: `SCANDAR_DATA` and `SCANDAR_OUT` move
every path the project uses, and `train.device` / `train.num_workers` default to `auto`.
`notebooks/00_colab_bootstrap.ipynb` is this page as executable cells.

## The one-time setup

Copy `data/` to `MyDrive/scandar/data`. Everything except `data/frozen/` is small (~75 MB); the six
frozen evaluation buckets are ~400 MB on top.

```
MyDrive/scandar/
├── data/        ← scans, cache, backgrounds, real, frozen, splits.json
└── outputs/     ← created by the first run: runs/<name>/{config.yaml,last.pt,best.pt,metrics.csv}
```

**Copy the frozen buckets; do not regenerate them casually.** They can be regenerated bit-for-bit,
but the enhancement buckets and the corner buckets were frozen from *different* configs, so the
obvious command produces enhancement buckets on the wrong canvas — and nothing downstream complains,
it just quietly measures a different distribution. If they really have to be rebuilt:

```bash
python scripts/freeze_eval_sets.py --config configs/enhance_realistic.yaml --task enhance
python scripts/freeze_eval_sets.py --config configs/corner.yaml           --task corner
```

Both are no-ops when the sets are already present and match.

## A session, start to finish

1. **Push first.** The notebook clones from GitHub and runs `git pull --ff-only`; a config that only
   exists on the laptop is a config Colab cannot run.
2. Run cells 1–3: GPU check, mount Drive, clone and `pip install -e .`.
3. Run cell 4. It copies the data from Drive to `/content/data` and points `SCANDAR_DATA` there,
   while `SCANDAR_OUT` stays on Drive. **This split is deliberate**: the generator opens scans and
   background photos inside every `__getitem__` across several worker processes, and Drive is a
   network filesystem, so reading training data through it is slow in exactly the place this project
   is already bottlenecked. Checkpoints are the opposite — written once an epoch, and worthless if
   the runtime dies with them on a disk that dies with it.
4. Run cell 5: `prepare_data.py` (idempotent) and `sanity_checks.py`. Three warnings about data still
   owed are expected; anything red is not.
5. Train **one** run at a time, with the wall-clock guard:

   ```bash
   python train.py --config configs/corner_reg_dropout.yaml --set train.max_hours=3
   ```

6. When the run has finished all its epochs, score it and refresh the tables:

   ```bash
   python evaluate.py --config configs/corner_reg_dropout.yaml
   python scripts/compare_dropout.py
   python scripts/make_figures.py --run corner_reg_dropout
   ```

   Tables land in `reports/tables/` and figures in `reports/figures/` **inside the clone**, which
   dies with the runtime — copy them to Drive, or commit them, before closing the session.

## Resuming, which is the normal case

Colab ends sessions. Every run writes `last.pt` each epoch with the optimiser, the AMP scaler and the
RNG state, and `train.resume` defaults to `auto`, so:

> **Re-running the exact same command is how you resume.**

`train.max_hours` exists so the run stops cleanly at a checkpoint boundary rather than being killed
mid-epoch; set it a little under the session length you expect. Resuming prints
`resume: last.pt at epoch N, step M` — if it does not, it is starting from scratch and something is
wrong with the path, usually `SCANDAR_OUT`.

Two things that break a resumed comparison:

* **Changing `train.epochs` between sessions restarts the cosine schedule rather than extending it.**
  Step 3000 of a 20-epoch schedule sits at a learning rate of 1e-6; re-declare it as 40 epochs and
  the same step is back at 1.1e-4. Decide the schedule before the first session.
* **Changing anything else** — batch size, canvas, patch count — makes the second half of the run a
  different experiment from the first. Only `train.max_hours` is safe to vary.

## What to expect from the hardware

**Colab is the slower machine for this project**, which inverts the usual advice. The bottleneck is
not the GPU: it is the CPU compositing and degrading synthetic photographs, and a Colab runtime has
about two cores against the laptop's sixteen. Measured through the real training loop on the 3060:

| Run | throughput | wall clock |
|---|---|---|
| enhancement, batch 16, 1920×2560 canvas, 8 patches per photo | ~18 samples/s | ~2.0 h for 20 × 400 steps |
| corner detectors, batch 16, 1152×1536 canvas | ~6 samples/s | ~2.4 h for 20 × 150 steps |

Expect a Colab run to take **two to four times** those figures, and to need two or three sessions
with resume. The number to trust is the one the trainer prints at the end of its first epoch: read
`samples/s`, multiply out the remaining steps, and decide before committing the night to it.

A bigger batch does not help — it is the same CPU work per sample either way. `batch_size` and
`grad_accum` are separate keys so the *effective* batch can be held constant on a machine where the
real one does not fit; at 256² patches the default of 16 peaks around 3 GB and fits everywhere.

Two other things not worth trying: `persistent_workers` is off deliberately (the dataset's per-epoch
state is a parent-side mutation that never reaches an already-forked worker, so persistent workers
would replay one epoch's samples for the whole run), and Colab's OpenCV is 4.x while the laptop's is
5.0 — the code sticks to the 4.x API surface for that reason.

## Bringing the results home

What matters is under `SCANDAR_OUT/runs/<name>/`: `best.pt` (weights only, ~31 MB) is all that
evaluation and the demo ever need, `metrics.csv` draws the curves, and `last.pt` (~93 MB) exists only
to resume — delete it once the run is finished if Drive or the laptop's disk is tight. Copy the run
directory back beside the local ones and every script finds it without being told where it came from.
