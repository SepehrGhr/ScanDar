"""Training loop.  *(brief §3.2)*

The brief names this file explicitly. One config-driven trainer serves all three
models — the enhancement network and both corner detectors — because they differ
in their data, loss and metrics, not in the shape of the loop.

Planned behaviour:

* device chosen automatically, mixed precision on CUDA, ``grad_accum`` so the
  *effective* batch is identical on the 6 GB laptop GPU and on a Colab runtime;
* the frozen validation set scored every epoch, and train-versus-validation loss
  plotted against epochs — the graph the brief asks for, and the one that
  diagnoses overfitting versus underfitting;
* Adam with **zero weight decay**: the first versions of every model carry no
  explicit regularisation, so the later dropout study isolates dropout alone;
* checkpoints storing model, optimiser, scaler, epoch *and* RNG state, so a run
  killed by a Colab timeout resumes exactly where it stopped;
* config, git commit and per-epoch metrics written next to the checkpoint, so
  every number in the report can be traced back to the run that produced it.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise SystemExit(
        "The training loop is not implemented yet. The synthetic generator has to "
        "exist before there is anything to train on."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
