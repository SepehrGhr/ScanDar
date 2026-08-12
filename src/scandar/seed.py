"""Determinism.

The synthetic dataset invents a fresh sample on every ``__getitem__``, which makes
reproducibility a design problem rather than an afterthought:

* the **frozen** validation and test sets must regenerate byte-identically, or the
  validation curve measures the dice as much as the model;
* a run interrupted by a Colab timeout must resume without silently reshuffling
  the samples it has already seen.

Both fall out of :func:`rng_for`, which derives a generator from a *stable* hash of
whatever keys identify the sample. Python's builtin ``hash()`` is salted per
process and cannot be used for this — ``hash("a")`` differs between interpreter
runs, so frozen sets would quietly drift.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

MAX_SEED = 2**32


def stable_hash(*keys) -> int:
    """A process-independent 64-bit hash of *keys*."""
    payload = "|".join(repr(key) for key in keys).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def rng_for(*keys) -> np.random.Generator:
    """A NumPy generator uniquely and reproducibly determined by *keys*.

    Typical use inside a dataset::

        rng = rng_for(self.seed, self.split, index, epoch)

    Same keys, same sample — on any machine, in any process, in any epoch order.
    """
    return np.random.default_rng(stable_hash(*keys) % MAX_SEED)


def seed_everything(seed: int = 1234, deterministic_torch: bool = False) -> int:
    """Seed python, numpy and torch. Returns the seed, for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % MAX_SEED)

    try:
        import torch
    except ImportError:  # torch is optional for the data-only entry points
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Costs throughput; worth it when a result has to be bit-reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own stream, derived from the base seed.

    Without this, forked workers inherit one numpy state and several of them hand
    back *the same* "random" degradation — a classic silent duplicate-sample bug.
    """
    try:
        import torch

        base = torch.initial_seed() % MAX_SEED
    except ImportError:
        base = 0
    seed = (base + worker_id) % MAX_SEED
    np.random.seed(seed)
    random.seed(seed)
