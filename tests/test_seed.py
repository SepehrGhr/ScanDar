"""Reproducibility is a correctness property here, not a nicety.

The frozen validation set has to regenerate byte-identically across processes, or
the validation curve measures the dice as much as the model.
"""

import subprocess
import sys

from scandar.seed import rng_for, stable_hash


def test_same_keys_give_the_same_stream():
    assert rng_for(1234, "val", 7).random(5).tolist() == rng_for(1234, "val", 7).random(5).tolist()


def test_different_keys_give_different_streams():
    assert rng_for(1234, "val", 7).random() != rng_for(1234, "val", 8).random()
    assert rng_for(1234, "val", 7).random() != rng_for(1234, "test", 7).random()
    assert rng_for(1234, "val", 7).random() != rng_for(9999, "val", 7).random()


def test_hash_is_stable_across_processes():
    """Python's builtin hash() is salted per process; ours must not be.

    Spawning real interpreters is the only honest way to test this — within one
    process even a salted hash looks stable.
    """
    code = "from scandar.seed import stable_hash; print(stable_hash(1234, 'val', 7))"
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1
    assert runs.pop() == str(stable_hash(1234, "val", 7))
