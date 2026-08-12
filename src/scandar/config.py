"""YAML configuration with inheritance and command-line overrides.

Every experiment in this project is a config file, never an edited constant. That
is what makes the ablations the brief asks for — MSE vs L1 vs L1+SSIM+gradient,
regression vs heatmap, with and without dropout — comparable rather than
anecdotal: each run stores the exact config it ran with, next to its checkpoint.

A config may inherit from another via ``_base_``, and any leaf can be overridden
from the command line::

    python train.py --config configs/enhance.yaml --set train.lr=3e-4 model.base=48
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

# PyYAML implements YAML 1.1, whose float resolver requires a decimal point in
# scientific notation: `2.0e-4` parses as a float but `2e-4` comes back as the
# *string* "2e-4". Left alone, a learning rate written the natural way would reach
# the optimiser as a string — which either crashes far from the cause or, worse,
# does not. Any scalar in that exact shape is coerced back to a float.
_SCIENTIFIC = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


class Config(dict):
    """A nested dict that also answers to attribute access.

    ``cfg.train.lr`` and ``cfg["train"]["lr"]`` are the same thing; the first reads
    better in code, the second is convenient when the key is computed.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(f"no config key {key!r}") from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    # -- dotted access -----------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot descend into {dotted!r}: {part!r} is a leaf")
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        return path

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Dicts merge key by key; anything else (including lists) is replaced outright —
    a half-overridden list of loss weights would be a debugging nightmare.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def coerce_scalars(value: Any) -> Any:
    """Recursively repair YAML 1.1's exponent-without-a-dot blind spot."""
    if isinstance(value, str):
        return float(value) if _SCIENTIFIC.match(value) else value
    if isinstance(value, dict):
        return {key: coerce_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [coerce_scalars(item) for item in value]
    return value


def parse_override(item: str) -> tuple[str, Any]:
    """``"train.lr=3e-4"`` -> ``("train.lr", 0.0003)``, with YAML value typing."""
    if "=" not in item:
        raise ValueError(f"override must look like key.path=value, got {item!r}")
    key, _, raw = item.partition("=")
    return key.strip(), coerce_scalars(yaml.safe_load(raw))


def load_config(
    path: Path | str,
    overrides: Iterable[str] | None = None,
    _seen: tuple[Path, ...] = (),
) -> Config:
    """Load a YAML config, resolving ``_base_`` inheritance and applying overrides.

    ``_base_`` is a path relative to the including file, so configs can live in
    subdirectories without absolute paths leaking in.
    """
    path = Path(path).resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ValueError(f"circular _base_ inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"config must be a mapping at the top level: {path}")
    raw = coerce_scalars(raw)

    base_ref = raw.pop("_base_", None)
    if base_ref is not None:
        parent = load_config(path.parent / base_ref, overrides=None, _seen=(*_seen, path))
        raw = deep_merge(parent.to_dict(), raw)

    config = Config(raw)
    for item in overrides or ():
        key, value = parse_override(item)
        config.set_path(key, value)
    config["_config_path"] = str(path)
    return config
