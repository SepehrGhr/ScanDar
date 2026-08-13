"""The config system is what makes the ablations comparable, so it gets tests."""

import pytest
import yaml

from scandar.config import Config, deep_merge, load_config, parse_override
from scandar.io import paths


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_attribute_and_item_access_agree():
    config = Config({"train": {"lr": 3e-4}})
    assert config.train.lr == config["train"]["lr"] == 3e-4


def test_dotted_get_and_set():
    config = Config({"train": {"lr": 1e-3}})
    assert config.get_path("train.lr") == 1e-3
    assert config.get_path("train.missing", "fallback") == "fallback"
    config.set_path("model.base", 48)
    assert config.model.base == 48


def test_deep_merge_replaces_lists_but_merges_dicts():
    base = {"train": {"lr": 1e-3, "epochs": 60}, "weights": [1.0, 0.5]}
    merged = deep_merge(base, {"train": {"lr": 2e-4}, "weights": [1.0]})
    assert merged["train"] == {"lr": 2e-4, "epochs": 60}
    # A half-overridden list of loss weights would be a debugging nightmare.
    assert merged["weights"] == [1.0]
    assert base["train"]["lr"] == 1e-3, "merging must not mutate the base"


def test_override_values_are_yaml_typed():
    assert parse_override("train.amp=false") == ("train.amp", False)
    assert parse_override("data.canvas=[512, 512]") == ("data.canvas", [512, 512])
    assert parse_override("log.keep_best_on=val_psnr") == ("log.keep_best_on", "val_psnr")
    with pytest.raises(ValueError):
        parse_override("no-equals-sign")


def test_exponent_notation_survives_yaml_1_1():
    """PyYAML resolves `2e-4` to a string, not a float — a silent way to hand the
    optimiser a learning rate it cannot use."""
    assert parse_override("train.lr=3e-4") == ("train.lr", 3e-4)
    assert parse_override("train.lr=3.0e-4") == ("train.lr", 3e-4)
    assert parse_override("train.min_lr=1E-6") == ("train.min_lr", 1e-6)
    # Only that exact shape is coerced; genuine strings are left alone.
    assert parse_override("project.name=scandar") == ("project.name", "scandar")
    assert parse_override("log.keep_best_on=1e_poch") == ("log.keep_best_on", "1e_poch")


def test_config_files_get_the_same_repair(tmp_path):
    path = write(tmp_path, "lr.yaml", {"train": {"lr": "2e-4", "note": "2e-4 lr"}})
    config = load_config(path)
    assert config.train.lr == pytest.approx(2e-4) and isinstance(config.train.lr, float)
    assert config.train.note == "2e-4 lr"


def test_base_inheritance_and_overrides(tmp_path):
    write(tmp_path, "base.yaml", {"train": {"lr": 1e-3, "epochs": 60}, "model": {"base": 32}})
    child = write(tmp_path, "child.yaml", {"_base_": "base.yaml", "train": {"lr": 2e-4}})

    config = load_config(child, overrides=["model.base=48"])
    assert config.train.lr == 2e-4  # child wins
    assert config.train.epochs == 60  # inherited
    assert config.model.base == 48  # command line wins over both


def test_circular_inheritance_is_reported(tmp_path):
    write(tmp_path, "a.yaml", {"_base_": "b.yaml"})
    write(tmp_path, "b.yaml", {"_base_": "a.yaml"})
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_shipped_base_config_holds_the_documented_defaults():
    config = load_config(paths.repo / "configs" / "base.yaml")
    # v1 models carry no explicit regularisation; the dropout study depends on it.
    assert config.train.weight_decay == 0.0
    # Splitting by source scan is the rule the whole dataset design rests on.
    assert config.data.n_val_scans + config.data.n_test_scans < 50
    assert config.data.patch_size <= min(config.data.rect_size)


@pytest.mark.parametrize(
    "baseline, arm, expected",
    [
        ("enhance_realistic", "enhance_dropout", {"bottleneck_dropout": 0.2}),
        ("enhance_realistic", "enhance_dropout_wide", {"dropout": 0.1}),
        ("corner_reg", "corner_reg_dropout", {"fc_dropout": 0.3}),
        ("corner_heat", "corner_heat_dropout", {"bottleneck_dropout": 0.2}),
        ("corner_heat", "corner_heat_dropout_wide", {"dropout": 0.1}),
    ],
)
def test_each_dropout_arm_differs_from_its_baseline_by_dropout_alone(baseline, arm, expected):
    """The regularisation study *(brief §6)* only means anything if dropout is the
    single variable, and an arm that quietly inherits a different schedule or a
    different canvas measures that instead. The epochs are the trap: the
    enhancement baseline was run with an override rather than the file's default,
    so the arms pin the schedule in the file."""
    base = load_config(paths.repo / "configs" / f"{baseline}.yaml")
    study = load_config(paths.repo / "configs" / f"{arm}.yaml")

    for key in ("data", "synth", "degradation", "loss", "project"):
        assert study.get(key) == base.get(key), f"{arm} changed {key}"
    assert study.train.weight_decay == 0.0, "dropout is supposed to be the only regulariser"
    assert study.train.epochs == 20, "the trained baselines are all 20-epoch runs"
    for key, value in base.train.items():
        if key != "epochs":
            assert study.train[key] == value, f"{arm} changed train.{key}"

    changed = {
        key for key in set(base.get("model")) | set(study.get("model"))
        if base.get("model").get(key) != study.get("model").get(key)
    }
    assert changed == set(expected)
    for key, value in expected.items():
        assert study.model[key] == value
