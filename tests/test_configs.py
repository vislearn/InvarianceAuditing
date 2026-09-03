"""The training configs under configs/.

Each one is a row of Figure 4 / Figure 14 / Table 3. Nothing checks them until
a training job starts, by which point the dataset has loaded and a GPU is
occupied; every name in them can be resolved statically instead.
"""

import glob
import importlib
import os
import re

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_paths():
    return sorted(glob.glob(os.path.join(REPO, "configs", "**", "*.yaml"),
                            recursive=True))


def fiber_model_paths():
    return sorted(glob.glob(os.path.join(
        REPO, "configs", "colormnist", "fiber_models", "*.yaml")))


def load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def ids(path):
    return os.path.relpath(path, os.path.join(REPO, "configs"))


def resolve_dotted(name):
    """`fff.model.ResNet` -> the class, or raise saying which part is missing."""
    module, _, attribute = name.rpartition(".")
    return getattr(importlib.import_module(module), attribute)


@pytest.mark.parametrize("path", config_paths(), ids=ids)
def test_config_is_yaml_with_a_model_and_a_dataset(path):
    config = load(path)
    assert isinstance(config, dict)
    assert "model" in config and "data_set" in config


@pytest.mark.parametrize("path", config_paths(), ids=ids)
def test_every_dotted_name_in_a_config_resolves(path):
    """A typo in `fff.model.ResNet` is only found when the model is built.

    lightning_trainable looks these up by string at construction time, so a
    misspelled class survives every import check in this suite and fails after
    the data has loaded.
    """
    text = open(path).read()
    for name in sorted(set(re.findall(r"\bfff\.[A-Za-z0-9_.]+", text))):
        resolve_dotted(name)


@pytest.mark.parametrize("path", config_paths(), ids=ids)
def test_optimizer_and_scheduler_names_exist_in_torch(path):
    import torch.optim

    def named(value):
        # lightning_trainable accepts either a bare name or {name: ..., kwargs: ...}
        return value if isinstance(value, str) else value["name"]

    config = load(path)
    if "optimizer" in config:
        assert hasattr(torch.optim, named(config["optimizer"]))
    if "lr_scheduler" in config:
        assert hasattr(torch.optim.lr_scheduler, named(config["lr_scheduler"]))


@pytest.mark.parametrize("path", config_paths(), ids=ids)
def test_every_recorded_path_can_be_retargeted_by_the_data_root(path):
    """No config may hard-code a machine-specific location.

    fff.data.paths.resolve rewrites a path that starts with `data/` against
    FFF_DATA_ROOT and leaves everything else alone, so anything else is a path
    that only exists on one machine.
    """
    text = open(path).read()
    recorded = re.findall(r"(?:root|path|_path|ckpt):\s*([^\s,}]+)", text)
    for value in recorded:
        if value in ("null", "true", "false") or not re.search(r"[/.]", value):
            continue
        assert value.startswith("data/"), (
            f"{value!r} is not under data/, so FFF_DATA_ROOT cannot point at it")


# --------------------------------------------------- the colorMNIST comparison

@pytest.mark.parametrize("path", fiber_model_paths(), ids=os.path.basename)
def test_the_lambda_in_the_filename_is_the_fiber_loss_weight(path):
    """The filename is how Figure 4 and Table 3 label the marker.

    A config named lambda10 that trains at a different weight would move a
    point on the trade-off curve with nothing to show it had moved. Weight 0 is
    spelled by leaving the key out.
    """
    expected = int(re.search(r"lambda(\d+)", os.path.basename(path)).group(1))
    weight = load(path).get("loss_weights", {}).get("fiber_loss", 0)
    assert weight == expected


def test_every_fiber_model_is_audited_by_the_same_subject_model():
    """The lambda sweep compares fiber models, not subject models."""
    seen = {}
    for path in fiber_model_paths():
        config = load(path)
        key = (config["data_set"].get("subject_model_path"),
               config["data_set"].get("subject_model_type"),
               config.get("sm_input_transform"),
               config.get("sm_empty_condition"))
        seen.setdefault(key, []).append(os.path.basename(path))
    assert len(seen) == 1, f"the sweep audits more than one phi: {seen}"


def test_every_fiber_model_trains_on_the_same_data():
    datasets = {yaml.dump(load(p)["data_set"], sort_keys=True)
                for p in fiber_model_paths()}
    assert len(datasets) == 1


def test_every_fiber_model_uses_the_released_lossless_autoencoder():
    """A different VAE is a different latent space and a different fiber loss."""
    for path in fiber_model_paths():
        config = load(path)
        assert config.get("load_lossless_ae_path") == "data/cc_mnist/lossless_vae.ckpt"
        assert config.get("train_lossless_ae") is False, (
            f"{os.path.basename(path)} would retrain the autoencoder")


def test_the_fiber_model_sweep_covers_the_families_the_paper_reports():
    """Figure 4 has one curve per fiber-model family."""
    families = {os.path.basename(p).split("_")[0] for p in fiber_model_paths()}
    assert families == {"fff", "fif", "nf", "dnf", "mlf", "diff", "fm"}


def test_every_config_is_reachable_from_the_statistics_script():
    """compute_statistics.py hard-codes the run names it evaluates.

    A config with no entry there is a model that gets trained and never read.
    """
    from experiments.colormnist.compute_statistics import __file__ as stats_file

    text = open(stats_file).read()
    for path in fiber_model_paths():
        name = os.path.splitext(os.path.basename(path))[0]
        assert f'"{name}"' in text, f"{name} is never evaluated"


# The optimiser settings each run was launched with. They are not uniform --
# MLF and DNF at lambda = 0 trained for 100 epochs at lr 2e-3 where their
# lambda > 0 siblings trained for 250 at 1e-3, and NF varies its learning rate
# across the sweep too -- so part of the movement along a Figure 4 curve is
# optimiser difference rather than lambda. That is what produced the published
# numbers, and pinning it here is what stops an accidental edit from changing
# one with nothing to show for it.
TRAINING_SETTINGS = {
    "diff_lambda0":  (1000, 0.001, 0.005),
    "dnf_lambda0":   (100,  0.002, None),
    "dnf_lambda1":   (100,  0.001, None),
    "dnf_lambda10":  (100,  0.002, 0.12),
    "dnf_lambda100": (100,  0.001, None),
    "fff_lambda0":   (250,  0.001, 0.12),
    "fff_lambda1":   (250,  0.001, 0.12),
    "fff_lambda10":  (250,  0.001, 0.12),
    "fff_lambda100": (250,  0.001, 0.12),
    "fif_lambda0":   (250,  0.001, 0.12),
    "fif_lambda1":   (250,  0.001, 0.12),
    "fif_lambda10":  (250,  0.001, 0.12),
    "fif_lambda100": (250,  0.001, 0.12),
    "fm_lambda0":    (250,  0.001, 0.12),
    "mlf_lambda0":   (100,  0.002, None),
    "mlf_lambda1":   (250,  0.001, 0.12),
    "mlf_lambda10":  (250,  0.001, 0.12),
    "nf_lambda0":    (250,  0.001, 0.12),
    "nf_lambda1":    (250,  0.002, 0.12),
    "nf_lambda10":   (250,  0.002, 0.12),
    "nf_lambda100":  (250,  0.002, 0.12),
}


@pytest.mark.parametrize("path", fiber_model_paths(), ids=os.path.basename)
def test_training_settings_are_unchanged(path):
    name = os.path.splitext(os.path.basename(path))[0]
    config = load(path)
    scheduler = config.get("lr_scheduler")
    kwargs = scheduler.get("kwargs", {}) if isinstance(scheduler, dict) else {}
    actual = (config["max_epochs"], config["optimizer"]["lr"],
              kwargs.get("pct_start"))
    assert actual == TRAINING_SETTINGS[name], (
        f"{name} no longer matches the run it came from")


def test_the_training_table_covers_every_config():
    names = {os.path.splitext(os.path.basename(p))[0] for p in fiber_model_paths()}
    assert names == set(TRAINING_SETTINGS)


def test_settings_cover_every_table5_row():
    """Every ImageNet row resolves a gamma and a variance_type without flags.

    A row missing from SETTINGS falls back to (default, learned_range), which
    is silently wrong for three of the four -- and wrong in a way that only shows
    up as a 10-50% fiber loss offset in a run that costs a day.
    """
    from experiments.imagenet.sample_ndtm import ETA, GAMMA_SCHEDULES, SETTINGS

    rows = {("dinov2", "imagenet"), ("inception", "imagenet"),
            ("dinov2", "cue_conflict"), ("resnet50", "cue_conflict")}
    assert set(SETTINGS) == rows
    for (model, dataset), (gamma, variance) in SETTINGS.items():
        assert model in ETA
        assert gamma in GAMMA_SCHEDULES
        assert variance in ("small", "large", "learned_range")
