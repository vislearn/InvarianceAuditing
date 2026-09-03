"""Which seed the HTRU2 analysis reports.

A seed sweep leaves several sampled_*.pt per gamma, and the reference values in
REPRODUCING.md are `--seed 0` alone. Keying results by gamma only would keep
whichever file `sorted(glob)` returned last, and since the seed is not in the
filename nothing in `analysis.json` would say which run it described -- while
consistency at gamma=1 spans 23% across three seeds, wide enough to read as a
regression that is not there.

These build the .pt files `analyze` reads, so they exercise the real grouping
without sampling anything.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_run(directory, gamma, seed, offset):
    """One sampled_*.pt, with `offset` shifting the fiber samples so that seeds
    are distinguishable in every reported metric."""
    os.makedirs(directory, exist_ok=True)
    n, dim = 16, 8
    torch.manual_seed(0)
    originals = torch.zeros(n, dim)
    torch.save({
        "invariances": torch.full((n, dim), float(offset)),
        "invariances_raw": torch.zeros(n, dim),
        "originals": originals,
        "originals_raw": originals,
        "invariances_embeddings": torch.full((n, 2), float(offset)),
        "original_embeddings": torch.zeros(n, 2),
        "feature_names": np.array([f"f{i}" for i in range(dim)]),
        "config": {"gamma": gamma, "seed": seed, "tag": "sweep"},
    }, os.path.join(directory,
                    f"sampled_htru2_invariances_gamma={gamma}_sweep_s{seed}_{offset}.pt"))


@pytest.fixture
def sweep(tmp_path):
    """A two-gamma, three-seed sweep plus the data and subject model analyze needs."""
    invariances = tmp_path / "invariances"
    for gamma in (1.0, 10.0):
        for seed, offset in enumerate((1, 2, 3)):
            make_run(str(invariances), gamma, seed, offset)

    data = tmp_path / "htru2.npz"
    np.savez(data, X_test=np.zeros((16, 8), dtype=np.float32),
             feature_names=np.array([f"f{i}" for i in range(8)]))

    from experiments.htru2.train_subject_model import HTRU2SubjectModel
    config = {"in_dim": 8, "hidden": 4, "n_hidden": 1, "n_classes": 2}
    ckpt = tmp_path / "subject_model.pt"
    torch.save({"model_config": config,
                "model_state_dict": HTRU2SubjectModel(**config).state_dict()}, ckpt)
    return tmp_path, invariances, data, ckpt


def run_analyze(sweep, *extra):
    tmp_path, invariances, data, ckpt = sweep
    out = tmp_path / "analysis.json"
    result = subprocess.run(
        [sys.executable, "-m", "experiments.htru2.analyze",
         "--invariance-dir", str(invariances), "--data", str(data),
         "--subject-ckpt", str(ckpt), "--out-json", str(out), *extra],
        cwd=REPO, capture_output=True, text=True)
    return result, out


def test_reports_seed_zero_by_default(sweep):
    """The reference is seed 0, so that is what by_gamma describes."""
    result, out = run_analyze(sweep)
    assert result.returncode == 0, result.stderr
    analysis = json.loads(out.read_text())

    assert analysis["reported_seed"] == 0
    for gamma in ("1.0", "10.0"):
        entry = analysis["by_gamma"][gamma]
        assert entry["seed"] == 0
        assert entry["seeds_available"] == [0, 1, 2]
        # seed 0 was written with offset 1; picking a later seed would give 2 or 3
        assert entry["fiber_l2_mean"] == pytest.approx(1.0)


def test_every_seed_is_kept_not_just_the_last(sweep):
    """The spread across seeds is what the figures' error bars want."""
    _, out = run_analyze(sweep)
    across = json.loads(out.read_text())["by_gamma"]["1.0"]["across_seeds"]
    assert across["fiber_l2_mean"]["n_seeds"] == 3
    assert across["fiber_l2_mean"]["mean"] == pytest.approx(2.0)   # offsets 1, 2, 3
    assert across["fiber_l2_mean"]["std"] == pytest.approx(1.0)


def test_seed_is_selectable(sweep):
    result, out = run_analyze(sweep, "--seed", "2")
    assert result.returncode == 0, result.stderr
    entry = json.loads(out.read_text())["by_gamma"]["1.0"]
    assert entry["seed"] == 2
    assert entry["fiber_l2_mean"] == pytest.approx(3.0)


def test_missing_seed_is_refused_rather_than_substituted(sweep):
    """The old failure was silent substitution; anything but seed 0 must be asked for."""
    result, _ = run_analyze(sweep, "--seed", "7")
    assert result.returncode != 0
    assert "no --seed 7 run" in result.stdout + result.stderr


def test_duplicate_gamma_seed_is_refused(sweep):
    """Two runs of the same setting cannot both be 'the' seed-0 run."""
    tmp_path, invariances, _, _ = sweep
    make_run(str(invariances), 1.0, 0, 99)      # a second seed-0 run at gamma 1
    result, _ = run_analyze(sweep)
    assert result.returncode != 0
    assert "gamma=1.0 seed=0" in result.stdout + result.stderr
