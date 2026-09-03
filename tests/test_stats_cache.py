"""That a model compute_statistics has already scored is not scored again.

Caching samples.pt but not stats.pt would make every rerun redo the whole FID
sweep -- an Inception pass over the 40k test set per sample slot, 10 slots, 21
models -- even though FID is a pure function of a samples.pt that has not
changed. These pin the cache down, including the two ways a cached entry must
NOT be trusted: stats older than the samples they describe, and a NaN FID left
behind by --skip_fid.
"""

import os

import numpy as np
import pytest
import torch

cs = pytest.importorskip("experiments.colormnist.compute_statistics",
                         reason="needs the fff training stack")


STATS = ((0.1, 0.01), [(0.2, 0.02)], [(0.3, 0.03)], (0.4, 0.04), (5.0, 0.5))


@pytest.fixture
def plot_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "plot_folder", str(tmp_path))
    return tmp_path


def write(plot_folder, name, fid=(5.0, 0.5), samples=False):
    d = plot_folder / name
    d.mkdir(parents=True, exist_ok=True)
    if samples:  # written first, so stats.pt is the newer file
        torch.save({"samples": torch.zeros(1)}, d / "samples.pt")
    cs.save_model_stats(*STATS[:-1], fid, name)
    return d


def test_no_stats_file_is_a_miss(plot_folder):
    assert cs.cached_stats("fff_lambda0") is None


def test_finished_model_is_reused(plot_folder):
    write(plot_folder, "fff_lambda0", samples=True)
    stats = cs.cached_stats("fff_lambda0")
    assert stats is not None
    assert stats[-1] == (5.0, 0.5)


def test_stats_older_than_samples_are_ignored(plot_folder):
    d = write(plot_folder, "fff_lambda0", samples=True)
    # re-sampling touches samples.pt: the stats now describe images that are gone
    os.utime(d / "samples.pt", (10**9, 10**9))
    os.utime(d / "stats.pt", (10**9 - 60, 10**9 - 60))
    assert cs.cached_stats("fff_lambda0") is None


def test_skipped_fid_is_not_reused_when_fid_is_wanted(plot_folder):
    write(plot_folder, "fff_lambda0", fid=(float("nan"), float("nan")), samples=True)
    assert cs.cached_stats("fff_lambda0") is None
    # ... but a run that is skipping FID anyway has nothing left to compute
    reused = cs.cached_stats("fff_lambda0", skip_fid=True)
    assert reused is not None and np.isnan(reused[-1][0])


def test_evaluate_model_returns_the_cache_without_a_model(plot_folder):
    """The point of the cache: no checkpoint is loaded, no sampling happens."""
    write(plot_folder, "fff_lambda0", samples=True)

    def explode(*a, **k):
        raise AssertionError("load_model must not be called for a cached model")

    saved = cs.load_model
    cs.load_model = explode
    try:
        stats = cs.evaluate_model("fff_lambda0")
    finally:
        cs.load_model = saved
    assert stats[-1] == (5.0, 0.5)


def test_recompute_bypasses_the_cache(plot_folder):
    write(plot_folder, "fff_lambda0", samples=True)

    def explode(*a, **k):
        raise RuntimeError("reached the model")

    saved = cs.load_model
    cs.load_model = explode
    try:
        with pytest.raises(RuntimeError, match="reached the model"):
            cs.evaluate_model("fff_lambda0", recompute=True)
    finally:
        cs.load_model = saved


class TestSeedRepeatRunNames:
    """That `--models fff_lambda10_seed1` is accepted and scored.

    A run trained under a non-default seed is named `<config stem>_seed<N>` so
    repeats sit beside the original instead of overwriting it. Rejecting any
    name not in the hardcoded table would make a seed sweep unscoreable -- and a
    seed sweep is how the spread at high lambda gets measured.
    """

    KNOWN = {"fff_lambda10", "dnf_lambda0", "fm_lambda0"}

    def test_configured_run_is_known(self):
        assert cs.is_known_run("fff_lambda10", self.KNOWN)

    def test_seed_repeat_is_known(self):
        for n in ("fff_lambda10_seed1", "dnf_lambda0_seed12", "fm_lambda0_seed0"):
            assert cs.is_known_run(n, self.KNOWN), n

    def test_typo_is_still_rejected(self):
        for n in ("fff_lambda11", "fff_lambda11_seed1", "", "_seed1"):
            assert not cs.is_known_run(n, self.KNOWN), n

    def test_seed_must_be_numeric(self):
        # `_seedbest` is a typo, not a seed repeat; only digits count
        assert not cs.is_known_run("fff_lambda10_seedbest", self.KNOWN)
        assert not cs.is_known_run("fff_lambda10_seed", self.KNOWN)

    def test_lambda_is_not_mistaken_for_a_seed(self):
        # the suffix rule must not let an unconfigured lambda in through it
        assert not cs.is_known_run("fff_lambda5_seed1", self.KNOWN)


class TestMissingRunDiagnostics:
    """That a missing run says where it looked, not just a bare relative path.

    `log_folder` defaults to a relative "lightning_logs" and only 38_/39_ export
    FIBER_MODEL_LOGS, so a hand-run command failed with
    "Checkpoint root directory 'lightning_logs/<run>' does not exist" -- which
    names neither the env var nor the runs that do exist.
    """

    def test_names_the_env_var_when_the_log_root_is_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "log_folder", str(tmp_path / "nope"))
        with pytest.raises(SystemExit) as e:
            cs.load_model("fff_lambda10_seed1")
        assert "FIBER_MODEL_LOGS" in str(e.value)

    def test_lists_available_runs_when_the_log_root_exists(self, tmp_path, monkeypatch):
        (tmp_path / "fff_lambda0").mkdir()
        (tmp_path / "dnf_lambda1").mkdir()
        monkeypatch.setattr(cs, "log_folder", str(tmp_path))
        with pytest.raises(SystemExit) as e:
            cs.load_model("fff_lambda10_seed1")
        msg = str(e.value)
        assert "fff_lambda0" in msg and "dnf_lambda1" in msg
        # the env-var hint would be wrong here: the directory is fine
        assert "FIBER_MODEL_LOGS" not in msg

    def test_empty_log_root_is_reported_as_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "log_folder", str(tmp_path))
        with pytest.raises(SystemExit, match="none"):
            cs.load_model("fff_lambda10_seed1")
