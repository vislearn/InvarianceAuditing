"""Loading checkpoints under torch >= 2.6.

`trusted_load` monkey-patches `torch.load` for the duration of a block. That is
a global mutation, so what matters is that it is always undone -- a leaked patch
would silently turn `weights_only=True` off for the rest of the process,
including for files the caller did not choose to trust.
"""

import pytest
import torch

from fff.utils.checkpoint import (default_map_location, load_checkpoint,
                                  trusted_load)


class HParams:
    """A checkpoint payload the restricted unpickler will refuse."""

    def __init__(self, a):
        self.a = a


def test_torch_load_is_restored_afterwards():
    original = torch.load
    with trusted_load():
        assert torch.load is not original
    assert torch.load is original


def test_torch_load_is_restored_after_an_exception():
    original = torch.load
    with pytest.raises(RuntimeError):
        with trusted_load():
            raise RuntimeError("boom")
    assert torch.load is original


def test_trusted_load_survives_a_nested_use():
    original = torch.load
    with trusted_load():
        patched = torch.load
        with trusted_load():
            pass
        assert torch.load is patched, "the inner exit undid the outer patch"
    assert torch.load is original


def test_a_trusted_load_reads_a_pickled_object(tmp_path):
    """weights_only=True cannot reconstruct the objects our checkpoints hold.

    Ours embed lightning_trainable AttributeDicts; any class the restricted
    unpickler does not know about stands in for one here.
    """
    path = tmp_path / "thing.pt"
    torch.save({"hyper_parameters": HParams(1)}, path)
    with pytest.raises(Exception):
        torch.load(path, weights_only=True)
    with trusted_load():
        assert torch.load(path)["hyper_parameters"].a == 1


def test_an_explicit_argument_still_wins(tmp_path):
    path = tmp_path / "thing.pt"
    torch.save(torch.ones(2), path)
    with trusted_load():
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded.device.type == "cpu"


def test_map_location_defaults_to_cpu_only_without_a_gpu(cpu_only):
    assert default_map_location() == "cpu"


def test_map_location_is_left_alone_when_there_is_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_map_location() is None


def test_load_checkpoint_passes_its_arguments_through(tmp_path, cpu_only):
    seen = {}

    class Fake:
        @staticmethod
        def load_from_checkpoint(path, **kwargs):
            seen.update(path=path, **kwargs)
            return "model"

    assert load_checkpoint(Fake, "some.ckpt", strict=False) == "model"
    assert seen == {"path": "some.ckpt", "strict": False, "map_location": "cpu"}
