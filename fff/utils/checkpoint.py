"""Checkpoint loading compatible with torch >= 2.6.

Checkpoints written by this codebase embed their hyperparameters as
``lightning_trainable`` ``AttributeDict`` objects. Since torch 2.6, ``torch.load``
defaults to ``weights_only=True``, whose restricted unpickler cannot reconstruct
those objects, so every ``load_from_checkpoint`` call fails. The helpers here load
with ``weights_only=False``, which executes pickled code and is therefore only
appropriate for checkpoints you trust -- ours and your own.
"""

from contextlib import contextmanager

import torch

_original_load = torch.load


def default_map_location():
    """``"cpu"`` where CUDA is unavailable, otherwise ``None`` (leave as stored).

    Our checkpoints were written on GPU machines, so their storages carry CUDA
    devices and unpickling one on a CPU-only box raises inside torch.load --
    before Lightning gets a chance to place the model.
    """
    return None if torch.cuda.is_available() else "cpu"


@contextmanager
def trusted_load():
    """Within this context, ``torch.load`` defaults to ``weights_only=False``.

    Re-entrant: the exit restores whatever ``torch.load`` was on entry, not the
    module-level original. Restoring the original would un-patch the outer
    block, so everything after an inner ``with`` would load with
    ``weights_only=True`` again -- and this nests in practice, since loading a
    FiberModel inside a ``trusted_load`` loads its subject model through
    ``load_checkpoint``.
    """
    outer = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        # An explicit map_location still wins.
        if not torch.cuda.is_available():
            kwargs.setdefault("map_location", "cpu")
        return _original_load(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = outer


def load_checkpoint(cls, path, **kwargs):
    """``cls.load_from_checkpoint(path)`` for a trusted checkpoint of ours."""
    if not torch.cuda.is_available():
        kwargs.setdefault("map_location", "cpu")
    with trusted_load():
        return cls.load_from_checkpoint(path, **kwargs)
