"""Shared fixtures, and the convention the suite is written to.

These tests run on CPU, without datasets and without checkpoints: everything
they need is built in the test itself. That is deliberate -- a test suite that
needs the 40 GB of ImageNet samples is a test suite nobody runs before pushing.
The generative model in test_ndtm.py is the exact denoiser for a Gaussian, so
NDTM's two claims (samples stay on the model's distribution, and land on the
subject model's fiber) are checked against closed forms rather than eyeballed.

A test written against a bug that is still open is marked `xfail(strict=True)`
with a reason beginning "BUG:". That keeps the suite green while the bug is open
and turns it red the moment the bug is fixed, so nobody has to remember to
unmark it. `pytest -rx` lists any that are open; there are none at present.
"""

import os
import sys

# The samplers wrap their timestep loop in tqdm, which writes a progress bar to
# stderr for every one of them. Under pytest that is thousands of lines of
# captured output on the first failure.
os.environ.setdefault("TQDM_DISABLE", "1")

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(autouse=True)
def deterministic():
    """Every test starts from the same RNG state."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def cpu_only(monkeypatch):
    """Pretend there is no GPU.

    Several modules read `torch.cuda.is_available()` at import time and again at
    call time to pick a device; the call-time reads have to keep working on a
    machine without a GPU.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
