"""That the FID path embeds each batch once.

`compute_fid_openai_tf` rebuilt the TensorFlow session, re-embedded the reference
batch, and computed Inception Score, sFID and precision/recall on every call.
Table 3 uses one of those numbers, and colorMNIST asks for it 10 times per model
across 21 models against one fixed set of originals -- so the reference batch was
embedded 210 times and the O(N^2) precision/recall was computed and discarded 210
times, over a 40k-image test set.

Nothing here touches TensorFlow: the evaluator is replaced with a counting stub,
so these check the bookkeeping that decides how much work gets done, which is
where the cost was.
"""

import numpy as np
import pytest
import torch

fid = pytest.importorskip("fff.evaluate.fid",
                          reason="FID path needs tensorflow")


class CountingEvaluator:
    """Stands in for Evaluator, recording how many images it embeds."""

    def __init__(self):
        self.calls = 0
        self.images = 0

    def compute_statistics(self, activations):
        return fid.Evaluator.compute_statistics(self, activations)


@pytest.fixture
def counting(monkeypatch):
    stub = CountingEvaluator()

    def fake_pool_activations(images, batch_size=64):
        stub.calls += 1
        stub.images += len(images)
        # A deterministic, well-conditioned embedding: enough for the Frechet
        # distance to be finite and for equal inputs to score 0.
        flat = images.reshape(len(images), -1).numpy().astype(np.float64)
        rng = np.random.default_rng(0)
        projection = rng.standard_normal((flat.shape[1], 8))
        return flat @ projection

    monkeypatch.setattr(fid, "_pool_activations", fake_pool_activations)
    monkeypatch.setattr(fid, "_evaluator", lambda: stub)
    fid._REFERENCE_CACHE.clear()
    yield stub
    fid._REFERENCE_CACHE.clear()


def batch(seed, n=32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, 8, 8, generator=g)


def test_reference_is_embedded_once_across_slots(counting):
    """The 10 slots of one model share one reference embedding."""
    reference = batch(0)
    slots = [batch(i + 1) for i in range(10)]

    stats = fid.reference_statistics(reference)
    for slot in slots:
        fid.compute_fid_fast(reference, slot, reference_stats=stats)

    # 1 reference + 10 samples, not 20.
    assert counting.calls == 11


def test_reference_cache_survives_across_models(counting):
    """Every colorMNIST model scores against the same originals."""
    reference = batch(0)
    for model in range(3):
        fid.compute_fid_fast(reference, batch(100 + model))

    # 1 reference embedded on the first model, then 3 samples.
    assert counting.calls == 4


def test_cache_is_keyed_by_content_not_identity(counting):
    """An equal tensor must hit the cache; a different one must not."""
    reference = batch(0)
    fid.reference_statistics(reference)
    fid.reference_statistics(reference.clone())
    assert counting.calls == 1

    fid.reference_statistics(batch(7))
    assert counting.calls == 2


def test_identical_batches_score_zero(counting):
    """A sanity check that the Frechet distance is still being computed."""
    reference = batch(0)
    assert fid.compute_fid_fast(reference, reference.clone()) == pytest.approx(0.0, abs=1e-6)


def test_differing_batches_score_above_zero(counting):
    reference = batch(0)
    assert fid.compute_fid_fast(reference, batch(1) + 0.5) > 1e-6
