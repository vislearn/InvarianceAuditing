"""`fff/evaluate/nearest_neighbour.py` -- what produces Table 5's NN column.

Distinct from `test_nearest_neighbour_search.py`, which covers the unused
standalone class in `fff/ndtm.py`.

The searches here are pooled: every run of a setting is concatenated into one
call so the dataset is embedded once per setting instead of once per run. That
is only sound if pooling changes nothing about the answer, which is what most of
these assert.
"""

import pytest
import torch
import torch.nn as nn

from fff.evaluate.nearest_neighbour import (nearest_neighbour_embeddings,
                                            nearest_neighbour_indices)


class Projection(nn.Module):
    """phi(x) = the first three coordinates."""

    def forward(self, x):
        return x.reshape(x.shape[0], -1)[:, :3]


def dataset(rows):
    return torch.utils.data.TensorDataset(torch.as_tensor(rows, dtype=torch.float32))


def brute_force(queries, data, model, threshold=1e-4, metric="l2"):
    """The definition, computed the obvious way over the whole search set."""
    candidates = model(data.tensors[0])
    if metric == "l2":
        dist = ((candidates[:, None, :] - queries[None, :, :]) ** 2).mean(-1)
    elif metric == "l1":
        dist = (candidates[:, None, :] - queries[None, :, :]).abs().mean(-1)
    else:
        dist = -(queries.softmax(-1)[None, :, :]
                 * candidates.log_softmax(-1)[:, None, :]).sum(-1)
    dist = torch.where(dist < threshold, torch.full_like(dist, torch.inf), dist)
    return dist.min(dim=0).indices


@pytest.fixture
def corpus():
    torch.manual_seed(0)
    return dataset(torch.randn(40, 5))


@pytest.mark.parametrize("metric", ["l2", "l1", "cross_entropy"])
def test_matches_brute_force(corpus, metric):
    torch.manual_seed(1)
    queries = torch.randn(7, 3)
    got = nearest_neighbour_indices(queries, corpus, Projection(), batch_size=6,
                                    metric=metric, progress=False)
    torch.testing.assert_close(got, brute_force(queries, corpus, Projection(),
                                                metric=metric))


def test_a_query_is_not_its_own_neighbour(corpus):
    """Otherwise the baseline is zero and says nothing."""
    queries = Projection()(corpus.tensors[0][:4])
    got = nearest_neighbour_indices(queries, corpus, Projection(), batch_size=5,
                                    progress=False)
    assert (got != torch.arange(4)).all()


@pytest.mark.parametrize("metric", ["l2", "l1", "cross_entropy"])
def test_pooling_runs_gives_what_searching_them_separately_gives(corpus, metric):
    """The whole point of the pooling: one pass per setting, same answer.

    Twenty shards searched together must land on the neighbours they would have
    found alone, and the concatenated result must split back in run order.
    """
    torch.manual_seed(2)
    runs = [torch.randn(n, 3) for n in (3, 5, 4)]
    separate = torch.cat([
        nearest_neighbour_indices(q, corpus, Projection(), batch_size=7,
                                  metric=metric, progress=False) for q in runs])
    pooled = nearest_neighbour_indices(torch.cat(runs), corpus, Projection(),
                                       batch_size=7, metric=metric, progress=False)
    torch.testing.assert_close(pooled, separate)


@pytest.mark.parametrize("metric", ["l1", "cross_entropy"])
def test_query_chunking_does_not_change_the_answer(corpus, metric):
    """l1 and cross-entropy walk the queries in blocks to bound a (B, N, D)
    tensor that a pooled N would otherwise make enormous."""
    torch.manual_seed(3)
    queries = torch.randn(11, 3)
    whole = nearest_neighbour_indices(queries, corpus, Projection(), batch_size=6,
                                      metric=metric, progress=False,
                                      query_chunk=1000)
    chunked = nearest_neighbour_indices(queries, corpus, Projection(), batch_size=6,
                                        metric=metric, progress=False,
                                        query_chunk=2)
    torch.testing.assert_close(whole, chunked)


def test_embeddings_stay_aligned_when_batched(corpus):
    """The neighbour images are fetched a batch at a time rather than stacked
    all at once; the embeddings must still line up with the indices."""
    torch.manual_seed(4)
    queries = torch.randn(9, 3)
    indices, embeddings, _ = nearest_neighbour_embeddings(
        queries, corpus, Projection(), batch_size=2, progress=False)
    expected = Projection()(corpus.tensors[0][indices])
    assert len(embeddings) == len(queries)
    torch.testing.assert_close(embeddings, expected)


def test_cross_embeddings_describe_the_same_neighbours(corpus):
    """also_embed is the cross-model view of Figures 8 and 18: a second model
    applied to the *first* model's neighbours, not to its own."""
    class Tail(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)[:, -2:]

    torch.manual_seed(5)
    queries = torch.randn(6, 3)
    indices, _, cross = nearest_neighbour_embeddings(
        queries, corpus, Projection(), also_embed={"tail": Tail()},
        batch_size=4, progress=False)
    torch.testing.assert_close(cross["tail"], Tail()(corpus.tensors[0][indices]))


def test_no_neighbour_at_all_is_an_error_not_a_silent_minus_one(corpus):
    """A -1 index would sail on and produce a garbage baseline."""
    only_itself = dataset([[1.0, 0, 0, 0, 0]])
    with pytest.raises(RuntimeError, match="no nearest neighbour"):
        nearest_neighbour_indices(torch.tensor([[1.0, 0.0, 0.0]]), only_itself,
                                  Projection(), batch_size=1, progress=False)
