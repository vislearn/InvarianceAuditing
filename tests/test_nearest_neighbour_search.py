"""`NearestNeighborSearch`, the brute-force nearest neighbour in fff/ndtm.py.

Note this is *not* what produces Table 5's nearest-neighbour column -- that is
`fff/evaluate/nearest_neighbour.py`, driven by the per-experiment
`nearest_neighbours.py` scripts. Nothing in this repository calls this class;
it is kept as a small standalone utility, and tested so that staying is cheap.
"""

import pytest
import torch
import torch.nn as nn

import fff.ndtm
from fff.ndtm import NearestNeighborSearch


class Projection(nn.Module):
    """phi(x) = the first three coordinates."""

    def forward(self, x):
        return x.reshape(x.shape[0], -1)[:, :3]


@pytest.fixture(autouse=True)
def on_cpu(monkeypatch):
    """The class reads the module-level `device`, fixed at import time."""
    monkeypatch.setattr(fff.ndtm, "device", "cpu")


def dataset(rows):
    return torch.utils.data.TensorDataset(torch.as_tensor(rows, dtype=torch.float32))


def test_nearest_neighbour_is_the_closest_in_representation_space():
    data = dataset([[0.0, 0, 0, 9], [5.0, 0, 0, 0], [1.0, 0, 0, 0]])
    search = NearestNeighborSearch(data, data, data, Projection(), batch_size=3)
    query = torch.tensor([[0.9, 0.0, 0.0, 0.0]])
    distance, neighbour = search.find_nearest_neighbor(query, use_datasets="train")
    assert distance.item() == pytest.approx(0.1, abs=1e-5)
    torch.testing.assert_close(neighbour, torch.tensor([1.0, 0, 0, 0]))


def test_the_query_itself_is_excluded():
    """Otherwise the baseline is zero and says nothing."""
    data = dataset([[1.0, 0, 0, 0], [3.0, 0, 0, 0]])
    search = NearestNeighborSearch(data, data, data, Projection(), batch_size=2)
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    distance, _ = search.find_nearest_neighbor(query, use_datasets="train")
    assert distance.item() == pytest.approx(2.0, abs=1e-5)


def test_searching_a_named_subset_only_reads_that_subset():
    train = dataset([[9.0, 0, 0, 0]])
    test = dataset([[1.0, 0, 0, 0]])
    search = NearestNeighborSearch(train, train, test, Projection(), batch_size=1)
    query = torch.zeros(1, 4)
    assert search.find_nearest_neighbor(query, use_datasets="test")[0].item() \
        == pytest.approx(1.0, abs=1e-5)
    assert search.find_nearest_neighbor(query, use_datasets=["train"])[0].item() \
        == pytest.approx(9.0, abs=1e-5)


def test_a_batch_holding_only_the_query_is_skipped_not_fatal():
    """The identity filter can empty a batch: the last one of a shard, or any
    batch of a dataset with duplicates. argmin over the empty result used to
    raise and kill the whole search."""
    data = dataset([[2.0, 0, 0, 0], [1.0, 0, 0, 0]])
    search = NearestNeighborSearch(data, data, data, Projection(), batch_size=1)
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    distance, neighbour = search.find_nearest_neighbor(query, use_datasets="train")
    assert distance.item() == pytest.approx(1.0, abs=1e-5)
    torch.testing.assert_close(neighbour, torch.tensor([2.0, 0, 0, 0]))


def test_every_batch_being_the_query_leaves_no_neighbour():
    data = dataset([[1.0, 0, 0, 0]])
    search = NearestNeighborSearch(data, data, data, Projection(), batch_size=1)
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    distance, neighbour = search.find_nearest_neighbor(query, use_datasets="train")
    assert distance == torch.inf and neighbour is None
