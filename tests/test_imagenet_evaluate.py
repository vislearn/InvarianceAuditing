"""`experiments/imagenet/evaluate.py`, pooling and the staleness warning.

The pooled report concatenates every key the runs share, so anything that is not
a tensor takes the whole pooling down. That is not hypothetical: recording the
query drift inside the run dict crashed exactly the two settings whose runs were
stale, which are the ones the warning exists for.
"""

import json

import pytest
import torch

from experiments.imagenet import evaluate


def make_run(directory, size=4, dim=3, neighbours=None, revision="abc123"):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(len(str(directory)))
    torch.save({"original_embeddings": torch.randn(size, dim),
                "invariances_embeddings": torch.randn(size, dim),
                "originals": torch.randn(size, 3, 4, 4),
                "labels": torch.zeros(size, dtype=torch.long)},
               directory / "chunk_0.pt")
    (directory / "config.json").write_text(json.dumps(
        {"args": {"subject_model": "dinov2", "dataset": "imagenet",
                  "num_images": size, "samples_per_image": 1, "seed": 0},
         "revision": revision}))
    if neighbours is not None:
        torch.save(neighbours, directory / "nearest_neighbours.pt")
    return directory


def neighbour_file(size=4, dim=3, agree=True, drift=1e-9):
    return {"nn_embeddings": torch.randn(size, dim),
            "nn_indices": torch.arange(size),
            "query_embeddings": torch.randn(size, dim),
            "queries": "reembed",
            "stored_embeddings_agree": agree,
            "query_drift": drift}


def run_cli(target, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["evaluate", str(target)])
    evaluate.main()
    return capsys.readouterr().out


def test_pooling_survives_the_neighbour_metadata(tmp_path, monkeypatch, capsys):
    """The regression: a float recorded beside the tensors broke torch.cat."""
    setting = tmp_path / "dinov2_imagenet"
    for name in ("run_a", "run_b"):
        make_run(setting / name, neighbours=neighbour_file(agree=False, drift=0.23))
    out = run_cli(setting, monkeypatch, capsys)
    # The exact label depends on whether the runs read as draws or shards; that
    # a pooled block was reached at all is the regression.
    assert "POOLED" in out


def test_a_stale_run_says_its_fiber_loss_is_not_comparable(tmp_path, monkeypatch, capsys):
    setting = tmp_path / "dinov2_imagenet"
    for name in ("run_a", "run_b"):
        make_run(setting / name, neighbours=neighbour_file(agree=False, drift=0.23))
    out = run_cli(setting, monkeypatch, capsys)
    assert "stale phi" in out and "2.3e-01" in out
    assert "have to be re-sampled" in out


def test_a_sound_run_carries_no_warning(tmp_path, monkeypatch, capsys):
    setting = tmp_path / "dinov2_imagenet"
    for name in ("run_a", "run_b"):
        make_run(setting / name, neighbours=neighbour_file(agree=True))
    out = run_cli(setting, monkeypatch, capsys)
    assert "stale phi" not in out
    assert "l2 (nearest)" in out


def test_the_neighbour_column_uses_the_queries_that_were_searched(tmp_path, monkeypatch, capsys):
    """Not the run's stored embeddings, which may be a different phi."""
    setting = tmp_path / "dinov2_imagenet" / "run_a"
    found = neighbour_file(agree=True)
    make_run(setting, neighbours=found)
    out = run_cli(setting, monkeypatch, capsys)
    expected = ((found["query_embeddings"] - found["nn_embeddings"]) ** 2).sum(-1).mean()
    assert f"{expected:9.1f}".strip() in out


def test_a_stale_run_without_recomputed_queries_omits_the_column(tmp_path, monkeypatch, capsys):
    """Differencing a recomputed baseline against stored embeddings would mix
    two representations and print a plausible wrong number."""
    setting = tmp_path / "dinov2_imagenet" / "run_a"
    found = neighbour_file(agree=False, drift=0.2)
    del found["query_embeddings"]
    make_run(setting, neighbours=found)
    out = run_cli(setting, monkeypatch, capsys)
    assert "nearest neighbour omitted" in out
    assert "l2 (nearest)" not in out
