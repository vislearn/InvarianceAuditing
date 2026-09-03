"""Where `nearest_neighbours.py` gets its query representations from.

A run sampled before a subject-model fix stores embeddings the current code
would not produce. The baseline itself is still recoverable, because it needs
only *which images were audited* -- which the run also stores -- and phi as it
is now. These check that the drift is noticed and that the recomputed queries
are what gets used and recorded.
"""

import json

import pytest
import torch
import torch.nn as nn

from experiments.imagenet import nearest_neighbours as nnb


class Projection(nn.Module):
    """phi(x) = the first three pixels; stands in for a subject model."""

    def forward(self, x):
        return x.reshape(x.shape[0], -1)[:, :3]


@pytest.fixture
def run(tmp_path):
    """A run directory whose stored embeddings are deliberately wrong."""
    torch.manual_seed(0)
    images = torch.randn(6, 3, 4, 4)
    directory = tmp_path / "dinov2_imagenet" / "run_a"
    directory.mkdir(parents=True)
    torch.save({"originals": images,
                # what a pre-fix phi produced: not Projection(images)
                "original_embeddings": Projection()(images) + 3.0,
                "invariances_embeddings": Projection()(images),
                "labels": torch.zeros(6, dtype=torch.long)},
               directory / "chunk_0.pt")
    (directory / "config.json").write_text(json.dumps(
        {"args": {"subject_model": "dinov2", "dataset": "imagenet",
                  "resize_to": 256}}))
    return directory, images


@pytest.fixture
def patched(monkeypatch):
    torch.manual_seed(1)
    corpus = torch.utils.data.TensorDataset(torch.randn(24, 3, 4, 4))
    monkeypatch.setattr(nnb, "build_subject_model", lambda name, device: Projection())
    monkeypatch.setattr(nnb, "load_dataset", lambda **kw: (None, None, corpus))
    monkeypatch.setattr(nnb.paths, "data", lambda *a, **k: "unused")
    return corpus


def run_cli(directory, monkeypatch, *extra):
    monkeypatch.setattr("sys.argv", ["nearest_neighbours", str(directory), *extra])
    nnb.main()
    return torch.load(directory / nnb.NEIGHBOUR_FILE, map_location="cpu",
                      weights_only=False)


def test_drift_is_recorded_and_queries_recomputed_by_default(run, patched, monkeypatch):
    directory, images = run
    written = run_cli(directory, monkeypatch)
    assert written["queries"] == "reembed"
    assert written["stored_embeddings_agree"] is False
    assert written["query_drift"] > 1e-5
    # what got searched is phi(images) as computed now, not what the run stored
    torch.testing.assert_close(written["query_embeddings"], Projection()(images))


def test_stored_queries_are_still_available_and_flagged(run, patched, monkeypatch):
    directory, images = run
    written = run_cli(directory, monkeypatch, "--queries", "stored")
    assert written["queries"] == "stored"
    assert written["stored_embeddings_agree"] is False
    torch.testing.assert_close(written["query_embeddings"],
                               Projection()(images) + 3.0)


def test_a_run_whose_embeddings_still_agree_is_not_flagged(tmp_path, patched, monkeypatch):
    torch.manual_seed(2)
    images = torch.randn(5, 3, 4, 4)
    directory = tmp_path / "dinov2_imagenet" / "run_ok"
    directory.mkdir(parents=True)
    torch.save({"originals": images,
                "original_embeddings": Projection()(images),
                "invariances_embeddings": Projection()(images),
                "labels": torch.zeros(5, dtype=torch.long)},
               directory / "chunk_0.pt")
    (directory / "config.json").write_text(json.dumps(
        {"args": {"subject_model": "dinov2", "dataset": "imagenet",
                  "resize_to": 256}}))
    written = run_cli(directory, monkeypatch)
    assert written["stored_embeddings_agree"] is True
    assert written["query_drift"] <= 1e-5


def test_results_split_back_to_the_runs_they_came_from(tmp_path, patched, monkeypatch):
    """Pooled search, per-run files: each must get its own slice, in order."""
    torch.manual_seed(3)
    setting = tmp_path / "dinov2_imagenet"
    sizes = [4, 7]
    for name, size in zip("ab", sizes):
        directory = setting / f"run_{name}"
        directory.mkdir(parents=True)
        images = torch.randn(size, 3, 4, 4)
        torch.save({"originals": images,
                    "original_embeddings": Projection()(images),
                    "invariances_embeddings": Projection()(images),
                    "labels": torch.zeros(size, dtype=torch.long)},
                   directory / "chunk_0.pt")
        (directory / "config.json").write_text(json.dumps(
            {"args": {"subject_model": "dinov2", "dataset": "imagenet",
                      "resize_to": 256}}))
    monkeypatch.setattr("sys.argv", ["nearest_neighbours", str(setting)])
    nnb.main()
    for name, size in zip("ab", sizes):
        written = torch.load(setting / f"run_{name}" / nnb.NEIGHBOUR_FILE,
                             map_location="cpu", weights_only=False)
        assert len(written["nn_embeddings"]) == size
        assert len(written["nn_indices"]) == size
        assert len(written["query_embeddings"]) == size


def test_search_set_queries_looks_only_among_the_audited_images(run, patched, monkeypatch):
    """The candidates are the queries themselves, so every neighbour must be one
    of them -- and the stored embeddings must be those, not the split's."""
    directory, images = run
    written = run_cli(directory, monkeypatch, "--search-set", "queries")
    assert written["search_set"] == "queries"
    queries = written["query_embeddings"]
    assert written["nn_indices"].max() < len(queries)
    torch.testing.assert_close(written["nn_embeddings"], queries[written["nn_indices"]])
    # and no query is its own neighbour
    assert (written["nn_indices"] != torch.arange(len(queries))).all()


def test_a_smaller_search_set_gives_neighbours_no_closer(tmp_path, monkeypatch):
    """The mechanism behind the ImageNet hypothesis: the audited images are a
    subset of the split, so searching only them cannot find a closer neighbour
    than searching all of it -- restricting can only raise the baseline.

    The run's images have to actually be part of the corpus for that to be the
    claim under test; a disjoint fixture would say nothing.
    """
    torch.manual_seed(7)
    corpus_images = torch.randn(30, 3, 4, 4)
    corpus = torch.utils.data.TensorDataset(corpus_images)
    monkeypatch.setattr(nnb, "build_subject_model", lambda name, device: Projection())
    monkeypatch.setattr(nnb, "load_dataset", lambda **kw: (None, None, corpus))
    monkeypatch.setattr(nnb.paths, "data", lambda *a, **k: "unused")

    audited = corpus_images[:8]
    directory = tmp_path / "dinov2_imagenet" / "run_a"
    directory.mkdir(parents=True)
    torch.save({"originals": audited,
                "original_embeddings": Projection()(audited),
                "invariances_embeddings": Projection()(audited),
                "labels": torch.zeros(8, dtype=torch.long)},
               directory / "chunk_0.pt")
    (directory / "config.json").write_text(json.dumps(
        {"args": {"subject_model": "dinov2", "dataset": "imagenet",
                  "resize_to": 256}}))

    wide = run_cli(directory, monkeypatch)
    narrow = run_cli(directory, monkeypatch, "--search-set", "queries", "--overwrite")

    def loss(written):
        return ((written["query_embeddings"] - written["nn_embeddings"]) ** 2).sum(-1)

    assert (loss(narrow) >= loss(wide) - 1e-6).all()
    assert loss(narrow).mean() > loss(wide).mean()


def test_queries_dataset_says_what_it_measured_on_a_partly_audited_split(
        run, patched, monkeypatch, capsys):
    """6 fibers against a 24-image split. Not an error -- a neighbour's distance
    depends on the candidate set, not on how many queries you ask about -- but
    the number is about the split, not the audited images, and must say so."""
    directory, _ = run
    monkeypatch.setattr("sys.argv", ["nearest_neighbours", str(directory),
                                     "--queries", "dataset"])
    nnb.main()
    out = capsys.readouterr().out
    assert "6 fibers were audited but this measures all 24" in out
    assert "nearest neighbour over the whole split" in out


def test_queries_dataset_reports_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """The cue conflict case: every image was audited, so the split is the query
    set and the baseline can be asked at any resolution."""
    torch.manual_seed(6)
    corpus_images = torch.randn(9, 3, 4, 4)
    corpus = torch.utils.data.TensorDataset(corpus_images)
    monkeypatch.setattr(nnb, "build_subject_model", lambda name, device: Projection())
    monkeypatch.setattr(nnb, "load_dataset", lambda **kw: (None, None, corpus))
    monkeypatch.setattr(nnb.paths, "data", lambda *a, **k: "unused")

    directory = tmp_path / "dinov2_cue_conflict" / "run_a"
    directory.mkdir(parents=True)
    torch.save({"originals": corpus_images,
                "original_embeddings": Projection()(corpus_images),
                "invariances_embeddings": Projection()(corpus_images),
                "labels": torch.zeros(9, dtype=torch.long)},
               directory / "chunk_0.pt")
    (directory / "config.json").write_text(json.dumps(
        {"args": {"subject_model": "dinov2", "dataset": "cue_conflict",
                  "resize_to": 256}}))
    monkeypatch.setattr("sys.argv", ["nearest_neighbours", str(directory),
                                     "--queries", "dataset"])
    nnb.main()
    out = capsys.readouterr().out
    assert "nearest neighbour over the whole split" in out
    assert "reports only; nothing written" in out
    assert not (directory / nnb.NEIGHBOUR_FILE).exists()


def test_report_only_leaves_the_existing_answer_alone(run, patched, monkeypatch, capsys):
    """A diagnostic must not replace the baseline evaluate.py reads."""
    directory, _ = run
    real = run_cli(directory, monkeypatch)
    monkeypatch.setattr("sys.argv", ["nearest_neighbours", str(directory),
                                     "--search-set", "queries", "--report-only"])
    nnb.main()
    assert "report only, nothing written" in capsys.readouterr().out
    after = torch.load(directory / nnb.NEIGHBOUR_FILE, map_location="cpu",
                       weights_only=False)
    torch.testing.assert_close(after["nn_embeddings"], real["nn_embeddings"])
    assert after["search_set"] == "dataset"


def test_queries_dataset_implies_report_only(tmp_path, monkeypatch, capsys):
    """Otherwise a directory that already has an answer is skipped before the
    diagnostic runs, and the job prints a heading with nothing under it."""
    torch.manual_seed(8)
    corpus_images = torch.randn(9, 3, 4, 4)
    corpus = torch.utils.data.TensorDataset(corpus_images)
    monkeypatch.setattr(nnb, "build_subject_model", lambda name, device: Projection())
    monkeypatch.setattr(nnb, "load_dataset", lambda **kw: (None, None, corpus))
    monkeypatch.setattr(nnb.paths, "data", lambda *a, **k: "unused")

    directory = tmp_path / "dinov2_cue_conflict" / "run_a"
    directory.mkdir(parents=True)
    torch.save({"originals": corpus_images,
                "original_embeddings": Projection()(corpus_images),
                "invariances_embeddings": Projection()(corpus_images),
                "labels": torch.zeros(9, dtype=torch.long)},
               directory / "chunk_0.pt")
    (directory / "config.json").write_text(json.dumps(
        {"args": {"subject_model": "dinov2", "dataset": "cue_conflict",
                  "resize_to": 256}}))
    # the answer is already there, as it is after a first run
    run_cli(directory, monkeypatch)
    assert (directory / nnb.NEIGHBOUR_FILE).exists()
    capsys.readouterr()

    monkeypatch.setattr("sys.argv", ["nearest_neighbours", str(directory),
                                     "--queries", "dataset"])
    nnb.main()
    out = capsys.readouterr().out
    assert "nearest neighbour over the whole split" in out
    assert "skipping" not in out
