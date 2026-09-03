"""The bookkeeping that decides which numbers get averaged together.

`experiments/common/sampling.py` is where a sampling run records what it was,
and where the evaluators decide what counts as a repeat of a run rather than a
piece of one. Getting that wrong does not raise -- it prints a mean over the
wrong set, which is exactly the failure the module's own docstrings describe
having been hit by. All of it is pure logic over directories and dicts, so it
can be checked without sampling anything.
"""

import os

import pytest
import torch

from experiments.common.sampling import (INCIDENTAL, ChunkWriter,
                                         check_one_setting, draw_arguments,
                                         fiber_identity, group_draws,
                                         load_config, run_directories,
                                         run_directory, save_config, shard,
                                         check_revisions)


def make_run(root, name, args=None, revision="abc123", chunks=1, draw_args=None):
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    for i in range(chunks):
        torch.save({"original_embeddings": torch.zeros(2, 4)},
                   os.path.join(path, f"chunk_{i}.pt"))
    if args is not None:
        save_config(path, {"args": args, "revision": revision},
                    draw_args=draw_args)
    return path


# ------------------------------------------------------------------ ChunkWriter

def test_chunk_writer_flushes_when_full_and_keeps_everything(tmp_path):
    writer = ChunkWriter(str(tmp_path), ["x", "y"], chunk_size=4)
    for _ in range(5):
        writer.add(x=torch.ones(2, 3), y=torch.zeros(2))
    writer.flush()

    chunks = sorted(os.listdir(tmp_path))
    # The evaluators sort chunks by the integer in the name and stop at the
    # first gap, so the numbering has to be dense.
    assert chunks == [f"chunk_{i}.pt" for i in range(len(chunks))]
    total = sum(len(torch.load(os.path.join(tmp_path, c), weights_only=False)["x"])
                for c in chunks)
    assert total == 10, "a flush must not drop buffered batches"


def test_chunk_writer_flush_is_idempotent(tmp_path):
    """The samplers flush per batch and again at the end."""
    writer = ChunkWriter(str(tmp_path), ["x"], chunk_size=100)
    writer.add(x=torch.ones(3))
    writer.flush()
    writer.flush()
    assert os.listdir(tmp_path) == ["chunk_0.pt"], "an empty flush wrote a chunk"


def test_chunk_writer_keeps_the_keys_it_was_given(tmp_path):
    writer = ChunkWriter(str(tmp_path), ["x"], chunk_size=1)
    writer.add(x=torch.ones(1), extra=torch.ones(1))
    saved = torch.load(os.path.join(tmp_path, "chunk_0.pt"), weights_only=False)
    assert set(saved) == {"x"}


def test_chunk_writer_reports_a_missing_field(tmp_path):
    writer = ChunkWriter(str(tmp_path), ["x", "y"], chunk_size=1)
    with pytest.raises(KeyError):
        writer.add(x=torch.ones(1))


def test_chunk_writer_rejects_a_ragged_batch(tmp_path):
    writer = ChunkWriter(str(tmp_path), ["x", "y"], chunk_size=100)
    with pytest.raises((ValueError, AssertionError)):
        writer.add(x=torch.ones(4, 2), y=torch.ones(3, 2))


# ------------------------------------------------------------------------ shard

def test_shards_partition_the_dataset_exactly_once():
    data = list(range(100))
    seen = []
    for i in range(7):
        seen.extend(shard(data, i, 7))
    assert sorted(seen) == data


def test_shards_are_interleaved_not_blocked():
    """Every shard has to be representative -- the runs are reported per-shard."""
    assert list(shard(list(range(10)), 0, 2)) == [0, 2, 4, 6, 8]


def test_a_single_shard_is_the_whole_dataset():
    data = list(range(10))
    assert list(shard(data, 0, 1)) == data


def test_an_unsharded_run_sees_the_same_permutation_as_a_sharded_one():
    """`order` has to apply before the early return, or shard 0 of 1 differs.

    A single job must audit the same images the array would have, in the same
    order, so that a one-off rerun is comparable with the sharded run.
    """
    data = list(range(20))
    one = list(shard(data, 0, 1, order=torch.Generator().manual_seed(0)))
    assert one != data, "the permutation was not applied"
    for i in range(4):
        piece = list(shard(data, i, 4, order=torch.Generator().manual_seed(0)))
        assert piece == one[i::4], f"shard {i} is not the stride it should be"


def test_an_out_of_range_shard_is_refused():
    with pytest.raises(ValueError):
        shard(list(range(10)), 3, 3)


# ----------------------------------------------------------- config round trip

def test_save_config_records_the_revision_and_reads_back(tmp_path):
    save_config(str(tmp_path), {"args": {"seed": 0}})
    config = load_config(str(tmp_path))
    assert config["args"] == {"seed": 0}
    assert "revision" in config


def test_load_config_is_none_for_a_run_that_never_wrote_one(tmp_path):
    assert load_config(str(tmp_path)) is None


def test_save_config_survives_a_gamma_schedule_closure(tmp_path):
    """The whole reason this is JSON and not torch.save."""
    from fff.ndtm import get_gamma_t_fct

    anchors = [(0, 0, 1000, 500), (5.0, 5.0, 500, 0)]
    save_config(str(tmp_path), {"gamma_t": get_gamma_t_fct(anchors)})
    recorded = load_config(str(tmp_path))["gamma_t"]
    assert recorded["anchorpoints"] == [list(a) for a in anchors]
    assert recorded["max_timesteps"] == 1000


def test_save_config_flattens_a_dataclass(tmp_path):
    from fff.ndtm import NDTMConfig

    save_config(str(tmp_path), {"ndtm": NDTMConfig(eta=0.5, N=3)})
    recorded = load_config(str(tmp_path))["ndtm"]
    assert recorded["eta"] == 0.5 and recorded["N"] == 3


def test_save_config_does_not_lose_a_tensor_valued_setting(tmp_path):
    """Not round-trippable, but it must at least be written and readable."""
    save_config(str(tmp_path), {"clip": torch.tensor([1.0])})
    assert load_config(str(tmp_path))["clip"]


def test_load_config_returns_none_for_a_truncated_file(tmp_path):
    """A job killed mid-write leaves half a JSON document."""
    with open(os.path.join(tmp_path, "config.json"), "w") as handle:
        handle.write('{"args": {"seed": 0}')
    assert load_config(str(tmp_path)) is None


def test_run_directory_never_reuses_a_name(tmp_path):
    a = run_directory(str(tmp_path), "setting")
    b = run_directory(str(tmp_path), "setting")
    assert a != b and os.path.isdir(a) and os.path.isdir(b)


# --------------------------------------------------------- resolving what to read

def test_run_directories_expands_a_settings_directory(tmp_path):
    setting = tmp_path / "setting"
    make_run(str(setting), "task_0")
    make_run(str(setting), "task_1")
    assert len(run_directories([str(setting)])) == 2


def test_run_directories_accepts_the_runs_themselves(tmp_path):
    run = make_run(str(tmp_path), "task_0")
    assert run_directories([run]) == [run]


def test_run_directories_skips_a_shard_that_died_before_its_first_flush(tmp_path):
    setting = tmp_path / "setting"
    make_run(str(setting), "task_0")
    os.makedirs(setting / "task_1")
    assert len(run_directories([str(setting)])) == 1


def test_run_directories_refuses_to_return_nothing(tmp_path):
    with pytest.raises(SystemExit):
        run_directories([str(tmp_path)])


def test_run_directories_does_not_report_the_same_run_twice(tmp_path):
    """`evaluate setting setting/*` is a shell glob away from happening.

    Counting one shard twice halves the apparent spread and doubles the
    apparent fiber count.
    """
    setting = tmp_path / "setting"
    run = make_run(str(setting), "task_0")
    resolved = run_directories([str(setting), run])
    assert len(resolved) == len(set(os.path.realpath(p) for p in resolved))


# ------------------------------------------------------------ pooling and draws

def test_two_seeds_of_the_same_setting_are_two_draws(tmp_path):
    base = dict(dataset="imagenet", subject_model="dinov2", num_images=10)
    a = make_run(str(tmp_path), "a", {**base, "seed": 0, "shard": 0})
    b = make_run(str(tmp_path), "b", {**base, "seed": 1, "shard": 0})
    assert fiber_identity(a) == fiber_identity(b)


def test_two_shards_of_the_same_setting_are_not_the_same_fibers(tmp_path):
    base = dict(dataset="imagenet", subject_model="dinov2", seed=0)
    a = make_run(str(tmp_path), "a", {**base, "shard": 0})
    b = make_run(str(tmp_path), "b", {**base, "shard": 1})
    assert fiber_identity(a) != fiber_identity(b)


@pytest.mark.parametrize("field", INCIDENTAL)
def test_incidental_arguments_do_not_split_a_setting(tmp_path, field):
    base = dict(dataset="imagenet", subject_model="dinov2", seed=0, shard=0)
    a = make_run(str(tmp_path), "a", {**base, field: "one"})
    b = make_run(str(tmp_path), "b", {**base, field: "two"})
    assert fiber_identity(a) == fiber_identity(b)


def test_a_declared_draw_argument_wins_over_the_fallback(tmp_path):
    base = dict(dataset="chexpert", seed=0, shard=0)
    a = make_run(str(tmp_path), "a", {**base, "noise": 1},
                 draw_args=["noise"])
    b = make_run(str(tmp_path), "b", {**base, "noise": 2},
                 draw_args=["noise"])
    assert fiber_identity(a) == fiber_identity(b)


def test_fiber_identity_falls_back_for_a_run_without_a_config(tmp_path):
    run = make_run(str(tmp_path), "old")
    assert fiber_identity(run, fallback="sentinel") == "sentinel"


def test_draw_arguments_prefers_sample_seed_when_there_is_one():
    assert draw_arguments({"seed": 0, "sample_seed": 1}) == ("sample_seed",)
    assert draw_arguments({"seed": 0}) == ("seed",)


def test_group_draws_splits_repeats_of_a_complete_set():
    items = [("s0", "d0"), ("s1", "d0"), ("s0", "d1"), ("s1", "d1")]
    sets, draws = group_draws(items, lambda item: item[0])
    assert draws == 2
    assert all(len(one) == 2 for one in sets)
    assert sorted(sum(sets, [])) == sorted(items)


def test_group_draws_pools_an_interrupted_repeat():
    """Two shards, one of them sampled twice: there is no draw index to split on."""
    items = [("s0", "d0"), ("s1", "d0"), ("s0", "d1")]
    sets, draws = group_draws(items, lambda item: item[0])
    assert draws == 1 and sets == [items]


def test_group_draws_treats_a_single_pass_as_one_set():
    items = [("s0", "d0"), ("s1", "d0")]
    sets, draws = group_draws(items, lambda item: item[0])
    assert draws == 1 and sets == [items]


def test_pooling_two_different_experiments_is_refused(tmp_path):
    a = make_run(str(tmp_path), "a", {"dataset": "imagenet", "seed": 0})
    b = make_run(str(tmp_path), "b", {"dataset": "cue_conflict", "seed": 0})
    with pytest.raises(SystemExit):
        check_one_setting([a, b])


def test_pooling_shards_of_one_setting_is_allowed(tmp_path):
    base = {"dataset": "imagenet", "subject_model": "dinov2"}
    a = make_run(str(tmp_path), "a", {**base, "shard": 0})
    b = make_run(str(tmp_path), "b", {**base, "shard": 1})
    check_one_setting([a, b])


def test_pooling_two_step_counts_is_refused(tmp_path):
    """The Qwen row was drawn at 100 steps and redrawn at 200 in the same tree.

    A step count is not a shard of anything: the guidance schedule is a function
    of the diffusion timestep, so the two runs guide by different amounts.
    """
    base = {"dataset": "chexpert", "queries": "chexpert"}
    a = make_run(str(tmp_path), "a", {**base, "num_steps": 100}, revision="same")
    b = make_run(str(tmp_path), "b", {**base, "num_steps": 200}, revision="same")
    with pytest.raises(SystemExit):
        check_one_setting([a, b])


def test_mixed_revisions_are_refused(tmp_path):
    a = make_run(str(tmp_path), "a", {"seed": 0}, revision="before")
    b = make_run(str(tmp_path), "b", {"seed": 0}, revision="after")
    with pytest.raises(SystemExit) as excinfo:
        check_revisions([a, b])
    assert "before" in str(excinfo.value) and "after" in str(excinfo.value)


def test_mixed_revisions_are_reported_when_explicitly_allowed(tmp_path, capsys):
    a = make_run(str(tmp_path), "a", {"seed": 0}, revision="before")
    b = make_run(str(tmp_path), "b", {"seed": 0}, revision="after")
    check_revisions([a, b], allow_mixed=True)
    out = capsys.readouterr().out
    assert "WARNING" in out and "before" in out and "after" in out


def test_one_revision_is_not_reported(tmp_path, capsys):
    a = make_run(str(tmp_path), "a", {"seed": 0}, revision="same")
    b = make_run(str(tmp_path), "b", {"seed": 1}, revision="same")
    check_revisions([a, b])
    assert capsys.readouterr().out == ""
