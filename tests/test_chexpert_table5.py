"""The CheXpert rows of Table 5.

Two things here decide a number rather than compute one: the metric is not the
same for the classifier rows as for the Qwen row, and the legacy combined file
holds two sample slots from a run with faulty settings that have to be dropped.
Both are made in code from data shapes, so both can be tested.
"""

import os

import pytest
import torch

from experiments.chexpert.table5_fiber_losses import (BAD_SLOTS, default_metric,
                                                      from_combined, key,
                                                      probability_distance,
                                                      split_repeats,
                                                      squared_l2,
                                                      subject_models)


# ------------------------------------------------------------------ metrics

def test_probability_distance_is_zero_for_an_exact_match():
    logits = torch.randn(5, 5)
    assert probability_distance(logits, logits).abs().max() < 1e-5


def test_probability_distance_is_a_percentage_bounded_by_two_hundred():
    """It is the l1 distance between two probability vectors, times 100."""
    far = probability_distance(torch.tensor([[50.0, -50, -50, -50, -50]]),
                               torch.tensor([[-50.0, 50, -50, -50, -50]]))
    assert far.item() == pytest.approx(200.0, abs=1e-2)
    assert (probability_distance(torch.randn(20, 5), torch.randn(20, 5)) <= 200).all()


def test_probability_distance_ignores_a_shared_logit_offset():
    """A softmax is shift-invariant, so the metric must be too."""
    a, b = torch.randn(4, 5), torch.randn(4, 5)
    torch.testing.assert_close(probability_distance(a, b),
                               probability_distance(a + 3.0, b + 3.0))


def test_squared_l2_is_the_sum_not_the_mean():
    target = torch.zeros(1, 4)
    samples = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    assert squared_l2(target, samples).item() == pytest.approx(4.0)


def test_the_metric_follows_the_width_of_the_representation():
    """Five logits get the probability metric; a pooled embedding gets l2.

    A softmax over several hundred embedding dimensions is meaningless but
    would still print a plausible number, so this must not be guessed wrong.
    """
    assert default_metric({"original_bio_embeddings": torch.zeros(2, 5)},
                          "bio") == "probability"
    assert default_metric({"original_embeddings": torch.zeros(2, 1536)},
                          "") == "l2"


# -------------------------------------------------------------- run key names

def test_subject_models_are_read_off_the_keys_in_a_stable_order():
    keys = ["original_convnext_embeddings", "original_biomedclip_embeddings",
            "invariances_convnext_embeddings"]
    assert subject_models(keys) == ["biomedclip", "convnext"]


def test_an_unnamed_pair_counts_as_one_model():
    assert subject_models(["original_embeddings", "invariances_embeddings"]) == [""]


def test_no_embedding_keys_at_all_is_reported_as_none():
    assert subject_models(["images", "config"]) == []


def test_key_spells_both_the_named_and_unnamed_form():
    assert key("original", "convnext") == "original_convnext_embeddings"
    assert key("original", "") == "original_embeddings"


# ------------------------------------------------------------------- repeats

def make_config(directory, args):
    from experiments.common.sampling import save_config

    os.makedirs(directory, exist_ok=True)
    save_config(directory, {"args": args})


def test_a_run_with_repeats_splits_into_that_many_draws(tmp_path):
    """A genuine repeat: the same four queries submitted three times, so the
    targets recur and only the samples differ."""
    directory = str(tmp_path / "run")
    make_config(directory, {"repeats": 3})
    queries = torch.arange(4).reshape(4, 1).float()
    run = {"original_embeddings": queries.repeat(3, 1),
           "invariances_embeddings": torch.arange(12).reshape(12, 1).float()}
    draws = split_repeats(directory, run, [""])
    assert len(draws) == 3
    assert [len(d["original_embeddings"]) for d in draws] == [4, 4, 4]
    assert draws[0]["invariances_embeddings"][0].item() == 0
    assert draws[1]["invariances_embeddings"][0].item() == 4


def test_passes_over_different_images_are_one_draw_not_several(tmp_path):
    """`sample_qwen.py --queries chexpert` draws --num-images distinct images
    and never reads --repeats, but records its default of 5 anyway. Splitting on
    the recorded value alone would measure four fifths of the row as the
    distance between unrelated radiographs."""
    directory = str(tmp_path / "run")
    make_config(directory, {"repeats": 5, "queries": "chexpert", "num_images": 60})
    torch.manual_seed(0)
    run = {"original_embeddings": torch.randn(60, 8),
           "invariances_embeddings": torch.randn(60, 8)}
    draws = split_repeats(directory, run, [""])
    assert len(draws) == 1
    assert len(draws[0]["original_embeddings"]) == 60


def test_a_repeat_is_recognised_through_forward_pass_jitter(tmp_path):
    """The passes are separate forward calls, so the targets are equal only to
    within what the subject model does twice -- but different images are not
    close by any tolerance, so the check must not demand exact equality."""
    directory = str(tmp_path / "run")
    make_config(directory, {"repeats": 2})
    queries = torch.randn(4, 8)
    run = {"original_embeddings": torch.cat([queries, queries + 1e-6]),
           "invariances_embeddings": torch.randn(8, 8)}
    assert len(split_repeats(directory, run, [""])) == 2


def test_every_subject_model_has_to_agree_before_a_run_is_split(tmp_path):
    """A pair run where one model's targets repeat and the other's do not is
    not a repeat; splitting it would misalign the second model."""
    directory = str(tmp_path / "run")
    make_config(directory, {"repeats": 2})
    same = torch.randn(3, 5)
    run = {"original_bio_embeddings": same.repeat(2, 1),
           "invariances_bio_embeddings": torch.randn(6, 5),
           "original_conv_embeddings": torch.randn(6, 5),
           "invariances_conv_embeddings": torch.randn(6, 5)}
    assert len(split_repeats(directory, run, ["bio", "conv"])) == 1


def test_a_run_without_repeats_is_one_draw(tmp_path):
    directory = str(tmp_path / "run")
    make_config(directory, {"repeats": 1})
    run = {"original_embeddings": torch.zeros(10, 1)}
    assert len(split_repeats(directory, run, [""])) == 1


def test_a_repeat_count_that_does_not_divide_the_rows_is_left_alone(tmp_path):
    """Better one pooled set than draws silently sliced at the wrong boundary."""
    directory = str(tmp_path / "run")
    make_config(directory, {"repeats": 4})
    run = {"original_embeddings": torch.zeros(10, 1)}
    assert len(split_repeats(directory, run, [""])) == 1


# ------------------------------------------------------------- faulty slots

def write_combined(path, n_slots=18, faulty=BAD_SLOTS):
    """The legacy file: one tensor per key, sample sets on a slot axis."""
    original = torch.zeros(6, 5)
    samples = torch.zeros(6, n_slots, 5)
    # a faulty slot sits near 35%; every good one is between 1.1 and 1.6%
    samples[:, :, 0] = 0.02
    for slot in faulty:
        samples[:, slot, 0] = 1.5
    torch.save({"original_biomedclip_embeddings": original,
                "invariances_biomedclip_embeddings": samples}, path)


def test_the_two_faulty_slots_are_excluded_by_default(tmp_path):
    path = str(tmp_path / "combined.pt")
    write_combined(path)
    table, _ = from_combined(path)
    _, _, sets = table["biomedclip"]
    assert len(sets) == 18 - len(BAD_SLOTS)


def test_excluding_the_faulty_slots_changes_the_number(tmp_path):
    """If it did not, the exclusion would be cosmetic and the fix meaningless."""
    path = str(tmp_path / "combined.pt")
    write_combined(path)
    data = torch.load(path, weights_only=False)
    original = data["original_biomedclip_embeddings"]
    all_slots = data["invariances_biomedclip_embeddings"]

    def mean(sets):
        return torch.stack([probability_distance(original, s).mean()
                            for s in sets]).mean()

    kept = from_combined(path)[0]["biomedclip"][2]
    every = [all_slots[:, i] for i in range(all_slots.shape[1])]
    assert mean(kept) < mean(every) / 2


def test_the_combined_file_reader_uses_the_probability_metric(tmp_path):
    """Every row in that file is a classifier, so the metric is not in doubt."""
    path = str(tmp_path / "combined.pt")
    write_combined(path)
    assert from_combined(path)[0]["biomedclip"][0] == "probability"
