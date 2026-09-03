"""The reported quantities: fiber loss, and the colorMNIST colour statistics.

Three different things are called "the fiber loss" in this repository -- the
terminal cost NDTM minimises, the squared distance Tables 5-7 report, and the
per-dimension RMS the colorMNIST and HTRU2 scripts report. They differ by a
square and by a division by the feature dimension, which is exactly the kind of
difference that survives a code review and shows up as a number an order of
magnitude off. These tests pin all three down against each other.
"""


import numpy as np
import pytest
import torch

from experiments.imagenet.evaluate import expected_fibers, fiber_loss, load_run


# ------------------------------------------------------------------ definitions

def test_l2_fiber_loss_is_the_squared_distance_not_the_norm():
    """Table 5's 1225 for DINOv2 is only possible for the squared distance.

    The features have norm about 48, so a plain norm could not exceed ~96.
    """
    target = torch.zeros(1, 4)
    samples = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
    assert fiber_loss(target, samples, "l2").item() == pytest.approx(25.0)


def test_l1_fiber_loss_is_the_absolute_sum():
    target = torch.zeros(1, 4)
    samples = torch.tensor([[3.0, -4.0, 0.0, 0.0]])
    assert fiber_loss(target, samples, "l1").item() == pytest.approx(7.0)


def test_the_reported_l2_is_the_square_of_ndtms_terminal_cost():
    """`NDTMConfig.fiber_loss="l2"` is the norm; `evaluate.py` reports its square.

    The comment in NDTM.sample turns on this: a terminal loss of ~16 printed
    during sampling is a reported fiber loss of ~270.
    """
    target = torch.randn(8, 16)
    samples = torch.randn(8, 16)
    terminal = torch.norm(samples - target, p=2, dim=-1)  # what NDTM minimises
    torch.testing.assert_close(fiber_loss(target, samples, "l2"), terminal ** 2)


def test_the_colormnist_metric_is_the_reported_l2_per_dimension():
    """colorMNIST and HTRU2 report sqrt(sum d^2 / dim), a third scale again."""
    target = torch.randn(8, 16)
    samples = torch.randn(8, 16)
    reported = torch.sqrt(((samples - target) ** 2).sum(-1) / target.shape[-1])
    torch.testing.assert_close(reported,
                               (fiber_loss(target, samples, "l2") / 16).sqrt())


def test_cross_entropy_matches_the_one_ndtm_optimises():
    import torch.nn.functional as F

    target = torch.randn(8, 10)
    samples = torch.randn(8, 10)
    torch.testing.assert_close(
        fiber_loss(target, samples, "cross_entropy"),
        F.cross_entropy(samples, target.softmax(-1), reduction="none"))


def test_cross_entropy_has_a_non_zero_floor():
    """A perfect fiber sample does not score 0 on the cross-entropy column.

    Table 7's cross-entropy numbers are bounded below by the entropy of the
    target distribution, so they are comparable across rows but not against 0.
    """
    target = torch.randn(4, 10)
    assert (fiber_loss(target, target, "cross_entropy") > 0).all()
    assert fiber_loss(target, target, "l2").sum() == 0
    assert fiber_loss(target, target, "l1").sum() == 0


def test_an_unknown_metric_is_refused():
    with pytest.raises(ValueError):
        fiber_loss(torch.zeros(1, 2), torch.zeros(1, 2), "cosine")


@pytest.mark.parametrize("metric", ["l2", "l1", "cross_entropy"])
def test_every_metric_returns_one_number_per_fiber(metric):
    assert fiber_loss(torch.randn(7, 5), torch.randn(7, 5), metric).shape == (7,)


# --------------------------------------------------------------- loading runs

def write_chunks(directory, counts):
    directory.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(counts):
        torch.save({"original_embeddings": torch.zeros(n, 4),
                    "invariances_embeddings": torch.ones(n, 4),
                    "images": torch.zeros(n, 3, 8, 8)},
                   directory / f"chunk_{i}.pt")


def test_chunks_are_concatenated_in_numeric_order(tmp_path):
    """chunk_10 sorts before chunk_2 as a string.

    A run of more than ten chunks would be reassembled out of order, which
    silently misaligns originals against samples when two runs are compared
    row by row.
    """
    directory = tmp_path / "run"
    directory.mkdir()
    for i in range(12):
        torch.save({"original_embeddings": torch.full((1, 2), float(i)),
                    "invariances_embeddings": torch.zeros(1, 2)},
                   directory / f"chunk_{i}.pt")
    loaded = load_run(str(directory))["original_embeddings"][:, 0]
    torch.testing.assert_close(loaded, torch.arange(12, dtype=torch.float32))


def test_loading_drops_the_images_it_does_not_need(tmp_path):
    directory = tmp_path / "run"
    write_chunks(directory, [3, 2])
    data = load_run(str(directory))
    assert "images" not in data
    assert len(data["invariances_embeddings"]) == 5


def test_loading_a_run_with_no_chunks_says_so(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_run(str(tmp_path / "empty"))


def test_expected_fibers_multiplies_images_by_samples_each(tmp_path):
    from experiments.common.sampling import save_config

    directory = tmp_path / "run"
    directory.mkdir()
    save_config(str(directory), {"args": {"num_images": 32, "samples_per_image": 4}})
    assert expected_fibers(str(directory)) == 128

    save_config(str(directory), {"args": {"num_images": 32}})
    assert expected_fibers(str(directory)) == 32


def test_expected_fibers_is_none_for_a_run_that_recorded_nothing(tmp_path):
    (tmp_path / "run").mkdir()
    assert expected_fibers(str(tmp_path / "run")) is None


# ------------------------------------------------- the colorMNIST colour metric

def sample_reference_colors(n, seed=0):
    """Draw from the mixture `gaussian_mix_dense` is the density of."""
    rng = np.random.default_rng(seed)
    which = rng.choice(3, size=n, p=[0.6, 0.35, 0.05])
    mu = np.array([0.7, 0.5, 0.1])[which]
    sigma = np.array([0.08, 0.015, 0.02])[which]
    return np.clip(rng.normal(mu, sigma), 0.0, 1.0)


def images_with_colors(colors):
    """A batch of colorMNIST-shaped images whose background is `colors`."""
    n = colors.shape[0]
    digit = torch.zeros(n, 1, 28, 28)
    digit[:, :, 8:20, 8:20] = 1.0  # any foreground; the metric reads the border
    c = torch.from_numpy(colors).float().reshape(n, 3, 1, 1)
    return (1 - digit) * c + digit * ((c + 0.5) % 1)


def test_colour_kl_is_near_zero_for_samples_from_the_reference_mixture():
    """The KL column measures the colour marginal against the training mixture."""
    from experiments.colormnist.ablations.analyze_kappa_tau_steps import kl_and_dev

    colors = np.stack([sample_reference_colors(20000, seed=s) for s in range(3)], 1)
    kl, dev = kl_and_dev(images_with_colors(colors))
    assert kl < 0.05, f"a faithful colour marginal scored KL = {kl}"
    assert dev < 1e-4, f"a clean digit scored deviation = {dev}"


def test_colour_kl_grows_when_the_colours_are_wrong():
    from experiments.colormnist.ablations.analyze_kappa_tau_steps import kl_and_dev

    faithful = np.stack([sample_reference_colors(20000, seed=s) for s in range(3)], 1)
    wrong = np.random.default_rng(0).uniform(size=(20000, 3))
    assert kl_and_dev(images_with_colors(wrong))[0] > \
        10 * kl_and_dev(images_with_colors(faithful))[0]


def test_the_two_colour_metric_implementations_agree():
    """compute_statistics.py and the ablation analyzer each carry a copy."""
    from experiments.colormnist.ablations.analyze_kappa_tau_steps import kl_and_dev
    from experiments.colormnist.compute_statistics import \
        compute_kl_w1_and_deviation

    colors = np.stack([sample_reference_colors(5000, seed=s) for s in range(3)], 1)
    images = images_with_colors(colors)
    kl, dev = kl_and_dev(images)
    kl_means, _, (dev_mean, _) = compute_kl_w1_and_deviation(images[None])
    assert np.mean([m[0] for m in kl_means]) == pytest.approx(kl, rel=1e-5)
    assert dev_mean == pytest.approx(dev, rel=1e-5)
