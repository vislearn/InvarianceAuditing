"""The forward diffusion process: betas, alpha-bars, and the Tweedie inverse.

Everything NDTM does sits on top of these three, and they are pure functions of
a config, so they can be checked exactly rather than by eyeballing samples.
"""

import numpy as np
import pytest
import torch

from fff.ndtm import DiffusionSchedule, DiffusionScheduleConfig
from fff.utils.diffusion import make_betas


def schedule(device="cpu", **kwargs):
    return DiffusionSchedule(DiffusionScheduleConfig(**kwargs), device=device)


# --------------------------------------------------------------------- alphas

def test_alpha_bar_starts_at_one_and_decreases():
    s = schedule()
    # t = -1 is the clean end: NDTM passes it as `s` on the last step, where the
    # update has to return x0 unchanged.
    assert s.alpha(torch.tensor(-1)).item() == pytest.approx(1.0)
    alphas = s.alphas
    assert (alphas.diff() < 0).all()
    assert alphas.min() > 0 and alphas.max() <= 1


def test_alpha_bar_is_the_cumulative_product_of_one_minus_beta():
    s = schedule(num_diffusion_timesteps=50)
    betas = torch.linspace(1e-4, 0.02, 50)
    expected = (1 - betas).cumprod(0)
    torch.testing.assert_close(s.alphas[1:], expected.float(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("beta_schedule", ["linear", "quad", "const", "sigmoid"])
def test_every_beta_schedule_is_a_valid_noising_process(beta_schedule):
    s = schedule(beta_schedule=beta_schedule, num_diffusion_timesteps=100)
    betas = s.betas[1:]
    assert (betas > 0).all(), "a non-positive beta is not a noising step"
    assert (s.alphas > 0).all(), "alpha_bar == 0 makes the Tweedie estimate inf"
    x0 = s.predict_x_from_eps(torch.randn(2, 4), torch.randn(2, 4),
                              torch.tensor([99, 99]))
    assert torch.isfinite(x0).all()


def test_unknown_beta_schedule_is_rejected():
    with pytest.raises(NotImplementedError):
        schedule(beta_schedule="cosine")


def test_a_schedule_that_destroys_the_signal_is_refused():
    """"jsd" ends at beta = 1, so alpha_bar(T) is exactly 0.

    The Tweedie estimate then divides by sqrt(0) and the whole trajectory comes
    back inf/NaN -- and NDTM reports a loss rather than an error, so the run
    looks like it worked.
    """
    with pytest.raises(ValueError, match="alpha_bar"):
        schedule(beta_schedule="jsd", num_diffusion_timesteps=100)


def test_given_betas_override_the_schedule():
    betas = torch.linspace(0.01, 0.5, 20, dtype=torch.float64)
    s = schedule(given_betas=betas)
    torch.testing.assert_close(s.betas[1:], betas.float())


# ------------------------------------------------------- noising and Tweedie

@pytest.mark.parametrize("shape", [(4, 2, 8, 8), (4, 16)])
def test_predict_x_from_eps_inverts_the_noising_exactly(shape):
    """The Tweedie estimate must recover x0 from (x_t, the eps that made it).

    Both shapes matter: the image experiments carry (B, C, H, W) and the HTRU2
    and latent experiments carry flat (B, D). The alpha reshape has to broadcast
    over either.
    """
    s = schedule()
    x0 = torch.randn(shape)
    t = torch.randint(0, 999, (shape[0],))
    eps = torch.randn(shape)
    alpha = s.alpha(t).view(-1, *([1] * (x0.ndim - 1)))
    xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * eps
    torch.testing.assert_close(s.predict_x_from_eps(xt, eps, t), x0,
                               rtol=1e-3, atol=1e-3)


def test_noise_image_matches_the_marginal_it_claims():
    """x_t | x0 must be N(sqrt(alpha_bar) x0, (1 - alpha_bar) I)."""
    s = schedule()
    x0 = torch.full((20000, 1), 3.0)
    t = torch.full((20000,), 500)
    xt = s.noise_image(x0, t)
    alpha = s.alpha(t[0]).item()
    assert xt.mean().item() == pytest.approx(np.sqrt(alpha) * 3.0, abs=0.05)
    assert xt.std().item() == pytest.approx(np.sqrt(1 - alpha), abs=0.05)


def test_noise_image_accepts_a_scalar_timestep():
    """The docstring offers "integer or tensor of shape (B,)"."""
    s = schedule()
    x0 = torch.randn(4, 3, 8, 8)
    assert s.noise_image(x0, 10).shape == x0.shape
    assert s.noise_image(x0, torch.tensor(10)).shape == x0.shape


def test_the_last_timestep_is_almost_pure_noise():
    """Otherwise x_T ~ N(0, I) is the wrong thing to start sampling from."""
    s = schedule()
    assert s.alpha(torch.tensor(999)).item() < 1e-4


# ------------------------------------------------------------------ device

def test_schedule_builds_on_cpu_when_asked(cpu_only):
    """The schedule has to build on a machine with no GPU."""
    s = schedule(device=None)
    assert s.alphas.device.type == "cpu"


def test_given_betas_do_not_pin_the_schedule_to_their_own_device():
    betas = torch.linspace(0.01, 0.5, 20, dtype=torch.float64)
    s = DiffusionSchedule(DiffusionScheduleConfig(given_betas=betas), device="cpu")
    assert s.betas.device.type == "cpu" and s.alphas.device.type == "cpu"


# -------------------------------------------------------------- make_betas

@pytest.mark.parametrize("alpha_transform_type", ["cosine", "linear"])
def test_make_betas_returns_the_number_of_betas_it_was_asked_for(alpha_transform_type):
    """The linear branch used to ignore this argument and always return 1000.

    fff.model.DiffusionModel passes hparams.num_timesteps straight through, so
    any model trained at a different number of steps got the wrong noise levels
    with nothing to say so. Every config in this repository trains at 1000.
    """
    for n in (250, 1000):
        assert len(make_betas(n, alpha_transform_type=alpha_transform_type)) == n


def test_make_betas_rejects_an_unknown_transform():
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        make_betas(100, alpha_transform_type="quadratic")


@pytest.mark.parametrize("alpha_transform_type", ["cosine", "linear"])
def test_make_betas_stays_within_max_beta(alpha_transform_type):
    betas = make_betas(1000, max_beta=0.02, alpha_transform_type=alpha_transform_type)
    assert (betas > 0).all() and (betas <= 0.02).all()
