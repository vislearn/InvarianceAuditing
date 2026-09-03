"""NDTM sampling, end to end, on CPU.

Nothing here needs a checkpoint or a dataset. The generative model is the exact
denoiser for a standard normal, which has a closed form, so unguided sampling
has a distribution we can assert on; the subject model is a fixed linear map, so
its fibers are affine subspaces we can compute the distance to. That makes the
two claims NDTM makes -- that samples stay on the model's distribution, and that
they land on the subject model's fiber -- into ordinary assertions.
"""

import math

import pytest
import torch
import torch.nn as nn

from fff.ndtm import (Additive, DiffusionModel, DiffusionSchedule,
                      DiffusionScheduleConfig, NDTM, NDTMConfig,
                      TimestepConfig, get_gamma_t_fct, get_timesteps)

DIM = 8
FEATURES = 3


# The toy data distribution: anisotropic, off-centre, so that getting the
# sampler right means getting both the mean shift and the per-coordinate scale
# right. An isotropic standard normal would hide either mistake -- for N(0, I)
# the deterministic DDIM map is the identity, and every coefficient error
# cancels.
MU = torch.linspace(-1.0, 1.0, DIM)
SIGMA = torch.linspace(0.5, 2.0, DIM)


class ExactGaussianDenoiser(nn.Module):
    """E[eps | x_t] for data drawn from N(MU, diag(SIGMA^2)).

    With x_t = sqrt(a) x_0 + sqrt(1-a) eps, the posterior mean has a closed
    form, and so does the deterministic DDIM map it induces: it takes x_T = z to
    x_0 = MU + SIGMA * z. That gives an exact answer to compare a sampled
    trajectory against, rather than a distributional one.
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def forward(self, x, t, y=None):
        alpha = self.schedule.alpha(t).view(-1, *([1] * (x.ndim - 1)))
        var = alpha * SIGMA ** 2 + (1 - alpha)
        return (x - alpha.sqrt() * MU) * (1 - alpha).sqrt() / var


class LinearSubjectModel(nn.Module):
    """phi(x) = A x, so the fiber through a target is an affine subspace."""

    def __init__(self, dim=DIM, features=FEATURES, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "A", torch.randn(features, dim, generator=generator) / math.sqrt(dim))

    def forward(self, x):
        return x.reshape(x.shape[0], -1) @ self.A.T


@pytest.fixture
def schedule():
    return DiffusionSchedule(DiffusionScheduleConfig(), device="cpu")


@pytest.fixture
def generative_model(schedule):
    return DiffusionModel(ExactGaussianDenoiser(schedule), schedule)


@pytest.fixture
def subject_model():
    return LinearSubjectModel()


def config(**kwargs):
    defaults = dict(N=2, gamma_t=0.0, u_lr=0.05, w_terminal=50.0, eta=0.0,
                    w_score_scheme="zero", w_control_scheme="zero",
                    clip_images=False, u_lr_scheduler="linear")
    defaults.update(kwargs)
    return NDTMConfig(**defaults)


def run(generative_model, subject_model, cfg, batch=16, num_steps=50, seed=0):
    ndtm = NDTM(generative_model, subject_model, cfg)
    ts = get_timesteps(TimestepConfig(num_steps=num_steps))
    x = torch.zeros(batch, DIM)
    # Seed before the queries as well as before the run: two calls in one test
    # must audit the same fibers, or the comparison between them means nothing.
    torch.manual_seed(seed)
    target = subject_model(MU + SIGMA * torch.randn(batch, DIM))
    torch.manual_seed(seed + 1000)  # a different draw, so x_T is not the query
    xt_s, x0_s = ndtm.sample(x, None, ts, y_0=target)
    return xt_s, x0_s, target


# ------------------------------------------------------- shape and bookkeeping

def test_sample_returns_both_trajectories_final_state_first(generative_model,
                                                            subject_model):
    """The samplers all read `x0_traj[0]` as the answer.

    Both lists come back reversed, so index 0 is the *last* step. `xt_s` also
    carries the initial x_T, which `x0_s` has no counterpart for, hence the
    off-by-one in their lengths.
    """
    num_steps = 20
    xt_s, x0_s, _ = run(generative_model, subject_model, config(),
                        num_steps=num_steps)
    assert len(xt_s) == num_steps + 1
    assert len(x0_s) == num_steps
    assert all(x.shape == (16, DIM) for x in xt_s)
    # reversed: the last element of xt_s is the noise the run started from
    assert xt_s[-1].std().item() == pytest.approx(1.0, abs=0.3)
    torch.testing.assert_close(xt_s[0], x0_s[0], rtol=2e-2, atol=2e-2)


def test_sample_does_not_mutate_its_input(generative_model, subject_model):
    ndtm = NDTM(generative_model, subject_model, config())
    x = torch.zeros(4, DIM)
    target = torch.zeros(4, FEATURES)
    before = x.clone()
    ndtm.sample(x, None, get_timesteps(TimestepConfig(num_steps=10)), y_0=target)
    torch.testing.assert_close(x, before)


def test_sampling_is_reproducible_from_a_seed(generative_model, subject_model):
    """Every ablation reports a spread over --seed; the seed has to fix a run."""
    a = run(generative_model, subject_model, config(gamma_t=2.0), num_steps=10, seed=7)
    b = run(generative_model, subject_model, config(gamma_t=2.0), num_steps=10, seed=7)
    torch.testing.assert_close(a[1][0], b[1][0])


def test_samples_are_finite(generative_model, subject_model):
    _, x0_s, _ = run(generative_model, subject_model, config(gamma_t=5.0),
                     num_steps=20)
    assert torch.isfinite(x0_s[0]).all()


# ------------------------------------------------------------ unguided sampling

def test_unguided_sampling_reproduces_the_model_distribution(generative_model,
                                                             subject_model):
    """gamma = 0 is plain DDIM, and this model's samples are N(MU, SIGMA^2).

    This is the assertion that would catch a swapped alpha_t/alpha_s or a
    dropped square root in the update: both would still give a plausible-looking
    cloud, with the wrong mean or the wrong spread.
    """
    _, x0_s, _ = run(generative_model, subject_model, config(gamma_t=0.0),
                     batch=2048, num_steps=100)
    samples = x0_s[0]
    torch.testing.assert_close(samples.mean(0), MU, rtol=0, atol=0.15)
    torch.testing.assert_close(samples.std(0), SIGMA, rtol=0.1, atol=0.05)


def test_unguided_sampling_matches_the_closed_form_ddim_map(generative_model,
                                                            subject_model):
    """For this model the deterministic DDIM map is x_T -> MU + SIGMA * x_T."""
    xt_s, x0_s, _ = run(generative_model, subject_model,
                        config(gamma_t=0.0, eta=0.0), batch=8, num_steps=500)
    x_T = xt_s[-1]
    torch.testing.assert_close(x0_s[0], MU + SIGMA * x_T, rtol=0.02, atol=0.02)


def test_unguided_update_matches_a_written_out_ddim_step(generative_model,
                                                         subject_model, schedule):
    """The DDIM update, spelled out, against the one NDTM applies."""
    cfg = config(gamma_t=0.0, eta=0.0)
    ts = get_timesteps(TimestepConfig(num_steps=25))
    ndtm = NDTM(generative_model, subject_model, cfg)

    torch.manual_seed(3)
    _, x0_s = ndtm.sample(torch.zeros(4, DIM), None, ts,
                          y_0=torch.zeros(4, FEATURES))

    torch.manual_seed(3)
    xt = torch.randn(4, DIM)  # what `initialize` draws for init_xT="random"
    ss = [-1] + list(ts[:-1])
    for ti, si in zip(reversed(ts), reversed(ss)):
        t = torch.full((4,), ti, dtype=torch.long)
        s = torch.full((4,), si, dtype=torch.long)
        alpha_t = schedule.alpha(t).view(-1, 1)
        alpha_s = schedule.alpha(s).view(-1, 1)
        et = generative_model(xt, None, t)
        x0 = (xt - (1 - alpha_t).sqrt() * et) / alpha_t.sqrt()
        xt = alpha_s.sqrt() * x0 + (1 - alpha_s).sqrt() * et
    torch.testing.assert_close(x0_s[0], x0, rtol=1e-4, atol=1e-4)


# -------------------------------------------------------------------- guidance

def fiber_loss(subject_model, samples, target):
    return ((subject_model(samples) - target) ** 2).sum(-1)


def test_guidance_moves_samples_onto_the_fiber(generative_model, subject_model):
    """The whole point: guided samples must satisfy phi(x) = phi(x_query)."""
    unguided = run(generative_model, subject_model, config(gamma_t=0.0),
                   num_steps=50)
    guided = run(generative_model, subject_model, config(gamma_t=5.0, N=4),
                 num_steps=50)
    loss_unguided = fiber_loss(subject_model, unguided[1][0], unguided[2])
    loss_guided = fiber_loss(subject_model, guided[1][0], guided[2])
    assert loss_guided.mean() < loss_unguided.mean() / 10, (
        f"guidance barely helped: {loss_guided.mean():.4f} vs "
        f"{loss_unguided.mean():.4f}")
    # The mean is set by the few queries the control does not reach in the
    # budget; the typical one should land on the fiber outright.
    assert loss_guided.median() < 1e-3


def test_more_inner_steps_do_not_make_the_fiber_loss_worse(generative_model,
                                                           subject_model):
    """N is the number of control-optimisation steps per denoising step."""
    losses = []
    for n in (1, 4):
        _, x0_s, target = run(generative_model, subject_model,
                              config(gamma_t=5.0, N=n), num_steps=50)
        losses.append(fiber_loss(subject_model, x0_s[0], target).mean().item())
    assert losses[1] <= losses[0] * 1.2


def test_zero_gamma_leaves_the_control_at_zero(generative_model, subject_model):
    """gamma = 0 must be exactly the unguided sampler, whatever N and w_terminal.

    The ablations lean on this: the gamma = 0 column is the "no guidance"
    reference, and if the control leaked through it would not be one.
    """
    a = run(generative_model, subject_model,
            config(gamma_t=0.0, N=0), num_steps=20, seed=11)
    b = run(generative_model, subject_model,
            config(gamma_t=0.0, N=8, w_terminal=1e6), num_steps=20, seed=11)
    torch.testing.assert_close(a[1][0], b[1][0])


def test_ancestral_sampling_runs_and_guides(generative_model, subject_model):
    """The DDPM-style update, the other branch of the timestep loop.

    It reads a beta schedule rescaled to the number of sampling steps, which is
    built once before the loop and is None for DDIM sampling -- so this is the
    only test that touches those tensors at all.
    """
    cfg = config(gamma_t=5.0, N=4, eta=1.0, ancestral_sampling=True)
    _, x0_s, target = run(generative_model, subject_model, cfg, num_steps=50)
    assert torch.isfinite(x0_s[0]).all()
    assert fiber_loss(subject_model, x0_s[0], target).median() < 1e-2


def test_ancestral_sampling_needs_a_linear_schedule_and_says_which(subject_model):
    """It rescales a linear schedule, so it can only have one to rescale."""
    schedule = DiffusionSchedule(DiffusionScheduleConfig(beta_schedule="quad"),
                                 device="cpu")
    model = DiffusionModel(ExactGaussianDenoiser(schedule), schedule)
    with pytest.raises(ValueError, match="ancestral"):
        run(model, subject_model, config(gamma_t=1.0, ancestral_sampling=True),
            num_steps=10)


def test_ddim_sampling_does_not_require_a_linear_beta_schedule(subject_model):
    """That requirement belongs to the ancestral update alone.

    It used to be asserted for every run, with a message about ancestral
    sampling, so DDIM sampling from a model trained on any other schedule was
    refused for a reason that did not apply to it.
    """
    schedule = DiffusionSchedule(DiffusionScheduleConfig(beta_schedule="quad"),
                                 device="cpu")
    model = DiffusionModel(ExactGaussianDenoiser(schedule), schedule)
    run(model, subject_model, config(gamma_t=0.0, ancestral_sampling=False),
        num_steps=10)


def test_clip_range_is_applied_to_the_returned_estimate(generative_model,
                                                        subject_model):
    cfg = config(gamma_t=1.0, clip_images=True, clip_range=[-0.5, 0.5])
    _, x0_s, _ = run(generative_model, subject_model, cfg, num_steps=20)
    assert x0_s[0].abs().max().item() <= 0.5 + 1e-6


@pytest.mark.parametrize("fiber_loss_name", ["l2", "l1"])
def test_each_terminal_cost_runs_and_guides(generative_model, subject_model,
                                            fiber_loss_name):
    cfg = config(gamma_t=5.0, N=4, fiber_loss=fiber_loss_name)
    _, x0_s, target = run(generative_model, subject_model, cfg, num_steps=50)
    assert fiber_loss(subject_model, x0_s[0], target).median() < 1e-3


# ------------------------------------------------------------- configuration

def test_unknown_combine_fn_is_rejected_at_construction(generative_model,
                                                        subject_model):
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        NDTM(generative_model, subject_model, config(combine_fn="multiplicative"))


def test_unknown_init_control_is_rejected(generative_model, subject_model):
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        run(generative_model, subject_model,
            config(gamma_t=1.0, init_control="warm_start"), num_steps=5)


def test_unknown_variance_type_is_rejected(generative_model, subject_model):
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        run(generative_model, subject_model,
            config(gamma_t=1.0, variance_type="learned"), num_steps=5)


def test_an_integer_weight_is_accepted_like_a_float(generative_model,
                                                    subject_model):
    run(generative_model, subject_model,
        config(gamma_t=1.0, w_control_scheme=1, w_score_scheme=1), num_steps=5)


@pytest.mark.parametrize("scheme", ["zero", "ones", "ddpm", "ddim", 1e-4])
def test_named_weight_schemes_are_non_negative(generative_model, subject_model,
                                               scheme):
    """A negative weight would reward the thing the cost is meant to penalise."""
    ndtm = NDTM(generative_model, subject_model, config())
    t = torch.tensor([500])
    s = torch.tensor([495])
    assert float(ndtm._get_score_weight(scheme, t, s)) >= 0
    assert float(ndtm._get_control_weight(scheme, t, s)) >= 0


# ------------------------------------------------------------ gamma schedules

def test_gamma_schedule_is_zero_above_the_first_anchor_and_gamma_below():
    """The const schedule the colorMNIST and HTRU2 samplers build."""
    gamma_t = get_gamma_t_fct([(0, 0, 1000, 500), (5.0, 5.0, 500, 0)])
    assert float(gamma_t(torch.tensor(999))) == pytest.approx(0.0, abs=1e-6)
    assert float(gamma_t(torch.tensor(750))) == pytest.approx(0.0, abs=1e-6)
    assert float(gamma_t(torch.tensor(250))) == pytest.approx(5.0)
    assert float(gamma_t(torch.tensor(0))) == pytest.approx(5.0)


def test_gamma_schedule_ramps_monotonically():
    gamma_t = get_gamma_t_fct(
        [(0, 0, 1000, 500), (0, 5.0, 500, 300), (5.0, 5.0, 300, 0)])
    values = [float(gamma_t(torch.tensor(t))) for t in range(500, 299, -20)]
    assert all(b >= a - 1e-6 for a, b in zip(values, values[1:]))
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    assert values[-1] == pytest.approx(5.0, abs=1e-6)


def test_gamma_schedule_rejects_a_timestep_it_does_not_cover():
    gamma_t = get_gamma_t_fct([(0, 0, 1000, 500)])
    with pytest.raises(ValueError):
        gamma_t(torch.tensor(100))


def test_gamma_schedule_accepts_a_plain_int_timestep():
    gamma_t = get_gamma_t_fct([(0, 0, 1000, 500), (5.0, 5.0, 500, 0)])
    assert float(gamma_t(250)) == pytest.approx(5.0)


def test_additive_combination_scales_the_control_by_gamma():
    xt = torch.zeros(2, DIM)
    ut = torch.ones(2, DIM)
    torch.testing.assert_close(Additive(gamma_t=3.0)(xt, ut), 3.0 * ut)
    torch.testing.assert_close(Additive(gamma_t=None)(xt, ut), ut)


# --------------------------------------------------- the DiffusionModel wrapper

def test_diffusion_model_splits_variance_channels_off_an_image_output():
    class SixChannel(nn.Module):
        def forward(self, x, t, y=None):
            return torch.cat([torch.ones_like(x), torch.zeros_like(x)], dim=1)

    model = DiffusionModel(SixChannel(), None)
    x = torch.randn(2, 3, 8, 8)
    et, logvar = model(x, None, torch.zeros(2).long(), predict_variance=True)
    assert et.shape == (2, 3, 8, 8) and logvar.shape == (2, 3, 8, 8)
    assert model(x, None, torch.zeros(2).long()).shape == (2, 3, 8, 8)


def test_diffusion_model_keeps_every_channel_of_a_four_channel_latent():
    class FourChannel(nn.Module):
        def forward(self, x, t, y=None):
            return torch.zeros_like(x)

    model = DiffusionModel(FourChannel(), None)
    x = torch.randn(2, 4, 8, 8)
    assert model(x, None, torch.zeros(2).long()).shape == x.shape


def test_diffusion_model_passes_the_class_label_only_when_asked():
    seen = []

    class Recorder(nn.Module):
        def forward(self, x, t, y=None):
            seen.append(y)
            return torch.zeros_like(x)

    x = torch.randn(2, 4)
    t = torch.zeros(2).long()
    DiffusionModel(Recorder(), None, class_cond_diffusion_model=False)(x, "label", t)
    DiffusionModel(Recorder(), None, class_cond_diffusion_model=True)(x, "label", t)
    assert seen == [None, "label"]


def test_cos_learning_rate_is_the_reverse_of_linear():
    """`cos` ramps the control learning rate UP as denoising proceeds.

    The two schedulers are mirror images, not variations: `current_step` is the
    loop index, which starts at the noisiest timestep, so `linear` spends the
    rate early and `cos` spends it late. A test that only checked both end at
    the same place would pass on a sign error, so check both ends of both.
    """
    from fff.ndtm import NDTM, NDTMConfig

    def rate(scheme, step, total=200):
        ndtm = NDTM.__new__(NDTM)
        ndtm.hparams = NDTMConfig(u_lr_scheduler=scheme)
        return ndtm.get_learning_rate(0.002, step, total)

    assert rate("linear", 0) == pytest.approx(0.002)
    assert rate("linear", 200) == pytest.approx(0.0)
    assert rate("cos", 0) == pytest.approx(0.0, abs=1e-12)
    assert rate("cos", 200) == pytest.approx(0.002)
    assert rate("cos", 100) == pytest.approx(0.001)
    assert rate("const", 0) == rate("const", 200) == pytest.approx(0.002)
