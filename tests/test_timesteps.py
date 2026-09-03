"""The denoising timestep grid.

`get_timesteps` decides how many network evaluations a sampling run makes, and
`--num-steps` is a swept axis in the colorMNIST ablations, so the grid it
returns is a reported quantity and not only an implementation detail.

Every stride returns the same thing: `num_steps` strictly increasing ints in
the half-open interval [t_start, t_end).
"""

import pytest
import torch

from fff.ndtm import (DiffusionSchedule, DiffusionScheduleConfig,
                      TimestepConfig, get_timesteps)


@pytest.mark.parametrize("num_steps", [25, 50, 100, 200, 400, 500, 800, 1000])
def test_ddpm_uniform_returns_the_requested_number_of_steps(num_steps):
    """`--num-steps N` must run N steps.

    `range(t0, t1, (t1 - t0) // n_steps)` gives n_steps only when n_steps
    divides t1 - t0. It does not for 400 or 800 out of 1000, which are two of
    the five points on the steps axis of the ablation.
    """
    ts = get_timesteps(TimestepConfig(num_steps=num_steps))
    assert len(ts) == num_steps


def test_ddpm_uniform_is_increasing_and_within_bounds():
    ts = get_timesteps(TimestepConfig(t_start=0, t_end=1000, num_steps=200))
    assert ts[0] == 0
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert max(ts) < 1000


def test_ddpm_uniform_explains_itself_when_asked_for_too_many_steps():
    with pytest.raises(ValueError) as excinfo:
        get_timesteps(TimestepConfig(t_start=0, t_end=1000, num_steps=1001))
    assert "range()" not in str(excinfo.value)


def test_unknown_stride_is_rejected():
    with pytest.raises(NotImplementedError):
        get_timesteps(TimestepConfig(stride="cosine"))


@pytest.mark.parametrize("stride", ["ddpm_uniform", "uniform", "quadratic"])
def test_every_stride_is_increasing_and_half_open(stride):
    """Half-open matters: t_end is one past the last alpha in the schedule."""
    ts = get_timesteps(TimestepConfig(t_start=1, t_end=1000, num_steps=50,
                                      stride=stride))
    assert len(ts) == 50
    assert all(isinstance(t, int) for t in ts), "a float cannot index alphas"
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert ts[0] >= 1 and ts[-1] < 1000


def test_the_quadratic_stride_concentrates_its_steps_at_the_clean_end():
    ts = get_timesteps(TimestepConfig(t_start=0, t_end=1000, num_steps=20,
                                      stride="quadratic"))
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert gaps[-1] > gaps[0]


def test_a_stride_that_cannot_fit_the_steps_says_so():
    """rho = 7 crowds the low end, so 1000 distinct integers do not fit."""
    with pytest.raises(ValueError, match="quadratic"):
        get_timesteps(TimestepConfig(t_start=0, t_end=1000, num_steps=1000,
                                     stride="quadratic"))


@pytest.mark.parametrize("stride", ["ddpm_uniform", "uniform", "quadratic"])
def test_every_stride_is_indexable_as_a_timestep(stride):
    """Whatever the stride, the result has to index the alpha table.

    NDTM does `self.diffusion.alpha(t)` on these values. A stride that returns
    the closed interval [t_start, t_end] runs one past the end of the schedule.
    """
    schedule = DiffusionSchedule(DiffusionScheduleConfig(), device="cpu")
    ts = get_timesteps(TimestepConfig(num_steps=10, stride=stride))
    schedule.alpha(torch.as_tensor(ts))
