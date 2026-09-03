import torch
import math

def make_betas(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="linear",
) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
                num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default `linear`): the type of noise schedule
                     for alpha_bar. Choose from `linear` or `cosine`.

    Returns:
                betas: `num_diffusion_timesteps` betas for the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    if alpha_transform_type == "linear":
        # Honour num_diffusion_timesteps rather than hard-coding 1000: a model
        # asking for any other number of steps would otherwise get a 1000-step
        # schedule and silently use the wrong noise levels.
        return torch.linspace(1.e-4, max_beta, num_diffusion_timesteps,
                              dtype=torch.float32)

    # Raise rather than falling through and returning an empty tensor, where a
    # typo would build a model with no schedule at all and fail much later on an
    # index error.
    raise ValueError(
        f"unknown alpha_transform_type {alpha_transform_type!r}; expected "
        f"'linear' or 'cosine'")
