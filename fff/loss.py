# MIT License
#
# Copyright (c) 2024 Computer Vision and Learning Lab, Heidelberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import functools
from collections import namedtuple
from functools import wraps, partial
from math import sqrt, prod
from typing import Callable, Union

import torch
from torch.distributions import Distribution
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual


Transform = Callable[[torch.Tensor], torch.Tensor]


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """Sum over all dimensions except the first.

    :param x: Input tensor. Shape: (batch_size, ...)
    :return: Sum over all dimensions except the first. Shape: (batch_size,)
    """
    return torch.sum(x.reshape(x.shape[0], -1), dim=1)


SurrogateOutput = namedtuple(
    "SurrogateOutput", ["surrogate", "z", "x1", "regularizations"]
)


def sample_v(x: torch.Tensor, hutchinson_samples: int) -> torch.Tensor:
    """
    Sample a random vector v of shape (*x.shape, hutchinson_samples)
    with scaled orthonormal columns.

    The reference data is used for shape, device and dtype.

    :param x: Reference data. Shape: (batch_size, ...).
    :param hutchinson_samples: Number of Hutchinson samples to draw.
    :return: Random vectors of shape (batch_size, ...)
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])

    if hutchinson_samples > total_dim:
        raise ValueError(
            f"Too many Hutchinson samples: got {hutchinson_samples}, \
                expected <= {total_dim}"
        )
    v = torch.randn(
        batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype
    )
    q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples)
    return q * sqrt(total_dim)


def volume_change_surrogate(
    x: torch.Tensor,
    encode: Transform,
    decode: Transform,
    hutchinson_samples: int = 1,
) -> SurrogateOutput:
    r"""Computes the surrogate for the volume change term in the change of
    variables formula. The surrogate is given by:
    $$
    v^T f_\theta'(x) \texttt{SG}(g_\phi'(z) v).
    $$
    The gradient of the surrogate is the gradient of the volume change term.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent
        representation `z` of shape (batch_size, latent_shape).
    :param decode: Decoder function. Takes a latent representation `z` as input
        and returns a reconstruction `x1`.
    :param hutchinson_samples: Number of Hutchinson samples to use for the
        volume change estimator. The number of hutchinson samples must be less
        than or equal to the total dimension of the data.
    :return: The computed surrogate of shape (batch_size,), latent representation
        `z`, reconstruction `x1` and regularization metrics computed on the fly.
    """
    regularizations = {}
    surrogate = 0

    x.requires_grad_()
    z = encode(x)

    vs = sample_v(z, hutchinson_samples)

    for k in range(hutchinson_samples):
        v = vs[..., k]

        # $ g'(z) v $ via forward-mode AD
        with dual_level():
            dual_z = make_dual(z, v)
            dual_x1 = decode(dual_z)
            x1, v1 = unpack_dual(dual_x1)

        # $ v^T f'(x) $ via backward-mode AD
        (v2,) = grad(z, x, v, create_graph=True)

        # $ v^T f'(x) stop_grad(g'(z)) v $
        surrogate += sum_except_batch(v2 * v1.detach()) / hutchinson_samples

    return SurrogateOutput(surrogate, z, x1, regularizations)


def reconstruction_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum((a - b) ** 2, dim=tuple(range(1, len(a.shape))))


def fff_loss(
    x: torch.Tensor,
    encode: Transform,
    decode: Transform,
    latent_distribution: Distribution,
    beta: Union[float, torch.Tensor],
    hutchinson_samples: int = 1,
) -> torch.Tensor:
    r"""Compute the per-sample FFF/FIF loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 - \log p_Z(z)
        - \sum_{k=1}^K v_k^T f'(x) stop_grad(g'(z)) v_k
    $$
    where $E[v_k^T v_k] = 1$, and $ f'(x) $ and $ g'(z) $ are the Jacobians of
    `encode` and `decode`.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent
        representation `z` of shape (batch_size, latent_shape).
    :param decode: Decoder function. Takes a latent representation `z` as input
        and returns a reconstruction `x1`.
    :param latent_distribution: Latent distribution of the model.
    :param beta: Weight of the mean squared error.
    :param hutchinson_samples: Number of Hutchinson samples to use for the
        volume change estimator.
    :return: Per-sample loss. Shape: (batch_size,)"""
    surrogate = volume_change_surrogate(x, encode, decode, hutchinson_samples)
    mse = reconstruction_loss(x, surrogate.x1)
    log_prob = latent_distribution.log_prob(surrogate.z)
    nll = -sum_except_batch(log_prob) - surrogate.surrogate
    return beta * mse + nll
