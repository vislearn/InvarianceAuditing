"""The estimators the fiber models are trained with.

`volume_change_surrogate` is a Hutchinson estimator whose expectation is the
log-determinant term of the change-of-variables formula. With as many Hutchinson
samples as there are dimensions the estimator stops being stochastic and equals
the trace exactly, so it can be checked against a closed form rather than by
averaging and hoping.
"""

import pytest
import torch

from fff.loss import (fff_loss, reconstruction_loss, sample_v, sum_except_batch,
                      volume_change_surrogate)
from fff.model.utils import guess_image_shape
from fff.utils.func import compute_jacobian, compute_volume_change
from fff.utils.utils import batch_wrap

DIM = 4


def linear_pair(seed=0):
    """An encoder and its exact inverse, both linear."""
    generator = torch.Generator().manual_seed(seed)
    A = torch.eye(DIM) + 0.1 * torch.randn(DIM, DIM, generator=generator)
    B = torch.linalg.inv(A)
    return (lambda x: x @ A.T), (lambda z: z @ B.T), A, B


# ---------------------------------------------------------------- sample_v

def test_hutchinson_vectors_are_orthogonal_and_scaled():
    x = torch.zeros(3, DIM)
    v = sample_v(x, hutchinson_samples=DIM)
    gram = torch.einsum("bik,bil->bkl", v, v)
    torch.testing.assert_close(gram, DIM * torch.eye(DIM).expand(3, -1, -1),
                               rtol=1e-4, atol=1e-4)


def test_too_many_hutchinson_samples_is_refused():
    with pytest.raises(ValueError):
        sample_v(torch.zeros(2, DIM), hutchinson_samples=DIM + 1)


def test_hutchinson_vectors_follow_the_shape_of_their_reference():
    v = sample_v(torch.zeros(2, 3, 4, 4), hutchinson_samples=2)
    assert v.shape == (2, 3, 4, 4, 2)


# ------------------------------------------------------- volume_change_surrogate

def test_the_surrogate_is_the_trace_when_it_is_not_an_estimate():
    """With D Hutchinson samples the estimator is exact: it equals tr(f' g')."""
    encode, decode, A, B = linear_pair()
    x = torch.randn(5, DIM)
    out = volume_change_surrogate(x, encode, decode, hutchinson_samples=DIM)
    expected = torch.trace(A @ B).expand(5)
    torch.testing.assert_close(out.surrogate, expected, rtol=1e-4, atol=1e-4)


def test_the_surrogate_of_an_exact_inverse_is_the_dimension():
    """f' g' = I, so its trace is D -- and the volume term vanishes."""
    encode, decode, _, _ = linear_pair()
    out = volume_change_surrogate(torch.randn(3, DIM), encode, decode,
                                  hutchinson_samples=DIM)
    torch.testing.assert_close(out.surrogate, torch.full((3,), float(DIM)),
                               rtol=1e-4, atol=1e-4)


def test_the_surrogates_gradient_is_the_gradient_of_the_log_determinant():
    """The point of the surrogate: same gradient, no Jacobian.

    d/dtheta tr(f'(x) SG(g'(z))) equals d/dtheta log|det f'(x)| when g = f^-1,
    which is what makes training on the surrogate train on the volume term.
    """
    theta = torch.randn(DIM, DIM, requires_grad=True)
    base = torch.eye(DIM) + 0.1 * torch.randn(DIM, DIM,
                                              generator=torch.Generator().manual_seed(1))

    def encode(x):
        return x @ (base + 0.1 * theta).T

    def decode(z):
        return z @ torch.linalg.inv((base + 0.1 * theta).detach()).T

    x = torch.randn(6, DIM)
    surrogate = volume_change_surrogate(x, encode, decode,
                                        hutchinson_samples=DIM).surrogate
    (grad_surrogate,) = torch.autograd.grad(surrogate.sum(), theta)

    theta2 = theta.detach().clone().requires_grad_(True)
    logdet = torch.linalg.slogdet(base + 0.1 * theta2)[1] * 6
    (grad_logdet,) = torch.autograd.grad(logdet, theta2)
    torch.testing.assert_close(grad_surrogate, grad_logdet, rtol=1e-3, atol=1e-4)


def test_the_surrogate_returns_the_reconstruction_it_used():
    encode, decode, _, _ = linear_pair()
    x = torch.randn(3, DIM)
    out = volume_change_surrogate(x, encode, decode, hutchinson_samples=2)
    torch.testing.assert_close(out.x1, x, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out.z, encode(x))


# ------------------------------------------------------------------- fff_loss

def test_fff_loss_of_an_exact_inverse_is_the_negative_log_likelihood():
    """No reconstruction error, so only the density term is left."""
    encode, decode, A, _ = linear_pair()
    latent = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(DIM), torch.ones(DIM)), 1)
    x = torch.randn(5, DIM)
    loss = fff_loss(x, encode, decode, latent, beta=1.0, hutchinson_samples=DIM)
    expected = -latent.log_prob(encode(x)) - DIM
    torch.testing.assert_close(loss, expected, rtol=1e-3, atol=1e-3)


def test_fff_loss_penalises_a_reconstruction_error_by_beta():
    encode, decode, _, _ = linear_pair()
    latent = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(DIM), torch.ones(DIM)), 1)
    x = torch.randn(5, DIM)
    broken = lambda z: decode(z) + 1.0  # noqa: E731
    a = fff_loss(x, encode, broken, latent, beta=1.0, hutchinson_samples=DIM)
    b = fff_loss(x, encode, broken, latent, beta=2.0, hutchinson_samples=DIM)
    torch.testing.assert_close(b - a, torch.full((5,), float(DIM)),
                               rtol=1e-3, atol=1e-3)


def test_reconstruction_loss_sums_over_every_non_batch_dimension():
    a = torch.zeros(2, 3, 4, 4)
    b = torch.ones(2, 3, 4, 4)
    torch.testing.assert_close(reconstruction_loss(a, b),
                               torch.full((2,), 48.0))


def test_sum_except_batch_keeps_one_number_per_sample():
    assert sum_except_batch(torch.ones(5, 2, 3)).tolist() == [6.0] * 5


# ------------------------------------------------------------------ jacobians

def test_compute_jacobian_matches_autograd():
    def fn(x):
        return torch.stack([x[:, 0] ** 2, (x[:, 1] * x[:, 2]).sin(),
                            x.sum(-1)], dim=-1)

    x = torch.randn(4, 3)
    out, jac = compute_jacobian(x, fn)
    torch.testing.assert_close(out, fn(x))
    for i in range(4):
        expected = torch.autograd.functional.jacobian(
            lambda xi: fn(xi[None])[0], x[i])
        torch.testing.assert_close(jac[i], expected, rtol=1e-4, atol=1e-4)


def test_forward_and_backward_jacobians_agree():
    fn = lambda x: (x ** 3).cumsum(-1)  # noqa: E731
    x = torch.randn(3, 4)
    _, back = compute_jacobian(x, fn, grad_type="backward")
    _, forward = compute_jacobian(x, fn, grad_type="forward")
    torch.testing.assert_close(back, forward, rtol=1e-4, atol=1e-4)


def test_volume_change_of_a_square_jacobian_is_its_log_determinant():
    jac = torch.randn(4, 3, 3) + 3 * torch.eye(3)
    torch.testing.assert_close(compute_volume_change(jac),
                               torch.linalg.slogdet(jac)[1])


@pytest.mark.parametrize("shape", [(6, 5, 3), (6, 3, 5)])
def test_volume_change_of_a_rectangular_jacobian_is_the_gramian_half_logdet(shape):
    """Injective (out > in) and projecting (out < in) both go through the Gram matrix."""
    jac = torch.randn(*shape)
    small = min(shape[1:])
    gram = jac @ jac.transpose(1, 2) if shape[1] == small else jac.transpose(1, 2) @ jac
    torch.testing.assert_close(compute_volume_change(jac),
                               torch.linalg.slogdet(gram)[1] / 2,
                               rtol=1e-4, atol=1e-4)


def test_batch_wrap_passes_a_scalar_argument_through():
    def fn(x, scale):
        return x * scale

    assert batch_wrap(fn)(torch.ones(3), 2.0).tolist() == [2.0, 2.0, 2.0]


# ------------------------------------------------------------- image shapes

@pytest.mark.parametrize("dim,shape", [
    (3 * 28 * 28, (3, 28, 28)),
    (28 * 28, (1, 28, 28)),
    (3 * 32 * 32, (3, 32, 32)),
    (3 * 256 * 256, (3, 256, 256)),
    (3 * 38804, (3, 178, 218)),  # CelebA, the one non-square special case
])
def test_guess_image_shape_recovers_the_usual_shapes(dim, shape):
    assert guess_image_shape(dim) == shape


def test_guess_image_shape_refuses_a_dimension_that_is_not_an_image():
    with pytest.raises(ValueError):
        guess_image_shape(1000)


def test_guess_image_shape_is_never_ambiguous():
    """3 k^2 = m^2 has no integer solutions, so the channel guess is safe.

    Worth stating: the function guesses 3 channels whenever the dimension is
    divisible by 3, and a grayscale image whose side length happened to make
    that work would be silently reshaped into a colour one.
    """
    for side in range(2, 200):
        grayscale = side * side
        if grayscale % 3 == 0:
            with pytest.raises(ValueError):
                guess_image_shape(grayscale)
