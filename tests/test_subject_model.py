"""SubjectModel: the wrapper every experiment loads its phi through.

The failure that cost a 21-job array (see test_entry_points.py) was in here, so
what this checks is mostly that the wrapper's error paths say what went wrong.
Loading a real checkpoint needs a checkpoint; the conditioning and transform
logic does not, so that is what is exercised directly.
"""

import warnings

import pytest
import torch
import torch.nn as nn

from fff.subject_model import SubjectModel


class Encoder(nn.Module):
    """Stands in for a fiber model: conditional, with an encode()."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def encode(self, x, *c, **kwargs):
        self.seen.append(("encode", [tuple(ci.shape) for ci in c]))
        return x[:, :2]

    def decode(self, z, *c, **kwargs):
        self.seen.append(("decode", [tuple(ci.shape) for ci in c]))
        return torch.zeros(z.shape[0], 4)

    def forward(self, x, *c, **kwargs):
        self.seen.append(("forward", [tuple(ci.shape) for ci in c]))
        return x[:, :2]


class PlainModel(nn.Module):
    """Stands in for a torchvision-style subject model: no encode()."""

    def forward(self, x, *c, **kwargs):
        return x.mean(-1, keepdim=True)


def wrap(model, **kwargs):
    """A SubjectModel around an already-constructed module."""
    subject = SubjectModel.__new__(SubjectModel)
    nn.Module.__init__(subject)
    subject.model = model
    subject.fixed_transform = kwargs.get("fixed_transform")
    subject.empty_condition = kwargs.get("empty_condition", False)
    return subject


def test_empty_condition_supplies_the_zero_width_tensor():
    """The colorMNIST subject model records cond_dim: 0 and must be told so.

    Its encode() still takes a condition argument; without this it is called
    with none and reports a missing argument to forward().
    """
    model = Encoder()
    out = wrap(model, empty_condition=True).encode(torch.randn(3, 4))
    assert out.shape == (3, 2)
    assert model.seen == [("encode", [(3, 0)])]


def test_without_empty_condition_the_caller_supplies_the_condition():
    model = Encoder()
    wrap(model).encode(torch.randn(3, 4), torch.zeros(3, 5))
    assert model.seen == [("encode", [(3, 5)])]


def test_empty_condition_applies_to_forward_and_decode_as_well():
    model = Encoder()
    subject = wrap(model, empty_condition=True)
    subject(torch.randn(3, 4))
    subject.decode(torch.randn(3, 2))
    assert model.seen == [("forward", [(3, 0)]), ("decode", [(3, 0)])]


def test_encode_falls_back_to_calling_a_model_without_one():
    assert wrap(PlainModel()).encode(torch.randn(3, 4)).shape == (3, 1)


def test_an_error_inside_encode_is_not_swallowed():
    """The bare `except:` this replaced reported the wrong call as the failure."""

    class Broken(nn.Module):
        def encode(self, x, *c, **kwargs):
            raise ValueError("the real problem")

        def forward(self, x, *c, **kwargs):
            return x

    with pytest.raises(ValueError, match="the real problem"):
        wrap(Broken()).encode(torch.randn(2, 3))


def test_the_fixed_transform_runs_before_the_model():
    seen = {}

    def transform(x):
        seen["called"] = True
        return x * 0

    out = wrap(PlainModel(), fixed_transform=transform).encode(torch.ones(2, 4))
    assert seen.get("called") and out.abs().sum() == 0


def test_the_fixed_transform_also_runs_in_forward():
    subject = wrap(PlainModel(), fixed_transform=lambda x: x * 0)
    assert subject(torch.ones(2, 4)).abs().sum() == 0


def test_a_named_fixed_transform_that_does_not_exist_is_refused(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(Encoder(), path)
    with pytest.raises(NotImplementedError, match="greyscale"):
        SubjectModel(str(path), fixed_transform="greyscale")


def test_calling_a_subject_model_that_was_never_loaded_says_so():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        subject = SubjectModel(None)
    with pytest.raises(RuntimeError, match="No subject model"):
        subject(torch.zeros(2, 4))
    with pytest.raises(RuntimeError, match="No subject model"):
        subject.encode(torch.zeros(2, 4))


@pytest.mark.parametrize("suffix", [".ckpt", ".pt"])
def test_a_missing_checkpoint_path_is_reported_as_a_missing_path(tmp_path, suffix):
    missing = str(tmp_path / f"absent{suffix}")
    with pytest.raises(FileNotFoundError) as excinfo:
        SubjectModel(missing)
    assert missing in str(excinfo.value)


def test_an_unsupported_model_type_is_refused(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(PlainModel(), path)
    with pytest.raises(NotImplementedError):
        SubjectModel(str(path), model_type="TransformerThing")


def test_a_pickled_model_without_encode_is_refused_by_name(tmp_path):
    """Inference has to fail on the model, not later on a missing method."""
    path = tmp_path / "model.pt"
    torch.save(PlainModel(), path)
    with pytest.raises(NotImplementedError, match="encode"):
        SubjectModel(str(path))


def test_a_pickled_model_with_encode_and_decode_is_accepted(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(Encoder(), path)
    subject = SubjectModel(str(path))
    assert subject.encode(torch.randn(2, 4)).shape == (2, 2)


def test_a_loaded_subject_model_is_frozen(tmp_path):
    """phi is fixed; a gradient reaching it would train the thing being audited."""
    path = tmp_path / "model.pt"
    torch.save(Encoder(), path)
    subject = SubjectModel(str(path))
    assert not any(p.requires_grad for p in subject.parameters())


def test_guidance_gradients_still_reach_the_input_of_a_frozen_model(tmp_path):
    """Frozen parameters must not mean a detached graph.

    NDTM differentiates the terminal cost with respect to the control, through
    phi. If freezing had blocked that, guidance would silently do nothing.
    """
    path = tmp_path / "model.pt"
    torch.save(Encoder(), path)
    subject = SubjectModel(str(path))
    x = torch.randn(2, 4, requires_grad=True)
    subject.encode(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
