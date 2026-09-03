"""How `ImageNetDataset` brings images to the sampling resolution.

The nearest-neighbour baseline hits on cue conflict (0.7%, 0.8%) and misses on
ImageNet (-9%, -23%), across three subject models with DINOv2 on both sides --
so the discriminating variable is the dataset, not the model. The square resize
is the candidate mechanism: cue conflict stimuli are natively 224x224, so it is
a clean upscale there, while ImageNet validation images are not square and get
squashed before any subject model's own resize sees them.
"""

import numpy as np
import pytest

from fff.data.imagenet import ImageNetDataset


def transform_of(resize_mode, resize_to=256):
    # The ImageNet archive is not present here; the constructor warns and leaves
    # the dataset uninitialised, but still builds the transform, which is what
    # this is about.
    with pytest.warns(UserWarning, match="Could not load split"):
        dataset = ImageNetDataset(mode="test", root="/nonexistent",
                                  resize_to=resize_to, resize_mode=resize_mode)
    return dataset.transform


def applied(resize_mode, height, width, resize_to=256):
    image = (np.random.default_rng(0).random((height, width, 3)) * 255).astype("uint8")
    return transform_of(resize_mode, resize_to)(image=image)["image"]


@pytest.mark.parametrize("mode", ["square", "shortest"])
def test_both_modes_produce_the_sampling_resolution(mode):
    """Whatever else changes, the UNet needs a square input of a fixed size."""
    assert applied(mode, 375, 500).shape[-2:] == (256, 256)
    assert applied(mode, 500, 375).shape[-2:] == (256, 256)


def block_extent(mode, height, width, side=40):
    """Height and width, in output pixels, of a centred square block."""
    canvas = np.zeros((height, width, 3), dtype="uint8")
    top, left = (height - side) // 2, (width - side) // 2
    canvas[top:top + side, left:left + side] = 255
    out = transform_of(mode)(image=canvas)["image"].numpy()[0]
    lit = out > out.max() / 2
    return int(lit.any(axis=1).sum()), int(lit.any(axis=0).sum())


def test_shortest_mode_keeps_a_square_square():
    """The property that matters: both axes scaled by the same factor, so
    content is cropped rather than distorted."""
    tall_h, tall_w = block_extent("shortest", 200, 400)
    assert tall_h == pytest.approx(tall_w, abs=2)


def test_square_mode_turns_a_square_into_a_rectangle():
    """200x400 squashed into 256x256 scales the axes by 1.28 and 0.64, so a
    square block comes out twice as tall as it is wide."""
    height, width = block_extent("square", 200, 400)
    assert height > width * 1.6


def test_a_square_image_is_unaffected_by_the_choice():
    """Which is why cue conflict, at 224x224, cannot tell the two apart."""
    square = (np.random.default_rng(1).random((224, 224, 3)) * 255).astype("uint8")
    a = transform_of("square")(image=square)["image"]
    b = transform_of("shortest")(image=square)["image"]
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5)


def test_an_unknown_mode_is_refused():
    with pytest.warns(UserWarning, match="Could not load split"):
        with pytest.raises(ValueError, match="resize_mode must be one of"):
            ImageNetDataset(mode="test", root="/nonexistent", resize_to=256,
                            resize_mode="stretch")


def test_crop_mode_keeps_a_square_square_too():
    """Like 'shortest' it preserves aspect ratio; unlike it, nothing is rescaled."""
    height, width = block_extent("crop", 400, 600)
    assert height == pytest.approx(width, abs=2)


def test_crop_mode_does_not_rescale():
    """A 40px block stays 40px, where 'shortest' rescales it.

    400x600: the shortest side goes 400 -> 256, a 0.64x scale, so the block
    shrinks to ~26. 'crop' takes the middle 256x256 at native scale and leaves
    the block alone -- the same aspect ratio, different detail.
    """
    height, _ = block_extent("crop", 400, 600, side=40)
    assert height == pytest.approx(40, abs=2)
    scaled, _ = block_extent("shortest", 400, 600, side=40)
    assert scaled == pytest.approx(40 * 256 / 400, abs=2)


# ------------------------------------------------------- guidance scaling

def test_gamma_scale_multiplies_strengths_and_leaves_timing_alone():
    """--gamma-scale separates "the schedule is the wrong shape" from "it pulls
    too weakly". Scaling a timestep boundary instead would silently retime the
    guidance, which is a different experiment."""
    from experiments.imagenet.sample_ndtm import GAMMA_SCHEDULES, scaled_schedule

    base = GAMMA_SCHEDULES["default"]
    scaled = scaled_schedule("default", 2.5)
    assert len(scaled) == len(base)
    for (s0, e0, t0, t1), (s1, e1, u0, u1) in zip(base, scaled):
        assert (s1, e1) == pytest.approx((s0 * 2.5, e0 * 2.5))
        assert (u0, u1) == (t0, t1)


def test_gamma_scale_of_one_is_the_schedule_itself():
    from experiments.imagenet.sample_ndtm import GAMMA_SCHEDULES, scaled_schedule

    assert scaled_schedule("default", 1.0) == GAMMA_SCHEDULES["default"]


# ------------------------------------- ancestral sampling and the eta argument

def parse(*argv):
    import sys
    from experiments.imagenet import sample_ndtm
    saved = sys.argv
    sys.argv = ["sample_ndtm", "--subject-model", "dinov2",
                "--base-model", "x.pt", *argv]
    try:
        return sample_ndtm.parse_args()
    finally:
        sys.argv = saved


def test_ancestral_sampling_is_on_unless_turned_off():
    """Table 5 was drawn with ancestral sampling, so the default must not move."""
    assert parse().ancestral_sampling is True
    assert parse("--no-ancestral-sampling").ancestral_sampling is False
    assert parse("--ancestral-sampling").ancestral_sampling is True


def test_eta_defaults_to_the_subject_models_value_and_can_be_overridden():
    """eta is inert under ancestral sampling and live under DDIM, so it has to be
    settable independently of which branch is running."""
    from experiments.imagenet.sample_ndtm import ETA

    assert parse().eta is None            # resolved from ETA in main()
    assert parse("--eta", "1.0").eta == pytest.approx(1.0)
    assert ETA["inception"] == pytest.approx(1.0)   # the DDPM-equivalent limit


def test_the_two_schedules_are_reachable_and_distinct():
    from experiments.imagenet.sample_ndtm import GAMMA_SCHEDULES, SETTINGS

    assert set(GAMMA_SCHEDULES) == {"default", "inception"}
    assert len({tuple(v) for v in GAMMA_SCHEDULES.values()}) == len(GAMMA_SCHEDULES)
    # every row's schedule is one of them
    assert {gamma for gamma, _ in SETTINGS.values()} <= set(GAMMA_SCHEDULES)


def test_every_schedule_covers_the_whole_interval():
    """get_gamma_t_fct raises if the anchorpoints leave a gap, at sampling time.

    A schedule that fails only at t=317 of a 20-hour run is the expensive way to
    find that out, so evaluate every one of them here instead.
    """
    import torch
    from fff.ndtm import get_gamma_t_fct
    from experiments.imagenet.sample_ndtm import GAMMA_SCHEDULES

    for name, schedule in GAMMA_SCHEDULES.items():
        gamma = get_gamma_t_fct(schedule, max_timesteps=1000)
        for t in (0, 1, 199, 200, 201, 599, 600, 799, 800, 995, 999):
            gamma(torch.tensor(t))          # raises ValueError on a gap
