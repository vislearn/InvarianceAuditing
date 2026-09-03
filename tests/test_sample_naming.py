"""That a sampling run's filename is one make_figures can find.

The seed reaches a colorMNIST NDTM filename only through `--tag`. Omit it and a
run lands as `gamma={g}_{timestamp}_{nonce}.pt`; `variant_present` still matches
on the prefix, so the variant looks present and `bucket_files` only then asserts
"expected exactly 1 file ... got []" -- after the sampling has run.

Both sides derive the name from `sample_naming`, and these check that the two
halves still meet.
"""

import fnmatch

import pytest

from experiments.colormnist.sample_naming import (bucket_glob, sample_basename,
                                                  seed_tag)

VARIANTS = {
    "": "sampled_colormnist_latent_space_invariances",
    "uncorrelated_": "sampled_colormnist_latent_space_uncorrelated_invariances",
    "uncorrelated_oodvae_": "sampled_colormnist_latent_space_uncorrelated_oodvae_invariances",
    "correlated_ctrl_": "sampled_colormnist_latent_space_correlated_ctrl_invariances",
}
GAMMAS = [1.0, 2.0, 5.0, 10.0]
SEEDS = [0, 1, 2]


@pytest.mark.parametrize("variant,prefix", VARIANTS.items())
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_a_tagged_run_is_found_by_its_bucket(variant, prefix, gamma, seed):
    name = sample_basename(variant, gamma, seed_tag(seed), when="12_00_00__01_01_2026", nonce=7)
    assert fnmatch.fnmatch(name, bucket_glob(prefix, gamma, seed))


def test_an_untagged_run_is_not_found():
    """The failure this module exists to prevent."""
    name = sample_basename("", 1.0, "", when="12_00_00__01_01_2026", nonce=7)
    assert not fnmatch.fnmatch(name, bucket_glob(VARIANTS[""], 1.0, 0))


def test_buckets_do_not_capture_each_other():
    """The in-distribution glob must not swallow the recolor variants."""
    indist = bucket_glob(VARIANTS[""], 5.0, 0)
    for variant, prefix in VARIANTS.items():
        if variant == "":
            continue
        other = sample_basename(variant, 5.0, seed_tag(0), when="12_00_00__01_01_2026", nonce=7)
        assert not fnmatch.fnmatch(other, indist), f"{variant} leaks into the in-dist bucket"


def test_seeds_do_not_capture_each_other():
    name = sample_basename("", 5.0, seed_tag(1), when="12_00_00__01_01_2026", nonce=7)
    assert not fnmatch.fnmatch(name, bucket_glob(VARIANTS[""], 5.0, 0))
    assert fnmatch.fnmatch(name, bucket_glob(VARIANTS[""], 5.0, 1))


def test_gammas_do_not_capture_each_other():
    """gamma=1.0 must not match a gamma=10.0 bucket or vice versa."""
    one = sample_basename("", 1.0, seed_tag(0), when="12_00_00__01_01_2026", nonce=7)
    ten = sample_basename("", 10.0, seed_tag(0), when="12_00_00__01_01_2026", nonce=7)
    assert not fnmatch.fnmatch(one, bucket_glob(VARIANTS[""], 10.0, 0))
    assert not fnmatch.fnmatch(ten, bucket_glob(VARIANTS[""], 1.0, 0))


def test_make_figures_variants_match_this_modules_view():
    """If make_figures gains or renames a variant, this test says so."""
    from experiments.colormnist import make_figures
    assert {p for p, _ in make_figures.VARIANTS.values()} == set(VARIANTS.values())
    assert make_figures.GAMMAS == GAMMAS and make_figures.SEEDS == SEEDS
