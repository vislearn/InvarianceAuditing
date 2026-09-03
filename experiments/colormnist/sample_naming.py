"""How colorMNIST NDTM sample files are named, in one place.

`sample_ndtm.py` writes these files and `make_figures.py` globs them back into
buckets of (variant, gamma, seed). Both derive the name from here, so a change
to one cannot silently strand the other.

The seed reaches the filename only through `--tag`. Without it a file lands as
`gamma={g}_{timestamp}_{nonce}.pt`, which still matches the variant prefix but
no longer matches the per-seed glob `make_figures` buckets on -- so pass
`--tag seed$seed` when sampling a seed sweep.
"""

from datetime import datetime
import random

PREFIX = "sampled_colormnist_latent_space"


def sample_basename(variant: str, gamma, tag: str, when: str | None = None,
                    nonce: int | None = None) -> str:
    """The filename a sampling run writes.

    `variant` is the empty string for in-distribution and otherwise ends in "_"
    ("uncorrelated_", "uncorrelated_oodvae_", "correlated_ctrl_"). `tag` carries
    the seed for runs that make_figures will bucket -- see `seed_tag`.
    """
    when = when or datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    nonce = random.getrandbits(16) if nonce is None else nonce
    suffix = (tag + "_" if tag else "") + when + "_" + str(nonce)
    return f"{PREFIX}_{variant}invariances_gamma={gamma}_{suffix}.pt"


def seed_tag(seed: int) -> str:
    """The tag a run must carry for make_figures to find it."""
    return f"seed{int(seed)}"


def bucket_glob(variant_prefix: str, gamma, seed: int) -> str:
    """The pattern make_figures uses to find the one file for a bucket.

    `variant_prefix` is the full prefix recorded in make_figures' VARIANTS, e.g.
    "sampled_colormnist_latent_space_uncorrelated_invariances".
    """
    return f"{variant_prefix}_gamma={gamma}_{seed_tag(seed)}_*.pt"
