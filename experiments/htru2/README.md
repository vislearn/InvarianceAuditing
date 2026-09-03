# HTRU2 — non-image (tabular) invariance auditing

Rebuttal experiment demonstrating that the paper's method (NDTM guidance + fiber loss)
generalizes beyond images to a **non-image, tabular** modality, as requested by the
reviewer.

## Why HTRU2

[HTRU2](https://archive.ics.uci.edu/dataset/372/htru2) (Lyon et al.) is a pulsar-candidate
dataset: 17,898 candidates, **8 continuous features**, binary label (1,639 real pulsars /
16,259 spurious). The 8 features are individually interpretable statistics of two curves:

| idx | feature | curve |
|-----|---------|-------|
| 0-3 | mean, std, excess kurtosis, skewness | integrated pulse profile |
| 4-7 | mean, std, excess kurtosis, skewness | DM-SNR curve |

It is the right non-image choice because its features are **natively continuous**, so the
diffusion model and NDTM guidance operate directly on the raw standardized feature vector.
There is **no VAE and no categorical relaxation** — the friction that makes most tabular
data (Adult, COMPAS, Folktables, …) awkward for gradient-based diffusion guidance. This is
the cleanest possible non-image analogue of the colorMNIST benchmark (Sec. 4.1).

## Pipeline

All state lives in the 8-dim **standardized** feature space (fit on train only). The
diffusion model and the subject model therefore share exactly one space.

```
prepare_htru2.py            # download + stratified split + standardize -> data/htru2/htru2.npz
train_htru2_subject_model.py# MLP pulsar classifier; phi(x) = class logits (the fiber target h)
train_htru2_diffusion.py    # unconditional MLP DDPM over p(x)  (reuses LatentDenoiser/DDPM
                            #   from ../ColorMNIST/train_colormnist_latent_diffusion.py)
sample_htru2_ndtm.py        # NDTM guided invariance sampling (no VAE; classifier is phi)
analyze_htru2_invariances.py# fidelity / consistency / per-feature invariance + grad x-check
```

Run order:

```bash
python notebooks/HTRU2/prepare_htru2.py
python notebooks/HTRU2/train_htru2_subject_model.py     # ~1 min, test AUROC ~0.98
python notebooks/HTRU2/train_htru2_diffusion.py         # ~3 min, mean feature W1 ~0.02
for g in 1 2 5 10; do
  python notebooks/HTRU2/sample_htru2_ndtm.py --gamma $g --tag sweep
done
python notebooks/HTRU2/analyze_htru2_invariances.py
```

Sampling the full 3,580-candidate test set takes well under a minute per gamma on a 2080 Ti.

## What is measured

- **Fidelity** — fiber loss on classifier logits, `sqrt(sum d^2 / dim)` (paper metric), plus
  summed absolute class-probability difference (as in Sec. B.4).
- **Consistency** — mean per-feature 1-Wasserstein between the fiber samples and the data
  marginal (are the generated candidates still in-distribution, Eq. 4).
- **Interpretability** — per-feature movement of fiber samples relative to the query reveals
  which pulsar statistics the classifier's decision is *invariant* to (free directions) vs.
  which are *pinned* (load-bearing). Cross-checked against a gradient-based feature
  importance of the classifier (they anti-correlate: important features stay pinned).

## Notes

- Standardized features reach ~±11 (features are skewed), so the NDTM Tweedie-estimate clip
  range is set to ±12 (`--clip`), unlike the ±1 image case.
- No external guidance package is needed: the two functions NDTM used from oc-guidance are vendored into `fff/ndtm.py`. Same as
  the colorMNIST NDTM scripts.
