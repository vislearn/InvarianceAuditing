---
license: other
license_name: chexpert-research-use
license_link: https://stanfordmlgroup.github.io/competitions/chexpert/
base_model: facebook/convnextv2-tiny-22k-384
library_name: transformers
tags:
  - medical
  - chest-xray
  - image-classification
  - multi-label
  - interpretability
  - chexpert
pipeline_tag: image-classification
---

# ConvNeXt V2-tiny CheXpert (seed 2)

A five-finding chest radiograph classifier: `facebook/convnextv2-tiny-22k-384`
fine-tuned end to end on CheXpert. It is one of the two subject models audited
in *"Show Me What You Don't Know: Efficient Sampling from Invariant Sets for
Model Validation"* (Rousselot, Wendebourg and Köthe, 2026), where it is not the
end product but the thing being examined — the paper samples from its **fibers**,
the sets of images it maps to the same representation.

**This is one of an identically trained pair.**
[`convnext-chexpert-seed1`](https://huggingface.co/RussellALA/convnext-chexpert-seed1)
is the same recipe under a different random seed. The paper audits both to ask
whether two models that differ only in training randomness end up sharing their
invariances (Figure 18) — so if you want a single classifier, take seed 1; this
one exists to make that pair experiment reproducible.

- Code: [vislearn/InvarianceAuditing](https://github.com/vislearn/InvarianceAuditing)
- Paper: [arXiv:2603.21782](https://arxiv.org/abs/2603.21782)
- Companion model: [`biomedclip-chexpert`](https://huggingface.co/RussellALA/biomedclip-chexpert)

## Labels

Multi-label, five CheXpert competition findings, in this order:

```
["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
```

The model outputs five raw logits. In the paper, φ(x) is that logit vector, and
the fiber loss is the per-class probability distance of Appendix B.4.

## Loading it

A standard `transformers` image classifier:

```python
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "RussellALA/convnext-chexpert-seed2").eval()
logits = model(pixel_values).logits            # (B, 5)
```

Or through the paper's subject-model wrapper, which takes a normalised
1-channel batch and repeats the channel to three:

```python
from experiments.chexpert.subject_models import ConvNextClassfierSubjectModel

model = ConvNextClassfierSubjectModel("RussellALA/convnext-chexpert-seed2",
                                      n_channels=1).eval()
```

## Training

Trained with `notebooks/chexpert_classifier.ipynb` in the repository above, which
is adapted from [a CheXpert notebook](https://www.kaggle.com/code/shreydan/chexpert-multi-label-classifier)
by Shreyas Daniel Gaddam ([`shreydan`](https://www.kaggle.com/shreydan), Apache
License 2.0) — the data pipeline, loss,
model and trainer below are that notebook's.

| | |
|---|---|
| Base | `facebook/convnextv2-tiny-22k-384`, fine-tuned end to end |
| Data | CheXpert v1.0-small, **frontal views only** |
| Split | `train.csv` divided by `StratifiedGroupKFold` **grouped on PatientID**, so no patient crosses the split; `valid.csv` (202 frontal studies) held out as the test set |
| Resolution | 384 × 384 |
| Augmentation | Rotate(15°), horizontal flip |
| Normalisation | Per-channel ImageNet statistics, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |
| Loss | Masked Asymmetric Loss ([Ridnik et al., 2020](https://arxiv.org/abs/2009.14119)), `gamma_neg=3`, with **uncertain (-1) labels masked out** rather than imputed |
| Optimiser | AdamW, lr 1e-3, weight decay 0.02, cosine schedule, 1600 warmup steps |
| Epochs | 3, batch size 16, best checkpoint by validation loss |

Missing labels are filled with 0; uncertain labels are excluded from the loss.

## ⚠️ Input convention

Two conventions are in play and they are not interchangeable:

- **As trained** (and what you should use for ordinary inference): 384×384, RGB,
  **per-channel** ImageNet normalisation.
- **As audited in the paper**: the invariance-auditing pipeline normalises the
  1-channel image with *grayscale-collapsed* statistics — the mean of the
  per-channel values, 0.4490 / 0.2260 — and then repeats that one channel three
  times. `renormalize_grayscale` in `experiments/chexpert/subject_models.py`
  converts between the two, and is deliberately never called.

Measured against the logits the paper's runs stored, the per-channel convention
sits 1.19% away from this model as a probability distance, where the repeat-only
convention sits 0.0016% away. The BiomedCLIP companion is twenty times more
sensitive, at 25.17% against 0.0001%.

## Evaluation

Validation used label-wise AUROC, plus exact-match, specificity and Hamming
distance (`torchmetrics` multilabel). The paper reports this model's **fiber
loss** and nearest-neighbour baseline in Table 5, and the agreement between
this model and its seed-1 twin in Figure 18 — measurements of what the model
treats as equivalent, not of its diagnostic accuracy.

## Intended use and limitations

This is a **research artifact**, published so that the paper's invariance audit
can be reproduced. It is not a diagnostic tool and must not be used for clinical
decisions, or on patients.

Known limitations:

- Trained on a single institution's data (Stanford Hospital) at low resolution,
  on five findings out of CheXpert's fourteen.
- The paper's own finding is the sharpest limitation: the model assigns the same
  representation to images that differ in ways a radiologist would not consider
  equivalent. That is what the fiber samples show.
- Uncertain labels were masked, so the model never learned to express
  uncertainty; its outputs are not calibrated.
- Inherits whatever demographic and acquisition biases are in CheXpert.

## Licence and data terms

The weights are derived from CheXpert, which is distributed under the
[Stanford University Dataset Research Use Agreement](https://stanfordmlgroup.github.io/competitions/chexpert/):
**research use only, non-commercial**. By using this model you accept those
terms as they extend to derivatives. The CheXpert images themselves are not
redistributed here or in the paper's repository — request them from Stanford.

## Citation

```bibtex
@article{rousselot2026show,
  title={Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model Validation},
  author={Rousselot, Armand and Wendebourg, Joran and K{\"o}the, Ullrich},
  journal={arXiv preprint arXiv:2603.21782},
  year={2026}
}
```
