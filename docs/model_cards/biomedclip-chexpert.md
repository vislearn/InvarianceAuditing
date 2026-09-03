---
license: other
license_name: chexpert-research-use
license_link: https://stanfordmlgroup.github.io/competitions/chexpert/
base_model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
tags:
  - medical
  - chest-xray
  - image-classification
  - multi-label
  - interpretability
  - chexpert
pipeline_tag: image-classification
---

# BiomedCLIP-CheXpert (frozen encoder + residual head)

A five-finding chest radiograph classifier: the **frozen** BiomedCLIP image tower
with a small residual MLP head trained on CheXpert. It is one of the two subject
models audited in *"Show Me What You Don't Know: Efficient Sampling from
Invariant Sets for Model Validation"* (Rousselot, Wendebourg and Köthe, 2026),
where it is not the end product but the thing being examined — the paper samples
from its **fibers**, the sets of images it maps to the same representation.

- Code: [vislearn/InvarianceAuditing](https://github.com/vislearn/InvarianceAuditing)
- Paper: [arXiv:2603.21782](https://arxiv.org/abs/2603.21782)
- Companion models: [`convnext-chexpert-seed1`](https://huggingface.co/RussellALA/convnext-chexpert-seed1),
  [`convnext-chexpert-seed2`](https://huggingface.co/RussellALA/convnext-chexpert-seed2)

## Labels

Multi-label, five CheXpert competition findings, in this order:

```
["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
```

The model outputs five raw logits. In the paper, φ(x) is that logit vector, and
the fiber loss is the per-class probability distance of Appendix B.4.

## Architecture

| | |
|---|---|
| Encoder | BiomedCLIP ViT-B/16 image tower, **frozen** (`open_clip`, from `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) |
| Head | 2 × residual MLP block (LayerNorm → Linear → SiLU → Dropout → Linear, hidden = 4 × embed), then LayerNorm → SiLU → Dropout → Linear(5) |
| Dropout | 0.05 |
| Trainable | The head only — the encoder's parameters are frozen |

The encoder applies its own preprocessing internally: shortest side to 224,
centre crop, grayscale repeated to 3 channels, CLIP normalisation.

## Loading it

This is **not** a `transformers` `AutoModel` — it is a `safetensors` state dict
for the `BiomedClipClassifier` module defined in the paper's repository, and the
BiomedCLIP tower is fetched from its own hub at construction time.

```python
from huggingface_hub import snapshot_download
from experiments.chexpert.subject_models import BiomedClipSubjectModel

path = snapshot_download("RussellALA/biomedclip-chexpert")
model = BiomedClipSubjectModel(path, n_channels=1).eval()   # forward(x) -> (B, 5) logits
```

`BiomedClipSubjectModel` expects a normalised **1-channel** 384×384 batch and
repeats the channel three times internally. See the input note below.

## Training

Trained with `notebooks/chexpert_classifier.ipynb` in the repository above. Its
data pipeline, loss, metrics and trainer are adapted from
[a CheXpert notebook](https://www.kaggle.com/code/shreydan/chexpert-multi-label-classifier)
by Shreyas Daniel Gaddam ([`shreydan`](https://www.kaggle.com/shreydan), Apache
License 2.0); the BiomedCLIP head and its
training section are ours.

| | |
|---|---|
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

Worth knowing before you use these weights, because there are two conventions in
play and they are not interchangeable:

- **As trained** (and what you should use for ordinary inference): 384×384, RGB,
  **per-channel** ImageNet normalisation.
- **As audited in the paper**: the invariance-auditing pipeline normalises the
  1-channel image with *grayscale-collapsed* statistics — the mean of the
  per-channel values, 0.4490 / 0.2260 — and then repeats that one channel three
  times. `renormalize_grayscale` in `experiments/chexpert/subject_models.py`
  converts between the two, and is deliberately never called.

This model is unusually sensitive to the difference. Measured against the logits
the paper's runs stored, the per-channel convention sits **25.17%** away as a
probability distance, where the repeat-only convention sits **0.0001%** away. The
same comparison moves the ConvNeXt companions by 1.19% and 0.0016% — twenty times
less, which is why the discrepancy was easy to miss on those.

## Evaluation

Validation used label-wise AUROC, plus exact-match, specificity and Hamming
distance (`torchmetrics` multilabel). The paper reports this model's **fiber
loss** and nearest-neighbour baseline in Table 5, which measures how tightly
NDTM can contract onto its fibers — a property of the model's invariances, not
of its diagnostic accuracy.

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

The BiomedCLIP encoder is Microsoft's and carries its own licence; it is fetched
from its own hub repository rather than included here.

## Citation

```bibtex
@article{rousselot2026show,
  title={Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model Validation},
  author={Rousselot, Armand and Wendebourg, Joran and K{\"o}the, Ullrich},
  journal={arXiv preprint arXiv:2603.21782},
  year={2026}
}
```
