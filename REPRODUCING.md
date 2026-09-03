# Reproducing the paper

Every number and figure in *Show Me What You Don't Know: Efficient Sampling from
Invariant Sets for Model Validation*, with the command that produces it.

## Setup

```bash
pip install -r requirements.txt
export FFF_DATA_ROOT=/path/to/datasets-and-checkpoints
export FFF_OUTPUT_ROOT=/path/to/samples-and-figures     # needs room: ~1 GB per colorMNIST setting
export FFF_DOWNLOAD_DATASETS=1                          # allow torchvision downloads in batch jobs
```

Nothing else is machine-specific. `FID` (Table 2) additionally needs the separate
TensorFlow environment described at the bottom of `requirements.txt`; it is the
only part that does.

Sanity check before submitting anything long:

```bash
python -m pytest tests/ -q          # every experiment module imports
```

## ImageNet settings

The DINOv2/ImageNet configuration in `sample_ndtm.py` is N=5, u_lr 2e-3,
w_terminal 1.0, eta 0.5, linear decay, kappa = tau = 0, ancestral sampling,
per-timestep target, **200 denoising steps**, and

    gamma = [(0, 0, 1000, 800), (3, 3, 800, 600), (1, 0.5, 600, 200), (2, 10, 200, 0)]

`variance_type` is `learned_range` for the two ImageNet rows and `small` for the
two cue conflict rows; `SETTINGS` in `sample_ndtm.py` holds the per-row table and
the section below gives it in full. The fibers are drawn from a seeded
*permutation* of the 50k validation images, not the first 10k, which
`--shuffle-seed 0` reproduces.

### Calibrating a setting

Sample 32 fibers before committing to a full run:

```bash
set -- dinov2:imagenet inception:imagenet dinov2:cue_conflict resnet50:cue_conflict
for spec in "$@"; do
  m=${spec%%:*}; d=${spec##*:}
  python -m experiments.imagenet.sample_ndtm --subject-model $m --dataset $d \
    --base-model $FFF_DATA_ROOT/256x256_diffusion_uncond.pt \
    --num-images 32 --out $FFF_OUTPUT_ROOT/calibration/${m}_${d}
  python -m experiments.imagenet.evaluate $FFF_OUTPUT_ROOT/calibration/${m}_${d}/*
done
```

Table 5 gives 273, 34.8, 487 and 38.1 in that order. Read a calibration with the
spread in mind: two runs of dinov2/cue_conflict under identical settings came out
597.9 and 538.6, 10% apart, and a w_terminal sweep that should be monotone gave
51.8 / 56.7 / 49.8 / 55.0 at w = 1 / 2 / 3 / 5. 32 fibers resolves a large offset,
not a 15% one; raise `--num-images` before concluding anything from a single
calibration.

`eta` is listed above for completeness only: it scales `c1`/`c2`, which the
ancestral update never reads, so it has no effect on any of these runs.

## ImageNet and cue conflict -- Table 5 (rows 1-4), Table 2, Figures 5, 6, 7, 19

Needs `256x256_diffusion_uncond.pt` from
[guided-diffusion](https://github.com/openai/guided-diffusion) and that package
importable. The paper draws 10k fiber samples per setting.

```bash
BASE=$FFF_DATA_ROOT/256x256_diffusion_uncond.pt

OUT=$FFF_OUTPUT_ROOT/imagenet

python -m experiments.imagenet.sample_ndtm --subject-model dinov2    --dataset imagenet     --base-model $BASE --num-images 10000 --out $OUT/dinov2_imagenet
python -m experiments.imagenet.sample_ndtm --subject-model inception --dataset imagenet     --base-model $BASE --num-images 10000 --out $OUT/inception_imagenet
python -m experiments.imagenet.sample_ndtm --subject-model dinov2    --dataset cue_conflict --base-model $BASE --num-images 10000 --out $OUT/dinov2_cue_conflict
python -m experiments.imagenet.sample_ndtm --subject-model resnet50  --dataset cue_conflict --base-model $BASE --num-images 10000 --out $OUT/resnet50_cue_conflict
```

`--out` is what keeps the four settings apart. Without it every run lands
directly in `$FFF_OUTPUT_ROOT/imagenet` under a prefix that names only the
dataset, so the two ImageNet settings become indistinguishable on disk and the
evaluator -- which pools every run it is given -- would average them.

Each of those four commands carries its own guidance schedule and
`variance_type`, from the `SETTINGS` table in `sample_ndtm.py`; `--gamma` and
`--variance-type` override them.

| row | gamma | variance_type |
|---|---|---|
| dinov2 / imagenet | `default` | `learned_range` |
| inception / imagenet | `inception` | `learned_range` |
| dinov2 / cue conflict | `default` | `small` |
| resnet50 / cue conflict | `default` | `small` |

`variance_type` is worth understanding before changing it: `learned_range`
injects strictly more noise per step than `small`, which moves the fiber loss by
10-50% depending on the row.

The two cue conflict settings are the exception: the texture-vs-shape stimulus
set holds **1280 images**, so `--num-images 10000` yields 1280 fibers however it
is sharded. Raise `--samples-per-image` to draw more than one fiber per image if
you need a larger set; the paper's own cue conflict rows rest on the 1280.

None of those finish in one job. Split each across an array with
`--shard $SLURM_ARRAY_TASK_ID --num-shards N --num-images <10000/N>`: the shards
stride the dataset, so they partition it and each stays representative. Every
shard writes its own run directory and the evaluation globs them together, so a
failed shard is just resubmitted.

Table 6 (modified vs unmodified NDTM) is the same command with the modification
switched off, and Table 7 is a re-evaluation of the DINOv2 run under all three
metrics:

```bash
python -m experiments.imagenet.sample_ndtm --subject-model dinov2 --base-model $BASE \
    --num-images 10000 --static-target --out $FFF_OUTPUT_ROOT/imagenet/dinov2_imagenet_static
python -m experiments.imagenet.evaluate $FFF_OUTPUT_ROOT/imagenet/dinov2_imagenet \
    --metrics l2 l1 cross_entropy
```

Name the one setting, not `imagenet/*`: the evaluator pools every run it is
given, so the glob averages all four settings into a single meaningless number.

Table 5's nearest-neighbour column is a separate pass, because finding it means
embedding the whole dataset once and doing that inside each of twenty shards
would be wasted work:

```bash
python -m experiments.imagenet.nearest_neighbours $FFF_OUTPUT_ROOT/imagenet/dinov2_imagenet
```

It writes `nearest_neighbours.pt` into each run directory; `evaluate.py` reports
the column as soon as it is there, and says nothing if it is not.

Figures and FID come from `notebooks/evaluate_imagenet.ipynb`.

> The fiber loss in Tables 5-7 is the **squared** l2 summed over dimensions.
> `NDTMConfig.fiber_loss="l2"`, the cost NDTM minimises, is the plain norm --
> the two differ by the square, which is worth remembering when reading the
> terminal loss printed during sampling.

## CheXpert -- Table 5 (rows 5-7), Figures 8, 10, 11, 16, 17, 18

The images are 384x384 and NDTM backpropagates through the diffusion model, so
memory is the binding constraint: the classifier pair needs `--batch-size 2` on
an 11 GB card, and Qwen does not fit there at all -- its vision tower is 665M
parameters in fp32 on top of the same UNet graph. Give the Qwen runs a 24 GB card
or larger.

Classifier pair, then the two identically trained ConvNeXts of Figure 18:

```bash
python -m experiments.chexpert.sample_ndtm --subject-models biomedclip convnext --num-images 202
python -m experiments.chexpert.sample_ndtm --subject-models convnext convnext2  --num-images 202
```

202 is the whole of CheXpert's own `valid.csv`, the frontal studies held out of
the classifiers' training, and `--split test` selects it. Table 5's "+/-" is
across independent draws of the *same* fibers, so run each of those eighteen
times with `--sample-seed 0..17` and `--seed 0` fixed; `--seed` chooses which
images are audited, `--sample-seed` only the noise. They take
`--shard`/`--num-shards` as well, sharding the shuffled order.

The nearest-neighbour column is the same separate pass as ImageNet's:

```bash
python -m experiments.chexpert.nearest_neighbours $FFF_OUTPUT_ROOT/chexpert/biomedclip_convnext
```

Table 5's BiomedCLIP and ConvNeXt rows, with the two faulty sample slots dropped:

```bash
python -m experiments.chexpert.table5_fiber_losses $FFF_OUTPUT_ROOT/chexpert/biomedclip_convnext
```

The two excluded slots are flagged by the file's own `masks` array; a fresh
sampling run has no slot axis and nothing to exclude.

Qwen-2B (Table 5 row 7 and Figure 10). The situs inversus radiographs come from
Mayer et al. (2025) and are not redistributed here; place them in
`$FFF_DATA_ROOT/rare_cases` as `situs_inversus_1.jpeg` ... `_4.jpeg`:

```bash
python -m experiments.chexpert.sample_qwen --queries chexpert       --num-images 60
python -m experiments.chexpert.sample_qwen --queries situs-inversus --repeats 5

python -m experiments.chexpert.table5_fiber_losses $FFF_OUTPUT_ROOT/chexpert/qwen
```

Both take **100 denoising steps**, where every other sampler here takes 200, and
that is the setting Table 5 row 7 was drawn at rather than an economy.

Do not raise it without meaning to. Guidance strength gamma is a function of the
diffusion timestep, not of the step index, so doubling the steps doubles the
number of corrections inside every anchor interval and roughly doubles the
accumulated control. At 200 steps this row reads 4.26 against the paper's 5.5 --
lower, because the extra guidance contracts harder -- and the samples leave the
data manifold visibly enough that the radiographs stop looking like radiographs,
which no fiber loss in Table 5 would have caught.

Qwen's row is on a different scale from the two classifier rows: phi is a
mean-pooled patch embedding rather than five logits, so its loss is the squared
l2 and not the per-class probability distance. `table5_fiber_losses` picks the
metric from the subject model's output width; `--metric` overrides it.

Generator and classifiers: `notebooks/chexpert_generator.ipynb`,
`notebooks/chexpert_classifier.ipynb`. Evaluation and figures:
`notebooks/evaluate_chexpert.ipynb`.

## colorMNIST benchmark -- Figure 4, Figure 14, Table 3

Train the VAE, then the 21 conditional fiber models (7 model classes x fiber-loss
weight lambda), then the latent diffusion model NDTM guides:

```bash
python -m lightning_trainable.launcher.fit configs/colormnist/lossless_vae.yaml --name lossless_vae
for cfg in configs/colormnist/fiber_models/*.yaml; do
  python -m lightning_trainable.launcher.fit "$cfg" --name "$(basename "$cfg" .yaml)"
done
python -m experiments.colormnist.train_latent_diffusion --epochs 600
```

`--name` is not optional: it is the log directory `compute_statistics.py` and
`make_figures.py` look each run up by, and without it all 21 collide. It writes
to `./lightning_logs`, so run this from where the logs should live and point
`FIBER_MODEL_LOGS` at that directory afterwards; `--log-dir` is broken upstream
in lightning-trainable. The VAE has to finish before the fiber models, which
load it.

Sampling with the tuned NDTM configuration (the script's defaults: gamma 5,
kappa 3e-4, tau 0, 200 steps, eta 0.25, N 10) and the benchmark statistics:

```bash
for g in 1.0 2.0 5.0 10.0; do
  for seed in 0 1 2; do
    python -m experiments.colormnist.sample_ndtm --gamma $g --seed $seed --tag seed$seed
  done
done
python -m experiments.colormnist.compute_statistics --model_class all
python -m experiments.colormnist.make_figures
```

**`--tag seed$seed` is not cosmetic.** The seed reaches the filename only through
the tag, and `make_figures` globs each bucket as
`..._gamma={g}_seed{n}_*.pt`. Without it the files land as
`gamma={g}_{timestamp}_{nonce}.pt`, the variant still looks present, and the
figures step fails with `expected exactly 1 file ... got []` -- after the
sampling has already run. `experiments/colormnist/sample_naming.py` is where both
sides get the convention from.

The middle line is doing more than its name suggests: `sample_ndtm` covers NDTM,
but the 21 conditional fiber models are sampled inside `compute_statistics`, which
draws 10 fibers for each of the 40,000 test images per model and then runs an
Inception pass per draw for FID. Both are cached per model, so it is a one-off,
but on one GPU it is the longest step in the colorMNIST pipeline rather than the
bookkeeping the name implies.

`--models` shards it -- each run writes its own directory, so one model per job
parallelises cleanly:

```bash
python -m experiments.colormnist.compute_statistics --models fff_lambda1
```

`--fid_slots 3` keeps the FID means and only widens their error bars; `--skip_fid`
leaves the column for `make_figures fid` to fill in later.

`make_figures` runs three phases -- `stats`, `fid`, `plots` -- and with no
argument runs all of them in order, skipping `fid` when TensorFlow is not
importable (the table's FID column is then left empty). Name a phase to run one
on its own.

## Causal MNIST -- Section 4.4, Figures 9, 12, 15

The two classifiers train in a couple of CPU-minutes, so there is nothing to
download:

```bash
python -m experiments.causal_mnist.train_classifiers
python -m experiments.causal_mnist.train_diffusion --epochs 250
python -m experiments.causal_mnist.sample_ndtm \
    --diffusion-ckpt $FFF_OUTPUT_ROOT/causal_mnist/checkpoints/ckpt_epoch_correlated_dist_250.pt \
    --erm-model $FFF_DATA_ROOT/causal_mnist/ERM.pt \
    --irm-model $FFF_DATA_ROOT/causal_mnist/IRM.pt
```

`train_classifiers.py` is the MLP, environments and IRMv1 penalty of Arjovsky et
al. (2019), with their published hyperparameters; ERM is the same run with the
penalty off. It prints its accuracies against theirs; check them before
sampling. An IRM test accuracy down near ERM's means the run collapsed onto the
colour-based solution, which would make Figures 9 and 15 meaningless.

## HTRU2 -- tabular experiment (rebuttal, Appendix B)

CPU-only, and slower than it looks: measured end to end on twelve CPU cores,
about 25 minutes -- 9 for the diffusion training and 75 seconds for each of the
twelve sampling runs below. On a GPU the whole thing is a few minutes; on a
login node budget half an hour, or send it to a compute node.

`prepare` needs outbound HTTP to `archive.ics.uci.edu`. Where a node has none it
now fails within 30 seconds and prints what to copy over, rather than waiting on
a connection that never opens.

```bash
python -m experiments.htru2.prepare
python -m experiments.htru2.train_subject_model
python -m experiments.htru2.train_diffusion
for g in 1.0 2.0 5.0 10.0; do
  python -m experiments.htru2.sample_ndtm --gamma $g --tag sweep       # add --seed 1, 2 for spread
done
python -m experiments.htru2.analyze
python -m experiments.htru2.make_figures
```

Reference values (single seed): fiber l2 mean 0.3580 / 0.1944 / 0.0705 / 0.0280
at gamma 1 / 2 / 5 / 10. Sweeping three seeds per gamma is what the figures'
error bars want; the means above are the `--seed 0` run alone, which is what
`analyze` reports by default -- compare against that rather than the three-seed
average.

### How exactly this reproduces, and what moves it

Measured on a 2080 Ti, torch 2.12+cu130, against the released checkpoints:

* **Same machine, released checkpoint, `--seed 0`: bit-exact.** Every metric in
  `analysis.json` -- fiber loss, median, prob-diff, consistency, per-feature
  movement, gradient importance -- matches the reference to the last decimal
  (max absolute deviation 0.0).
* **Retraining the diffusion model is deterministic.** `train_diffusion --seed 0`
  reproduces the released checkpoint's weights exactly (max |dw| = 0), so the
  documented pipeline reproduces the reference end to end. The diffusion *seed*
  barely matters either: seeds 1 and 2 give unrelated networks (relative weight
  distance 1.4) at the same fit quality (mean feature W1 0.0197 / 0.0206 /
  0.0197) and move the downstream numbers by 2-4%.
* **The sampling seed is what moves these numbers.** Same checkpoint, same
  machine, `sample_ndtm --seed 0/1/2`:

  | gamma | fiber l2 (seed 0 / 1 / 2) | consistency (seed 0 / 1 / 2) |
  |---|---|---|
  | 1  | 0.3580 / 0.3341 / 0.3432 | 0.1110 / 0.1214 / 0.1363 |
  | 2  | 0.1944 / 0.1892 / 0.1938 | 0.1542 / 0.1663 / 0.1787 |
  | 5  | 0.0705 / 0.0697 / 0.0725 | 0.1479 / 0.1517 / 0.1627 |
  | 10 | 0.0280 / 0.0283 / 0.0275 | 0.1410 / 0.1393 / 0.1462 |

  Consistency at gamma=1 spans 23% across three seeds. **Consistency is the
  seed-sensitive metric, and low gamma is the seed-sensitive end** -- weak
  guidance leaves the NDTM optimum loosely determined, while strong guidance
  pins it. This is why `analyze` reports one named seed and puts the spread in
  `across_seeds`, and why the reference values are `--seed 0` specifically:
  quoting a different seed as though it were the reference reads as a 23%
  regression that is not there.
* **Hardware matters less than the seed, but not nothing.** Sampling the same
  checkpoint with the same seed on CPU rather than GPU moves fiber loss at
  gamma=1 from 0.3580 to 0.3302 (-7.8%) and consistency from 0.1110 to 0.1253,
  with gamma=10 inside 5%. Expect small cross-machine differences on top of the
  seed spread above.

So: compare seed 0 against seed 0, metric by metric. A cross-machine run that
differs by a couple of percent at `--seed 0` has reproduced this experiment.

## Ablations -- Appendix C (rebuttal)

One-at-a-time sweeps around the tuned colorMNIST configuration, 3 seeds each:

```bash
python -m experiments.colormnist.ablations.run_kappa_tau_steps
python -m experiments.colormnist.ablations.run_eta
python -m experiments.colormnist.ablations.run_steps_gamma_matched

python -m experiments.colormnist.ablations.analyze_kappa_tau_steps
python -m experiments.colormnist.ablations.analyze_eta
python -m experiments.colormnist.ablations.analyze_steps_gamma_matched
```

Add `--dry-run` to any runner to print the job list without executing it.

## Re-running a setting that has already been sampled

Every sampling run writes a timestamped directory under its setting directory,
records the git revision it was produced by, and the evaluators pool whatever
they are given. Runs made before and after a change would otherwise be averaged
into one number -- and because each old shard finds a new partner, the result
reads as two independent draws and comes out looking exactly like the number you
wanted.

So the evaluators refuse to pool across revisions. Old runs can stay where they
are; you do not have to delete anything. Sample into a fresh directory:

```bash
python -m experiments.imagenet.sample_ndtm ... --out $FFF_OUTPUT_ROOT/imagenet/dinov2_imagenet_v2
```

or move the older ones aside, or pass `--allow-mixed-revisions` when comparing
them is the point. Runs written before revisions were recorded all report the
same (absent) revision, so a directory of purely historical runs still evaluates
without complaint.

## Assets

The GitHub release holds what we trained and what we derived: the colorMNIST
benchmark (`cc_mnist/`) with its VAE and subject model, the 21 conditional fiber
models, and the causal MNIST ERM/IRM pair. Unpack it under `$FFF_DATA_ROOT`.

The three CheXpert classifiers are on the Hugging Face Hub instead, since they
are the artifacts most likely to be wanted on their own:

| subject model | hub id | `$FFF_DATA_ROOT` directory |
|---|---|---|
| `biomedclip` | [`RussellALA/biomedclip-chexpert`](https://huggingface.co/RussellALA/biomedclip-chexpert) | `biomedclip-pretrained-larger-chexpert_384` |
| `convnext` | [`RussellALA/convnext-chexpert-seed1`](https://huggingface.co/RussellALA/convnext-chexpert-seed1) | `convnextv2-tiny-chexpert_384` |
| `convnext2` | [`RussellALA/convnext-chexpert-seed2`](https://huggingface.co/RussellALA/convnext-chexpert-seed2) | `convnextv2-tiny-chexpert_384_2` |

`SUBJECT_MODEL_PATHS` in `experiments/chexpert/sample_ndtm.py` resolves the
right-hand column against `$FFF_DATA_ROOT`, so download each into the directory
named there.

Four things are deliberately not in it:

- **The CheXpert images** are Stanford's, under a research use agreement that
  does not permit redistribution. Request them from the
  [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)
  and put them in `$FFF_DATA_ROOT/chexpert`.
- **The CheXpert diffusion model.** A generative model trained on CheXpert is
  close enough to the images themselves that we do not redistribute it either.
  `notebooks/chexpert_generator.ipynb` trains it -- `train_ct_ddpm` is the run
  that produced `diffusion-chexpert/epoch_10`, ten epochs at 384x384 -- and
  everything downstream of it then reproduces.
- **The cue conflict stimuli** are from
  [rgeirhos/texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape),
  built on ImageNet images whose permissions do not allow us to mirror them.
  Take `stimuli/style-transfer-preprocessed-512/` from that repository and point
  `$FFF_DATA_ROOT/cue_conflict` at it; the loader wants one directory per shape
  class, which is how it already ships.
- **The situs inversus radiographs** (Figure 10) are from Mayer et al. (2025).
  We have permission to show them in the paper's figures, not to redistribute
  them -- obtain them from the authors and place them in
  `$FFF_DATA_ROOT/rare_cases` as `situs_inversus_1.jpeg` ... `_4.jpeg`. Every
  other CheXpert result is independent of them.

ImageNet itself is not redistributable either; point `$FFF_DATA_ROOT/imagenet`
at a copy of the validation set. The ImageNet fiber sample sets are too large to
release and are regenerated with the commands above.

BiomedCLIP itself is not in the release -- `BiomedClipVisionEncoder` builds it
from `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` through
`open_clip`, and the fine-tuned head comes from the released
`biomedclip-pretrained-larger-chexpert_384/model.safetensors`.
