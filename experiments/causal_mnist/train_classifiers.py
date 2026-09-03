"""Train the ERM and IRM classifiers audited in Section 4.4.

Follows the public implementation of Arjovsky et al. (2019) [1]: the same MLP,
the same three environments, and the same IRMv1 gradient-norm penalty with its
published hyperparameters. ERM is the identical run with the penalty switched
off, which is how the paper's baseline is defined.

Colour agrees with the label 80% and 90% of the time in the two training
environments and only 10% of the time at test, so colour is the stronger
predictor in training but reverses at test. ERM leans on it and collapses below
chance; IRM does not. Those two behaviours are what invariance auditing has to
detect, so check the printed test accuracies before sampling: an "ERM" that
generalises, or an "IRM" that does not, makes Figures 9 and 15 meaningless.

Models are saved as whole pickled modules, matching how the released weights were
stored and how CausalMNISTSubjectModel loads them.

    python -m experiments.causal_mnist.train_classifiers

[1] https://github.com/facebookresearch/InvariantRiskMinimization
"""

# Derived from the reference implementation of Arjovsky et al. (2019),
# "Invariant Risk Minimization": https://github.com/facebookresearch/InvariantRiskMinimization
# (Copyright (c) Facebook, Inc. and its affiliates), which is licensed
# CC BY-NC 4.0 -- https://creativecommons.org/licenses/by-nc/4.0/
#
# The MLP, the environment construction, `mean_nll`, `mean_accuracy` and the
# IRMv1 penalty follow that implementation closely; the training loop and the
# ERM/IRM split around it are ours.
# Those parts are used here under that licence, which is more restrictive than
# this repository's BSD 3-Clause: they are for non-commercial use.

import argparse
import os
import sys

import torch
from torch import autograd, nn
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import paths
from experiments.causal_mnist.subject_models import MLP, denormalize
from experiments.causal_mnist.train_diffusion import ColoredMNIST

# The published hyperparameters from the reference implementation's README.
DEFAULTS = dict(
    hidden_dim=390,
    l2_regularizer_weight=0.00110794568,
    lr=0.0004898536566546834,
    penalty_anneal_iters=190,
    penalty_weight=91257.18613115903,
    steps=501,
)


def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def mean_accuracy(logits, y):
    return (((logits > 0.0).float() - y).abs() < 1e-2).float().mean()


def irm_penalty(logits, y):
    """IRMv1: squared gradient of the risk w.r.t. a dummy unit scaling."""
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


def build_environments(data_root, device, seed=0):
    """The two training environments and the reversed test environment."""
    mnist = datasets.MNIST(data_root, train=True, download=True)
    images, labels = mnist.data.clone(), mnist.targets.clone()
    # The reference implementation carves the two training environments out of
    # the first 50k only, interleaved, and holds out the last 10k as the reversed
    # test environment. Interleaving the full 60k instead leaks the test split
    # into training and leaves IRM barely better than ERM.
    train_images, train_labels = images[:50000], labels[:50000]
    # Shuffling before the interleave is not cosmetic. MNIST is not stored in a
    # random order, so the even and odd halves differ in more than the colour
    # correlation, and IRMv1 can then drive its penalty to zero using that other
    # difference while still classifying by colour: every run collapses onto the
    # ERM solution (~0.84 train / ~0.19 test). With the shuffle it reaches
    # Arjovsky et al.'s 0.71 / 0.66.
    order = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(seed))
    train_images, train_labels = train_images[order], train_labels[order]
    specs = [(train_images[::2], train_labels[::2], 0.2),
             (train_images[1::2], train_labels[1::2], 0.1),
             (images[50000:], labels[50000:], 0.9)]
    envs = []
    for env_images, env_labels, e in specs:
        ds = ColoredMNIST(env_images, env_labels, e=e, device=device)
        envs.append({
            # ColoredMNIST stores images in [-1, 1] for the diffusion model;
            # denormalize is exactly what the subject model applies at inference
            "images": denormalize(ds.images),
            "labels": ds.labels.float()[:, None].to(device),
        })
    return envs


def train(method, envs, args, device, seed):
    torch.manual_seed(seed)
    mlp = MLP(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    use_penalty = method == "irm"

    for step in range(args.steps):
        for env in envs:
            logits = mlp(env["images"])
            env["nll"] = mean_nll(logits, env["labels"])
            env["acc"] = mean_accuracy(logits, env["labels"])
            env["penalty"] = irm_penalty(logits, env["labels"])

        train_nll = torch.stack([envs[0]["nll"], envs[1]["nll"]]).mean()
        train_acc = torch.stack([envs[0]["acc"], envs[1]["acc"]]).mean()
        train_penalty = torch.stack([envs[0]["penalty"], envs[1]["penalty"]]).mean()

        weight_norm = sum(w.norm().pow(2) for w in mlp.parameters())
        loss = train_nll + args.l2_regularizer_weight * weight_norm
        penalty_weight = (args.penalty_weight
                          if use_penalty and step >= args.penalty_anneal_iters else 1.0)
        loss = loss + penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # rescale so the total loss stays in a sane range once the penalty kicks in
            loss = loss / penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == args.steps - 1:
            print(f"  [{method}] step {step:4d}  train acc {train_acc:.3f}  "
                  f"test acc {envs[2]['acc']:.3f}  penalty {train_penalty.item():.2e}")

    return mlp, float(train_acc), float(envs[2]["acc"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None, help="torchvision MNIST root")
    p.add_argument("--out", default=None, help="where ERM.pt and IRM.pt are written")
    p.add_argument("--seed", type=int, default=0)
    for name, value in DEFAULTS.items():
        p.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = args.data_root or paths.data("mnist")
    out = args.out or paths.data("causal_mnist")
    os.makedirs(out, exist_ok=True)

    envs = build_environments(data_root, device, seed=args.seed)
    print(f"environments: {[len(e['labels']) for e in envs]} examples "
          f"(colour agrees 80% / 90% / 10%)")

    results = {}
    for method in ("erm", "irm"):
        model, train_acc, test_acc = train(method, envs, args, device, args.seed)
        path = os.path.join(out, f"{method.upper()}.pt")
        torch.save(model, path)          # whole module, as the released weights were
        results[method] = (train_acc, test_acc)
        print(f"saved {path}")

    print("\n           train acc   test acc     Arjovsky et al., Table 1")
    reference = {"erm": (0.864, 0.140), "irm": (0.708, 0.669)}
    for method, (train_acc, test_acc) in results.items():
        ref_train, ref_test = reference[method]
        print(f"  {method.upper():4s}     {train_acc:.3f}       {test_acc:.3f}"
              f"        {ref_train:.3f}       {ref_test:.3f}")
    print("\nAn IRM test accuracy down near ERM's means the run collapsed onto the\n"
          "colour-based solution, which would make Figures 9 and 15 meaningless.")


if __name__ == "__main__":
    main()
