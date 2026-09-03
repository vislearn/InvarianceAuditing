"""The ERM and IRM classifiers audited on causal MNIST (paper Section 4.4).

Both are the two-hidden-layer MLP of Arjovsky et al. (2019), trained with their
public implementation [1]; ERM leans on the spurious colour-label correlation and
IRM does not, which is what invariance auditing is asked to detect.

The released weights are whole pickled modules, so the MLP class has to be
importable under the same name to unpickle them.

[1] https://github.com/facebookresearch/InvariantRiskMinimization
"""

import torch
import torch.nn as nn

HIDDEN_DIM = 390  # the default in the reference implementation


# `MLP` below is taken from the reference implementation of Arjovsky et al.
# (2019), "Invariant Risk Minimization":
# https://github.com/facebookresearch/InvariantRiskMinimization
# (Copyright (c) Facebook, Inc. and its affiliates), licensed CC BY-NC 4.0 --
# https://creativecommons.org/licenses/by-nc/4.0/. It is kept verbatim because
# the released ERM.pt / IRM.pt are pickled instances of it. Used under that
# licence, which is more restrictive than this repository's BSD 3-Clause.
class MLP(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super(MLP, self).__init__()
        lin1 = nn.Linear(2 * 14 * 14, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        out = input[:, :2].view(input.shape[0], 2 * 14 * 14)
        return self._main(out)


def denormalize(x):
    """Undo normalize(): back to [0, 1] and re-append the empty third channel."""
    x = (x + 1) / 2
    empty = torch.zeros_like(x[:, :1] if x.ndim == 4 else x[:1])
    return torch.cat((x, empty), dim=1 if x.ndim == 4 else 0)


def normalize(x):
    """Two-channel image in [0, 1] to the diffusion model's [-1, 1] range."""
    return (x[:, :2] if x.ndim == 4 else x[:2]) * 2 - 1


class CausalMNISTSubjectModel(nn.Module):
    """Wraps a pickled ERM or IRM classifier; phi(x) is the single class logit."""

    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.model = torch.load(model_path, map_location=device, weights_only=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(denormalize(x))
