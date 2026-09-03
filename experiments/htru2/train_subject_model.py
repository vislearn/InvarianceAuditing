"""Train the HTRU2 subject model: a small MLP pulsar classifier.

The subject model phi maps a standardized 8-dim feature vector to class logits h in
R^2. Its fiber phi^{-1}(h) is the set of candidates the classifier finds indistinguish-
able from a query. This is the tabular counterpart of the CheXpert / ImageNet subject
models (paper Sec. 4.5 / B.4): as there, the fiber loss is measured on the classifier
logits.

The .encode() method returns logits and is the interface consumed by the NDTM sampler
(fff.ndtm.NDTM calls subject_model(x) at each Tweedie estimate). The network is small
and fully differentiable so gradient-based guidance is cheap.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import paths  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


class HTRU2SubjectModel(nn.Module):
    """MLP classifier over standardized HTRU2 features. forward/encode -> logits (B, 2)."""

    def __init__(self, in_dim=8, hidden=128, n_hidden=2, n_classes=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x))

    def encode(self, x):
        """Subject-model representation used as the fiber target h (class logits)."""
        return self.forward(x)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X.to(device))
    prob = logits.softmax(-1)[:, 1].cpu().numpy()
    pred = logits.argmax(-1).cpu().numpy()
    yt = y.cpu().numpy()
    acc = float((pred == yt).mean())
    # AUROC via rank statistic (no sklearn dependency in the hot path)
    order = np.argsort(prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(prob) + 1)
    n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())
    auroc = (ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return acc, float(auroc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=paths.data("htru2", "htru2.npz"))
    parser.add_argument("--output-dir",
                        default=paths.output("htru2", "subject_model", create=False))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X_train = torch.from_numpy(d["X_train"]).float()
    y_train = torch.from_numpy(d["y_train"]).long()
    X_test = torch.from_numpy(d["X_test"]).float()
    y_test = torch.from_numpy(d["y_test"]).long()

    # class weights counter the ~9% pulsar imbalance
    counts = torch.bincount(y_train, minlength=2).float()
    class_w = (counts.sum() / (2 * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    model = HTRU2SubjectModel(in_dim=X_train.shape[1], hidden=args.hidden,
                              n_hidden=args.n_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n = X_train.shape[0]
    Xtr, ytr = X_train.to(device), y_train.to(device)
    best_auroc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = criterion(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 20 == 0 or epoch == 1:
            acc, auroc = evaluate(model, X_test, y_test)
            print(f"epoch {epoch:4d} | test acc {acc:.4f} | test AUROC {auroc:.4f}", flush=True)
            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    acc, auroc = evaluate(model, X_test, y_test)
    if auroc > best_auroc:
        best_auroc, best_state = auroc, {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}
    print(f"FINAL best test AUROC {best_auroc:.4f} (last-epoch acc {acc:.4f})")

    ckpt_path = os.path.join(args.output_dir, "subject_model.pt")
    torch.save({
        "model_state_dict": best_state,
        "model_config": {"in_dim": int(X_train.shape[1]), "hidden": args.hidden,
                         "n_hidden": args.n_hidden, "n_classes": 2},
        "feat_mean": d["feat_mean"], "feat_std": d["feat_std"],
        "feature_names": d["feature_names"],
        "test_auroc": best_auroc,
    }, ckpt_path)
    print("saved", ckpt_path)


if __name__ == "__main__":
    main()
