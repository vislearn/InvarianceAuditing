"""Train the conditional diffusion model for the causal MNIST experiment.

Paper Section 4.4 and Appendix B.3.2. The model is trained on the environment
where colour agrees with the label 80% of the time, which fixes p(x) for the
audit; query representations then come from the 10%-agreement environment. A
shift in the colour-label correlation of the fiber samples is what reveals that
the subject model encodes colour (Figures 9 and 15).

Not to be confused with the colorMNIST benchmark of Section 4.1, which is the
dataset of Rombach et al. (2020); this is the variant from Arjovsky et al. (2019).

    python -m experiments.causal_mnist.train_diffusion --epochs 250
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from functools import partial
from math import ceil
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, utils

from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import numpy as np
from torch.distributions import Normal, Uniform
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

# `ColoredMNIST` reproduces the environment construction of the reference
# implementation of Arjovsky et al. (2019), "Invariant Risk Minimization":
# https://github.com/facebookresearch/InvariantRiskMinimization
# (Copyright (c) Facebook, Inc. and its affiliates), licensed CC BY-NC 4.0 --
# https://creativecommons.org/licenses/by-nc/4.0/. Used under that licence,
# which is more restrictive than this repository's BSD 3-Clause.
class ColoredMNIST(Dataset):
    def __init__(self, images, labels, e, device="cpu"):
        super().__init__()
        self.device = device

        # Subsample (28x28 → 14x14)
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]

        # Binary labels
        labels = (labels < 5).float()

        # Helpers
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return (a - b).abs()

        # Flip labels with prob 0.25
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))

        # Assign colors with flip prob e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))

        # Build 2-channel images
        images = torch.stack([images, images], dim=1)

        # Zero out one channel depending on color
        idx = torch.arange(len(images))
        images[idx, (1 - colors).long(), :, :] *= 0

        self.images = normalize(images.float() / 255.).to(device)
        self.labels = labels.long()  # shape (B,)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def make_dataloaders(root: str, batch_size: int, num_workers: int):
    mnist = datasets.MNIST(root, train=True, download=True)
    
    images = mnist.data.clone()
    labels = mnist.targets.clone()
    
    # ✅ Split FIRST (deterministic)
    train_images, val_images = images[:50000], images[50000:]
    train_labels, val_labels = labels[:50000], labels[50000:]
    
    # ✅ Shuffle ONLY training data
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    # e is the probability that colour disagrees with the label. The paper uses
    # three environments: two for training, at 80% and 90% colour-label agreement
    # (e=0.2 and e=0.1), and one for testing where the correlation is reversed to
    # 10% agreement (e=0.9). The diffusion model is trained on e=0.2, which fixes
    # p(x) for the audit; sample_ndtm.py draws its queries from e=0.9.
    train_ds = ColoredMNIST(train_images[:40000], train_labels[:40000], e=0.2)
    test_ds = ColoredMNIST(train_images[40000:], train_labels[40000:], e=0.1)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def sample_grid(model, scheduler, device, n_steps=100, n_row=8, n_col=8, labels=None):
    model.eval()
    num_images = n_row * n_col
    labels = labels[:num_images]
    # start from random noise
    sample_shape = (num_images, 2, 14, 14)
    sample = torch.randn(sample_shape, device=device)
    # configure scheduler timesteps for sampling
    if n_steps is not None:
        scheduler.set_timesteps(n_steps)

    timesteps = scheduler.timesteps  # already in the right order

    with torch.no_grad():
        for t in tqdm(timesteps, desc=f"sampling ({len(timesteps)} steps)"):
            t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
            out = model(sample, t_batch, class_labels=labels).sample
            # scheduler step expects model_output (predicted noise) and returns a previous sample
            step = scheduler.step(model_output=out, timestep=t, sample=sample)
            sample = step.prev_sample
    # denormalize from [-1,1] to [0,1]
    sample = denormalize(sample.clamp(-1, 1))
    grid = utils.make_grid(sample, nrow=n_col)
    model.train()
    return grid

def normalize(x):
    if x.ndim == 4:
        x = x[:,:2]
    elif x.ndim == 3:
        x = x[:2]
    return x*2 - 1
    
def denormalize(x):
    x = (x+1)/2
    if x.ndim == 4:
        missing_channels = torch.zeros_like(x[:,:1])
        x = torch.cat((x, missing_channels), dim=1)
    elif x.ndim == 3:
        missing_channels = torch.zeros_like(x[:1])
        x = torch.cat((x, missing_channels), dim=0)
    return x
    
def sample_timesteps_cosine(batch_size, num_steps, device):
    # Cosine weighting over t in [0, 1]
    t = torch.rand(batch_size, device=device)
    # Convert uniform t to cosine-weighted schedule
    # The function below roughly inverts the cosine schedule from Improved DDPM
    alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    # Map continuous t to discrete indices
    timesteps = (alpha_bar * (num_steps - 1)).long().clamp(0, num_steps - 1)
    return timesteps

    
def save_sample_grid(grid, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = make_dataloaders(args.data_root, args.batch_size, args.num_workers)
    model = UNet2DModel(
        sample_size=14,
        in_channels=2,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=(64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        class_embed_type=None, 
        num_class_embeds=2,
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.e-5)
    # Cosine annealing with warmup
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.05,
        anneal_strategy='cos',
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device).squeeze()  # ensure (B,)
            batch_size = images.shape[0]

            noise = torch.randn_like(images)

            timesteps = sample_timesteps_cosine(batch_size, scheduler.num_train_timesteps, device)
            
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp and torch.cuda.is_available()):
                outputs = model(noisy_images, timesteps, class_labels=labels)
                pred_noise = outputs.sample
                loss = F.smooth_l1_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1
            if global_step % args.log_every == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': float(loss.detach().cpu()), 'lr': current_lr})

            if global_step % args.save_every == 0 and args.save_every != -1:
                ckpt_path = os.path.join(args.output_dir, f"checkpoints/ckpt_step_correlated_dist_{global_step}.pt")
                save_checkpoint(model, optimizer, epoch, ckpt_path)
                grid = sample_grid(model, scheduler, device, n_row=8, n_col=8, labels=labels)
                save_sample_grid(grid, os.path.join(args.output_dir, f"samples/sample_correlated_dist_{global_step}.png"))

        # Save end-of-epoch checkpoint and samples periodically
        if not (epoch % 10) or (epoch == args.epochs):
            save_checkpoint(model, optimizer, epoch, os.path.join(args.output_dir, f"checkpoints/ckpt_epoch_correlated_dist_{epoch}.pt"))
            grid = sample_grid(model, scheduler, device, n_row=8, n_col=8, labels=labels)
            save_sample_grid(grid, os.path.join(args.output_dir, f"samples/sample_epoch_correlated_dist_{epoch}.png"))

    print("Training finished.")



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=None,
                   help="defaults to $FFF_OUTPUT_ROOT/causal_mnist")
    p.add_argument("--data-root", default=None,
                   help="torchvision MNIST root; defaults to $FFF_DATA_ROOT")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    from experiments.common import paths

    args = parse_args()
    args.output_dir = args.output_dir or paths.output("causal_mnist")
    args.data_root = args.data_root or paths.data_root()
    train(args)
