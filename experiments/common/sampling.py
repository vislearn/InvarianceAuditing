"""Shared plumbing for the NDTM sampling scripts."""

import dataclasses
import glob
import json
import os
import subprocess
import uuid
from datetime import datetime

import torch


def run_directory(root: str, prefix: str) -> str:
    """A fresh directory for one sampling run.

    Runs are chunked and collected by prefix afterwards, so several jobs can
    sample the same setting in parallel without coordinating.
    """
    stamp = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    path = os.path.join(root, f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=False)
    return path


class ChunkWriter:
    """Accumulates batches and flushes them to `chunk_<n>.pt` files.

    Writing in chunks means a preempted job still leaves usable output, and the
    evaluation notebooks concatenate whatever chunks they find.
    """

    def __init__(self, directory: str, keys, chunk_size: int = 500):
        self.directory = directory
        self.keys = list(keys)
        self.chunk_size = chunk_size
        self.buffers = {k: [] for k in self.keys}
        self.n_buffered = 0
        self.n_chunks = 0

    def add(self, **batch):
        rows = {key: len(batch[key]) for key in self.keys}
        if len(set(rows.values())) > 1:
            # Check every field, not just keys[0]: a batch whose fields disagree
            # would otherwise be written anyway, leaving the chunk with a
            # different number of rows per field and every later pairing of an
            # original with its sample off by the difference.
            raise ValueError(
                f"the fields of this batch hold different numbers of rows "
                f"({rows}); they are supposed to line up row for row")
        for key in self.keys:
            self.buffers[key].append(batch[key].detach().cpu())
        self.n_buffered += next(iter(rows.values()))
        if self.n_buffered >= self.chunk_size:
            self.flush()

    def flush(self):
        if self.n_buffered == 0:
            return
        path = os.path.join(self.directory, f"chunk_{self.n_chunks}.pt")
        torch.save({k: torch.cat(v, dim=0) for k, v in self.buffers.items()}, path)
        self.buffers = {k: [] for k in self.keys}
        self.n_buffered = 0
        self.n_chunks += 1


def shard(dataset, index: int, count: int, order=None):
    """The slice of `dataset` this array task should sample.

    A full ImageNet setting is 10k fibers and does not finish in one job, so the
    work is split across tasks. Striding rather than blocking keeps every shard
    representative of the whole set, which matters because the runs are also
    reported per-shard. `order` shuffles first, for datasets whose file order is
    meaningful.
    """
    import torch
    if count <= 1 and order is None:
        return dataset
    if not 0 <= index < count:
        raise ValueError(f"shard {index} out of range for {count} shards")
    indices = torch.arange(len(dataset))
    if order is not None:
        # before the early return: a single unsharded job still has to see the
        # same permutation the sharded ones do
        indices = indices[torch.randperm(len(dataset), generator=order)]
    return torch.utils.data.Subset(dataset, indices[index::count].tolist())


def _plain(value):
    """A JSON-writable stand-in for anything a run configuration may hold."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value) and hasattr(value, "anchorpoints"):
        # the gamma schedule is a closure; store what it was built from
        return {"anchorpoints": _plain(value.anchorpoints),
                "max_timesteps": value.max_timesteps}
    return repr(value)


def _git_revision() -> str | None:
    """The commit the running code is at, or None outside a git checkout."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        out = subprocess.run(["git", "-C", repo, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def save_config(directory: str, config: dict, draw_args=None):
    """Record a run's settings as JSON.

    Not `torch.save`: the gamma schedule is a closure and pickling it fails, and
    a run configuration is more useful as something you can read.

    `draw_args` names the arguments that vary between independent draws of the
    same fibers, so the evaluators can tell a repeat from a different shard.

    The git revision goes in alongside them. Sampling runs accumulate in one
    setting directory across weeks and the evaluators pool whatever they find,
    so without this there is nothing to tell a run made before a change to the
    code from one made after it, and the two are silently averaged.
    """
    config = dict(config)
    config.setdefault("revision", _git_revision())
    if draw_args is not None:
        config.setdefault("draw_args", list(draw_args))
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(_plain(config), handle, indent=2)


def load_config(directory: str) -> dict | None:
    """The `config.json` a run wrote, or None for runs that predate it."""
    path = os.path.join(directory, "config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def revisions(directories) -> set:
    """The distinct code revisions a set of runs was produced by."""
    found = set()
    for directory in directories:
        config = load_config(directory) or {}
        found.add(config.get("revision"))
    return found


def check_revisions(directories, allow_mixed=False):
    """Refuse to pool runs that came from different code revisions.

    A warning is not enough here. Re-sampling a setting into the directory that
    already holds the superseded runs gives every shard a partner, so the
    evaluator reads them as two independent draws and reports a mean and a
    spread across two different versions of the code -- a number that looks
    exactly like the one you wanted. Stopping is the only safe default; pass
    --allow-mixed-revisions when you genuinely mean to compare.

    Runs written before revisions were recorded all report None, so a directory
    of purely historical runs still evaluates without complaint.
    """
    found = revisions(directories)
    if len(found) <= 1:
        return
    listed = ", ".join(sorted(str(r) for r in found))
    message = (f"these runs were made by different code revisions ({listed}), and "
               f"pooling them would average results from before and after a change.")
    if not allow_mixed:
        raise SystemExit(
            message + "\nSample into a fresh directory, move the older runs "
            "aside, or pass --allow-mixed-revisions if you mean to compare them.")
    print(f"WARNING: {message} Continuing because --allow-mixed-revisions was "
          f"given.\n")


def run_directories(paths):
    """Resolve what the user pointed at into a list of chunk-holding runs.

    A path that holds `chunk_*.pt` is a run. A path that holds run directories --
    which is what `--out some/setting` produces, one per array task -- expands to
    those. This means both `evaluate .../setting` and `evaluate .../setting/*`
    work, rather than the first failing with "no chunks".
    """
    def is_run(path):
        return bool(glob.glob(os.path.join(path, "chunk_*.pt")))

    resolved, empty, seen = [], [], set()

    def take(path):
        # `evaluate setting setting/*` names the same shard twice. Pooling it
        # twice halves the apparent spread and doubles the apparent fiber count,
        # and group_draws reads it as two draws of a set sampled once.
        real = os.path.realpath(path)
        if real in seen:
            return
        seen.add(real)
        resolved.append(path)

    for path in paths:
        if is_run(path):
            take(path)
            continue
        if os.path.isdir(path):
            children = sorted(c for c in glob.glob(os.path.join(path, "*"))
                              if is_run(c))
            if children:
                for child in children:
                    take(child)
                continue
        empty.append(path)
    if not resolved:
        listed = "\n  ".join(empty)
        raise SystemExit(f"no sampling runs under:\n  {listed}")
    for path in empty:
        print(f"skipping {os.path.basename(os.path.normpath(path))}: no chunks")
    return resolved


# Arguments that never change which fibers a run audits or what it computes:
# where the output goes, how it is batched, where the inputs were read from.
INCIDENTAL = ("out", "data_root", "base_model", "diffusion_model", "batch_size",
              "chunk_size", "num_workers", "device")


def draw_arguments(args) -> tuple:
    """Which recorded arguments distinguish two draws of the *same* fibers.

    A sampler states this itself by passing `draw_args` to `save_config`. For
    runs that predate that, the rule below matches both samplers as written:
    `sample_ndtm.py` for CheXpert takes `--seed` to fix which images are audited
    and `--sample-seed` to fix the noise, while the ImageNet and Qwen samplers
    have no `--sample-seed` and use `--seed` for the noise alone.
    """
    if "sample_seed" in args:
        return ("sample_seed",)
    return ("seed",)


def fiber_identity(directory, fallback=None):
    """What identifies the fibers a run audited, or `fallback` if it cannot be told.

    Read from the arguments the run recorded: everything except the draw
    arguments and the incidental ones. Callers pass `fallback` for runs written
    before `config.json` existed -- historically the target representations'
    exact bytes, which put two draws of the same images in different groups
    whenever they happened to run on different GPU models.
    """
    config = load_config(directory) or {}
    args = config.get("args")
    if not args:
        return fallback
    skip = set(INCIDENTAL) | set(config.get("draw_args") or draw_arguments(args))
    return json.dumps({k: v for k, v in sorted(args.items()) if k not in skip},
                      sort_keys=True, default=str)


# Fields that name a different *experiment* rather than a different piece of one.
# Shards of a setting differ in `shard`; two settings differ in one of these, and
# pooling them produces a number that describes neither.
# `num_steps` is here because the guidance schedule is a function of the
# diffusion timestep: the same gamma anchors run at 100 and at 200 steps put a
# different number of corrections inside every interval, so the two are different
# experiments however alike their configs look. The Qwen row was drawn at 100 and
# redrawn at 200, and nothing but the revision check stood between them.
SETTING_FIELDS = ("dataset", "subject_model", "subject_models", "queries", "split",
                  "static_target", "num_steps")


def check_one_setting(directories):
    """Raise if the runs are from different experiments, not shards of one."""
    seen = {}
    for directory in directories:
        args = (load_config(directory) or {}).get("args") or {}
        key = tuple((f, json.dumps(args[f], default=str))
                    for f in SETTING_FIELDS if f in args)
        seen.setdefault(key, []).append(os.path.basename(os.path.normpath(directory)))
    if len(seen) > 1:
        described = "\n  ".join(
            ", ".join(f"{f}={v}" for f, v in key) + f"  ({len(runs)} runs)"
            for key, runs in seen.items())
        raise SystemExit(
            "these runs are different experiments and pooling them would describe "
            f"neither:\n  {described}\nEvaluate them one setting at a time.")


def group_draws(items, identity):
    """Split `(directory, run)` pairs into sample sets.

    Returns `(sets, draws)`, where `sets[i]` is one complete sample set -- every
    shard of the setting, once -- and `draws` is how many such sets there are.
    Uneven groups mean an interrupted repeat, with no draw index every shard
    has; those fall back to a single pooled set, which is also the single-pass
    case.
    """
    groups = {}
    for item in items:
        groups.setdefault(identity(item), []).append(item)
    ordered = [groups[k] for k in sorted(groups, key=str)]
    sizes = {len(g) for g in ordered}
    if len(sizes) > 1 or not ordered:
        return [list(items)], 1
    draws = sizes.pop()
    if draws == 1:
        return [list(items)], 1
    return [[group[d] for group in ordered] for d in range(draws)], draws
