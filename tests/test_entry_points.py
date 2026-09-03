"""That every experiment script can actually be run, not merely imported.

`test_imports.py` checks that every module imports, which is necessary and not
sufficient: a script whose `__main__` block never calls anything imports fine
and exits 0 having done nothing, and a command line that names an option the
parser does not have fails only when it is run.
"""

import ast
import functools
import glob
import os
import re
import subprocess
import sys

import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every check here spawns a subprocess that imports torch, ~1.5 s each across 196
# cases -- about 290 s, which is 85% of the whole suite. Marked so the edit/verify
# loop can run `pytest -m "not slow"` in ~40 s; the default run still includes it.
pytestmark = pytest.mark.slow


def experiment_scripts():
    paths = sorted(glob.glob(os.path.join(REPO, "experiments", "**", "*.py"),
                             recursive=True))
    return [p for p in paths if "__pycache__" not in p]


def main_block(tree):
    """The body of `if __name__ == "__main__":`, or None."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name) and test.left.id == "__name__"):
            return node.body
    return None


@pytest.mark.parametrize("path", experiment_scripts(),
                         ids=lambda p: os.path.relpath(p, REPO))
def test_main_block_calls_something(path):
    """A __main__ block that only parses arguments is a script that does nothing.

    `causal_mnist/train_diffusion.py` set up its arguments and ended without
    calling `train(args)`. It exited 0, so `set -e` in the job script did not
    catch it either -- Section 4.4 failed two commands later, on the checkpoint
    that was never written.
    """
    with open(path) as handle:
        tree = ast.parse(handle.read(), path)
    body = main_block(tree)
    if body is None:
        pytest.skip("not a script")

    def calls(nodes):
        return any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   for n in nodes) or any(
            calls(child.body) for child in nodes
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)))

    assert calls(body), (
        f"the __main__ block of {os.path.relpath(path, REPO)} never calls "
        f"anything, so running it as a script does nothing and exits 0")


def documented_commands():
    """Every `python -m experiments...` command line REPRODUCING.md gives."""
    with open(os.path.join(REPO, "REPRODUCING.md")) as handle:
        text = handle.read()
    # join continuation lines so a wrapped command is one string
    text = re.sub(r"\\\n\s*", " ", text)
    found = []
    for line in text.splitlines():
        line = re.sub(r"\s+#.*$", "", line.strip())   # drop trailing shell comments
        match = re.match(r"^python -m (experiments\.[\w.]+)", line)
        if match:
            found.append((match.group(1), line))
    assert found, "no commands found in REPRODUCING.md -- has its format changed?"
    return sorted(set(found))


@functools.lru_cache(maxsize=None)
def help_text(module):
    """`--help` for a module, with nothing else on the command line."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=REPO, capture_output=True,
        text=True, timeout=300,
        env={**os.environ, "FFF_DOWNLOAD_DATASETS": "0"})
    return result.stdout


def stand_in(flag, module):
    """A value the parser will accept for `flag`.

    A shell variable has no value here, and dropping the token would leave its
    flag without an argument -- which fails for a different reason than the one
    under test. "1" covers `type=float` and `type=int` and reads as a path
    fragment, but not a flag with `choices=`, so those are read off the help
    text rather than guessed. Getting this wrong reports a documented command
    as broken when only the stand-in was.
    """
    if flag is not None:
        choices = re.search(re.escape(flag) + r"[= ]\{([^}]*)\}", help_text(module))
        if choices:
            return choices.group(1).split(",")[0]
    return "1"


@pytest.mark.parametrize("module,line", documented_commands(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_documented_command_parses(module, line):
    """The command lines in REPRODUCING.md must at least reach their parser.

    `make_figures` required a positional `phase` that neither REPRODUCING.md nor
    the run book passed, so Figure 4, Figure 14 and Table 3 died on an argparse
    error. --help exercises the parser without touching data or a GPU.
    """
    args = line.split()[3:]  # drop "python -m <module>"
    filled = []
    for i, arg in enumerate(args):
        if re.search(r"\$\{?\w+\}?", arg):
            previous = args[i - 1] if i and args[i - 1].startswith("--") else None
            arg = re.sub(r"\$\{?\w+\}?", stand_in(previous, module), arg)
        filled.append(arg)
    result = subprocess.run(
        [sys.executable, "-m", module, *filled, "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
        env={**os.environ, "FFF_DOWNLOAD_DATASETS": "0"})
    assert result.returncode == 0, (
        f"`{line}` does not parse:\n{result.stderr[-1500:]}")


@pytest.mark.parametrize("path", experiment_scripts() + sorted(
    p for p in glob.glob(os.path.join(REPO, "fff", "**", "*.py"), recursive=True)
    if "__pycache__" not in p), ids=lambda p: os.path.relpath(p, REPO))
def test_no_bare_except(path):
    """A bare `except:` hides the error that actually happened.

    `SubjectModel.encode` caught everything from `self.model.encode(x)` and fell
    through to `self.model(x)`, so a conditional model called without its
    condition reported a missing argument to `forward()` -- pointing away from
    the call that failed. That cost a 21-job array.
    """
    with open(path) as handle:
        tree = ast.parse(handle.read(), path)
    bare = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.ExceptHandler) and n.type is None]
    assert not bare, (
        f"{os.path.relpath(path, REPO)} has a bare `except:` at line(s) "
        f"{bare}; name the exceptions you mean to handle")


@pytest.mark.parametrize("path", sorted(
    glob.glob(os.path.join(REPO, "configs", "colormnist", "fiber_models", "*.yaml"))),
    ids=lambda p: os.path.basename(p))
def test_colormnist_configs_pass_an_empty_condition(path):
    """The colorMNIST subject model is unconditional and must be told so.

    Its checkpoint records `cond_dim: 0`, and `SubjectModel` only supplies the
    zero-width condition tensor its `encode` needs when `sm_empty_condition` is
    set. Without it, validation raises as soon as `ae_rec_fiber_loss` is
    computed -- which is what killed 19 of these 21 runs.
    """
    config = yaml.safe_load(open(path))
    assert config.get("sm_empty_condition") is True, (
        f"{os.path.basename(path)} loads a subject model without "
        f"sm_empty_condition: true")


def modules_with_schedule_tensors():
    """Every nn.Module in fff/ that precomputes tensors in __init__."""
    import fff.model.diffusion
    return [fff.model.diffusion.DiffusionModel]


@pytest.mark.parametrize("cls", modules_with_schedule_tensors(),
                         ids=lambda c: c.__name__)
def test_precomputed_tensors_are_buffers(cls):
    """Precomputed constants must be buffers, or they stay on the CPU.

    A plain tensor attribute does not follow its module across `.to(device)`.
    DiffusionModel set its beta/alpha schedule that way, so on a GPU the
    schedule stayed behind while the timesteps moved, and the first validation
    batch died in `alpha_cumprod[t]` with "indices should be either on cpu or
    on the same device as the indexed tensor". On one device the bug is
    invisible, which is exactly why a CPU-only check passed all 21 colorMNIST
    configs and the GPU array still failed.

    This test needs no GPU: it asserts the tensors are registered, which is the
    property that makes them move.
    """
    model = cls(dict(data_dim=8, cond_dim=0, layers_spec=[[32, 32]]))
    registered = dict(model.named_buffers())
    stray = [name for name, value in vars(model).items()
             if isinstance(value, torch.Tensor) and name not in registered]
    assert not stray, (
        f"{cls.__name__} holds tensor attribute(s) {stray} that are not "
        f"registered buffers, so .to(device) leaves them behind")
    assert registered, f"{cls.__name__} registered no buffers at all"

    # persistent=False keeps them out of state_dict: they are derived from
    # hparams, and adding keys would break every checkpoint that predates this.
    keys = model.state_dict().keys()
    leaked = [n for n in registered if n in keys]
    assert not leaked, (
        f"{cls.__name__} buffers {leaked} entered state_dict; pass "
        f"persistent=False so old checkpoints still load")
