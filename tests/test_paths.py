"""Where the code looks for data and puts output.

Two environment variables are the whole story (see experiments/common/paths.py),
and released checkpoints carry paths recorded relative to the repository that
have to be rewritten against them. A mistake here is a FileNotFoundError on
someone else's machine and nowhere else.
"""

import os

import pytest

from experiments.common import paths as experiment_paths
from fff.data import paths as data_paths


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FFF_DATA_ROOT", str(tmp_path))
    return str(tmp_path)


def test_a_recorded_relative_path_is_rewritten_against_the_root(data_root):
    assert data_paths.resolve("data/cc_mnist") == os.path.join(data_root, "cc_mnist")
    assert data_paths.resolve("./data/cc_mnist") == os.path.join(data_root, "cc_mnist")
    assert data_paths.resolve("data") == data_root
    assert data_paths.resolve("./data") == data_root


def test_an_absolute_path_is_left_alone(data_root):
    assert data_paths.resolve("/mnt/scratch/cc_mnist") == "/mnt/scratch/cc_mnist"


def test_none_resolves_to_none(data_root):
    assert data_paths.resolve(None) is None


def test_a_directory_merely_starting_with_data_is_not_a_data_root(data_root):
    """"database/" is not "data/"."""
    assert data_paths.resolve("database/things") == "database/things"


def test_the_default_root_is_the_repository_data_directory(monkeypatch):
    monkeypatch.delenv("FFF_DATA_ROOT", raising=False)
    assert data_paths.data_root() == "data"
    assert experiment_paths.data_root() == "data"


def test_the_two_data_roots_agree(data_root):
    """fff.data.paths and experiments.common.paths read the same variable.

    They are separate implementations of the same rule; if one of them ever
    reads a different variable, the sampler and the model would disagree about
    where the dataset is.
    """
    assert data_paths.data_root() == experiment_paths.data_root()


def test_output_root_is_separate_from_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FFF_DATA_ROOT", str(tmp_path / "in"))
    monkeypatch.setenv("FFF_OUTPUT_ROOT", str(tmp_path / "out"))
    assert experiment_paths.output("colormnist").startswith(str(tmp_path / "out"))
    assert experiment_paths.data("cc_mnist").startswith(str(tmp_path / "in"))


def test_output_creates_the_directory_unless_told_not_to(tmp_path, monkeypatch):
    monkeypatch.setenv("FFF_OUTPUT_ROOT", str(tmp_path))
    made = experiment_paths.output("a", "b")
    assert os.path.isdir(made)
    assert not os.path.exists(experiment_paths.output("c", create=False))


def test_importing_an_experiment_module_writes_nothing(tmp_path, monkeypatch):
    import subprocess
    import sys

    env = {**os.environ, "FFF_OUTPUT_ROOT": str(tmp_path / "out"),
           "FFF_DATA_ROOT": str(tmp_path / "in"), "TQDM_DISABLE": "1"}
    subprocess.run(
        [sys.executable, "-c",
         "import experiments.colormnist.compute_statistics"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, check=True, capture_output=True)
    assert not os.path.exists(tmp_path / "out")
