"""Every experiment module must import.

The scripts under experiments/ were extracted from notebooks and standalone files
that lived in one flat directory and imported each other by bare module name
(`from train_htru2_subject_model import ...`). Those imports break once the files
become a package, and they break late: at module level they surface as an
ImportError on the first run, but inside a function body they survive an import
check and only fail partway through a job, and one of them sat inside a
try/except that degraded silently. Importing every module here catches the first
kind; the grep in test_no_flat_imports catches the rest.
"""

import importlib
import os
import pkgutil
import re
import subprocess

import pytest

import experiments


def experiment_modules():
    return sorted(
        m.name
        for m in pkgutil.walk_packages(experiments.__path__, "experiments.")
        if not m.ispkg
    )


@pytest.mark.parametrize("module", experiment_modules())
def test_module_imports(module):
    importlib.import_module(module)


def test_no_flat_imports():
    """No module may import a sibling script by bare name, at any indentation."""
    import pathlib
    import re

    root = pathlib.Path(experiments.__path__[0])
    pattern = re.compile(
        r"^\s*(?:from|import) (train_|sample_|prepare_|analyze_|make_|compute_)\w*",
        re.MULTILINE,
    )
    offenders = [
        f"{path.relative_to(root)}: {match.group().strip()}"
        for path in root.rglob("*.py")
        for match in pattern.finditer(path.read_text())
    ]
    assert not offenders, "flat cross-script imports: " + "; ".join(offenders)


def test_no_source_file_is_gitignored():
    """Every source file must actually be in the repository.

    A bare `data/` in .gitignore matches at every depth, so the fff.data package
    was excluded while the working tree looked complete and these tests passed.
    Only a clone showed it. This asks git directly instead.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracked = set(subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True,
        check=True).stdout.split())

    missing = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", ".pytest_cache", "outputs", "data"}]
        for name in filenames:
            if not name.endswith((".py", ".yaml", ".yml", ".ipynb")):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if rel not in tracked:
                missing.append(rel)
    assert not missing, ("source files absent from the repository, so no clone "
                         f"has them: {sorted(missing)}")
