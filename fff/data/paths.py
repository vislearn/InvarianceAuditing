"""Resolution of dataset and checkpoint paths.

Configs and released checkpoints record data roots relative to the repository
(`data/cc_mnist`, `data`). Set FFF_DATA_ROOT to point those at wherever the data
actually lives; absolute paths are left untouched.
"""

import os


def data_root() -> str:
    return os.environ.get("FFF_DATA_ROOT", "data")


def resolve(path: str | None) -> str | None:
    """Resolve a recorded data path against FFF_DATA_ROOT."""
    if path is None or os.path.isabs(path):
        return path
    root = data_root()
    # Recorded paths are relative to the repository, whose data directory is
    # "data/"; strip that prefix so FFF_DATA_ROOT replaces it.
    for prefix in ("data/", "./data/"):
        if path.startswith(prefix):
            return os.path.join(root, path[len(prefix):])
    if path in ("data", "./data"):
        return root
    return path
