"""Where experiment inputs and outputs live.

Nothing in this repository hard-codes a machine-specific path. Two environment
variables cover everything:

    FFF_DATA_ROOT     datasets and released checkpoints   (default: ./data)
    FFF_OUTPUT_ROOT   samples, figures and statistics     (default: ./outputs)

Sample sets are large -- a full colorMNIST setting is about 1 GB -- so point
FFF_OUTPUT_ROOT at a disk with room.
"""

import os


def data_root() -> str:
    return os.environ.get("FFF_DATA_ROOT", "data")


def output_root() -> str:
    return os.environ.get("FFF_OUTPUT_ROOT", "outputs")


def data(*parts: str) -> str:
    return os.path.join(data_root(), *parts)


def output(*parts: str, create: bool = True) -> str:
    """A path under FFF_OUTPUT_ROOT, created unless `create=False`.

    Pass `create=False` for a path named at module level -- a constant, or an
    argparse default. Those run on import, so creating there scatters empty
    directories anywhere the module is merely imported, including under pytest.
    Create at the point of writing instead.
    """
    path = os.path.join(output_root(), *parts)
    if create:
        os.makedirs(path, exist_ok=True)
    return path
