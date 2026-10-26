"""Main entry point of the Mammoth library"""

from importlib.metadata import PackageNotFoundError, version

from .trainer import Trainer
from .utils import optimizers

# FIXME: what is the purpose of this hack?
# import sys
# mammoth.utils.optimizers.Optim = mammoth.utils.optimizers.Optimizer
# sys.modules["mammoth.Optim"] = mammoth.utils.optimizers

__all__ = ["optimizers", "Trainer"]

# Single source of truth is `version=` in setup.py.
try:
    __version__ = version("mammoth-nlp")
except PackageNotFoundError:  # running from a source tree that was never pip-installed
    __version__ = "unknown"
