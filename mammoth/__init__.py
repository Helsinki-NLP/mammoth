""" Main entry point of the Mammoth library """
from mammoth.trainer import Trainer
from mammoth.utils import optimizers

# FIXME: what is the purpose of this hack?
# import sys
# mammoth.utils.optimizers.Optim = mammoth.utils.optimizers.Optimizer
# sys.modules["mammoth.Optim"] = mammoth.utils.optimizers

__all__ = [
    "optimizers",
    "Trainer"
]

__version__ = "2.2.0"
