""" Main entry point of the Mammoth library """
import mammoth.inputters
import mammoth.models
import mammoth.utils
import mammoth.modules
import mammoth.opts
from mammoth.trainer import Trainer
import sys
import mammoth.utils.optimizers

mammoth.utils.optimizers.Optim = mammoth.utils.optimizers.Optimizer
sys.modules["mammoth.Optim"] = mammoth.utils.optimizers

__all__ = [
    mammoth.inputters,
    mammoth.models,
    mammoth.utils,
    mammoth.modules,
    mammoth.opts,
    "Trainer"
]

__version__ = "2.2.0"
