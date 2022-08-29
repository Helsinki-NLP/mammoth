""" Main entry point of the ONMT library """
import onmt.inputters_mvp
import onmt.encoders
import onmt.decoders
import onmt.models
import onmt.utils
import onmt.modules
import onmt.opts
from onmt.trainer import Trainer
import sys
import onmt.utils.optimizers

onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

__all__ = [
    onmt.inputters_mvp,
    onmt.encoders,
    onmt.decoders,
    onmt.models,
    onmt.utils,
    onmt.modules,
    onmt.opts,
    "Trainer"
]

__version__ = "2.2.0"
