"""Resuming with reset_optim=none when some components are frozen.

Scenario: a model was trained with encoder stacks "frozen" and "trained", so the checkpoint has
an optimizer-state file for both. The new run freezes "frozen", which therefore gets no
suboptimizer. Loading must restore both components' weights and "trained"'s optimizer state,
and skip "frozen"'s optimizer state instead of failing with KeyError.
"""
import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mammoth.utils.model_saver import load_parameters_from_checkpoint

PREFIX = "model_step_100"


class _Component:
    """Duck-typed stand-in for a DistributedComponent: one named module in the model."""

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def load_state_dict(self, model, state_dict):
        return model[self.name].load_state_dict(state_dict, strict=False)


class _TQM:
    def __init__(self, names):
        self.components = [_Component(n) for n in names]

    def get_my_distributed_components(self):
        return self.components


def _adam_state(module):
    """Optimizer state of `module` after one step, as saved in a *_optim.pt file."""
    optim = torch.optim.AdamW(module.parameters())
    module(torch.ones(1, 4)).sum().backward()
    optim.step()
    return optim.state_dict()


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    names = ["encoder_0_frozen", "encoder_0_trained"]
    old = nn.ModuleDict({n: nn.Linear(4, 4, bias=False) for n in names})
    for n in names:
        torch.save(_adam_state(old[n]), tmp_path / f"{PREFIX}_{n}_optim.pt")
        torch.save(old[n].state_dict(), tmp_path / f"{PREFIX}_{n}.pt")

    model = nn.ModuleDict({n: nn.Linear(4, 4, bias=False) for n in names})
    model["encoder_0_frozen"].weight.requires_grad = False
    # Like MultipleOptimizer: only components with trainable params get a suboptimizer
    optim = SimpleNamespace(suboptimizers={
        "encoder_0_trained": torch.optim.AdamW(model["encoder_0_trained"].parameters()),
    })
    return str(tmp_path / f"{PREFIX}_frame.pt"), model, optim, _TQM(names), old


def test_frozen_component_optim_state_is_skipped(setup, caplog):
    frame, model, optim, tqm, old = setup
    with caplog.at_level(logging.INFO):
        # Strict load: a skipped frozen optimizer state must not count as missing
        load_parameters_from_checkpoint(frame, model, optim=optim, task_queue_manager=tqm,
                                        reset_optim=False, allow_partial_checkpoint=False)

    for n in ["encoder_0_frozen", "encoder_0_trained"]:
        assert torch.equal(model[n].weight, old[n].weight)  # weights restored for both
    state = optim.suboptimizers["encoder_0_trained"].state
    assert len(state) == 1 and state[model["encoder_0_trained"].weight]["step"] == 1
    assert "Skipping optimizer state of frozen component encoder_0_frozen" in caplog.text
    assert "All modules and optimizer restored" in caplog.text


def test_frozen_component_without_optim_file_is_not_missing(setup, tmp_path):
    """A frozen component needs no optimizer state, so a missing *_optim.pt is fine."""
    frame, model, optim, tqm, _ = setup
    (tmp_path / f"{PREFIX}_encoder_0_frozen_optim.pt").unlink()
    load_parameters_from_checkpoint(frame, model, optim=optim, task_queue_manager=tqm,
                                    reset_optim=False, allow_partial_checkpoint=False)
