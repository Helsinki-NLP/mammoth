"""--allow_partial_checkpoint: loading a checkpoint that does not cover the whole model.

Scenario: a model was trained with an encoder stack "old"; the config now adds a new stack "new".
The checkpoint has no file for "new", so a strict load must fail and a partial load must keep
"old"'s trained weights while leaving "new" at its fresh initialization.
"""
import logging

import configargparse
import pytest
import torch
import torch.nn as nn

from mammoth.opts import _add_train_general_opts
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


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    trained = nn.Linear(4, 4, bias=False)
    torch.save(trained.state_dict(), tmp_path / f"{PREFIX}_encoder_0_old.pt")
    # Freshly built model from the *new* config: "old" (in checkpoint) + "new" (not in checkpoint)
    model = nn.ModuleDict({"encoder_0_old": nn.Linear(4, 4, bias=False),
                           "encoder_0_new": nn.Linear(4, 4, bias=False)})
    fresh_new = model["encoder_0_new"].weight.detach().clone()
    tqm = _TQM(["encoder_0_old", "encoder_0_new"])
    return str(tmp_path / f"{PREFIX}_frame.pt"), model, tqm, trained, fresh_new


def _load(frame, model, tqm, allow):
    load_parameters_from_checkpoint(frame, model, optim=None, task_queue_manager=tqm,
                                    reset_optim=True, allow_partial_checkpoint=allow)


def test_strict_load_rejects_missing_component(setup):
    frame, model, tqm, _, _ = setup
    with pytest.raises(Exception) as exc:
        _load(frame, model, tqm, allow=False)
    msg = str(exc.value)
    assert "encoder_0_new.pt (entire file missing)" in msg
    assert "--allow_partial_checkpoint" in msg  # error tells you how to opt in


def test_partial_load_keeps_trained_and_new_weights(setup, caplog):
    frame, model, tqm, trained, fresh_new = setup
    with caplog.at_level(logging.WARNING):
        _load(frame, model, tqm, allow=True)
    assert torch.equal(model["encoder_0_old"].weight, trained.weight)  # restored
    assert torch.equal(model["encoder_0_new"].weight, fresh_new)       # left at its init
    warning = caplog.text
    assert "--allow_partial_checkpoint" in warning
    assert "encoder_0_new.pt (entire file missing)" in warning           # reinitialized parts are listed
    assert "encoder_0_old" not in warning.replace("encoder_0_new", "")  # restored parts are not


def test_partial_load_lists_every_key(tmp_path, caplog):
    """The warning is uncapped (unlike the error) so all reinitialized parts can be checked."""
    names = [f"encoder_0_new{i}" for i in range(15)]
    model = nn.ModuleDict({n: nn.Linear(2, 2, bias=False) for n in names})
    with caplog.at_level(logging.WARNING):
        _load(str(tmp_path / f"{PREFIX}_frame.pt"), model, _TQM(names), allow=True)
    assert "Missing parameters (15):" in caplog.text
    assert all(f"{n}.pt" in caplog.text for n in names)
    assert "more" not in caplog.text


@pytest.mark.parametrize("argv,yaml_text,expected", [
    (["--allow_partial_checkpoint"], None, True),
    (["-yes_i_messed_with_the_checkpoint"], None, True),  # deprecated alias
    ([], "allow_partial_checkpoint: true\n", True),
    ([], "yes_i_messed_with_the_checkpoint: true\n", True),
    ([], None, False),
])
def test_option_and_deprecated_alias(argv, yaml_text, expected, tmp_path):
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        args_for_setting_config_path=["-config"],
    )
    _add_train_general_opts(parser)
    if yaml_text:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(yaml_text)
        argv = ["-config", str(cfg)]
    opts, _ = parser.parse_known_args(argv + ["--node_rank", "0", "--seed", "1", "--max_length", "10"])
    assert opts.allow_partial_checkpoint is expected
    assert not hasattr(opts, "yes_i_messed_with_the_checkpoint")
