import os
import torch
import torch.distributed
import torch.nn as nn
from collections import OrderedDict, deque
from glob import glob
from typing import Dict, Any, Tuple

from mammoth.distributed.tasks import LocalTaskQueueManager
from mammoth.model_builder import build_model
from mammoth.models import NMTModel
from mammoth.utils.logging import logger
from mammoth.utils.misc import use_gpu
from mammoth.utils.parse import ArgumentParser


def build_model_saver(model_opts, opts, model, vocabs_dict, optim, task_queue_manager):
    # _check_save_model_path
    save_model_path = os.path.abspath(opts.save_model)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_saver = ModelSaver(
        opts.save_model, model, model_opts, vocabs_dict, optim, opts.keep_checkpoint, task_queue_manager
    )
    return model_saver


def load_frame_checkpoint(checkpoint_path):
    """
    Load only the frame checkpoint from `checkpoint_path` if any else return `None`.

    This function is intended to be called before the fork:
    the model itself has not yet been constructed, so we don't want to load its parameters.
    We need the vocabs and data loader state from the frame.
    """
    checkpoint = None
    if checkpoint_path:
        if not checkpoint_path.endswith('.pt'):
            frames = glob(os.path.join(checkpoint_path + '*frame*pt'))
            frames.sort(key=lambda s: int(s.split('step_')[-1].split('_frame')[0]))
            checkpoint_path = frames[-1]
        logger.info('Loading frame checkpoint from %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint, checkpoint_path


def explode_model(
    model: NMTModel,
    optim,
    task_queue_manager: LocalTaskQueueManager,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Splits the model into distributed components and retrieves the state dict of each."""
    my_components = task_queue_manager.get_my_distributed_components()
    my_global_rank = task_queue_manager.global_rank
    state_dicts = OrderedDict()
    optim_state_dicts = OrderedDict()
    for component in my_components:
        name = component.get_name()
        if component.min_rank == my_global_rank:
            # Only the lowest ranked device saves a component
            state_dicts[name] = component.state_dict(model)
            # The optimizer parameters are distributed the same way as the components
            # Not all components have trainable (unfrozen) parameters, though
            if name in optim.suboptimizers:
                optim_state_dicts[name] = optim.suboptimizers[name].state_dict()
    return state_dicts, optim_state_dicts


def load_parameters_from_checkpoint(
    frame_checkpoint_path,
    model,
    optim,
    task_queue_manager,
    reset_optim=False,
    yes_i_messed_with_the_checkpoint=False,
):
    """
    Splits the model into distributed components
    and restores the state dict of each component from a checkpoint file.
    """
    if not frame_checkpoint_path:
        return
    checkpoint_prefix = frame_checkpoint_path.removesuffix('_frame.pt')

    my_components = task_queue_manager.get_my_distributed_components()
    all_ok = True
    for component in my_components:
        name = component.get_name()
        checkpoint_path = f'{checkpoint_prefix}_{name}.pt'
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path)
            incompatible_keys = component.load_state_dict(model=model, state_dict=state_dict)
            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                logger.info(f'Module {name} incompatible keys: {incompatible_keys}')
                all_ok = False
        else:
            logger.warning(
                f'Could not find model checkpoint file {checkpoint_path}. Affected parameters are reinitialized.'
            )
            all_ok = False

        if not reset_optim:
            optimizer_path = f'{checkpoint_prefix}_{name}_optim.pt'
            if os.path.isfile(optimizer_path):
                # The optimizer parameters are distributed the same way as the components
                optim_state_dict = torch.load(optimizer_path)
                incompatible_keys = optim.suboptimizers[name].load_state_dict(optim_state_dict)
                if incompatible_keys and (incompatible_keys.missing_keys or incompatible_keys.unexpected_keys):
                    logger.info(f'Optim {name} incompatible keys: {incompatible_keys}')
                    all_ok = False
            else:
                logger.warning(
                    f'Could not find optim checkpoint file {optimizer_path}. Affected parameters are reinitialized.'
                )
                all_ok = False
    if all_ok:
        if reset_optim:
            logger.info(f'All modules restored from checkpoint {checkpoint_prefix}')
            if optim is not None:
                logger.info('Optimizer was reset')
        else:
            logger.info(f'All modules and optimizer restored from checkpoint {checkpoint_prefix}')
    else:
        if yes_i_messed_with_the_checkpoint:
            logger.warning('Proceeding with a partial checkpoint due to --yes_i_messed_with_the_checkpoint')
        else:
            raise Exception('Some parameters are missing from the checkpoint.')


def load_model_for_translation(opts, task_queue_manager, task=None, model_path=None):
    if task is None:
        raise ValueError('Must set task')
    if model_path is None:
        model_path = opts.models[0]

        # Load only the frame
    frame, frame_checkpoint_path = load_frame_checkpoint(checkpoint_path=opts.train_from)

    vocabs_dict = {
        'src': frame["vocab"].get(('src', task.src_lang)),
        'tgt': frame["vocab"].get(('tgt', task.tgt_lang)),
    }

    model_opts = ArgumentParser.checkpoint_model_opts(frame['opts'])

    model = build_model(
        model_opts,
        opts,
        vocabs_dict,
        task_queue_manager,
        single_task=task.corpus_id,
    )

    load_parameters_from_checkpoint(
        frame_checkpoint_path,
        model,
        optim=None,
        task_queue_manager=task_queue_manager,
        reset_optim=True,
        yes_i_messed_with_the_checkpoint=opts.yes_i_messed_with_the_checkpoint,
    )

    device = torch.device("cuda" if use_gpu(opts) else "cpu")
    model.to(device)
    model.eval()

    return vocabs_dict, model, model_opts


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(
        self,
        base_path,
        model,
        model_opts,
        vocabs_dict,
        optim,
        keep_checkpoint=-1,
        task_queue_manager=None,
    ):
        self.base_path = base_path
        self.model = model
        self.model_opts = model_opts
        self.vocabs_dict = vocabs_dict
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)
        assert task_queue_manager is not None
        self.task_queue_manager = task_queue_manager

    def save(self, step, data_state, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        save_model = self.model
        if moving_average:
            model_params_data = []
            for avg, param in zip(moving_average, save_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        chkpt_names = self._save(step, save_model, data_state, self.task_queue_manager)
        self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data, save_model.parameters()):
                param.data = param_data

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_names)

    def _save(self, step, save_model, data_state, task_queue_manager):
        """Save a resumable checkpoint.

        Args:
            step (int): step number
            save_model (nn.Module): torch model to save
            data_state (dict): data streaming info
            task_queue_manager: distributed structure of modules

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model, data_state, task_queue_manager):
        model = model.module if isinstance(model, nn.DataParallel) else model
        device_context = task_queue_manager.device_context

        tmp_checkpoint_paths = []

        module_state_dicts, optim_state_dicts = explode_model(model, self.optim, task_queue_manager)

        # The master device stores the frame
        if device_context.is_master():
            module_state_dicts['frame'] = {
                'vocab': self.vocabs_dict,
                'opts': self.model_opts,
                'global_training_step': self.optim.global_training_step,
            }

        # In a distributed context, aggregate all data states for corpus restoration
        if device_context.is_distributed():
            data_states = [None for _ in range(device_context.world_size)]
            torch.distributed.all_gather_object(data_states, data_state)
            data_state = {k: v for state in data_states for k, v in state.items()}
        if device_context.is_master():
            module_state_dicts['frame']['data_state'] = data_state

        for key, state_dict in module_state_dicts.items():
            # The exploded state_dicts across different devices only contain one copy of each module:
            # on the lowest ranked device having that module.
            # There is no race condition.
            checkpoint_path = f'{self.base_path}_step_{step}_{key}.pt'
            optimizer_path = f'{self.base_path}_step_{step}_{key}_optim.pt'
            if os.path.isfile(checkpoint_path):
                logger.debug("{} - not saving {} as it is already present".format(device_context.id, checkpoint_path))
            else:
                if key != 'frame' and key in optim_state_dicts:
                    logger.info(f'Saving module checkpoint {checkpoint_path} and optimizer {optimizer_path}')
                    torch.save(optim_state_dicts[key], optimizer_path)
                    tmp_checkpoint_paths.append(optimizer_path)
                else:
                    logger.info(f'Saving module checkpoint {checkpoint_path} (no optimizer to save)')
                torch.save(state_dict, checkpoint_path)
                tmp_checkpoint_paths.append(checkpoint_path)

        return tmp_checkpoint_paths

    def _rm_checkpoint(self, names):
        for name in names:
            if os.path.exists(name):
                try:
                    os.remove(name)
                except BaseException:
                    logger.warning(f'Failed to delete {name}')
