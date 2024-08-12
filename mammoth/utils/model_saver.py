import os
from glob import glob
from collections import deque
from mammoth.utils.logging import logger

import torch
import torch.distributed
import torch.nn as nn

from mammoth.utils.module_splitter import explode_model


def build_model_saver(model_opts, opts, model, vocabs_dict, optim, task_queue_manager):
    # _check_save_model_path
    save_model_path = os.path.abspath(opts.save_model)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_saver = ModelSaver(
        opts.save_model, model, model_opts, vocabs_dict, optim, opts.keep_checkpoint, task_queue_manager, opts.save_all_gpus
    )
    return model_saver


def load_checkpoint(ckpt_path):
    """Load checkpoint from `ckpt_path` if any else return `None`."""
    checkpoint = None
    if ckpt_path:
        if not ckpt_path.endswith('.pt'):
            frames = glob(os.path.join(ckpt_path + '*frame*pt'))
            frames.sort(key=lambda s: int(s.split('step_')[-1].split('_frame')[0]))
            ckpt_path = frames[-1]
        logger.info('Loading checkpoint from %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    return checkpoint


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
        all_gpus=False,
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
        self.all_gpus = all_gpus

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

        module_state_dicts = explode_model(model, task_queue_manager)

        # The master device stores the frame
        if device_context.is_master():
            module_state_dicts['frame'] = {
                'vocab': self.vocabs_dict,
                'opts': self.model_opts,
                'optim': self.optim.state_dict(),
            }

        # In a distributed context, aggregate all data states for corpus restoration
        if device_context.is_distributed():
            data_states = [None for _ in range(device_context.world_size)]
            torch.distributed.all_gather_object(data_states, data_state)
            data_state = {k: v for state in data_states for k, v in state.items()}
            if device_context.is_master():
                module_state_dicts['frame']['data_state'] = data_state

        for key, state_dict in module_state_dicts.items():
            # The state_dicts across different devices only contain one copy of each module:
            # on the lowest ranked device having that module.
            # There is no race condition.
            checkpoint_path = f'{self.base_path}_step_{step}_{key}.pt'
            if os.path.isfile(checkpoint_path):
                logger.debug("{} - not saving {} as it is already present".format(device_context.id, checkpoint_path))
            else:
                logger.info("Saving module checkpoint {}".format(checkpoint_path))
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
