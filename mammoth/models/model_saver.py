import os
from glob import glob
from collections import deque
from mammoth.utils.logging import logger

import torch
import torch.nn as nn

from mammoth.utils.module_splitter import explode_model


def build_model_saver(model_opts, opts, model, vocabs_dict, optim, device_context):
    # _check_save_model_path
    save_model_path = os.path.abspath(opts.save_model)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_saver = ModelSaver(
        opts.save_model, model, model_opts, vocabs_dict, optim, opts.keep_checkpoint, device_context, opts.save_all_gpus
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
        device_context=None,
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
        assert device_context is not None
        self.device_context = device_context
        self.all_gpus = all_gpus

    def save(self, step, moving_average=None):
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

        chkpt_names = self._save(step, save_model, self.device_context)
        self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data, save_model.parameters()):
                param.data = param_data

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_names)

    def _save(self, step):
        """Save a resumable checkpoint.

        Args:
            step (int): step number
            model (nn.Module): torch model to save

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

    def _save(self, step, model, device_context):
        real_model = model.module if isinstance(model, nn.DataParallel) else model

        model_state_dict = real_model.state_dict()

        checkpoint = {
            "model": model_state_dict,
            # 'generator': generator_state_dict,
            "vocab": self.vocabs_dict,
            "opts": self.model_opts,
            "optim": self.optim.state_dict(),
            "whole_model": self.model,
        }

        tmp_checkpoint_paths = []

        if self.all_gpus:
            # save models trained in each gpu
            checkpoint_path = "{}_step_{}_gpu_{}.pt".format(self.base_path, step, device_context.global_rank)
            logger.info("Saving full checkpoint {}".format(checkpoint_path))
            torch.save(checkpoint, checkpoint_path)
            tmp_checkpoint_paths.append(checkpoint_path)

        modules, model_frame = explode_model(checkpoint)

        for key, module in modules.items():
            # All processes will try to save the modules present on that device
            # Not that a race condition is possible:
            # the process can be preempted after the check for existence, but before the save.
            # This shouldn't be a problem, if writes are atomic.
            checkpoint_path = f'{self.base_path}_step_{step}_{key}.pt'
            if os.path.isfile(checkpoint_path):
                logger.debug("{} - not saving {} as it is already present".format(device_context.id, checkpoint_path))
            else:
                logger.info("Saving module checkpoint {}".format(checkpoint_path))
                torch.save(module, checkpoint_path)
                tmp_checkpoint_paths.append(checkpoint_path)

        if device_context.is_master():
            # TODO: not sure how to deal with model_state_dict, fields, model_opts and optim.state_dict() in a multi-gpu
            #  setting. Is it OK to save only from master?

            # model frame
            checkpoint_path = "{}_step_{}_frame.pt".format(self.base_path, step)
            logger.info("Saving model frame checkpoint {}".format(checkpoint_path))
            torch.save(model_frame, checkpoint_path)
            tmp_checkpoint_paths.append(checkpoint_path)

        return tmp_checkpoint_paths

    def _rm_checkpoint(self, names):
        for name in names:
            if os.path.exists(name):
                try:
                    os.remove(name)
                except BaseException:
                    logger.warning(f'Failed to delete {name}')
