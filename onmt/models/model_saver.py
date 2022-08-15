import os
from collections import deque
from onmt.utils.logging import logger

import torch
import torch.nn as nn

from onmt.utils.distributed import is_master
from onmt.utils.module_splitter import explode_model


def build_model_saver(model_opt, opt, model, fields_dict, optim, device_id):
    # _check_save_model_path
    save_model_path = os.path.abspath(opt.save_model)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_saver = ModelSaver(
        opt.save_model, model, model_opt, fields_dict, optim, opt.keep_checkpoint, device_id, opt.save_all_gpus
    )
    return model_saver


def load_checkpoint(ckpt_path):
    """Load checkpoint from `ckpt_path` if any else return `None`."""
    checkpoint = None
    if ckpt_path:
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
        model_opt,
        fields_dict,
        optim,
        keep_checkpoint=-1,
        device_id="0",
        all_gpus=False,
    ):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields_dict = fields_dict
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)
        self.device_id = device_id
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

        chkpt_names = self._save(step, save_model, self.device_id)
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

    def _save(self, step, model, device_id):
        real_model = model.module if isinstance(model, nn.DataParallel) else model

        model_state_dict = real_model.state_dict()
        encoder_ids = {
            index: lang[0].replace('encoder', '') for index, lang in enumerate(model.encoder.named_children())
        }
        decoder_ids = {
            index: lang[0].replace('decoder', '') for index, lang in enumerate(model.decoder.named_children())
        }

        checkpoint = {
            "model": model_state_dict,
            # 'generator': generator_state_dict,
            "vocab": self.fields_dict,
            "opt": self.model_opt,
            "optim": {k: v.state_dict() for k, v in self.optim._optimizer.optimizers.items()},
            "whole_model": self.model,
        }

        tmp_checkpoint_paths = []

        if self.all_gpus:
            # save models trained in each gpu
            checkpoint_path = "{}_step_{}_gpu_{}.pt".format(self.base_path, step, device_id)
            logger.info("Saving full checkpoint {}".format(checkpoint_path))
            torch.save(checkpoint, checkpoint_path)
            tmp_checkpoint_paths.append(checkpoint_path)

        encoders, decoders, attention_bridge, generators, model_frame = explode_model(checkpoint)

        # TODO: refactor (in a dedicated saver class?)
        # encoder modules
        for i, encoder in enumerate(encoders):
            checkpoint_path = "{}_step_{}_{}_enc.pt".format(self.base_path, step, encoder_ids[i])
            if os.path.isfile(checkpoint_path):
                logger.debug("GPU {} - not saving {} as it is already present".format(device_id, checkpoint_path))
            else:
                logger.info("Saving encoder checkpoint {}".format(checkpoint_path))
                torch.save(encoder, checkpoint_path)
                tmp_checkpoint_paths.append(checkpoint_path)
        # decoder modules
        for i, decoder in enumerate(decoders):
            checkpoint_path = "{}_step_{}_{}_dec.pt".format(self.base_path, step, decoder_ids[i])
            if os.path.isfile(checkpoint_path):
                logger.debug("GPU {} - not saving {} as it is already present".format(device_id, checkpoint_path))
            else:
                logger.info("Saving decoder checkpoint {}".format(checkpoint_path))
                torch.save(decoder, checkpoint_path)
                tmp_checkpoint_paths.append(checkpoint_path)
        # generator modules
        for i, generator in enumerate(generators):
            checkpoint_path = "{}_step_{}_{}_gen.pt".format(self.base_path, step, decoder_ids[i])
            if os.path.isfile(checkpoint_path):
                logger.debug("GPU {} - not saving {} as it is already present".format(device_id, checkpoint_path))
            else:
                logger.info("Saving generator checkpoint {}".format(checkpoint_path))
                torch.save(generator, checkpoint_path)
                tmp_checkpoint_paths.append(checkpoint_path)

        if is_master(device_id):
            # TODO: not sure how to deal with model_state_dict, fields, model_opt and optim.state_dict() in a multi-gpu
            #  setting. Is it OK to save only from master?
            # attention bridge module
            checkpoint_path = "{}_step_{}_bridge.pt".format(self.base_path, step)
            logger.info("Saving attention bridge checkpoint {}".format(checkpoint_path))
            torch.save(attention_bridge, checkpoint_path)
            tmp_checkpoint_paths.append(checkpoint_path)

            # model frame
            checkpoint_path = "{}_step_{}_frame.pt".format(self.base_path, step)
            logger.info("Saving model frame checkpoint {}".format(checkpoint_path))
            torch.save(model_frame, checkpoint_path)
            tmp_checkpoint_paths.append(checkpoint_path)

        return tmp_checkpoint_paths

    def _rm_checkpoint(self, names):
        for name in names:
            if os.path.exists(name):
                os.remove(name)
