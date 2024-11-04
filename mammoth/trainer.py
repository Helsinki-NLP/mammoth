"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""


import torch
import torch.distributed
import torch.nn as nn
from einops import rearrange
from itertools import islice

import mammoth.distributed
from mammoth.utils.logging import logger
from mammoth.utils.loss import build_loss_function
from mammoth.utils.statistics import Statistics


class NanLossException(Exception):
    pass


def iter_on_device(iterator, device_context):
    if device_context.is_gpu():
        device = torch.device(f'cuda:{device_context.local_rank}')
    else:
        device = torch.device('cpu')
    for batch, meta, comm_batch_id in iterator:
        yield batch.to(device), meta, comm_batch_id


def build_trainer(
    opts,
    device_context,
    model,
    vocabs_dict,
    optim,
    task_queue_manager,
    model_saver=None,
):
    """
    Simplify `Trainer` creation based on user `opts`s*

    Args:
        opts (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`mammoth.models.NMTModel`): the model to train
        vocabs_dict (dict): dict of vocabs
        optim (:obj:`mammoth.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text"
        model_saver(:obj:`mammoth.models.ModelSaverBase`): the utility object
            used to save the model
    """

    loss_functions = nn.ModuleDict()
    logger.info("BUILD TRAINER")

    for (side, lang, component_id, tgt_vocab) in task_queue_manager.get_my_vocabs('tgt', vocabs_dict):
        # Note that the old OpenNMT losses required a separate generator, which is not available in x_transformers
        # In MAMMOTH, pytorch losses are used instead.
        loss_functions[lang] = build_loss_function(
            tgt_vocab,
            label_smoothing=opts.label_smoothing,
        )

    norm_method = opts.normalization
    accum_count = opts.accum_count
    accum_steps = opts.accum_steps
    average_decay = opts.average_decay
    average_every = opts.average_every
    dropout = opts.dropout
    dropout_steps = opts.dropout_steps
    gpu_verbose_level = opts.gpu_verbose_level

    earlystopper = (
        mammoth.utils.EarlyStopping(opts.early_stopping, scorers=mammoth.utils.scorers_from_opts(opts))
        if opts.early_stopping > 0
        else None
    )

    report_manager = mammoth.utils.build_report_manager(opts, device_context.node_rank, device_context.local_rank)
    trainer = mammoth.Trainer(
        model,
        loss_functions,
        optim,
        norm_method,
        accum_count,
        accum_steps,
        device_context=device_context,
        gpu_verbose_level=gpu_verbose_level,
        report_manager=report_manager,
        model_saver=model_saver,
        average_decay=average_decay,
        average_every=average_every,
        model_dtype=opts.model_dtype,
        earlystopper=earlystopper,
        dropout=dropout,
        dropout_steps=dropout_steps,
        task_queue_manager=task_queue_manager,
        report_stats_from_parameters=opts.report_stats_from_parameters,
        report_training_accuracy=opts.report_training_accuracy,
    )
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`mammoth.models.model.NMTModel`): translation model
                to train
            loss_functions:
               ModelDict containing loss functions
            optim(:obj:`mammoth.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            data_type(string): type of the source input: [text]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`mammoth.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`mammoth.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(
        self,
        model,
        loss_functions,
        optim,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        device_context=None,
        gpu_verbose_level=0,
        report_manager=None,
        model_saver=None,
        average_decay=0,
        average_every=1,
        model_dtype='fp32',
        earlystopper=None,
        dropout=[0.3],
        dropout_steps=[0],
        task_queue_manager=None,
        report_stats_from_parameters=False,
        report_training_accuracy=False,
    ):
        # Basic attributes.
        self.model = model
        self.loss_functions = loss_functions
        self.optim = optim
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.device_context = device_context
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.report_stats_from_parameters = report_stats_from_parameters
        self.report_training_accuracy = report_training_accuracy
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps

        self.task_queue_manager = task_queue_manager

        self._data_state = {}

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        if step == 0:
            _accum = self.accum_count_l[0]
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d" % (self.dropout[i], step))

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float() for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay, 1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average), self.model.parameters()):
                self.moving_average[i] = (1 - average_decay) * avg + cpt.detach().float() * average_decay

    def train(
        self,
        train_iter,
        train_steps,
        save_checkpoint_steps=5000,
        valid_iter=None,
        valid_steps=10000,
        device_context=None,
    ):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        train_iter = iter_on_device(train_iter, device_context)
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...', valid_steps)

        n_correct = 0 if self.report_training_accuracy else None
        total_stats = mammoth.utils.Statistics(n_correct=n_correct)
        report_stats = mammoth.utils.Statistics(n_correct=n_correct)
        self._start_report_manager(start_time=total_stats.start_time)
        self.optim.zero_grad()

        i = -1
        while True:
            i += 1

            # global training step
            step = self.optim.training_step
            self._maybe_update_dropout(step)

            self.accum_count = self._accum_count(self.optim.training_step)
            self.task_queue_manager.accum_count = self.accum_count
            batches_with_meta = islice(train_iter, self.accum_count)

            batch_task_sample = self.task_queue_manager.sample_corpus_ids()
            my_task = batch_task_sample.tasks[self.task_queue_manager.global_rank]

            gradient_syncs = self.task_queue_manager.distributed_component_gradient_sync(batch_task_sample)

            self._gradient_accumulation(
                batches_with_meta,
                total_stats,
                report_stats,
                my_task,
                gradient_syncs,
            )

            for gradient_sync in gradient_syncs:
                component = gradient_sync.component
                if not component.needs_communication():
                    # Omit components not found elsewhere, as these don't need to be communicated
                    # logger.warning(f'Omitting (single device) {component.get_name()}')   # DEBUG
                    continue
                # logger.warning(f'Syncing {component.get_name()}')   # DEBUG
                params = component.named_parameters(self.model)
                # gradient_sync.gradient_norm counts the number of devices that trained this component
                # this doesn't normalize the number of masked tokens
                mammoth.distributed.externally_managed_reduce_and_rescale_grads(
                    named_parameters=params,
                    has_local_gradient=gradient_sync.has_local_gradient,
                    gradient_norm=gradient_sync.gradient_norm,
                    group=component.group,
                )

            self._maybe_update_stats_from_parameters(report_stats, self.model.named_parameters())

            # Including single-device components
            self.optim.externally_managed_step(gradient_syncs)
            self.optim.zero_grad()

            # if step % 1000 == 0 and step > 0:
            #     TODO: if you are going to uncomment that block, please make it optional
            #     logger.info(f'After gradient sync {step}')
            #     for name, p in self.model.named_parameters():
            #         logger.info(
            #             f'{device_context.node_rank}:{device_context.local_rank}'
            #             f' {name}: {p.flatten()[:10]}'
            #         )

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            # Learning rate used to be retrieved with: self.optim.learning_rate()
            # However, as each optimizer has its own learning rate, it is not obvious what to log here.
            # We might log the mean or the range of learning rates, but the simplest thing is to log nothing.
            report_lr = None
            if device_context.is_master():
                sampled_task_counts = self.task_queue_manager.sampled_task_counts
            else:
                sampled_task_counts = None
            report_stats = self._maybe_report_training(
                step,
                train_steps,
                report_lr,
                report_stats,
                sampled_task_counts=sampled_task_counts,
            )

            if step % valid_steps == 0 and valid_iter is not None:
                if self.gpu_verbose_level > 0:
                    logger.info(f'{device_context.node_rank}:{device_context.local_rank} validate step {step}')
                valid_stats = self.validate(
                    iter_on_device(valid_iter, device_context),
                    moving_average=self.moving_average,
                )
                if self.gpu_verbose_level > 0:
                    logger.info(f'{device_context.node_rank}:{device_context.local_rank} gather valid stat step {step}')
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info(f'{device_context.node_rank}:{device_context.local_rank} report stat step {step}')
                if device_context.is_master():
                    self._report_step(
                        None,
                        step,
                        valid_stats=valid_stats,
                    )

                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0):
                self.model_saver.save(step, self._data_state, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, self._data_state, moving_average=self.moving_average)
        if device_context.is_master() and self.report_manager is not None:
            self.report_manager.report_end(step)
        return total_stats

    def validate(self, valid_iter, moving_average=None, task=None):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():

            # Tasks need not define validation paths: hence, a device need not contain
            # any validation path. This would cause statistics equals to 0 word seen,
            # which would then cause a zero devision when normalizing PPL per words.
            stats = None  # mammoth.utils.Statistics()

            for batch, metadata, _ in valid_iter:
                if stats is None:
                    stats = mammoth.utils.Statistics(n_correct=0)

                stats.n_src_words += batch.src.mask.sum().item()
                src = batch.src.tensor
                src_mask = batch.src.mask
                decoder_input = batch.tgt.tensor[:-1]
                target = batch.tgt.tensor[1:]
                # if self.norm_method == "tokens":
                #     normalization = batch.tgt.mask.sum().item()
                # else:
                #     normalization = batch.batch_size

                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    # F-prop through the model.
                    logits, decoder_output = valid_model(
                        rearrange(src, 't b 1 -> b t'),
                        rearrange(decoder_input, 't b 1 -> b t'),
                        rearrange(src_mask, 't b -> b t'),
                        metadata=metadata,
                    )
                    logits = rearrange(logits, 'b t i -> t b i')
                    decoder_output = rearrange(decoder_output, 'b t d -> t b d')

                    # Compute loss.
                    loss = self.loss_functions[metadata.tgt_lang](
                        rearrange(logits, 't b i -> (t b) i'),
                        rearrange(target, 't b 1 -> (t b)'),
                    )
                    # loss /= normalization

                # Update statistics.
                padding_idx = self.loss_functions[metadata.tgt_lang].ignore_index
                batch_stats = Statistics.from_loss_logits_target(
                    loss.item(),
                    logits,
                    target,
                    padding_idx,
                )
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _gradient_accumulation(
        self,
        batches_with_meta,
        total_stats,
        report_stats,
        my_task,
        gradient_syncs,
    ):
        normalization = 0
        seen_comm_batches = set()
        expected_metadata = my_task.get_serializable_metadata()
        for k, (batch, metadata, comm_batch) in enumerate(batches_with_meta):
            if metadata != expected_metadata:
                raise Exception(
                    f'Mismatch in task sampling for batch {comm_batch}.\n '
                    f'Received {metadata},\n expected {expected_metadata}'
                )
            seen_comm_batches.add(comm_batch)

            # update data state
            self._data_state[metadata.corpus_id] = batch.line_idx

            num_tokens = batch.tgt.mask.sum().item()
            if self.norm_method == "tokens":
                normalization += num_tokens
            else:
                normalization += batch.batch_size
            report_stats.n_src_words += batch.src.mask.sum().item()

            # logger.info(f'batch with metadata {metadata}')

            src = batch.src.tensor
            src_mask = batch.src.mask

            decoder_input = batch.tgt.tensor[:-1]
            target = batch.tgt.tensor[1:]
            # tgt_mask = batch.tgt.mask

            # shapes are: (t b i)   i.e.   (time, batch, vocab_index)

            with torch.cuda.amp.autocast(enabled=self.optim.amp):
                logits, decoder_output = self.model(
                    src=rearrange(src, 't b 1 -> b t'),
                    decoder_input=rearrange(decoder_input, 't b 1 -> b t'),
                    src_mask=rearrange(src_mask, 't b -> b t'),
                    metadata=metadata,
                )
                logits = rearrange(logits, 'b t i -> t b i')
                decoder_output = rearrange(decoder_output, 'b t d -> t b d')

                # 3. Compute loss.
                loss = self.loss_functions[metadata.tgt_lang](
                    rearrange(logits, 't b i -> (t b) i'),
                    rearrange(target, 't b 1 -> (t b)'),
                )
                # logger.info(loss)

            if loss is not None:
                if torch.isnan(loss):
                    raise NanLossException('Loss blowout')
                # loss /= normalization
                self.optim.backward(loss)

            if self.report_training_accuracy:
                # Slow: requires max over logits, eq, masked_select
                batch_stats = Statistics.from_loss_logits_target(
                    loss.item(),
                    logits,
                    target,
                    padding_idx=self.loss_functions[metadata.tgt_lang].ignore_index,
                )
            else:
                batch_stats = Statistics(
                    loss.item(),
                    num_tokens,
                    n_correct=None,
                )

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
            report_stats.update_task_loss(batch_stats.loss, metadata)

        if len(seen_comm_batches) != 1:
            logger.warning('Communication batches out of synch with batch accumulation')

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:mammoth.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.device_context.is_distributed():
            return mammoth.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_update_stats_from_parameters(self, report_stats, named_parameters):
        if self.report_manager is not None and self.report_stats_from_parameters:
            report_stats.update_from_parameters(named_parameters)

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats, sampled_task_counts):
        """
        Simple function to report training stats (if report_manager is set)
        see `mammoth.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step,
                num_steps,
                learning_rate,
                None if self.earlystopper is None else self.earlystopper.current_tolerance,
                report_stats,
                multigpu=self.device_context.is_distributed(),
                sampled_task_counts=sampled_task_counts,
                optimizer=self.optim,
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `mammoth.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate,
                None if self.earlystopper is None else self.earlystopper.current_tolerance,
                step,
                train_stats=train_stats,
                valid_stats=valid_stats,
            )
