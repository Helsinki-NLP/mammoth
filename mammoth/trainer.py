"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""


import mammoth.distributed
import torch
import torch.distributed
import torch.nn as nn
import traceback

from itertools import islice
from mammoth.utils.logging import logger


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
    generators_md=None,
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

    train_loss_md = nn.ModuleDict()
    valid_loss_md = nn.ModuleDict()
    logger.info("BUILD TRAINER")

    for (side, lang, component_id, tgt_vocab) in task_queue_manager.get_my_vocabs('tgt', vocabs_dict):
        generator = generators_md[f'generator_{lang}']
        train_loss_md.add_module(
            f'trainloss{lang}',
            mammoth.utils.loss.build_loss_compute(model, tgt_vocab, opts, train=True, generator=generator),
        )
        valid_loss_md.add_module(
            f'valloss{lang}',
            mammoth.utils.loss.build_loss_compute(model, tgt_vocab, opts, train=False, generator=generator),
        )

    trunc_size = opts.truncated_decoder  # Badly named...
    shard_size = opts.max_generator_batches if opts.model_dtype == 'fp32' else 0
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
        train_loss_md,
        valid_loss_md,
        optim,
        trunc_size,
        shard_size,
        norm_method,
        accum_count,
        accum_steps,
        device_context=device_context,
        gpu_verbose_level=gpu_verbose_level,
        report_manager=report_manager,
        with_align=True if opts.lambda_align > 0 else False,
        model_saver=model_saver,
        average_decay=average_decay,
        average_every=average_every,
        model_dtype=opts.model_dtype,
        earlystopper=earlystopper,
        dropout=dropout,
        dropout_steps=dropout_steps,
        task_queue_manager=task_queue_manager,
        report_stats_from_parameters=opts.report_stats_from_parameters,
    )
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`mammoth.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`mammoth.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`mammoth.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`mammoth.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
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
        train_loss_md,
        valid_loss_md,
        optim,
        trunc_size=0,
        shard_size=32,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        device_context=None,
        gpu_verbose_level=0,
        report_manager=None,
        with_align=False,
        model_saver=None,
        average_decay=0,
        average_every=1,
        model_dtype='fp32',
        earlystopper=None,
        dropout=[0.3],
        dropout_steps=[0],
        task_queue_manager=None,
        report_stats_from_parameters=False,
    ):
        # Basic attributes.
        self.model = model
        self.train_loss_md = train_loss_md
        self.valid_loss_md = valid_loss_md
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.device_context = device_context
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.report_stats_from_parameters = report_stats_from_parameters
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps

        self.task_queue_manager = task_queue_manager
        my_component_groups = self.task_queue_manager.get_my_distributed_groups()
        self.my_encoder_groups = my_component_groups['encoder']
        self.my_decoder_groups = my_component_groups['decoder']
        self.my_src_emb_groups = my_component_groups['src_emb']
        self.my_tgt_emb_groups = my_component_groups['tgt_emb']
        self.my_encoder_adapter_groups = my_component_groups['encoder_adapters']
        self.my_decoder_adapter_groups = my_component_groups['decoder_adapters']

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert (
                    self.trunc_size == 0
                ), """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
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

        total_stats = mammoth.utils.Statistics()
        report_stats = mammoth.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        self.optim.zero_grad()

        i = -1
        while True:
            i += 1

            step = self.optim.training_step
            self._maybe_update_dropout(step)

            self.accum_count = self._accum_count(self.optim.training_step)
            self.task_queue_manager.tasks_per_communication_batch = self.accum_count
            batches_with_meta = islice(train_iter, self.accum_count)

            self._gradient_accumulation_over_lang_pairs(
                batches_with_meta,
                total_stats,
                report_stats,
            )

            # Note that all group ids are tuples, some with length 1
            for (layer_stack_index, encoder_id), (_, group) in self.my_encoder_groups.items():
                params = [
                    (name, p) for (name, p)
                    in self.model.encoder.get_submodule(layer_stack_index, encoder_id).named_parameters()
                    if 'embeddings' not in name and 'adapter' not in name
                ]
                mammoth.distributed.only_ready_reduce_and_rescale_grads(params, group=group)

            for (layer_stack_index, decoder_id), (_, group) in self.my_decoder_groups.items():
                params = [
                    (name, p) for (name, p)
                    in self.model.decoder.get_submodule(layer_stack_index, decoder_id).named_parameters()
                    if 'embeddings' not in name and 'adapter' not in name
                ]
                mammoth.distributed.only_ready_reduce_and_rescale_grads(params, group=group)

            for (src_lang,), (_, group) in self.my_src_emb_groups.items():
                embs = self.model.encoder.embeddings[f'embeddings_{src_lang}']
                mammoth.distributed.only_ready_reduce_and_rescale_grads(embs.named_parameters(), group=group)

            for (tgt_lang,), (_, group) in self.my_tgt_emb_groups.items():
                embs = self.model.decoder.embeddings[f'embeddings_{tgt_lang}']
                mammoth.distributed.only_ready_reduce_and_rescale_grads(embs.named_parameters(), group=group)

                mammoth.distributed.only_ready_reduce_and_rescale_grads(
                    self.model.generator[f'generator_{tgt_lang}'].named_parameters(), group=group
                )

            for adapter_id, (_, group) in self.my_encoder_adapter_groups.items():
                layer_stack_index, encoder_id, adapter_group, sub_id = adapter_id
                adapter = self.model.encoder.get_submodule(layer_stack_index, encoder_id).get_adapter(
                    adapter_group, sub_id
                )
                mammoth.distributed.only_ready_reduce_and_rescale_grads(adapter.named_parameters(), group=group)

            for adapter_id, (_, group) in self.my_decoder_adapter_groups.items():
                layer_stack_index, decoder_id, adapter_group, sub_id = adapter_id
                adapter = self.model.decoder.get_submodule(layer_stack_index, decoder_id).get_adapter(
                    adapter_group, sub_id
                )
                mammoth.distributed.only_ready_reduce_and_rescale_grads(adapter.named_parameters(), group=group)

            # a group is not specified: reduce across all devices
            if device_context.is_distributed():
                mammoth.distributed.only_ready_reduce_and_rescale_grads(
                    self.model.attention_bridge.named_parameters()
                )

            self._maybe_update_stats_from_parameters(report_stats, self.model.named_parameters())

            self.optim.step()
            self.optim.zero_grad()
            for p in self.model.parameters():
                if hasattr(p, 'has_grad'):
                    p.has_grad = False

            if step % 1000 == 0 and step > 0:
                # TODO: if you are going to uncomment that block, please make it optional
                # logger.info(f'After gradient sync {step}')
                # for name, p in self.model.named_parameters():
                #     logger.info(
                #         f'{device_context.node_rank}:{device_context.local_rank}'
                #         f' {name}: {p.flatten()[:10]}'
                #     )
                if hasattr(self.optim._optimizer, 'report_steps'):
                    for line in self.optim._optimizer.report_steps():
                        logger.info(f'{device_context.node_rank}:{device_context.local_rank} {line}')

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step,
                train_steps,
                self.optim.learning_rate(),
                report_stats,
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
                        self.optim.learning_rate(),  # learning_rate_to_show, #self.optim.learning_rate(),
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
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
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
                    stats = mammoth.utils.Statistics()

                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt

                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    # F-prop through the model.
                    outputs, attns = valid_model(
                        src, tgt, src_lengths, with_align=self.with_align, metadata=metadata
                    )

                    # Compute loss.
                    _, batch_stats = self.valid_loss_md[f"valloss{metadata.tgt_lang}"](batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        for p in self.model.parameters():
            if hasattr(p, 'has_grad'):
                p.has_grad = False

        return stats

    def _gradient_accumulation_over_lang_pairs(
        self,
        batches_with_meta,
        total_stats,
        report_stats,
    ):
        normalization = 0
        seen_comm_batches = set()
        for k, (batch, metadata, comm_batch) in enumerate(batches_with_meta):
            seen_comm_batches.add(comm_batch)
            if self.norm_method == "tokens":
                num_tokens = (
                    batch.labels[1:, :, 0].ne(self.train_loss_md[f'trainloss{metadata.tgt_lang}'].padding_idx).sum()
                )
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size

            # logger.info(f'batch with metadata {metadata}')

            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                raise Exception('Truncated BPTT not supported')
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            # tgt_outer corresponds to the target-side input. The expected
            # decoder output will be read directly from the batch:
            # cf. `onmt.utils.loss.CommonLossCompute._make_shard_state`
            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j:(j + trunc_size)]
                # TODO: AMP == TRUE If fp16
                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    outputs, attns = self.model(
                        src, tgt, src_lengths, bptt=bptt, with_align=self.with_align, metadata=metadata
                    )
                    bptt = True

                    # 3. Compute loss.
                    loss, batch_stats = self.train_loss_md[f'trainloss{metadata.tgt_lang}'](
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size,
                    )
                    # logger.info(loss)

                try:
                    if loss is not None:
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)
                    report_stats.update_task_loss(batch_stats.loss, metadata)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d", self.training_step_all, k)
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

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
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
