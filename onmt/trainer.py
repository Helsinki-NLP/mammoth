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
import traceback

import onmt.utils
from onmt.utils.logging import logger
import torch.nn as nn
import torch.distributed


def build_trainer(
    opt,
    device_id,
    model,
    fields_dict,
    optim,
    scheduler,
    model_saver=None,
    generators_md=None,
):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    train_loss_md = nn.ModuleDict()
    valid_loss_md = nn.ModuleDict()
    logger.info("BUILD TRAINER")

    for (side, lang, component_id, fields) in scheduler.get_fields('tgt', fields_dict):
        generator = generators_md[f"generator{lang}"]
        # This retrieves the primary field of this crappy datastructure
        tgt_field = fields['tgt'].fields[0][1]
        train_loss_md.add_module(
            f'trainloss{lang}',
            onmt.utils.loss.build_loss_compute(model, tgt_field, opt, train=True, generator=generator),
        )
        valid_loss_md.add_module(
            f'valloss{lang}',
            onmt.utils.loss.build_loss_compute(model, tgt_field, opt, train=False, generator=generator),
        )

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id is not None and device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = -1
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = (
        onmt.utils.EarlyStopping(opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt))
        if opt.early_stopping > 0
        else None
    )

    report_manager = onmt.utils.build_report_manager(opt, scheduler.node_rank, scheduler.local_rank)
    trainer = onmt.Trainer(
        model,
        train_loss_md,
        valid_loss_md,
        optim,
        trunc_size,
        shard_size,
        norm_method,
        accum_count,
        accum_steps,
        n_gpu,
        gpu_rank,
        gpu_verbose_level,
        report_manager,
        with_align=True if opt.lambda_align > 0 else False,
        model_saver=model_saver,
        average_decay=average_decay,
        average_every=average_every,
        model_dtype=opt.model_dtype,
        earlystopper=earlystopper,
        dropout=dropout,
        dropout_steps=dropout_steps,
        scheduler=scheduler,
        report_stats_from_parameters=opt.report_stats_from_parameters,
        lca_loginterval=opt.lca_loginterval,
    )
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
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
        n_gpu=1,
        gpu_rank=1,
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
        scheduler=None,
        report_stats_from_parameters=False,
        lca_loginterval=-1,
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
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
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

        self.scheduler = scheduler

        my_component_groups = self.scheduler.get_distributed_groups()
        self.my_encoder_groups = my_component_groups['encoder']
        self.my_decoder_groups = my_component_groups['decoder']
        self.my_src_emb_groups = my_component_groups['src_emb']
        self.my_tgt_emb_groups = my_component_groups['tgt_emb']

        self.lca_loginterval = lca_loginterval

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

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        current_comm_batch_id = None
        comm_batch_count = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch, metadata, communication_batch_id in iterator:
            if current_comm_batch_id is None:
                current_comm_batch_id = communication_batch_id
            # when communication_batch_id changes, we might be ready to synch
            if current_comm_batch_id != communication_batch_id:
                comm_batch_count += 1
                current_comm_batch_id = communication_batch_id
            # accum_count refers to number of communication batches
            # (i.e. one batch from all language pairs)
            # seen before synching gradients
            if comm_batch_count == self.accum_count:
                # logger.info(f'yielding {len(batches)} batches, accum {comm_batch_count} == {self.accum_count}')
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
                comm_batch_count = 0
            # logger.info(f'appending batch with {metadata}, communication_batch_id {communication_batch_id}')
            batches.append((batch, metadata))
            if self.norm_method == "tokens":
                num_tokens = (
                    batch.tgt[1:, :, 0].ne(self.train_loss_md[f'trainloss{metadata.tgt_lang}'].padding_idx).sum()
                )
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float() for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay, 1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average), self.model.parameters()):
                self.moving_average[i] = (1 - average_decay) * avg + cpt.detach().float() * average_decay

    def train(
        self, train_iter, train_steps, save_checkpoint_steps=5000, valid_iter=None, valid_steps=10000, global_rank=None
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
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...', valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        # LCA
        if self.lca_loginterval > 0:
            lca_logs = {
                k: dict() for k, v in self.model.named_parameters() if v.requires_grad and 'attention_bridge' in k
            }
            lca_params = {
                k: torch.zeros_like(v.data)
                for k, v in self.model.named_parameters()
                if v.requires_grad and 'attention_bridge' in k
            }
        # /LCA
        self.optim.zero_grad()
        trainEnum = enumerate(self._accum_batches(train_iter))

        while True:

            step = self.optim.training_step
            self._maybe_update_dropout(step)

            # LCA:
            #   1. get θ_t
            #   2. train step; grad = ∇_t, params: θ_{t+1}
            #   3. log changelca_params = (θ_{t+1} - θ_t) * ∇_t
            if self.lca_loginterval > 0:
                theta_t = {
                    k: v.data.clone()
                    for k, v in self.model.named_parameters()
                    if v.requires_grad and k.find('attention_bridge') >= 0
                }
            # /LCA

            i, (batches_with_meta, normalization) = next(trainEnum)
            # logger.info(f'{j} {i} global_rank {global_rank}')

            self._gradient_accumulation_overLangPair(
                batches_with_meta,
                normalization,
                total_stats,
                report_stats,
            )

            for encoder_id, (_, group) in self.my_encoder_groups:
                grads = [
                    p.grad.data
                    for name, p in self.model.encoder[f'encoder{encoder_id}'].named_parameters()
                    if 'embeddings' not in name and p.requires_grad and p.grad is not None
                ]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, rescale_denom=1.0, group=group)
            for decoder_id, (_, group) in self.my_decoder_groups:
                grads = [
                    p.grad.data
                    for name, p in self.model.decoder[f'decoder{decoder_id}'].named_parameters()
                    if 'embeddings' not in name and p.requires_grad and p.grad is not None
                ]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, rescale_denom=1.0, group=group)
            for src_emb_id, (_, group) in self.my_src_emb_groups:
                src_lang, encoder_id = src_emb_id
                embs = self.model.encoder[f'encoder{encoder_id}'].embeddings[f'embeddings{src_lang}']
                grads = [p.grad.data for p in embs.parameters() if p.requires_grad and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, rescale_denom=1.0, group=group)
            for tgt_emb_id, (_, group) in self.my_tgt_emb_groups:
                tgt_lang, decoder_id = tgt_emb_id
                embs = self.model.decoder[f'decoder{decoder_id}'].embeddings[f'embeddings{tgt_lang}']
                grads = [p.grad.data for p in embs.parameters() if p.requires_grad and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, rescale_denom=1.0, group=group)
                grads = [
                    p.grad.data
                    for p in self.model.generator[f'generator{tgt_lang}'].parameters()
                    if p.requires_grad and p.grad is not None
                ]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, rescale_denom=1.0, group=group)

            grads = [
                p.grad.data for p in self.model.attention_bridge.parameters() if p.requires_grad and p.grad is not None
            ]
            if grads and self.scheduler.node_rank is not None and self.scheduler.local_rank is not None:
                # a group is not specified: reduce across all devices
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, rescale_denom=1.0)

            self._maybe_update_stats_from_parameters(report_stats, self.model.named_parameters())

            # LCA
            if (self.lca_loginterval > 0) and (step % self.lca_loginterval == 0) and (global_rank == 0):
                for k, v in self.model.named_parameters():
                    if not v.requires_grad or isinstance(v.grad, type(None)) or k.find('attention_bridge') < 0:
                        continue
                    lca_params[k] = (v.data - theta_t[k]) * v.grad

                # dump logs at each checkpoint and 10 times during training
                dump_logs = (step % (train_steps // 10) == 0) or (
                    save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0
                )
                dumppath = f'{self.model_saver.base_path}_lca_logs.json'
                onmt.utils.logging.log_lca_values(step, lca_logs, lca_params, dumppath, dump_logs)

            # /LCA

            self.optim.step()
            self.optim.zero_grad()

            if step % 1000 == 0:
                logger.info(f'After gradient sync {step}')
                for name, p in self.model.named_parameters():
                    logger.info(
                        f'{self.scheduler.node_rank}:{self.scheduler.local_rank}' f' {name}: {p.flatten()[:10]}'
                    )

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step,
                train_steps,
                self.optim.learning_rate(),
                report_stats,
            )

            # if valid_iter is not None and step % valid_steps == 0:
            #     if self.gpu_verbose_level > 0:
            #         logger.info('GpuRank %d: validate step %d'
            #                     % (self.gpu_rank, step))
            #     valid_stats = self.validate(
            #         valid_iter, moving_average=self.moving_average, sourceLang=sourceLang, targetLang=targetLang)
            #     if self.gpu_verbose_level > 0:
            #         logger.info('GpuRank %d: gather valid stat \
            #                         step %d' % (self.gpu_rank, step))
            #     valid_stats = self._maybe_gather_stats(valid_stats)
            #     if self.gpu_verbose_level > 0:
            #         logger.info('GpuRank %d: report stat step %d'
            #                     % (self.gpu_rank, step))
            #     self._report_step(self.optim.learning_rate(), #learning_rate_to_show, #self.optim.learning_rate(),
            #                       step, valid_stats=valid_stats)
            #     # Run patience mechanism
            #     if self.earlystopper is not None:
            #         self.earlystopper(valid_stats, step)
            #         # If the patience has reached the limit, stop training
            #         if self.earlystopper.has_stopped():
            #             break

            if self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None, sourceLang=None, targetLang=None):
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
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt

                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    # F-prop through the model.
                    outputs, attns = valid_model(
                        src, tgt, src_lengths, with_align=self.with_align, src_task=sourceLang, tgt_task=targetLang
                    )

                    # Compute loss.
                    _, batch_stats = self.valid_loss_md["valloss" + targetLang](batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _gradient_accumulation_overLangPair(
        self,
        batches_with_meta,
        normalization,
        total_stats,
        report_stats,
    ):
        for k, (batch, metadata) in enumerate(batches_with_meta):
            # logger.info(f'batch with metadata {metadata}')

            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

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

                # If truncated, don't backprop fully.
                if self.model.decoder[f'decoder{metadata.decoder_id}'].state is not None:
                    self.model.decoder[f'decoder{metadata.decoder_id}'].detach_state()

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
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_update_stats_from_parameters(self, report_stats, named_parameters):
        if self.report_manager is not None and self.report_stats_from_parameters:
            report_stats.update_from_parameters(named_parameters)

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step,
                num_steps,
                learning_rate,
                None if self.earlystopper is None else self.earlystopper.current_tolerance,
                report_stats,
                multigpu=self.n_gpu > 1,
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate,
                None if self.earlystopper is None else self.earlystopper.current_tolerance,
                step,
                train_stats=train_stats,
                valid_stats=valid_stats,
            )
