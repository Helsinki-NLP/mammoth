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
from mammoth.utils.flops import compute_transformer_flops
from mammoth.inputters.vocab import HFTokenizerVocab

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

from mammoth.utils.profiling import get_profiler_range


class NanLossException(Exception):
    pass


def iter_on_device(iterator, device_context):
    """Move batches to device with ROCTx profiling annotation"""
    if device_context.is_gpu():
        device = torch.device(f'cuda:{device_context.local_rank}')
    else:
        device = torch.device('cpu')

    profiler_range = get_profiler_range()

    for batch, meta, comm_batch_id in iterator:
        with profiler_range("data_transfer_to_device"):
            # Use non_blocking=True for async transfer (requires pinned memory)
            # Pinned memory is enabled in train_single.py (_reattach_batch_tensors)
            batch_on_device = batch.to(device, non_blocking=True)
        yield batch_on_device, meta, comm_batch_id


def build_trainer(
    opts,
    device_context,
    model,
    vocabs_dict,
    optim,
    task_queue_manager,
    model_saver=None,
    world_group_sync=None,
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

    # Extract model config for FLOP counting
    x_opts = opts.x_transformers_opts if opts.x_transformers_opts else {}
    flops_config = {
        'model_dim': opts.model_dim,
        'enc_layers': sum(opts.enc_layers) if opts.enc_layers else 6,
        'dec_layers': sum(opts.dec_layers) if opts.dec_layers else 6,
        'vocab_size': opts.tgt_vocab_size or 0,
        'ff_mult': x_opts.get('ff_mult', x_opts.get('dec_ff_mult', 4.0)),
        'use_glu': x_opts.get('ff_glu', x_opts.get('dec_ff_glu', False)),
    }

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
        valid_metrics=opts.valid_metrics,
        valid_max_length=opts.valid_max_length,
        valid_max_batches=opts.valid_max_batches,
        valid_timeout=opts.valid_timeout,
        valid_decode_timeout=opts.valid_decode_timeout,
        valid_start=opts.valid_start,
        vocabs_dict=vocabs_dict,
        beam_size=opts.beam_size,
        max_length=opts.max_length,
        world_group_sync=world_group_sync,
        flops_config=flops_config,
        report_tflops=getattr(opts, 'report_tflops', True),
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
        valid_metrics=None,
        valid_max_length=None,
        valid_max_batches=None,
        valid_timeout=None,
        valid_decode_timeout=None,
        valid_start=0,
        vocabs_dict=None,
        beam_size=1,
        max_length=100,
        world_group_sync=None,
        flops_config=None,
        report_tflops=True,
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
        self.flops_config = flops_config or {}
        self.report_tflops = report_tflops
        self.valid_metrics = valid_metrics or []
        self.valid_max_length = valid_max_length
        self.valid_max_batches = valid_max_batches
        self.valid_timeout = valid_timeout
        self.valid_decode_timeout = valid_decode_timeout
        self.valid_start = valid_start
        self.vocabs_dict = vocabs_dict or {}
        self.beam_size = beam_size
        self.max_length = max_length
        self.world_group_sync = world_group_sync

        # Get ROCTx range function (markers always active, profiling controlled by rocprofv3)
        self.profiler_range = get_profiler_range()

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

        # Validate configuration for metric-based checkpoint strategies
        if self.model_saver is not None:
            save_strategy = getattr(self.model_saver, 'save_strategy', 'steps')
            if save_strategy in ['best_only', 'best_and_last', 'best_n']:
                if valid_iter is None:
                    raise ValueError(
                        f"save_strategy='{save_strategy}' requires validation, but no validation data provided. "
                        f"Either provide validation data or use save_strategy='steps'."
                    )
                if valid_steps <= 0:
                    raise ValueError(
                        f"save_strategy='{save_strategy}' requires valid_steps > 0, but got {valid_steps}"
                    )

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

            with self.profiler_range(f"data_preparation_step_{step}"):
                batches_with_meta = islice(train_iter, self.accum_count)
                # Convert to list to materialize all batches and measure data loading time
                batches_with_meta = list(batches_with_meta)

            batch_task_sample = self.task_queue_manager.sample_corpus_ids()
            my_task = batch_task_sample.tasks[self.task_queue_manager.global_rank]

            gradient_syncs = self.task_queue_manager.distributed_component_gradient_sync(batch_task_sample)

            with self.profiler_range(f"gradient_accumulation_step_{step}"):
                self._gradient_accumulation(
                    batches_with_meta,
                    total_stats,
                    report_stats,
                    my_task,
                    gradient_syncs,
                )

            with self.profiler_range(f"gradient_sync_step_{step}"):
                if self.world_group_sync is not None:
                    # World-group gradient sync: single allreduce on world group
                    # Replaces per-component allreduce to avoid NCCL deadlock at scale
                    self.world_group_sync.sync(self.model, gradient_syncs)
                else:
                    # Fallback: per-component gradient sync (non-distributed or legacy mode)
                    for idx, gradient_sync in enumerate(gradient_syncs):
                        component = gradient_sync.component
                        if not component.needs_communication():
                            continue

                        component_name = component.get_name() if hasattr(component, 'get_name') else f"component_{idx}"

                        with self.profiler_range(f"allreduce_{component_name}"):
                            params = component.named_parameters(self.model)
                            mammoth.distributed.externally_managed_reduce_and_rescale_grads(
                                named_parameters=params,
                                has_local_gradient=gradient_sync.has_local_gradient,
                                gradient_norm=gradient_sync.gradient_norm,
                                group=component.group,
                            )

            self._maybe_update_stats_from_parameters(report_stats, self.model.named_parameters())

            # Filter to owned components only: the optimizer should only step components this GPU owns
            owned_gradient_syncs = [gs for gs in gradient_syncs if gs.owns_component]
            with self.profiler_range(f"optimizer_step_{step}"):
                self.optim.externally_managed_step(owned_gradient_syncs)
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

            # Validation step - each device validates its own tasks with validation data
            if (step % valid_steps == 0) and (step >= self.valid_start):
                valid_stats = None

                # Only validate if this device has tasks with validation data
                if valid_iter is not None:
                    # Each device validates its own assigned tasks (those with validation paths)
                    if self.gpu_verbose_level > 0:
                        logger.info(f'{device_context.node_rank}:{device_context.local_rank} validate step {step}')

                    valid_stats = self.validate(
                        iter_on_device(valid_iter, device_context),
                        moving_average=self.moving_average,
                    )

                    # Store last validation stats for use at end of training
                    self._last_valid_stats = valid_stats

                    # Report validation stats for this device's tasks
                    if self.gpu_verbose_level > 0:
                        logger.info(f'{device_context.node_rank}:{device_context.local_rank} report valid stat step {step}')
                    self._report_step(
                        None,
                        step,
                        valid_stats=valid_stats,
                    )
                else:
                    # This device has no tasks with validation data
                    if self.gpu_verbose_level > 0:
                        logger.info(f'{device_context.node_rank}:{device_context.local_rank} skipping validation (no validation data)')

                # Synchronize all devices after validation to avoid timeouts
                # This barrier ensures all devices (with or without validation data) stay in sync
                if device_context.is_distributed():
                    with self.profiler_range("barrier_post_validation"):
                        torch.distributed.barrier()

                # Clean up GPU memory after validation to free any remaining tensors
                if device_context.is_gpu():
                    torch.cuda.empty_cache()

                # All ranks must call save_with_metric to participate in collective operations
                # (metric aggregation and data state gathering), but only master actually saves files
                if self.model_saver is not None:
                    self.model_saver.save_with_metric(
                        step, self._data_state, valid_stats, device_context, moving_average=self.moving_average
                    )

                # Early stopping: master evaluates, then broadcasts decision to all ranks
                # All ranks must break together to avoid NCCL deadlock from asymmetric exits
                should_stop = False
                if device_context.is_master():
                    if self.earlystopper is not None and valid_stats is not None:
                        self.earlystopper(valid_stats, step)
                        if self.earlystopper.has_stopped():
                            logger.info(f"Early stopping triggered at step {step}")
                            if self.earlystopper.current_step_best is not None:
                                logger.info(f"Best checkpoint was at step {self.earlystopper.current_step_best}")
                            should_stop = True

                if device_context.is_distributed():
                    stop_tensor = torch.tensor([1 if should_stop else 0], device='cuda')
                    torch.distributed.broadcast(stop_tensor, src=0)
                    should_stop = stop_tensor.item() == 1

                if should_stop:
                    break

            # Regular checkpoint saving (for save_strategy='steps' or when no validation has run yet)
            # This is skipped when using metric-based strategies after validation
            if self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0):
                # Only use regular save if not already saved via save_with_metric in validation
                if not (step % valid_steps == 0 and valid_iter is not None and device_context.is_master()):
                    self.model_saver.save(step, self._data_state, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        # Final checkpoint save — all ranks must participate because _save() contains
        # collective ops (all_gather_object for data state). Only master writes files.
        if self.model_saver is not None:
            if hasattr(self, '_last_valid_stats') and self._last_valid_stats is not None:
                self.model_saver.save_with_metric(
                    step, self._data_state, self._last_valid_stats, device_context, moving_average=self.moving_average
                )
            else:
                self.model_saver.save(step, self._data_state, moving_average=self.moving_average)

            # Log final best checkpoint info (master only)
            if device_context.is_master() and self.model_saver.best_checkpoint_step is not None:
                logger.info("=" * 80)
                logger.info("Training completed!")
                logger.info(f"Best checkpoint: step {self.model_saver.best_checkpoint_step}")
                logger.info(
                    f"Best {self.model_saver.metric_for_best_model}: "
                    f"{self.model_saver.best_metric_value:.4f}"
                )
                logger.info("=" * 80)

        if device_context.is_master() and self.report_manager is not None:
            self.report_manager.report_end(step)
        return total_stats

    def _compute_validation_metrics(self, predictions_by_direction, references_by_direction, valid_metrics):
        """Compute additional validation metrics like BLEU.

        Args:
            predictions_by_direction: Dict mapping (src_lang, tgt_lang) to list of predictions
            references_by_direction: Dict mapping (src_lang, tgt_lang) to list of references
            valid_metrics: List of metric names to compute

        Returns:
            Dict of metric names to values
        """
        metrics = {}

        if not SACREBLEU_AVAILABLE and 'bleu' in valid_metrics:
            logger.warning("sacrebleu not available, skipping BLEU computation")
            return metrics

        if 'bleu' in valid_metrics and SACREBLEU_AVAILABLE:
            bleu_scores = []

            # Compute BLEU for each translation direction separately
            for (src_lang, tgt_lang), preds in predictions_by_direction.items():
                refs = references_by_direction.get((src_lang, tgt_lang), [])

                if not preds or not refs:
                    logger.warning(f"Skipping BLEU for {src_lang}→{tgt_lang}: empty predictions or references")
                    continue

                try:
                    bleu = sacrebleu.corpus_bleu(preds, [refs])
                    metric_name = f'bleu/{src_lang}-{tgt_lang}'
                    metrics[metric_name] = bleu.score
                    bleu_scores.append(bleu.score)
                except Exception as e:
                    logger.warning(f"Error computing BLEU for {src_lang}→{tgt_lang}: {e}")

            # Compute average BLEU across all directions
            # if bleu_scores:
            #     metrics['bleu/avg'] = sum(bleu_scores) / len(bleu_scores)

        return metrics

    def _generate_predictions_autoregressive(self, batch, metadata, valid_model, decode_timeout=None):
        """Generate predictions using autoregressive decoding (like inference).

        Args:
            batch: Validation batch
            metadata: Task metadata
            valid_model: Model to use for generation

        Returns:
            List of predicted token sequences
        """
        from mammoth.translate.greedy_search import GreedySearch
        from mammoth.translate.beam_search import BeamSearch, GNMTGlobalScorer
        import time

        batch_size = batch.batch_size
        decode_start = time.monotonic()

        # Activate the correct components
        active_encoder = valid_model.encoder.activate(
            task_id=metadata.corpus_id,
            adapter_ids=metadata.encoder_adapter_ids,
        )
        active_decoder = valid_model.decoder.activate(
            task_id=metadata.corpus_id,
            adapter_ids=metadata.decoder_adapter_ids,
        )

        # Get device and dtype for mixed precision
        device = torch.device(f'cuda:{self.device_context.local_rank}' if self.device_context.is_gpu() else 'cpu')
        device_type = 'cuda' if self.device_context.is_gpu() else 'cpu'
        dtype = torch.float16 if self.model_dtype == 'fp16' else torch.bfloat16 if self.model_dtype == 'bf16' else torch.float32

        # Run encoder with autocast for consistent dtype
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.optim.amp):
            src = rearrange(batch.src.tensor, 't b 1 -> b t')
            src_mask = rearrange(batch.src.mask, 't b -> b t')
            encoder_output = active_encoder(x=src, mask=src_mask, return_embeddings=True)

            # Apply attention bridge if it exists
            if valid_model.attention_bridge is not None:
                encoder_output, alphas = valid_model.attention_bridge(encoder_output, src_mask)
                if valid_model.attention_bridge.is_fixed_length:
                    src_mask = None

        # Get special tokens - need to access vocabulary
        tgt_vocab = self.vocabs_dict.get(('tgt', metadata.tgt_lang))

        if isinstance(tgt_vocab, HFTokenizerVocab):
            # HF tokenizer vocab - use specials dictionary
            from mammoth.constants import DefaultTokens
            pad_idx = tgt_vocab.specials.get(DefaultTokens.PAD)
            bos_idx = tgt_vocab.specials.get(DefaultTokens.BOS)
            eos_idx = tgt_vocab.specials.get(DefaultTokens.EOS)
            unk_idx = tgt_vocab.specials.get(DefaultTokens.UNK)

            if None in (pad_idx, bos_idx, eos_idx, unk_idx):
                raise ValueError(
                    f"Missing special tokens in vocabulary. Found: "
                    f"PAD={pad_idx}, BOS={bos_idx}, EOS={eos_idx}, UNK={unk_idx}"
                )
        elif hasattr(tgt_vocab, 'stoi'):
            # Traditional vocab
            pad_idx = tgt_vocab.stoi.get('<pad>', 1)
            bos_idx = tgt_vocab.stoi.get('<s>', 0)
            eos_idx = tgt_vocab.stoi.get('</s>', 2)
            unk_idx = tgt_vocab.stoi.get('<unk>', 3)
        else:
            raise ValueError(f"Unknown vocabulary type: {type(tgt_vocab)}")

        # Get beam size and max_length from trainer attributes
        beam_size = self.beam_size
        max_length = self.valid_max_length if self.valid_max_length else self.max_length
        

        # Create global scorer with default parameters
        global_scorer = GNMTGlobalScorer(
            alpha=0.0,
            beta=0.0,
            length_penalty='none',
            coverage_penalty='none',
        )

        # Create decode strategy
        if beam_size == 1:
            # Greedy search for speed
            decode_strategy = GreedySearch(
                pad=pad_idx,
                bos=bos_idx,
                eos=eos_idx,
                unk=unk_idx,
                batch_size=batch_size,
                global_scorer=global_scorer,
                min_length=0,
                max_length=max_length,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                sampling_temp=0,
                keep_topk=1,
                keep_topp=0,
                beam_size=1,
                ban_unk_token=False,
                device=device,
            )
        else:
            # Beam search
            decode_strategy = BeamSearch(
                beam_size=beam_size,
                batch_size=batch_size,
                pad=pad_idx,
                bos=bos_idx,
                eos=eos_idx,
                unk=unk_idx,
                n_best=1,
                global_scorer=global_scorer,
                min_length=0,
                max_length=max_length,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                stepwise_penalty=False,
                ratio=-0.0,
                ban_unk_token=False,
                device=device,
                dtype=dtype,
            )

        # Initialize decode strategy
        decode_strategy.initialize(
            target_prefix=None,
            encoder_output=encoder_output,
            src_mask=src_mask,
        )

        # Autoregressive generation loop with autocast for consistent dtype
        for step in range(max_length):
            decoder_input = decode_strategy.alive_seq

            with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.optim.amp):
                # Forward pass through decoder
                logits_for_whole_sequence, new_cache = active_decoder(
                    decoder_input,
                    context=decode_strategy.encoder_output_tiled,
                    context_mask=decode_strategy.src_mask_tiled,
                    return_attn=False,
                    return_embeddings=False,
                    return_intermediates=True,
                    cache=decode_strategy.cache,
                    seq_start_pos=None,
                )

                if active_decoder.can_cache_kv:
                    decode_strategy.set_cache(new_cache)

                # Get logits for the last position
                logits = logits_for_whole_sequence[:, -1]
                log_probs = torch.log_softmax(logits, dim=-1)

            # Advance decode strategy
            decode_strategy.advance(log_probs)

            # Check if done
            if decode_strategy.is_finished.any():
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            # Check for timeout
            if decode_timeout:
                elapsed = time.monotonic() - decode_start
                if elapsed > decode_timeout:
                    logger.info(f"[DECODING TIMEOUT] time={elapsed}")
                    break

        # Extract predictions (take first from n_best for each batch item)
        predictions = []
        for batch_idx in range(batch_size):
            if len(decode_strategy.predictions[batch_idx]) > 0:
                # Take the best prediction (first one)
                pred_tokens = decode_strategy.predictions[batch_idx][0]
                if isinstance(pred_tokens, torch.Tensor):
                    pred_tokens = pred_tokens.tolist()
                # Remove BOS and EOS tokens
                pred_tokens = [t for t in pred_tokens if t not in (bos_idx, eos_idx, pad_idx)]
                predictions.append(pred_tokens)
            else:
                predictions.append([])

        return predictions

    def validate(self, valid_iter, moving_average=None, task=None):
        """Validate model using autoregressive generation.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model

        # Initialize collections for BLEU computation if needed
        # Group by translation direction to avoid mixing languages in BLEU computation
        predictions_by_direction = {}  # {(src_lang, tgt_lang): [predictions]}
        references_by_direction = {}   # {(src_lang, tgt_lang): [references]}
        compute_metrics = bool(self.valid_metrics)
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():

            # Tasks need not define validation paths: hence, a device need not contain
            # any validation path. This would cause statistics equals to 0 word seen,
            # which would then cause a zero devision when normalizing PPL per words.
            stats = None  # mammoth.utils.Statistics()

            import random
            import time

            # Log samples from only a few batches instead of every batch
            max_batches_to_log = 3  # Only log samples from first 3 batches
            batch_count = 0
            valid_start_time = time.monotonic()
            valid_max_time = self.valid_timeout
            valid_max_decode_time = self.valid_decode_timeout
            valid_max_batches = self.valid_max_batches

            for batch, metadata, _ in valid_iter:
                elapsed_time = time.monotonic() - valid_start_time
                if valid_max_time and elapsed_time > valid_max_time:
                    logger.info(f"[VALIDATION TIMEOUT] corpus_id={metadata.corpus_id}, direction={metadata.src_lang}->{metadata.tgt_lang}, time={elapsed_time:.1f}s after {batch_count} batches")
                    break
                if valid_max_batches and batch_count >= valid_max_batches:
                    logger.info(f"[VALIDATION MAX BATCHES] corpus_id={metadata.corpus_id}, direction={metadata.src_lang}->{metadata.tgt_lang}, batch-count={batch_count}")
                    break

                batch_count += 1
                

                # Only set logged_sample_idx for the first few batches
                logged_sample_idx = random.randint(0, batch.batch_size - 1) if batch_count <= max_batches_to_log else -1
                if stats is None:
                    stats = mammoth.utils.Statistics(n_correct=0)

                stats.n_src_words += batch.src.mask.sum().item()
                stats.n_sents += batch.batch_size
                src = batch.src.tensor
                src_mask = batch.src.mask
                decoder_input = batch.tgt.tensor[:-1]
                target = batch.tgt.tensor[1:]
                # if self.norm_method == "tokens":
                #     normalization = batch.tgt.mask.sum().item()
                # else:
                #     normalization = batch.batch_size

                # Determine device and dtype for mixed precision training
                device_type = 'cuda' if self.device_context.is_gpu() else 'cpu'
                dtype = torch.float16 if self.model_dtype == 'fp16' else torch.bfloat16 if self.model_dtype == 'bf16' else torch.float32

                with self.profiler_range("validation_forward_pass"):
                    with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.optim.amp):
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
                
                # Collect predictions and references for additional metrics using AUTOREGRESSIVE GENERATION
                if compute_metrics:
                    # Generate predictions autoregressively (like real inference)
                    pred_token_seqs = self._generate_predictions_autoregressive(batch, metadata, valid_model, valid_max_decode_time)

                    # Get target vocab for decoding
                    tgt_vocab = self.vocabs_dict.get(('tgt', metadata.tgt_lang))

                    if tgt_vocab is not None:
                        # Decode generated predictions and references
                        for b in range(batch.batch_size):
                            # Get reference sequence
                            ref_seq = target[:, b, 0].tolist()
                            # Filter out padding
                            ref_seq = [t for t in ref_seq if t != padding_idx]

                            # Get generated prediction (already filtered in _generate_predictions_autoregressive)
                            pred_seq = pred_token_seqs[b] if b < len(pred_token_seqs) else []

                            if pred_seq and ref_seq:
                                # Decode to text
                                if isinstance(tgt_vocab, HFTokenizerVocab):
                                    pred_text = tgt_vocab.decode_tokens(pred_seq, skip_special_tokens=True).strip()
                                    ref_text = tgt_vocab.decode_tokens(ref_seq, skip_special_tokens=True).strip()
                                else:
                                    # Traditional vocab
                                    pred_words = []
                                    ref_words = []
                                    for token in pred_seq:
                                        if hasattr(tgt_vocab, 'itos') and token < len(tgt_vocab.itos):
                                            word = tgt_vocab.itos[token]
                                            if not (word.startswith('<') and word.endswith('>')):
                                                pred_words.append(word)
                                    for token in ref_seq:
                                        if hasattr(tgt_vocab, 'itos') and token < len(tgt_vocab.itos):
                                            word = tgt_vocab.itos[token]
                                            if not (word.startswith('<') and word.endswith('>')):
                                                ref_words.append(word)
                                    pred_text = ' '.join(pred_words)
                                    ref_text = ' '.join(ref_words)

                                # Log randomly sampled example from a few batches only
                                if b == logged_sample_idx:
                                    logger.info(f"[VALIDATION SAMPLE] corpus_id={metadata.corpus_id}, direction={metadata.src_lang}->{metadata.tgt_lang}, example={b} (AUTOREGRESSIVE)")
                                    # logger.info(f"  pred_tokens: {pred_seq[:20]}")
                                    # logger.info(f"  ref_tokens: {ref_seq[:20]}")
                                    logger.info(f"  pred_text ({len(pred_text.split())} words): {pred_text[:200]}")
                                    logger.info(f"  ref_text ({len(ref_text.split())} words):  {ref_text[:200]}")

                                if pred_text and ref_text:
                                    # Group by translation direction to avoid mixing languages in BLEU
                                    direction_key = (metadata.src_lang, metadata.tgt_lang)
                                    if direction_key not in predictions_by_direction:
                                        predictions_by_direction[direction_key] = []
                                        references_by_direction[direction_key] = []
                                    predictions_by_direction[direction_key].append(pred_text)
                                    references_by_direction[direction_key].append(ref_text)
                                else:
                                    logger.warning(f"[VALIDATION] Empty text after decoding: pred_empty={not pred_text}, ref_empty={not ref_text}")
                            else:
                                logger.warning(f"[VALIDATION] Empty token sequences: pred_len={len(pred_seq)}, ref_len={len(ref_seq)}")
                    else:
                        logger.warning(f"Could not find vocabulary for target language '{metadata.tgt_lang}', skipping BLEU computation for this batch")
                
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        # Compute additional validation metrics
        if compute_metrics and predictions_by_direction:
            metrics = self._compute_validation_metrics(predictions_by_direction, references_by_direction, self.valid_metrics)
            if stats is not None:
                stats.validation_metrics.update(metrics)

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

        # Determine device and dtype for mixed precision training (once per accumulation)
        device_type = 'cuda' if self.device_context.is_gpu() else 'cpu'
        dtype = torch.float16 if self.model_dtype == 'fp16' else torch.bfloat16 if self.model_dtype == 'bf16' else torch.float32

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
            report_stats.n_sents += batch.batch_size

            # Track cumulative sentence count (this persists across report_stats resets)
            report_stats.cumulative_sents += batch.batch_size

            # logger.info(f'batch with metadata {metadata}')

            src = batch.src.tensor
            src_mask = batch.src.mask

            decoder_input = batch.tgt.tensor[:-1]
            target = batch.tgt.tensor[1:]
            # tgt_mask = batch.tgt.mask

            # shapes are: (t b i)   i.e.   (time, batch, vocab_index)

            with self.profiler_range(f"forward_pass_batch_{k}"):
                with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.optim.amp):
                    logits, decoder_output = self.model(
                        src=rearrange(src, 't b 1 -> b t'),
                        decoder_input=rearrange(decoder_input, 't b 1 -> b t'),
                        src_mask=rearrange(src_mask, 't b -> b t'),
                        metadata=metadata,
                    )
                    logits = rearrange(logits, 'b t i -> t b i')
                    decoder_output = rearrange(decoder_output, 'b t d -> t b d')

            with self.profiler_range(f"loss_computation_batch_{k}"):
                with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.optim.amp):
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
                with self.profiler_range(f"backward_pass_batch_{k}"):
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

        # Compute FLOPs for this step and record in report_stats
        if self.report_tflops and self.flops_config.get('model_dim', 0) > 0:
            n_src = report_stats.n_src_words
            n_tgt = report_stats.n_words
            # Use batch sequence lengths as approximation
            batch_size = report_stats.n_sents if report_stats.n_sents > 0 else 1
            src_seq_len = n_src // batch_size if batch_size > 0 else 0
            tgt_seq_len = n_tgt // batch_size if batch_size > 0 else 0
            report_stats.flops_per_step = compute_transformer_flops(
                n_src_tokens=n_src,
                n_tgt_tokens=n_tgt,
                src_seq_len=src_seq_len,
                tgt_seq_len=tgt_seq_len,
                **self.flops_config,
            )

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
