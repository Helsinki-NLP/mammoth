#!/usr/bin/env python
"""Training on a single process."""
import torch

from mammoth.model_builder import build_model, validate_optimizer_coverage
from mammoth.utils.optimizers import MultipleOptimizer
from mammoth.utils.misc import set_random_seed
from mammoth.trainer import build_trainer, iter_on_device
from mammoth.utils.model_saver import build_model_saver, load_parameters_from_checkpoint
from mammoth.utils.logging import init_logger, logger
from mammoth.utils.parse import ArgumentParser

import pickle
from collections import OrderedDict

from mammoth.distributed import _reattach_batch_tensors, WorldGroupGradientSync
from mammoth.distributed.communication import all_gather_list
from mammoth.inputters import DynamicDatasetIter
from mammoth.transforms import get_transforms_cls

from mammoth.utils.profiling import get_profiler_range


def set_cpu_affinity(local_rank):
    """Bind this process to the CPU cores nearest to its GPU (GCD) on LUMI-G.
    Each MI250X GCD is wired to a specific NUMA node; binding avoids
    cross-NUMA memory traffic that slows down data loading and kernel launches.
    Silently skips if not on LUMI or psutil is unavailable."""
    LUMI_GPU_CPU_map = {
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    if local_rank not in LUMI_GPU_CPU_map:
        return
    try:
        import psutil
        cpu_list = LUMI_GPU_CPU_map[local_rank]
        psutil.Process().cpu_affinity(cpu_list)
        logger.info(f"Bound local_rank {local_rank} to CPUs {cpu_list}")
    except (ImportError, AttributeError, OSError) as e:
        logger.info(f"CPU affinity binding skipped for local_rank {local_rank}: {e}")


def configure_process(opts, device_id):
    logger.info("logger set device {} ".format(device_id))
    if device_id >= 0:
        torch.cuda.set_device(device_id)
        # CPU affinity mapping is only valid when all 8 GCDs on a LUMI-G node are in use.
        # With fewer GPUs, the hardcoded GPU→NUMA mapping may not match actual device assignment.
        if sorted(opts.gpu_ranks) == list(range(8)):
            set_cpu_affinity(device_id)
        else:
            logger.info(
                f"CPU affinity binding skipped: gpu_ranks {opts.gpu_ranks} does not use all 8 GCDs"
            )
    set_random_seed(opts.seed, device_id >= 0)


def _get_model_opts(opts, frame_checkpoint=None):
    """Get `model_opts` to build model, may load from `checkpoint` if any."""
    if frame_checkpoint is not None:
        model_opts = ArgumentParser.checkpoint_model_opts(frame_checkpoint["opts"])
        ArgumentParser.update_model_opts(model_opts)
        ArgumentParser.validate_model_opts(model_opts)
        # if opts.tensorboard_log_dir == model_opts.tensorboard_log_dir and \
                # hasattr(model_opts, 'tensorboard_log_dir_dated'):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            # opts.tensorboard_log_dir_dated = model_opts.tensorboard_log_dir_dated
        # Override checkpoint's update_embeddings as it defaults to false
        # model_opts.update_vocab = opts.update_vocab
    else:
        model_opts = opts
    return model_opts


def _build_valid_iter(opts, vocabs_dict, transforms_cls, task_queue_manager):
    """Build iterator used for validation."""
    if not any(opts.tasks[corpus_id].get('path_valid_src', False) for corpus_id in opts.tasks.keys()):
        logger.info("Validation set missing for: {}".format(
            [
                corpus_id for corpus_id in opts.tasks.keys()
                if not opts.tasks[corpus_id].get('path_valid_src', False)
            ]
        ))
        return None
    logger.info("creating validation iterator")
    valid_iter = DynamicDatasetIter.from_opts(
        task_queue_manager=task_queue_manager,
        transforms_cls=transforms_cls,
        vocabs_dict=vocabs_dict,
        opts=opts,
        is_train=False,
        line_idx_restore=None,
    )
    return valid_iter


def init_distributed(model, task_queue_manager):
    """
    Synchronize initial weights across GPUs using a single world-group allreduce.

    For each shared component (on multiple devices), only the min_rank GPU
    contributes its weights into a flat buffer; all others contribute zeros.
    After allreduce(SUM), every GPU has the min_rank's weights and can unpack
    the components it owns.

    This replaces the previous per-component broadcast approach, which required
    per-component NCCL process groups. Creating hundreds of process groups via
    new_group() causes RCCL to deadlock at scale during communicator bootstrap.
    """
    all_components = task_queue_manager.distributed_components
    my_global_rank = task_queue_manager.global_rank

    # Only shared (multi-device) components need synchronization
    shared_components = [c for c in all_components if c.needs_communication()]
    if not shared_components:
        logger.info("init_distributed: no shared components to synchronize")
        return

    # Step 1: Compute param counts for each shared component on this GPU
    my_param_counts = OrderedDict()
    for component in shared_components:
        name = component.get_name()
        if my_global_rank in component.global_ranks:
            count = sum(p.numel() for _, p in component.named_parameters(model))
            my_param_counts[name] = count
        else:
            my_param_counts[name] = 0

    # Step 2: All-gather param counts so every GPU agrees on buffer layout
    # IMPORTANT: max_size must be identical across all ranks, otherwise all_gather
    # will deadlock due to mismatched buffer sizes. Compute it from a zero-valued
    # dict (same keys on all ranks → identical pickle size) plus generous padding.
    zero_dict = OrderedDict((name, 0) for name in my_param_counts.keys())
    base_enc_size = len(pickle.dumps(zero_dict))
    # Pad for varying integer sizes: large ints add ~10 bytes each in pickle encoding
    gather_max_size = base_enc_size + len(my_param_counts) * 20 + 256
    gather_max_size = max(gather_max_size, 4096)
    all_counts = all_gather_list(my_param_counts, max_size=gather_max_size)

    # Step 3: Build buffer layout (max param count per component across all GPUs)
    global_param_counts = OrderedDict()
    for name in sorted(my_param_counts.keys()):
        sizes = [counts.get(name, 0) for counts in all_counts]
        global_param_counts[name] = max(sizes)

    layout = OrderedDict()
    offset = 0
    for name in sorted(global_param_counts.keys()):
        size = global_param_counts[name]
        if size > 0:
            layout[name] = (offset, size)
            offset += size

    total_size = offset
    if total_size == 0:
        logger.info("init_distributed: no parameters to synchronize")
        return

    # Step 4: Allocate flat buffer on GPU
    dtype = torch.float32
    for p in model.parameters():
        dtype = p.dtype
        break
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    buffer = torch.zeros(total_size, dtype=dtype, device=device)

    # Step 5: Pack — only the min_rank of each component contributes weights
    for component in shared_components:
        name = component.get_name()
        if name not in layout:
            continue
        if my_global_rank not in component.global_ranks:
            continue
        if my_global_rank != component.min_rank:
            continue
        buf_offset, _ = layout[name]
        pos = buf_offset
        for _, p in component.named_parameters(model):
            numel = p.numel()
            buffer[pos:pos + numel].copy_(p.data.view(-1))
            pos += numel

    # Step 6: Single allreduce on world group (only min_rank contributed non-zero)
    torch.distributed.all_reduce(buffer)

    # Step 7: Unpack — each GPU copies weights for shared components it owns
    # (skip min_rank since it already has the correct weights)
    for component in shared_components:
        name = component.get_name()
        if name not in layout:
            continue
        if my_global_rank not in component.global_ranks:
            continue
        if my_global_rank == component.min_rank:
            continue
        buf_offset, _ = layout[name]
        pos = buf_offset
        for _, p in component.named_parameters(model):
            numel = p.numel()
            p.data.copy_(buffer[pos:pos + numel].view_as(p.data))
            pos += numel

    logger.info(
        f"init_distributed: synchronized {len(shared_components)} shared components "
        f"via world-group allreduce (buffer: {total_size * buffer.element_size() / 1024 / 1024:.1f} MB)"
    )


def main(
    opts,
    vocabs_dict,
    device_context,
    error_queue=None,
    batch_queue=None,
    semaphore=None,
    task_queue_manager=None,
    frame_checkpoint=None,
    frame_checkpoint_path=None,
):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opts`` has been validated and updated
    # at this point.
    # N.B: task_queue_manager is already local

    init_logger(opts.log_file, gpu_id=device_context.id)
    if device_context.is_distributed():
        configure_process(opts, device_context.local_rank)
        gpu_rank_t = torch.distributed.get_rank()
        logger.info("RANK GPU FROM TORCH %s", str(gpu_rank_t))

    transforms_cls = get_transforms_cls(opts._all_transform)
    model_opts = _get_model_opts(opts, frame_checkpoint=frame_checkpoint)

    task_queue_manager.create_all_distributed_components(
        use_attention_bridge=(model_opts.ab_layers is not None and len(model_opts.ab_layers) != 0),
    )

    # Build model.
    model = build_model(model_opts, opts, vocabs_dict, task_queue_manager)

    # Apply parameter freezing based on configuration
    if device_context.is_master():
        from mammoth.model_builder import freeze_model_components
        freeze_model_components(model, opts, task_queue_manager)

    logger.info("{} - Init model".format(device_context.id))
    world_group_sync = None
    if device_context.is_distributed():
        init_distributed(model, task_queue_manager)
        # Create world-group gradient sync: replaces per-component allreduce
        # with a single allreduce on the world group to avoid NCCL deadlock at scale
        world_group_sync = WorldGroupGradientSync(
            all_components=task_queue_manager.distributed_components,
            model=model,
            global_rank=task_queue_manager.global_rank,
        )
    enc, dec = model.count_parameters()
    logger.info("{} - total encoder parameters: {}".format(device_context.id, enc))
    logger.info("{} - total decoder parameters: {}".format(device_context.id, dec))

    # Build optimizer.
    logger.info("{} - Build optimizer".format(device_context.id))
    optim = MultipleOptimizer.from_opts(
        model,
        opts,
        task_queue_manager=task_queue_manager,
        frame_checkpoint=frame_checkpoint,
    )
    logger.info("{} - total optimized parameters: {}".format(
        device_context.id,
        optim.count_parameters()
    ))
    validate_optimizer_coverage(model, optim)

    # Load parameters from checkpoint
    if opts.train_from:
        # Determine whether to load optimizer state based on reset_optim
        # 'none' and 'keep_states' → load state
        # 'all' and 'states' → reset state
        should_reset_optimizer_state = opts.reset_optim in {'all', 'states'}

        load_parameters_from_checkpoint(
            frame_checkpoint_path=frame_checkpoint_path,
            model=model,
            optim=optim,
            task_queue_manager=task_queue_manager,
            reset_optim=should_reset_optimizer_state,
        )

        # Determine whether to load training step based on reset_optim
        # 'none' and 'states' → keep training step
        # 'all' and 'keep_states' → reset training step to 1
        if opts.reset_optim in {'none', 'states'}:
            optim.global_training_step = frame_checkpoint['global_training_step']
            logger.info(
                f"Loaded global_training_step={optim.global_training_step} from checkpoint "
                f"(reset_optim={opts.reset_optim})"
            )
        else:
            # Reset to step 1 (reset_optim in {'all', 'keep_states'})
            optim.global_training_step = 1
            logger.info(
                f"Reset global_training_step to 1 (reset_optim={opts.reset_optim}, "
                f"checkpoint was at step {frame_checkpoint['global_training_step']})"
            )

    # Build model saver
    model_saver = build_model_saver(
        model_opts,
        opts,
        model,
        vocabs_dict,
        optim,
        task_queue_manager=task_queue_manager,
    )

    logger.info("{} - Build trainer".format(device_context.id))
    trainer = build_trainer(
        opts,
        device_context,
        model,
        vocabs_dict,
        optim,
        task_queue_manager=task_queue_manager,
        model_saver=model_saver,
        world_group_sync=world_group_sync,
    )
    logger.info("{} - Trainer built".format(device_context.id))

    # It is no longer possible to train without multiprocessing
    assert batch_queue is not None
    assert semaphore is not None

    def _train_iter():
        profiler_range = get_profiler_range()

        while True:
            with profiler_range("batch_queue_get"):
                batch, metadata, communication_batch_id = batch_queue.get()

            # Reconstruct tensors from NumPy arrays (inverse of _detach_batch_tensors)
            # Pin memory for faster async GPU transfers
            with profiler_range("batch_tensor_reattach_from_cpu"):
                batch = _reattach_batch_tensors(batch, pin_memory=True)
                metadata = _reattach_batch_tensors(metadata, pin_memory=True)
            semaphore.release()
            # TODO: confirm that batch-providing corpus has already been to'd to the correct place
            yield batch, metadata, communication_batch_id

    train_iter = _train_iter()
    # train_iter = iter_on_device(train_iter, device_context)
    valid_iter = _build_valid_iter(opts, vocabs_dict, transforms_cls, task_queue_manager)

    # Perform validation before training starts if requested
    # Each device validates its own assigned tasks (those with validation paths)
    if opts.valid_at_start and valid_iter is not None:
        logger.info("{} - Performing validation before training starts".format(device_context.id))
        valid_stats = trainer.validate(iter_on_device(valid_iter, device_context))

        # Display BLEU validation results
        if valid_stats is not None:
            # Check for BLEU score in validation metrics
            if hasattr(valid_stats, 'validation_metrics') and valid_stats.validation_metrics:
                if 'bleu' in valid_stats.validation_metrics:
                    bleu_score = valid_stats.validation_metrics['bleu']
                    logger.info("{} - Pre-training validation BLEU: {:.2f}".format(
                        device_context.id, bleu_score))
                else:
                    logger.info("{} - Pre-training validation completed, but no BLEU score computed (check if sacrebleu is available)".format(device_context.id))
            else:
                logger.info("{} - Pre-training validation completed, but no BLEU metrics available".format(device_context.id))
        else:
            logger.info("{} - Pre-training validation returned no statistics".format(device_context.id))

        # Synchronize all devices after pre-training validation
        if device_context.is_distributed():
            torch.distributed.barrier()

    if len(opts.gpu_ranks):
        if device_context.is_master():
            logger.info('Starting training on GPU: %s' % opts.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opts.train_steps
    logger.info("{} - Starting training".format(device_context.id))

    # Training loop (ROCTx profiling markers always active; controlled by rocprofv3 wrapper)
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opts.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opts.valid_steps,
        device_context=device_context,
    )

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()

    # Properly cleanup PyTorch distributed resources before exit
    if device_context.is_distributed():
        logger.info("{} - Cleaning up distributed process group".format(device_context.id))
        torch.distributed.destroy_process_group()
        logger.info("{} - Distributed cleanup complete".format(device_context.id))
