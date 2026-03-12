"""Module defining low-level comunication utilities (initialization, brodcasting, etc.)"""

import math
import os
import pickle
import signal
from collections import OrderedDict

import torch
import torch.distributed

from mammoth.distributed.contexts import DeviceContextEnum
from mammoth.utils.logging import init_logger, logger
from mammoth.utils.misc import set_random_seed
from mammoth.utils.profiling import get_roctx_range


def _detach_batch_tensors(obj):
    """
    Recursively convert all tensors in a batch to CPU NumPy arrays to prevent shared memory usage.
    This avoids /dev/shm race conditions in multi-node containerized environments.

    PyTorch's custom pickle implementation automatically uses shared memory (/dev/shm) for tensors,
    even after .detach().clone(). Converting to NumPy forces standard Python serialization.

    The consumer will convert NumPy arrays back to tensors and move to the appropriate device.
    """
    if isinstance(obj, torch.Tensor):
        # Convert to CPU NumPy array to bypass PyTorch's shared memory pickling
        # Store dtype as string to preserve it through serialization
        return {'__tensor__': True, 'data': obj.detach().cpu().numpy(), 'dtype': str(obj.dtype)}
    elif isinstance(obj, dict):
        return {k: _detach_batch_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # Handle namedtuples (which have _fields attribute)
        # Reconstruct by passing detached values to constructor
        return type(obj)(*(_detach_batch_tensors(item) for item in obj))
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_detach_batch_tensors(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects with attributes
        new_obj = type(obj).__new__(type(obj))
        for key, value in obj.__dict__.items():
            setattr(new_obj, key, _detach_batch_tensors(value))
        return new_obj
    else:
        return obj


def _reattach_batch_tensors(obj, pin_memory=False):
    """
    Recursively convert NumPy arrays back to PyTorch tensors.
    This is the inverse operation of _detach_batch_tensors.

    Args:
        obj: Object to process (may contain serialized tensors)
        pin_memory: If True, pin reconstructed tensors for faster GPU transfer

    Tensors are reconstructed on CPU; the consumer will move them to the appropriate device.
    """
    if isinstance(obj, dict):
        # Check if this is a serialized tensor
        if obj.get('__tensor__') is True:
            # Convert NumPy array back to tensor
            numpy_array = obj['data']
            dtype_str = obj['dtype']
            # Parse dtype string (e.g., "torch.float32" -> torch.float32)
            dtype = getattr(torch, dtype_str.replace('torch.', ''))
            tensor = torch.from_numpy(numpy_array).to(dtype)
            if pin_memory:
                tensor = tensor.pin_memory()
            return tensor
        else:
            # Regular dict, recurse
            return {k: _reattach_batch_tensors(v, pin_memory=pin_memory) for k, v in obj.items()}
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # Handle namedtuples
        return type(obj)(*(_reattach_batch_tensors(item, pin_memory=pin_memory) for item in obj))
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_reattach_batch_tensors(item, pin_memory=pin_memory) for item in obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects with attributes
        new_obj = type(obj).__new__(type(obj))
        for key, value in obj.__dict__.items():
            setattr(new_obj, key, _reattach_batch_tensors(value, pin_memory=pin_memory))
        return new_obj
    else:
        return obj


def multi_init(opts, global_rank, local_rank=None):
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip=opts.master_ip, master_port=opts.master_port
    )

    dist_world_size = opts.world_size

    # Prepare init_process_group arguments
    init_args = {
        'backend': opts.gpu_backend,
        'init_method': dist_init_method,
        'rank': global_rank,
        'world_size': dist_world_size,
    }

    # Add device_id for NCCL backend to avoid warnings
    # This tells NCCL which GPU this process should use
    if local_rank is not None and opts.gpu_backend == 'nccl':
        init_args['device_id'] = torch.device(f'cuda:{local_rank}')

    torch.distributed.init_process_group(**init_args)

    gpu_rank = torch.distributed.get_rank()

    return gpu_rank


def broadcast_tensors(tensors, src=0, group=None):
    for t in tensors:
        if group is None:
            torch.distributed.broadcast(t, src)
        else:
            torch.distributed.broadcast(t, src, group=group)


def externally_managed_reduce_and_rescale_grads(
    named_parameters,
    has_local_gradient: bool,
    gradient_norm: int,
    group=None,
):
    """
    Gradient synch tolerant to missing grads.

    Missing grads occur when some parameters are trained on some devices in
    a communication group but not on others, between two gradient synchs.

    The "managed" implementation relies on a deterministic sampling of tasks
    known to all devices. This allows a device to figure out that even though
    it didn't train some parameters itself, some other device did. In this case
    the device must send a dummy gradient of all zeros.

    Only if no device trains some parameters (or if the parameters exist on
    exactly one device) is it possible to skip communication entirely.

    Args:
        named_parameters: tuples of (str, Parameter) defining the parameters to consider
        group: torch.distributed communication group
    """
    require_grad = [(name, p) for (name, p) in named_parameters if p.requires_grad]
    if not require_grad:
        # Exit early if the component has no parameters that require a gradient
        return
    # Set missing gradients to zero
    for name, p in require_grad:
        if p.grad is None or not has_local_gradient:
            p.grad = torch.zeros_like(p)

    grads = [p.grad.data for name, p in require_grad]

    # All devices communicate either a real gradient or a dummy zeros of the same size
    all_reduce_and_rescale_tensors(grads, rescale_denom=gradient_norm, group=group)

    # Note: p.has_grad is not used in the "managed" implementation:
    # the optimizer can not use it to prevent the untrained components from being stepped


def all_reduce_and_rescale_tensors(
    tensors, rescale_denom, group=None, buffer_size=10485760
):
    """
    All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = (
        tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    )
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset : offset + numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        if group is None:
            torch.distributed.all_reduce(buffer_t[:offset])
        else:
            torch.distributed.all_reduce(buffer_t[:offset], group=group)
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset : offset + numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            if group is None:
                torch.distributed.all_reduce(t)
            else:
                torch.distributed.all_reduce(t, group=group)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if (
        not hasattr(all_gather_list, "_in_buffer")
        or max_size != all_gather_list._in_buffer.size()
    ):
        all_gather_list._in_buffer = torch.zeros(max_size, dtype=torch.uint8, device='cuda')
        all_gather_list._out_buffers = [
            torch.zeros(max_size, dtype=torch.uint8, device='cuda') for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError("encoded data exceeds max_size: {}".format(enc_size + 2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2 : enc_size + 2] = torch.tensor(list(enc), dtype=torch.uint8, device='cuda')

    torch.distributed.all_gather(out_buffers, in_buffer)

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2 : size + 2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


class WorldGroupGradientSync:
    """
    Replaces per-component allreduce calls with a single allreduce on the world group.

    Problem: With many languages, Mammoth creates hundreds of overlapping NCCL process groups
    (one per shared model component). At scale, per-component allreduce across these overlapping
    communicators causes NCCL deadlock/timeout.

    Solution: Pack all component gradients into a single flat buffer, do one allreduce on the
    world group (group=None), then unpack. One collective operation, zero overlapping groups.

    Trade-off: The buffer includes zeros for components a GPU doesn't own (~bandwidth overhead).
    This is acceptable because (a) compute dominates communication and (b) one large allreduce
    is more efficient per-byte than many small ones.
    """

    def __init__(self, all_components, model, global_rank):
        """
        Build the global buffer layout by gathering param counts across all GPUs.

        Args:
            all_components: list of all DistributedComponent objects (globally consistent order)
            model: the NMTModel (already on GPU)
            global_rank: this GPU's global rank
        """
        self.global_rank = global_rank
        self.roctx_range = get_roctx_range()

        # Step 1: For each component this GPU owns, compute param count
        my_param_counts = OrderedDict()
        for component in all_components:
            name = component.get_name()
            if global_rank in component.global_ranks:
                count = sum(p.numel() for _, p in component.named_parameters(model) if p.requires_grad)
                my_param_counts[name] = count
            else:
                my_param_counts[name] = 0

        # Step 2: All-gather param counts across all GPUs so every GPU knows every component's size
        # Use all_gather_object instead of all_gather_list: at large scale (many adapters per
        # language pair), the serialized dict can exceed the ~65 KB hard limit of all_gather_list.
        # all_gather_object has no size limit.
        all_counts = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(all_counts, my_param_counts)

        # Step 3: Compute global buffer layout
        # For each component, take the max param count across all GPUs that own it
        # (they should all agree, but max is safe)
        global_param_counts = OrderedDict()
        for name in sorted(my_param_counts.keys()):
            sizes = [counts.get(name, 0) for counts in all_counts]
            global_param_counts[name] = max(sizes)

        # Compute offsets in sorted name order
        self.component_layout = OrderedDict()  # {name: (offset, size)}
        offset = 0
        for name in sorted(global_param_counts.keys()):
            size = global_param_counts[name]
            if size > 0:
                self.component_layout[name] = (offset, size)
                offset += size

        self.total_size = offset

        # Step 4: Pre-allocate the flat buffer (on GPU, matching model dtype)
        # Find the dtype from any model parameter
        dtype = torch.float32
        for p in model.parameters():
            dtype = p.dtype
            break
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.buffer = torch.zeros(self.total_size, dtype=dtype, device=device)

        logger.info(
            f"WorldGroupGradientSync: rank {global_rank}, buffer size {self.total_size} elements "
            f"({self.total_size * self.buffer.element_size() / 1024 / 1024:.1f} MB), "
            f"{len(self.component_layout)} components"
        )

    def sync(self, model, all_gradient_syncs):
        """
        Perform a single world-group allreduce for all component gradients.

        Args:
            model: the NMTModel
            all_gradient_syncs: list of DistributedComponentGradientSync (for ALL components,
                including those this GPU doesn't own)
        """
        with self.roctx_range("world_group_sync"):
            # Step 1: Zero the buffer
            self.buffer.zero_()

            # Step 2: Pack gradients from owned components into the buffer
            with self.roctx_range("world_group_sync_pack"):
                for gradient_sync in all_gradient_syncs:
                    component = gradient_sync.component
                    name = component.get_name()
                    if name not in self.component_layout:
                        continue
                    if not gradient_sync.owns_component:
                        # This GPU doesn't own this component — contributes zeros (already zeroed)
                        continue
                    if not gradient_sync.has_local_gradient:
                        # This GPU owns the component but didn't train it this step — contributes zeros
                        continue

                    offset, size = self.component_layout[name]
                    pos = offset
                    for _, p in component.named_parameters(model):
                        if not p.requires_grad:
                            continue
                        numel = p.numel()
                        if p.grad is not None:
                            self.buffer[pos:pos + numel].copy_(p.grad.data.view(-1))
                        pos += numel

            # Step 3: Single allreduce on world group
            with self.roctx_range("world_group_sync_allreduce"):
                torch.distributed.all_reduce(self.buffer)

            # Step 4: Unpack from buffer back to gradients (only for owned components)
            with self.roctx_range("world_group_sync_unpack"):
                for gradient_sync in all_gradient_syncs:
                    component = gradient_sync.component
                    name = component.get_name()
                    if name not in self.component_layout:
                        continue
                    if not gradient_sync.owns_component:
                        continue

                    offset, size = self.component_layout[name]
                    pos = offset
                    for _, p in component.named_parameters(model):
                        if not p.requires_grad:
                            continue
                        numel = p.numel()
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad.data.copy_(
                            self.buffer[pos:pos + numel].view_as(p.grad.data) / gradient_sync.gradient_norm
                        )
                        pos += numel


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """init error handler"""
        import signal
        import threading

        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """error handler"""
        self.children_pids.append(pid)

    def error_listener(self):
        """error listener"""
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """signal handler"""
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def batch_producer(generator_to_serve, queue, semaphore, opts, device_id):
    """
    Produce batches to `queues` from `generator_to_serve` with background prefetching.

    Uses a background thread to prepare batches in advance, hiding disk I/O latency
    and preventing GPU stalls when waiting for the next batch.
    """
    # Set sharing strategy in spawned subprocess (not inherited from parent)
    torch.multiprocessing.set_sharing_strategy('file_system')

    log_level = "INFO" if opts.verbose or device_id == 0 else "WARNING"
    init_logger(opts.log_file, log_level=log_level)
    set_random_seed(opts.seed, False)
    logger.info("BATCH PRODUCER")
    logger.info(generator_to_serve)

    # Initialize ROCTx profiling markers (always active, controlled by rocprofv3 wrapper)
    roctx_range = get_roctx_range()

    # Get prefetch buffer size
    prefetch_buffer_size = getattr(opts, 'prefetch_buffer_size', 16)
    logger.info(f"BATCH PRODUCER {device_id} - Prefetching with buffer size {prefetch_buffer_size}")

    _batch_producer_with_prefetch(
        generator_to_serve, queue, semaphore, opts, device_id,
        prefetch_buffer_size, roctx_range
    )


def _batch_producer_with_prefetch(generator_to_serve, queue, semaphore, opts, device_id,
                                   prefetch_buffer_size, roctx_range):
    """
    Batch producer with background thread prefetching.

    Architecture:
    - Main thread: Pulls from prefetch_buffer → Puts in main queue for consumer
    - Prefetch thread: Reads from generator → Prepares batches → Puts in prefetch_buffer

    This decouples slow disk I/O (in prefetch thread) from queue serving (main thread),
    preventing GPU stalls when waiting for next batch.
    """
    import threading
    import queue as queue_module

    # Internal buffer for prefetched batches (thread-safe queue)
    prefetch_buffer = queue_module.Queue(maxsize=prefetch_buffer_size)
    stop_event = threading.Event()
    prefetch_exception = [None]  # Mutable container to share exceptions across threads

    def prefetch_worker():
        """
        Background worker thread that continuously fetches batches from generator.

        This runs concurrently with main thread, preparing batches in advance so
        they're ready when consumer needs them. The thread:
        1. Fetches batch from generator (may block on disk I/O)
        2. Detaches tensors to CPU (prevents shared memory issues)
        3. Puts prepared batch in prefetch_buffer
        """
        try:
            for batch, metadata, communication_batch_id in generator_to_serve:
                if stop_event.is_set():
                    break

                # Do expensive work (disk I/O, tensor detach) in background thread
                with roctx_range("batch_prefetch_read"):
                    batch_detached = _detach_batch_tensors(batch)
                    metadata_detached = _detach_batch_tensors(metadata)

                # Put in prefetch buffer (blocks if buffer is full)
                prefetch_buffer.put((batch_detached, metadata_detached, communication_batch_id))

        except Exception as e:
            # Capture exception to propagate to main thread
            logger.error(f"BATCH PRODUCER {device_id} - Prefetch thread exception: {e}")
            import traceback
            prefetch_exception[0] = (e, traceback.format_exc())
        finally:
            # Signal end of data
            prefetch_buffer.put(None)

    # Start prefetch thread
    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True, name=f"BatchPrefetch-{device_id}")
    prefetch_thread.start()
    logger.info(f"BATCH PRODUCER {device_id} - Started prefetch thread")

    try:
        batch_count = 0
        while True:
            # Check for exceptions in prefetch thread
            if prefetch_exception[0] is not None:
                exc, tb = prefetch_exception[0]
                logger.error(f"BATCH PRODUCER {device_id} - Prefetch thread failed:\n{tb}")
                raise exc

            # Get batch from prefetch buffer (blocks if buffer is empty)
            # This should rarely block since prefetch thread is continuously filling it
            with roctx_range("batch_queue_get"):
                item = prefetch_buffer.get()

            if item is None:
                # End of data signal from prefetch thread
                logger.info(f"BATCH PRODUCER {device_id} - Received end-of-data signal after {batch_count} batches")
                break

            batch_detached, metadata_detached, communication_batch_id = item
            batch_count += 1

            # Wait for space in main queue (blocks if consumer is slow)
            semaphore.acquire()

            # Put in main queue for consumer (GPU trainer)
            queue.put((batch_detached, metadata_detached, communication_batch_id))

    except KeyboardInterrupt:
        # Graceful shutdown on termination signal
        logger.info(f"BATCH PRODUCER {device_id} - Received shutdown signal, cleaning up...")
        stop_event.set()
    finally:
        # Stop prefetch thread
        stop_event.set()

        # Wait for prefetch thread to finish (with timeout)
        prefetch_thread.join(timeout=5.0)
        if prefetch_thread.is_alive():
            logger.warning(f"BATCH PRODUCER {device_id} - Prefetch thread did not exit cleanly")

        # ROCTx profiling markers always active (traces saved by rocprofv3 if enabled)
        logger.info(f"BATCH PRODUCER {device_id} - Batch producer complete (produced {batch_count} batches)")

        # Send sentinel value to signal end of data stream
        try:
            semaphore.acquire()
            queue.put(None)
            logger.info(f"BATCH PRODUCER {device_id} - Shutdown complete")
        except Exception:
            # Ignore errors during shutdown cleanup
            pass


def consumer(
    process_fn,
    opts,
    device_context,
    error_queue,
    batch_queue,
    semaphore,
    task_queue_manager,
    frame_checkpoint,
    frame_checkpoint_path,
):
    """Run `process_fn` on `device_id` with data from `batch_queue`."""
    # Set sharing strategy in spawned subprocess (not inherited from parent)
    torch.multiprocessing.set_sharing_strategy('file_system')

    try:
        logger.info(
            f"global_rank {device_context.global_rank} "
            f"node_rank {device_context.node_rank} "
            f"local_rank {device_context.local_rank}"
        )
        logger.info(f"opts.gpu_ranks {opts.gpu_ranks}")
        if device_context.context == DeviceContextEnum.MULTI_GPU:
            multi_init(opts, device_context.global_rank, device_context.local_rank)
        # error_queue not passed (is this intentional?)
        process_fn(
            opts,
            device_context=device_context,
            batch_queue=batch_queue,
            semaphore=semaphore,
            task_queue_manager=task_queue_manager,
            frame_checkpoint=frame_checkpoint,
            frame_checkpoint_path=frame_checkpoint_path,
        )

    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((device_context.node_rank, traceback.format_exc()))
