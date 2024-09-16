"""Module defining low-level comunication utilities (initialization, brodcasting, etc.)"""
import math
import os
import pickle
import signal

import torch
import torch.distributed

from mammoth.distributed.contexts import DeviceContextEnum
from mammoth.utils.logging import init_logger, logger
from mammoth.utils.misc import set_random_seed


def multi_init(opts, global_rank):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip=opts.master_ip, master_port=opts.master_port)

    dist_world_size = opts.world_size
    torch.distributed.init_process_group(
        backend=opts.gpu_backend,
        init_method=dist_init_method,
        rank=global_rank,
        world_size=dist_world_size,
    )

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


def all_reduce_and_rescale_tensors(tensors, rescale_denom, group=None, buffer_size=10485760):
    """
    All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset + numel].copy_(t.view(-1))
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
            t.view(-1).copy_(buffer_t[offset:offset + numel])
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
    if not hasattr(all_gather_list, '_in_buffer') or max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [torch.cuda.ByteTensor(max_size) for i in range(world_size)]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size + 2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2:size + 2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


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
    """Produce batches to `queues` from `generator_to_serve`."""
    log_level = "INFO" if opts.verbose or device_id == 0 else "WARNING"
    init_logger(opts.log_file, log_level=log_level)
    set_random_seed(opts.seed, False)
    logger.info("BATCH PRODUCER")
    logger.info(generator_to_serve)

    for batch, metadata, communication_batch_id in generator_to_serve:
        semaphore.acquire()
        # Move batch to correspond device_id when consumer iterate
        # hack to dodge unpicklable `dict_keys`
        # batch.fields = list(batch.fields)
        queue.put((batch, metadata, communication_batch_id))


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
    try:
        logger.info(
            f'global_rank {device_context.global_rank} '
            f'node_rank {device_context.node_rank} '
            f'local_rank {device_context.local_rank}'
        )
        logger.info(f'opts.gpu_ranks {opts.gpu_ranks}')
        if device_context.context == DeviceContextEnum.MULTI_GPU:
            multi_init(opts, device_context.global_rank)
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
