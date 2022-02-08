""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""
import os
import signal
import math
import pickle

import torch.distributed
#import deepspeed
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from collections import OrderedDict

def is_master(opt, device_id):
    return opt.gpu_ranks[device_id] == 0


def multi_init(opt, device_id):
    #print("MULTI INIT2")
    #print(device_id)
    #print(opt.master_ip)
    #print(opt.master_port)
    #current_env = os.environ.copy()
    #current_env["RANK"] = str(device_id)

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=opt.master_ip,
        master_port=opt.master_port)

    #rank = int(os.environ['SLURM_PROCID'])
    #gpuz = rank % torch.cuda.device_count()

    dist_world_size = opt.world_size
    #logger.info("dist world size {} ".format(dist_world_size))
    #logger.info("dist_world_size in distributed : %s ", str(dist_world_size))
    #print("dist_world_size in distributed : "+str(dist_world_size))
    #print(rank)
    #logger.info("GPU RANKS in distributed %s - ", str(device_id))
    #print("GPU RANKS in distributed "+str(device_id))
    #print(opt.gpu_ranks[device_id])
    #dist_world_size
    #'env://'
    torch.distributed.init_process_group(backend=opt.gpu_backend, init_method=dist_init_method, rank=device_id, world_size=dist_world_size) # init_method='env://') #dist_init_method, rank=opt.gpu_ranks[device_id], world_size=dist_world_size)

    #deepspeed.init_distributed()

    gpu_rank = torch.distributed.get_rank()
    #logger.info("GPU RANKS in distributed out of torch %s - ", str(gpu_rank))
    #print("GPU RANKS in distributed out of torch ")#+ str(gpu_rank))
    #print(str(gpu_rank))
    #if not is_master(opt, device_id):
    #    logger.disabled = True

    return gpu_rank

def all_reduce_tensors_init(tensors, numtoaverage, group=None):
    for t in tensors:
        if group == None:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        else:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group) # 
        #t.div_(numtoaverage)

"""
def all_reduce_and_rescale_tensors(tensors, rescale_denom, group=None):
    for t in tensors:
        if group == None:
            torch.distributed.all_reduce(t)
        else:
            torch.distributed.all_reduce(t, group=group) 

"""
def all_reduce_and_rescale_tensors(tensors, rescale_denom, group=None,
                                   buffer_size=10485760):
#    All-reduce and rescale tensors in chunks of the specified size.
#
#    Args:
#        tensors: list of Tensors to all-reduce
#        rescale_denom: denominator for rescaling summed Tensors
#        buffer_size: all-reduce chunk size in bytes
#    
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset+numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        if group == None:
            torch.distributed.all_reduce(buffer_t[:offset])
        else:
            torch.distributed.all_reduce(buffer_t[:offset], group=group)
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset+numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            if group == None:
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
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size)
            for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size+2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2:size+2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def batch_producer(generator_to_serve_map, queue, semaphore, opt, device_id):
    """Produce batches to `queues` from `generator_to_serve`."""
    log_level = "INFO" if opt.verbose or device_id == 0 else "WARNING"
    init_logger(opt.log_file, log_level=log_level)
    set_random_seed(opt.seed, False)
    logger.info("BATCH PRODUCER")
    logger.info(generator_to_serve_map)

    def pred(x):
        """
        Filters batches that belong only
        to gpu_ranks of current node
        """
        for rank in opt.gpu_ranks:
            if x[0] % opt.world_size == rank:
                return True

    #for name, gen in generator_to_serve_map.items():
    #    generator_to_serve = gen
    ll = list(generator_to_serve_map.keys())
    train_iters = {k:
                (filter(pred, enumerate(f)))
                for k, f in generator_to_serve_map.items()}

    sizeLL = len(ll)
    #    generator_to_serve = filter(
    #        pred, enumerate(generator_to_serve))
    first = True
    while True:
        #            src_lang, tgt_lang = random.choices(langpairweights[0],weights=list(langpairweights[1]))[0]
        for idx in range(0, sizeLL):
            train_enum = train_iters[ll[idx]]
            #print("TRIAN ENUM DIST")
            #print(train_enum)
            def next_batch(langPairName):
                # NOTE: stride (if needed) is handled at the
                # generator (train_iter) level
                new_batch = next(train_enum)
                semaphore.acquire()
                return new_batch[1], langPairName
            
            if first:
                b, langPairName = next_batch(ll[idx])
                first = False
                #print(b)
                #print(langPairName)
            #    while True:
            b.dataset = None
            # Move batch to correspond device_id when consumer iterate
    
            # hack to dodge unpicklable `dict_keys`
            b.fields = list(b.fields)
            queue.put((b, langPairName))
            b, langPairName = next_batch(ll[idx])


def consumer(process_fn, opt, device_id, error_queue, batch_queue, semaphore, node_rank):  # noqa: E501
    """Run `process_fn` on `device_id` with data from `batch_queue`."""
    try:
        #print("MULTI INIT")
        gpu_rank = multi_init(opt, device_id)
        #if gpu_rank != opt.gpu_ranks[device_id]:
        #    raise AssertionError("An error occurred in \
        #          Distributed initialization")

        process_fn(opt, device_id=device_id,
                   batch_queue=batch_queue, semaphore=semaphore, nodeRank=node_rank)

    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        #error_queue.put((device_id, traceback.format_exc()))
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))
