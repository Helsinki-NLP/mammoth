""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""
import math
import os
import pickle
import signal
import torch.distributed

from argparse import Namespace
from collections import OrderedDict
from itertools import compress, cycle, islice
from onmt.inputters.corpus import get_corpus
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import set_random_seed
from typing import Any, Dict, Optional


def is_master(opt, device_id):
    return opt.gpu_ranks[device_id] == 0


def multi_init(opt, device_id):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=opt.master_ip, master_port=opt.master_port
    )

    dist_world_size = opt.world_size
    torch.distributed.init_process_group(
        backend=opt.gpu_backend,
        init_method=dist_init_method,
        rank=device_id,
        world_size=dist_world_size,
    )

    gpu_rank = torch.distributed.get_rank()

    return gpu_rank


def all_reduce_tensors_init(tensors, numtoaverage, group=None):
    for t in tensors:
        if group is None:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        else:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)


def all_reduce_and_rescale_tensors(tensors, rescale_denom, group=None,
                                   buffer_size=10485760):
    """
    All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
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
        if group is None:
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

    ll = list(generator_to_serve_map.keys())
    train_iters = {
        k: (filter(pred, enumerate(f)))
        for k, f in generator_to_serve_map.items()
    }

    sizeLL = len(ll)

    first = True
    while True:
        for idx in range(0, sizeLL):
            train_enum = train_iters[ll[idx]]

            def next_batch(langPairName):
                # NOTE: stride (if needed) is handled at the
                # generator (train_iter) level
                new_batch = next(train_enum)
                semaphore.acquire()
                return new_batch[1], langPairName

            if first:
                b, langPairName = next_batch(ll[idx])
                first = False
            b.dataset = None
            # Move batch to correspond device_id when consumer iterate
            # hack to dodge unpicklable `dict_keys`
            b.fields = list(b.fields)
            queue.put((b, langPairName))
            b, langPairName = next_batch(ll[idx])


def consumer(
    process_fn, opt, device_id, error_queue, batch_queue, semaphore, node_rank
):  # noqa: E501
    """Run `process_fn` on `device_id` with data from `batch_queue`."""
    try:
        multi_init(opt, device_id)
        process_fn(
            opt,
            device_id=device_id,
            batch_queue=batch_queue,
            semaphore=semaphore,
            nodeRank=node_rank,
        )

    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class Scheduler:
    def __init__(
        self,
        opt: Namespace,
        node_rank: Optional[int] = None,
        local_rank: Optional[int] = None
    ):
        """
        Has the responsibility for all resources that need to be
        consistently assigned to nodes and GPUs.
        This includes data, parameters, and vocabularies.

        `local_rank` is the local rank of the GPU on this node.
        When `node_rank` and `local_rank` are given, the methods return only
        the items needed in the specified process.
        When set to None, all items are returned.
        """
        self.opt = opt
        self.node_rank = node_rank
        self.local_rank = local_rank

        self.gpus_per_node = len(self.opt.gpu_ranks)
        if self.gpus_per_node > 0:
            self.n_nodes = self.opt.world_size // self.gpus_per_node
        else:
            self.n_nodes = 1
            self.gpus_per_node = 1

        print(f'in scheduler: node_rank {node_rank} local_rank {local_rank}')
        assert node_rank is None or 0 <= node_rank < self.n_nodes
        assert local_rank is None or 0 <= local_rank < self.gpus_per_node
        # TODO: All the configuration lists should be the same length
        self.n_tasks = len(self.opt.src_tgt)

        # When --node_gpu is not set, assume an assigment that fills gpus in rank order
        if 'node_gpu' not in self.opt:
            self.opt.node_gpu = self._default_node_gpu()

        self.lang_pairs = [lang_pair.split('-') for lang_pair in self.opt.src_tgt]
        if 'enc_sharing_group' in self.opt:
            self.encoder_ids = self.opt.enc_sharing_group
        else:
            # if no encoder sharing groups are defined, encoders are language specific
            self.encoder_ids = [src_lang for src_lang, tgt_lang in self.lang_pairs]
        if 'dec_sharing_group' in self.opt:
            self.decoder_ids = self.opt.dec_sharing_group
        else:
            # if no decoder sharing groups are defined, decoders are language specific
            self.decoder_ids = [tgt_lang for src_lang, tgt_lang in self.lang_pairs]

        # A list of booleans, selecting only relevant parts of the configuration lists
        self._selector = self._get_selector(self.node_rank, self.local_rank)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'..., node_rank={self.node_rank}, local_rank={self.local_rank})'
        )

    def _get_selector(self, node_rank: Optional[int], local_rank: Optional[int]):
        if node_rank is None or local_rank is None:
            # Keep all items in global mode
            return [True] * self.n_tasks
        my_id = f'{node_rank}:{local_rank}'
        return [assignment == my_id for assignment in self.opt.node_gpu]

    def _default_node_gpu(self):
        def yield_each_gpu():
            for node_rank in range(self.n_nodes):
                for local_rank in range(self.gpus_per_node):
                    yield f'{node_rank}:{local_rank}'

        # yield GPUs in rank order, repeat as necessary
        return list(islice(cycle(yield_each_gpu()), self.n_tasks))

    def create_all_distributed_groups(
        self,
        new_group_func=torch.distributed.new_group,
    ):
        # encoder_id -> set of global_ranks
        encoder_to_gpus = OrderedDict()
        decoder_to_gpus = OrderedDict()
        for encoder_id in self.encoder_ids:
            encoder_to_gpus[encoder_id] = set()
        for decoder_id in self.decoder_ids:
            decoder_to_gpus[decoder_id] = set()
        for node_rank in range(self.n_nodes):
            for local_rank in range(self.gpus_per_node):
                global_rank = node_rank * self.gpus_per_node + local_rank
                selector = self._get_selector(node_rank, local_rank)

                encoders_on_this_gpu = compress(self.encoder_ids, selector)
                for encoder_id in encoders_on_this_gpu:
                    encoder_to_gpus[encoder_id].add(global_rank)

                decoders_on_this_gpu = compress(self.decoder_ids, selector)
                for decoder_id in decoders_on_this_gpu:
                    decoder_to_gpus[decoder_id].add(global_rank)

        encoder_to_group = OrderedDict()
        for encoder_id, global_ranks in encoder_to_gpus.items():
            if len(global_ranks) < 2:
                # only create a process group if the component is on 2 or more gpus
                continue
            sorted_global_ranks = list(sorted(global_ranks))
            encoder_to_group[encoder_id] = new_group_func(sorted_global_ranks)

        decoder_to_group = OrderedDict()
        for decoder_id, global_ranks in decoder_to_gpus.items():
            if len(global_ranks) < 2:
                # only create a process group if the component is on 2 or more gpus
                continue
            sorted_global_ranks = list(sorted(global_ranks))
            decoder_to_group[decoder_id] = new_group_func(sorted_global_ranks)

        return encoder_to_group, decoder_to_group

    def get_distributed_groups(
        self,
        new_group_func=torch.distributed.new_group,
    ):
        """
        Returns pairs of (component_id, process_group).
        Only components present on this GPU are returned.
        The pairs are returned in a consistent order across GPUs.
        """
        encoder_to_group, decoder_to_group = self.create_all_distributed_groups(new_group_func)
        my_encoder_ids = set(compress(self.encoder_ids, self._selector))
        my_encoder_groups = [
            (encoder_id, group) for (encoder_id, group) in encoder_to_group.items()
            if encoder_id in my_encoder_ids
        ]
        my_decoder_ids = set(compress(self.decoder_ids, self._selector))
        my_decoder_groups = [
            (decoder_id, group) for (decoder_id, group) in decoder_to_group.items()
            if decoder_id in my_decoder_ids
        ]
        return my_encoder_groups, my_decoder_groups

    def get_corpora(self, is_train=False) -> Dict[str, Any]:
        corpus_ids = self.opt.data.keys()
        my_corpus_ids = compress(corpus_ids, self._selector)
        return {
            corpus_id: get_corpus(self.opt, corpus_id, is_train=is_train)
            for corpus_id in my_corpus_ids
        }

    def get_vocabularies(self, opt: Namespace, side: str):
        my_lang_pairs = compress(self.lang_pairs, self._selector)
        result = []
        for lang_pair in my_lang_pairs:
            src_lang, tgt_lang = lang_pair
            lang = src_lang if side == 'src' else tgt_lang
            vocab_path = opt.__getattribute__(f'{side}_vocab')[lang]
            result.append((lang, vocab_path))
        return result

    def get_fields(self, side: str, fields_dict):
        """Returns a list of tuples: (side, lang, component_id, fields).
        side:           Either 'src' or 'tgt'.
        lang:           The language code. Vocabularies are language specific.
        component_id:   The encoder or decoder id. Embeddings are stored in
                        the encoders/decoders, so that component needs to be identified
                        in order to access the correct embeddings,
                        even if the embeddings are language specific.
        fields:         The actual Fields.
        """
        my_lang_pairs = compress(self.lang_pairs, self._selector)
        component_ids = self.encoder_ids if side == 'src' else self.decoder_ids
        my_component_ids = compress(component_ids, self._selector)
        seen = set()
        result = []
        for lang_pair, component_id in zip(my_lang_pairs, my_component_ids):
            src_lang, tgt_lang = lang_pair
            lang = src_lang if side == 'src' else tgt_lang
            if not (side, lang, component_id) in seen:
                result.append((side, lang, component_id, fields_dict[(side, lang)]))
            seen.add((side, lang, component_id))
        return result

    def get_dataset_specs(self, fields_dict):
        my_lang_pairs = compress(self.lang_pairs, self._selector)
        my_encoder_ids = compress(self.encoder_ids, self._selector)
        my_decoder_ids = compress(self.decoder_ids, self._selector)
        corpus_ids = self.opt.data.keys()
        my_corpus_ids = compress(corpus_ids, self._selector)
        corpus_dict = self.get_corpora(is_train=True)

        selected = [my_lang_pairs, my_encoder_ids, my_decoder_ids, my_corpus_ids]

        result = []
        for lang_pair, encoder_id, decoder_id, corpus_id in zip(*selected):
            src_lang, tgt_lang = lang_pair
            result.append((
                lang_pair,
                encoder_id,
                decoder_id,
                corpus_id,
                corpus_dict[corpus_id],
                fields_dict[('src', src_lang)],
                fields_dict[('tgt', tgt_lang)],
            ))
        return result

    def get_encoders(self):
        # TODO: also return how many times each component occurs, for normalization?
        my_encoder_ids = compress(self.encoder_ids, self._selector)
        return my_encoder_ids

    def get_decoders(self):
        my_decoder_ids = compress(self.decoder_ids, self._selector)
        return my_decoder_ids

    def get_generators(self):
        my_lang_pairs = compress(self.lang_pairs, self._selector)
        return [tgt_lang for (src_lang, tgt_lang) in my_lang_pairs]
