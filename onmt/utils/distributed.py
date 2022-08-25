""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""
import math
import numpy as np
import os
import pickle
import signal
import torch.distributed

from abc import ABC, abstractmethod
from argparse import Namespace
from collections import OrderedDict
from itertools import compress, cycle, islice
from typing import Any, Dict, Optional, List

from onmt.inputters.corpus import get_corpus
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import set_random_seed


def is_master(global_rank):
    return global_rank == 0


def multi_init(opt, global_rank):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip=opt.master_ip, master_port=opt.master_port)

    dist_world_size = opt.world_size
    torch.distributed.init_process_group(
        backend=opt.gpu_backend,
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


def only_ready_reduce_and_rescale_grads(named_parameters, group=None):
    """
    Gradient synch tolerant to missing grads.

    Missing grads occur when some parameters are not trained between two
    gradient synchs, e.g. the embeddings of a low-resource language with low
    sampling weight.

    The algorithm first uses the 'has_grad' attribute set by the forward hook
    'has_grad_hook'. This hook ensures that all parameters of the modules
    selected for use during the current training computation have 'has_grad'
    set to True. This gives the list of parameters that have been trained on
    this device ("ready").

    A bit mask covering the parameters that are ready on this device is
    communicated to the other devices in the group. The bit masks are reduced
    using summation. The sum gives the number of real gradients for that
    parameter, and can be used for normalization.

    If a parameter is ready on any device, all devices communicate a value.
    Devices on which the parameter is ready communicate the actual gradient,
    while devices on which it is not ready communicate a dummy zero tensor
        instead. The sum computed previously is used for normalization.

    Args:
        named_parameters: tuples of (str, Parameter) defining the parameters to consider
        group: torch.distributed communication group
    """
    # Set missing gradients to zero, keeping track of true gradients
    require_grad = [(name, p) for (name, p) in named_parameters if p.requires_grad]
    if not require_grad:
        # Exit early if the component has no parameters that require a gradient
        return
    device = require_grad[0][1].device
    ready_list = []
    for name, p in require_grad:
        if hasattr(p, 'has_grad') and p.has_grad:
            ready_list.append(1.0)
        else:
            ready_list.append(0.0)
            if p.grad is None:
                p.grad = torch.zeros_like(p)

    # Communicate the ready bits, and reduce them using summation.
    # This gives the number of non-dummy gradients participating, for normalization
    ready_t = torch.tensor(ready_list).to(device)
    if group is None:
        torch.distributed.all_reduce(ready_t)
    else:
        torch.distributed.all_reduce(ready_t, group=group)
    rescale_denoms = ready_t  # after reduction

    # Omit if all nodes sent a zero ready bit
    grads = [p.grad.data for name, p in require_grad]
    grads = [grad for (grad, denom) in zip(grads, rescale_denoms) if denom > 0]
    rescale_denoms = [denom for denom in rescale_denoms if denom > 0]
    assert len(grads) == len(rescale_denoms)
    if len(grads) == 0:
        return

    # All devices communicate either a real gradient or a dummy zeros of the same size
    # Can not use rescale_denom, as each grad may have its own denominator
    all_reduce_and_rescale_tensors(grads, rescale_denom=1)

    # Normalize using the previously computed values
    for grad, denom in zip(grads, rescale_denoms):
        if denom > 0:
            grad.div_(denom)
    # Note: p.has_grad is reused in the optimizer to prevent the untrained components from being stepped


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


def batch_producer(generator_to_serve, queue, semaphore, opt, device_id):
    """Produce batches to `queues` from `generator_to_serve`."""
    log_level = "INFO" if opt.verbose or device_id == 0 else "WARNING"
    init_logger(opt.log_file, log_level=log_level)
    set_random_seed(opt.seed, False)
    logger.info("BATCH PRODUCER")
    logger.info(generator_to_serve)

    for batch, metadata, communication_batch_id in generator_to_serve:
        semaphore.acquire()
        batch.dataset = None
        # Move batch to correspond device_id when consumer iterate
        # hack to dodge unpicklable `dict_keys`
        batch.fields = list(batch.fields)
        queue.put((batch, metadata, communication_batch_id))


def consumer(process_fn, opt, global_rank, error_queue, batch_queue, semaphore, node_rank, local_rank):  # noqa: E501
    """Run `process_fn` on `device_id` with data from `batch_queue`."""
    try:
        logger.info(f'global_rank {global_rank} node_rank {node_rank} local_rank {local_rank}')
        logger.info(f'opt.gpu_ranks {opt.gpu_ranks}')
        multi_init(opt, global_rank)
        process_fn(
            opt,
            global_rank=global_rank,
            batch_queue=batch_queue,
            semaphore=semaphore,
            node_rank=node_rank,
            local_rank=local_rank,
        )

    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((opt.gpu_ranks[node_rank], traceback.format_exc()))


class Scheduler:
    def __init__(self, opt: Namespace, node_rank: Optional[int] = None, local_rank: Optional[int] = None):
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

        self.tasks_per_communication_batch = opt.accum_count

        logger.info(f'in scheduler: node_rank {node_rank} local_rank {local_rank}')
        assert node_rank is None or 0 <= node_rank < self.n_nodes
        assert local_rank is None or 0 <= local_rank < self.gpus_per_node
        # TODO: All the configuration lists should be the same length
        self.n_tasks = len(self.opt.src_tgt)

        # When --node_gpu is not set, assume an assigment that fills gpus in rank order
        if not self.opt.node_gpu:
            self.opt.node_gpu = self._default_node_gpu()

        self.lang_pairs = [lang_pair.split('-') for lang_pair in self.opt.src_tgt]
        assert len(self.lang_pairs) == self.n_tasks
        if self.opt.enc_sharing_group:
            self.encoder_ids = self.opt.enc_sharing_group
            assert len(self.encoder_ids) == self.n_tasks
        else:
            # if no encoder sharing groups are defined, encoders are language specific
            self.encoder_ids = [src_lang for src_lang, tgt_lang in self.lang_pairs]
        if self.opt.dec_sharing_group:
            self.decoder_ids = self.opt.dec_sharing_group
            assert len(self.decoder_ids) == self.n_tasks
        else:
            # if no decoder sharing groups are defined, decoders are language specific
            self.decoder_ids = [tgt_lang for src_lang, tgt_lang in self.lang_pairs]

        # A list of booleans, selecting only relevant parts of the configuration lists
        self._selector = self._get_selector(self.node_rank, self.local_rank)

        self.strategy = SCHEDULING_STRATEGIES[opt.scheduling_strategy].from_opt(
            my_corpus_ids=list(compress(self.opt.data.keys(), self._selector)),
            opt=self.opt,
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(' f'..., node_rank={self.node_rank}, local_rank={self.local_rank})'

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
        # decoder_id -> set of global_ranks
        decoder_to_gpus = OrderedDict()
        # (src_lang, encoder_id) -> set of global_ranks
        src_emb_to_gpus = OrderedDict()
        # (tgt_lang, decoder_id) -> set of global_ranks
        tgt_emb_to_gpus = OrderedDict()
        for encoder_id in self.encoder_ids:
            encoder_to_gpus[encoder_id] = set()
        for decoder_id in self.decoder_ids:
            decoder_to_gpus[decoder_id] = set()

        for node_rank in range(self.n_nodes):
            for local_rank in range(self.gpus_per_node):
                global_rank = node_rank * self.gpus_per_node + local_rank
                selector = self._get_selector(node_rank, local_rank)

                encoders_on_this_gpu = list(compress(self.encoder_ids, selector))
                for encoder_id in encoders_on_this_gpu:
                    encoder_to_gpus[encoder_id].add(global_rank)

                decoders_on_this_gpu = list(compress(self.decoder_ids, selector))
                for decoder_id in decoders_on_this_gpu:
                    decoder_to_gpus[decoder_id].add(global_rank)

                lang_pairs_on_this_gpu = list(compress(self.lang_pairs, selector))
                for lang_pair, encoder_id in zip(lang_pairs_on_this_gpu, encoders_on_this_gpu):
                    src_lang, tgt_lang = lang_pair
                    key = (src_lang, encoder_id)
                    if key not in src_emb_to_gpus:
                        src_emb_to_gpus[key] = set()
                    src_emb_to_gpus[key].add(global_rank)

                for lang_pair, decoder_id in zip(lang_pairs_on_this_gpu, decoders_on_this_gpu):
                    src_lang, tgt_lang = lang_pair
                    key = (tgt_lang, decoder_id)
                    if key not in tgt_emb_to_gpus:
                        tgt_emb_to_gpus[key] = set()
                    tgt_emb_to_gpus[key].add(global_rank)

        result = {}
        components = [
            ('encoder', encoder_to_gpus),
            ('decoder', decoder_to_gpus),
            ('src_emb', src_emb_to_gpus),
            ('tgt_emb', tgt_emb_to_gpus),
        ]
        for key, component_to_gpus in components:
            component_to_group = OrderedDict()
            for component_id, global_ranks in component_to_gpus.items():
                if len(global_ranks) < 2:
                    # only create a process group if the component is on 2 or more gpus
                    continue
                sorted_global_ranks = list(sorted(global_ranks))
                min_rank = sorted_global_ranks[0]
                component_to_group[component_id] = (min_rank, new_group_func(sorted_global_ranks))
            result[key] = component_to_group

        return result

    def get_distributed_groups(
        self,
        new_group_func=torch.distributed.new_group,
    ):
        """
        Returns pairs of (component_id, process_group).
        Only components present on this GPU are returned.
        The pairs are returned in a consistent order across GPUs.
        """
        components_to_group = self.create_all_distributed_groups(new_group_func)
        logger.info(f'components_to_group: {components_to_group}')
        my_encoder_ids = set(compress(self.encoder_ids, self._selector))
        for encoder_id in my_encoder_ids:
            if encoder_id not in components_to_group['encoder']:
                logger.info(f'encoder {encoder_id} is on a single device')
        my_encoder_groups = [
            (encoder_id, group)
            for (encoder_id, group) in components_to_group['encoder'].items()
            if encoder_id in my_encoder_ids
        ]

        my_decoder_ids = set(compress(self.decoder_ids, self._selector))
        for decoder_id in my_decoder_ids:
            if decoder_id not in components_to_group['decoder']:
                logger.info(f'decoder {decoder_id} is on a single device')
        my_decoder_groups = [
            (decoder_id, group)
            for (decoder_id, group) in components_to_group['decoder'].items()
            if decoder_id in my_decoder_ids
        ]

        my_src_emb_ids = self.get_src_embs()
        for src_emb_id in my_src_emb_ids:
            if src_emb_id not in components_to_group['src_emb']:
                logger.info(f'src_emb {src_emb_id} is on a single device')
        my_src_emb_groups = [
            (src_emb_id, group)
            for (src_emb_id, group) in components_to_group['src_emb'].items()
            if src_emb_id in my_src_emb_ids
        ]

        my_tgt_emb_ids = self.get_tgt_embs()
        for tgt_emb_id in my_tgt_emb_ids:
            if tgt_emb_id not in components_to_group['tgt_emb']:
                logger.info(f'tgt_emb {tgt_emb_id} is on a single device')
        my_tgt_emb_groups = [
            (tgt_emb_id, group)
            for (tgt_emb_id, group) in components_to_group['tgt_emb'].items()
            if tgt_emb_id in my_tgt_emb_ids
        ]

        gpu_str = f'{self.node_rank}:{self.local_rank}'
        logger.info("{} my_encoder_ids: {}, my_decoder_ids: {}".format(gpu_str, my_encoder_ids, my_decoder_ids))
        logger.info("{} my_src_emb_ids: {}, my_tgt_emb_ids: {}".format(gpu_str, my_src_emb_ids, my_tgt_emb_ids))
        logger.info(
            "{} my_decoder groups: {}, my_encoder groups: {}".format(gpu_str, my_decoder_groups, my_encoder_groups)
        )
        logger.info(
            "{} my_src_emb groups: {}, my_tgt_emb_groups groups: {}".format(
                gpu_str, my_src_emb_groups, my_tgt_emb_groups
            )
        )
        return {
            'encoder': my_encoder_groups,
            'decoder': my_decoder_groups,
            'src_emb': my_src_emb_groups,
            'tgt_emb': my_tgt_emb_groups,
        }

    def get_corpora(self, is_train=False) -> Dict[str, Any]:
        corpus_ids = self.opt.data.keys()
        my_corpus_ids = compress(corpus_ids, self._selector)
        return {corpus_id: get_corpus(self.opt, corpus_id, is_train=is_train) for corpus_id in my_corpus_ids}

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
            result.append(
                (
                    src_lang,
                    tgt_lang,
                    encoder_id,
                    decoder_id,
                    corpus_id,
                    corpus_dict[corpus_id],
                    fields_dict[('src', src_lang)],
                    fields_dict[('tgt', tgt_lang)],
                )
            )
        return result

    def sample_corpus_ids(self, communication_batch_id: int):
        return self.strategy.sample_corpus_ids(
            self.tasks_per_communication_batch,
            communication_batch_id,
        )

    def get_encoders(self):
        # TODO: also return how many times each component occurs, for normalization?
        my_encoder_ids = compress(self.encoder_ids, self._selector)
        return my_encoder_ids

    def get_decoders(self):
        my_decoder_ids = compress(self.decoder_ids, self._selector)
        return my_decoder_ids

    def get_src_embs(self):
        my_lang_pairs = list(compress(self.lang_pairs, self._selector))
        my_encoder_ids = list(compress(self.encoder_ids, self._selector))
        return [(lang_pair[0], encoder_id) for (lang_pair, encoder_id) in zip(my_lang_pairs, my_encoder_ids)]

    def get_tgt_embs(self):
        my_lang_pairs = compress(self.lang_pairs, self._selector)
        my_decoder_ids = compress(self.decoder_ids, self._selector)
        return [(lang_pair[1], decoder_id) for (lang_pair, decoder_id) in zip(my_lang_pairs, my_decoder_ids)]

    def get_generators(self):
        my_lang_pairs = compress(self.lang_pairs, self._selector)
        return [tgt_lang for (src_lang, tgt_lang) in my_lang_pairs]


class SchedulingStrategy(ABC):
    @abstractmethod
    def __init__(self, my_corpus_ids: List[str], **kwargs):
        pass

    @classmethod
    @abstractmethod
    def from_opt(cls, my_corpus_ids: List[str], opt: dict):
        pass

    @abstractmethod
    def sample_corpus_ids(self, n_samples: int, communication_batch_id: int) -> List[str]:
        pass


class WeightedSamplingSchedulingStrategy(SchedulingStrategy):
    """
    Schedules tasks by sampling with replacement from a categorical distribution.
    The probabilities are found by normalizing the weights of all valid tasks (corpora).
    Valid tasks are those that are present on this device, and have already reached
    their curriculum starting point "introduce_at_training_step".
    """

    def __init__(
        self,
        my_corpus_ids: List[str],
        my_weights: List[float],
        my_introduce_at_training_step: List[int]
    ):
        self.my_corpus_ids = my_corpus_ids
        self.my_weights = my_weights
        self.my_introduce_at_training_step = my_introduce_at_training_step

        # Sanity check of weights and curriculum
        assert len(self.my_corpus_ids) == len(self.my_weights)
        assert len(self.my_corpus_ids) == len(self.my_introduce_at_training_step)
        if sum(my_weights) <= 0:
            raise ValueError('Can not set "weight" of all corpora on a device to zero')
        if all(x > 0 for x in my_introduce_at_training_step):
            raise ValueError('Can not set "introduce_at_training_step" of all corpora on a device to nonzero')
        if all(weight == 0 or start > 0 for (weight, start) in zip(my_weights, my_introduce_at_training_step)):
            raise ValueError('Invalid curriculum: no corpus is ready to start in the first step')

    @classmethod
    def from_opt(cls, my_corpus_ids: List[str], opt: dict):
        my_weights = [opt.data[corpus_id]['weight'] for corpus_id in my_corpus_ids]
        my_introduce_at_training_step = [
            opt.data[corpus_id]['introduce_at_training_step'] for corpus_id in my_corpus_ids
        ]
        return cls(my_corpus_ids, my_weights, my_introduce_at_training_step)

    def sample_corpus_ids(
        self,
        n_samples: int,
        communication_batch_id: int,
    ):
        weights = [
            weight if introduce_at_training_step <= communication_batch_id else 0
            for (corpus_id, weight, introduce_at_training_step) in zip(
                self.my_corpus_ids, self.my_weights, self.my_introduce_at_training_step
            )
        ]
        sum_w = sum(weights)
        assert sum_w > 0
        p = [weight / sum_w for weight in weights]
        # sampling with replacement from weighted corpora (language pairs)
        sampled_corpus_ids = np.random.choice(self.my_corpus_ids, size=n_samples, p=p)
        return sampled_corpus_ids


class RoundRobinSchedulingStrategy(SchedulingStrategy):
    """
    Schedules tasks (corpora) in a round-robin fashion.
    Yields a communication batch of n_samples at a time.
    When reaching the end of the list of tasks, starts over from the beginning.
    """

    def __init__(self, my_corpus_ids: List[str]):
        self.infinite_corpus_ids = cycle(my_corpus_ids)

    def from_opt(cls, my_corpus_ids: List[str], opt: dict):
        return cls(my_corpus_ids)

    def sample_corpus_ids(
        self,
        n_samples: int,
        communication_batch_id: int,
    ):
        return list(islice(self.infinite_corpus_ids, n_samples))


SCHEDULING_STRATEGIES = {
    'weighted_sampling': WeightedSamplingSchedulingStrategy,
    'roundrobin': RoundRobinSchedulingStrategy,
}
