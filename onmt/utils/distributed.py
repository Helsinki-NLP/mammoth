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
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from itertools import cycle, islice
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import set_random_seed
from pprint import pformat
from typing import Any, Optional, List


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


DatasetMetadata = namedtuple('DatasetMetadata', 'src_lang tgt_lang encoder_id decoder_id corpus_id')


@dataclass
class TaskSpecs():
    node_rank: int
    local_rank: int
    src_lang: str
    tgt_lang: str
    encoder_id: str
    decoder_id: str
    corpus_id: str
    weight: int
    corpus_opt: dict
    src_vocab: Any  # FIXME: type
    tgt_vocab: Any

    def get_serializable_metadata(self):
        """
        TaskSpecs contains objects that should not be serialized
        and sent over the multiprocessing message queue.
        The DatasetMetadata namedtuple can be serialized.
        """
        return DatasetMetadata(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            encoder_id=self.encoder_id,
            decoder_id=self.decoder_id,
            corpus_id=self.corpus_id,
        )


class TaskQueueManager:
    def __init__(
        self,
        tasks: List[TaskSpecs],
        gpus_per_node: int,
        n_nodes: int,
        components_to_gpus=None,
        components_to_groups=None,
        node_rank: Optional[int] = None,
        local_rank: Optional[int] = None
    ):
        """
        Schedules tasks (language pairs) to devices.
        Has the responsibility for all resources that need to be
        consistently assigned to nodes and GPUs.
        This includes data, parameters, and vocabularies.

        `local_rank` is the local rank of the GPU on this node.
        When `node_rank` and `local_rank` are given, the methods return only
        the items needed in the specified process.
        When set to None, all items are returned.
        """
        self.tasks = tasks
        self.node_rank = node_rank
        self.local_rank = local_rank

        self.gpus_per_node = gpus_per_node
        self.n_nodes = n_nodes

        logger.info(f'in task_queue_manager: node_rank {node_rank} local_rank {local_rank}')
        assert node_rank is None or 0 <= node_rank < self.n_nodes
        assert local_rank is None or 0 <= local_rank < self.gpus_per_node

        self.components_to_gpus = components_to_gpus
        self.components_to_groups = components_to_groups

    @classmethod
    def from_opt(cls, opt: Namespace):
        n_tasks = len(opt.src_tgt)
        gpus_per_node = len(opt.gpu_ranks)
        if gpus_per_node > 0:
            n_nodes = opt.world_size // gpus_per_node
        else:
            n_nodes = 1
            gpus_per_node = 1

        # When --node_gpu is not set, assume an assigment that fills gpus in rank order
        node_gpu = (
            [tuple(int(y) for y in x.split(':', 1)) for x in opt.node_gpu] if opt.node_gpu
            else cls._default_node_gpu(n_tasks, n_nodes, gpus_per_node)
        )
        lang_pairs = [lang_pair.split('-') for lang_pair in opt.src_tgt]

        if opt.enc_sharing_group:
            encoder_ids = opt.enc_sharing_group
        else:
            # if no encoder sharing groups are defined, encoders are language specific
            encoder_ids = [src_lang for src_lang, tgt_lang in lang_pairs]
        if opt.dec_sharing_group:
            decoder_ids = opt.dec_sharing_group
        else:
            # if no decoder sharing groups are defined, decoders are language specific
            decoder_ids = [tgt_lang for src_lang, tgt_lang in lang_pairs]

        corpus_ids = opt.data.keys()

        assert len(node_gpu) == n_tasks, f'{len(node_gpu)} != {n_tasks}'
        assert len(lang_pairs) == n_tasks, f'{len(lang_pairs)} != {n_tasks}'
        assert len(encoder_ids) == n_tasks, f'{len(encoder_ids)} != {n_tasks}'
        assert len(decoder_ids) == n_tasks, f'{len(decoder_ids)} != {n_tasks}'
        assert len(corpus_ids) == n_tasks, f'{len(corpus_ids)} != {n_tasks}'

        tasks = []
        for (
            (node_rank, local_rank),
            (src_lang, tgt_lang),
            encoder_id,
            decoder_id,
            corpus_id
        ) in zip(
            node_gpu,
            lang_pairs,
            encoder_ids,
            decoder_ids,
            corpus_ids
        ):
            corpus_opt = opt.data[corpus_id]
            weight = corpus_opt.get('weight', 1.0)
            task = TaskSpecs(
                node_rank=node_rank,
                local_rank=local_rank,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                encoder_id=encoder_id,
                decoder_id=decoder_id,
                corpus_id=corpus_id,
                weight=weight,
                corpus_opt=corpus_opt,
                src_vocab=None,
                tgt_vocab=None,
            )
            tasks.append(task)
        return cls(tasks, gpus_per_node=gpus_per_node, n_nodes=n_nodes)

    def global_to_local(self, node_rank, local_rank):
        return self.__class__(
            self.tasks,
            gpus_per_node=self.gpus_per_node,
            n_nodes=self.n_nodes,
            components_to_gpus=self.components_to_gpus,
            components_to_groups=self.components_to_groups,
            node_rank=node_rank,
            local_rank=local_rank,
        )

    def __repr__(self):
        kwargs = ',\n '.join(
            f'{key}={pformat(self.__getattribute__(key))}'
            for key in ['tasks', 'gpus_per_node', 'n_nodes', 'node_rank', 'local_rank']
        )
        return f'{self.__class__.__name__}(\n{kwargs}\n)'

    def _tasks_on_device(self, node_rank, local_rank):
        return [task for task in self.tasks if (task.node_rank, task.local_rank) == (node_rank, local_rank)]

    def get_tasks(self):
        if self.node_rank is None or self.local_rank is None:
            # global mode: return all
            return self.tasks
        else:
            return self._tasks_on_device(self.node_rank, self.local_rank)

    @staticmethod
    def _default_node_gpu(n_tasks, n_nodes, gpus_per_node):
        def yield_each_gpu():
            for node_rank in range(n_nodes):
                for local_rank in range(gpus_per_node):
                    yield (node_rank, local_rank)

        # yield GPUs in rank order, repeat as necessary
        return list(islice(cycle(yield_each_gpu()), n_tasks))

    def create_all_distributed_groups(
        self,
        new_group_func=torch.distributed.new_group,
    ):
        # Single OrderedDict contains all components.
        # Keys are tuples of strings.
        # The length of the key varies depending on the component:
        # ('encoder', encoder_id)
        # ('decoder', decoder_id)
        # ('src_emb', lang, encoder_id)
        # ('tgt_emb', lang, decoder_id)
        self.components_to_gpus = OrderedDict()

        for node_rank in range(self.n_nodes):
            for local_rank in range(self.gpus_per_node):
                global_rank = node_rank * self.gpus_per_node + local_rank
                tasks = self._tasks_on_device(node_rank, local_rank)

                for task in tasks:
                    keys = [
                        ('encoder', task.encoder_id),
                        ('decoder', task.decoder_id),
                        ('src_emb', task.src_lang, task.encoder_id),
                        ('tgt_emb', task.tgt_lang, task.decoder_id),
                    ]
                    for key in keys:
                        # Using setdefault to treat OrderedDict as defaultdict
                        self.components_to_gpus.setdefault(key, set()).add(global_rank)

        # Structured, each component in a separate OrderedDict
        self.components_to_groups = {
            component_type:  OrderedDict() for component_type in ('encoder', 'decoder', 'src_emb', 'tgt_emb')
        }
        for key, global_ranks in self.components_to_gpus.items():
            if len(global_ranks) < 2:
                # only create a process group if the component is on 2 or more gpus
                continue
            sorted_global_ranks = list(sorted(global_ranks))
            min_rank = sorted_global_ranks[0]
            group_tpl = (min_rank, new_group_func(sorted_global_ranks))
            component_type = key[0]
            component_id = key[1:]
            self.components_to_groups.setdefault(component_type, OrderedDict())[component_id] = group_tpl

        return self.components_to_groups

    @property
    def global_rank(self):
        if self.node_rank is None or self.local_rank is None:
            return None
        return self.node_rank * self.gpus_per_node + self.local_rank

    def get_distributed_groups(
        self,
        new_group_func=torch.distributed.new_group,
    ):
        """
        Returns pairs of (component_id, process_group).
        Only components present on this GPU are returned.
        The pairs are returned in a consistent order across GPUs.
        """
        if self.components_to_groups is None:
            self.create_all_distributed_groups(new_group_func)
        logger.info(f'components_to_groups: {self.components_to_groups}')

        my_distributed_groups = {
            'encoder': OrderedDict(),
            'decoder': OrderedDict(),
            'src_emb': OrderedDict(),
            'tgt_emb': OrderedDict(),
        }

        if self.global_rank is None:
            # Training on CPU, or called on global TaskQueueManager
            for component_type, components in self.components_to_groups.items():
                my_distributed_groups[component_type] = components

        global_rank = self.global_rank

        for key, global_ranks in self.components_to_gpus.items():
            if global_rank not in global_ranks:
                # omit groups that are not on this device
                continue
            component_type = key[0]
            component_id = key[1:]
            if component_id not in self.components_to_groups[component_type]:
                # omit components on a single device
                logger.info(f'{component_type} {component_id} is on a single device')
                continue
            my_distributed_groups[component_type][component_id] = \
                self.components_to_groups[component_type][component_id]

        return my_distributed_groups

    # TODO: soon deprecated by #18 Data pipeline refactoring
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
        seen = set()
        result = []
        for task in self.get_tasks():
            if side == 'src':
                lang = task.src_lang
                component_id = task.encoder_id
            else:
                lang = task.tgt_lang
                component_id = task.decoder_id
            fields = fields_dict[(side, lang)]
            if not (side, lang, component_id) in seen:
                result.append((side, lang, component_id, fields))
            seen.add((side, lang, component_id))
        return result

    def get_encoders(self):
        my_encoder_ids = [task.encoder_id for task in self.get_tasks()]
        return my_encoder_ids

    def get_decoders(self):
        my_decoder_ids = [task.decoder_id for task in self.get_tasks()]
        return my_decoder_ids

    def get_src_embs(self):
        return [(task.src_lang, task.encoder_id) for task in self.get_tasks()]

    def get_tgt_embs(self):
        return [(task.tgt_lang, task.decoder_id) for task in self.get_tasks()]

    def get_generators(self):
        return [task.tgt_lang for task in self.get_tasks()]

    def get_langs(self, side):
        if side == 'src':
            return [task.src_lang for task in self.get_tasks()]
        elif side == 'tgt':
            return [task.tgt_lang for task in self.get_tasks()]
        else:
            raise ValueError(f'side "{side}" not in {{src, tgt}}')
