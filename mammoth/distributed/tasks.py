"""sub-module defining tasks, task specifications and task management objects."""
from abc import ABC, abstractmethod
from argparse import Namespace
from collections import OrderedDict, namedtuple, Counter
from dataclasses import dataclass
from itertools import cycle, islice
from pprint import pformat
from typing import Any, Optional, List, Tuple

import numpy as np
import torch
import torch.distributed

from mammoth.distributed.contexts import DeviceContext, WorldContext
from mammoth.utils.logging import logger


class TaskDistributionStrategy(ABC):
    """
    An abstract task distribution strategy, controls which task will be scheduled next.
    """
    @abstractmethod
    def __init__(self, my_corpus_ids: List[str], **kwargs):
        pass

    @classmethod
    @abstractmethod
    def from_opts(cls, my_corpus_ids: List[str], opts: dict):
        """Alternative constructor."""
        pass

    @abstractmethod
    def sample_corpus_ids(self, n_samples: int, communication_batch_id: int) -> List[str]:
        """Select corpora to sample from."""
        pass


class WeightedSamplingTaskDistributionStrategy(TaskDistributionStrategy):
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
        if len(self.my_corpus_ids) == 0:
            raise ValueError('No corpora on device')
        if sum(my_weights) <= 0:
            raise ValueError('Can not set "weight" of all corpora on a device to zero')
        if all(x > 0 for x in my_introduce_at_training_step):
            raise ValueError('Can not set "introduce_at_training_step" of all corpora on a device to nonzero')
        if all(weight == 0 or start > 0 for (weight, start) in zip(my_weights, my_introduce_at_training_step)):
            raise ValueError('Invalid curriculum: no corpus is ready to start in the first step')

    @classmethod
    def from_opts(cls, my_corpus_ids: List[str], opts: dict):
        my_weights = [opts.tasks[corpus_id]['weight'] for corpus_id in my_corpus_ids]
        my_introduce_at_training_step = [
            opts.tasks[corpus_id]['introduce_at_training_step'] for corpus_id in my_corpus_ids
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


class RoundRobinTaskDistributionStrategy(TaskDistributionStrategy):
    """
    Schedules tasks (corpora) in a round-robin fashion.
    Yields a communication batch of n_samples at a time.
    When reaching the end of the list of tasks, starts over from the beginning.
    """

    def __init__(self, my_corpus_ids: List[str]):
        self.infinite_corpus_ids = cycle(my_corpus_ids)

    @classmethod
    def from_opts(cls, my_corpus_ids: List[str], opts: dict):
        return cls(my_corpus_ids)

    def sample_corpus_ids(
        self,
        n_samples: int,
        communication_batch_id: int,
    ):
        return list(islice(self.infinite_corpus_ids, n_samples))


TASK_DISTRIBUTION_STRATEGIES = {
    'weighted_sampling': WeightedSamplingTaskDistributionStrategy,
    'roundrobin': RoundRobinTaskDistributionStrategy,
}

DatasetMetadata = namedtuple(
    'DatasetMetadata',
    'src_lang tgt_lang encoder_id decoder_id corpus_id encoder_adapter_ids decoder_adapter_ids'
)


@dataclass
class TaskSpecs():
    node_rank: int
    local_rank: int
    src_lang: str
    tgt_lang: str
    encoder_id: List[str]
    decoder_id: List[str]
    corpus_id: str
    weight: int
    corpus_opts: dict
    src_vocab: Any  # FIXME: type
    tgt_vocab: Any
    encoder_adapter_ids: Optional[List[Tuple[int, str, str]]]
    decoder_adapter_ids: Optional[List[Tuple[int, str, str]]]

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
            encoder_adapter_ids=self.encoder_adapter_ids,
            decoder_adapter_ids=self.decoder_adapter_ids,
        )


def get_adapter_ids(opts, corpus_opts, side):
    if 'adapters' not in opts or 'adapters' not in corpus_opts:
        return []
    global_adapters_opt = opts.adapters.get(side, None)
    corpus_adapter_opt = corpus_opts['adapters'].get(side, None)
    if not global_adapters_opt or not corpus_adapter_opt:
        return []
    result = []
    for adapter_group, sub_id in corpus_adapter_opt:
        layer_stack_index = global_adapters_opt[adapter_group]['layer_stack_index']
        result.append((layer_stack_index, adapter_group, sub_id))
    return result


class TaskQueueManager:
    def __init__(
        self,
        tasks: List[TaskSpecs],
        accum_count: int,
        world_context: WorldContext,
        components_to_gpus=None,
        components_to_groups=None,
        task_distribution_strategy: Optional[TaskDistributionStrategy] = None,
        uses_adapters: bool = False,
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
        # TODO: no support for variable accumulation across training
        self.accum_count = accum_count[0] if isinstance(accum_count, list) else accum_count
        self.task_distribution_strategy = task_distribution_strategy
        self.world_context = world_context
        self.uses_adapters = uses_adapters

        self.components_to_gpus = components_to_gpus
        self.components_to_groups = components_to_groups
        self.sampled_task_counts = Counter()

    @property
    def gpus_per_node(self):
        return self.world_context.gpus_per_node

    @property
    def n_nodes(self):
        return self.world_context.n_nodes

    @classmethod
    def from_opts(cls, opts: Namespace, world_context: WorldContext):
        n_tasks = len(opts.tasks)

        # Sorting the keys, to ensure that tasks have a consistent order across devices.
        # This in turn ensures the order in which components are created from those tasks.
        corpus_ids = sorted(opts.tasks.keys())

        if world_context.is_distributed():
            if any(task.get('node_gpu', None) is not None for task in opts.tasks.values()):
                node_gpu = [
                    tuple(int(y) for y in opts.tasks[corpus_id]['node_gpu'].split(':', 1))
                    for corpus_id in corpus_ids]
            else:
                # When --node_gpu is not set, assume an assigment that fills gpus in rank order
                node_gpu = cls._default_node_gpu(n_tasks, world_context.n_nodes, world_context.gpus_per_node)
        else:
            node_gpu = [(0, 0)] * n_tasks

        enc_sharing_group = [
            opts.tasks[corpus_id].get('enc_sharing_group', None) for corpus_id in corpus_ids
        ]
        dec_sharing_group = [
            opts.tasks[corpus_id].get('dec_sharing_group', None) for corpus_id in corpus_ids
        ]
        if any(x is not None for x in enc_sharing_group):
            assert all(len(enc_ids) == len(opts.enc_layers) for enc_ids in enc_sharing_group)
        else:
            # if no encoder sharing groups are defined,
            # it is assumed that there is only one encoder stack and it is language specific
            if not len(opts.enc_layers) == 1:
                raise Exception('With more than one encoder stack, you must explictly define enc_sharing_group')
        if any(x is not None for x in dec_sharing_group):
            assert all(len(dec_ids) == len(opts.dec_layers) for dec_ids in dec_sharing_group)
        else:
            # if no decoder sharing groups are defined,
            # it is assumed that there is only one decoder stack and it is language specific
            if not len(opts.dec_layers) == 1:
                raise Exception('With more than one decoder stack, you must explictly define dec_sharing_group')

        tasks = []
        uses_adapters = False
        for (
            (node_rank, local_rank),
            corpus_id
        ) in zip(
            node_gpu,
            corpus_ids
        ):
            corpus_opts = opts.tasks[corpus_id]
            src_lang, tgt_lang = corpus_opts['src_tgt'].split('-', 1)
            encoder_id = corpus_opts.get('enc_sharing_group', [src_lang])
            decoder_id = corpus_opts.get('dec_sharing_group', [tgt_lang])
            weight = corpus_opts.get('weight', 1.0)
            if 'adapters' in corpus_opts:
                encoder_adapter_ids = get_adapter_ids(opts, corpus_opts, 'encoder')
                decoder_adapter_ids = get_adapter_ids(opts, corpus_opts, 'decoder')
                uses_adapters = True
            else:
                encoder_adapter_ids = None
                decoder_adapter_ids = None
            task = TaskSpecs(
                node_rank=node_rank,
                local_rank=local_rank,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                encoder_id=encoder_id,
                decoder_id=decoder_id,
                corpus_id=corpus_id,
                weight=weight,
                corpus_opts=corpus_opts,
                src_vocab=None,
                tgt_vocab=None,
                encoder_adapter_ids=encoder_adapter_ids,
                decoder_adapter_ids=decoder_adapter_ids,
            )
            tasks.append(task)
        return cls(
            tasks,
            world_context=world_context,
            accum_count=opts.accum_count,
            uses_adapters=uses_adapters,
        )

    def global_to_local(self, node_rank, local_rank, opts):
        assert node_rank is not None
        assert local_rank is not None
        task_distribution_strategy = self._get_strategy(node_rank=node_rank, local_rank=local_rank, opts=opts)
        device_context = self.world_context.global_to_local(node_rank, local_rank)
        return LocalTaskQueueManager(
            self.tasks,
            accum_count=self.accum_count,
            world_context=self.world_context,
            components_to_gpus=self.components_to_gpus,
            components_to_groups=self.components_to_groups,
            task_distribution_strategy=task_distribution_strategy,
            uses_adapters=self.uses_adapters,
            device_context=device_context,
        )

    def _get_strategy(self, node_rank, local_rank, opts):
        assert node_rank is not None
        assert local_rank is not None
        # Global TQM does not have a task distribution strategy, but the local ones do
        my_corpus_ids = [task.corpus_id for task in self._tasks_on_device(node_rank, local_rank)]
        try:
            strategy = TASK_DISTRIBUTION_STRATEGIES[opts.task_distribution_strategy].from_opts(
                my_corpus_ids=my_corpus_ids,
                opts=opts,
            )
            return strategy
        except Exception as e:
            raise Exception(
                f'Exception when creating task distribution strategy on {node_rank}:{local_rank} {e}'
            )

    def __repr__(self):
        kwargs = ',\n '.join(
            f'{key}={pformat(self.__getattribute__(key))}'
            for key in [
                'tasks',
                'gpus_per_node',
                'n_nodes',
                'node_rank',
                'local_rank',
                'task_distribution_strategy',
                'uses_adapters',
            ]
        )
        return f'{self.__class__.__name__}(\n{kwargs}\n)'

    def _tasks_on_device(self, node_rank, local_rank):
        return [task for task in self.tasks if (task.node_rank, task.local_rank) == (node_rank, local_rank)]

    def get_all_tasks(self):
        return self.tasks

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
        if not self.world_context.is_distributed():
            self.components_to_gpus = dict()
            self.components_to_groups = dict()
            return self.components_to_groups

        # Single OrderedDict contains all components.
        # Keys are tuples of strings.
        # The length of the key varies depending on the component:
        # ('encoder', layer_stack_index, encoder_id)
        # ('decoder', layer_stack_index, decoder_id)
        # ('src_emb', lang)
        # ('tgt_emb', lang)
        # ('encoder_adapters', layer_stack_index, encoder_id, adapter_group, sub_id)
        # ('decoder_adapters', layer_stack_index, decoder_id, adapter_group, sub_id)
        self.components_to_gpus = OrderedDict()

        for node_rank in range(self.n_nodes):
            for local_rank in range(self.gpus_per_node):
                global_rank = node_rank * self.gpus_per_node + local_rank
                tasks = self._tasks_on_device(node_rank, local_rank)

                for task in tasks:
                    keys = [
                        ('src_emb', task.src_lang),
                        ('tgt_emb', task.tgt_lang),
                    ]
                    for layer_stack_index, encoder_id in enumerate(task.encoder_id):
                        keys.append(('encoder', layer_stack_index, encoder_id))
                    for layer_stack_index, decoder_id in enumerate(task.decoder_id):
                        keys.append(('decoder', layer_stack_index, decoder_id))
                    for key in keys:
                        # Using setdefault to treat OrderedDict as defaultdict
                        self.components_to_gpus.setdefault(key, set()).add(global_rank)

                    if task.encoder_adapter_ids:
                        for layer_stack_index, adapter_group, sub_id in task.encoder_adapter_ids:
                            encoder_id = task.encoder_id[layer_stack_index]
                            key = ('encoder_adapters', layer_stack_index, encoder_id, adapter_group, sub_id)
                            self.components_to_gpus.setdefault(key, set()).add(global_rank)
                    if task.decoder_adapter_ids:
                        for layer_stack_index, adapter_group, sub_id in task.decoder_adapter_ids:
                            decoder_id = task.decoder_id[layer_stack_index]
                            key = ('decoder_adapters', layer_stack_index, decoder_id, adapter_group, sub_id)
                            self.components_to_gpus.setdefault(key, set()).add(global_rank)

        # Structured, each component in a separate OrderedDict
        self.components_to_groups = {
            component_type: OrderedDict() for component_type
            in ('encoder', 'decoder', 'src_emb', 'tgt_emb')
        }
        if self.uses_adapters:
            self.components_to_groups['encoder_adapters'] = OrderedDict()
            self.components_to_groups['decoder_adapters'] = OrderedDict()
        for key, global_ranks in self.components_to_gpus.items():
            if len(global_ranks) < 2:
                # only create a process group if the component is on 2 or more gpus
                continue
            sorted_global_ranks = list(sorted(global_ranks))
            min_rank = sorted_global_ranks[0]
            # The torch.distributed.new_group function requires that all
            # processes in the main group (i.e. all processes that are part of
            # the distributed job) enter the function, even if they are not
            # going to be members of the group. Additionally, groups should be
            # created in the same order in all processes.
            group_tpl = (min_rank, new_group_func(sorted_global_ranks))
            component_type = key[0]
            component_id = key[1:]
            self.components_to_groups.setdefault(component_type, OrderedDict())[component_id] = group_tpl

        return self.components_to_groups

    def get_langs(self, side):
        if side == 'src':
            return [task.src_lang for task in self.get_all_tasks()]
        elif side == 'tgt':
            return [task.tgt_lang for task in self.get_all_tasks()]
        else:
            raise ValueError(f'side "{side}" not in {{src, tgt}}')


class LocalTaskQueueManager(TaskQueueManager):
    def __init__(
        self,
        tasks: List[TaskSpecs],
        accum_count: int,
        world_context: WorldContext,
        components_to_gpus=None,
        components_to_groups=None,
        task_distribution_strategy: Optional[TaskDistributionStrategy] = None,
        uses_adapters: bool = False,
        device_context: Optional[DeviceContext] = None,
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
        super().__init__(
            tasks=tasks,
            accum_count=accum_count,
            world_context=world_context,
            task_distribution_strategy=task_distribution_strategy,
            uses_adapters=uses_adapters,
            components_to_gpus=components_to_gpus,
            components_to_groups=components_to_groups,
        )

        assert device_context is not None
        self.device_context = device_context

        logger.info(f'in task_queue_manager: node_rank {self.node_rank} local_rank {self.local_rank}')
        self.device_context.validate(self.world_context)

        self.sampled_task_counts = Counter()

    @property
    def node_rank(self):
        return self.device_context.node_rank

    @property
    def local_rank(self):
        return self.device_context.local_rank

    @property
    def global_rank(self):
        return self.node_rank * self.gpus_per_node + self.local_rank

    def get_my_distributed_groups(
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
            'encoder_adapters': OrderedDict(),
            'decoder_adapters': OrderedDict(),
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

    def get_my_grouped_components(self, model):
        """
        Returns nested dict of component_type -> component_id -> nn.Module.
        Only components present on this GPU are returned.
        Unlike get_my_distributed_groups, this method also returns components on a single device,
        and it does not retrieve communication groups.
        """
        if self.components_to_groups is None:
            raise Exception('Must call get_my_distributed_groups first')

        my_grouped_components = {
            'encoder': OrderedDict(),
            'decoder': OrderedDict(),
            'src_emb': OrderedDict(),
            'tgt_emb': OrderedDict(),
            'encoder_adapters': OrderedDict(),
            'decoder_adapters': OrderedDict(),
        }

        if not self.world_context.is_distributed():
            tasks = self.tasks
        else:
            tasks = self.get_my_tasks()

        for task in tasks:
            # loop over my tasks, getting all the relevant module ids and modules
            my_grouped_components['src_emb'][task.src_lang] = model.encoder.embeddings[f'embeddings_{task.src_lang}']
            my_grouped_components['tgt_emb'][task.tgt_lang] = model.decoder.embeddings[f'embeddings_{task.tgt_lang}']
            for layer_stack_index, encoder_id in enumerate(task.encoder_id):
                component = model.encoder.get_submodule(layer_stack_index, encoder_id)
                my_grouped_components['encoder'][(layer_stack_index, encoder_id)] = component
            for layer_stack_index, decoder_id in enumerate(task.decoder_id):
                component = model.decoder.get_submodule(layer_stack_index, decoder_id)
                my_grouped_components['decoder'][(layer_stack_index, decoder_id)] = component
            if task.encoder_adapter_ids:
                for layer_stack_index, adapter_group, sub_id in task.encoder_adapter_ids:
                    encoder_id = task.encoder_id[layer_stack_index]
                    key = (layer_stack_index, encoder_id, adapter_group, sub_id)
                    component = model.encoder.get_submodule(
                        layer_stack_index, encoder_id
                    ).get_adapter(adapter_group, sub_id)
                    my_grouped_components['encoder_adapters'][key] = component
            if task.decoder_adapter_ids:
                for layer_stack_index, adapter_group, sub_id in task.decoder_adapter_ids:
                    decoder_id = task.decoder_id[layer_stack_index]
                    key = (layer_stack_index, decoder_id, adapter_group, sub_id)
                    component = model.decoder.get_submodule(
                        layer_stack_index, decoder_id
                    ).get_adapter(adapter_group, sub_id)
                    my_grouped_components['decoder_adapters'][key] = component

        return my_grouped_components

    def sample_corpus_ids(self, communication_batch_id: int):
        corpus_id = self.task_distribution_strategy.sample_corpus_ids(
            1,
            communication_batch_id,
        )[0]
        corpus_ids = [corpus_id for _ in range(self.accum_count)]
        self.sampled_task_counts.update(corpus_ids)
        return corpus_ids

    def get_my_encoders(self, layer_stack_index: int):
        my_encoder_ids = [task.encoder_id[layer_stack_index] for task in self.get_my_tasks()]
        return my_encoder_ids

    def get_my_decoders(self, layer_stack_index: int):
        my_decoder_ids = [task.decoder_id[layer_stack_index] for task in self.get_my_tasks()]
        return my_decoder_ids

    def get_my_src_langs(self):
        return [task.src_lang for task in self.get_my_tasks()]

    def get_my_tgt_langs(self):
        return [task.tgt_lang for task in self.get_my_tasks()]

    def get_my_generators(self):
        return [task.tgt_lang for task in self.get_my_tasks()]

    def get_my_vocabs(self, side: str, vocabs_dict):
        """Returns a list of tuples: (side, lang, component_id, vocabs).
        side:           Either 'src' or 'tgt'.
        lang:           The language code. Vocabularies are language specific.
        component_id:   None
        vocabs_dict:    The actual vocabs.
        """
        seen = set()
        result = []
        component_id = None     # for hysterical raisins
        for task in self.get_my_tasks():
            if side == 'src':
                lang = task.src_lang
            else:
                lang = task.tgt_lang
            if not (side, lang, component_id) in seen:
                result.append((side, lang, component_id, vocabs_dict[(side, lang)]))
            seen.add((side, lang, component_id))
        return result

    def get_my_tasks(self):
        return self._tasks_on_device(self.node_rank, self.local_rank)
