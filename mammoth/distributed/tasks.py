"""sub-module defining tasks, task specifications and task management objects."""
from abc import ABC, abstractmethod
from argparse import Namespace
from collections import namedtuple, defaultdict, Counter
from dataclasses import dataclass
from itertools import cycle, islice
from pprint import pformat
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.distributed

from mammoth.distributed.contexts import DeviceContext, WorldContext
from mammoth.distributed.components import (
    Side,
    DistributedComponentBuilder,
    DistributedComponent,
    DistributedEncoder,
    DistributedDecoder,
    DistributedEmbedding,
    DistributedGenerator,
    DistributedAdapter,
    DistributedAttentionBridge,
    # DistributedComponentAction,
    # DistributedComponentActionWithGradient,
)
from mammoth.utils.logging import logger


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
    introduce_at_training_step: int
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


@dataclass
class BatchTaskSample:
    """
    A deterministicly random sample of one task per device, to be trained in a single batch.
    """
    # maps from global rank to Task
    tasks: Dict[int, TaskSpecs]
    training_step: int


class TaskDistributionStrategy(ABC):
    """
    An abstract task distribution strategy, controls which tasks will be scheduled next.
    """
    def __init__(self):
        self.training_step = 0

    @abstractmethod
    def sample_corpus_ids(self, active_tasks: Dict[int, List[TaskSpecs]]) -> BatchTaskSample:
        """
        Select one task per device, to train on.
        active_tasks[global_rank] -> (task_id, weight)
        """
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
        seed: int,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)

    def sample_corpus_ids(self, active_tasks: Dict[int, List[TaskSpecs]]) -> BatchTaskSample:
        result: Dict[int, str] = dict()
        for global_rank in sorted(active_tasks.keys()):
            tasks = active_tasks[global_rank]
            weights = [task.weight for task in tasks]
            sum_w = sum(weights)
            assert sum_w > 0
            p = [weight / sum_w for weight in weights]
            # sampling with replacement from weighted corpora (language pairs)
            sampled_corpus_id = self.rng.choice(tasks, size=1, p=p)[0]
            result[global_rank] = sampled_corpus_id
        bts = BatchTaskSample(tasks=result, training_step=self.training_step)
        self.training_step += 1
        return bts


class RoundRobinTaskDistributionStrategy(TaskDistributionStrategy):
    """
    Schedules tasks (corpora) in a round-robin fashion.
    Yields a communication batch of n_samples at a time.
    When reaching the end of the list of tasks, starts over from the beginning.
    """

    def __init__(self, seed: int):
        super().__init__()

    def sample_corpus_ids(self, active_tasks: Dict[int, List[TaskSpecs]]) -> BatchTaskSample:
        result: Dict[int, str] = dict()
        for global_rank in sorted(active_tasks.keys()):
            tasks = active_tasks[global_rank]
            sampled_corpus_id = tasks[self.training_step % len(tasks)]
            result[global_rank] = sampled_corpus_id
        bts = BatchTaskSample(tasks=result, training_step=self.training_step)
        self.training_step += 1
        return bts


TASK_DISTRIBUTION_STRATEGIES = {
    'weighted_sampling': WeightedSamplingTaskDistributionStrategy,
    'roundrobin': RoundRobinTaskDistributionStrategy,
}


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
        distributed_components=None,
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

        self.distributed_components = distributed_components

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

        task_distribution_strategy = TASK_DISTRIBUTION_STRATEGIES[opts.task_distribution_strategy](
            seed=opts.seed,
        )
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
            introduce_at_training_step = corpus_opts.get('introduce_at_training_step', 0)
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
                introduce_at_training_step=introduce_at_training_step,
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
            task_distribution_strategy=task_distribution_strategy,
            uses_adapters=uses_adapters,
        )

    def global_to_local(self, node_rank, local_rank, opts):
        assert node_rank is not None
        assert local_rank is not None
        device_context = self.world_context.global_to_local(node_rank, local_rank)
        return LocalTaskQueueManager(
            self.tasks,
            accum_count=self.accum_count,
            world_context=self.world_context,
            distributed_components=self.distributed_components,
            task_distribution_strategy=self.task_distribution_strategy,
            uses_adapters=self.uses_adapters,
            device_context=device_context,
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

    def get_active_tasks(self) -> Dict[int, List[TaskSpecs]]:
        result = defaultdict(list)
        for task in self.tasks:
            # TODO: DRY violation, this computation is implemented in many places
            global_rank = task.node_rank * self.gpus_per_node + task.local_rank
            if task.introduce_at_training_step <= self.task_distribution_strategy.training_step:
                result[global_rank].append(task)
        return result

    @staticmethod
    def _default_node_gpu(n_tasks, n_nodes, gpus_per_node):
        def yield_each_gpu():
            for node_rank in range(n_nodes):
                for local_rank in range(gpus_per_node):
                    yield (node_rank, local_rank)

        # yield GPUs in rank order, repeat as necessary
        return list(islice(cycle(yield_each_gpu()), n_tasks))

    def create_all_distributed_components(
        self,
        use_attention_bridge: bool,
        new_group_func=torch.distributed.new_group,
    ) -> List[DistributedComponent]:
        """
        Creates DistributedComponent objects.
        For all components that are on more than one device, creats a communication group.
        """
        builder = DistributedComponentBuilder()
        for task in self.tasks:
            # TODO: DRY violation, this computation is implemented in many places
            global_rank = task.node_rank * self.gpus_per_node + task.local_rank
            builder.add(
                DistributedEmbedding(
                    global_ranks={global_rank},
                    group=None,
                    side=Side.encoder,
                    lang=task.src_lang,
                )
            )
            builder.add(
                DistributedEmbedding(
                    global_ranks={global_rank},
                    group=None,
                    side=Side.decoder,
                    lang=task.tgt_lang,
                )
            )
            builder.add(
                DistributedGenerator(
                    global_ranks={global_rank},
                    group=None,
                    lang=task.tgt_lang,
                )
            )
            for layer_stack_index, encoder_id in enumerate(task.encoder_id):
                builder.add(
                    DistributedEncoder(
                        global_ranks={global_rank},
                        group=None,
                        layer_stack_index=layer_stack_index,
                        xcoder_id=encoder_id,
                    )
                )
            for layer_stack_index, decoder_id in enumerate(task.decoder_id):
                builder.add(
                    DistributedDecoder(
                        global_ranks={global_rank},
                        group=None,
                        layer_stack_index=layer_stack_index,
                        xcoder_id=decoder_id,
                    )
                )
            if task.encoder_adapter_ids:
                for layer_stack_index, adapter_group, sub_id in task.encoder_adapter_ids:
                    builder.add(
                        DistributedAdapter(
                            global_ranks={global_rank},
                            group=None,
                            side=Side.encoder,
                            layer_stack_index=layer_stack_index,
                            adapter_group=adapter_group,
                            sub_id=sub_id,
                        )
                    )
            if task.decoder_adapter_ids:
                for layer_stack_index, adapter_group, sub_id in task.decoder_adapter_ids:
                    builder.add(
                        DistributedAdapter(
                            global_ranks={global_rank},
                            group=None,
                            side=Side.decoder,
                            layer_stack_index=layer_stack_index,
                            adapter_group=adapter_group,
                            sub_id=sub_id,
                        )
                    )
            if use_attention_bridge:
                builder.add(DistributedAttentionBridge(global_ranks={global_rank}, group=None))

        # once all DistributedComponents are created, we can initialize communication groups
        if self.world_context.is_distributed():
            for component in builder:
                # do not create communication groups for components on a single device
                if len(component.global_ranks) > 1:
                    # The torch.distributed.new_group function requires that all
                    # processes in the main group (i.e. all processes that are part of
                    # the distributed job) enter the function, even if they are not
                    # going to be members of the group. Additionally, groups should be
                    # created in the same order in all processes.
                    component.group = new_group_func(sorted(component.global_ranks))
                else:
                    logger.info(f'{component.get_name()} is on a single device')

        sorted_components = list(builder)
        self.distributed_components = sorted_components
        return sorted_components

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
        distributed_components=None,
        task_distribution_strategy: TaskDistributionStrategy = None,
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
            distributed_components=distributed_components,
        )

        assert device_context is not None
        self.device_context = device_context

        logger.info(f'in task_queue_manager: node_rank {self.node_rank} local_rank {self.local_rank}')
        self.device_context.validate(self.world_context)
        self._sanity_check_tasks()

        self.sampled_task_counts = Counter()
        self.my_distributed_components = None

    def _sanity_check_tasks(self):
        my_corpus_ids = [task.corpus_id for task in self.get_my_tasks()]
        my_weights = [task.weight for task in self.get_my_tasks()]
        my_introduce_at_training_step = [
            task.introduce_at_training_step for task in self.get_my_tasks()
        ]
        # Sanity check of weights and curriculum
        assert len(my_corpus_ids) == len(my_weights)
        assert len(my_corpus_ids) == len(my_introduce_at_training_step)
        if len(my_corpus_ids) == 0:
            raise ValueError('No corpora on device')
        if sum(my_weights) <= 0:
            raise ValueError('Can not set "weight" of all corpora on a device to zero')
        if all(x > 0 for x in my_introduce_at_training_step):
            raise ValueError('Can not set "introduce_at_training_step" of all corpora on a device to nonzero')
        if all(weight == 0 or start > 0 for (weight, start) in zip(my_weights, my_introduce_at_training_step)):
            raise ValueError('Invalid curriculum: no corpus is ready to start in the first step')

    @property
    def node_rank(self):
        return self.device_context.node_rank

    @property
    def local_rank(self):
        return self.device_context.local_rank

    @property
    def global_rank(self):
        return self.node_rank * self.gpus_per_node + self.local_rank

    def get_my_distributed_components(self) -> List[DistributedComponent]:
        if self.distributed_components is None:
            raise Exception('Call create_all_distributed_components first')
        if not self.my_distributed_components:
            my_global_rank = self.global_rank
            self.my_distributed_components = [
                component
                for component in self.distributed_components
                if my_global_rank in component.global_ranks
            ]
        return self.my_distributed_components

    def sample_corpus_ids(self) -> BatchTaskSample:
        active_tasks: Dict[int, List[TaskSpecs]] = self.get_active_tasks()
        batch_task_sample = self.task_distribution_strategy.sample_corpus_ids(active_tasks)
        if self.global_rank is None or self.global_rank == 0:
            # Only track sampled_task_counts on the master device.
            # Every TQM (both data loader and trainer for every device) has access to the global info
            self.sampled_task_counts.update(
                [task.corpus_id for task in batch_task_sample.tasks.values()]
            )
        return batch_task_sample

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
