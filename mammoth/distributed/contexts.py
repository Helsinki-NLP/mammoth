from dataclasses import dataclass
from enum import Enum


class DeviceContextEnum(Enum):
    CPU = 1
    SINGLE_GPU = 2
    MULTI_GPU = 3


@dataclass
class WorldContext:
    context: DeviceContextEnum
    # Size of the world: total number of nodes, gpus on each node
    n_nodes: int
    gpus_per_node: int

    @property
    def world_size(self):
        """Total number of training GPUs"""
        return self.n_nodes * self.gpus_per_node

    def is_distributed(self):
        """When training is distributed over several devices,
        multiprocessing is used to communicate gradients"""
        return self.context == DeviceContextEnum.MULTI_GPU

    def is_gpu(self):
        """Data tensors must be moved to the GPU for compute"""
        return self.context != DeviceContextEnum.CPU

    def global_to_local(self, node_rank, local_rank):
        assert node_rank is not None
        assert local_rank is not None
        return DeviceContext(
            context=self.context,
            n_nodes=self.n_nodes,
            gpus_per_node=self.gpus_per_node,
            node_rank=node_rank,
            local_rank=local_rank,
        )

    @classmethod
    def from_opts(cls, opts):
        gpus_per_node = len(opts.gpu_ranks)
        world_size = int(opts.world_size) if gpus_per_node > 0 else 0
        multinode = gpus_per_node != world_size
        if world_size <= 0:
            # setting a non-positive world size means use CPU
            device_context_enum = DeviceContextEnum.CPU
            if opts.n_nodes != 1:
                raise ValueError('CPU training is only possible on a single node')
        elif world_size == 1:
            # world size 1 uses GPU, but is not distributed
            device_context_enum = DeviceContextEnum.SINGLE_GPU
            if opts.n_nodes != 1:
                raise ValueError(
                    f'Invalid single-gpu node configuration: '
                    f'n_nodes {opts.n_nodes} gpus_per_node {gpus_per_node} world_size {world_size}'
                )
        else:
            # world size > 1
            if multinode and opts.n_nodes == 1:
                raise ValueError(
                    f'Invalid multi-node configuration: '
                    f'n_nodes {opts.n_nodes} gpus_per_node {gpus_per_node} world_size {world_size}'
                )
            device_context_enum = DeviceContextEnum.MULTI_GPU
        world_context = WorldContext(context=device_context_enum, n_nodes=opts.n_nodes, gpus_per_node=gpus_per_node)
        return world_context


@dataclass
class DeviceContext(WorldContext):
    # Our place in the world
    node_rank: int
    local_rank: int

    @property
    def global_rank(self) -> int:
        return self.gpus_per_node * self.node_rank + self.local_rank

    @property
    def id(self) -> str:
        if self.is_gpu():
            return f'GPU {self.node_rank}:{self.local_rank}'
        else:
            return 'CPU'

    def is_master(self):
        """For code that should only run in one process:
        - saving fully shared modules from one device only
        - avoiding log spam when all devices would log the same result
        """
        return not self.is_distributed() or self.global_rank == 0

    def validate(self, world_context):
        # check that this DeviceContext is consistent with given WorldContext
        assert self.context == world_context.context
        assert self.n_nodes == world_context.n_nodes
        assert self.gpus_per_node == world_context.gpus_per_node
        # check that ranks are within the specified size of the world
        assert 0 <= self.node_rank < self.n_nodes
        if self.is_gpu():
            assert 0 <= self.local_rank < self.gpus_per_node
