import pytest
from argparse import Namespace
from collections import OrderedDict
from unittest.mock import MagicMock

from mammoth.distributed import TaskQueueManager, WorldContext


def test_init_minimal():
    opt_dict = {
        'accum_count': 1,
        'task_distribution_strategy': 'roundrobin',
        'world_size': 2,
        'n_nodes': 1,
        'gpu_ranks': [0, 1],
        'enc_layers': [1],
        'dec_layers': [1],
        'tasks': {
            'train_a-b': {'path_src': 'dummy', 'path_tgt': 'dummy', 'src_tgt': 'a-b'},
            'train_c-d': {'path_src': 'dummy', 'path_tgt': 'dummy', 'src_tgt': 'c-d'},
        }
    }
    opts = Namespace(**opt_dict)
    world_context = WorldContext.from_opts(opts)
    task_queue_manager = TaskQueueManager.from_opts(opts, world_context)
    assert world_context.is_gpu()
    assert world_context.is_distributed()
    assert len(task_queue_manager.tasks) == 2
    assert task_queue_manager.gpus_per_node == 2
    assert task_queue_manager.n_nodes == 1
    with pytest.raises(Exception):
        # global TQM does not allow accessing node_rank or local_rank
        task_queue_manager.node_rank
    with pytest.raises(Exception):
        task_queue_manager.local_rank
    assert [task.node_rank for task in task_queue_manager.tasks] == [0, 0]
    assert [task.local_rank for task in task_queue_manager.tasks] == [0, 1]


def create_basic_task_queue_manager():
    opt_dict = {
        'accum_count': 8,
        'task_distribution_strategy': 'weighted_sampling',
        'world_size': 4,
        'n_nodes': 2,
        'gpu_ranks': [0, 1],
        'enc_layers': [1],
        'dec_layers': [1],
        # node_gpu unconventional assignment: two on 0:1, none on 1:1
        # enc_sharing_group x is twice, on two devices 0:0 and 0:1
        # dec_sharing_group y is twice, on two devices 0:0 and 1:0
        # dec_sharing_group yy is twice, but only on a single device 0:1
        'tasks': {
            'train_0_a-b': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'weight': 2,
                'introduce_at_training_step': 0,
                'src_tgt': 'a-b',
                'node_gpu': '0:0',
                'enc_sharing_group': ['x'],
                'dec_sharing_group': ['y'],
            },
            'train_2_a-d': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'weight': 1,
                'introduce_at_training_step': 10,
                'src_tgt': 'a-d',
                'node_gpu': '0:1',
                'enc_sharing_group': ['x'],
                'dec_sharing_group': ['yy'],
            },
            'train_3_e-b': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'weight': 1,
                'introduce_at_training_step': 0,
                'src_tgt': 'e-b',
                'node_gpu': '1:0',
                'enc_sharing_group': ['xxx'],
                'dec_sharing_group': ['y'],
            },
            'train_1_c-d': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'weight': 1,
                'introduce_at_training_step': 0,
                'src_tgt': 'c-d',
                'node_gpu': '0:1',
                'enc_sharing_group': ['xx'],
                'dec_sharing_group': ['yy'],
            },
        }
    }
    opts = Namespace(**opt_dict)
    world_context = WorldContext.from_opts(opts)
    task_queue_manager = TaskQueueManager.from_opts(opts, world_context)
    return task_queue_manager, opts


def test_init_basic():
    global_task_queue_manager, opts = create_basic_task_queue_manager()
    task_queue_manager = global_task_queue_manager.global_to_local(node_rank=0, local_rank=1, opts=opts)
    world_context = task_queue_manager.world_context
    assert world_context.is_gpu()
    assert world_context.is_distributed()
    assert len(task_queue_manager.tasks) == 4
    assert task_queue_manager.gpus_per_node == 2
    assert task_queue_manager.n_nodes == 2
    assert task_queue_manager.node_rank == 0
    assert task_queue_manager.local_rank == 1
    # accessing task_queue_manager data structures directly: not filtered by rank
    assert [task.encoder_id for task in task_queue_manager.tasks] == [['x'], ['xx'], ['x'], ['xxx']]
    assert [task.decoder_id for task in task_queue_manager.tasks] == [['y'], ['yy'], ['yy'], ['y']]
    assert [task.src_lang for task in task_queue_manager.tasks] == ['a', 'c', 'a', 'e']
    assert [task.tgt_lang for task in task_queue_manager.tasks] == ['b', 'd', 'd', 'b']


def test_create_all_distributed_groups():
    class MockGroup:
        def __init__(self):
            self.group_idx = 0

        def __call__(self, sorted_global_ranks):
            result = f'Group {self.group_idx} with GPU ranks {sorted_global_ranks}'
            self.group_idx += 1
            return result

    global_task_queue_manager, opts = create_basic_task_queue_manager()
    all_groups = global_task_queue_manager.create_all_distributed_groups(new_group_func=MockGroup())
    assert all_groups == {
        'src_emb': OrderedDict({
            ('a',): (0, 'Group 0 with GPU ranks [0, 1]'),
        }),
        'tgt_emb': OrderedDict({
            ('b',): (0, 'Group 1 with GPU ranks [0, 2]'),
        }),
        'encoder': OrderedDict({
            (0, 'x'): (0, 'Group 2 with GPU ranks [0, 1]'),
        }),
        'decoder': OrderedDict({
            (0, 'y'): (0, 'Group 3 with GPU ranks [0, 2]'),
        }),
    }


def test_get_my_distributed_groups():
    class MockGroup:
        def __init__(self):
            self.group_idx = 0

        def __call__(self, sorted_global_ranks):
            result = f'Group {self.group_idx} with GPU ranks {sorted_global_ranks}'
            self.group_idx += 1
            return result

    global_task_queue_manager, opts = create_basic_task_queue_manager()
    task_queue_manager = global_task_queue_manager.global_to_local(node_rank=0, local_rank=1, opts=opts)
    my_groups = task_queue_manager.get_my_distributed_groups(new_group_func=MockGroup())
    assert my_groups == {
        'encoder': OrderedDict({
            (0, 'x'): (0, 'Group 2 with GPU ranks [0, 1]'),
        }),
        'decoder': OrderedDict(),
        'src_emb': OrderedDict({
            ('a',): (0, 'Group 0 with GPU ranks [0, 1]'),
        }),
        'tgt_emb': OrderedDict(),
        'encoder_adapters': OrderedDict(),
        'decoder_adapters': OrderedDict(),
    }


def test_cpu_distributed_groups():
    opt_dict = {
        'accum_count': 4,
        'task_distribution_strategy': 'roundrobin',
        'world_size': 0,
        'gpu_ranks': [],
        'n_nodes': 1,
        'enc_layers': [1],
        'dec_layers': [1],
        'tasks': {
            'train_a-b': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'src_tgt': 'a-b',
            },
            'train_c-d': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'src_tgt': 'c-d',
            },
        }
    }
    opts = Namespace(**opt_dict)
    world_context = WorldContext.from_opts(opts)
    global_task_queue_manager = TaskQueueManager.from_opts(opts, world_context)
    task_queue_manager = global_task_queue_manager.global_to_local(node_rank=0, local_rank=0, opts=opts)
    new_group_func = MagicMock().new_group_func
    my_groups = task_queue_manager.get_my_distributed_groups(new_group_func=new_group_func)
    # No groups should be created when running on CPU
    new_group_func.assert_not_called()
    # The component keys should still exist, but be empty
    for component in ['encoder', 'decoder', 'src_emb', 'tgt_emb']:
        assert len(my_groups[component]) == 0
    assert not world_context.is_gpu()
    assert not world_context.is_distributed()


def test_distributed_groups_no_encoder_group():
    opt_dict = {
        'accum_count': 1,
        'task_distribution_strategy': 'roundrobin',
        'world_size': 4,
        'n_nodes': 2,
        'enc_layers': [1],
        'dec_layers': [1],
        'gpu_ranks': [0, 1],
        # every language pair on its own gpu: no overlap
        'tasks': {
            'train_a-b': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'src_tgt': 'a-b',
                'node_gpu': '0:0',
            },
            'train_c-d': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'src_tgt': 'c-d',
                'node_gpu': '0:1',
            },
            'train_b-a': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'src_tgt': 'b-a',
                'node_gpu': '1:0',
            },
            'train_d-c': {
                'path_src': 'dummy',
                'path_tgt': 'dummy',
                'src_tgt': 'd-c',
                'node_gpu': '1:1',
            },
        }
    }
    opts = Namespace(**opt_dict)
    world_context = WorldContext.from_opts(opts)
    global_task_queue_manager = TaskQueueManager.from_opts(opts, world_context)
    task_queue_manager = global_task_queue_manager.global_to_local(node_rank=0, local_rank=0, opts=opts)
    new_group_func = MagicMock().new_group_func
    my_groups = task_queue_manager.get_my_distributed_groups(new_group_func=new_group_func)
    # No groups should be created:
    # AB is fully shared (doesn't need a group),
    # and all other components are not shared at all
    new_group_func.assert_not_called()
    # The component keys should still exist, but be empty
    for component in ['encoder', 'decoder', 'src_emb', 'tgt_emb']:
        assert len(my_groups[component]) == 0


# FIXME
# def test_get_fields():
#     mock_fields = {
#         (side, lang): f'{side} {lang}' for (side, lang) in
#         [('src', 'a'), ('src', 'c'), ('src', 'e'), ('tgt', 'b'), ('tgt', 'd')]
#     }
#     global_task_queue_manager, opts = create_basic_task_queue_manager()
#     task_queue_manager = global_task_queue_manager.global_to_local(node_rank=0, local_rank=0, opts=opts)
#     fields = task_queue_manager.get_fields('src', mock_fields)
#     assert fields == [('src', 'a', None, 'src a')]
#     fields = task_queue_manager.get_fields('tgt', mock_fields)
#     assert fields == [('tgt', 'b', None, 'tgt b')]
#
#     task_queue_manager = global_task_queue_manager.global_to_local(node_rank=0, local_rank=1, opts=opts)
#     fields = task_queue_manager.get_fields('src', mock_fields)
#     assert fields == [('src', 'c', None, 'src c'), ('src', 'a', 'x', 'src a')]
#     fields = task_queue_manager.get_fields('tgt', mock_fields)
#     assert fields == [('tgt', 'd', None, 'tgt d')]
#
#     task_queue_manager = global_task_queue_manager.global_to_local(node_rank=1, local_rank=0, opts=opts)
#     fields = task_queue_manager.get_fields('src', mock_fields)
#     assert fields == [('src', 'e', None, 'src e')]
#     fields = task_queue_manager.get_fields('tgt', mock_fields)
#     assert fields == [('tgt', 'b', None, 'tgt b')]


def test_basic_getters():
    global_task_queue_manager, opts = create_basic_task_queue_manager()
    task_queue_manager = global_task_queue_manager.global_to_local(node_rank=0, local_rank=0, opts=opts)
    encoders = list(task_queue_manager.get_my_encoders(0))
    assert encoders == ['x']
    decoders = list(task_queue_manager.get_my_decoders(0))
    assert decoders == ['y']
    src_langs = list(task_queue_manager.get_my_src_langs())
    assert src_langs == ['a']
    tgt_langs = list(task_queue_manager.get_my_tgt_langs())
    assert tgt_langs == ['b']
    generators = list(task_queue_manager.get_my_generators())
    assert generators == ['b']

    task_queue_manager = global_task_queue_manager.global_to_local(node_rank=0, local_rank=1, opts=opts)
    encoders = list(task_queue_manager.get_my_encoders(0))
    assert encoders == ['xx', 'x']
    decoders = list(task_queue_manager.get_my_decoders(0))
    assert decoders == ['yy', 'yy']
    src_langs = list(task_queue_manager.get_my_src_langs())
    assert src_langs == ['c', 'a']
    tgt_langs = list(task_queue_manager.get_my_tgt_langs())
    assert tgt_langs == ['d', 'd']
    generators = list(task_queue_manager.get_my_generators())
    assert generators == ['d', 'd']
