from argparse import Namespace
from collections import OrderedDict

from onmt.inputters.corpus import ParallelCorpus
from onmt.utils.distributed import Scheduler


def test_init_minimal():
    opt_dict = {
        'world_size': 2,
        'gpu_ranks': [0, 1],
        'src_tgt': ['a-b', 'c-d'],
        'node_gpu': None,
        'enc_sharing_group': None,
        'dec_sharing_group': None,
    }
    opt = Namespace(**opt_dict)
    scheduler = Scheduler(opt)
    assert str(scheduler) == 'Scheduler(..., node_rank=None, local_rank=None)'
    assert scheduler.opt.node_gpu == ['0:0', '0:1']
    assert scheduler.encoder_ids == ['a', 'c']
    assert scheduler.decoder_ids == ['b', 'd']


def create_basic_scheduler(node_rank, local_rank):
    opt_dict = {
        'world_size': 4,
        'gpu_ranks': [0, 1],
        'src_tgt': ['a-b', 'c-d', 'a-d', 'e-b'],
        # unconventional assignment: two on 0:1, none on 1:1
        'node_gpu': ['0:0', '0:1', '0:1', '1:0'],
        # x is twice, on two devices 0:0 and 0:1
        'enc_sharing_group': ['x', 'xx', 'x', 'xxx'],
        # y is twice, on two devices 0:0 and 1:0
        # yy is twice, but only on a single device 0:1
        'dec_sharing_group': ['y', 'yy', 'yy', 'y'],
        'data': {
            'train_a-b': {'path_src': 'dummy', 'path_tgt': 'dummy'},
            'train_c-d': {'path_src': 'dummy', 'path_tgt': 'dummy'},
            'train_a-d': {'path_src': 'dummy', 'path_tgt': 'dummy'},
            'train_e-b': {'path_src': 'dummy', 'path_tgt': 'dummy'},
        }
    }
    opt = Namespace(**opt_dict)
    scheduler = Scheduler(opt, node_rank=node_rank, local_rank=local_rank)
    return scheduler


def test_init_basic():
    scheduler = create_basic_scheduler(node_rank=0, local_rank=1)
    assert str(scheduler) == 'Scheduler(..., node_rank=0, local_rank=1)'
    # accessing scheduler data structures directly: not filtered by rank
    assert scheduler.lang_pairs == [['a', 'b'], ['c', 'd'], ['a', 'd'], ['e', 'b']]
    assert scheduler.encoder_ids == ['x', 'xx', 'x', 'xxx']
    assert scheduler.decoder_ids == ['y', 'yy', 'yy', 'y']


def test_distributed_groups():
    class MockGroup:
        def __init__(self):
            self.group_idx = 0

        def __call__(self, sorted_global_ranks):
            result = f'Group {self.group_idx} with GPU ranks {sorted_global_ranks}'
            self.group_idx += 1
            return result

    scheduler = create_basic_scheduler(node_rank=0, local_rank=1)
    all_groups = scheduler.create_all_distributed_groups(new_group_func=MockGroup())
    assert all_groups == {
        'encoder': OrderedDict({
            'x': (0, 'Group 0 with GPU ranks [0, 1]'),
        }),
        'decoder': OrderedDict({
            'y': (0, 'Group 1 with GPU ranks [0, 2]'),
        }),
        'src_emb': OrderedDict({
            ('a', 'x'): (0, 'Group 2 with GPU ranks [0, 1]'),
        }),
        'tgt_emb': OrderedDict({
            ('b', 'y'): (0, 'Group 3 with GPU ranks [0, 2]'),
        }),
    }

    my_groups = scheduler.get_distributed_groups(new_group_func=MockGroup())
    assert my_groups == {
        'encoder': [('x', (0, 'Group 0 with GPU ranks [0, 1]'))],
        'decoder': [],
        'src_emb': [(('a', 'x'), (0, 'Group 2 with GPU ranks [0, 1]'))],
        'tgt_emb': [],
    }


def test_get_corpora():
    scheduler = create_basic_scheduler(node_rank=0, local_rank=0)
    corpora = scheduler.get_corpora(is_train=True)
    assert isinstance(corpora['train_a-b'], ParallelCorpus)
    assert len(corpora.keys()) == 1

    scheduler = create_basic_scheduler(node_rank=0, local_rank=1)
    corpora = scheduler.get_corpora(is_train=True)
    assert isinstance(corpora['train_c-d'], ParallelCorpus)
    assert isinstance(corpora['train_a-d'], ParallelCorpus)
    assert len(corpora.keys()) == 2

    scheduler = create_basic_scheduler(node_rank=1, local_rank=0)
    corpora = scheduler.get_corpora(is_train=True)
    assert isinstance(corpora['train_e-b'], ParallelCorpus)
    assert len(corpora.keys()) == 1


def test_get_fields():
    mock_fields = {
        (side, lang): f'{side} {lang}' for (side, lang) in
        [('src', 'a'), ('src', 'c'), ('src', 'e'), ('tgt', 'b'), ('tgt', 'd')]
    }
    scheduler = create_basic_scheduler(node_rank=0, local_rank=0)
    fields = scheduler.get_fields('src', mock_fields)
    assert fields == [('src', 'a', 'x', 'src a')]
    fields = scheduler.get_fields('tgt', mock_fields)
    assert fields == [('tgt', 'b', 'y', 'tgt b')]

    scheduler = create_basic_scheduler(node_rank=0, local_rank=1)
    fields = scheduler.get_fields('src', mock_fields)
    assert fields == [('src', 'c', 'xx', 'src c'), ('src', 'a', 'x', 'src a')]
    fields = scheduler.get_fields('tgt', mock_fields)
    assert fields == [('tgt', 'd', 'yy', 'tgt d')]

    scheduler = create_basic_scheduler(node_rank=1, local_rank=0)
    fields = scheduler.get_fields('src', mock_fields)
    assert fields == [('src', 'e', 'xxx', 'src e')]
    fields = scheduler.get_fields('tgt', mock_fields)
    assert fields == [('tgt', 'b', 'y', 'tgt b')]


def test_basic_getters():
    scheduler = create_basic_scheduler(node_rank=0, local_rank=0)
    encoders = list(scheduler.get_encoders())
    assert encoders == ['x']
    decoders = list(scheduler.get_decoders())
    assert decoders == ['y']
    src_embs = list(scheduler.get_src_embs())
    assert src_embs == [('a', 'x')]
    tgt_embs = list(scheduler.get_tgt_embs())
    assert tgt_embs == [('b', 'y')]
    generators = list(scheduler.get_generators())
    assert generators == ['b']

    scheduler = create_basic_scheduler(node_rank=0, local_rank=1)
    encoders = list(scheduler.get_encoders())
    assert encoders == ['xx', 'x']
    decoders = list(scheduler.get_decoders())
    assert decoders == ['yy', 'yy']
    src_embs = list(scheduler.get_src_embs())
    assert src_embs == [('c', 'xx'), ('a', 'x')]
    tgt_embs = list(scheduler.get_tgt_embs())
    assert tgt_embs == [('d', 'yy'), ('d', 'yy')]
    generators = list(scheduler.get_generators())
    assert generators == ['d', 'd']
