import pytest

import torch

import mammoth
import mammoth.opts
from mammoth.model_builder import build_model, build_xcoder, add_zeroshot_task
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS
from mammoth.utils.parse import ArgumentParser
from mammoth.distributed.components import Side
from mammoth.distributed.tasks import TaskSpecs, TaskQueueManager, RoundRobinTaskDistributionStrategy
from mammoth.distributed.contexts import WorldContext, DeviceContextEnum

parser = ArgumentParser(description='train.py')
mammoth.opts.model_opts(parser)
mammoth.opts._add_train_general_opts(parser)

DEFAULT_ARGS = '-tasks dummy -node_rank 0 -model_dim 500 -seed 1 --max_length 200'

VOCABS = {
    ('src', 'a'): Vocab(None, items=['a'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('src', 'b'): Vocab(None, items=['b', 'bb'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('tgt', 'a'): Vocab(None, items=['_a'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('tgt', 'b'): Vocab(None, items=['_b', '_bb', '_bbb'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    # Zero-shot scenario: spa-eng, por-eng, eng-cat. cat exists only as a target.
    ('src', 'spa'): Vocab(None, items=['hola', 'mundo'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('src', 'por'): Vocab(None, items=['ola', 'mundo'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('src', 'eng'): Vocab(None, items=['hello', 'world'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('tgt', 'eng'): Vocab(None, items=['hello', 'world'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('tgt', 'cat'): Vocab(None, items=['hola', 'mon'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
}

TASK_SPECS = {
    'dummy_a-b': TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang='a',
        tgt_lang='b',
        encoder_id=['foo'],
        decoder_id=['bar'],
        corpus_id='a-b',
        weight=1,
        introduce_at_training_step=0,
        corpus_opts=dict(),
        src_vocab=VOCABS[('src', 'a')],
        tgt_vocab=VOCABS[('tgt', 'b')],
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    ),
    # Zero-shot scenario: two tasks share 'shared' encoder + 'eng' decoder,
    # third task trains an 'eng' encoder + 'cat' decoder. The cat embedding
    # is trained only as a target embedding.
    'spa-eng': TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang='spa',
        tgt_lang='eng',
        encoder_id=['shared'],
        decoder_id=['eng'],
        corpus_id='spa-eng',
        weight=1,
        introduce_at_training_step=0,
        corpus_opts=dict(),
        src_vocab=VOCABS[('src', 'spa')],
        tgt_vocab=VOCABS[('tgt', 'eng')],
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    ),
    'por-eng': TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang='por',
        tgt_lang='eng',
        encoder_id=['shared'],
        decoder_id=['eng'],
        corpus_id='por-eng',
        weight=1,
        introduce_at_training_step=0,
        corpus_opts=dict(),
        src_vocab=VOCABS[('src', 'por')],
        tgt_vocab=VOCABS[('tgt', 'eng')],
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    ),
    'eng-cat': TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang='eng',
        tgt_lang='cat',
        encoder_id=['eng'],
        decoder_id=['cat'],
        corpus_id='eng-cat',
        weight=1,
        introduce_at_training_step=0,
        corpus_opts=dict(),
        src_vocab=VOCABS[('src', 'eng')],
        tgt_vocab=VOCABS[('tgt', 'cat')],
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    ),
}





class TestModel():
    def __init__(self, args, tasks):
        self.opts = self.parse_args(args)
        world_context = WorldContext(DeviceContextEnum.MULTI_GPU, n_nodes=1, gpus_per_node=2)
        self.tasks = [TASK_SPECS[task] for task in tasks]
        self.tqm = TaskQueueManager(
            tasks=self.tasks,
            accum_count=1,
            world_context=world_context,
            task_distribution_strategy_cls=RoundRobinTaskDistributionStrategy,
            uses_adapters=False,
        ).global_to_local(
            node_rank=0,
            local_rank=0,
            opts=self.opts,
        )
        self.vocabs_dict = {
            (side, lang): vocab for (side, lang, _, vocab) in self.tqm.get_my_vocabs('src', VOCABS)
        }
        self.vocabs_dict.update({
            (side, lang): vocab for (side, lang, _, vocab) in self.tqm.get_my_vocabs('tgt', VOCABS)
        })

        self.tqm.create_all_distributed_components(
            use_attention_bridge=False,
        )

    def parse_args(self, args):
        opts = parser.parse_known_args(
            ' '.join([DEFAULT_ARGS, args]).split(),
            strict=False
        )[0]
        return opts

    def get_batch(self, source_l=3, bsize=1, task=None):
        # x-transformers takes shape (batch, time)
        test_src = torch.ones(bsize, source_l).long()
        test_tgt = torch.ones(bsize, source_l).long()
        test_mask = torch.ones(bsize, source_l).bool()
        metadata = task.get_serializable_metadata()
        return test_src, test_tgt, test_mask, metadata

# Broken in x-transformers
#    def embeddings_forward(self, opts, source_l=3, bsize=1):
#        '''
#        Tests if the embeddings works as expected
#
#        args:
#            opts: set of options
#            source_l: Length of generated input sentence
#            bsize: Batchsize of generated input
#        '''
#        word_field = self.get_field()
#        emb = build_embeddings(opts, word_field)
#        test_src, _, __ = self.get_batch(source_l=source_l, bsize=bsize)
#        if opts.decoder_type == 'transformer':
#            input = torch.cat([test_src, test_src], 0)
#            res = emb(input)
#            compare_to = torch.zeros(source_l * 2, bsize, opts.model_dim)
#        else:
#            res = emb(test_src)
#            compare_to = torch.zeros(source_l, bsize, opts.model_dim)
#
#        self.assertEqual(res.size(), compare_to.size())

    def encoder_forward(self, source_l=3, bsize=1):
        '''
        Tests if the encoder works as expected

        args:
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        token_embs = None
        device = 'cpu'

        task = self.tasks[0]
        test_src, test_tgt, test_mask, metadata = self.get_batch(source_l=source_l, bsize=bsize, task=task)

        enc = build_xcoder(
            Side.encoder,
            self.opts,
            self.vocabs_dict,
            device,
            task_queue_manager=self.tqm,
            single_task=None,
            token_embs=token_embs,
        )
        active_encoder = enc.activate(task_id=task.corpus_id, adapter_ids=task.encoder_adapter_ids)

        encoder_output = active_encoder(test_src, mask=test_mask, return_embeddings=True)

        # Make sure that output has the correct size and type
        # x-transformers returns (batch, time, dim/vocab_index)
        outputsize = torch.zeros(bsize, source_l, self.opts.model_dim)
        assert encoder_output.size() == outputsize.size()
        assert isinstance(encoder_output, torch.Tensor)

    def nmtmodel_forward(self, source_l=3, bsize=1):
        """
        Creates a nmtmodel with a custom opts function.
        Forwards a testbatch and checks output size.

        Args:
            source_l: length of input sequence
            bsize: batchsize
        """
        model = build_model(
            self.opts,
            self.opts,
            self.vocabs_dict,
            task_queue_manager=self.tqm,
            single_task=None,
        )

        task = self.tasks[0]
        tgt_vocab = self.vocabs_dict[('tgt', task.tgt_lang)]
        test_src, test_tgt, test_mask, metadata = self.get_batch(source_l=source_l, bsize=bsize, task=task)
        # currently caller must adjust for the autoregressive step
        # shape: (batch, time)
        decoder_input = test_tgt[:, :-1]
        logits, decoder_output = model(test_src, decoder_input, test_mask, metadata=metadata)
        # Make sure that output has the correct size and type
        # x-transformers returns (batch, time, dim/vocab_index)
        logitsize = torch.zeros(bsize, source_l - 1, len(tgt_vocab))
        outputsize = torch.zeros(bsize, source_l - 1, self.opts.model_dim)
        assert logits.size() == logitsize.size()
        assert isinstance(logits, torch.Tensor)
        assert decoder_output.size() == outputsize.size()
        assert isinstance(decoder_output, torch.Tensor)


@pytest.mark.parametrize(
    ('args', 'tasks', 'source_l', 'bsize'),
    [
        (
            '--enc_layers 1 --dec_layers 1',
            ['dummy_a-b'],
            3,
            1,
        ),
        (
            '--enc_layers 1 --dec_layers 1',
            ['dummy_a-b'],
            5,
            7,
        ),
        (
            '--enc_layers 3 --dec_layers 2',
            ['dummy_a-b'],
            4,
            1,
        ),
    ],
)
def test_nmtmodel(args, tasks, source_l, bsize):
    tm = TestModel(args, tasks)
    tm.nmtmodel_forward(source_l=source_l, bsize=bsize)
    tm.encoder_forward(source_l=source_l, bsize=bsize)


def test_add_zeroshot_task():
    """
    Zero-shot scenario: spa-eng, por-eng, eng-cat were trained.
    At inference, register a zero-shot 'cat-eng' task that reuses
    the trained 'cat' target embedding as a source embedding and
    routes through the shared encoder + eng decoder.
    """
    tm = TestModel(
        '--enc_layers 2 --dec_layers 2',
        ['spa-eng', 'por-eng', 'eng-cat'],
    )
    model = build_model(
        tm.opts,
        tm.opts,
        tm.vocabs_dict,
        task_queue_manager=tm.tqm,
        single_task=None,
    )

    # Precondition: cat embedding exists only on the decoder side; not on encoder.
    assert 'cat' in model.decoder.token_embs
    assert 'cat' not in model.encoder.token_embs
    trained_cat_emb = model.decoder.token_embs['cat']

    # Register a zero-shot cat->eng task reusing the shared encoder (from spa-eng)
    # and the eng decoder (from spa-eng).
    add_zeroshot_task(
        model,
        zero_shot_task_id='zs_cat-eng',
        src_lang='cat',
        template_task_id='spa-eng',
    )

    # The encoder now exposes the new task_id and the cat embedding is the
    # SAME object (parameter reuse, not a copy).
    assert 'zs_cat-eng' in model.encoder
    assert 'cat' in model.encoder.token_embs
    assert model.encoder.token_embs['cat'] is trained_cat_emb
    assert model.encoder['zs_cat-eng'].token_emb is trained_cat_emb

    # The shared encoder attention layers should be reused (same underlying
    # AdaptedAttentionLayers, though the AdaptedAttentionLayersStack wrapper
    # is a new instance, matching how build_xcoder constructs one wrapper
    # per task while sharing the layers themselves).
    zs_layers = model.encoder['zs_cat-eng'].attn_layers.attention_layers_stack
    tpl_layers = model.encoder['spa-eng'].attn_layers.attention_layers_stack
    assert len(zs_layers) == len(tpl_layers)
    for zs_l, tpl_l in zip(zs_layers, tpl_layers):
        assert zs_l is tpl_l

    # The decoder side should alias the template task's decoder wrapper.
    assert 'zs_cat-eng' in model.decoder
    assert model.decoder['zs_cat-eng'] is model.decoder['spa-eng']

    # Forward pass: tokens must be in range of the cat (source) vocab now.
    bsize, source_l = 2, 4
    cat_vocab_size = len(tm.vocabs_dict[('tgt', 'cat')])
    eng_vocab_size = len(tm.vocabs_dict[('tgt', 'eng')])
    test_src = torch.randint(0, cat_vocab_size, (bsize, source_l)).long()
    test_tgt = torch.randint(0, eng_vocab_size, (bsize, source_l)).long()
    test_mask = torch.ones(bsize, source_l).bool()

    # Build metadata for the new zero-shot task. We build it from the template
    # task's metadata since the shapes of encoder_adapter_ids, etc. match.
    template_task = next(t for t in tm.tasks if t.corpus_id == 'spa-eng')
    metadata = template_task.get_serializable_metadata()
    metadata = metadata._replace(corpus_id='zs_cat-eng')

    decoder_input = test_tgt[:, :-1]
    logits, decoder_output = model(test_src, decoder_input, test_mask, metadata=metadata)

    assert logits.shape == (bsize, source_l - 1, eng_vocab_size)
    assert decoder_output.shape == (bsize, source_l - 1, tm.opts.model_dim)
