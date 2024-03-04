import copy
import unittest

import torch

import mammoth
import mammoth.opts
from mammoth.model_builder import build_embeddings, build_encoder, build_decoder
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS
from mammoth.utils.parse import ArgumentParser

parser = ArgumentParser(description='train.py')
mammoth.opts.model_opts(parser)
mammoth.opts._add_train_general_opts(parser)

# -data option is required, but not used in this test, so dummy.
opts = parser.parse_known_args(['-tasks', 'dummy', '-node_rank', '0', '-model_dim', '500'], strict=False)[0]


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.opts = opts

    def get_field(self):
        return Vocab(None, items=[], tag='dummy', specials=list(DEFAULT_SPECIALS))

    def get_batch(self, source_l=3, bsize=1):
        # len x batch x nfeat
        test_src = torch.ones(source_l, bsize, 1).long()
        test_tgt = torch.ones(source_l, bsize, 1).long()
        test_length = torch.ones(bsize).fill_(source_l).long()
        return test_src, test_tgt, test_length

    def embeddings_forward(self, opts, source_l=3, bsize=1):
        '''
        Tests if the embeddings works as expected

        args:
            opts: set of options
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        word_field = self.get_field()
        emb = build_embeddings(opts, word_field)
        test_src, _, __ = self.get_batch(source_l=source_l, bsize=bsize)
        if opts.decoder_type == 'transformer':
            input = torch.cat([test_src, test_src], 0)
            res = emb(input)
            compare_to = torch.zeros(source_l * 2, bsize, opts.model_dim)
        else:
            res = emb(test_src)
            compare_to = torch.zeros(source_l, bsize, opts.model_dim)

        self.assertEqual(res.size(), compare_to.size())

    def encoder_forward(self, opts, source_l=3, bsize=1):
        '''
        Tests if the encoder works as expected

        args:
            opts: set of options
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        word_field = self.get_field()
        embeddings = build_embeddings(opts, word_field)
        enc = build_encoder(opts, embeddings)

        test_src, test_tgt, test_length = self.get_batch(source_l=source_l, bsize=bsize)

        hidden_t, outputs, test_length = enc(test_src, test_length)

        # Initialize vectors to compare size with
        test_hid = torch.zeros(self.opts.enc_layers, bsize, opts.model_dim)
        test_out = torch.zeros(source_l, bsize, opts.model_dim)

        # Ensure correct sizes and types
        self.assertEqual(test_hid.size(), hidden_t[0].size(), hidden_t[1].size())
        self.assertEqual(test_out.size(), outputs.size())
        self.assertEqual(type(outputs), torch.Tensor)

    def nmtmodel_forward(self, opts, source_l=3, bsize=1):
        """
        Creates a nmtmodel with a custom opts function.
        Forwards a testbatch and checks output size.

        Args:
            opts: Namespace with options
            source_l: length of input sequence
            bsize: batchsize
        """
        word_field = self.get_field()

        embeddings = build_embeddings(opts, word_field)
        enc = build_encoder(opts, embeddings)

        embeddings = build_embeddings(opts, word_field, for_encoder=False)
        dec = build_decoder(opts, embeddings)

        model = mammoth.models.model.NMTModel(enc, dec)

        test_src, test_tgt, test_length = self.get_batch(source_l=source_l, bsize=bsize)
        outputs, attn = model(test_src, test_tgt, test_length)
        outputsize = torch.zeros(source_l - 1, bsize, opts.model_dim)
        # Make sure that output has the correct size and type
        self.assertEqual(outputs.size(), outputsize.size())
        self.assertEqual(type(outputs), torch.Tensor)


def _add_test(param_setting, methodname):
    """
    Adds a Test to TestModel according to settings

    Args:
        param_setting: list of tuples of (param, setting)
        methodname: name of the method that gets called
    """

    def test_method(self):
        opts = copy.deepcopy(self.opts)
        if param_setting:
            for param, setting in param_setting:
                setattr(opts, param, setting)
        ArgumentParser.update_model_opts(opts)
        getattr(self, methodname)(opts)

    if param_setting:
        name = 'test_' + methodname + "_" + "_".join(str(param_setting).split())
    else:
        name = 'test_' + methodname + '_standard'
    setattr(TestModel, name, test_method)
    test_method.__name__ = name


'''
TEST PARAMETERS
'''
opts.brnn = False

# FIXME: Most tests disabled: MAMMOTH only supports Transformer
test_embeddings = [
    # [],
    [('decoder_type', 'transformer')]
]

for p in test_embeddings:
    _add_test(p, 'embeddings_forward')

# FIXME: All tests disabled: MAMMOTH only supports Transformer, and the test for Transformer is broken
tests_encoder = [
    # [],
    # [('encoder_type', 'mean')],
    # [('encoder_type', 'transformer'), ('word_vec_size', 16), ('model_dim', 16)],
    # [],
]

for p in tests_encoder:
    _add_test(p, 'encoder_forward')

# FIXME: Most tests disabled: MAMMOTH only supports Transformer
tests_nmtmodel = [
    # [('rnn_type', 'GRU')],
    # [('layers', 10)],
    # [('input_feed', 0)],
    [
        ('decoder_type', 'transformer'),
        ('encoder_type', 'transformer'),
        ('src_word_vec_size', 16),
        ('tgt_word_vec_size', 16),
        ('model_dim', 16),
    ],
    [
        ('decoder_type', 'transformer'),
        ('encoder_type', 'transformer'),
        ('src_word_vec_size', 16),
        ('tgt_word_vec_size', 16),
        ('model_dim', 16),
        ('position_encoding', True),
    ],
    # [('coverage_attn', True)],
    # [('copy_attn', True)],
    # [('global_attention', 'mlp')],
    # [('context_gate', 'both')],
    # [('context_gate', 'target')],
    # [('context_gate', 'source')],
    # [('encoder_type', "brnn"), ('brnn_merge', 'sum')],
    # [('encoder_type', "brnn")],
    # [('decoder_type', 'cnn'), ('encoder_type', 'cnn')],
    # [('encoder_type', 'rnn'), ('global_attention', None)],
    # [('encoder_type', 'rnn'), ('global_attention', None), ('copy_attn', True), ('copy_attn_type', 'general')],
    # [('encoder_type', 'rnn'), ('global_attention', 'mlp'), ('copy_attn', True), ('copy_attn_type', 'general')],
    # [],
]


# ## FIXME: Broken in MAMMOTH
# for p in tests_nmtmodel:
#     _add_test(p, 'nmtmodel_forward')
