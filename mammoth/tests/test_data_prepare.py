#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # FIXME: Broken in MAMMOTH
#
# import copy
# import unittest
# import glob
# import os
#
# from mammoth.utils.parse import ArgumentParser
# from mammoth.opts import dynamic_prepare_opts
# from mammoth.bin.train import prepare_fields_transforms
# from mammoth.constants import CorpusName
#
#
# SAVE_DATA_PREFIX = 'data/test_data_prepare'
#
#
# def get_default_opts():
#     parser = ArgumentParser(description='data sample prepare')
#     dynamic_prepare_opts(parser)
#
#     default_opts = [
#         '-config', 'data/data.yaml',
#         '-src_vocab', 'data/vocab-train.src',
#         '-tgt_vocab', 'data/vocab-train.tgt'
#     ]
#
#     opts = parser.parse_known_args(default_opts)[0]
#     # Inject some dummy training options that may needed when build fields
#     opts.copy_attn = False
#     ArgumentParser.validate_prepare_opts(opts)
#     return opts
#
#
# default_opts = get_default_opts()
#
#
# class TestData(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         super(TestData, self).__init__(*args, **kwargs)
#         self.opts = default_opts
#
#     def dataset_build(self, opts):
#         try:
#             prepare_fields_transforms(opts)
#         except SystemExit as err:
#             print(err)
#         except IOError as err:
#             if opts.skip_empty_level != 'error':
#                 raise err
#             else:
#                 print(f"Catched IOError: {err}")
#         finally:
#             # Remove the generated *pt files.
#             for pt in glob.glob(SAVE_DATA_PREFIX + '*.pt'):
#                 os.remove(pt)
#             if self.opts.save_data:
#                 # Remove the generated data samples
#                 sample_path = os.path.join(
#                     os.path.dirname(self.opts.save_data),
#                     CorpusName.SAMPLE)
#                 if os.path.exists(sample_path):
#                     for f in glob.glob(sample_path + '/*'):
#                         os.remove(f)
#                     os.rmdir(sample_path)
#
#
# def _add_test(param_setting, methodname):
#     """
#     Adds a Test to TestData according to settings
#
#     Args:
#         param_setting: list of tuples of (param, setting)
#         methodname: name of the method that gets called
#     """
#
#     def test_method(self):
#         if param_setting:
#             opts = copy.deepcopy(self.opts)
#             for param, setting in param_setting:
#                 setattr(opts, param, setting)
#         else:
#             opts = self.opts
#         getattr(self, methodname)(opts)
#     if param_setting:
#         name = 'test_' + methodname + "_" + "_".join(
#             str(param_setting).split())
#     else:
#         name = 'test_' + methodname + '_standard'
#     setattr(TestData, name, test_method)
#     test_method.__name__ = name
#
#
# test_databuild = [[],
#                   [('src_vocab_size', 1),
#                    ('tgt_vocab_size', 1)],
#                   [('src_vocab_size', 10000),
#                    ('tgt_vocab_size', 10000)],
#                   [('src_seq_len', 1)],
#                   [('src_seq_len', 5000)],
#                   [('src_seq_length_trunc', 1)],
#                   [('src_seq_length_trunc', 5000)],
#                   [('tgt_seq_len', 1)],
#                   [('tgt_seq_len', 5000)],
#                   [('tgt_seq_length_trunc', 1)],
#                   [('tgt_seq_length_trunc', 5000)],
#                   [('copy_attn', True)],
#                   [('share_vocab', True)],
#                   [('n_sample', 30),
#                    ('save_data', SAVE_DATA_PREFIX)],
#                   [('n_sample', 30),
#                    ('save_data', SAVE_DATA_PREFIX),
#                    ('skip_empty_level', 'error')]
#                   ]
#
# for p in test_databuild:
#     _add_test(p, 'dataset_build')
