# -*- coding: utf-8 -*-
import logging
import torch

logger = logging.getLogger()


def quishape(name, val):
    """Debug utils: QUI logging for tensor shapes"""
    if val is None:
        print(f'{name} is None')
        return
    if not isinstance(val, torch.Tensor):
        print(f'{name} is of type {type(val)}')
        return
    if len(val.shape) >= 2:
        print(f'{name} {val.shape}  {val.shape[0] * val.shape[1]}')
    else:
        print(f'{name} {val.shape}')


def layer_intermediates_shapes(cache):
    """Debug utils: QUI logging for all shapes in a LayerIntermediates object"""
    quishape('last_hidden', cache.last_hidden)
    quishape('attn_z_loss', cache.attn_z_loss)
    quishape('mems', cache.mems)
    quishape('memory_tokens', cache.memory_tokens)

    if cache.hiddens is not None:
        for x in cache.hiddens:
            quishape('hiddens', x)
    if cache.layer_hiddens is not None:
        for x in cache.layer_hiddens:
            quishape('layer_hiddens', x)

    if cache.attn_intermediates is not None:
        for attn_i in cache.attn_intermediates:
            quishape('qk_similarities', attn_i.qk_similarities)
            quishape('pre_softmax_attn', attn_i.pre_softmax_attn)
            quishape('post_softmax_attn', attn_i.post_softmax_attn)
            quishape('cached_kv[0]', attn_i.cached_kv[0])
            quishape('cached_kv[1]', attn_i.cached_kv[1])
            print('layer_type', attn_i.layer_type)
