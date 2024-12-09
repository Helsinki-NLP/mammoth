"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
from collections import defaultdict, OrderedDict
from functools import partial
from torch.nn.init import xavier_uniform_
from typing import Optional, List, Dict, Tuple
from x_transformers import TransformerWrapper
from x_transformers.x_transformers import TokenEmbedding

from mammoth.distributed.components import (
    DistributedAdapter,
    DistributedComponent,
    DistributedDecoderAttentionLayersBlock,
    DistributedEncoderAttentionLayersBlock,
    Side,
)
from mammoth.modules.adapters import (
    AdaptedAttentionLayers,
    Adapter,
    FeedForwardAdapterLayer,
    LoraAdapterLayer,
)
from mammoth.inputters.vocab import Vocab
from mammoth.models import NMTModel
from mammoth.modules.attention_bridge import AttentionBridge
from mammoth.modules.layer_stack import AdaptedAttentionLayersStack, StackXcoder
from mammoth.utils.logging import logger
from mammoth.utils.misc import use_gpu

TRANSFORMER_WRAPPER_OPTS = {
    'post_emb_norm',
    'tie_embedding',
    'use_abs_pos_emb',
    'scaled_sinu_pos_emb',
    'emb_frac_gradient',
}


def _combine_ordered_dicts(input_dicts: Dict[str, OrderedDict]) -> OrderedDict:
    result = []
    for prefix, input_dict in input_dicts.items():
        for key, item in input_dict.items():
            result.append((f'{prefix}{key}', item))
    return OrderedDict(result)


def uses_adapters(opts):
    return 'adapters' in opts and opts.adapters


def get_attention_layers_kwargs(
    side: Side,
    layer_stack_index,
    xcoder_id,
    model_opts,
):
    """Return arguments for x_transformers.AttentionLayers"""
    assert side in {Side.encoder, Side.decoder}, f'Invalid side "{side}"'
    depths = model_opts.enc_layers if side == Side.encoder else model_opts.dec_layers
    depth = depths[layer_stack_index]
    causal = side == Side.decoder
    cross_attend = side == Side.decoder
    is_last = layer_stack_index == len(depths) - 1
    pre_norm_has_final_norm = is_last
    kwargs = model_opts.x_transformers_opts if model_opts.x_transformers_opts else dict()
    kwargs = {key: val for key, val in kwargs.items() if key not in TRANSFORMER_WRAPPER_OPTS}
    kwargs.update({
        'dim': model_opts.model_dim,
        'depth': depth,
        'causal': causal,
        'cross_attend': cross_attend,
        'pre_norm_has_final_norm': pre_norm_has_final_norm,
    })
    return kwargs


def get_transformer_wrapper_kwargs(
    side: Side,
    model_opts,
):
    """Return arguments for x_transformers.TransformerWrapper"""
    assert side in {Side.encoder, Side.decoder}, f'Invalid side "{side}"'
    kwargs = model_opts.x_transformers_opts if model_opts.x_transformers_opts else dict()
    kwargs = {key: val for key, val in kwargs.items() if key in TRANSFORMER_WRAPPER_OPTS}
    max_seq_len = 0 if model_opts.max_length is None else model_opts.max_length
    kwargs.update({
        'max_seq_len': max_seq_len,
    })
    if side == Side.encoder:
        kwargs['return_only_embed'] = True
    return kwargs


def build_adapters(
    side: Side,
    model_opts,
    task_queue_manager,
    single_task: Optional[str] = None,
) -> Optional[Dict[str, Adapter]]:
    """
    Create AdapterLayer objects and Adapter objects
    """
    adapters_by_name: Optional[Dict[str, Adapter]]
    if side == Side.encoder:
        side_str = 'encoder'
    else:
        side_str = 'decoder'
    my_components: List[DistributedComponent] = task_queue_manager.get_my_distributed_components()
    my_side_specific_components = [
        component for component in my_components
        if hasattr(component, 'side') and component.side == side
    ]

    if single_task:
        components_to_create = [
            component for component in my_side_specific_components
            if single_task in component.task_ids
        ]
    else:
        components_to_create = my_side_specific_components

    if uses_adapters(model_opts):
        adapter_components = [
            component for component in components_to_create
            if isinstance(component, DistributedAdapter) and component.side == side
        ]
        adapters_by_name = dict()
        adapter_params_by_group = dict()
        for adapter_group, adapter_opts in model_opts.adapters[side_str].items():
            adapter_params_by_group[adapter_group] = {
                'layer_stack_index': adapter_opts['layer_stack_index'],
                'hidden_dim': adapter_opts['hidden_dim'],
                'layers': adapter_opts['layers'],
                'sub_ids': adapter_opts['ids'],
            }
        for component in adapter_components:
            adapter_params = adapter_params_by_group[component.adapter_group]
            if adapter_opts['adapter_type'].lower() == 'lora':
                adapter_layer_func = partial(
                        LoraAdapterLayer,
                        dim=model_opts.model_dim,
                        r=adapter_params['hidden_dim'],
                    )
            elif adapter_opts['adapter_type'].lower() == 'ff':
                mult = adapter_params['hidden_dim'] / model_opts.model_dim
                # TODO: make norm locations and glu configurable
                adapter_layer_func = partial(
                    FeedForwardAdapterLayer,
                    dim=model_opts.model_dim,
                    mult=mult,
                    pre_norm=True,
                    sandwich_norm=False,
                    glu=True,
                )
            else:
                raise ValueError(f'Unrecognized adapter_type {adapter_opts["adapter_type"]}')
            layer_stack_index = adapter_params['layer_stack_index']
            adapter = Adapter(
                adapter_group=component.adapter_group,
                sub_id=component.sub_id,
                layer_stack_index=layer_stack_index,
            )
            adapters_by_name[adapter.name] = adapter
            for layer_idx in adapter_params['layers']:
                adapter_layer = adapter_layer_func()
                adapter.add_layer(layer_idx, adapter_layer)
    else:
        adapters_by_name = None
    return adapters_by_name


def build_xcoder(
    side: Side,
    model_opts,
    vocabs_dict: Dict[Tuple[str, str], Vocab],
    device,
    task_queue_manager,
    single_task: Optional[str] = None,
    token_embs: Optional[Dict[str, Vocab]] = None,
    adapters_by_name: Optional[Dict[str, Adapter]] = None,
) -> StackXcoder:
    """
    Build a StackXcoder for use as either Encoder or Decoder.
    side: a Side enum from distributed components
    model_opts: options
    vocabs_dict: A dict mapping ('src'|'tgt', lang) to a Vocab.
    device: torch.device
    task_queue_manager: TaskQueueManager
    single_task: if a task_id string is given, the built model contains only the components necessary for that task.
    token_embs: to tie encoder and decoder embeddings, pass existing embeddings here.
    """
    my_components: List[DistributedComponent] = task_queue_manager.get_my_distributed_components()
    my_side_specific_components = [
        component for component in my_components
        if hasattr(component, 'side') and component.side == side
    ]

    if single_task:
        components_to_create = [
            component for component in my_side_specific_components
            if single_task in component.task_ids
        ]
    else:
        components_to_create = my_side_specific_components

    # Create AdaptedAttentionLayers objects (an extension of an x_transformers.AttentionLayers block)
    distributed_xcoder_class: type
    if side == Side.encoder:
        distributed_xcoder_class = DistributedEncoderAttentionLayersBlock
    elif side == Side.decoder:
        distributed_xcoder_class = DistributedDecoderAttentionLayersBlock
    else:
        raise TypeError(type(side))
    attention_layers_components = [
        component for component in components_to_create
        if isinstance(component, distributed_xcoder_class)
    ]

    attention_layer_blocks: Dict[int, Dict[str, AdaptedAttentionLayers]] = defaultdict(dict)
    for component in attention_layers_components:
        layer_stack_index = component.layer_stack_index
        xcoder_id = component.xcoder_id
        attention_layers_kwargs = get_attention_layers_kwargs(
            side=side,
            layer_stack_index=layer_stack_index,
            xcoder_id=xcoder_id,
            model_opts=model_opts,
        )
        attention_layer_blocks[layer_stack_index][xcoder_id] = AdaptedAttentionLayers(
            layer_stack_index=layer_stack_index,
            xcoder_id=xcoder_id,
            **attention_layers_kwargs
        )

    # Add pre-created Adapters to the AdaptedAttentionLayers objects
    if adapters_by_name is not None:
        for adapter_name, adapter in adapters_by_name.items():
            for xcoder_id, attention_layers in attention_layer_blocks[adapter.layer_stack_index].items():
                # TODO: allow limiting which xcoder_ids get the adapter?
                logger.info(f'adding {adapter.name} to {adapter.layer_stack_index}:{xcoder_id}:{adapter.sub_id}')
                try:
                    attention_layers.add_adapter(adapter)
                except Exception as e:
                    logger.error(repr(attention_layers))
                    raise e

    # Create TokenEmbedding objects
    l2norm_embed = False
    if side == Side.encoder:
        all_langs = sorted(set(task_queue_manager.get_my_src_langs()))
    else:
        all_langs = sorted(set(task_queue_manager.get_my_tgt_langs()))
    side_alt_str = 'src' if side == Side.encoder else 'tgt'
    if token_embs is None:
        token_embs = dict()
    for lang in all_langs:
        if lang not in token_embs:
            vocab = vocabs_dict[(side_alt_str, lang)]
            token_embs[lang] = TokenEmbedding(
                dim=model_opts.model_dim,
                num_tokens=len(vocab),
                l2norm_embed=l2norm_embed
            )
    # Create AdaptedAttentionLayersStack objects and TransformerWrapper objects
    tasks = task_queue_manager.get_my_tasks()
    if single_task:
        tasks = [task for task in tasks if task.corpus_id == single_task]
    transformer_wrappers = dict()
    transformer_wrapper_kwargs = get_transformer_wrapper_kwargs(
        side=side,
        model_opts=model_opts,
    )
    for task in tasks:
        if side == Side.encoder:
            xcoder_ids = task.encoder_id
        else:
            xcoder_ids = task.decoder_id
        attention_layers_stack = [
            attention_layer_blocks[layer_stack_index][xcoder_id]
            for layer_stack_index, xcoder_id in enumerate(xcoder_ids)
        ]
        adapted_attention_layers_stack = AdaptedAttentionLayersStack(
            attention_layers_stack=attention_layers_stack
        )

        lang = task.src_lang if side == Side.encoder else task.tgt_lang
        vocab = vocabs_dict[(side_alt_str, lang)]
        # Using custom extended TransformerWrapper to allow passing in an embedding
        transformer_wrapper = TransformerWrapper(
            num_tokens=len(vocab),
            attn_layers=adapted_attention_layers_stack,
            emb_dim=model_opts.model_dim,
            token_emb=token_embs[lang],
            **transformer_wrapper_kwargs,
        )
        transformer_wrappers[task.corpus_id] = transformer_wrapper

    # Create a StackXcoder
    stack_xcoder = StackXcoder(
        transformer_wrappers=transformer_wrappers,
        attention_layer_blocks=attention_layer_blocks,
        token_embs=token_embs,
        adapters=adapters_by_name,
    )
    return stack_xcoder


def build_attention_bridge(model_opts):
    attention_bridge = AttentionBridge.from_opts(model_opts)

    if model_opts.param_init != 0.0:
        for p in attention_bridge.parameters():
            p.data.uniform_(-model_opts.param_init, model_opts.param_init)
    if model_opts.param_init_glorot:
        for p in attention_bridge.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    return attention_bridge


def build_model(
    model_opts,
    opts,
    vocabs_dict,
    task_queue_manager,
    single_task=None,
):
    """Build a model from opts.

    Args:
        model_opts: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`mammoth.utils.parse.ArgumentParser`.
        opts: overriding options.
        vocabs_dict (dict[str, mammoth.inputters.Vocab]):
            `Vocab` objects for the model.
        task_queue_manager: TaskQueueManager
        single_task: corpus_id of task, to create a single-task model

    Returns:
        the NMTModel.
    """
    logger.info('Building model...')
    gpu = use_gpu(opts)
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(device)

    enc_adapters_by_name: Optional[Dict[str, Adapter]] = build_adapters(
        side=Side.encoder,
        model_opts=model_opts,
        task_queue_manager=task_queue_manager,
        single_task=single_task,
    )
    encoder = build_xcoder(
        side=Side.encoder,
        model_opts=model_opts,
        vocabs_dict=vocabs_dict,
        device=device,
        task_queue_manager=task_queue_manager,
        single_task=single_task,
        adapters_by_name=enc_adapters_by_name,
    )
    # TODO: to tie embeddings between encoder and decoder,
    # take the token_embs from the encoder and pass them in the next build_xcoder call
    dec_adapters_by_name: Optional[Dict[str, Adapter]] = build_adapters(
        side=Side.decoder,
        model_opts=model_opts,
        task_queue_manager=task_queue_manager,
        single_task=single_task,
    )
    decoder = build_xcoder(
        side=Side.decoder,
        model_opts=model_opts,
        vocabs_dict=vocabs_dict,
        device=device,
        task_queue_manager=task_queue_manager,
        single_task=single_task,
        adapters_by_name=dec_adapters_by_name,
    )
    attention_bridge = build_attention_bridge(model_opts)
    model = NMTModel(
        encoder=encoder,
        decoder=decoder,
        attention_bridge=attention_bridge
    )

    model.to(device)
    if opts.log_model_structure:
        logger.info(model)
        for component in task_queue_manager.get_my_distributed_components():
            logger.info(component)
        for name, p in model.named_parameters():
            logger.info(f'{p.requires_grad} {name}')
    logger.info('Building model - done!')
    return model


def validate_optimizer_coverage(model, optimizer):
    """
    Given a model (complete with all locally loaded modules),
    confirm that all trained (non-frozen) parameters are covered
    by a suboptimizer.
    """
    trainable_model_params = {
        name: p for name, p in model.named_parameters()
        if p.requires_grad
    }
    optimized_params = set()
    for group in optimizer.param_groups:
        optimized_params.update(group['params'])
    missing_params = [
        name for name, p in trainable_model_params.items()
        if p not in optimized_params
    ]
    if len(missing_params) > 0:
        raise Exception(f'Missing optimizer for params: {sorted(missing_params)}')
    else:
        logger.info('All non-frozen parameters have an optimizer')
