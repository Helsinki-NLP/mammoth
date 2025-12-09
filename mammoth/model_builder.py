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
from mammoth.x_transformers import TransformerWrapper
from mammoth.x_transformers.x_transformers import TokenEmbedding, AbsolutePositionalEmbedding

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
    'post_emb_norm_bias',
    'tie_embedding',
    'final_logits_bias',  # BART-style final logits bias
    'use_abs_pos_emb',
    'scaled_sinu_pos_emb',
    'scaled_embeddings',  # Gemma3-style scaled embeddings (scaling in forward pass)
    'emb_frac_gradient',
    'max_seq_len',
    'emb_dropout',
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
    """Return arguments for x_transformers.AttentionLayers

    Supports encoder/decoder-specific options via enc_/dec_ prefixes:
    - Options with 'enc_' prefix apply only to encoder
    - Options with 'dec_' prefix apply only to decoder
    - Options without prefix apply to both
    """
    assert side in {Side.encoder, Side.decoder}, f'Invalid side "{side}"'
    depths = model_opts.enc_layers if side == Side.encoder else model_opts.dec_layers
    depth = depths[layer_stack_index]
    causal = side == Side.decoder
    cross_attend = side == Side.decoder
    is_last = layer_stack_index == len(depths) - 1
    pre_norm_has_final_norm = is_last

    # Start with base options (excluding TRANSFORMER_WRAPPER_OPTS)
    all_opts = model_opts.x_transformers_opts if model_opts.x_transformers_opts else dict()

    # Filter side-specific options
    kwargs = {}
    prefix = 'enc_' if side == Side.encoder else 'dec_'
    other_prefix = 'dec_' if side == Side.encoder else 'enc_'

    for key, val in all_opts.items():
        # Skip options for the other side
        if key.startswith(other_prefix):
            continue

        # Determine the unprefixed key name
        if key.startswith(prefix):
            unprefixed_key = key[len(prefix):]
        else:
            unprefixed_key = key

        # Skip TRANSFORMER_WRAPPER_OPTS (they go to TransformerWrapper instead)
        if unprefixed_key in TRANSFORMER_WRAPPER_OPTS:
            continue

        # Include side-specific options (strip prefix) or shared options
        if key.startswith(prefix):
            kwargs[unprefixed_key] = val  # Strip 'enc_' or 'dec_' prefix
        else:
            kwargs[key] = val

    # Handle model_dim with side-specific support
    # Priority: enc_model_dim/dec_model_dim > model_dim
    if side == Side.encoder and hasattr(model_opts, 'enc_model_dim'):
        dim = model_opts.enc_model_dim
    elif side == Side.decoder and hasattr(model_opts, 'dec_model_dim'):
        dim = model_opts.dec_model_dim
    else:
        dim = model_opts.model_dim

    # Handle sliding window attention parameters (top-level model_opts attributes)
    # These are not in x_transformers_opts but are needed for AttentionLayers
    sliding_window_attrs = [
        'sliding_window',
        'global_attn_every_n_layers',
        'global_rope_theta',
        'local_rope_theta',
    ]

    for attr in sliding_window_attrs:
        # Check for side-specific attribute first (e.g., enc_sliding_window)
        side_specific_attr = f"{prefix}{attr}"
        if hasattr(model_opts, side_specific_attr):
            kwargs[attr] = getattr(model_opts, side_specific_attr)
        # Fall back to generic attribute if it exists
        elif hasattr(model_opts, attr):
            kwargs[attr] = getattr(model_opts, attr)

    kwargs.update({
        'dim': dim,
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
    """Return arguments for x_transformers.TransformerWrapper

    Supports encoder/decoder-specific options via enc_/dec_ prefixes:
    - Options with 'enc_' prefix apply only to encoder
    - Options with 'dec_' prefix apply only to decoder
    - Options without prefix apply to both
    """
    assert side in {Side.encoder, Side.decoder}, f'Invalid side "{side}"'
    all_opts = model_opts.x_transformers_opts if model_opts.x_transformers_opts else dict()

    # Filter side-specific options
    kwargs = {}
    prefix = 'enc_' if side == Side.encoder else 'dec_'
    other_prefix = 'dec_' if side == Side.encoder else 'enc_'

    for key, val in all_opts.items():
        # Only include TRANSFORMER_WRAPPER_OPTS
        if key not in TRANSFORMER_WRAPPER_OPTS and not key.startswith(prefix):
            continue
        # Skip options for the other side
        if key.startswith(other_prefix):
            continue
        # Include side-specific options (strip prefix) or shared options
        if key.startswith(prefix):
            unprefixed_key = key[len(prefix):]
            if unprefixed_key in TRANSFORMER_WRAPPER_OPTS:
                kwargs[unprefixed_key] = val
        elif key in TRANSFORMER_WRAPPER_OPTS:
            kwargs[key] = val

    # Handle max_seq_len with side-specific support
    # Priority: enc_max_seq_len/dec_max_seq_len > max_seq_len from x_transformers_opts > model_opts.max_length
    max_seq_len_key = f'{prefix}max_seq_len'
    if max_seq_len_key in all_opts:
        max_seq_len = all_opts[max_seq_len_key]
    elif 'max_seq_len' in all_opts:
        max_seq_len = all_opts['max_seq_len']
    elif model_opts.max_length is not None:
        max_seq_len = model_opts.max_length
    else:
        max_seq_len = 0

    kwargs.update({
        'max_seq_len': max_seq_len,
    })
    if side == Side.encoder:
        kwargs['return_only_embed'] = True

    # Debug: Log if final_logits_bias is being passed
    if 'final_logits_bias' in kwargs:
        logger.info(f"get_transformer_wrapper_kwargs: final_logits_bias present for {side.name} (shape: {kwargs['final_logits_bias'].shape})")

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

    # Get side-specific model_dim
    if side == Side.encoder and hasattr(model_opts, 'enc_model_dim'):
        emb_dim = model_opts.enc_model_dim
    elif side == Side.decoder and hasattr(model_opts, 'dec_model_dim'):
        emb_dim = model_opts.dec_model_dim
    else:
        emb_dim = model_opts.model_dim

    if token_embs is None:
        token_embs = dict()
    for lang in all_langs:
        if lang not in token_embs:
            vocab = vocabs_dict[(side_alt_str, lang)]
            token_embs[lang] = TokenEmbedding(
                dim=emb_dim,
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
            emb_dim=emb_dim,  # Use side-specific dimension
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

    # Determine dtype for model initialization
    dtype = torch.float32
    if hasattr(model_opts, 'model_dtype'):
        if model_opts.model_dtype == 'fp16':
            dtype = torch.float16
            logger.info('Initializing model in fp16 precision')
        elif model_opts.model_dtype == 'bf16':
            dtype = torch.bfloat16
            logger.info('Initializing model in bf16 precision')
        else:
            logger.info('Initializing model in fp32 precision')

    # Set default dtype for model initialization
    torch.set_default_dtype(dtype)

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

    # Optionally share embeddings between encoder and decoder
    dec_token_embs = None
    share_embeddings = getattr(model_opts, 'share_encoder_decoder_embeddings', False)

    if share_embeddings:
        # Get dimensions
        enc_dim = model_opts.enc_model_dim if hasattr(model_opts, 'enc_model_dim') else model_opts.model_dim
        dec_dim = model_opts.dec_model_dim if hasattr(model_opts, 'dec_model_dim') else model_opts.model_dim

        if enc_dim != dec_dim:
            logger.warning(
                f'Cannot share encoder-decoder embeddings: encoder dim ({enc_dim}) != decoder dim ({dec_dim}). '
                'Embeddings will not be shared.'
            )
        else:
            # Check if encoder and decoder share vocabularies for any language pairs
            encoder_langs = set(task_queue_manager.get_my_src_langs())
            decoder_langs = set(task_queue_manager.get_my_tgt_langs())
            shared_langs = encoder_langs & decoder_langs

            if shared_langs:
                # Use encoder embeddings for decoder where languages overlap
                # First, validate that vocabularies match for shared languages
                dec_token_embs = {}
                successfully_shared = []
                for lang in shared_langs:
                    src_vocab = vocabs_dict[('src', lang)]
                    tgt_vocab = vocabs_dict[('tgt', lang)]

                    # Check if vocabularies match
                    if len(src_vocab) != len(tgt_vocab):
                        logger.warning(
                            f'Cannot share embeddings for language "{lang}": '
                            f'vocab sizes differ (src: {len(src_vocab)}, tgt: {len(tgt_vocab)})'
                        )
                        continue

                    # For now, we assume matching vocab sizes means matching vocabularies
                    # A more thorough check would compare vocab tokens, but that's expensive
                    dec_token_embs[lang] = encoder.token_embs[lang]
                    successfully_shared.append(lang)

                if successfully_shared:
                    logger.info(
                        f'Sharing encoder-decoder embeddings for languages: {sorted(successfully_shared)}'
                    )
                else:
                    logger.warning(
                        'Could not share embeddings for any language due to vocabulary mismatches'
                    )
                    dec_token_embs = None
            else:
                logger.warning(
                    'Cannot share encoder-decoder embeddings: no overlapping languages. '
                    f'Encoder languages: {sorted(encoder_langs)}, Decoder languages: {sorted(decoder_langs)}'
                )

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
        token_embs=dec_token_embs,
        adapters_by_name=dec_adapters_by_name,
    )

    # Check if encoder and decoder have different dimensions
    # If so, skip attention bridge (it requires matching dimensions)
    enc_dim = model_opts.enc_model_dim if hasattr(model_opts, 'enc_model_dim') else model_opts.model_dim
    dec_dim = model_opts.dec_model_dim if hasattr(model_opts, 'dec_model_dim') else model_opts.model_dim

    if enc_dim != dec_dim:
        logger.info(f'Encoder dim ({enc_dim}) != Decoder dim ({dec_dim}): Skipping attention bridge')
        attention_bridge = None
    else:
        attention_bridge = build_attention_bridge(model_opts)

    model = NMTModel(
        encoder=encoder,
        decoder=decoder,
        attention_bridge=attention_bridge
    )


    model.to(device)

    if opts.log_model_structure and task_queue_manager.global_rank == 0:
        logger.info(model)
        for component in task_queue_manager.get_my_distributed_components():
            logger.info(component)
        for name, p in model.named_parameters():
            logger.info(f'{p.requires_grad} {name}')
    logger.info('Building model - done!')
    return model


def freeze_model_components(model, opts, task_queue_manager):
    """
    Freeze model components based on configuration options.
    Must be called after model initialization but before optimizer creation.

    This function implements granular freezing for parameter-efficient fine-tuning,
    allowing independent control over encoder, decoder, cross-attention, embeddings,
    and attention bridge freezing.

    Args:
        model: The NMTModel instance
        opts: Configuration options containing freeze flags
        task_queue_manager: TaskQueueManager for distributed component access

    Notes:
        - Adapters are never frozen, even when base model is frozen
        - Frozen parameters are excluded from optimizer automatically
        - Freezing happens before distributed initialization to ensure consistency
    """
    from mammoth.utils.logging import logger

    frozen_params_count = 0
    total_params_count = sum(p.numel() for p in model.parameters())

    # Freeze encoder
    if getattr(opts, 'freeze_encoder', False):
        logger.info("Freezing encoder attention and feedforward layers")
        frozen_count = _freeze_encoder_layers(model)
        frozen_params_count += frozen_count
        logger.info(f"Frozen {frozen_count:,} encoder parameters")

    # Freeze decoder (three mutually exclusive options)
    if getattr(opts, 'freeze_decoder', False):
        logger.info("Freezing decoder attention and feedforward layers (including cross-attention)")
        frozen_count = _freeze_decoder_layers(model, freeze_cross_attn=True)
        frozen_params_count += frozen_count
        logger.info(f"Frozen {frozen_count:,} decoder parameters")

    # Freeze decoder except cross-attention (if decoder not already frozen)
    elif getattr(opts, 'freeze_decoder_except_cross_attention', False):
        logger.info("Freezing decoder self-attention and feedforward, keeping cross-attention trainable")
        frozen_count = _freeze_decoder_layers(model, freeze_cross_attn=False)
        frozen_params_count += frozen_count
        logger.info(f"Frozen {frozen_count:,} decoder parameters (cross-attention remains trainable)")

    # Freeze cross-attention only (if decoder not already frozen)
    elif getattr(opts, 'freeze_cross_attention', False):
        logger.info("Freezing decoder cross-attention layers only")
        frozen_count = _freeze_cross_attention_only(model)
        frozen_params_count += frozen_count
        logger.info(f"Frozen {frozen_count:,} cross-attention parameters")

    # Freeze encoder embeddings
    if getattr(opts, 'freeze_encoder_embeddings', False):
        logger.info("Freezing encoder embeddings")
        frozen_count = _freeze_embeddings(model.encoder)
        frozen_params_count += frozen_count
        logger.info(f"Frozen {frozen_count:,} encoder embedding parameters")

    # Freeze decoder embeddings
    if getattr(opts, 'freeze_decoder_embeddings', False):
        logger.info("Freezing decoder embeddings")
        frozen_count = _freeze_embeddings(model.decoder)
        frozen_params_count += frozen_count
        logger.info(f"Frozen {frozen_count:,} decoder embedding parameters")

    # Freeze attention bridge
    if getattr(opts, 'freeze_attention_bridge', False):
        if model.attention_bridge is not None:
            logger.info("Freezing attention bridge")
            frozen_count = 0
            for param in model.attention_bridge.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
            frozen_params_count += frozen_count
            logger.info(f"Frozen {frozen_count:,} attention bridge parameters")
        else:
            logger.warning(
                "Cannot freeze attention bridge: model has no attention bridge. "
                "Attention bridge is only present when ab_layers is configured and "
                "encoder/decoder dimensions match."
            )

    # Report freezing summary
    if frozen_params_count > 0:
        frozen_percentage = (frozen_params_count / total_params_count) * 100
        trainable_params = total_params_count - frozen_params_count
        logger.info(
            f"Freezing summary: {frozen_params_count:,} / {total_params_count:,} parameters frozen "
            f"({frozen_percentage:.2f}%). {trainable_params:,} parameters remain trainable."
        )
    else:
        logger.info("No parameters frozen - all parameters are trainable")

    return frozen_params_count


def _freeze_encoder_layers(model):
    """
    Freeze all encoder attention and feedforward layers, excluding adapters.

    Args:
        model: NMTModel instance

    Returns:
        Number of frozen parameters
    """
    frozen_count = 0

    # Iterate through all encoder layer stacks and language-specific components
    for layer_stack_index in model.encoder.attention_layers_by_xcoder_id.keys():
        for xcoder_id, attention_layers in model.encoder.attention_layers_by_xcoder_id[layer_stack_index].items():
            # Freeze base layers (not adapters)
            for name, param in attention_layers.named_parameters():
                # Skip adapter parameters - they should remain trainable
                if not name.startswith('adapters.'):
                    param.requires_grad = False
                    frozen_count += param.numel()

    return frozen_count


def _freeze_decoder_layers(model, freeze_cross_attn=True):
    """
    Freeze decoder layers, with optional control over cross-attention freezing.

    Args:
        model: NMTModel instance
        freeze_cross_attn: Whether to freeze cross-attention layers (default: True)

    Returns:
        Number of frozen parameters
    """
    frozen_count = 0

    # Iterate through all decoder layer stacks and language-specific components
    for layer_stack_index in model.decoder.attention_layers_by_xcoder_id.keys():
        for xcoder_id, attention_layers in model.decoder.attention_layers_by_xcoder_id[layer_stack_index].items():
            # Get layer types and structures
            layer_types = attention_layers.layer_types
            layers = attention_layers.layers

            # Freeze based on layer type
            for layer_type, layer_struct in zip(layer_types, layers):
                should_freeze = False

                if layer_type == 'a':  # Self-attention - always freeze
                    should_freeze = True
                elif layer_type == 'c':  # Cross-attention - conditional
                    should_freeze = freeze_cross_attn
                elif layer_type == 'f':  # Feedforward - always freeze
                    should_freeze = True

                if should_freeze:
                    # layer_struct is a ModuleList: typically [norms, block, residual]
                    for module in layer_struct:
                        for name, param in module.named_parameters():
                            # Skip adapter parameters
                            if not name.startswith('adapters.'):
                                param.requires_grad = False
                                frozen_count += param.numel()

    return frozen_count


def _freeze_cross_attention_only(model):
    """
    Freeze only cross-attention layers in decoder, leave all other layers trainable.

    Args:
        model: NMTModel instance

    Returns:
        Number of frozen parameters
    """
    frozen_count = 0

    # Iterate through all decoder components
    for layer_stack_index in model.decoder.attention_layers_by_xcoder_id.keys():
        for xcoder_id, attention_layers in model.decoder.attention_layers_by_xcoder_id[layer_stack_index].items():
            layer_types = attention_layers.layer_types
            layers = attention_layers.layers

            # Freeze only where layer_type == 'c' (cross-attention)
            for layer_type, layer_struct in zip(layer_types, layers):
                if layer_type == 'c':
                    for module in layer_struct:
                        for param in module.parameters():
                            param.requires_grad = False
                            frozen_count += param.numel()

    return frozen_count


def _freeze_embeddings(xcoder):
    """
    Freeze all token embeddings in an encoder or decoder.

    Args:
        xcoder: Encoder or Decoder instance (StackXcoder)

    Returns:
        Number of frozen parameters
    """
    frozen_count = 0

    # Freeze all language-specific token embeddings
    for lang, token_emb in xcoder.token_embs.items():
        for param in token_emb.parameters():
            param.requires_grad = False
            frozen_count += param.numel()

    return frozen_count


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
