"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
from collections import defaultdict
from torch.nn.init import xavier_uniform_
from typing import Optional, List, Dict, Tuple

from mammoth.distributed.components import (
    DistributedDecoderAttentionLayersBlock,
    DistributedEncoderAttentionLayersBlock,
    Side,
)
from mammoth.inputters.vocab import Vocab
from mammoth.models import NMTModel
from mammoth.modules.attention_bridge import AttentionBridge
from mammoth.modules.layer_stack import StackXcoder
from mammoth.modules.transformer import (
    NativeTransformerWrapper,
    TransformerStack,
    EncoderBlock,
    DecoderBlock,
    RotaryEmbedding,
)
from mammoth.utils.logging import logger
from mammoth.utils.misc import use_gpu




def build_xcoder(
    side: Side,
    model_opts,
    vocabs_dict: Dict[Tuple[str, str], Vocab],
    device,
    task_queue_manager,
    single_task: Optional[str] = None,
    token_embs: Optional[Dict[str, nn.Embedding]] = None,
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
    side_str = 'src' if side == Side.encoder else 'tgt'
    return_only_embed = (side == Side.encoder)

    if side == Side.encoder and getattr(model_opts, 'enc_model_dim', None) is not None:
        dim = model_opts.enc_model_dim
    elif side == Side.decoder and getattr(model_opts, 'dec_model_dim', None) is not None:
        dim = model_opts.dec_model_dim
    else:
        dim = model_opts.model_dim

    heads = getattr(model_opts, 'heads', 8)
    dim_head = dim // heads
    activation = getattr(model_opts, 'ff_activation', 'swiglu')
    default_ff_mult = 4.0 if activation == 'gelu' else getattr(model_opts, 'ff_mult', 2.67)
    ff_mult = getattr(model_opts, 'ff_mult', default_ff_mult)
    attn_drop = getattr(model_opts, 'attn_dropout', 0.0)
    ff_drop = getattr(model_opts, 'ff_dropout', 0.0)
    use_rotary = getattr(model_opts, 'rotary_pos_emb', False)
    use_post_emb_norm = getattr(model_opts, 'post_emb_norm', False)
    depths = model_opts.enc_layers if side == Side.encoder else model_opts.dec_layers
    block_cls = EncoderBlock if side == Side.encoder else DecoderBlock

    # 1. Build one TransformerStack per (layer_stack_index, xcoder_id)
    distributed_xcoder_class = (
        DistributedEncoderAttentionLayersBlock if side == Side.encoder
        else DistributedDecoderAttentionLayersBlock
    )
    my_components = task_queue_manager.get_my_distributed_components()
    xcoder_components = [
        c for c in my_components
        if isinstance(c, distributed_xcoder_class)
        and (not single_task or single_task in c.task_ids)
    ]

    attention_layer_blocks: Dict[int, Dict[str, TransformerStack]] = defaultdict(dict)
    for component in xcoder_components:
        layer_stack_index = component.layer_stack_index
        xcoder_id = component.xcoder_id
        depth = depths[layer_stack_index]
        is_last_stack = (layer_stack_index == len(depths) - 1)
        blocks = nn.ModuleList([
            block_cls(dim=dim, heads=heads, ff_mult=ff_mult,
                      attn_dropout=attn_drop, ff_dropout=ff_drop, activation=activation)
            for _ in range(depth)
        ])
        final_norm = nn.RMSNorm(dim) if is_last_stack else None
        attention_layer_blocks[layer_stack_index][xcoder_id] = TransformerStack(
            blocks=blocks,
            final_norm=final_norm,
            dim=dim,
            layer_stack_index=layer_stack_index,
            xcoder_id=xcoder_id,
        )

    # 2. Build token embeddings
    if token_embs is None:
        token_embs = {}
    all_langs = sorted(set(
        task_queue_manager.get_my_src_langs() if side == Side.encoder
        else task_queue_manager.get_my_tgt_langs()
    ))
    for lang in all_langs:
        if lang not in token_embs:
            vocab = vocabs_dict[(side_str, lang)]
            token_embs[lang] = nn.Embedding(len(vocab), dim)

    # 3. Build per-component shared modules (keyed by xcoder_id tuple)
    tasks = task_queue_manager.get_my_tasks()
    if single_task:
        tasks = [t for t in tasks if t.corpus_id == single_task]

    component_vocab_sizes: Dict[tuple, int] = {}
    for task in tasks:
        xcoder_ids = task.encoder_id if side == Side.encoder else task.decoder_id
        component_key = tuple(xcoder_ids)
        if component_key not in component_vocab_sizes:
            lang = task.src_lang if side == Side.encoder else task.tgt_lang
            component_vocab_sizes[component_key] = len(vocabs_dict[(side_str, lang)])

    per_component_post_emb_norms: Dict[tuple, nn.Module] = {}
    per_component_rotary_embs: Dict[tuple, Optional[RotaryEmbedding]] = {}
    per_component_to_logits: Dict[tuple, nn.Module] = {}

    for component_key, vocab_size in component_vocab_sizes.items():
        per_component_post_emb_norms[component_key] = (
            nn.RMSNorm(dim) if use_post_emb_norm else nn.Identity()
        )
        per_component_rotary_embs[component_key] = (
            RotaryEmbedding(dim_head) if use_rotary else None
        )
        if not return_only_embed:
            per_component_to_logits[component_key] = nn.Linear(dim, vocab_size, bias=False)

    # 4. Build one NativeTransformerWrapper per task
    transformer_wrappers: Dict[str, NativeTransformerWrapper] = {}
    for task in tasks:
        xcoder_ids = task.encoder_id if side == Side.encoder else task.decoder_id
        component_key = tuple(xcoder_ids)
        lang = task.src_lang if side == Side.encoder else task.tgt_lang
        stacks = [
            attention_layer_blocks[layer_stack_index][xcoder_id]
            for layer_stack_index, xcoder_id in enumerate(xcoder_ids)
        ]
        transformer_wrappers[task.corpus_id] = NativeTransformerWrapper(
            token_emb=token_embs[lang],
            post_emb_norm=per_component_post_emb_norms[component_key],
            stacks=stacks,
            rotary_emb=per_component_rotary_embs[component_key],
            to_logits=per_component_to_logits.get(component_key),
            return_only_embed=return_only_embed,
        )

    return StackXcoder(
        transformer_wrappers=transformer_wrappers,
        attention_layer_blocks=dict(attention_layer_blocks),
        token_embs=token_embs,
        per_component_post_emb_norms=per_component_post_emb_norms,
        per_component_to_logits=per_component_to_logits,
    )


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
        elif model_opts.model_dtype == 'bf16':
            dtype = torch.bfloat16
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f'Initializing model in {dtype} precision')

    # Set default dtype for model initialization
    torch.set_default_dtype(dtype)

    encoder = build_xcoder(
        side=Side.encoder,
        model_opts=model_opts,
        vocabs_dict=vocabs_dict,
        device=device,
        task_queue_manager=task_queue_manager,
        single_task=single_task,
    )

    # Optionally share embeddings between encoder and decoder
    dec_token_embs = None
    share_embeddings = getattr(model_opts, 'share_encoder_decoder_embeddings', False)

    if share_embeddings:
        # Get dimensions
        enc_dim = model_opts.enc_model_dim if getattr(model_opts, 'enc_model_dim', None) is not None else model_opts.model_dim
        dec_dim = model_opts.dec_model_dim if getattr(model_opts, 'dec_model_dim', None) is not None else model_opts.model_dim

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
                # No overlapping language labels, but check if vocabularies are identical
                # This handles bilingual models like BART with shared vocab but different language labels
                # (e.g., es→en translation with same tokenizer for both sides)

                # Check if this is a simple bilingual case (one encoder lang, one decoder lang)
                # or if all encoder-decoder vocab pairs are identical
                all_vocabs_match = True
                dec_token_embs = {}
                cross_lingual_pairs = []

                for enc_lang in encoder_langs:
                    for dec_lang in decoder_langs:
                        src_vocab = vocabs_dict.get(('src', enc_lang))
                        tgt_vocab = vocabs_dict.get(('tgt', dec_lang))

                        if src_vocab is None or tgt_vocab is None:
                            all_vocabs_match = False
                            break

                        if len(src_vocab) != len(tgt_vocab):
                            all_vocabs_match = False
                            break

                        # Vocabularies match - map decoder lang to encoder embeddings
                        dec_token_embs[dec_lang] = encoder.token_embs[enc_lang]
                        cross_lingual_pairs.append(f'{enc_lang}→{dec_lang}')

                    if not all_vocabs_match:
                        break

                if all_vocabs_match and cross_lingual_pairs:
                    logger.info(
                        f'Sharing encoder-decoder embeddings across language pairs: {", ".join(cross_lingual_pairs)} '
                        f'(vocabularies have matching sizes)'
                    )
                else:
                    logger.warning(
                        'Cannot share encoder-decoder embeddings: no overlapping languages and vocabularies do not match. '
                        f'Encoder languages: {sorted(encoder_langs)}, Decoder languages: {sorted(decoder_langs)}'
                    )
                    dec_token_embs = None

    decoder = build_xcoder(
        side=Side.decoder,
        model_opts=model_opts,
        vocabs_dict=vocabs_dict,
        device=device,
        task_queue_manager=task_queue_manager,
        single_task=single_task,
        token_embs=dec_token_embs,
    )

    # Check if encoder and decoder have different dimensions
    # If so, skip attention bridge (it requires matching dimensions)
    enc_dim = model_opts.enc_model_dim if getattr(model_opts, 'enc_model_dim', None) is not None else model_opts.model_dim
    dec_dim = model_opts.dec_model_dim if getattr(model_opts, 'dec_model_dim', None) is not None else model_opts.model_dim

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

    def _freeze(module):
        nonlocal frozen_count
        for param in module.parameters():
            param.requires_grad = False
            frozen_count += param.numel()

    for layer_stack_index in model.decoder.attention_layers_by_xcoder_id.keys():
        for xcoder_id, stack in model.decoder.attention_layers_by_xcoder_id[layer_stack_index].items():
            for block in stack.blocks:
                _freeze(block.norm1)
                _freeze(block.self_attn)
                if freeze_cross_attn:
                    _freeze(block.norm2)
                    _freeze(block.cross_attn)
                _freeze(block.norm3)
                _freeze(block.ff)

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

    for layer_stack_index in model.decoder.attention_layers_by_xcoder_id.keys():
        for xcoder_id, stack in model.decoder.attention_layers_by_xcoder_id[layer_stack_index].items():
            for block in stack.blocks:
                for param in block.norm2.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
                for param in block.cross_attn.parameters():
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
