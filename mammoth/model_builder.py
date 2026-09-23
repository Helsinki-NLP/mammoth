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

    # side-prefixed opts (e.g. dec_attn_dim_head) override the shared/global
    # ones, so encoder and decoder can have independently-shaped attention
    # (needed for Gemma3: head_dim=256 while dim/heads=160, and Gemma3-only
    # features like GQA/QK-norm/sandwich-norm/sliding-window/dual-RoPE must
    # not leak into a plain encoder stack). See CLAUDE.md "Gemma3-270M ->
    # Mammoth Conversion" for the source of these knobs
    # (transformers.models.gemma3.configuration_gemma3.Gemma3TextConfig).
    prefix = 'enc' if side == Side.encoder else 'dec'

    def side_opt(name, default):
        value = getattr(model_opts, f'{prefix}_{name}', None)
        return value if value is not None else getattr(model_opts, name, default)

    heads = side_opt('heads', 8)
    dim_head = side_opt('attn_dim_head', dim // heads)
    activation = side_opt('ff_activation', 'swiglu')
    # ff_mult has no static argparse default (see opts.py) so that an unset
    # value here genuinely means "not provided" and falls through to the
    # activation-based default below, rather than always resolving to
    # swiglu's 2.67 regardless of ff_activation.
    ff_mult = side_opt('ff_mult', None)
    if ff_mult is None:
        ff_mult = 4.0 if activation == 'gelu' else 2.67
    attn_drop = side_opt('attn_dropout', 0.0)
    ff_drop = side_opt('ff_dropout', 0.0)
    use_rotary = getattr(model_opts, 'rotary_pos_emb', False)
    use_post_emb_norm = getattr(model_opts, 'post_emb_norm', False)
    depths = model_opts.enc_layers if side == Side.encoder else model_opts.dec_layers
    block_cls = EncoderBlock if side == Side.encoder else DecoderBlock

    # Gemma3-style decoder-only extensions (all no-ops unless set; EncoderBlock
    # does not accept them, so they are never resolved/used on the encoder side).
    kv_heads = getattr(model_opts, f'{prefix}_attn_kv_heads', None)
    qk_norm = getattr(model_opts, f'{prefix}_attn_qk_norm', False)
    attn_scale = getattr(model_opts, f'{prefix}_attn_scale', None)
    sandwich_norm = getattr(model_opts, f'{prefix}_sandwich_norm', False)
    sliding_window = getattr(model_opts, f'{prefix}_sliding_window', -1) or -1
    global_attn_every_n_layers = getattr(model_opts, f'{prefix}_global_attn_every_n_layers', 3)
    global_rope_theta = getattr(model_opts, f'{prefix}_global_rope_theta', None)
    local_rope_theta = getattr(model_opts, f'{prefix}_local_rope_theta', None)
    embed_scale = dim ** 0.5 if getattr(model_opts, f'{prefix}_scaled_embeddings', False) else None
    # RMSNorm epsilon: None keeps torch's dtype-based default (pre-existing
    # behavior); Gemma3 requires the fixed 1e-6 from its config (see
    # DecoderBlock's norm_eps docstring for why this isn't just cosmetic).
    norm_eps = getattr(model_opts, f'{prefix}_norm_eps', None)
    # Cross-attn K/V context dim: defaults to enc_model_dim so a decoder
    # naturally cross-attends to whatever the encoder actually produces,
    # without requiring dim_head/dim to match encoder and decoder (see
    # CLAUDE.md "Gemma3-270M -> Mammoth Conversion" known-limitations note).
    context_dim = getattr(model_opts, 'dec_cross_attn_dim_context', None)
    if context_dim is None:
        context_dim = getattr(model_opts, 'enc_model_dim', None)
    # True decoder-only mode (CLAUDE.md "Gemma3-270M -> Mammoth Conversion",
    # "option 1"): no cross-attn module is built or run at all, as opposed to
    # the fake-encoder path which keeps cross-attn but zeros its output.
    decoder_only = getattr(model_opts, 'decoder_only', False)

    # Dual-theta rope (Gemma3-style): every DecoderBlock gets its own
    # per-layer RotaryEmbedding below (`layer_rotary_emb`) and always
    # overrides whatever `rotary` tuple is passed into it (see
    # DecoderBlock.forward). In that case the generic per-component
    # RotaryEmbedding built further down (`per_component_rotary_embs`) would
    # never be used in forward -- just dead weight held as a submodule and
    # written into the wrapper checkpoint file. Skip building it whenever
    # blocks own their own rotary.
    blocks_own_their_rotary = (
        block_cls is DecoderBlock
        and use_rotary
        and global_rope_theta is not None
        and local_rope_theta is not None
    )

    def make_block(layer_idx: int) -> nn.Module:
        if block_cls is not DecoderBlock:
            return EncoderBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult,
                                 attn_dropout=attn_drop, ff_dropout=ff_drop, activation=activation,
                                 norm_eps=norm_eps)

        # Gemma3 layer_types pattern: full/global attention is the LAST layer
        # of every group of `global_attn_every_n_layers`, the rest are local
        # sliding-window layers (see Gemma3TextConfig.__init__).
        is_global_layer = (sliding_window <= 0) or ((layer_idx + 1) % global_attn_every_n_layers == 0)
        layer_sliding_window = None if is_global_layer else sliding_window
        layer_rotary_emb = None
        if use_rotary and global_rope_theta is not None and local_rope_theta is not None:
            theta = global_rope_theta if is_global_layer else local_rope_theta
            layer_rotary_emb = RotaryEmbedding(dim_head, base=theta)

        return DecoderBlock(
            dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult,
            attn_dropout=attn_drop, ff_dropout=ff_drop, activation=activation,
            norm_eps=norm_eps, kv_heads=kv_heads, qk_norm=qk_norm, attn_scale=attn_scale,
            sandwich_norm=sandwich_norm, sliding_window=layer_sliding_window,
            rotary_emb=layer_rotary_emb, context_dim=context_dim,
            use_cross_attn=not decoder_only,
        )

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
        blocks = nn.ModuleList([make_block(i) for i in range(depth)])
        final_norm = nn.RMSNorm(dim, eps=norm_eps) if is_last_stack else None
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
            nn.RMSNorm(dim, eps=norm_eps) if use_post_emb_norm else nn.Identity()
        )
        per_component_rotary_embs[component_key] = (
            RotaryEmbedding(dim_head) if (use_rotary and not blocks_own_their_rotary) else None
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
            embed_scale=embed_scale,
        )

    # Build shared nn.ModuleDicts: StackXcoder is the single owner of all parameters.
    shared_stacks_dict = {
        f'{layer_stack_index}__{xcoder_id}': stack
        for layer_stack_index, xcoder_dict in attention_layer_blocks.items()
        for xcoder_id, stack in xcoder_dict.items()
    }
    shared_rotary_embs_dict = {
        '__'.join(k): v
        for k, v in per_component_rotary_embs.items()
        if v is not None
    }
    return StackXcoder(
        task_wrappers=transformer_wrappers,
        attention_layer_blocks=dict(attention_layer_blocks),
        token_embs=token_embs,
        shared_stacks=nn.ModuleDict(shared_stacks_dict),
        shared_token_embs=nn.ModuleDict(token_embs),
        shared_post_emb_norms=nn.ModuleDict({'__'.join(k): v for k, v in per_component_post_emb_norms.items()}),
        shared_rotary_embs=nn.ModuleDict(shared_rotary_embs_dict),
        shared_to_logits=nn.ModuleDict({'__'.join(k): v for k, v in per_component_to_logits.items()}),
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

    # True decoder-only mode (CLAUDE.md "Gemma3-270M -> Mammoth Conversion",
    # "option 1"): no encoder is built at all -- not even the small
    # randomly-initialized "fake encoder" used by the alternative path
    # (convert_gemma3_native.py). See NMTModel.forward for how encoder=None
    # is handled at inference time.
    decoder_only = getattr(model_opts, 'decoder_only', False)

    encoder = None
    dec_token_embs = None
    if not decoder_only:
        encoder = build_xcoder(
            side=Side.encoder,
            model_opts=model_opts,
            vocabs_dict=vocabs_dict,
            device=device,
            task_queue_manager=task_queue_manager,
            single_task=single_task,
        )

    # Optionally share embeddings between encoder and decoder
    share_embeddings = (not decoder_only) and getattr(model_opts, 'share_encoder_decoder_embeddings', False)

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
                    dec_token_embs[lang] = encoder.shared_token_embs[lang]
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
    if decoder_only:
        attention_bridge = None
    else:
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

    Iterates stack.named_parameters() rather than listing block submodules by
    name, so this automatically covers every parameter the stack owns -
    including DecoderBlock's optional sandwich-norm post-branch norms
    (norm1_post/norm3_post, only present when sandwich_norm=True, e.g. the
    Gemma3 conversion) and TransformerStack.final_norm - without having to
    keep a manually-enumerated list in sync with block.py.

    Also freezes the shared vocabulary output projection (to_logits), since
    it is decoder "body" output, not embeddings or cross-attention, and is
    stored outside the stack (StackXcoder.shared_to_logits) so the stack loop
    can't reach it.

    Args:
        model: NMTModel instance
        freeze_cross_attn: Whether to freeze cross-attention layers (default: True)

    Returns:
        Number of frozen parameters
    """
    frozen_count = 0

    for layer_stack_index in model.decoder.attention_layers_by_xcoder_id.keys():
        for xcoder_id, stack in model.decoder.attention_layers_by_xcoder_id[layer_stack_index].items():
            for name, param in stack.named_parameters():
                if not freeze_cross_attn and ('.cross_attn.' in name or '.norm2.' in name):
                    continue
                param.requires_grad = False
                frozen_count += param.numel()

    for head in model.decoder.shared_to_logits.values():
        for param in head.parameters():
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

    for layer_stack_index in model.decoder.attention_layers_by_xcoder_id.keys():
        for xcoder_id, stack in model.decoder.attention_layers_by_xcoder_id[layer_stack_index].items():
            for name, param in stack.named_parameters():
                if '.cross_attn.' in name or '.norm2.' in name:
                    param.requires_grad = False
                    frozen_count += param.numel()

    return frozen_count


def _freeze_embeddings(xcoder):
    """
    Freeze all token embeddings in an encoder or decoder, along with the
    shared post-embedding norm (StackXcoder.shared_post_emb_norms) - it is
    applied directly to the token embedding output before any transformer
    block runs, so it is grouped with "embeddings" rather than with the
    attention/feedforward freeze functions.

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

    for norm in xcoder.shared_post_emb_norms.values():
        for param in norm.parameters():
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
