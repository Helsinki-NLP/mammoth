"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
from collections import defaultdict
from functools import partial
from pathlib import Path
from torch.nn.init import xavier_uniform_
from typing import Optional, List
from x_transformers import TransformerWrapper

from mammoth.distributed.components import (
    DistributedAdapter,
    DistributedComponent,
    DistributedDecoder,
    DistributedEncoder,
    Side,
)
from mammoth.models import NMTModel
from mammoth.modules.adapters import (
    AdaptedAttentionLayers,
    Adapter,
    FeedForwardAdapterLayer,
    LoraAdapterLayer,
)
from mammoth.modules.layer_stack import AdaptedAttentionLayersStack, StackXcoder
from mammoth.utils.logging import logger
from mammoth.utils.misc import use_gpu
from mammoth.utils.module_splitter import _combine_ordered_dicts
from mammoth.utils.parse import ArgumentParser

from mammoth.modules.attention_bridge import AttentionBridge


def uses_adapters(opts):
    return 'adapters' in opts and opts.adapters


def load_test_multitask_model(opts, task_queue_manager, task=None, model_path=None):
    if task is None:
        raise ValueError('Must set task')
    if model_path is None:
        model_path = opts.models[0]

    checkpoint_modules = [
        (f'encoder.embeddings.embeddings_{task.src_lang}.', f'src_embeddings_{task.src_lang}'),
        (f'decoder.embeddings.embeddings_{task.tgt_lang}.', f'tgt_embeddings_{task.tgt_lang}'),
        (f'generator.generator_{task.tgt_lang}.', f'generator_{task.tgt_lang}'),
        ('attention_bridge.', 'attention_bridge'),
    ]

    for layer_stack_idx, layer_stack_key in enumerate(task.encoder_id):
        checkpoint_modules.append(
            (
                f'encoder.encoders.{layer_stack_idx}.{layer_stack_key}.',
                f'encoder_{layer_stack_idx}_{layer_stack_key}'
            )
        )
    if task.encoder_adapter_ids:
        for layer_stack_idx, adapter_group, sub_id in task.encoder_adapter_ids:
            checkpoint_modules.append(
                (
                    f'encoder.encoders.{layer_stack_idx}.{layer_stack_key}.adapters.adapter_{adapter_group}_{sub_id}.',    # noqa
                    f'encoder_adapter_{layer_stack_idx}_{layer_stack_key}_{adapter_group}_{sub_id}'
                )
            )
    for layer_stack_idx, layer_stack_key in enumerate(task.decoder_id):
        checkpoint_modules.append(
            (
                f'decoder.decoders.{layer_stack_idx}.{layer_stack_key}.',
                f'decoder_{layer_stack_idx}_{layer_stack_key}'
            )
        )
    if task.decoder_adapter_ids:
        for layer_stack_idx, adapter_group, sub_id in task.decoder_adapter_ids:
            checkpoint_modules.append(
                (
                    f'decoder.decoders.{layer_stack_idx}.{layer_stack_key}.adapters.adapter_{adapter_group}_{sub_id}.',    # noqa
                    f'decoder_adapter_{layer_stack_idx}_{layer_stack_key}_{adapter_group}_{sub_id}'
                )
            )

    model_path = model_path.rstrip('_')
    checkpoint_paths = [
        (prefix, f'{model_path}_{key}.pt') for (prefix, key) in checkpoint_modules
    ]

    opts.model_frame = model_path + '_frame.pt'
    frame = torch.load(opts.model_frame, map_location=lambda storage, loc: storage)

    checkpoint_state_dicts = {
        prefix: torch.load(path, map_location=lambda storage, loc: storage)
        for prefix, path in checkpoint_paths
    }

    combined_state_dict = _combine_ordered_dicts(checkpoint_state_dicts)

    vocabs_dict = {
        'src': frame["vocab"].get(('src', task.src_lang)),
        'tgt': frame["vocab"].get(('tgt', task.tgt_lang)),
    }
    # FIXME
    # fields["indices"] = Field(use_vocab=False, dtype=torch.long, sequential=False)

    model_opts = ArgumentParser.ckpt_model_opts(frame['opts'])
    # Avoid functionality on inference
    # model_opts.update_vocab = False
    model = build_model(
        model_opts,
        opts,
        vocabs_dict,
        task_queue_manager,
        checkpoint=None,
        single_task=task.corpus_id,
    )
    model_params = {name for name, p in model.named_parameters()}
    model_params.update(name for name, p in model.named_buffers())
    for key in set(combined_state_dict.keys()):
        if key not in model_params:
            print(f'Deleting unnecessary key: {key}')
            del combined_state_dict[key]
    for key in model_params:
        if key not in combined_state_dict:
            print(f'Key missing {key}')
    model.load_state_dict(combined_state_dict)
    device = torch.device("cuda" if use_gpu(opts) else "cpu")
    model.to(device)

    model.eval()

    return vocabs_dict, model, model_opts


def get_attention_layers_kwargs(
    side: Side,
    layer_stack_index,
    xcoder_id,
    model_opts,
):
    """Return arguments for x_transformers.AttentionLayers"""
    depths = model_opts.enc_layers if side == Side.decoder else model_opts.dec_layers
    depth = depths[layer_stack_index]
    causal = side == Side.decoder
    cross_attend = side == Side.decoder
    is_last = layer_stack_index == len(depths) - 1
    # changed from default
    use_simple_rmsnorm = True
    attn_flash = True
    ff_glu = True
    pre_norm_has_final_norm = is_last
    # Mostly x_transformers defaults. Make (some of this) configurable.
    return {
        'dim': model_opts.model_dim,
        'depth': depth,
        'heads': model_opts.heads,
        'causal': causal,
        'cross_attend': cross_attend,
        'only_cross': False,
        'use_scalenorm': False,
        'use_rmsnorm': False,
        'use_simple_rmsnorm': use_simple_rmsnorm,
        'use_adaptive_layernorm': False,
        'use_adaptive_rmsnorm': False,
        'use_adaptive_layerscale': False,
        'norm_add_unit_offset': True,
        'dim_condition': None,
        'adaptive_condition_mlp': False,
        'adaptive_condition_mlp_expansion': 4,
        'alibi_pos_bias': False,
        'alibi_num_heads': None,
        'rel_pos_bias': False,
        'rel_pos_num_buckets': 32,
        'rel_pos_max_distance': 128,
        'dynamic_pos_bias': False,
        'dynamic_pos_bias_log_distance': False,
        'dynamic_pos_bias_mlp_depth': 2,
        'dynamic_pos_bias_norm': False,
        'rotary_pos_emb': False,
        'rotary_emb_dim': None,
        'rotary_xpos': False,
        'rotary_interpolation_factor': 1.,
        'rotary_xpos_scale_base': 512,
        'rotary_base_rescale_factor': 1.,
        'weight_tie_layers': False,
        'custom_layers': None,
        'layers_execute_order': None,
        # 'sandwich_coef': None,    # Sandwich would be very unintuitive with multiple layerstacks
        'par_ratio': None,
        'residual_attn': False,
        'cross_residual_attn': False,
        # 'macaron': False,         # Can not support macaron and inject adapters at each 'f' layer
        'pre_norm': True,
        'pre_norm_has_final_norm': pre_norm_has_final_norm,
        'gate_residual': False,
        'scale_residual': False,
        'scale_residual_constant': 1.,
        'shift_tokens': 0,
        'sandwich_norm': False,
        'softclamp_output': False,
        'softclamp_output_value': 30.,
        'resi_dual': False,
        'resi_dual_scale': 1.,
        'zero_init_branch_output': False,
        'layer_dropout': 0.,
        'cross_attn_tokens_dropout': 0.,
        'disable_abs_pos_emb': None,
        'use_layerscale': False,
        'layerscale_init_value': 0.,

        'ff_dim_out': None,
        'ff_mult': model_opts.ff_mult,
        'ff_glu': ff_glu,
        'ff_glu_mult_bias': False,
        'ff_swish': False,
        'ff_relu_squared': False,
        'ff_post_act_ln': False,
        'ff_dropout': 0.,
        'ff_no_bias': False,
        'ff_zero_init_output': False,

        'attn_dim_context': None,
        'attn_flash': attn_flash,
        'attn_talking_heads': False,
        'attn_head_scale': False,
        'attn_sparse_topk': None,
        'attn_num_mem_kv': 0,
        'attn_dropout': 0.,
        'attn_on_attn': False,
        'attn_gate_value_heads': False,
        'attn_swiglu_values': False,
        'attn_gate_values': False,
        'attn_zero_init_output': False,
        'attn_max_attend_past': None,
        'attn_qk_norm': False,
        'attn_qk_norm_groups': 1,
        'attn_qk_norm_scale': 10,
        'attn_qk_norm_dim_scale': False,
        'attn_one_kv_head': False,
        'attn_kv_heads': None,
        'attn_shared_kv': False,
        'attn_value_dim_head': None,
        'attn_tensor_product': False,      # https://arxiv.org/abs/2208.06061
        'attn_add_zero_kv': False,         # same as add_zero_attn in pytorch
        'attn_rotary_embed_values': False,
        'attn_use_cope': False,
        'attn_cope_max_pos': 16,
        'attn_cope_soft_onehot_pos': False,
        'attn_cope_talking_heads': False,
        'attn_softclamp_logits': False,
        'attn_logit_softclamp_value': 50.,
        'attn_onnxable': False,
    }


def build_xcoder(
    side: Side,
    model_opts,
    vocabs_dict,
    device,
    task_queue_manager,
    single_task: Optional[str] = None,
):
    my_components: List[DistributedComponent] = task_queue_manager.get_my_distributed_components()
    my_components = [
        component for component in my_components
        if hasattr(component, 'side') and component.side == side
    ]
    distributed_xcoder_class: type
    if side == Side.encoder:
        distributed_xcoder_class = DistributedEncoder
        side_str = 'encoder'
    else:
        distributed_xcoder_class = DistributedDecoder
        side_str = 'decoder'
    if single_task:
        my_components = [
            component for component in my_components
            if single_task in component.task_ids
        ]

    # Create AdaptedAttentionLayers objects (an extension of an x_transformers.AttentionLayers block)
    attention_layers_components = [
        component for component in my_components
        if isinstance(component, distributed_xcoder_class)
    ]
    attention_layer_blocks = defaultdict(dict)
    for component in attention_layers_components:
        layer_stack_index = component.layer_stack_index
        xcoder_id = component.xcoder_id
        attention_layers_kwargs = get_attention_layers_kwargs(
            side=side,
            layer_stack_index=layer_stack_index,
            xcoder_id=xcoder_id,
            model_opts=model_opts,
        )
        attention_layer_blocks[layer_stack_index][xcoder_id] = AdaptedAttentionLayers(**attention_layers_kwargs)

    # Create AdapterLayer objects and Adapter objects
    if uses_adapters(model_opts):
        adapter_components = [
            component for component in my_components
            if isinstance(component, DistributedAdapter) and component.side == side
        ]
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
            if model_opts.adapter_type.lower() == 'lora':
                adapter_layer_func = partial(
                        LoraAdapterLayer,
                        dim=model_opts.model_dim,
                        r=adapter_params['hidden_dim'],
                    )
            elif model_opts.adapter_type.lower() == 'ff':
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
                raise ValueError(f'Unrecognized adapter_type {model_opts.adapter_type}')
            for sub_id in adapter_params['sub_ids']:
                for layer_idx in adapter_params['layers']:
                    adapter_layer = adapter_layer_func()
                    adapter = Adapter(
                        adapter_group=component.adapter_group,
                        sub_id=sub_id,
                    )
                    adapter.add_layer(layer_idx, adapter_layer)
                    layer_stack_index = adapter_params['layer_stack_index']
                    for attention_layers in attention_layer_blocks[layer_stack_index]:
                        attention_layers.add_adapter(adapter)

    # Create AdaptedAttentionLayersStack objects and TransformerWrapper objects
    tasks = task_queue_manager.get_my_tasks()
    if single_task:
        tasks = [task for task in tasks if task.corpus_id == single_task]
    transformer_wrappers = dict()
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

        side_alt_str = 'src' if side == Side.encoder else 'tgt'
        lang = task.src_lang if side == Side.encoder else task.tgt_lang
        vocab = vocabs_dict[(side_alt_str, lang)]
        max_seq_len = 0 if model_opts.max_length is None else model_opts.max_length
        post_emb_norm = True
        tie_embedding = True
        use_abs_pos_emb = True
        emb_frac_gradient = 1.
        # FIXME: this won't work: creates embeddings for each task, not for each language
        # Have to reimplement TransformerWrapper to allow passing in an embedding
        transformer_wrapper = TransformerWrapper(
            num_tokens=len(vocab),
            max_seq_len=max_seq_len,
            attn_layers=adapted_attention_layers_stack,
            emb_dim=model_opts.model_dim,
            post_emb_norm=post_emb_norm,
            tie_embedding=tie_embedding,
            use_abs_pos_emb=use_abs_pos_emb,
            emb_frac_gradient=emb_frac_gradient,
        )
        transformer_wrappers[task.corpus_id] = transformer_wrapper

    # Create a StackXcoder
    stack_xcoder = StackXcoder(transformer_wrappers)
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


def restore_from_checkpoint(stack_xcoder, checkpoint):
    # FIXME: saving and loading are broken
    trainstep = int(checkpoint['optim']['training_step']) - 1
    for modname, gen in generators_md.items():
        mod_path = Path(checkpoint['opts'].save_model + f"_step_{trainstep}_{modname}.pt")
        if mod_path.exists():
            module = torch.load(mod_path)
            gen.load_state_dict(module)
            logger.info(f"Successfully loaded {modname} from the checkpoint.")

    pluggable_tgt_emb = PluggableEmbeddings(tgt_embs)
    decoder = build_only_dec(model_opts, pluggable_tgt_emb, task_queue_manager, checkpoint)

    # TODO: implement hierarchical approach to layer sharing
    attention_bridge = build_attention_bridge(model_opts)

    if checkpoint:
        # trainstep= int(checkpoint['optim']['training_step'])-1 - already recoderd in generators
        attn_path = Path(checkpoint['opts'].save_model + f"_step_{trainstep}_attention_bridge.pt")
        if attn_path.exists():
            attention_bridge.load_state_dict(torch.load(attn_path))
            logger.info("Successfully loaded the attention bridge  from the checkpoint.")

    if model_opts.model_dtype == 'fp16' and model_opts.optim == 'fusedadam':
        attention_bridge.half()

    if uses_adapters(model_opts):
        logger.info('Creating adapters...')
        create_all_adapters(nmt_model, model_opts, task_queue_manager)
        if checkpoint:
            # TODO: plug in properly
            logger.warning("Adapters' parameters are NOT being loaded from the checkpoint.")
    print('built model:')
    print(nmt_model)

    return nmt_model, generators_md


def build_model(
    model_opts,
    opts,
    vocabs_dict,
    task_queue_manager,
    checkpoint=None,
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
        checkpoint: the model generated by train phase, or a resumed snapshot
                    model from a stopped training.
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

    encoder = build_xcoder(
        side=Side.encoder,
        model_opts=model_opts,
        vocabs_dict=vocabs_dict,
        device=device,
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
    )
    attention_bridge = build_attention_bridge(model_opts)
    model = NMTModel(
        encoder=encoder,
        decoder=decoder,
        attention_bridge=attention_bridge
    )

    model.to(device)
    # logger.info(model)
    logger.info('Building model - done!')
    return model
