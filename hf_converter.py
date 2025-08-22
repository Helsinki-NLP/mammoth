#!/usr/bin/env python3
"""
HuggingFace BART to Mammoth Model Converter
Usage: python converter.py <hf_model_path> <save_path> [--src-lang en] [--tgt-lang es]
"""

import os
import torch
from collections import OrderedDict
from argparse import Namespace
from transformers import AutoConfig, BartForConditionalGeneration, AutoTokenizer
from mammoth.x_transformers import XTransformer
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS
from mammoth.distributed.tasks import TaskQueueManager
from mammoth.distributed.contexts import WorldContext, DeviceContext, DeviceContextEnum
from mammoth.model_builder import build_model
from mammoth.utils.optimizers import MultipleOptimizer
from mammoth.utils.model_saver import build_model_saver


# =============================================================================
# SECTION 1: HuggingFace to x-transformers conversion
# =============================================================================

def create_xtransformer_model(model_path):
    """Create XTransformer model from HF BART config"""
    config = AutoConfig.from_pretrained(model_path)
    
    xt_model = XTransformer(
        dim=config.d_model,
        tie_token_emb=True,
        layernorm_bias=True,
        # Encoder settings
        enc_num_tokens=config.vocab_size,
        enc_max_seq_len=config.max_position_embeddings + 2,
        enc_use_abs_pos_emb=True,
        enc_pre_norm=getattr(config, "normalize_before", False),
        enc_depth=config.encoder_layers,
        enc_heads=config.encoder_attention_heads,
        enc_layer_dropout=getattr(config, "encoder_layerdrop", 0.0),
        enc_post_emb_norm=getattr(config, "normalize_embedding", True),
        enc_post_emb_norm_bias=getattr(config, "normalize_embedding", True),
        enc_attn_qkv_bias=True,
        enc_attn_out_bias=True,
        enc_attn_dropout=getattr(config, "attention_dropout", 0.1),
        enc_ff_dropout=getattr(config, "activation_dropout", 0.1),
        enc_emb_dropout=getattr(config, "dropout", 0.1),
        enc_attn_flash=True,
        # Decoder settings
        dec_num_tokens=config.vocab_size,
        dec_max_seq_len=config.max_position_embeddings + 2,
        dec_use_abs_pos_emb=True,
        dec_pre_norm=getattr(config, "normalize_before", False),
        dec_depth=config.decoder_layers,
        dec_heads=config.decoder_attention_heads,
        dec_layer_dropout=getattr(config, "decoder_layerdrop", 0.0),
        dec_attn_qkv_bias=True,
        dec_attn_out_bias=True,
        dec_use_final_logits_bias=True,
        dec_post_emb_norm=getattr(config, "normalize_embedding", True),
        dec_post_emb_norm_bias=getattr(config, "normalize_embedding", True),
        dec_attn_dropout=getattr(config, "attention_dropout", 0.1),
        dec_ff_dropout=getattr(config, "activation_dropout", 0.1),
        dec_emb_dropout=getattr(config, "dropout", 0.1),
        dec_attn_flash=True,
    )
    return xt_model


def create_weight_mapping(encoder_layers, decoder_layers):
    """Create mapping from HuggingFace BART to x-transformers"""
    mapping = {}
    
    # Token embedding mapping
    mapping["model.shared.weight"] = "encoder.token_emb.emb.weight"
    mapping["model.encoder.embed_tokens.weight"] = "encoder.token_emb.emb.weight"
    mapping["model.encoder.embed_positions.weight"] = "encoder.pos_emb.emb.weight"
    mapping["model.encoder.layernorm_embedding.weight"] = "encoder.post_emb_norm.ln.weight"
    mapping["model.encoder.layernorm_embedding.bias"] = "encoder.post_emb_norm.ln.bias"
    
    # Decoder embeddings
    mapping["model.decoder.embed_tokens.weight"] = "decoder.net.token_emb.emb.weight"
    mapping["model.decoder.embed_positions.weight"] = "decoder.net.pos_emb.emb.weight"
    mapping["model.decoder.layernorm_embedding.weight"] = "decoder.net.post_emb_norm.ln.weight"
    mapping["model.decoder.layernorm_embedding.bias"] = "decoder.net.post_emb_norm.ln.bias"
    
    # Encoder layers
    for i in range(encoder_layers):
        attn_idx = i * 2
        # Self-attention
        mapping[f"model.encoder.layers.{i}.self_attn_layer_norm.weight"] = f"encoder.attn_layers.layers.{attn_idx}.0.2.ln.weight"
        mapping[f"model.encoder.layers.{i}.self_attn_layer_norm.bias"] = f"encoder.attn_layers.layers.{attn_idx}.0.2.ln.bias"
        mapping[f"model.encoder.layers.{i}.self_attn.q_proj.weight"] = f"encoder.attn_layers.layers.{attn_idx}.1.to_q.weight"
        mapping[f"model.encoder.layers.{i}.self_attn.q_proj.bias"] = f"encoder.attn_layers.layers.{attn_idx}.1.to_q.bias"
        mapping[f"model.encoder.layers.{i}.self_attn.k_proj.weight"] = f"encoder.attn_layers.layers.{attn_idx}.1.to_k.weight"
        mapping[f"model.encoder.layers.{i}.self_attn.k_proj.bias"] = f"encoder.attn_layers.layers.{attn_idx}.1.to_k.bias"
        mapping[f"model.encoder.layers.{i}.self_attn.v_proj.weight"] = f"encoder.attn_layers.layers.{attn_idx}.1.to_v.weight"
        mapping[f"model.encoder.layers.{i}.self_attn.v_proj.bias"] = f"encoder.attn_layers.layers.{attn_idx}.1.to_v.bias"
        mapping[f"model.encoder.layers.{i}.self_attn.out_proj.weight"] = f"encoder.attn_layers.layers.{attn_idx}.1.to_out.weight"
        mapping[f"model.encoder.layers.{i}.self_attn.out_proj.bias"] = f"encoder.attn_layers.layers.{attn_idx}.1.to_out.bias"
        
        # Feedforward
        ff_idx = attn_idx + 1
        mapping[f"model.encoder.layers.{i}.final_layer_norm.weight"] = f"encoder.attn_layers.layers.{ff_idx}.0.2.ln.weight"
        mapping[f"model.encoder.layers.{i}.final_layer_norm.bias"] = f"encoder.attn_layers.layers.{ff_idx}.0.2.ln.bias"
        mapping[f"model.encoder.layers.{i}.fc1.weight"] = f"encoder.attn_layers.layers.{ff_idx}.1.ff.0.weight"
        mapping[f"model.encoder.layers.{i}.fc1.bias"] = f"encoder.attn_layers.layers.{ff_idx}.1.ff.0.bias"
        mapping[f"model.encoder.layers.{i}.fc2.weight"] = f"encoder.attn_layers.layers.{ff_idx}.1.ff.3.weight"
        mapping[f"model.encoder.layers.{i}.fc2.bias"] = f"encoder.attn_layers.layers.{ff_idx}.1.ff.3.bias"
    
    # Decoder layers
    for i in range(decoder_layers):
        base_idx = i * 3
        # Self-attention
        self_attn_idx = base_idx
        mapping[f"model.decoder.layers.{i}.self_attn_layer_norm.weight"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.0.2.ln.weight"
        mapping[f"model.decoder.layers.{i}.self_attn_layer_norm.bias"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.0.2.ln.bias"
        mapping[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.1.to_q.weight"
        mapping[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.1.to_q.bias"
        mapping[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.1.to_k.weight"
        mapping[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.1.to_k.bias"
        mapping[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.1.to_v.weight"
        mapping[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.1.to_v.bias"
        mapping[f"model.decoder.layers.{i}.self_attn.out_proj.weight"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.1.to_out.weight"
        mapping[f"model.decoder.layers.{i}.self_attn.out_proj.bias"] = f"decoder.net.attn_layers.layers.{self_attn_idx}.1.to_out.bias"
        
        # Cross-attention
        cross_attn_idx = base_idx + 1
        mapping[f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.0.2.ln.weight"
        mapping[f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.0.2.ln.bias"
        mapping[f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.1.to_q.weight"
        mapping[f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.1.to_q.bias"
        mapping[f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.1.to_k.weight"
        mapping[f"model.decoder.layers.{i}.encoder_attn.k_proj.bias"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.1.to_k.bias"
        mapping[f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.1.to_v.weight"
        mapping[f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.1.to_v.bias"
        mapping[f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.1.to_out.weight"
        mapping[f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"] = f"decoder.net.attn_layers.layers.{cross_attn_idx}.1.to_out.bias"
        
        # Feedforward
        ff_idx = base_idx + 2
        mapping[f"model.decoder.layers.{i}.final_layer_norm.weight"] = f"decoder.net.attn_layers.layers.{ff_idx}.0.2.ln.weight"
        mapping[f"model.decoder.layers.{i}.final_layer_norm.bias"] = f"decoder.net.attn_layers.layers.{ff_idx}.0.2.ln.bias"
        mapping[f"model.decoder.layers.{i}.fc1.weight"] = f"decoder.net.attn_layers.layers.{ff_idx}.1.ff.0.weight"
        mapping[f"model.decoder.layers.{i}.fc1.bias"] = f"decoder.net.attn_layers.layers.{ff_idx}.1.ff.0.bias"
        mapping[f"model.decoder.layers.{i}.fc2.weight"] = f"decoder.net.attn_layers.layers.{ff_idx}.1.ff.3.weight"
        mapping[f"model.decoder.layers.{i}.fc2.bias"] = f"decoder.net.attn_layers.layers.{ff_idx}.1.ff.3.bias"
    
    # Output layers
    mapping["lm_head.weight"] = "decoder.net.to_logits.weight"
    mapping["final_logits_bias"] = "decoder.net.to_logits.bias"
    return mapping


def load_hf_weights_to_xtransformer(hf_model_path, xt_model):
    """Load weights from HuggingFace model to x-transformers model"""
    print(f"Loading HF model from {hf_model_path}")
    config = AutoConfig.from_pretrained(hf_model_path)
    hf_model = BartForConditionalGeneration.from_pretrained(hf_model_path)
    hf_state_dict = hf_model.state_dict()
    
    mapping = create_weight_mapping(config.encoder_layers, config.decoder_layers)
    x_state_dict = OrderedDict()
    
    for hf_key, x_key in mapping.items():
        if hf_key in hf_state_dict:
            if hf_key == "final_logits_bias":
                x_state_dict[x_key] = hf_state_dict[hf_key].squeeze(0)
            else:
                x_state_dict[x_key] = hf_state_dict[hf_key]
    
    xt_model.load_state_dict(x_state_dict, strict=False)
    print("✓ HF weights loaded to x-transformers model")
    return xt_model


# =============================================================================
# SECTION 2: Mammoth utilities and configuration
# =============================================================================

def create_vocabs_dict_from_hf_tokenizer(hf_model_path, src_lang="en", tgt_lang="es"):
    """Create vocabs_dict from HuggingFace tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    vocab_items = [tok for tok, idx in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])]
    
    specials = list(DEFAULT_SPECIALS)
    src_vocab = Vocab(path=None, items=vocab_items, tag=f"src_{src_lang}", 
                      size=len(tokenizer), specials=specials)
    # print(f"Check src_vocabs...")
    # for i in range(min(10, len(src_vocab.itos))):
    #     print(f"  - {src_vocab.itos[i]}")
    tgt_vocab = Vocab(path=None, items=vocab_items, tag=f"tgt_{tgt_lang}", 
                      size=len(tokenizer), specials=specials)
    # print(f"Check tgt_vocabs...")
    # for i in range(min(10, len(tgt_vocab.itos))):
    #     print(f"  - {tgt_vocab.itos[i]}")
    vocabs_dict = {("src", src_lang): src_vocab, ("tgt", tgt_lang): tgt_vocab}
    print(f"✓ Created vocabularies: {len(src_vocab)} tokens")
    return vocabs_dict


def create_model_opts_from_xt_model(xt_model, hf_model_path):
    """Create model_opts for Mammoth from x-transformer model"""
    config = AutoConfig.from_pretrained(hf_model_path)
    model_opts = Namespace()
    
    # Basic settings
    model_opts.model_type = "text"
    model_opts.model_dtype = "fp32"
    model_opts.pos_ffn_activation_fn = getattr(config, "activation_function", "relu")
    
    # Architecture
    model_opts.model_dim = config.d_model
    model_opts.enc_layers = [config.encoder_layers]
    model_opts.dec_layers = [config.decoder_layers]
    
    ff_mult = getattr(config, "encoder_ffn_dim", config.d_model * 4) / config.d_model
    
    model_opts.x_transformers_opts = {
        "heads": getattr(config, "encoder_attention_heads"),
        "ff_mult": ff_mult,
        "attn_dropout": getattr(config, "attention_dropout", 0.1),
        "ff_dropout": getattr(config, "dropout", 0.1),
        "pre_norm": False,
        "post_emb_norm": getattr(config, "normalize_embedding", True),
        "post_emb_norm_bias": getattr(config, "normalize_embedding", True),
        "layernorm_bias": True,
        "use_simple_rmsnorm": False,
        "attn_flash": True,
        "ff_glu": False,
        "scaled_sinu_pos_emb": False,

    }
    
    model_opts.max_length = getattr(config, "max_position_embeddings", 1024) + 2
    model_opts.param_init = 0.0
    model_opts.param_init_glorot = True
    model_opts.attention_bridge = None
    model_opts.ab_layers = []
    model_opts.adapters = None
    model_opts.enable_embeddingless = False
    model_opts.normformer = False
    model_opts.self_attn_type = "scaled-dot"
    model_opts.dropout = [getattr(config, "dropout", 0.1)]
    model_opts.attention_dropout = [getattr(config, "attention_dropout", 0.1)]
    
    return model_opts


class SimpleWorldContext(WorldContext):
    def __init__(self):
        super().__init__(context=1, n_nodes=1, gpus_per_node=1)
        self.context = DeviceContextEnum.SINGLE_GPU
        self.n_nodes = 1
        self.gpus_per_node = 1
    
    def is_distributed(self):
        return False
    
    def global_to_local(self, node_rank, local_rank):
        return SimpleDeviceContext(
            context=self.context, n_nodes=self.n_nodes, gpus_per_node=self.gpus_per_node,
            node_rank=node_rank, local_rank=local_rank)


class SimpleDeviceContext(DeviceContext):
    def __init__(self, context, n_nodes, gpus_per_node, node_rank, local_rank):
        super().__init__(context=context, n_nodes=n_nodes, gpus_per_node=gpus_per_node,
                        node_rank=node_rank, local_rank=local_rank)
    
    def validate(self, world_context):
        super().validate(world_context)
    
    def is_master(self):
        return True
    
    def is_distributed(self):
        return False


def create_task_queue_manager(src_lang="en", tgt_lang="es", corpus_id="bart_translation", vocabs_dict=None):
    """Create TaskQueueManager for single GPU setup"""
    opts = Namespace()
    opts.tasks = {
        corpus_id: {
            "src_tgt": f"{src_lang}-{tgt_lang}",
            "weight": 1.0,
            "introduce_at_training_step": 0,
            "node_gpu": "0:0",
            "enc_sharing_group": [src_lang],
            "dec_sharing_group": [tgt_lang],
        }
    }
    opts.enc_layers = [6]
    opts.dec_layers = [6]
    opts.task_distribution_strategy = "weighted_sampling"
    opts.accum_count = [1]
    opts.seed = 42
    
    world_context = SimpleWorldContext()
    task_manager = TaskQueueManager.from_opts(opts, world_context)
    
    src_vocab = vocabs_dict.get(("src", src_lang))
    tgt_vocab = vocabs_dict.get(("tgt", tgt_lang))
    
    for task in task_manager.tasks:
        task.src_vocab = src_vocab
        task.tgt_vocab = tgt_vocab
    
    local_task_manager = task_manager.global_to_local(node_rank=0, local_rank=0, opts=opts)
    local_task_manager.create_all_distributed_components(
        use_attention_bridge=False,
        new_group_func=lambda ranks: None)
    
    return local_task_manager


def create_opts():
    """Create complete opts object for optimizer"""
    opts = Namespace()
    opts.train_from = None
    opts.reset_optim = "all"
    opts.model_dtype = "fp32"
    opts.optim = "adamw"
    opts.learning_rate = 0.001
    opts.adam_beta1 = 0.9
    opts.adam_beta2 = 0.999
    opts.weight_decay = 0.0
    opts.max_grad_norm = 1.0
    opts.decay_method = "none"
    opts.learning_rate_decay = 0.5
    opts.start_decay_steps = 50000
    opts.decay_steps = 10000
    opts.adagrad_accumulator_init = 0.0
    opts.gpu_ranks = []
    opts.log_model_structure = True
    opts.adapters = None
    return opts


# =============================================================================
# SECTION 3: x-transformers to Mammoth conversion
# =============================================================================

def create_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers):
    """Create mapping from xt_model keys to mammoth_model keys"""
    mapping = {}
    
    # Token embeddings and positional embeddings
    mapping['encoder.token_emb.emb.weight'] = 'encoder.bart_translation.token_emb.emb.weight'
    mapping['encoder.pos_emb.emb.weight'] = 'encoder.bart_translation.pos_emb.emb.weight'
    mapping['encoder.post_emb_norm.ln.weight'] = 'encoder.bart_translation.post_emb_norm.ln.weight'
    mapping['encoder.post_emb_norm.ln.bias'] = 'encoder.bart_translation.post_emb_norm.ln.bias'
    
    # Decoder embeddings
    mapping['decoder.net.token_emb.emb.weight'] = 'decoder.bart_translation.token_emb.emb.weight'
    mapping['decoder.net.pos_emb.emb.weight'] = 'decoder.bart_translation.pos_emb.emb.weight'
    mapping['decoder.net.post_emb_norm.ln.weight'] = 'decoder.bart_translation.post_emb_norm.ln.weight'
    mapping['decoder.net.post_emb_norm.ln.bias'] = 'decoder.bart_translation.post_emb_norm.ln.bias'
    
    # Encoder layers
    for layer_idx in range(num_encoder_layers):
        # Attention layer
        attn_idx = layer_idx * 2
        xt_attn_base = f'encoder.attn_layers.layers.{attn_idx}'
        mammoth_attn_base = f'encoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{attn_idx}'
        
        mapping[f'{xt_attn_base}.0.2.ln.weight'] = f'{mammoth_attn_base}.0.2.ln.weight'
        mapping[f'{xt_attn_base}.0.2.ln.bias'] = f'{mammoth_attn_base}.0.2.ln.bias'
        mapping[f'{xt_attn_base}.1.to_q.weight'] = f'{mammoth_attn_base}.1.to_q.weight'
        mapping[f'{xt_attn_base}.1.to_q.bias'] = f'{mammoth_attn_base}.1.to_q.bias'
        mapping[f'{xt_attn_base}.1.to_k.weight'] = f'{mammoth_attn_base}.1.to_k.weight'
        mapping[f'{xt_attn_base}.1.to_k.bias'] = f'{mammoth_attn_base}.1.to_k.bias'
        mapping[f'{xt_attn_base}.1.to_v.weight'] = f'{mammoth_attn_base}.1.to_v.weight'
        mapping[f'{xt_attn_base}.1.to_v.bias'] = f'{mammoth_attn_base}.1.to_v.bias'
        mapping[f'{xt_attn_base}.1.to_out.weight'] = f'{mammoth_attn_base}.1.to_out.weight'
        mapping[f'{xt_attn_base}.1.to_out.bias'] = f'{mammoth_attn_base}.1.to_out.bias'
        
        # Feedforward layer
        ff_idx = layer_idx * 2 + 1
        xt_ff_base = f'encoder.attn_layers.layers.{ff_idx}'
        mammoth_ff_base = f'encoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{ff_idx}'
        
        mapping[f'{xt_ff_base}.0.2.ln.weight'] = f'{mammoth_ff_base}.0.2.ln.weight'
        mapping[f'{xt_ff_base}.0.2.ln.bias'] = f'{mammoth_ff_base}.0.2.ln.bias'
        mapping[f'{xt_ff_base}.1.ff.0.weight'] = f'{mammoth_ff_base}.1.ff.0.weight'
        mapping[f'{xt_ff_base}.1.ff.0.bias'] = f'{mammoth_ff_base}.1.ff.0.bias'
        mapping[f'{xt_ff_base}.1.ff.3.weight'] = f'{mammoth_ff_base}.1.ff.3.weight'
        mapping[f'{xt_ff_base}.1.ff.3.bias'] = f'{mammoth_ff_base}.1.ff.3.bias'
    
    # Decoder layers
    for layer_idx in range(num_decoder_layers):
        base_idx = layer_idx * 3
        
        # Self-attention
        self_attn_idx = base_idx
        xt_self_attn_base = f'decoder.net.attn_layers.layers.{self_attn_idx}'
        mammoth_self_attn_base = f'decoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{self_attn_idx}'

        mapping[f'{xt_self_attn_base}.0.2.ln.weight'] = f'{mammoth_self_attn_base}.0.2.ln.weight'
        mapping[f'{xt_self_attn_base}.0.2.ln.bias'] = f'{mammoth_self_attn_base}.0.2.ln.bias'
        mapping[f'{xt_self_attn_base}.1.to_q.weight'] = f'{mammoth_self_attn_base}.1.to_q.weight'
        mapping[f'{xt_self_attn_base}.1.to_q.bias'] = f'{mammoth_self_attn_base}.1.to_q.bias'
        mapping[f'{xt_self_attn_base}.1.to_k.weight'] = f'{mammoth_self_attn_base}.1.to_k.weight'
        mapping[f'{xt_self_attn_base}.1.to_k.bias'] = f'{mammoth_self_attn_base}.1.to_k.bias'
        mapping[f'{xt_self_attn_base}.1.to_v.weight'] = f'{mammoth_self_attn_base}.1.to_v.weight'
        mapping[f'{xt_self_attn_base}.1.to_v.bias'] = f'{mammoth_self_attn_base}.1.to_v.bias'
        mapping[f'{xt_self_attn_base}.1.to_out.weight'] = f'{mammoth_self_attn_base}.1.to_out.weight'
        mapping[f'{xt_self_attn_base}.1.to_out.bias'] = f'{mammoth_self_attn_base}.1.to_out.bias'
        
        # Cross-attention
        cross_attn_idx = base_idx + 1
        xt_cross_attn_base = f'decoder.net.attn_layers.layers.{cross_attn_idx}'
        mammoth_cross_attn_base = f'decoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{cross_attn_idx}'

        mapping[f'{xt_cross_attn_base}.0.2.ln.weight'] = f'{mammoth_cross_attn_base}.0.2.ln.weight'
        mapping[f'{xt_cross_attn_base}.0.2.ln.bias'] = f'{mammoth_cross_attn_base}.0.2.ln.bias'
        mapping[f'{xt_cross_attn_base}.1.to_q.weight'] = f'{mammoth_cross_attn_base}.1.to_q.weight'
        mapping[f'{xt_cross_attn_base}.1.to_q.bias'] = f'{mammoth_cross_attn_base}.1.to_q.bias'
        mapping[f'{xt_cross_attn_base}.1.to_k.weight'] = f'{mammoth_cross_attn_base}.1.to_k.weight'
        mapping[f'{xt_cross_attn_base}.1.to_k.bias'] = f'{mammoth_cross_attn_base}.1.to_k.bias'
        mapping[f'{xt_cross_attn_base}.1.to_v.weight'] = f'{mammoth_cross_attn_base}.1.to_v.weight'
        mapping[f'{xt_cross_attn_base}.1.to_v.bias'] = f'{mammoth_cross_attn_base}.1.to_v.bias'
        mapping[f'{xt_cross_attn_base}.1.to_out.weight'] = f'{mammoth_cross_attn_base}.1.to_out.weight'
        mapping[f'{xt_cross_attn_base}.1.to_out.bias'] = f'{mammoth_cross_attn_base}.1.to_out.bias'
        
        # Feedforward
        ff_idx = base_idx + 2
        xt_ff_base = f'decoder.net.attn_layers.layers.{ff_idx}'
        mammoth_ff_base = f'decoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{ff_idx}'

        mapping[f'{xt_ff_base}.0.2.ln.weight'] = f'{mammoth_ff_base}.0.2.ln.weight'
        mapping[f'{xt_ff_base}.0.2.ln.bias'] = f'{mammoth_ff_base}.0.2.ln.bias'
        mapping[f'{xt_ff_base}.1.ff.0.weight'] = f'{mammoth_ff_base}.1.ff.0.weight'
        mapping[f'{xt_ff_base}.1.ff.0.bias'] = f'{mammoth_ff_base}.1.ff.0.bias'
        mapping[f'{xt_ff_base}.1.ff.3.weight'] = f'{mammoth_ff_base}.1.ff.3.weight'
        mapping[f'{xt_ff_base}.1.ff.3.bias'] = f'{mammoth_ff_base}.1.ff.3.bias'
    
    # Output layers
    mapping['decoder.net.to_logits.weight'] = 'decoder.bart_translation.to_logits.weight'
    mapping['decoder.net.to_logits.bias'] = 'decoder.bart_translation.to_logits.bias'
    
    return mapping


def map_xt_to_mammoth_weights(xt_model, mammoth_model, num_encoder_layers, num_decoder_layers):
    """Map weights from xt_model to mammoth_model"""
    mapping = create_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers)
    
    xt_sd = xt_model.state_dict()
    mammoth_sd = mammoth_model.state_dict()
    
    n_copied = 0
    n_missed = 0
    missed_keys = [] # Initialize a list to store missed keys
    
    for xt_key, mammoth_key in mapping.items():
        if xt_key not in xt_sd or mammoth_key not in mammoth_sd:
            n_missed += 1
            missed_keys.append(f"Missing: XT key '{xt_key}' or Mammoth key '{mammoth_key}'") # Add to missed_keys
            continue

        xt_val = xt_sd[xt_key]
        mammoth_val = mammoth_sd[mammoth_key]

        if xt_val.shape != mammoth_val.shape:
            n_missed += 1
            missed_keys.append(f"Shape mismatch: XT key '{xt_key}' ({xt_val.shape}) vs Mammoth key '{mammoth_key}' ({mammoth_val.shape})") # Add to missed_keys
            continue

        mammoth_sd[mammoth_key] = xt_val.clone()
        n_copied += 1

    mammoth_model.load_state_dict(mammoth_sd, strict=False)
    print(f"✓ Weight mapping: {n_copied} copied, {n_missed} missed")
    if missed_keys: # Print missed keys if any
        print("Missed keys:")
        for key in missed_keys:
            print(f"  - {key}")


def create_mammoth_model(xt_model, hf_model_path, src_lang="en", tgt_lang="es", corpus_id="bart_translation"):
    """Create Mammoth model from x-transformer model"""
    print("Creating Mammoth model...")
    
    # Create vocabularies and configuration
    vocabs_dict = create_vocabs_dict_from_hf_tokenizer(hf_model_path, src_lang, tgt_lang)
    model_opts = create_model_opts_from_xt_model(xt_model, hf_model_path)
    task_queue_manager = create_task_queue_manager(src_lang, tgt_lang, corpus_id, vocabs_dict)
    opts = create_opts()
    
    # Build Mammoth model
    mammoth_model = build_model(
        model_opts=model_opts,
        opts=opts,
        vocabs_dict=vocabs_dict,
        task_queue_manager=task_queue_manager,
        single_task=corpus_id
    )
    
    # Map weights from x-transformers to Mammoth
    config = AutoConfig.from_pretrained(hf_model_path)
    map_xt_to_mammoth_weights(xt_model, mammoth_model, config.encoder_layers, config.decoder_layers)
    
    # Create optimizer
    optimizer = MultipleOptimizer.from_opts(
        model=mammoth_model,
        opts=opts,
        task_queue_manager=task_queue_manager,
        frame_checkpoint=None
    )
    
    print("✓ Mammoth model created")
    return mammoth_model, model_opts, vocabs_dict, task_queue_manager, optimizer

# =============================================================================
# SECTION 4: Main conversion pipeline
# =============================================================================

def convert_hf_to_mammoth(hf_model_path, save_path, src_lang="en", tgt_lang="es"):
    """
    Complete pipeline to convert HuggingFace BART to Mammoth model
    
    Args:
        hf_model_path: Path to HuggingFace BART model
        save_path: Path to save the converted Mammoth model
        src_lang: Source language code
        tgt_lang: Target language code
    """
    print(f"Converting HF BART model: {hf_model_path}")
    print(f"Target save path: {save_path}")
    print(f"Language pair: {src_lang} → {tgt_lang}")
    print("="*50)
    
    # Stage 1: HuggingFace → x-transformers
    print("Stage 1: HuggingFace → x-transformers")
    xt_model = create_xtransformer_model(hf_model_path)
    xt_model = load_hf_weights_to_xtransformer(hf_model_path, xt_model)

    # Save x-transformer model keys to a file
    with open("xt_model_keys.txt", "w") as f:
        for key in xt_model.state_dict().keys():
            f.write(key + "\n")
    print("✓ x-transformer model keys saved to xt_model_keys.txt")
    
    # Stage 2: x-transformers → Mammoth
    print("Stage 2: x-transformers → Mammoth")
    corpus_id = f"bart_translation"
    mammoth_model, model_opts, vocabs_dict, task_queue_manager, optimizer = create_mammoth_model(
        xt_model, hf_model_path, src_lang, tgt_lang, corpus_id
    )
    # Save Mammoth model keys to a file
    with open("mammoth_model_keys.txt", "w") as f:
        for key in mammoth_model.state_dict().keys():
            f.write(key + "\n")
    print("✓ Mammoth model keys saved to mammoth_model_keys.txt")

    
    # Stage 3: Save Mammoth model
    print("Stage 3: Saving Mammoth model")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_opts = Namespace()
    save_opts.save_model = save_path
    save_opts.keep_checkpoint = -1
    
    model_saver = build_model_saver(
        model_opts=model_opts,
        opts=save_opts,
        model=mammoth_model,
        vocabs_dict=vocabs_dict,
        optim=optimizer,
        task_queue_manager=task_queue_manager
    )
    
    model_saver.save(step=0, data_state={})
    
    print("="*50)
    print(f"✓ Conversion complete! Mammoth model saved to: {save_path}")
    return mammoth_model


# =============================================================================
# SECTION 5: Command line interface
# =============================================================================

def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace BART model to Mammoth format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python converter.py /path/to/hf/bart/model ./saved_models/mammoth_bart
  python converter.py facebook/bart-base ./models/bart_base --src-lang en --tgt-lang fr
        """
    )
    
    parser.add_argument(
        "hf_model_path",
        help="Path to HuggingFace BART model (local path or model name)"
    )
    
    parser.add_argument(
        "save_path",
        help="Path to save the converted Mammoth model"
    )
    
    parser.add_argument(
        "--src-lang",
        default="en",
        help="Source language code (default: en)"
    )
    
    parser.add_argument(
        "--tgt-lang", 
        default="es",
        help="Target language code (default: es)"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.hf_model_path) and "/" not in args.hf_model_path:
        print(f"Warning: {args.hf_model_path} not found locally, trying as HuggingFace model name...")
    
    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        print(f"Creating save directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Run conversion
        convert_hf_to_mammoth(
            hf_model_path=args.hf_model_path,
            save_path=args.save_path,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())