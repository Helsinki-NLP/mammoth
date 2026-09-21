#!/usr/bin/env python3
"""
Convert HuggingFace Gemma3 (text-only, e.g. gemma3_270m) into a Mammoth
model on the NATIVE PyTorch backend (mammoth/mammoth/modules/transformer/).

See CLAUDE.md "Gemma3-270M -> Mammoth Conversion" for the full architecture
gap analysis and the "fake encoder" design decision implemented here: Gemma3
is decoder-only, but Mammoth's build_model()/task plumbing assumes every
task has an encoder. We build a small randomly-initialized encoder whose
cross-attention contribution into the decoder is zero-initialized
(cross_attn.to_out.weight = 0), so the converted Mammoth model's forward
pass is numerically IDENTICAL to running Gemma3 alone -- the fake encoder
only exists to satisfy the NMT-shaped task plumbing. See
mammoth/tests/test_gemma3_native_conversion.py for the equivalence check
against the real HF model.

Every decoder architecture knob below is borrowed directly from
`transformers.models.gemma3.configuration_gemma3.Gemma3TextConfig` and
`modeling_gemma3.py` (Gemma3RMSNorm, Gemma3Attention, Gemma3MLP,
Gemma3DecoderLayer) rather than re-derived, per project convention.

Unlike the earlier x_transformers-backend converter
(hf_integration/from_hf/gemma3/gemma2mammoth.py), which mapped weights via
fragile string key tables, this script copies weights by walking the actual
built nn.Module objects on both sides -- there is no string-based key
mapping to get subtly wrong.

Usage:
    python convert_gemma3_native.py <hf_model_path> <save_path>
"""
import argparse
import json
import os
import sys
import tempfile
from argparse import Namespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from mammoth.distributed.contexts import DeviceContextEnum, WorldContext  # noqa: E402
from mammoth.distributed.tasks import (  # noqa: E402
    TaskQueueManager,
    TaskSpecs,
    WeightedSamplingTaskDistributionStrategy,
)
from mammoth.inputters.vocab import HFTokenizerVocab  # noqa: E402
from mammoth.model_builder import build_model  # noqa: E402
from mammoth.utils.model_saver import build_model_saver  # noqa: E402
from mammoth.utils.optimizers import MultipleOptimizer  # noqa: E402


DEFAULT_LANG = "gemma3"
DEFAULT_ENCODER_GROUP = "fake_enc"
DEFAULT_DECODER_GROUP = "gemma3_dec"


def get_text_config(hf_model_path):
    config = AutoConfig.from_pretrained(hf_model_path, local_files_only=True)
    return config.text_config if hasattr(config, "text_config") else config


def build_model_opts(config, enc_layers=1, enc_model_dim=64):
    """
    Mammoth model_opts matching Gemma3's decoder exactly, plus a tiny fake
    encoder (see module docstring). `enc_model_dim` can be much smaller than
    the decoder's hidden size: DecoderBlock's `context_dim` (defaulted here
    from `enc_model_dim` by model_builder.py) gives cross-attn's K/V their
    own projection from the encoder's actual output size, so the two no
    longer need to match.
    """
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    opts = Namespace()
    opts.seed = 1
    opts.model_dtype = "bf16"
    opts.log_model_structure = False
    # Always weighted_sampling (see build_task_queue_manager) -- recorded
    # here too so the opts Namespace we save alongside the checkpoint is
    # self-consistent with the strategy actually used.
    opts.task_distribution_strategy = "weighted_sampling"

    opts.dec_layers = [config.num_hidden_layers]
    opts.dec_model_dim = config.hidden_size
    opts.dec_heads = config.num_attention_heads
    opts.dec_attn_dim_head = head_dim
    opts.dec_attn_kv_heads = config.num_key_value_heads
    opts.dec_attn_dropout = config.attention_dropout
    opts.dec_attn_qk_norm = True
    opts.dec_norm_eps = config.rms_norm_eps
    opts.dec_attn_scale = config.query_pre_attn_scalar ** -0.5
    opts.dec_sandwich_norm = True
    opts.dec_ff_activation = "geglu"
    opts.dec_ff_mult = config.intermediate_size / config.hidden_size
    opts.dec_ff_dropout = 0.0
    opts.dec_scaled_embeddings = True
    opts.dec_sliding_window = config.sliding_window
    opts.dec_global_attn_every_n_layers = getattr(config, "_sliding_window_pattern", 6)
    opts.dec_global_rope_theta = config.rope_theta
    opts.dec_local_rope_theta = config.rope_local_base_freq

    # Fake encoder: small dim, tiny depth, plain swiglu block. Its cross-attn
    # contribution is zeroed at weight-copy time (see copy_gemma3_weights)
    # so it is a mathematical no-op regardless of its size.
    opts.enc_layers = [enc_layers]
    opts.enc_model_dim = enc_model_dim
    opts.enc_heads = 8
    opts.enc_ff_mult = 4.0

    opts.rotary_pos_emb = True
    opts.post_emb_norm = False

    opts.param_init = 0.0
    opts.param_init_glorot = True
    opts.attention_bridge = None
    opts.ab_layers = []
    opts.adapters = None
    opts.enable_embeddingless = False
    opts.dropout = [0.0]
    opts.attention_dropout = [config.attention_dropout]

    # Only read by MultipleOptimizer.from_opts when actually saving/training;
    # the converted model is not intended to be trained from this script.
    opts.train_from = None
    opts.reset_optim = "all"
    opts.optim = "adafactor"
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
    return opts


def prepare_text_only_tokenizer(hf_model_path, vocab_size):
    """
    Gemma3's tokenizer.json (this is the multimodal Gemma3 tokenizer, shared
    across text/vision variants) includes a `<image_soft_token>` added token
    at id `vocab_size` (262144 for gemma3_270m), one past the text model's
    embedding matrix (262144 rows, ids 0..262143). Strip any vocab/added
    tokens at or beyond `vocab_size` so HFTokenizerVocab's length matches the
    actual embedding size, and write the result to a temp file.
    """
    src_path = os.path.join(hf_model_path, "tokenizer.json")
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["model"]["vocab"]
    for token in [tok for tok, idx in vocab.items() if idx >= vocab_size]:
        del vocab[token]
    if "added_tokens" in data:
        data["added_tokens"] = [t for t in data["added_tokens"] if t["id"] < vocab_size]

    tmp_dir = tempfile.mkdtemp(prefix="mammoth_gemma3_tokenizer_")
    tmp_path = os.path.join(tmp_dir, "tokenizer.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return tmp_path


def build_vocabs_dict(tokenizer_path, src_lang, tgt_lang):
    src_vocab = HFTokenizerVocab(tokenizer_path=tokenizer_path, tag=src_lang)
    tgt_vocab = src_vocab if tgt_lang == src_lang else HFTokenizerVocab(tokenizer_path=tokenizer_path, tag=tgt_lang)
    return {("src", src_lang): src_vocab, ("tgt", tgt_lang): tgt_vocab}


def build_task_queue_manager(opts, vocabs_dict, task_id, encoder_group, decoder_group, src_lang, tgt_lang, weight=1.0):
    task = TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        encoder_id=[encoder_group],
        decoder_id=[decoder_group],
        corpus_id=task_id,
        weight=weight,
        introduce_at_training_step=0,
        corpus_opts=dict(),
        src_vocab=vocabs_dict[("src", src_lang)],
        tgt_vocab=vocabs_dict[("tgt", tgt_lang)],
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )
    world_context = WorldContext(DeviceContextEnum.SINGLE_GPU, n_nodes=1, gpus_per_node=1)
    # Mammoth's task-scheduling strategy is always weighted_sampling for
    # this converter, regardless of task count: it's the strategy actually
    # used in training, so the converted checkpoint's task registration
    # should reflect it rather than a placeholder like round-robin.
    tqm = TaskQueueManager(
        tasks=[task],
        accum_count=1,
        world_context=world_context,
        task_distribution_strategy_cls=WeightedSamplingTaskDistributionStrategy,
        uses_adapters=False,
    ).global_to_local(node_rank=0, local_rank=0, opts=opts)
    tqm.create_all_distributed_components(use_attention_bridge=False)
    return tqm


def copy_rmsnorm_unit_offset(mammoth_norm, hf_weight):
    """
    Gemma3RMSNorm computes `x_norm * (1 + weight)`; nn.RMSNorm computes
    `x_norm * weight`. Loading `hf_weight + 1.0` makes the two identical.
    """
    with torch.no_grad():
        mammoth_norm.weight.copy_(hf_weight.float() + 1.0)


def copy_linear(mammoth_linear, hf_linear):
    with torch.no_grad():
        mammoth_linear.weight.copy_(hf_linear.weight.float())


@torch.no_grad()
def copy_gemma3_weights(hf_model, mammoth_model, decoder_group, tgt_lang):
    hf_layers = hf_model.model.layers
    stack = mammoth_model.decoder.get_attention_layers_by_xcoder_id(0, decoder_group)
    assert len(stack.blocks) == len(hf_layers), (
        f"layer count mismatch: mammoth={len(stack.blocks)} hf={len(hf_layers)}"
    )

    for block, hf_layer in zip(stack.blocks, hf_layers):
        copy_rmsnorm_unit_offset(block.norm1, hf_layer.input_layernorm.weight)
        copy_rmsnorm_unit_offset(block.norm1_post, hf_layer.post_attention_layernorm.weight)
        copy_rmsnorm_unit_offset(block.norm3, hf_layer.pre_feedforward_layernorm.weight)
        copy_rmsnorm_unit_offset(block.norm3_post, hf_layer.post_feedforward_layernorm.weight)

        attn, hf_attn = block.self_attn, hf_layer.self_attn
        copy_linear(attn.to_q, hf_attn.q_proj)
        copy_linear(attn.to_k, hf_attn.k_proj)
        copy_linear(attn.to_v, hf_attn.v_proj)
        copy_linear(attn.to_out, hf_attn.o_proj)
        copy_rmsnorm_unit_offset(attn.q_norm, hf_attn.q_norm.weight)
        copy_rmsnorm_unit_offset(attn.k_norm, hf_attn.k_norm.weight)

        # Fake encoder's cross-attn output must contribute exactly zero so
        # the converted model reproduces Gemma3's decoder-only forward pass.
        block.cross_attn.to_out.weight.zero_()

        ff, hf_mlp = block.ff, hf_layer.mlp
        copy_linear(ff.w1, hf_mlp.gate_proj)  # gate
        copy_linear(ff.w3, hf_mlp.up_proj)    # up
        copy_linear(ff.w2, hf_mlp.down_proj)  # down

    copy_rmsnorm_unit_offset(stack.final_norm, hf_model.model.norm.weight)

    token_emb = mammoth_model.decoder.get_embedding_by_lang(tgt_lang)
    token_emb.weight.copy_(hf_model.model.embed_tokens.weight.float())

    to_logits = mammoth_model.decoder.get_to_logits_by_component((decoder_group,))
    to_logits.weight.copy_(hf_model.lm_head.weight.float())


def convert(
    hf_model_path,
    save_path=None,
    enc_layers=1,
    enc_model_dim=64,
    task_id=None,
    encoder_group=DEFAULT_ENCODER_GROUP,
    decoder_group=DEFAULT_DECODER_GROUP,
    src_lang=DEFAULT_LANG,
    tgt_lang=DEFAULT_LANG,
    weight=1.0,
):
    """
    task_id: Mammoth corpus/task id. Defaults to "<src_lang>-<tgt_lang>".
    encoder_group / decoder_group: enc_sharing_group / dec_sharing_group
        xcoder ids -- these are the keys other tasks would reference to
        share this encoder/decoder stack.
    src_lang / tgt_lang: language tags for the fake encoder's vocab/embedding
        and Gemma3's vocab/embedding respectively (tgt_lang is the
        meaningful one -- it owns the real converted weights).
    """
    if task_id is None:
        task_id = f"{src_lang}-{tgt_lang}"

    config = get_text_config(hf_model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path, torch_dtype=torch.bfloat16, local_files_only=True,
    ).eval()

    tokenizer_path = prepare_text_only_tokenizer(hf_model_path, config.vocab_size)
    vocabs_dict = build_vocabs_dict(tokenizer_path, src_lang, tgt_lang)

    model_opts = build_model_opts(config, enc_layers=enc_layers, enc_model_dim=enc_model_dim)
    tqm = build_task_queue_manager(
        model_opts, vocabs_dict, task_id, encoder_group, decoder_group, src_lang, tgt_lang, weight=weight,
    )

    mammoth_model = build_model(model_opts, model_opts, vocabs_dict, task_queue_manager=tqm, single_task=None)
    copy_gemma3_weights(hf_model, mammoth_model, decoder_group, tgt_lang)
    mammoth_model.eval()

    if save_path is not None:
        save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        optimizer = MultipleOptimizer.from_opts(
            model=mammoth_model, opts=model_opts, task_queue_manager=tqm, frame_checkpoint=None,
        )
        save_opts = Namespace(save_model=save_path, keep_checkpoint=-1)
        model_saver = build_model_saver(
            model_opts=model_opts, opts=save_opts, model=mammoth_model,
            vocabs_dict=vocabs_dict, optim=optimizer, task_queue_manager=tqm,
        )
        model_saver.save(step=0, data_state={})

    return mammoth_model, hf_model, tqm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_model_path")
    parser.add_argument("save_path")
    parser.add_argument("--enc-layers", type=int, default=1, help="Depth of the fake encoder.")
    parser.add_argument("--enc-model-dim", type=int, default=64, help="Hidden dim of the fake encoder.")
    parser.add_argument(
        "--task-id", default=None,
        help="Mammoth corpus/task id. Defaults to '<src-lang>-<tgt-lang>'.",
    )
    parser.add_argument(
        "--encoder-group", default=DEFAULT_ENCODER_GROUP,
        help="enc_sharing_group xcoder id for the fake encoder.",
    )
    parser.add_argument(
        "--decoder-group", default=DEFAULT_DECODER_GROUP,
        help="dec_sharing_group xcoder id hosting the converted Gemma3 decoder.",
    )
    parser.add_argument(
        "--src-lang", default=DEFAULT_LANG,
        help="Language tag for the fake encoder's vocab/embedding.",
    )
    parser.add_argument(
        "--tgt-lang", default=DEFAULT_LANG,
        help="Language tag for Gemma3's vocab/embedding (the meaningful side).",
    )
    parser.add_argument("--weight", type=float, default=1.0, help="Task weight for weighted_sampling.")
    args = parser.parse_args()
    convert(
        args.hf_model_path,
        args.save_path,
        enc_layers=args.enc_layers,
        enc_model_dim=args.enc_model_dim,
        task_id=args.task_id,
        encoder_group=args.encoder_group,
        decoder_group=args.decoder_group,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        weight=args.weight,
    )


if __name__ == "__main__":
    main()
