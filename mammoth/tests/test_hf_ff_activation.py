"""The HF export must rebuild the FFN exactly as training did.

Regression: config_from_opts used to read a non-existent ``ff_swiglu`` opt (always
True) and fall back to ff_mult=2.67, so a gelu/geglu model was exported as swiglu.
With an explicit ff_mult the shapes matched, the converter loaded non-strictly and
the extra swiglu ``w3`` kept its random init -> a silently broken model.
"""

import argparse

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from mammoth.hf_integration.to_hf import native_transformer as vendored
from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import config_from_opts, convert
from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration
from mammoth.modules.transformer import DecoderBlock, EncoderBlock, TransformerStack
from mammoth.modules.transformer.ffn import FeedForward
from mammoth.tests.test_hf_modular import _Vocab

PREFIX = "model_best"
DIM, HEADS, SRC_V, TGT_V = 32, 4, 40, 50


def _opts(**ff):
    return argparse.Namespace(
        model_dim=DIM, heads=HEADS, enc_layers=[1], dec_layers=[2],
        attn_dropout=0.0, ff_dropout=0.0, dropout=[0.0], rotary_pos_emb=True, post_emb_norm=True,
        tasks={"train_eng-spa": {"src_tgt": "eng-spa", "enc_sharing_group": ["all"],
                                 "dec_sharing_group": ["all"]}},
        **ff,
    )


@pytest.mark.parametrize("ff, expected", [
    ({"ff_activation": "gelu"}, (4.0, "gelu", 4.0, "gelu")),
    ({"ff_activation": "gelu", "ff_mult": 3.0}, (3.0, "gelu", 3.0, "gelu")),
    ({}, (2.67, "swiglu", 2.67, "swiglu")),
    ({"ff_activation": "swiglu", "dec_ff_activation": "geglu", "dec_ff_mult": 2.0},
     (2.67, "swiglu", 2.0, "geglu")),
])
def test_config_from_opts_ff(ff, expected):
    cfg = config_from_opts(_opts(**ff), SRC_V, TGT_V)
    assert (cfg.enc_ff_mult, cfg.enc_ff_activation, cfg.dec_ff_mult, cfg.dec_ff_activation) == expected


def test_legacy_config_fields():
    """config.json files written before per-side FFN fields still load."""
    cfg = MammothConfig(ff_mult=3.0, ff_swiglu=False)
    assert (cfg.enc_ff_mult, cfg.enc_ff_activation, cfg.dec_ff_mult, cfg.dec_ff_activation) == \
        (3.0, "gelu", 3.0, "gelu")
    reloaded = MammothConfig.from_dict(MammothConfig(dec_ff_activation="geglu").to_dict())
    assert (reloaded.enc_ff_activation, reloaded.dec_ff_activation) == ("swiglu", "geglu")


@pytest.mark.parametrize("activation", ["swiglu", "gelu", "geglu"])
def test_vendored_ffn_matches_native(activation):
    """native_transformer.py ships in the artifact; its FFN must equal the training one."""
    torch.manual_seed(0)
    native = FeedForward(DIM, 2.5, 0.0, activation)
    copy = vendored.FeedForward(DIM, 2.5, 0.0, activation)
    copy.load_state_dict(native.state_dict())
    x = torch.randn(2, 3, DIM)
    assert torch.equal(native(x), copy(x))


def _write_checkpoint(ckpt, opts, enc_ff, dec_ff):
    """Sharded checkpoint whose stacks are native Mammoth modules with the given FFNs."""
    torch.manual_seed(0)
    enc = TransformerStack(nn.ModuleList(
        [EncoderBlock(dim=DIM, heads=HEADS, ff_mult=enc_ff[0], activation=enc_ff[1])]),
        nn.RMSNorm(DIM), DIM, 0, "all")
    dec = TransformerStack(nn.ModuleList(
        [DecoderBlock(dim=DIM, heads=HEADS, ff_mult=dec_ff[0], activation=dec_ff[1]) for _ in range(2)]),
        nn.RMSNorm(DIM), DIM, 0, "all")
    shards = {
        "src_embeddings_eng": nn.Embedding(SRC_V, DIM),
        "tgt_embeddings_spa": nn.Embedding(TGT_V, DIM),
        "encoder_0_all": enc,
        "decoder_0_all": dec,
        "encoder_wrapper_all": nn.ModuleDict({"post_emb_norm": nn.RMSNorm(DIM)}),
        "decoder_wrapper_all": nn.ModuleDict({"post_emb_norm": nn.RMSNorm(DIM),
                                              "to_logits": nn.Linear(DIM, TGT_V, bias=False)}),
    }
    vocab = {("src", "eng"): _Vocab(SRC_V, "eng"), ("tgt", "spa"): _Vocab(TGT_V, "spa")}
    torch.save({"opts": opts, "vocab": vocab}, ckpt / f"{PREFIX}_frame.pt")
    for name, module in shards.items():
        with torch.no_grad():
            for p in module.parameters():
                p.normal_(0, 0.2)
        torch.save(module.state_dict(), ckpt / f"{PREFIX}_{name}.pt")
    return shards


@torch.no_grad()
def test_gelu_geglu_checkpoint_exact_match(tmp_path):
    """gelu encoder (ff_mult unset -> 4.0) + geglu decoder (dec_ff_mult) round-trips exactly."""
    opts = _opts(ff_activation="gelu", dec_ff_activation="geglu", dec_ff_mult=2.0)
    ckpt, out = tmp_path / "ckpt", tmp_path / "hf"
    ckpt.mkdir()
    shards = _write_checkpoint(ckpt, opts, enc_ff=(4.0, "gelu"), dec_ff=(2.0, "geglu"))
    convert(str(ckpt), PREFIX, "eng", "spa", str(out))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(out), trust_remote_code=True).eval()

    ref = MammothForConditionalGeneration(config_from_opts(opts, SRC_V, TGT_V)).eval()
    ref.encoder.token_emb.load_state_dict(shards["src_embeddings_eng"].state_dict())
    ref.decoder.token_emb.load_state_dict(shards["tgt_embeddings_spa"].state_dict())
    ref.encoder.stacks[0].load_state_dict(shards["encoder_0_all"].state_dict())
    ref.decoder.stacks[0].load_state_dict(shards["decoder_0_all"].state_dict())
    ref.encoder.post_emb_norm.load_state_dict(shards["encoder_wrapper_all"]["post_emb_norm"].state_dict())
    ref.decoder.post_emb_norm.load_state_dict(shards["decoder_wrapper_all"]["post_emb_norm"].state_dict())
    ref.decoder.to_logits.load_state_dict(shards["decoder_wrapper_all"]["to_logits"].state_dict())
    # The reference must really be built from the native modules being compared against.
    assert ref.encoder.stacks[0].blocks[0].ff.w3 is None
    assert ref.decoder.stacks[0].blocks[0].ff.activation == "geglu"

    g = torch.Generator().manual_seed(1)
    src_ids = torch.randint(4, SRC_V, (2, 7), generator=g)
    dec_ids = torch.randint(4, TGT_V, (2, 5), generator=g)
    kwargs = dict(input_ids=src_ids, attention_mask=torch.ones_like(src_ids),
                  decoder_input_ids=dec_ids, use_cache=False)
    assert torch.equal(model(**kwargs).logits, ref(**kwargs).logits)


def test_mismatched_checkpoint_is_rejected(tmp_path):
    """Opts that disagree with the stored weights must fail loudly, not load non-strictly."""
    # Frame claims swiglu x3.0, but the shards are gelu x3.0: shapes match, w3 is missing.
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_checkpoint(ckpt, _opts(ff_activation="swiglu", ff_mult=3.0),
                      enc_ff=(3.0, "gelu"), dec_ff=(3.0, "gelu"))
    with pytest.raises(RuntimeError, match="missing"):
        convert(str(ckpt), PREFIX, "eng", "spa", str(tmp_path / "hf"))
