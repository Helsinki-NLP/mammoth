"""Smoke tests for KV cache wiring in modeling_mammoth.py."""

import torch

from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration
from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.modules.transformer.cache import KVCache


def _tiny_model(dec_layers=(4,)) -> MammothForConditionalGeneration:
    cfg = MammothConfig(
        model_dim=32,
        heads=4,
        ff_mult=2.0,
        enc_layers=[2],
        dec_layers=list(dec_layers),
        src_vocab_size=200,
        tgt_vocab_size=200,
        rotary_pos_emb=True,
        post_emb_norm=True,
        use_cache=True,
    )
    return MammothForConditionalGeneration(cfg).eval()


@torch.no_grad()
def test_cache_populated_after_forward():
    """past_key_values returned by forward() is a KVCache with filled tensors."""
    model = _tiny_model()
    src = torch.randint(3, 200, (2, 6))
    dec = torch.full((2, 1), fill_value=2, dtype=torch.long)
    out = model(input_ids=src, decoder_input_ids=dec, use_cache=True)

    cache = out.past_key_values
    assert isinstance(cache, KVCache), "past_key_values should be a KVCache"
    assert cache.layers[0].self_k is not None, "self-attn cache not populated"
    assert cache.layers[0].cross_k is not None, "cross-attn cache not populated"


@torch.no_grad()
def test_step_by_step_matches_full_sequence():
    """Cached step-by-step decoding produces identical logits to uncached full-sequence decoding."""
    torch.manual_seed(0)
    model = _tiny_model()

    batch, src_len, dec_len = 2, 8, 5
    src = torch.randint(3, 200, (batch, src_len))
    enc_mask = torch.ones(batch, src_len, dtype=torch.long)
    dec_ids = torch.randint(3, 200, (batch, dec_len))

    enc_out = model.encoder(src, attention_mask=enc_mask)
    enc_h = enc_out.last_hidden_state
    ctx_mask = enc_mask[:, None, None, :].bool()

    # Reference: full sequence, no cache
    logits_full = model.decoder(dec_ids, context=enc_h, context_mask=ctx_mask)

    # Cached: one token at a time
    num_layers = sum(s.depth for s in model.decoder.stacks)
    cache = KVCache(num_layers)
    cached_logits = []
    for t in range(dec_len):
        logits_t = model.decoder(dec_ids[:, t:t + 1], context=enc_h,
                                 context_mask=ctx_mask, cache=cache)
        cached_logits.append(logits_t)
    logits_cached = torch.cat(cached_logits, dim=1)

    assert torch.allclose(logits_cached, logits_full, atol=1e-5), \
        f"max diff: {(logits_cached - logits_full).abs().max().item():.2e}"


@torch.no_grad()
def test_generate_greedy_uses_cache():
    """Greedy generate() with use_cache=True runs without error and cache stays consistent."""
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randint(3, 200, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.long)

    ids = model.generate(src, attention_mask=mask, max_new_tokens=10, use_cache=True)
    assert ids.shape[1] > 1, "should generate at least one token"


@torch.no_grad()
def test_generate_cached_equals_uncached():
    """Greedy generate() with and without cache produces identical token sequences."""
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randint(3, 200, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.long)

    ids_cached = model.generate(src, attention_mask=mask, max_new_tokens=10, use_cache=True)
    ids_uncached = model.generate(src, attention_mask=mask, max_new_tokens=10, use_cache=False)
    assert torch.equal(ids_cached, ids_uncached), \
        f"cached={ids_cached.tolist()} uncached={ids_uncached.tolist()}"


@torch.no_grad()
def test_generate_beam_cached_equals_uncached():
    """Beam search with and without cache produces identical token sequences."""
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randint(3, 200, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.long)

    ids_cached = model.generate(src, attention_mask=mask, max_new_tokens=8,
                                num_beams=4, use_cache=True)
    ids_uncached = model.generate(src, attention_mask=mask, max_new_tokens=8,
                                  num_beams=4, use_cache=False)
    assert torch.equal(ids_cached, ids_uncached), \
        f"cached={ids_cached.tolist()} uncached={ids_uncached.tolist()}"


@torch.no_grad()
def test_multi_stack_decoder_cache():
    """Multi-stack decoder (2 stacks of 2) uses correct layer offsets per stack."""
    torch.manual_seed(0)
    model = _tiny_model(dec_layers=(2, 2))

    batch, src_len, dec_len = 1, 6, 4
    src = torch.randint(3, 200, (batch, src_len))
    enc_mask = torch.ones(batch, src_len, dtype=torch.long)
    dec_ids = torch.randint(3, 200, (batch, dec_len))

    enc_out = model.encoder(src, attention_mask=enc_mask)
    enc_h = enc_out.last_hidden_state
    ctx_mask = enc_mask[:, None, None, :].bool()

    logits_full = model.decoder(dec_ids, context=enc_h, context_mask=ctx_mask)

    num_layers = sum(s.depth for s in model.decoder.stacks)
    assert num_layers == 4
    cache = KVCache(num_layers)
    cached_logits = []
    for t in range(dec_len):
        logits_t = model.decoder(dec_ids[:, t:t + 1], context=enc_h,
                                 context_mask=ctx_mask, cache=cache)
        cached_logits.append(logits_t)
    logits_cached = torch.cat(cached_logits, dim=1)

    assert torch.allclose(logits_cached, logits_full, atol=1e-5), \
        f"max diff: {(logits_cached - logits_full).abs().max().item():.2e}"
