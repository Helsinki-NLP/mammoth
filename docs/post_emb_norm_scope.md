# Per-language post-embedding norm

Branch: `per-language-post-emb-norm`. Tests: `mammoth/tests/test_post_emb_norm_scope.py`.

## Why

The goal is to add a language to a trained model by training **only that language's embedding**, for example a Danish source embedding that feeds into an existing North Germanic encoder already used by Swedish and Norwegian. The frozen encoder forces the Danish embedding into the same representation space, and Danish→X zero-shot translation can then be tested.

Think of the trained model as an airport. The shared encoder and decoder layers are the terminal. Each language embedding is a gate into it. Adding Danish means building one new gate without renovating the terminal.

Previously one piece of the route was shared: the **post-embedding RMSNorm** (`token_emb → post_emb_norm → stacks`). It was owned by the encoder/decoder *path* (component), so all languages on a path went through the same norm. Training a Danish embedding gave that shared norm real gradients, so it drifted and changed Swedish and Norwegian too. None of the freeze flags could freeze it while keeping the Danish embedding trainable.

Now every language has its own norm, stored with its embedding. A new language arrives with its own embedding *and* its own norm, so plugging it in touches nothing shared.

## Two layouts, decided by the checkpoint

There is **no config option**. The layout is a fact about how a model was built, and it is recorded in the checkpoint frame (`frame["opts"].post_emb_norm_scope`).

| | `language` | `component` |
|---|---|---|
| Used by | every new model | checkpoints that predate this change (no key in the frame) |
| Norms | one per language, per side | one per xcoder path (tuple of `xcoder_id`s), shared by its languages |
| Stored in | `{src,tgt}_embeddings_{lang}.pt` (`weight` + `post_emb_norm.weight`) | `{encoder,decoder}_wrapper_{path}.pt` (`post_emb_norm.weight`) |
| Wrapper file | no norm; the encoder wrapper is empty | `post_emb_norm.*` (+ `to_logits.*` on the decoder) |

Each task still has exactly one norm: a task has one source and one target language. `NativeTransformerWrapper.forward` is unchanged. Only the *owner* of the norm differs.

All of this applies only with `post_emb_norm: true`.

How the layout is chosen:

- **New training run:** `language`, written into the frame (`train_single.py::_get_model_opts`).
- **Translation / inference:** whatever the frame says. A frame without the key is `component` (`ArgumentParser.checkpoint_model_opts`).
- **`--train_from`:** whatever the frame says, so a run resumes exactly as it was built.
- **HF / LiteRT export:** reads the norm from the embedding shard when it is there, otherwise from the wrapper shard.

**Layouts are never converted.** Old (component) models are kept for inference and load exactly as before. New experiments, such as adding a language, start from a model trained with per-language norms.

## Exporters

- HF flat (`convert_mammoth_to_hf.py`) and modular (`--modular`, `mammoth_hub.py`): an embedding component may also carry `post_emb_norm.*`, which maps to `{encoder,decoder}.post_emb_norm.*`. With per-language norms the empty encoder wrapper is exported as an empty component. Old HF repos (embedding = `weight` only) load as before.
- LiteRT (`litert/convert_multi.py`): reads the norm from the embedding shard if present, otherwise from the wrapper shard.

## Recipe: add a new source language (Danish) to a trained model

This works for a model trained with per-language norms (anything trained on this branch or later).

1. Train an HF tokenizer for Danish and add it as the `dan` source vocab.
2. Add a task, for example `dan-eng`, whose `enc_sharing_group` is **identical** to the Swedish/Norwegian tasks' and whose `dec_sharing_group` reuses an existing decoder. No new layers may be created.
3. Train with:

   ```
   --train_from <ckpt_dir> --reset_optim all --yes_i_messed_with_the_checkpoint
   --freeze_encoder --freeze_decoder --freeze_decoder_embeddings
   ```

   Add `--freeze_attention_bridge` if the model has one.
4. Check the load log. Only `src_embeddings_dan.pt` and its `_optim.pt` should be reported missing. Any other missing file means the Danish sharing groups don't match the existing tasks. The Danish embedding and its norm start freshly initialized.
5. Evaluate zero-shot directions such as `dan→deu` with a translation config for a task that was never trained.

What stays fixed:

- The shared stacks, decoder and `to_logits` are frozen by the flags. No shared norm exists any more.
- The Swedish/Norwegian embeddings and norms remain trainable in principle, but with Danish-only data they get zero gradient. They stay unchanged as long as **`weight_decay` is 0** (the default). AdamW decays every parameter it steps, even with a zero gradient, and multi-GPU sync fills missing gradients with zeros.
- Cheap verification after the run: compare `*_src_embeddings_swe.pt` and `*_src_embeddings_nob.pt` against the original checkpoint with `torch.equal`.

## Limits and notes

- **Target languages:** a new *target* language is a different problem. `to_logits` is per decoder path and sized to that path's vocabulary, so a new target language with its own vocabulary needs a new decoder path.
- **Tied embeddings** (`share_encoder_decoder_embeddings`): the embedding weight is shared, but each side gets its own norm.
- **Extra parameters:** per-language norms add `model_dim` parameters per language and side, which is negligible. It is a (tiny) modelling change: languages on the same path can scale their inputs differently.
- **Rotary embeddings, `to_logits` and the stacks** remain per component, as before.

## Code map

| File | Change |
|---|---|
| `model_builder.py` | builds one `nn.RMSNorm` per language unless the frame says `component` (the component slot is then `Identity`); `_freeze_embeddings` also freezes per-language norms |
| `modules/layer_stack.py` | `StackXcoder.shared_lang_post_emb_norms`, `get_post_emb_norm_by_lang()` |
| `distributed/components.py` | `DistributedEmbedding` exposes `_EmbeddingWithNorm` (keys `weight`, `post_emb_norm.weight`), so the norm is saved, loaded, optimized and gradient-synced with the embedding |
| `train_single.py` | new runs record `post_emb_norm_scope = 'language'` in the frame |
| `utils/parse.py` | `checkpoint_model_opts`: a frame without the key → `component` |
| `hf_integration/to_hf/{mammoth_hub,convert_mammoth_to_hf}.py`, `litert/convert_multi.py` | read the norm from the embedding shard when present |
