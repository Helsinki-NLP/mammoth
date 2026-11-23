# Extending a Trained Model with a New Language

Say you have a model trained on `swe-eng` and `nob-eng`, and you want to add Northern Sami (`sme-eng`) without changing what the model already knows. Think of the trained model as a finished building: you add a new room (the `sme` parts) and lock all the existing rooms (the trained parts), so the work on the new room cannot damage the rest.

In MAMMOTH the "rooms" are **components**. Each component is one checkpoint file: a language embedding, a layer stack, a wrapper (output projection), an adapter, or the attention bridge. You lock (freeze) or unlock components by name.

## Component names

| Component | Name |
|---|---|
| encoder / decoder layer stack | `{encoder,decoder}_{stack}_{xcoder_id}`, e.g. `encoder_0_nordic` |
| language embedding (with its post-embedding norm) | `{src,tgt}_embeddings_{lang}`, e.g. `src_embeddings_sme` |
| wrapper (`to_logits`; post-embedding norm in older checkpoints) | `{encoder,decoder}_wrapper_{xcoder_ids}`, e.g. `decoder_wrapper_eng` |
| adapter | `{encoder,decoder}_adapter_{stack}_{group}_{sub_id}` |
| attention bridge | `attention_bridge` |

The names are the checkpoint file names without the `model_step_N_` prefix. To list them with their sizes, run `python mammoth/scripts/count_params.py --checkpoint-dir <dir> --by component`.

## Example config

Add the new task next to the old ones, start from the trained checkpoint, and freeze everything the checkpoint contains:

```yaml
train_from: pretrained/model_step_100000_frame.pt
allow_partial_checkpoint: true   # sme has no files in the checkpoint: it starts from random init
reset_optim: all                 # new optimizer and learning rate schedule for the new phase

freeze_components: ['@pretrained']

src_vocab:
  swe: tokenizers/swe/tokenizer.json
  nob: tokenizers/nob/tokenizer.json
  sme: tokenizers/sme/tokenizer.json
tgt_vocab:
  eng: tokenizers/eng/tokenizer.json

tasks:
  swe-eng:
    src_tgt: swe-eng
    enc_sharing_group: [nordic]
    dec_sharing_group: [eng]
    path_src: data/swe-eng.swe
    path_tgt: data/swe-eng.eng
  nob-eng:
    src_tgt: nob-eng
    enc_sharing_group: [nordic]
    dec_sharing_group: [eng]
    path_src: data/nob-eng.nob
    path_tgt: data/nob-eng.eng
  sme-eng:
    src_tgt: sme-eng
    enc_sharing_group: [nordic]   # reuses the trained (frozen) encoder stack
    dec_sharing_group: [eng]      # reuses the trained (frozen) decoder
    path_src: data/sme-eng.sme
    path_tgt: data/sme-eng.eng
```

Here only `src_embeddings_sme` is new, so only the sme embedding is trained. The frozen tasks still run forward passes, but their weights do not change. In a modular model (`enc_layers: [3, 3]`, existing tasks using e.g. `enc_sharing_group: [nordic, swe]`), you can also give sme its own stack in a slot with `enc_sharing_group: [nordic, sme]`. That new stack `encoder_1_sme` is not in the checkpoint, so it is trained as well.

## Choosing what to freeze

There are two options. You can set **one** of them, not both:

- `freeze_components: [<pattern>, ...]` freezes the components that match and trains the rest.
- `train_components: [<pattern>, ...]` trains only the components that match and freezes the rest.

Patterns are shell-style wildcards (`*`, `?`, `[0-9]`), matched against the full component name. `@pretrained` matches every component that has a file in the `train_from` checkpoint.

```yaml
# Freeze all trained parts; train only new ones
freeze_components: ['@pretrained']

# Train the new sme parts, and also keep fine-tuning the eng target embedding
train_components: ['*_sme', 'tgt_embeddings_eng']

# Freeze only the shared encoder stack
freeze_components: ['encoder_0_nordic']
```

On the command line, quote the patterns so the shell does not expand them: `--freeze_components '@pretrained'`.

Rules:

- **Every pattern must match at least one component.** A typo such as `encoder_0_nordc` stops training at startup and prints the available names.
- **Adapters stay trainable** unless a `freeze_components` pattern contains the word `adapter` (e.g. `'*adapter*'`). With `train_components`, adapters are never frozen.
- **The older options still work.** `freeze_encoder`, `freeze_cross_attention`, etc. are applied too, and a parameter is frozen if any option freezes it. Use them for finer control than whole components, e.g. freezing only cross-attention.

At startup each GPU logs every component it holds as `trainable`, `frozen` or `partial` (partly frozen by one of the older options), together with parameter counts. Check this list before a long run.

## What happens to frozen components

- They get no optimizer, so weight decay and learning-rate changes cannot touch them.
- They are not part of the gradient all-reduce.
- They are still saved in every checkpoint, unchanged, so the new checkpoint is complete on its own.
