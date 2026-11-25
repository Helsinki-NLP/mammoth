# Extending a Trained Model with a New Language

Say you have a model trained on `swe-eng` and `nob-eng`, and you want to add Northern Sami (`sme-eng`) without changing what the model already knows. Think of the trained model as a finished building: you add a new room (the `sme` parts) and lock all the existing rooms (the trained parts), so the work on the new room cannot damage the rest.

In MAMMOTH the "rooms" are **components**. Each component is one checkpoint file: a language embedding, a layer stack, a wrapper (output projection), an adapter, or the attention bridge. You lock (freeze) or unlock components by name.

In this example the component names you need are `src_embeddings_sme` (the new language) and `encoder_0_nordic` / `decoder_0_eng` (the trained, shared stacks). The full naming scheme is in [Freezing Components](../freezing_components.md).

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

## Other ways to choose

`freeze_components: ['@pretrained']` is the simplest choice. You can also list patterns, or say what to train instead of what to freeze:

```yaml
# Train the new sme parts, and also keep fine-tuning the eng target embedding
train_components: ['*_sme', 'tgt_embeddings_eng']
```

Pattern syntax, the rules (e.g. every GPU must have something to train), and what happens to frozen weights are described in [Freezing Components](../freezing_components.md). At startup each GPU logs which of its components are `trainable` or `frozen`. Check this list before a long run.

## The new checkpoint holds the whole model

You do not need to list the old tasks in the new config. Every checkpoint of a `train_from` run contains:

- the components of this run (trained or frozen);
- every other component of the `train_from` checkpoint, e.g. the encoder of an old source language this run does not use. These are hard-linked (copied across filesystems), so they stay valid if the old checkpoint is deleted;
- a frame whose `tasks` and `vocab` list the old tasks and this run's tasks.

So the new checkpoint directory is complete: translation and the HF/LiteRT converters see every language pair, old and new.
