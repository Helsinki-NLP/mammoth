# Freezing Components

A MAMMOTH model is made of **components**. A component is one checkpoint file: a language embedding, a layer stack, a wrapper (output projection), an adapter, or the attention bridge. You can freeze components by name, so their weights stay fixed while the rest of the model trains. A typical use is adding a new language to a trained model without changing what the model already knows (see [Extending a Trained Model with a New Language](examples/extending_with_a_new_language.md)).

## Component names

| Component | Name |
|---|---|
| encoder / decoder layer stack | `{encoder,decoder}_{stack}_{xcoder_id}`, e.g. `encoder_0_nordic` |
| language embedding (with its post-embedding norm) | `{src,tgt}_embeddings_{lang}`, e.g. `src_embeddings_sme` |
| wrapper (`to_logits`) | `{encoder,decoder}_wrapper_{xcoder_ids}`, e.g. `decoder_wrapper_eng` |
| adapter | `{encoder,decoder}_adapter_{stack}_{group}_{sub_id}` |
| attention bridge | `attention_bridge` |

The names are the checkpoint file names without the `model_step_N_` prefix. To list the components of a checkpoint with their sizes, see [Counting Parameters](counting_parameters.md).

## Choosing what to freeze

There are two options. You can set **one** of them, not both:

- `freeze_components: [<pattern>, ...]` freezes the components that match and trains the rest.
- `train_components: [<pattern>, ...]` trains only the components that match and freezes the rest.

Patterns are shell-style wildcards (`*`, `?`, `[0-9]`), matched against the full component name. The keyword `@pretrained` matches every component that has a file in the `train_from` checkpoint. Using `@pretrained` without `train_from` is an error.

```yaml
# Freeze all trained parts; train only new ones
freeze_components: ['@pretrained']

# Train the new sme parts, and also keep fine-tuning the eng target embedding
train_components: ['*_sme', 'tgt_embeddings_eng']

# Freeze only the shared encoder stack
freeze_components: ['encoder_0_nordic']
```

On the command line, quote the patterns so the shell does not expand them: `--freeze_components '@pretrained'`.

## Rules

- **Every pattern must match at least one component.** A typo such as `encoder_0_nordc` stops training at startup and prints the available names.
- **Adapters stay trainable** unless a `freeze_components` pattern contains the word `adapter` (e.g. `'*adapter*'`). With `train_components`, adapters are never frozen.
- **The `freeze_*` options still work.** `freeze_encoder`, `freeze_decoder`, `freeze_cross_attention`, `freeze_{encoder,decoder}_embeddings`, etc. are applied too, and a parameter is frozen if any option freezes it. Use them for finer control than whole components, e.g. freezing only cross-attention. See the [training options](options/train.rst).
- **Every GPU must have something to train.** If all components on a GPU are frozen, training stops at startup and names that GPU's tasks. Example: with a frozen decoder, an old task such as `eng-swe` whose encoder is also frozen has nothing to learn. Remove it from the config.
- **All GPUs make the same decision.** Patterns are matched against the full list of components in the config, not only the ones on one GPU, and each GPU then freezes the components it holds.

## Checking the result

At startup each GPU logs every component it holds as `trainable`, `frozen`, `partial` (partly frozen by one of the `freeze_*` options) or `empty` (no parameters, e.g. an encoder wrapper), together with parameter counts. Check this list before a long run.

## What happens to frozen components

- They get no optimizer, so weight decay and learning-rate changes cannot touch them.
- They are not part of the gradient all-reduce.
- They are still saved in every checkpoint, unchanged. No optimizer state (`*_optim.pt`) is written for them.
