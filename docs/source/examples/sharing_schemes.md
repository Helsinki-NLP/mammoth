# MAMMOTH Sharing Schemes
MAMMOTH is designed as a flexible modular system, allowing users to configure, train, and test various sharing schemes. This tutorial walks through setting up and experimenting with different sharing schemes, including:

- fully shared
- fully unshared
- encoder shared
- decoder shared
- modular (partially shared) stacks

The scheme is controlled entirely by the YAML config, so switching between schemes never requires a code change.

The commands below follow the ready-to-use setups in `csc_env/` for the CSC supercomputers ([LUMI](https://docs.lumi-supercomputer.eu/) with AMD MI250X, and Roihu with NVIDIA GH200). The same configs work on any SLURM cluster, or on a single workstation GPU, once you adapt the paths.


## Data and Tokenizers
Each translation task needs:

- **Parallel training data** — one source file and one target file per task (plain text or `.gz`, one sentence per line, line-aligned).
- **Validation data (Optional)** — a source/target pair per task (the `csc_env` configs use [FLORES-200](https://github.com/openlanguagedata/flores) `dev`).
- **One HuggingFace tokenizer per language** (`tokenizer.json`). MAMMOTH uses **separate source and target vocabularies**: `src_vocab` tokenizes the input, `tgt_vocab` tokenizes the output. HuggingFace tokenizer is suggested while SentencePiece model is also supported as a legacy option.

Language codes are arbitrary strings (such as `eng`, `spa`, `fin`); they are used as keys in `src_vocab`/`tgt_vocab` and inside the sharing groups.


## Sharing Schemes Overview

Think of the encoder and the decoder as a row of **components**, where each component holds one block of transformer layers (a "stack"). `enc_layers: [6]` means one component with 6 layers; `enc_layers: [6,6]` means two components with 6 layers each.

Every task then puts a **name tag** on each component, using `enc_sharing_group` for the encoder components and `dec_sharing_group` for the decoder components. The rule is:

> If two tasks put the same name tag on the same component, they use the **same weights** for it. Different tags mean separate weights.

So the name tag decides who shares with whom:

- a language name such as `eng` → only the tasks that also tag that component `eng` share it;
- a common name such as `all` → every task that uses `all` shares it.

The tag is just a label, so `all` is a convention and not a keyword. Any name works, as long as tasks that should share use the same one.

Each list needs exactly one tag per component, so its length equals the length of `enc_layers` (or `dec_layers`). With a single component, that is a list with one tag, e.g. `["eng"]`.

All examples below use the languages `eng`, `spa`, `fin` and `swe`, with one task per language pair. Only the sharing-group lines are shown; the rest of each task entry is identical to the full config in the next section.

### 1. **Fully Unshared:**
   - Each language maintains a distinct set of parameters for both encoder and decoder.
   - No parameter sharing occurs between languages.
```yaml
tasks:
  eng-spa:
    src_tgt: "eng-spa"
    enc_sharing_group: ["eng"]
    dec_sharing_group: ["spa"]
  fin-swe:
    src_tgt: "fin-swe"
    enc_sharing_group: ["fin"]
    dec_sharing_group: ["swe"]
```
- `src_tgt`: the source and target language of the task; it selects the `src_vocab` and `tgt_vocab` entries.
- `enc_sharing_group`: which encoder parameters this task uses. `["eng"]` is used only by tasks that also list `eng`, so nothing is shared with `fin`.
- `dec_sharing_group`: same for the decoder. The two tasks have different source and target languages, so no parameters are shared at all. (If two tasks used the same decoder id, e.g. both `["spa"]`, they would share that decoder.)

### 2. **Shared Encoder, Separate Decoder:**
   - Encoder parameters are shared across all languages.
   - Each language has a separate set of parameters for the decoder.
```yaml
tasks:
  eng-spa:
    src_tgt: "eng-spa"
    enc_sharing_group: ["all"]
    dec_sharing_group: ["spa"]
  eng-fin:
    src_tgt: "eng-fin"
    enc_sharing_group: ["all"] # Notice the "all" component is shared by the both tasks
    dec_sharing_group: ["fin"] # The decoder component is not shared
```

### 3. **Separate Encoder, Shared Decoder:**
   - Each language has a separate set of parameters for the encoder.
   - Decoder parameters are shared across all languages.

```yaml
tasks:
  eng-spa:
    src_tgt: "eng-spa"
    enc_sharing_group: ["eng"]
    dec_sharing_group: ["all"]
  fin-spa:
    src_tgt: "fin-spa"
    enc_sharing_group: ["fin"] # The encoder component is not shared
    dec_sharing_group: ["all"] # Notice the "all" component is shared by the both tasks
```

### 4. **Fully Shared:**
   - Both encoder and decoder parameters are shared across all languages.
   - The entire transformer is shared among all language pairs.
```yaml
tasks:
  eng-spa:
    src_tgt: "eng-spa"
    enc_sharing_group: ["all"]
    dec_sharing_group: ["all"]
  fin-spa:
    src_tgt: "fin-spa"
    enc_sharing_group: ["all"] # Both encoder and decoder are shared. 
    dec_sharing_group: ["all"] # Note the "all" in encoder and "all" in decoder are TWO DIFFERENT components even if they use same name.
```
Note that embeddings and vocabularies stay per-language even in the fully shared scheme; only the transformer layer stacks are shared.

### 5. **Modular (Partially Shared) Stacks:**
   - `enc_layers` / `dec_layers` take a list, one entry per stack, and each stack gets its own sharing id.
   - This lets one part of the encoder be language-specific while another part is shared.

The multi-node Roihu example (`csc_env/roihu/multi_nodes.yaml`) uses a two-stack encoder and a single shared-by-target decoder stack:
```yaml
enc_layers: [6,6]    # two encoder stacks of 6 layers each
dec_layers: [12]     # one decoder stack of 12 layers

tasks:
  fin-swe:
    src_tgt: "fin-swe"
    enc_sharing_group: ["fin","all"]   # stack 0: Finnish-specific, stack 1: shared by everyone
    dec_sharing_group: ["swe"]         # Swedish-specific decoder
```
The `enc_sharing_group` list must have as many entries as `enc_layers` (likewise for the decoder).


## Example Configuration

A complete single-task config (`csc_env/lumi/single_node.yaml`; `csc_env/roihu/single_node.yaml` is identical apart from the paths). Adapt the paths to your machine:

```yaml
tasks:
  eng-spa:
    src_tgt: "eng-spa"
    weight: 1                       # relative sampling weight of this task
    introduce_at_training_step: 0   # step at which the task starts being sampled
    node_gpu: "0:0"                 # "node:gpu" this task is placed on
    enc_sharing_group: ["eng"]
    dec_sharing_group: ["spa"]
    transforms: [filtertoolong]
    path_src: /path/to/train.eng.gz
    path_tgt: /path/to/train.spa.gz
    path_valid_src: /path/to/flores200/dev/eng_Latn.dev
    path_valid_tgt: /path/to/flores200/dev/spa_Latn.dev

src_vocab:
   eng: /path/to/tokenizer/eng/32000/tokenizer.json
tgt_vocab:
   spa: /path/to/tokenizer/spa/32000/tokenizer.json

# Model architecture
enc_layers: [6]
dec_layers: [6]
model_dim: 1024
model_dtype: bf16
add_language_tokens: false

# Native PyTorch transformer options
heads: 16              # model_dim must be divisible by heads
rotary_pos_emb: true   # RoPE (currently the only option)
post_emb_norm: true    # RMSNorm after the token embedding
attn_dropout: 0.1
ff_dropout: 0.1
ff_activation: swiglu  # "swiglu" (default) or "gelu"

# Sequence lengths
src_seq_length_min: 1
tgt_seq_length_min: 1
src_seq_length_max: 512
tgt_seq_length_max: 512
max_length: 512

# Training
train_steps: 100000
early_stopping: 5
accum_count: [4]
lookahead_minibatches: 8
batch_size: 23000
batch_type: tokens
normalization: tokens
queue_size: 120

# Optimizer and LR schedule
optim: adamw
learning_rate: 0.0003
adam_beta1: 0.9
adam_beta2: 0.95
weight_decay: 0.01
max_grad_norm: 1.0
label_smoothing: 0.1
warmup_steps: 1000
decay_method: linear_warmup
learning_rate_decay: 0.5
start_decay_steps: 10000

# Distributed setup
world_size: 1          # total number of GPUs across all nodes
gpu_ranks: [0]         # GPU ids used on each node
node_rank: 0
n_nodes: 1
task_distribution_strategy: weighted_sampling
seed: 42

# Validation decoding
valid_batch_size: 16
valid_steps: 1500
valid_metrics: [bleu, chrf]
beam_size: 1

# Checkpointing
save_model: /path/to/output/model/
save_strategy: best_and_last
save_checkpoint_steps: 2500
keep_checkpoint: 1
```

The example configs also enable a BART-style denoising objective (`denoising_objective: bart`, `mask_ratio`, `mask_length`, `poisson_lambda`, `replace_length`); remove those keys if you only want plain translation training.

For multi-GPU / multi-node runs, add one entry to `tasks` per language pair and place each on a GPU with `node_gpu: "<node>:<gpu>"`. Set `n_nodes`, `gpu_ranks` (GPU ids per node) and `world_size` (total GPU count) to match. See `csc_env/roihu/multi_nodes.yaml` for a 2-node × 4-GPU example with eight tasks.

## Notes:
- To test a different scheme, change only the `enc_sharing_group` / `dec_sharing_group` entries (and `node_gpu` if you add tasks), retrain, and point `task_id` at the task you want to decode.