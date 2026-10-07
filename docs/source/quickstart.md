# Quickstart

This quickstart trains a small English→German translation model on [Multi30k](https://github.com/multi30k/dataset) on a single GPU (or on CPU, only to test your setup). It covers the whole workflow:

1. Download the data.
2. Train a tokenizer for each language.
3. Write a training config.
4. Train.
5. Translate and score.

The result is a small baseline, not a production model. The point is to see how the pieces fit together.

For running on the CSC supercomputers (LUMI, Roihu), see the [LUMI & Roihu quickstart](CSC_quickstart.md) instead.

## Before you start

- Install MAMMOTH from source by following the [installation guide](install.md). Activate the environment you created there.
- A GPU (NVIDIA or AMD) is recommended. MAMMOTH also trains and translates on CPU, which is enough to check your setup but too slow for a useful model. To use the CPU, change the hardware settings as shown in Step 3.
- Run every command below from the root of the MAMMOTH repository.

## Step 1: Download the data

Multi30k is a small parallel corpus of image captions (about 29,000 training sentences per language).

```bash
mkdir -p data/multi30k
cd data/multi30k
for language in en de; do
    for split in train val test_2016_flickr; do
        wget "https://github.com/multi30k/dataset/raw/refs/heads/master/data/task1/raw/${split}.${language}.gz"
    done
done
cd ../..
```

This gives you `train`, `val` (validation) and `test_2016_flickr` files for each language. The files are gzipped, one sentence per line, and MAMMOTH reads `.gz` files directly.

## Step 2: Train the tokenizers

MAMMOTH uses [Hugging Face tokenizers](HF_TOKENIZERS.md). Each language gets its own tokenizer, trained with `mammoth/bin/build_vocab.py`. The script reads plain text, so decompress the training data first (`gzip -dc` works on both Linux and macOS):

```bash
for language in en de; do
    gzip -dc data/multi30k/train.${language}.gz > data/multi30k/train.${language}.txt
    python mammoth/bin/build_vocab.py \
        --input_file data/multi30k/train.${language}.txt \
        --output_dir models/tokenizers/${language} \
        --vocab_size 8000
done
```

Each output directory contains a `tokenizer.json`, which is the file you point the config at.

A tokenizer only splits text into subwords. MAMMOTH adds the beginning and end of sequence tokens itself.

## Step 3: Write the training config

Save the following as `multi30k_en_de.yaml`. It defines one translation task and a small Transformer.

```yaml
# ---- Task ----
tasks:
  en-de:
    src_tgt: en-de
    path_src: data/multi30k/train.en.gz
    path_tgt: data/multi30k/train.de.gz
    path_valid_src: data/multi30k/val.en.gz
    path_valid_tgt: data/multi30k/val.de.gz
    # Parameter-sharing groups: which encoder and decoder stack this task uses.
    enc_sharing_group: [en]
    dec_sharing_group: [de]
    # Run this task on node 0, GPU 0.
    node_gpu: "0:0"
    transforms: [filtertoolong]
    weight: 1
    introduce_at_training_step: 0

# ---- Tokenizers (one per language) ----
src_vocab:
  en: models/tokenizers/en/tokenizer.json
tgt_vocab:
  de: models/tokenizers/de/tokenizer.json
use_hf_tokenizer: true

# ---- Hardware ----
# One GPU. For CPU-only, use `world_size: 0` and `gpu_ranks: []`.
n_nodes: 1
world_size: 1
gpu_ranks: [0]

# ---- Model ----
model_dim: 256
heads: 4
enc_layers: [3]
dec_layers: [3]
rotary_pos_emb: true
dropout: 0.1
label_smoothing: 0.1

# ---- Data ----
batch_size: 4096
batch_type: tokens
normalization: tokens
valid_batch_size: 2048
src_seq_length_max: 100
tgt_seq_length_max: 100

# ---- Optimization ----
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
learning_rate: 0.0005
decay_method: linear_warmup
warmup_steps: 1000
max_grad_norm: 1.0
train_steps: 10000
valid_steps: 1000
report_every: 100
save_checkpoint_steps: 2500
keep_checkpoint: 3
seed: 3435

# ---- Output ----
save_model: models/multi30k_en_de
max_length: 200
```

The keys worth knowing about:

- **`tasks`**: every translation direction is a task. The task id (`en-de`) is what you pass to `--task_id` when translating. A multilingual setup lists many tasks, and tasks that name the same sharing group share parameters. That sharing is MAMMOTH's main feature. See [Modular model](modular_model.md) and [Sharing schemes](examples/sharing_schemes.md).
- **`enc_sharing_group` / `dec_sharing_group`**: one entry per layer stack. Here there is one encoder stack (`en`) and one decoder stack (`de`), so `enc_layers` and `dec_layers` each have one number.
- **`src_vocab` / `tgt_vocab`**: one tokenizer per language. The tokenizer for a language must be the same one used at translation time.
- **`gpu_ranks`, `world_size`, `n_nodes`**: these must describe the same hardware. `gpu_ranks: [0]` means one GPU in this node. An empty `gpu_ranks` with `world_size: 0` selects the CPU, and both training and translation then run on CPU.
- **`src_seq_length_max` / `tgt_seq_length_max`**: used by the `filtertoolong` transform, which drops training pairs longer than this many tokens.

On a small GPU, lower `batch_size`. To check that everything works before a full run (or when using the CPU), set `train_steps: 200` and `save_checkpoint_steps: 200`. A model trained for so few steps will produce poor or empty translations, which is expected.

## Step 4: Train

```bash
mammoth_train --config multi30k_en_de.yaml --node_rank 0
```

`--node_rank` is always required. It is `0` when you train on a single machine.

Training prints statistics every `report_every` steps and runs validation every `valid_steps`. Checkpoints are written under `models/`, named from `save_model` and the step number. The tokenizer files in `models/tokenizers/` are also updated in place if the config adds language tokens, so keep them next to the model.

## Step 5: Translate

Point `--model` at the checkpoint directory. MAMMOTH loads the best checkpoint if there is one, and otherwise the latest.

```bash
mkdir -p translations
mammoth_translate \
    --config multi30k_en_de.yaml \
    --model models/ \
    --task_id en-de \
    --src data/multi30k/test_2016_flickr.en.gz \
    --output translations/test_2016_flickr.en-de.trans \
    --beam_size 5
```

Translation uses the hardware settings from the config (the GPU, or the CPU if `gpu_ranks` is empty).

- `--task_id` selects which task's encoder, decoder and tokenizers to use.
- `--random_sampling_topk 1` switches to greedy decoding (instead of beam search).
- If you get `CUDA out of memory`, add `--batch_size 50`.

### Score the output

`sacrebleu` is installed with MAMMOTH's requirements.

```bash
gzip -dc data/multi30k/test_2016_flickr.de.gz > translations/test_2016_flickr.de.ref
sacrebleu translations/test_2016_flickr.de.ref -i translations/test_2016_flickr.en-de.trans
```

## Where to go next

- **More languages.** Add one entry per direction under `tasks`, one tokenizer per language under `src_vocab` / `tgt_vocab`, and choose sharing groups. For a multilingual model, add the `prefix` transform so the model knows the target language. See [Sharing schemes](examples/sharing_schemes.md).
- **More GPUs or nodes.** Set `n_nodes`, `world_size` and `gpu_ranks`, and give each task a `node_gpu` of the form `"<node>:<gpu>"`. See [Modular model](modular_model.md).
- [Best practices for training](training_tips.md).
- [Preparing your own data](prepare_data.md).
- [Hugging Face tokenizers guide](HF_TOKENIZERS.md).
- [Export a trained model to Hugging Face](exporting_to_huggingface.md).
- A multi-task, multi-GPU walkthrough on Europarl, including multi-node training: [MAMMOTH 101](examples/train_mammoth_101.md).
