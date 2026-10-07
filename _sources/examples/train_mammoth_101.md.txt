# Training MAMMOTH 101

This walkthrough trains a multilingual translation model with language-specific encoders and decoders on the [Europarl parallel corpus](https://www.statmt.org/europarl/), a multilingual resource extracted from European Parliament proceedings that covers 21 European languages. If you use the data in your research, please cite Philipp Koehn, "Europarl: A Parallel Corpus for Statistical Machine Translation," MT Summit 2005.

It is the multi-GPU follow-up to the [Quickstart](../quickstart.md), which trains one language pair on one GPU. Here you will:

1. Download and split the data.
2. Train a tokenizer.
3. Write a config with several tasks and assign them to GPUs.
4. Scale the config to more languages and to several nodes.
5. Launch training.

Tokenization uses [Hugging Face tokenizers](../HF_TOKENIZERS.md), and the model is MAMMOTH's native PyTorch Transformer. Run every command from the root of the MAMMOTH repository, with the environment from the [installation guide](../install.md) active.

## The model you will train

Every language gets its own encoder stack and its own decoder stack. A task such as `bg-en` uses the Bulgarian encoder and the English decoder, so each stack is trained by every task that uses it. The English decoder, for example, learns from all the `xx-en` tasks. This is the "language-specific" sharing scheme described in [Sharing schemes](sharing_schemes.md).

For each non-English language `xx` there are three tasks:

| Task | Encoder | Decoder | Data | Transforms |
|---|---|---|---|---|
| `xx-en` | `xx` | `en` | Europarl `xx`→`en` | `filtertoolong` |
| `en-xx` | `en` | `xx` | Europarl `en`→`xx` | `filtertoolong` |
| `xx-xx` | `xx` | `xx` | `xx` side of Europarl, as both source and target | `filtertoolong`, `denoising` |

The `xx-xx` task is an autoencoder: the `denoising` transform corrupts the source (BART-style masking) and the model learns to reconstruct it. It trains the `xx` encoder and decoder on text from their own language, without needing a translation.

To keep things small we start with two languages next to English (`bg` and `cs`, 6 tasks, 2 GPUs). [Step 4](#step-4-scale-up-more-languages-more-nodes) scales the same config to all 21 languages on several nodes.

## Step 1: Download and split the data

Download Europarl Release v7 (the version of 15 May 2012) from the original website, or use our processed copy. The three-language archive (about 666 MB) covers this walkthrough, and the full archive needs about 30 GB:

```bash
wget https://mammoth101.a3s.fi/europarl-3langs.tar.gz   # or europarl.tar.gz for all languages
mkdir europarl_data
tar -xvzf europarl-3langs.tar.gz -C europarl_data
```

The next step is to check that source and target are line-aligned, clean the pairs, shuffle, and hold out a validation set. This is done by the `prepare_parallel.py` script from [Preparing your data](../prepare_data.md#step-2-check-clean-and-split-trainvalid). Save that script, then run:

```bash
for lang in bg cs; do
    python prepare_parallel.py \
        --src europarl_data/europarl/${lang}-en/europarl-v7.${lang}-en.${lang} \
        --tgt europarl_data/europarl/${lang}-en/europarl-v7.${lang}-en.en \
        --src_lang ${lang} --tgt_lang en \
        --out_dir europarl_data/split/${lang}-en \
        --valid_size 1000
done
```

You end up with `europarl_data/split/bg-en/{train,val}.{bg,en}` and the same for `cs-en`. See [Preparing your data](../prepare_data.md) for details on what the script does.

## Step 2: Train the tokenizer

All languages share one BPE tokenizer here, trained on the training files of every language. `build_vocab.py` accepts any number of input files:

```bash
python mammoth/bin/build_vocab.py \
    --input_file europarl_data/split/bg-en/train.bg europarl_data/split/bg-en/train.en \
                 europarl_data/split/cs-en/train.cs europarl_data/split/cs-en/train.en \
    --output_dir tokenizer \
    --vocab_size 32000
```

The result is `tokenizer/tokenizer.json`. MAMMOTH adds the beginning and end of sequence tokens itself, so the tokenizer only splits text into subwords. For 21 languages, a larger vocabulary such as `--vocab_size 64000` is a better fit. See the [HF tokenizers guide](../HF_TOKENIZERS.md) for details.

You can instead train one tokenizer per language and point each language at its own file in `src_vocab` / `tgt_vocab`.

## Step 3: Write the config

Save the following as `europarl_2langs.yaml`. The tasks run on one node with two GPUs: all three Bulgarian tasks on GPU `0:0` and all three Czech tasks on GPU `0:1`.

<details>
<summary>Single-node configuration (2 GPUs)</summary>

```yaml
use_hf_tokenizer: true

# ---- Tokenizers (one entry per language; here they all share one file) ----
src_vocab:
  bg: tokenizer/tokenizer.json
  cs: tokenizer/tokenizer.json
  en: tokenizer/tokenizer.json
tgt_vocab:
  bg: tokenizer/tokenizer.json
  cs: tokenizer/tokenizer.json
  en: tokenizer/tokenizer.json

# ---- Tasks ----
tasks:
  # GPU 0:0
  train_bg-en:
    src_tgt: bg-en
    enc_sharing_group: [bg]
    dec_sharing_group: [en]
    node_gpu: "0:0"
    path_src: europarl_data/split/bg-en/train.bg
    path_tgt: europarl_data/split/bg-en/train.en
    path_valid_src: europarl_data/split/bg-en/val.bg
    path_valid_tgt: europarl_data/split/bg-en/val.en
    transforms: [filtertoolong]
    weight: 1
  train_bg-bg:
    src_tgt: bg-bg
    enc_sharing_group: [bg]
    dec_sharing_group: [bg]
    node_gpu: "0:0"
    path_src: europarl_data/split/bg-en/train.bg
    path_tgt: europarl_data/split/bg-en/train.bg
    path_valid_src: europarl_data/split/bg-en/val.bg
    path_valid_tgt: europarl_data/split/bg-en/val.bg
    transforms: [filtertoolong, denoising]
    weight: 1
  train_en-bg:
    src_tgt: en-bg
    enc_sharing_group: [en]
    dec_sharing_group: [bg]
    node_gpu: "0:0"
    path_src: europarl_data/split/bg-en/train.en
    path_tgt: europarl_data/split/bg-en/train.bg
    path_valid_src: europarl_data/split/bg-en/val.en
    path_valid_tgt: europarl_data/split/bg-en/val.bg
    transforms: [filtertoolong]
    weight: 1
  # GPU 0:1
  train_cs-en:
    src_tgt: cs-en
    enc_sharing_group: [cs]
    dec_sharing_group: [en]
    node_gpu: "0:1"
    path_src: europarl_data/split/cs-en/train.cs
    path_tgt: europarl_data/split/cs-en/train.en
    path_valid_src: europarl_data/split/cs-en/val.cs
    path_valid_tgt: europarl_data/split/cs-en/val.en
    transforms: [filtertoolong]
    weight: 1
  train_cs-cs:
    src_tgt: cs-cs
    enc_sharing_group: [cs]
    dec_sharing_group: [cs]
    node_gpu: "0:1"
    path_src: europarl_data/split/cs-en/train.cs
    path_tgt: europarl_data/split/cs-en/train.cs
    path_valid_src: europarl_data/split/cs-en/val.cs
    path_valid_tgt: europarl_data/split/cs-en/val.cs
    transforms: [filtertoolong, denoising]
    weight: 1
  train_en-cs:
    src_tgt: en-cs
    enc_sharing_group: [en]
    dec_sharing_group: [cs]
    node_gpu: "0:1"
    path_src: europarl_data/split/cs-en/train.en
    path_tgt: europarl_data/split/cs-en/train.cs
    path_valid_src: europarl_data/split/cs-en/val.en
    path_valid_tgt: europarl_data/split/cs-en/val.cs
    transforms: [filtertoolong]
    weight: 1

# ---- Hardware ----
n_nodes: 1
world_size: 2
gpu_ranks: [0, 1]

# ---- Model ----
model_dim: 512
heads: 8
ff_mult: 4
enc_layers: [6]
dec_layers: [6]
rotary_pos_emb: true
dropout: 0.1
label_smoothing: 0.1

# ---- Data ----
batch_size: 4096
batch_type: tokens
normalization: tokens
valid_batch_size: 4096
src_seq_length_max: 200
tgt_seq_length_max: 200

# ---- Denoising (used by the xx-xx autoencoder tasks) ----
mask_ratio: 0.2
mask_length: span-poisson
poisson_lambda: 3.0
replace_length: 1
denoising_objective: bart

# ---- Optimization ----
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
learning_rate: 0.0005
decay_method: linear_warmup
warmup_steps: 4000
max_grad_norm: 1.0
train_steps: 100000
valid_steps: 5000
report_every: 100
early_stopping: 5
early_stopping_criteria: accuracy
seed: 3435

# ---- Output ----
save_model: models/europarl
save_checkpoint_steps: 10000
keep_checkpoint: 3
max_length: 200
```
</details>

### What the config says

**Data and tokenizers**
- `src_vocab` / `tgt_vocab` list one tokenizer per language. The keys also define which languages exist. With `use_hf_tokenizer: true`, MAMMOTH loads each `.json` file as a Hugging Face tokenizer and tokenizes on the fly, so the data files are plain text.
- `src_seq_length_max` / `tgt_seq_length_max` are used by the `filtertoolong` transform, which drops training pairs longer than this many tokens.
- The denoising options (`mask_ratio`, `mask_length`, ...) only affect tasks that list the `denoising` transform.

**Tasks**
- Each entry under `tasks` is one translation direction, with the paths to its training and validation files, the transforms to apply, and a sampling `weight`.
- `enc_sharing_group` / `dec_sharing_group` name the encoder and decoder stack the task uses (one entry per layer stack). Tasks that name the same group share those parameters. Here the encoder group `[en]` is shared by all the `en-xx` tasks, and the decoder group `[en]` is shared by all the `xx-en` tasks.
- `node_gpu: "<node>:<gpu>"` assigns the task to a device, e.g. `"0:1"` is the second GPU of the first node. Keep the quotes. Tasks that share a GPU are sampled by `weight`.

**Hardware.** `n_nodes`, `world_size` and `gpu_ranks` must describe the same hardware: `world_size` is the total number of GPUs, and `gpu_ranks` lists the GPUs on one node. Every `node_gpu` must be one of those GPUs.

**Model.** A 6-layer, 512-dimensional Transformer with rotary position embeddings. `enc_layers: [6]` is a single layer stack of 6 layers. See [Modular model](../modular_model.md) for several stacks per side.

**Optimization.** Adam with a linear warmup and decay, gradient clipping per component, and early stopping on validation accuracy after 5 validations without improvement. These are reasonable starting values for a model of this size, not tuned ones. See [Training tips](../training_tips.md).

## Step 4: Scale up (more languages, more nodes)

Writing the tasks for 20 languages by hand is 60 entries. Because each language follows the same three-task pattern, generate them instead. The script below keeps everything that is not language-specific (model, optimization, ...) in a base file, and fills in the vocabularies, the tasks, and the hardware section.

First save the settings you want to share as `europarl_base.yaml`. It holds the `use_hf_tokenizer: true` line and the Model, Data, Denoising, Optimization and Output sections of the config above. The script adds the `src_vocab`, `tgt_vocab`, `tasks`, `n_nodes`, `world_size` and `gpu_ranks` keys. Then save this as `make_config.py`:

```python
"""Write a MAMMOTH config with one en->xx, xx->en and xx->xx (denoising) task per language."""
import argparse

import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", required=True, help="YAML file with the settings shared by all setups")
parser.add_argument("--langs", nargs="+", required=True, help="non-English languages, e.g. bg cs da")
parser.add_argument("--data_dir", default="europarl_data/split")
parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
parser.add_argument("--n_nodes", type=int, default=1)
parser.add_argument("--gpus_per_node", type=int, default=1)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.base) as f:
    config = yaml.safe_load(f)

all_langs = sorted(args.langs + ["en"])
config["src_vocab"] = {lang: args.tokenizer for lang in all_langs}
config["tgt_vocab"] = {lang: args.tokenizer for lang in all_langs}

n_gpus = args.n_nodes * args.gpus_per_node
tasks = {}
for i, lang in enumerate(args.langs):
    # All three tasks of a language share one GPU; languages are spread over the GPUs round-robin.
    slot = i % n_gpus
    node_gpu = f"{slot // args.gpus_per_node}:{slot % args.gpus_per_node}"
    pair_dir = f"{args.data_dir}/{lang}-en"
    for src, tgt, src_file, tgt_file, transforms in [
        (lang, "en", lang, "en", ["filtertoolong"]),
        (lang, lang, lang, lang, ["filtertoolong", "denoising"]),
        ("en", lang, "en", lang, ["filtertoolong"]),
    ]:
        tasks[f"train_{src}-{tgt}"] = {
            "src_tgt": f"{src}-{tgt}",
            "enc_sharing_group": [src],
            "dec_sharing_group": [tgt],
            "node_gpu": node_gpu,
            "path_src": f"{pair_dir}/train.{src_file}",
            "path_tgt": f"{pair_dir}/train.{tgt_file}",
            "path_valid_src": f"{pair_dir}/val.{src_file}",
            "path_valid_tgt": f"{pair_dir}/val.{tgt_file}",
            "transforms": transforms,
            "weight": 1,
        }
config["tasks"] = tasks
config["n_nodes"] = args.n_nodes
config["world_size"] = n_gpus
config["gpu_ranks"] = list(range(args.gpus_per_node))

with open(args.output, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
print(f"wrote {len(tasks)} tasks to {args.output}")
```

Run it for the setup you want. For example, the two-language config from Step 3 (one node, two GPUs):

```bash
python make_config.py --base europarl_base.yaml --langs bg cs --gpus_per_node 2 --output europarl_2langs.yaml
```

And all 21 languages (20 plus English) on 5 nodes with 4 GPUs each, which puts one language's three tasks on each of the 20 GPUs:

```bash
python make_config.py --base europarl_base.yaml \
    --langs bg cs da de el es et fi fr hu it lt lv nl pl pt ro sk sl sv \
    --n_nodes 5 --gpus_per_node 4 \
    --output europarl_21langs_5nodes.yaml
```

Before running the full set, you need the data split (Step 1) for every language, and a tokenizer trained on all of them (Step 2, with every language's training files and a larger vocabulary).

When there are more languages than GPUs, the script puts several languages on the same GPU, and that GPU samples among its tasks according to `weight`. You can also write `node_gpu` by hand to balance the load differently, for example by giving large corpora a GPU of their own.

The relevant lines of a multi-node config are these (the rest is unchanged):

```yaml
n_nodes: 5
world_size: 20          # 5 nodes x 4 GPUs
gpu_ranks: [0, 1, 2, 3] # the GPUs on each node
# ... and node_gpu values from "0:0" to "4:3"
```

## Step 5: Train

On a single machine (here, the 2-GPU config), `--node_rank` is always `0`:

```bash
mammoth_train --config europarl_2langs.yaml --node_rank 0 \
    --tensorboard --tensorboard_log_dir models/logs
```

Checkpoints are written under `models/`, named from `save_model` and the step number. Keep the tokenizer files next to the model, because you need them to translate.

### Several nodes with Slurm

Start the same command once per node, each with its own `--node_rank`, and point every node at the same master address. With Slurm, `srun` starts one wrapper per node, and the wrapper reads the rank from the environment. Save this as `train_multinode.sh`:

```bash
#!/bin/bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00

MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=9973

srun bash -c "mammoth_train \
    --config europarl_21langs_5nodes.yaml \
    --node_rank \${SLURM_NODEID} \
    --master_ip ${MASTER_NODE} \
    --master_port ${MASTER_PORT}"
```

Then submit it with `sbatch train_multinode.sh`.

- `n_nodes` in the config must equal the number of Slurm nodes, and `--node_rank` must equal the node's index. MAMMOTH checks both against `SLURM_NNODES` / `SLURM_NODEID` and stops with an error if they differ.
- The partition, account, modules or container, and GPU-type options in the `#SBATCH` header depend on your cluster. For LUMI and Roihu, see the [CSC quickstart](../CSC_quickstart.md).

## Step 6: Translate

Translate with the task you want, using the same config:

```bash
mammoth_translate \
    --config europarl_2langs.yaml \
    --model models/ \
    --task_id train_bg-en \
    --src europarl_data/split/bg-en/val.bg \
    --output val.bg-en.trans \
    --beam_size 5
```

`--task_id` selects the task's encoder, decoder and tokenizers; it is the key under `tasks`. See the [Quickstart](../quickstart.md#step-5-translate) for scoring the output.

Hooray! Take a moment to celebrate the progress you've made. Wait for hours (or days, for the full set) and the model training should be completed soon.
