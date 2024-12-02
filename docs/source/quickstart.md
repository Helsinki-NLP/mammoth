

# Quickstart

MAMMOTH is specifically designed for distributed training of modular systems in multi-GPUs SLURM environments.

In the example below, we will show you how to configure Mammoth.
We will use two small experiments as examples

1. A simple set of toy tasks with synthetic data. Easy and fast, requiring no resources except for the Mammoth package.
2. A machine translation model with language-specific encoders and decoders.

### Step 0: Install mammoth

```bash
pip install mammoth-nlp
```

Check out the [installation guide](install) to install in specific clusters.

## Experiment 1: Synthetic toy data

A simple set of toy tasks with synthetic data. 
Easy and fast, requiring no resources except for the Mammoth git repo.
This example uses a very small vocabulary, so we can use a "word level" model without sentencepiece.
The opts `--n_nodes`, `--n_gpus_per_node`, `--node_rank`, and `--gpu_rank` are set to use a single GPU.

### Step 1: Activate your virtual env

```bash
source ~/venvs/mammoth/bin/activate
```

### Step 2: Copy the config template from the Mammoth repo

```bash
mkdir config
pushd config
wget "https://raw.githubusercontent.com/Helsinki-NLP/mammoth/refs/heads/main/examples/synthdata.template.yaml"
popd 
```

### Step 3: Generate synthetic data

(this might take about 5 min)

```bash
mammoth_generate_synth_data \
    --config_path config/synthdata.template.yaml \
    --shared_vocab data/synthdata/shared_vocab
```

### Step 4: Generate the actual config from the config template 

(this should only take a few seconds)

```bash
mammoth_config_config \
    config_all \
    --in_config config/synthdata.template.yaml \
    --out_config config/synthdata.yaml \
    --n_nodes 1 \
    --n_gpus_per_node 1
```

### Step 5: Train the model

(This might take about 1h. To speed things up, train for a shorter time, e.g. `--train_steps 5000 --warmup_steps 600`)

```bash
mammoth_train --config config/synthdata.yaml --node_rank 0 --gpu_rank 0
```

### Step 6: Translate

(this might take a few minutes)

  - `--model` takes a prefix of the checkpoint files, of the form `{save_model}_step_{step}`.
  - `--random_sampling_topk 1` turns on greedy decoding.
  - If you get `CUDA out of memory` try reducing the batch size, e.g. `--batch_size 50`.

```bash
mammoth_translate \
    --config config/synthdata.yaml \
    --node_rank 0 --gpu_rank 0 \
    --model models/synthdata_step_50000 \
    --random_sampling_topk 1 \
    --max_length 200 \
    --task_id copy_source-copy_source \
    --src data/synthdata/test.copy_source-copy_source.src \
    --output translations/synthdata/test.copy_source-copy_source.greedy.trans
```

## Experiment 2: Machine translation with multi30k

### Step 1: Activate your virtual env

```bash
source ~/venvs/mammoth/bin/activate
```

### Step 2: Download data

```bash
mkdir data/multi30k
pushd data/multi30k

for language in cs en de fr; do
    wget "https://github.com/multi30k/dataset/raw/refs/heads/master/data/task1/raw/test_2016_flickr.${language}.gz"
    wget "https://github.com/multi30k/dataset/raw/refs/heads/master/data/task1/raw/val.${language}.gz"
    wget "https://github.com/multi30k/dataset/raw/refs/heads/master/data/task1/raw/train.${language}.gz"
done
popd
```

### Step 3: Train sentencepiece models

```bash
mkdir -p models/spm
for language in cs en de fr; do
    zcat data/multi30k/train.${language}.gz > /tmp/spm_train_${language}.txt
    spm_train --input /tmp/spm_train_${language}.txt --model_prefix=models/spm/spm.${language} --vocab_size 8000
    rm /tmp/spm_train_${language}.txt
done
```

### Step 4: Copy the config template from the Mammoth repo

```bash
mkdir config
pushd config
wget "https://raw.githubusercontent.com/Helsinki-NLP/mammoth/refs/heads/main/examples/multi30k.template.yaml"
popd 
```

### Step 5: Generate the actual config from the config template 

(this should only take a few seconds)

```bash
mammoth_config_config.py \
    config_all \
    --in_config config/multi30k.template.yaml \
    --out_config config/multi30k.yaml \
    --n_nodes 1 \
    --n_gpus_per_node 1
```

### Step 6: Train the model

(this might take a while)

```bash
mammoth_train --config config/multi30k.yaml --node_rank 0 --gpu_rank 0
```

### Step 7: Translate

(this might take a while)

  - `--model` takes a prefix of the checkpoint files, of the form `{save_model}_step_{step}`.
  - `--random_sampling_topk 1` turns on greedy decoding.
  - If you get `CUDA out of memory` try reducing the batch size, e.g. `--batch_size 50`.

Note that this time there are 16 language pairs, so we use the `iterate_tasks` and a loop to translate all language pairs in one command.
  - The `iterate_tasks` tool prints strings to use as parts of a command line. The strings contain spaces, so use a while-read-do loop.

```bash
export EXP_NAME=multi30k
export STEP=50000
CONFIG="config/${EXP_NAME}.yaml"
MODEL="models/${EXP_NAME}_step_${STEP}"

# Translate all language pairs
mkdir -p "translations/${EXP_NAME}/"
mammoth_iterate_tasks --config ${CONFIG} \
    --src "data/${EXP_NAME}/test_2016_flickr.{src_lang}.gz" \
    --output "translations/${EXP_NAME}/test_2016_flickr.{task_id}.greedy.trans" \
    | while read task_flags; do \
        mammoth_translate --config ${CONFIG} --node_rank 0 --gpu_rank 0 --model ${MODEL} --random_sampling_topk 1 --max_length 200       ${task_flags}; \
    done
```

Congratulations! You've successfully translated text using your Mammoth model. Adjust the parameters as needed for your specific translation tasks.

### Further reading

Reference documentation for the `config_config` tool can be found at [The config_config tool](config_config.md).

A complete example for configuring different parameter sharing schemes is available at [MAMMOTH sharing schemes](examples/sharing_schemes.md).

An older complete example of training on the Europarl dataset is available at [MAMMOTH101](examples/train_mammoth_101.md).

An [older version of the quickstart](old_quickstart.md) describes a manual procedure for configuring Mammoth. It may be difficult to adapt to your use case.
