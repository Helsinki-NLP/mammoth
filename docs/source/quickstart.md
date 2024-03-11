

# Quickstart

Mammoth is a library for training modular language models supporting multi-node multi-GPU training. 

In the example below, we will show you how to configure Mamooth to train a machine translation model with language-specific encoders and decoders.

### Step 0: Install mammoth

```bash
pip install mammoth-nlp
```

Check out the [installation guide](install) to install in specific clusters.

### Step 1: Prepare the data

Before running the training, we will download data for chosen pairs of languages and create a sentencepiece tokenizer for the model.

**Refer to the data preparation [tutorial](prepare_data) for more details.**

In the following steps, we assume that you already have an encoded dataset containing `*.sp` file for `europarl` dataset, and languages `cs` and `bg`. Thus, your data directory `europarl_data/encoded` should contain 8 files in a format `{train/valid}.{cs/bg}-en.{cs/bg}.sp`. If you use other datasets, please update the paths in the configurations below.

### Step 2: Configurations

Mamooth uses configurations to build a new transformer model and configure your training settings, such as which modules are trained with the data from which languages.

Below are a few examples of training configurations that will work for you out-of-box in a one-node, two-GPU environment.

<details>
<summary>Task-specific encoders and decoders</summary>

In this example, we create a model with encoders and decoders **shared** for the specified languages. This is defined by `enc_sharing_group` and `enc_sharing_group`.

```yaml
# TRAINING CONFIG
world_size: 2
gpu_ranks: [0, 1]

batch_type: tokens
batch_size: 4096

# INPUT/OUTPUT VOCABULARY CONFIG

src_vocab:
  bg: vocab/opusTC.mul.vocab.onmt
  cs: vocab/opusTC.mul.vocab.onmt
  en: vocab/opusTC.mul.vocab.onmt
tgt_vocab:
  cs: vocab/opusTC.mul.vocab.onmt
  en: vocab/opusTC.mul.vocab.onmt

# MODEL CONFIG

model_dim: 512

tasks:
  train_bg-en:
    src_tgt: bg-en
    enc_sharing_group: [bg]
    dec_sharing_group: [en]
    node_gpu: "0:0"
    path_src: europarl_data/encoded/train.bg-en.bg.sp
    path_tgt: europarl_data/encoded/train.bg-en.en.sp
  train_cs-en:
    src_tgt: cs-en
    enc_sharing_group: [cs]
    dec_sharing_group: [en]
    node_gpu: "0:1"
    path_src: europarl_data/encoded/train.cs-en.cs.sp
    path_tgt: europarl_data/encoded/train.cs-en.en.sp
  train_en-cs:
    src_tgt: en-cs
    enc_sharing_group: [en]
    dec_sharing_group: [cs]
    node_gpu: "0:1"
    path_src: europarl_data/encoded/train.cs-en.en.sp
    path_tgt: europarl_data/encoded/train.cs-en.cs.sp

enc_layers: [6]
dec_layers: [6]
```
</details>


<details>
<summary>Arbitrarily shared layers in encoders and task-specific decoders</summary>

The training and vocab config is the same as in the previous example.

```yaml
# TRAINING CONFIG
world_size: 2
gpu_ranks: [0, 1]

batch_type: tokens
batch_size: 4096

# INPUT/OUTPUT VOCABULARY CONFIG

src_vocab:
  bg: vocab/opusTC.mul.vocab.onmt
  cs: vocab/opusTC.mul.vocab.onmt
  en: vocab/opusTC.mul.vocab.onmt
tgt_vocab:
  cs: vocab/opusTC.mul.vocab.onmt
  en: vocab/opusTC.mul.vocab.onmt

# MODEL CONFIG

model_dim: 512

tasks:
  train_bg-en:
    src_tgt: bg-en
    enc_sharing_group: [bg, all]
    dec_sharing_group: [en]
    node_gpu: "0:0"
    path_src: europarl_data/encoded/train.bg-en.bg.sp
    path_tgt: europarl_data/encoded/train.bg-en.en.sp
  train_cs-en:
    src_tgt: cs-en
    enc_sharing_group: [cs, all]
    dec_sharing_group: [en]
    node_gpu: "0:1"
    path_src: europarl_data/encoded/train.cs-en.cs.sp
    path_tgt: europarl_data/encoded/train.cs-en.en.sp
  train_en-cs:
    src_tgt: en-cs
    enc_sharing_group: [en, all]
    dec_sharing_group: [cs]
    node_gpu: "0:1"
    path_src: europarl_data/encoded/train.cs-en.en.sp
    path_tgt: europarl_data/encoded/train.cs-en.cs.sp

enc_layers: [4, 4]
dec_layers: [4]
```
</details>

<details>
<summary>Non-modular multilingual system </summary>

In this example, we share the input/output vocabulary over all languages. Hence, we define a vocabulary for an `all` language, that we use in the definition of the model.

```yaml
# TRAINING CONFIG
world_size: 2
gpu_ranks: [0, 1]

batch_type: tokens
batch_size: 4096

# INPUT/OUTPUT VOCABULARY CONFIG

src_vocab:
  all: vocab/opusTC.mul.vocab.onmt
tgt_vocab:
  all: vocab/opusTC.mul.vocab.onmt

# MODEL CONFIG

model_dim: 512

tasks:
  train_bg-en:
    src_tgt: all-all
    enc_sharing_group: [shared_enc]
    dec_sharing_group: [shared_dec]
    node_gpu: "0:0"
    path_src: europarl_data/encoded/train.bg-en.bg.sp
    path_tgt: europarl_data/encoded/train.bg-en.en.sp
  train_cs-en:
    src_tgt: all-all
    enc_sharing_group: [shared_enc]
    dec_sharing_group: [shared_dec]
    node_gpu: "0:1"
    path_src: europarl_data/encoded/train.cs-en.cs.sp
    path_tgt: europarl_data/encoded/train.cs-en.en.sp
  train_en-cs:
    src_tgt: all-all
    enc_sharing_group: [shared_enc]
    dec_sharing_group: [shared_dec]
    node_gpu: "0:1"
    path_src: europarl_data/encoded/train.cs-en.en.sp
    path_tgt: europarl_data/encoded/train.cs-en.cs.sp

enc_layers: [6]
dec_layers: [6]
```
</details>

**To proceed, copy-paste one of these configurations into a new file named `my_config.yaml`.**

For further information, check out the documentation of all parameters in **[train.py](options/train)**.

For more complex scenarios, we recommend our [automatic configuration generation tool](config_config) for generating your configurations. 

## Step 3: Start training

The running script will slightly differ depending on whether you want to run the training in a single-node (i.e. a single-machine) or multi-node setting:

### Single-node training

If you want to run your training on a single machine, simply run a python script `train.py`, possibly with a definition of your desired GPUs. 
Note that the example config above assumes two GPUs available on one machine.

```shell
CUDA_VISIBLE_DEVICES=0,1 python train.py -config my_config.yaml -save_model output_dir -tensorboard -tensorboard_log_dir log_dir -node_rank 0
```

Note that when running `train.py`, you can use all the parameters from [train.py](options/train) as cmd arguments. In the case of duplicate arguments, the cmd parameters override the ones found in your config.yaml.

### Multi-node training

Now that you've prepared your data and configured the settings, it's time to initiate the training of your multilingual machine translation model using Mammoth. Follow these steps to launch the training script, for example, through the Slurm manager:

```bash
python -u "$@" --node_rank $SLURM_NODEID -u train.py \
    -config my_config.yaml \
    -save_model output_dir \
    -master_port 9974 -master_ip $SLURMD_NODENAME \
    -tensorboard -tensorboard_log_dir ${LOG_DIR}/${EXP_ID}
```
Explanation of Command:
   - `python -u "$@"`: Initiates the training script using Python.
   - `--node_rank $SLURM_NODEID`: Specifies the node rank using the environment variable provided by Slurm.
   - `-u ${PATH_TO_MAMMOTH}/train.py`: Specifies the path to the Mammoth training script.
   - `-config ${CONFIG_DIR}/your_config.yml`: Specifies the path to your configuration file.
   - `-save_model ${SAVE_DIR}/models/${EXP_ID}`: Defines the directory to save the trained models, incorporating an experiment identifier (`${EXP_ID}`).
   - `-master_port 9974 -master_ip $SLURMD_NODENAME`: Sets the master port and IP for communication.
   - `-tensorboard -tensorboard_log_dir ${LOG_DIR}/${EXP_ID}`: Enables TensorBoard logging, specifying the directory for TensorBoard logs.

Your training process has been initiated through the Slurm manager, leveraging the specified configuration settings. Monitor the progress through the provided logging and visualization tools. Adjust parameters as needed for your specific training requirements. You can also run the command on other workstations by modifying the parameters accordingly.



### Step 4: Translate

Now that you have successfully trained your multilingual machine translation model using Mammoth, it's time to put it to use for translation. 

```bash
python3 -u $MAMMOTH/translate.py \
  --config "${CONFIG_DIR}/your_config.yml" \
  --model "$model_checkpoint" \
  --task_id  "train_$src_lang-$tgt_lang" \
  --src "$path_to_src_language/$lang_pair.$src_lang.sp" \
  --output "$out_path/$src_lang-$tgt_lang.hyp.sp" \
  --gpu 0 --shard_size 0 \
  --batch_size 512
```

Follow these configs to translate text with your trained model.

- Provide necessary details using the following options:
   - Configuration File: `--config "${CONFIG_DIR}/your_config.yml"`
   - Model Checkpoint: `--model "$model_checkpoint"`
   - Translation Task: `--task_id "train_$src_lang-$tgt_lang"`

- Point to the source language file for translation:
   `--src "$path_to_src_language/$lang_pair.$src_lang.sp"`
- Define the path for saving the translated output: `--output "$out_path/$src_lang-$tgt_lang.hyp.sp"`
- Adjust GPU and batch size settings based on your requirements: `--gpu 0 --shard_size 0 --batch_size 512`
- We provide the model checkpoint trained using the encoder shared scheme described in [this tutorial](examples/sharing_schemes.md).
    ```bash
    wget https://mammoth-share.a3s.fi/encoder-shared-models.tar.gz
    ```

Congratulations! You've successfully translated text using your Mammoth model. Adjust the parameters as needed for your specific translation tasks.

### Further reading
A complete example of training on the Europarl dataset is available at [MAMMOTH101](examples/train_mammoth_101.md), and a complete example for configuring different sharing schemes is available at [MAMMOTH sharing schemes](examples/sharing_schemes.md).
