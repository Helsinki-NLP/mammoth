

# Quickstart

### Step 0: Install mammoth

```bash
git clone https://github.com/Helsinki-NLP/mammoth.git
cd mammoth
pip3 install -e .
pip3 install sentencepiece==0.1.97 sacrebleu==2.3.1
```

Check out the [installation guide](install) to install in specific clusters.

### Step 1: Prepare the data

Prepare the data for training. You can refer to the data preparation [tutorial](prepare_data) for more details.


### Step 2: Configurations
You will need to configure your training settings. 
Below is a list of configuration examples:

<details>
<summary>Task-specific encoders and decoders</summary>

```yaml
tasks:
  train_bg-en:
    src_tgt: bg-en
    enc_sharing_group: [bg]
    dec_sharing_group: [en]
    node_gpu: "0:0"
    path_src: /path/to/train.bg-en.bg
    path_tgt: /path/to/train.bg-en.en
  train_cs-en:
    src_tgt: cs-en
    enc_sharing_group: [cs]
    dec_sharing_group: [en]
    node_gpu: "0:1"
    path_src: /path/to/train.cs-en.cs
    path_tgt: /path/to/train.cs-en.en
  train_en-cs:
    src_tgt: en-cs
    enc_sharing_group: [en]
    dec_sharing_group: [cs]
    node_gpu: "0:1"
    path_src: /path/to/train.cs-en.en
    path_tgt: /path/to/train.cs-en.cs

enc_layers: [6]
dec_layers: [6]
```
</details>


<details>
<summary>Arbitrarily shared layers in encoders and task-specific decoders</summary>

```yaml
tasks:
  train_bg-en:
    src_tgt: bg-en
    enc_sharing_group: [bg, all]
    dec_sharing_group: [en]
    node_gpu: "0:0"
    path_src: /path/to/train.bg-en.bg
    path_tgt: /path/to/train.bg-en.en
  train_cs-en:
    src_tgt: cs-en
    enc_sharing_group: [cs, all]
    dec_sharing_group: [en]
    node_gpu: "0:1"
    path_src: /path/to/train.cs-en.cs
    path_tgt: /path/to/train.cs-en.en
  train_en-cs:
    src_tgt: en-cs
    enc_sharing_group: [en, all]
    dec_sharing_group: [cs]
    node_gpu: "0:1"
    path_src: /path/to/train.cs-en.en
    path_tgt: /path/to/train.cs-en.cs

enc_layers: [4, 4]
dec_layers: [4]
```
</details>

<details>
<summary>Non-modular multilingual system </summary>

```yaml
tasks:
  train_bg-en:
    src_tgt: all-all
    enc_sharing_group: [all]
    dec_sharing_group: [all]
    node_gpu: "0:0"
    path_src: /path/to/train.bg-en.bg
    path_tgt: /path/to/train.bg-en.en
  train_cs-en:
    src_tgt: all-all
    enc_sharing_group: [all]
    dec_sharing_group: [all]
    node_gpu: "0:1"
    path_src: /path/to/train.cs-en.cs
    path_tgt: /path/to/train.cs-en.en
  train_en-cs:
    src_tgt: all-all
    enc_sharing_group: [all]
    dec_sharing_group: [all]
    node_gpu: "0:1"
    path_src: /path/to/train.cs-en.en
    path_tgt: /path/to/train.cs-en.cs

enc_layers: [6]
dec_layers: [6]
```
</details>


We recommend our [automatic configuration generation tool](config_config) for generating your configurations. 


### Step 3: Start training

Finally, launch the training script, for example, through the Slurm manager, via:

```bash
python -u "$@" --node_rank $SLURM_NODEID -u ${PATH_TO_MAMMOTH}/train.py \
    -config ${CONFIG_DIR}/your_config.yml \
    -save_model ${SAVE_DIR}/models/${EXP_ID} \
    -master_port 9974 -master_ip $SLURMD_NODENAME \
    -tensorboard -tensorboard_log_dir ${LOG_DIR}/${EXP_ID}
```

A complete example of training on the Europarl dataset is available at [MAMMOTH101](examples/train_mammoth_101.md).