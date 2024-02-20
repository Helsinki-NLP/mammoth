

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

Now that you've prepared your data and configured the settings, it's time to initiate the training of your multilingual machine translation model using Mammoth. Follow these steps to launch the training script, for example, through the Slurm manager:

```bash
python -u "$@" --node_rank $SLURM_NODEID -u ${PATH_TO_MAMMOTH}/train.py \
    -config ${CONFIG_DIR}/your_config.yml \
    -save_model ${SAVE_DIR}/models/${EXP_ID} \
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

Congratulations! You've successfully translated text using your Mammoth model. Adjust the parameters as needed for your specific translation tasks.

### Further reading
A complete example of training on the Europarl dataset is available at [MAMMOTH101](examples/train_mammoth_101.md), and a complete example for configuring different sharing schemes is available at [MAMMOTH sharing schemes](examples/sharing_schemes.md)