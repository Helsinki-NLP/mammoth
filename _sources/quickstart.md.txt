

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


### Step 2: Start training

You will need to configure your training settings. We recommend our [automatic configuration generation tool](config_config) for generating your configurations. 
Then launch the training script, for example, through the Slurm manager, via:

```bash
python -u "$@" --node_rank $SLURM_NODEID -u ${PATH_TO_MAMMOTH}/train.py \
    -config ${CONFIG_DIR}/your_config.yml \
    -save_model ${SAVE_DIR}/models/${EXP_ID} \
    -master_port 9974 -master_ip $SLURMD_NODENAME \
    -tensorboard -tensorboard_log_dir ${LOG_DIR}/${EXP_ID}
```

A complete example of training on the Europarl dataset is available at [MAMMOTH101](examples/train_mammoth_101.md).