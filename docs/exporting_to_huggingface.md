We provide scripts to convert Mammoth pre-trained models to Hugging Face Model Hub compatible format, so users can download Mammoth models from Model Hub for inference without setting up Mammoth.

**This guide covers:** model conversion → pushing to Hub → running inference → downloading locally.

## How We Convert the Model

Since Mammoth uses [x-transformers](https://github.com/lucidrains/x-transformers/tree/main/x_transformers) as its transformer backend, converting a Mammoth pre-trained model perfectly into Hugging Face format would require rewriting all transformer-related implementations in Hugging Face format. As a practical alternative, we bundle the Mammoth model together with its backend (x-transformers related files) and upload the bundle to Model Hub, so users can use the model without worrying about the dependencies.

## Model Conversion

### Single-Task Model

If the pre-trained Mammoth model has only one task, you can convert it into a standalone NMT model:

```bash
python mammoth/hf_integration/to_hf/convert_mammoth_to_hf.py \
  --checkpoint-dir <path> \
  --output-dir <path>
```

By default, the script loads the best checkpoint (`_best_frame.pt`) if one exists; otherwise it falls back to the checkpoint with the highest step number. You can also use `--step` to load a specific checkpoint step.

The output directory will have the following structure:

```
hf_single_task_model/
├── attend.py                     # vendored: attention mechanism from x-transformers
├── autoregressive_wrapper.py     # vendored: wraps model for autoregressive decoding
├── config.json                   # model architecture hyperparameters (HF format)
├── configuration_mammoth.py      # custom HF Config class definition for Mammoth
├── generation_config.json        # generation settings (beam size, max length, etc.)
├── model.safetensors             # model weights
├── modeling_mammoth.py           # custom HF Model class (forward pass and generate)
├── x_transformers.py             # vendored: core transformer implementation
├── src_tokenizer/                # source language tokenizer
│   ├── special_tokens_map.json   # maps special token names to their string values
│   ├── tokenizer_config.json     # tokenizer class and settings
│   └── tokenizer.json            # full vocabulary and tokenization rules
└── tgt_tokenizer/                # target language tokenizer (same structure as above)
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── tokenizer.json
```

### Multi-Task Model

If the pre-trained Mammoth model has more than one task, there are two approaches:

- **Per-task standalone** — Each task is converted as a separate, standalone model.
- **Single artifact** — All tasks are bundled into one artifact; users specify `--task` to select the model for inference.

#### Per-Task Standalone

Each task is converted as a standalone model (same structure as single-task above). By default, the script converts all tasks:

```bash
python mammoth/hf_integration/to_hf/convert_mammoth_to_hf.py \
  --checkpoint-dir <path> \
  --output-dir <path>
```

Each task gets its own subdirectory:

```
hf_multi_task/
├── task1/                        # one subdirectory per task (named by task ID)
│   ├── attend.py
│   ├── autoregressive_wrapper.py
│   ├── config.json
│   ├── configuration_mammoth.py
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── modeling_mammoth.py
│   ├── x_transformers.py
│   ├── src_tokenizer/
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   └── tgt_tokenizer/
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── tokenizer.json
└── task2/                        # same structure as task1
    └── ...
```

Use the `--task` flag to convert only a specific task:

```bash
python mammoth/hf_integration/to_hf/convert_mammoth_to_hf.py \
  --checkpoint-dir <path> \
  --output-dir <path> \
  --task eng-spa
```

#### Single Artifact

All submodels are bundled into one artifact:

```bash
python mammoth/hf_integration/to_hf/convert_mammoth_to_hf.py \
  --checkpoint-dir <path> \
  --output-dir <path> \
  --single-artifact
```

The output directory will have the following structure:

```
hf_single_artifact/
├── attend.py
├── autoregressive_wrapper.py
├── config.json                       # shared model architecture config for all tasks
├── configuration_mammoth.py
├── mammoth_hub.py                    # entry point: routes inference to the correct task
├── modeling_mammoth.py
├── x_transformers.py
├── bul-eng.safetensors               # weights for bul-eng (one .safetensors per task)
├── bul-eng_src_tokenizer/            # source tokenizer for bul-eng
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── bul-eng_tgt_tokenizer/            # target tokenizer for bul-eng
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── task2.safetensors                 # same pattern repeated for task2
├── task2_src_tokenizer/
├── task2_tgt_tokenizer/
└── ...                               # remaining tasks follow the same pattern
```

## Push the Converted Model to Model Hub

```bash
python mammoth/hf_integration/to_hf/push_to_hub.py \
  --model-dir <path> \
  --repo-id <org/model-name> \
  [--private] \
  [--token YOUR_HF_TOKEN]
```

This script uploads the entire directory, so it works for single-task, multi-task, and single-artifact models alike.

## Inference

Use `mammoth/hf_integration/to_hf/inference.py` to run inference with a converted Mammoth model from Hugging Face Model Hub.

No Mammoth installation is required. Set up a fresh environment with the minimum dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers torch einops einx loguru packaging
# optional: faster attention on supported GPUs
pip install flash-attn
```

### Standalone Model

```bash
python inference.py \
  --model-dir <repo_id_on_HF> \
  --input-file <path>
```

### Single Task from a Multi-Task Artifact

```bash
python inference.py \
  --model-dir <repo_id_on_HF> \
  --task <task-id> \
  --input-file <path>
```

### Additional Parameters

```bash
--device
--num-beams
--max-new-tokens
--sentences
--batch-size
--output-file
```

## Download Models for Local Use

By default, Hugging Face Transformers downloads the model to its cache directory. To download the model to a specific path for repeated use, use `mammoth/hf_integration/to_hf/model_downloader.py`:

```bash
# Single-task model — download everything
python model_downloader.py --repo-id org/my-model --local-dir ./my_model

# Multi-task single artifact — download everything
python model_downloader.py --repo-id org/my-bundle --local-dir ./my_bundle

# Multi-task single artifact — download ONE task only (saves disk space / bandwidth)
python model_downloader.py --repo-id org/my-bundle --local-dir ./eng_spa --task eng-spa
```