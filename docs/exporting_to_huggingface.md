We provide conversion scripts for users to convert Mammoth pre-trained models to Hugging Face Model Hub compatible format, so users can download Mammoth models on Model Hub for inference without setting up Mammoth.

## How We Convert the Model
Since Mammoth uses x-transformers [https://github.com/lucidrains/x-transformers/tree/main/x_transformers] as its transformer backend, converting a Mammoth pre-trained model perfectly into Hugging Face format would require rewriting all transformer-related implementations in Hugging Face format. To save time, we instead bundle the Mammoth model together with its backend (x-transformers related files) and upload the bundle to Model Hub, so users can use the model without worrying about the dependencies.

## Model Conversion
If the pre-trained Mammoth model has only one task, you can convert it into a standalone NMT model:

```bash
python mammoth/hf_integration/to_hf/convert_mammoth_to_hf.py \
--checkpoint-dir \  # path to the checkpoint directory
--output-dir        # output directory for the converted HF model
```
By default, the script loads the best checkpoint (`_best_frame.pt`) if one exists, otherwise it falls back to the checkpoint with the highest step number. You can also use `--step` to load a specific checkpoint step.

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

If the pre-trained Mammoth model has more than one task, there are two ways to convert it:
- Convert each task as a standalone model (like the approach above). By default, the `convert_mammoth_to_hf.py` script converts all tasks in the Mammoth model checkpoint:
```bash
python mammoth/hf_integration/to_hf/convert_mammoth_to_hf.py \
--checkpoint-dir \
--output-dir
```
Each task gets its own subdirectory, and each subdirectory has the same structure as the single-task model:

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
You can also use the `--task` flag to convert only a specific task, since each converted model is standalone.

- Convert all submodels into one artifact:
```bash
python mammoth/hf_integration/to_hf/convert_mammoth_to_hf.py \
--checkpoint-dir \
--output-dir \
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
--model-dir \ 
--repo-id \ 
[--private] \
[--token YOUR_HF_TOKEN]
```

This script uploads the entire directory, so it works regardless of whether you are pushing a single-task model, a multi-task model, or a single artifact.

## Inference with the Mammoth Model on Hugging Face
Users can use `mammoth/hf_integration/to_hf/inference.py` to run inference with a converted Mammoth model from Hugging Face Model Hub.

No Mammoth installation is required. Set up a fresh environment with the minimum dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       
pip install transformers torch einops einx loguru packaging
# optional: faster attention on supported GPUs
pip install flash-attn
```

### To run inference on a standalone model from Hugging Face
```bash
python inference.py \
--model-dir <repo_id_on_HF> \
--input-file \
```
### To run inference on a single task from a Mammoth artifact on Model Hub
```bash
python inference.py \
--model-dir <repo_id_on_HF> \
--task <task-id> \
--input-file \
```
