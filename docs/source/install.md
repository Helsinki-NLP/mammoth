# Installation

This guide covers installing MAMMOTH on a local machine or a generic GPU server. For the CSC supercomputers (LUMI and Roihu), which need cluster-specific modules or a container, see the [LUMI & Roihu quickstart](CSC_quickstart.md).

## Requirements

- Python 3.10 or newer
- PyTorch 2.8 or newer (the test suite is verified on 2.8.0). A GPU build is needed for real training. CPU-only is fine for tests and small experiments.
- A Unix-like OS (Linux or macOS)

## Install from PyPI (recommended)

```bash
python -m venv .venv
source .venv/bin/activate

pip install mammoth-nlp
```

This installs MAMMOTH, its dependencies and its command-line tools.

### Choosing the right PyTorch build

`mammoth-nlp` asks for `torch>=2.8`, which pip resolves to the default PyPI build. For a specific CUDA or ROCm version, install PyTorch first by following the selector at [pytorch.org](https://pytorch.org/get-started/locally/), then install MAMMOTH:

```bash
# example: CUDA 12.6 wheels
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install mammoth-nlp   # keeps the torch you just installed if it satisfies torch>=2.8
```

## Install from source

Install from source if you want the latest unreleased code or want to edit MAMMOTH itself.

```bash
git clone https://github.com/Helsinki-NLP/mammoth.git
cd mammoth

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

To pick a PyTorch build, install it first as shown above. If you need a torch older than 2.8, skip the torch line with `grep -v '^torch' requirements.txt | pip install -r /dev/stdin`.

## Verify the installation

```bash
python -c "import mammoth, torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
mammoth_train --help
```

The command-line tools installed with the package are:

| Command | Purpose |
|---|---|
| `mammoth_train` | train a model |
| `mammoth_translate` | translate with a trained model |
| `mammoth_build_vocab` | train a BPE tokenizer |
| `mammoth_iterate_tasks` | list the tasks in a config, optionally as `--task_id` arguments with `--src`/`--output` path templates (handy for translation scripts) |
| `mammoth_generate_synth_data` | generate synthetic toy data for the quickstart |
| `mammoth_convert2hf` | convert a trained model to Hugging Face format |
| `mammoth_push2hf` | upload a converted model to the Hugging Face Hub |
| `mammoth_download_hf` | download a model (or one task of a bundle) from the Hugging Face Hub |

See [HF tokenizers](HF_TOKENIZERS.md) for `mammoth_build_vocab`, the [quickstart](quickstart.md) for training and translating, and [export to Hugging Face](exporting_to_huggingface.md) for the three Hugging Face commands.

Other tools run as Python modules, for example `python -m mammoth.bin.config_config` ([config_config](config_config.md)).

In a source checkout, the repository root also has `train.py` and `translate.py`, which do the same as `mammoth_train` and `mammoth_translate`.

## Optional dependencies

- **Hugging Face tokenizers and export.** `requirements.txt` already includes `tokenizers`, `transformers`, `huggingface-hub` and `safetensors`. See the [HF tokenizers guide](HF_TOKENIZERS.md) and the [export guide](exporting_to_huggingface.md).


## Next steps

- [Quickstart](quickstart.md): a first training run on synthetic toy data.
- [LUMI & Roihu quickstart](CSC_quickstart.md): running on the CSC supercomputers.
- [Data preparation](prepare_data.md): preparing your own corpora.
