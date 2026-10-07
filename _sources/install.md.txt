# Installation

This guide covers installing MAMMOTH on a local machine or a generic GPU server. For the CSC supercomputers (LUMI and Roihu), which need cluster-specific modules or a container, see the [LUMI & Roihu quickstart](CSC_quickstart.md).

## Requirements

- Python 3.10 or newer
- PyTorch 2.8 or newer (the test suite is verified on 2.8.0). A GPU build is needed for real training. CPU-only is fine for tests and small experiments.
- A Unix-like OS (Linux or macOS)

## Install from source (recommended)

Installing from source gives you the latest code and lets you edit it.

```bash
git clone https://github.com/Helsinki-NLP/mammoth.git
cd mammoth

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```


### Choosing the right PyTorch build

`requirements.txt` asks for `torch>=2.8`, which pip resolves to the default PyPI build. For a specific CUDA or ROCm version, install PyTorch first by following the selector at [pytorch.org](https://pytorch.org/get-started/locally/), then install the rest:

```bash
# example: CUDA 12.6 wheels
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt   # keeps the torch you just installed if it satisfies torch>=2.8
pip install -e .
```

If you need a torch older than 2.8, skip that line with `grep -v '^torch' requirements.txt | pip install -r /dev/stdin`.

## Install from PyPI

```bash
pip install mammoth-nlp
```

This installs the package and its console scripts, but not the dependencies (see the note above). Install PyTorch and the packages from `requirements.txt` separately.

## Verify the installation

```bash
python -c "import mammoth, torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
mammoth_train --help
```

The command-line tools installed by `pip install -e .` are:

| Command | Purpose |
|---|---|
| `mammoth_train` | train a model (equivalent to `python train.py`) |
| `mammoth_translate` | translate with a trained model |
| `mammoth_config_config` | generate multi-task configs, see [config_config](config_config.md) |
| `mammoth_iterate_tasks` | list the tasks in a config, optionally as `--task_id` arguments with `--src`/`--output` path templates (handy for translation scripts) |
| `mammoth_generate_synth_data` | generate synthetic toy data for the [quickstart](quickstart.md) |

The repository root also has `train.py` and `translate.py`, which do the same thing without installing the package.

## Optional dependencies

- **Hugging Face tokenizers and export.** `requirements.txt` already includes `tokenizers`, `transformers`, `huggingface-hub` and `safetensors`. See the [HF tokenizers guide](HF_TOKENIZERS.md) and the [export guide](exporting_to_huggingface.md).


## Next steps

- [Quickstart](quickstart.md): a first training run on synthetic toy data.
- [LUMI & Roihu quickstart](CSC_quickstart.md): running on the CSC supercomputers.
- [Data preparation](prepare_data.md): preparing your own corpora.
