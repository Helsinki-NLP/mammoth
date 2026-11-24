# Installation

This guide covers installing MAMMOTH on a local machine or a generic GPU server. For the CSC supercomputers (LUMI and Roihu), which need cluster-specific modules or a container, see the [LUMI & Roihu quickstart](CSC_quickstart.md).

## Requirements

- Python 3.11 or newer
- PyTorch 2.8 or newer (the test suite is verified on 2.8.0). A GPU build is needed for real training. A CPU-only build is enough for inference: Mammoth trains big and runs small. Training a multilingual model needs many GPUs, but translating one direction loads only the encoder and decoder components that direction uses, so a single task runs comfortably on a laptop CPU (see [Running a Translation Server](serving.md)). CPU-only is also fine for tests and small experiments.
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
| `mammoth_export_hf` | convert a trained model to Hugging Face format |
| `mammoth_upload` | upload a converted model to the Hugging Face Hub |
| `mammoth_download` | download a model (or one task of a bundle) from the Hugging Face Hub |
| `mammoth_serve` | run a local translation server with a web page ([serving](serving.md)); needs the `server` extra |

See [HF tokenizers](HF_TOKENIZERS.md) for `mammoth_build_vocab`, the [quickstart](quickstart.md) for training and translating, and [export to Hugging Face](exporting_to_huggingface.md) for the three Hugging Face commands.

In a source checkout, the repository root also has `train.py` and `translate.py`, which do the same as `mammoth_train` and `mammoth_translate`.

## Optional dependencies

- **Hugging Face tokenizers and export.** `requirements.txt` already includes `tokenizers`, `transformers`, `huggingface-hub` and `safetensors`. See the [HF tokenizers guide](HF_TOKENIZERS.md) and the [export guide](exporting_to_huggingface.md).
- **Translation server.** `pip install 'mammoth-nlp[server]'` adds FastAPI and uvicorn for `mammoth_serve`.

## Troubleshooting

**`mammoth_<cmd>: command not found`.** The commands exist only after `pip install` (from PyPI or `pip install -e .`), and only while that venv is active. In a source checkout without installing, run the module instead, from the repository root:

| Command | Module |
|---|---|
| `mammoth_train` | `python -m mammoth.bin.train` |
| `mammoth_translate` | `python -m mammoth.bin.translate` |
| `mammoth_build_vocab` | `python -m mammoth.bin.build_vocab` |
| `mammoth_serve` | `python -m mammoth.bin.serve` |
| `mammoth_export_hf` | `python -m mammoth.hf_integration.to_hf.convert_mammoth_to_hf` |
| `mammoth_upload` | `python -m mammoth.hf_integration.to_hf.push_to_hub` |
| `mammoth_download` | `python -m mammoth.hf_integration.to_hf.model_downloader` |

**`ImportError` from `transformers` or another dependency** (for example `cannot import name 'PreTrainedModel' from 'transformers'`) while the traceback shows a path under `~/.local/lib/python3.*/site-packages`. Python also loads packages from your user site directory (`pip install --user`), and those can shadow the versions in your venv or container. This is common on clusters such as LUMI. Turn the user site off:

```bash
export PYTHONNOUSERSITE=1
python -c "import transformers; print(transformers.__version__, transformers.__file__)"   # should be >= 5.24, inside your venv
```

If the path still points outside your venv, reinstall in a fresh venv.


## Next steps

- [Quickstart](quickstart.md): a first training run on synthetic toy data.
- [LUMI & Roihu quickstart](CSC_quickstart.md): running on the CSC supercomputers.
- [Data preparation](prepare_data.md): preparing your own corpora.
