# Frequently Asked Questions


## About MAMMOTH

### What is MAMMOTH?

MAMMOTH (Massively Multilingual Modular Open Translation Toolkit) is an open-source toolkit for training neural machine translation (NMT) models. It is built on top of [OpenNMT-py](https://opennmt.net/OpenNMT-py/) and uses its own native PyTorch transformer implementation as the transformer backend. MAMMOTH is designed for large-scale multilingual training across multiple GPUs and nodes.

### What can I use MAMMOTH for?

You can use MAMMOTH to:

- Train translation models from scratch using your own parallel corpora (any language pair)
- Train massively multilingual models — hundreds of language pairs in a single training run
- Fine-tune from pretrained HuggingFace models (Gemma3)
- Export trained models to HuggingFace format for easy sharing and inference

### How does MAMMOTH relate to OpenNMT-py?

MAMMOTH builds on top of OpenNMT-py (the PyTorch version of OpenNMT). It inherits the core training pipeline and extends it with modular architecture, flexible parameter sharing, scalable distributed training, and HuggingFace integration.

### How can I contribute?

We welcome contributions from the community. Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on how to get involved, report issues, or submit pull requests.


## Getting Started

### How do I install MAMMOTH and run my first training?

For installing on HPC clusters (LUMI, Roihu, etc.), see the [LUMI/Roihu quickstart guide](CSC_quickstart.md).

The basic workflow is:

1. Prepare your data and tokenizer.
2. Write a config template (YAML).
3. Run the training/inferencing job


<!-- ## Configuration

### What is the `config_config` tool?

Writing training configs by hand gets difficult once you have many language pairs, multiple GPUs, or complex parameter sharing. The `config_config` tool reads a human-friendly template and generates the full config automatically. It handles:

- GPU and node assignments for all tasks
- Task weights and curriculum learning schedules
- Parameter sharing groups based on language clustering

See the [config_config documentation](config_config.md) for details. -->

### How do I set up parameter sharing between languages?

MAMMOTH supports dynamic parameter sharing, configured per LayerStack via `enc_sharing_groups` and `dec_sharing_groups`:

You can mix these across layers. For example:

```yaml
enc_sharing_groups: [Language, FULL]   # Language-specific layers + fully shared layers
dec_sharing_groups: [FULL, Language]  # Fully shared layers + language-specific layers 
enc_layers: [3, 3]
dec_layers: [3, 3]
```

See the [sharing schemes example](examples/sharing_schemes.md) for more patterns.


## Tokenizers and Data

### Should I use HuggingFace tokenizers or SentencePiece?

Both work, but we recommend HuggingFace tokenizers for new projects. They are faster (implemented in Rust), working on-the-fly (you don't need to pre-tokenize the data), easier to set up, and integrate smoothly with the HuggingFace ecosystem. See the [HF tokenizers guide](HF_TOKENIZERS.md) for instructions.

SentencePiece is still supported if you prefer it or have existing models.

### Can I use pretrained HuggingFace models as a starting point?

Yes. MAMMOTH can convert pretrained HuggingFace models into MAMMOTH checkpoints for fine-tuning:

- **Gemma3** — use as decoder (a new encoder is randomly initialized)

If you need a model that is not yet supported, please [open an issue](https://github.com/Helsinki-NLP/mammoth/issues) — we also welcome contributions for new model converters.

### How do I prepare data for training?

You need parallel text files (one sentence per line) for each language pair. For example:

```
data/
├── eng-fin/
│   ├── train.eng
│   ├── train.fin
│   ├── val.eng
│   └── val.fin
└── eng-spa/
    ├── train.eng
    ├── train.spa
    ...
```

Your training files can be plain, untokenized text. With a HuggingFace tokenizer (the .json path in your vocab config), MAMMOTH tokenizes on the fly during training, so you don't need to pre-process anything. Use transforms for other on-the-fly steps such as filtertoolong, prefix or denoising.

To tokenize once up front instead, set data_type: indexed. This writes .bin/.idx files next to your text, which removes tokenization cost during training. Transforms don't work with indexed data, so filter your data first. See [INDEXED_DATASET_GUIDE.md](INDEXED_DATASET_GUIDE.md).

See [prepare_data.md](prepare_data.md) for full details.


## Exporting and Inference

### How do I export a trained model to HuggingFace?

Use the conversion script:

```bash
python mammoth/hf_integration/to_hf/convert_mammoth_to_hf.py \
    --checkpoint-dir /path/to/checkpoints \
    --output-dir /path/to/hf_model
```

For multi-task models, you can convert each task separately (per-task) or bundle everything into a single artifact. See the [exporting guide](exporting_to_huggingface.md) for all options.

### Can I run inference without installing MAMMOTH?

Yes. Once you convert a model to HuggingFace format, anyone can run inference using standard HuggingFace dependencies plus extra dependencies (`transformers`, `torch`). No MAMMOTH installation needed.

### How do I download Mammoth (or any) models from HuggingFace Hub?

Use the provided downloader:

```bash
python mammoth/hf_integration/to_hf/model_downloader.py \
    --repo-id org/my-model \
    --local-dir ./my_model
```

For multi-task single artifact, you can download just one task to save space (Mammoth model only):

```bash
python mammoth/hf_integration/to_hf/model_downloader.py \
    --repo-id org/my-bundle --local-dir ./eng_spa --task eng-spa
```


## Training Tips

### How do I speed up training?

Several options:

- Use **token-based batching**: `batch_type: tokens` (balances workload across GPUs better than sentence-based)
- Use **gradient accumulation**: `accum_count: 10` (do more work before communicating gradients between GPUs)
- Sort by length with `lookahead_minibatches` (reduces padding waste, set `lookahead_minibatches` equals `accum_count` )

See [training_tips.md](training_tips.md) for a full guide.

### I'm getting `CUDA out of memory`. What should I do?

Try these in order:

1. Reduce `batch_size` (e.g., from 8192 to 4096 tokens)
2. Increase `accum_count` to keep the same effective batch size with smaller per-step memory (set `lookahead_minibatches` equals `accum_count`)

### What learning rate schedule should I use?

We recommend `decay_method: linear_warmup`. It ramps the learning rate up over `warmup_steps`, then decays it linearly until `train_steps`. Other schedules exist but have inconsistent scaling behavior. See [training_tips.md](training_tips.md) for details.

### How do I monitor training?

Enable TensorBoard in your config:

```yaml
tensorboard: true
report_every: 100
report_training_accuracy: true # Note turning on this slows down training
```

On LUMI, you can use the TensorBoard app on the dashboard. You can also tail the log files directly:

```bash
tail -f ./log/training.<job_id>.out
```


## Distributed Training

### How does distributed training work in MAMMOTH?

MAMMOTH distributes training across multiple GPUs and nodes. Each GPU handles a subset of the language pairs (tasks). After each training step, the GPUs share (synchronize) their gradient updates so the model stays consistent.

You control this with:

- `world_size`: total number of GPUs (one process per GPU) across all nodes, e.g. 2 nodes × 4 GPUs = 8.
- `--node_rank`: which node this process runs on (0, 1, 2, ...)
- `--master_ip` / `--master_port`: where all nodes connect to coordinate
- `gpu_ranks`: which GPUs to use on the current node

