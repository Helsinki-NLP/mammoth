"""
Generate the README.md model card for a converted Mammoth HF artifact.

The card is the first thing users see on the Hub, so it is a self-contained guide:
what the repo contains, which translation directions can be loaded, and the exact
commands to run them with the shipped inference.py. It is built from the artifact's
own config.json, so it always matches what the loader accepts.

The Hub repo id is not known at conversion time: the card uses REPO_ID_PLACEHOLDER,
which push_to_hub.py replaces with the real repo id before uploading.
"""

import json
import os

REPO_ID_PLACEHOLDER = "<repo_id>"
EXAMPLE_SENTENCE = "Helsinki is the capital of Finland."


def write_model_card(output_dir: str, src: str | None = None, tgt: str | None = None) -> str:
    """Write output_dir/README.md from output_dir/config.json. `src`/`tgt` are needed
    for a standalone model, whose config.json does not record its languages."""
    with open(os.path.join(output_dir, "config.json")) as f:
        config = json.load(f)
    model_type = config.get("model_type")
    if model_type == "mammoth_modular":
        card = _modular_card(config)
    elif model_type == "mammoth_hub":
        tasks = {name: (t["_src"], t["_tgt"]) for name, t in config["tasks"].items()}
        card = _bundle_card(tasks)
    else:
        if not (src and tgt):
            raise ValueError("A standalone model card needs src and tgt")
        card = _standalone_card(config, src, tgt)
    path = os.path.join(output_dir, "README.md")
    with open(path, "w") as f:
        f.write(card)
    print(f"Model card written → {path}")
    return path


# ---------------------------------------------------------------------------
# Shared sections
# ---------------------------------------------------------------------------

def _front_matter(langs) -> str:
    lines = ["---", "library_name: transformers", "pipeline_tag: translation",
             "tags:", "- translation", "- mammoth", "language:"]
    lines += [f"- {code}" for code in sorted({lang.split("_")[0].lower() for lang in langs})]
    return "\n".join(lines + ["---", ""])


def _setup_section() -> str:
    return f"""## Setup

No Mammoth installation is needed. Create an environment and fetch the inference script from this repo:

```bash
python -m venv .venv && source .venv/bin/activate
pip install transformers torch huggingface_hub safetensors
hf download {REPO_ID_PLACEHOLDER} inference.py --local-dir .
```

`inference.py` downloads the model files it needs on first use: into the Hugging Face cache by default, or into a folder of your choice with `--local-dir` (see [Offline Use](#offline-use)).
"""


def _options_section() -> str:
    return """### Options

| Option | Meaning |
|---|---|
| `--sentences "..." "..."` | Sentences to translate (instead of `--input-file`) |
| `--input-file <path>` | Text file, one source sentence per line |
| `--output-file <path>` | Write translations here, one per line (default: print to the terminal) |
| `--device cpu\\|cuda\\|mps` | Where to run the model (default: `cpu`) |
| `--num-beams N` | Beam size (default: 4) |
| `--max-new-tokens N` | Maximum output length in tokens (default: 128) |
| `--batch-size N` | Sentences per batch (default: 32) |
| `--local-dir <path>` | Download the model files into this folder instead of the Hugging Face cache |
"""


def _python_section(src: str, load_args: str, extra: str = "") -> str:
    sentence = EXAMPLE_SENTENCE if src == "eng" else f"<a sentence in {src}>"
    return f"""## Using the Model from Python

`inference.py` is also a module (it must be in the current directory or on `sys.path`):

```python
from inference import load_model, translate

src_tok, tgt_tok, model = load_model("{REPO_ID_PLACEHOLDER}"{load_args})
print(translate(src_tok, tgt_tok, model, ["{sentence}"]))
```

{extra}`translate()` takes `num_beams` and `max_new_tokens`. `model` is a standard `transformers` model, so `model.generate()` also works if you need more control.
"""


def _offline_section(task_example: str | None) -> str:
    if task_example:
        folder = f"./{task_example}"
        task = f" --task {task_example}"
        scope = "only the files needed for that task are downloaded"
        download_all = (f"\n\n# Every task\n"
                        f"python inference.py --model-dir {REPO_ID_PLACEHOLDER} "
                        f"--local-dir ./model --download-only")
    else:
        folder, task, scope, download_all = "./model", "", "the whole model is downloaded", ""
    return f"""## Offline Use

By default the model files go into the Hugging Face cache, which is hard to browse. Add `--local-dir` to download them into a folder of your choice instead ({scope}):

```bash
python inference.py --model-dir {REPO_ID_PLACEHOLDER}{task} --local-dir {folder} --input-file in.txt
```

The folder contains everything needed, including its own copy of `inference.py`. On later runs, pass the folder as `--model-dir` (no internet connection needed):

```bash
python {folder}/inference.py --model-dir {folder}{task} --input-file in.txt
```

To download without translating, add `--download-only`:

```bash
python inference.py --model-dir {REPO_ID_PLACEHOLDER}{task} --local-dir {folder} --download-only{download_all}
```

From Python: `load_model("{REPO_ID_PLACEHOLDER}", ..., local_dir="{folder}")` downloads and loads, `download("{REPO_ID_PLACEHOLDER}", "{folder}", ...)` only downloads; later, `load_model("{folder}", ...)`.
"""


def _notes_section() -> str:
    return """## Notes

- Mammoth uses a separate vocabulary per language, so each direction has a source tokenizer (for the input) and a target tokenizer (for the output). `inference.py` and `load_model()` pick the right pair for you.
- The model code is shipped in this repo (`modeling_mammoth.py`, `native_transformer.py`, ...), which is why loading uses `trust_remote_code=True`.
- Trained with [MAMMOTH](https://github.com/Helsinki-NLP/mammoth).
"""


def _example_task(pairs) -> str:
    """A representative task name: prefer English → other, else any cross-language pair."""
    names = sorted(pairs)
    for pred in (lambda s, t: s == "eng" and t != "eng", lambda s, t: s != t, lambda s, t: True):
        for name in names:
            if pred(*pairs[name]):
                return name


def _example_input(src: str) -> str:
    """CLI input for an example command: the English example sentence only fits English sources."""
    if src == "eng":
        return f'--sentences "{EXAMPLE_SENTENCE}"'
    return f"--input-file input.{src}.txt"


def _codes(langs) -> str:
    return " ".join(f"`{lang}`" for lang in sorted(langs))


# ---------------------------------------------------------------------------
# Per-format cards
# ---------------------------------------------------------------------------

def _standalone_card(config: dict, src: str, tgt: str) -> str:
    layers = f"{sum(config['enc_layers'])} encoder + {sum(config['dec_layers'])} decoder layers"
    return "\n".join([
        _front_matter({src, tgt}),
        f"# Mammoth Translation Model: `{src}` → `{tgt}`",
        "",
        f"A transformer translation model ({layers}, dimension {config['model_dim']}) "
        f"that translates from `{src}` to `{tgt}`.",
        "",
        _setup_section(),
        "## Translate",
        "",
        "```bash",
        f"python inference.py --model-dir {REPO_ID_PLACEHOLDER} {_example_input(src)}",
        f"python inference.py --model-dir {REPO_ID_PLACEHOLDER} --input-file in.txt --output-file out.txt",
        "```",
        "",
        _options_section(),
        _python_section(src, ""),
        _offline_section(None),
        _notes_section(),
    ])


def _targets(trained: list, all_tgts: set) -> str:
    """Trained targets of one source language; collapsed when it covers every target."""
    if set(trained) == set(all_tgts):
        return f"all {len(all_tgts)} target languages"
    return _codes(trained)


def _task_table(tasks: dict, all_tgts: set) -> str:
    """One row per source language with the targets it was trained for."""
    by_src = {}
    for src, tgt in tasks.values():
        by_src.setdefault(src, []).append(tgt)
    rows = ["| Source | Trained targets |", "|---|---|"]
    rows += [f"| `{src}` | {_targets(tgts, all_tgts)} |" for src, tgts in sorted(by_src.items())]
    return "\n".join(rows)


def _bundle_card(tasks: dict) -> str:
    srcs = {s for s, _ in tasks.values()}
    tgts = {t for _, t in tasks.values()}
    example = _example_task(tasks)
    return "\n".join([
        _front_matter(srcs | tgts),
        "# Mammoth Multilingual Translation Model",
        "",
        f"A multilingual transformer translation model with {len(tasks)} translation "
        f"directions (tasks), covering {len(srcs)} source and {len(tgts)} target languages. "
        "Each task is a complete model; you pick one with `--task <src>-<tgt>`.",
        "",
        _setup_section(),
        "## Translate",
        "",
        "```bash",
        f"python inference.py --model-dir {REPO_ID_PLACEHOLDER} --task {example} {_example_input(tasks[example][0])}",
        f"python inference.py --model-dir {REPO_ID_PLACEHOLDER} --task {example} --input-file in.txt --output-file out.txt",
        "```",
        "",
        "Only the files of the chosen task are downloaded.",
        "",
        _options_section(),
        "## Available Tasks",
        "",
        "Task names are `<source>-<target>`, e.g. `--task " + example + "`.",
        "",
        _task_table(tasks, tgts),
        "",
        _python_section(tasks[example][0], f', task="{example}"'),
        _offline_section(example),
        _notes_section(),
    ])


def _paths(tasks: dict, side: str) -> dict:
    """lang → set of encoder (side='src') or decoder (side='tgt') paths trained tasks use."""
    key = "encoder" if side == "src" else "decoder"
    out = {}
    for t in tasks.values():
        out.setdefault(t[side], set()).add(tuple(t[key]))
    return out


def _modular_card(manifest: dict) -> str:
    tasks = manifest["tasks"]
    arch = manifest["architecture"]
    pairs = {name: (t["src"], t["tgt"]) for name, t in tasks.items()}
    enc_paths, dec_paths = _paths(tasks, "src"), _paths(tasks, "tgt")
    srcs, tgts = set(enc_paths), set(dec_paths)
    trained = set(pairs.values())

    # --src/--tgt works when each language has exactly one trained path to infer.
    auto_srcs = {s for s, p in enc_paths.items() if len(p) == 1}
    auto_tgts = {t for t, p in dec_paths.items() if len(p) == 1}
    untrained = sorted((s, t) for s in auto_srcs for t in auto_tgts
                       if s != t and (s, t) not in trained)
    example = _example_task(pairs)

    # Encoder/Decoder columns only carry information when a path is not just the language itself.
    show_paths = any(p != (lang,) for paths in (enc_paths, dec_paths)
                     for lang, ps in paths.items() for p in ps)

    multi_stack = len(arch["enc_layers"]) > 1 or len(arch["dec_layers"]) > 1
    layers = (f"{sum(arch['enc_layers'])} encoder + {sum(arch['dec_layers'])} decoder layers, "
              f"dimension {arch['model_dim']}")

    lines = [
        _front_matter(srcs | tgts),
        "# Mammoth Multilingual Translation Model",
        "",
        f"A multilingual transformer translation model ({layers}) covering {len(srcs)} source "
        f"and {len(tgts)} target languages, trained on {len(tasks)} translation directions (tasks).",
        "",
        "The model is stored as separate **components** — one per language embedding, per "
        "encoder/decoder layer stack, and so on — each stored once even when many directions "
        "share it. When you choose a direction, only the components it uses are downloaded and "
        "assembled into a model.",
        "",
        _setup_section(),
        "## Translate",
        "",
        "```bash",
        "# A trained direction",
        f"python inference.py --model-dir {REPO_ID_PLACEHOLDER} --task {example} {_example_input(pairs[example][0])}",
    ]
    if untrained:
        s, t = untrained[0]
        lines += [
            "",
            "# A direction that was never trained (zero-shot): combine a source and a target language",
            f"python inference.py --model-dir {REPO_ID_PLACEHOLDER} --src {s} --tgt {t} {_example_input(s)}",
        ]
    lines += ["```", "", _options_section(), "## Choosing a Direction", ""]

    lines += [
        f"**Trained directions** — `--task <source>-<target>`. These {len(tasks)} directions "
        "were trained directly and are listed in the table below.",
        "",
    ]
    if untrained:
        lines += [
            f"**Zero-shot directions** — `--src <source> --tgt <target>`. Any source language "
            "can be combined with any target language, including pairs that were never trained "
            f"together ({len(untrained)} more direction{'s' if len(untrained) > 1 else ''}). "
            "The source language brings its "
            "encoder and the target language its decoder. Quality of untrained pairs varies; "
            "check the output before relying on it.",
            "",
        ]
    ambiguous = sorted(set(srcs - auto_srcs) | set(tgts - auto_tgts))
    if ambiguous or multi_stack or show_paths:
        lines += [
            "**Choosing layer stacks yourself** — `--encoder ... --decoder ...` takes one id per "
            "layer stack (see the *Encoder* / *Decoder* columns below). Only stack combinations "
            "that some trained direction used can be loaded.",
            "",
        ]
    if ambiguous:
        lines += [
            f"Languages with several encoder or decoder options ({_codes(ambiguous)}) need "
            "`--encoder` / `--decoder` when used with `--src` / `--tgt`.",
            "",
        ]

    def fmt(paths):
        return "<br>".join(" ".join(p) for p in sorted(paths)) if paths else "—"

    by_src = {}
    for s, t in pairs.values():
        by_src.setdefault(s, []).append(t)
    lines += ["## Languages", ""]
    if show_paths:
        lines += ["| Language | As source: trained targets | Encoder | Decoder |", "|---|---|---|---|"]
    else:
        lines += ["| Language | As source: trained targets |", "|---|---|"]
    for lang in sorted(srcs | tgts):
        row = f"| `{lang}` | {_targets(by_src[lang], tgts) if lang in by_src else '—'} |"
        if show_paths:
            row += f" {fmt(enc_paths.get(lang))} | {fmt(dec_paths.get(lang))} |"
        lines.append(row)
    lines.append("")
    if show_paths:
        lines += [
            "*Encoder* / *Decoder* show the layer-stack ids used for the language (one id per "
            "stack); `—` means the language is not available on that side.",
            "",
        ]
    lines += [
        _python_section(pairs[example][0], f', task="{example}"',
                        "`load_model` also takes `src=` / `tgt=` (and `encoder=` / `decoder=`), "
                        "like the command line. "),
        _offline_section(example),
        _notes_section(),
    ]
    return "\n".join(lines)
