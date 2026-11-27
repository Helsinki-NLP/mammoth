# Running a Translation Server

`mammoth_serve` runs a Mammoth model as a translation service on your own machine: a web page to try it out, and an HTTP API that is compatible with the [OPUS-MT server](https://github.com/Helsinki-NLP/Opus-MT). It runs on a laptop CPU; a GPU is optional.

Think of a Mammoth model as a shelf of parts: language embeddings, encoder and decoder layer stacks, output layers. A translation direction such as `eng-spa` is assembled from a few of those parts, and directions that share a part use the very same copy in memory. So the server runs one process for all directions, adding a direction only loads the parts that are not on the shelf yet, and a direction the model was never trained on (zero-shot) is assembled and served just like a trained one.

## Install

The server is an optional extra of the `mammoth-nlp` package (it adds FastAPI and uvicorn):

```bash
pip install 'mammoth-nlp[server]'
```

Or run it without installing anything permanently, using [uv](https://docs.astral.sh/uv/):

```bash
uvx --from 'mammoth-nlp[server]' mammoth_serve --model <model>
```

To pin a version, write `'mammoth-nlp[server]==<version>'`.

## Start the server

`--model` accepts every format Mammoth writes:

```bash
# A training checkpoint directory (the *_frame.pt and component files); no conversion needed
mammoth_serve --model path/to/checkpoint_dir

# An HF export, as a local directory or a Hub repo id: modular, --single-artifact bundle or single task
mammoth_serve --model Helsinki-NLP/<repo> --tasks eng-spa,deu-fra
```

For a checkpoint directory the server uses the best checkpoint (`*_best_frame.pt`), or else the one with the largest step; `--step N` picks a specific step. Hub repos are downloaded into the Hugging Face cache, only the files of the directions you load; for a private repo, log in first with `hf auth login`.

Then open <http://127.0.0.1:8888> in a browser.

### Which directions are loaded

`--tasks` decides what is loaded at start:

| `--tasks` | Loads |
|---|---|
| `trained` (default) | every direction the model was trained on |
| `all` | also every zero-shot direction: each source language with each target language |
| `none` | nothing; load directions later from the web page or the API |
| `eng-spa,deu-fra` | exactly these directions, trained or zero-shot |

For a zero-shot direction the server uses the encoder path the source language was trained with and the decoder path the target language was trained with (the path most trained tasks use for that language, if there are several). Whether zero-shot output is any good depends on the model: a model trained on only a few directions usually produces nonsense for directions it has never seen. Each direction is marked `trained` or `zero-shot` on the web page and in the API.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `cuda`, `cuda:N` or `mps`; `auto` = `cuda` if available, else `cpu` |
| `--dtype` | `fp32` | `fp32`, `bf16` or `fp16`. `bf16` halves the memory |
| `--threads` | torch's choice | CPU threads for torch |
| `--host` | `127.0.0.1` | interface to listen on; the default accepts connections from this machine only |
| `--port` | `8888` | port, as in OPUS-MT |
| `--num-beams` | `4` | beam size |
| `--max-new-tokens` | `512` | longest translation, in tokens; also the upper limit a request may ask for |
| `--max-length-factor` | `2.0` | a translation may also be at most this many times as long as its input (+ 10 tokens) |
| `--max-batch-size` | `32` | segments translated together |
| `--max-batch-tokens` | `4096` | padded tokens translated together (segments × longest segment) |

Run `mammoth_serve --help` for the full list.

## The web page

Pick a source and a target language, type or paste text, and press **Translate** (or Ctrl+Enter / ⌘+Enter). A direction that is not loaded yet shows a **Load direction** button. Each line of the input is translated as one segment and empty lines are kept, so the layout of the text survives. Hover over the `trained` / `zero-shot` mark to see which encoder and decoder path the direction uses.

## The HTTP API

### OPUS-MT compatible

These endpoints behave as in the OPUS-MT server, so its clients work unchanged:

```bash
# Loaded directions
curl http://127.0.0.1:8888/api/languages
# {"languages": {"eng": ["spa"], "fra": ["ita"]}}

# Translate
curl http://127.0.0.1:8888/api/translate -H 'content-type: application/json' \
     -d '{"from": "eng", "to": "spa", "source": "Where is the train station?"}'
# {"translation": "¿Dónde está la estación de tren?"}

# Ready check: 204 once a direction is loaded, 503 before
curl -i http://127.0.0.1:8888/api/ready
```

Language codes are the ones the model was trained with (for example `eng`), not necessarily OPUS-MT's (`en`). Errors come back as `{"error": "..."}`.

### Mammoth extensions

`/api/translate` also takes a list, and optionally `num_beams` (1–16) and `max_new_tokens` (at most the server's `--max-new-tokens`):

```bash
curl http://127.0.0.1:8888/api/translate -H 'content-type: application/json' \
     -d '{"from": "fra", "to": "ita", "source": ["Bonjour, comment allez-vous ?", "Je voudrais un café, s il vous plaît.", ""]}'
# {"translation": ["Ciao, come stai?", "Vorrei un caffè, per favore.", ""]}
```

Directions can be listed, loaded and unloaded while the server runs:

```bash
# Every direction the model can serve
curl http://127.0.0.1:8888/api/tasks
# {"eng-ita": {"src": "eng", "tgt": "ita", "encoder": ["eng", "all"], "decoder": ["ita", "all"],
#              "trained": false, "loaded": false}, ...}

# Load one (only parts not in memory yet are read)
curl http://127.0.0.1:8888/api/tasks -H 'content-type: application/json' -d '{"task": "eng-ita"}'

# Unload one (parts no other loaded direction uses are freed)
curl -X DELETE http://127.0.0.1:8888/api/tasks/eng-ita

# Device, dtype, loaded directions, and which direction uses which part
curl http://127.0.0.1:8888/api/health
```

While a direction loads, the other endpoints keep answering, and loaded directions keep translating.

The interactive API reference (OpenAPI) is at <http://127.0.0.1:8888/docs>.

### From Python

```python
import requests

r = requests.post("http://127.0.0.1:8888/api/translate",
                  json={"from": "eng", "to": "spa", "source": ["Hello!", "Good night."]})
r.raise_for_status()
print(r.json()["translation"])
```

## Things to know

- **No sentence splitting.** Each string sent to `/api/translate` is translated as one segment (the web page sends one segment per line). Models are trained on sentences, so a long paragraph translated in one piece may come out worse, or be cut off by the length limits. Split long text into sentences before sending it.
- **One translation at a time.** Requests are translated one after another; on a CPU, translating several at once would only compete for the same cores. Send a list rather than many single requests: a list is batched.
- **Memory.** What counts is the set of parts in memory, not the number of directions: directions that share parts cost little extra. `/api/health` shows which parts are loaded and by whom. `--dtype bf16` halves the memory; on CPU it can be slower or faster depending on the processor.
- **No authentication.** The server trusts everyone who can reach it, including loading and unloading directions. Keep the default `--host 127.0.0.1` unless the machine is on a network you trust; to reach a server on a remote machine, use an SSH tunnel: `ssh -L 8888:127.0.0.1:8888 <remote>`.
- **No container image yet.** Install with pip or uvx as above.
