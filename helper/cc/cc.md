# Config-config Tool

A meta-configuration tool, or config generator for MAMMOTH.

The MAMMOTH configuration options have become unwieldy as complexity has increased.
Especially the introduction of LayerStacks (the mechanism for dividing encoders and 
decoders into several subcomponents with different parameter sharing) and Adapters 
has made specifying parameters on the command line into a total nightmare, and even writing yaml configs by hand is cumbersome.
Other functionality that is cumbersome to specify by hand includes:

  - Node and GPU assignments for massively multilingual models,
  - Weights and curricula (starting step for specific tasks),
  - Parameter sharing groups based on language similarity.

To ease the creation of configs, the config-config tool reads in a human-writable 
configuration template, and computes the specific values expected by OpenNMT, 
writing them out as a less-readable yaml file.

## Command
```bash
python3 mammoth/tools/config_config.py config_all --in_config input.yaml --out_config output.yaml
```

## Inputs

The primary input is a yaml file, which contains two types of parameters

  - Passed through parameters: values copied from the input into the output, i.e. an OpenNMT yaml config file.
  - Meta-parameters, defining what config-config should do. These are contained under the `config_config` key.

### Input yaml

See the example in `mammoth/examples/config_config.yaml`


The meta-parameters under the `config_config` key:

#### `src_path` and `tgt_path`

Path templates for source and target corpora, respectively.
The path templates can contain the following variables that will be substituted by `config_config`:

- Directional corpus mode
  - `{src_lang}`: The source language of the task
  - `{tgt_lang}`: The target language of the task
  - `{lang_pair}`: `{src_lang}-{tgt_lang}` for convenience
- Symmetric corpus mode
  - `{lang_a}`: The alphabetically first language
  - `{lang_b}`: The alphabetically second language
  - `{side_a}`: 'src' if the language pair is used in the "forward" direction, otherwise 'trg'.  Tatoeba uses 'trg', not 'tgt'. Deal with it.
  - `{side_b}`: 'trg' if the language pair is used in the "forward" direction, otherwise 'src'.
  - `{sorted_pair}`: the source and target languages in alphabetical order, separated by a hyphen.

So for example, let's say your corpus contains the files `eng-ben/train.src.gz` (English side) and `eng-ben/train.trg.gz` (Bengali side).
You want to use the data symmetrically for both ben-to-eng and eng-to-ben directions.
For the first, `{lang_pair}` and `{sorted_pair}` are the same.
For the second, `{lang_pair}` is "eng-ben", but `{sorted_pair}` is "ben-eng".
In order to use the files in the correct order, you should use the template `{sorted_pair}/train.{side_a}.gz` for the source template, and `{sorted_pair}/train.{side_b}.gz` for the target template.

#### `ae_path`

Path templates for monolingual data for autoencoder tasks.
The same data will be used as both the source and target for the task: noise is introduced using transforms.
The path templates can contain the following variables that will be substituted by `config_config`:
`{src_lang}`, `{tgt_lang}`, and `{sorted_pair}`.
If unset, autoencoder pairs will use `src_path` and `tgt_path` instead.

#### `autoencoder`

If set to `True`, autoencoder tasks will be added.

#### `distance_matrix`

Path to the distance matrix comma-separated value (csv) file.

#### `n_groups`

The number of language groups to create when clustering.

#### `use_weight`

If set to `True`, use corpus weights based on temperature-adjusted corpus size.

Note that the actual weight is proportional to the weights of the the other tasks assigned to the same GPU. E.g. if only one task is assigned to a GPU, it will receive 100% weight regardless of what the computed weight is.

#### `use_introduce_at_training_step`

If set to `True`, use a curriculum introducing corpora based on temperature-adjusted corpus size.

Note that if both `use_weight` and `use_introduce_at_training_step` are specified, the weight is distributed to the two according to the square root, so that when both of them are applied (multiplicatively), the desired weight is achieved.

Note that high-resource language pairs (would train for over 75% of the training time) all start at 0. This avoids starting training with only one GPU doing work, while the other GPUs are idle waiting for their LPs to start.

#### `use_src_lang_token`

Only has an effect when using the `prefix` transform.
Normally, the prefix transform only includes a target language selector token: `<to_yyy>` where `yyy` is the code of the target language.
If this flag is set, then also the source language is specified, e.g. `<from_xxx> <to_yyy>`.

#### `translation_config_dir`

The directory in which to generate translation configs.
One config per language pair will be generated.
Only supervised pairs are generated, unless `zero_shot` is True.

#### `zero_shot`

Generate translation configs for zero-shot directions.

#### `transforms` and `ae_transforms`

A list of transforms, for translation tasks and autoencoder tasks, respectively.
Use this to apply subword segmentation, e.g. using `sentencepiece`, and `denoising` noise for autoencoder.
Both of these may change the sequence length, necessitating a `filtertoolong` transform.

#### `enc_sharing_groups` and `dec_sharing_groups`

A list of parameter sharing patterns, one for each LayerStack in the (enc|dec)oder.
Each list element takes one of 7 values:

  - `FULL`: fully shared parameters. Will be named using the constant "full".
  - `SRC_GROUP`: groupwise shared parameters. Will be named according to the cluster id of the *source* language.
  - `TGT_GROUP`: groupwise shared parameters. Will be named according to the cluster id of the *target* language.
  - `GROUP`: groupwise shared parameters. Same as `SRC_GROUP` for encoder and `TGT_GROUP` for decoder.
  - `SRC_LANGUAGE`: language specific parameters. Will be named according to the *source* language code.
  - `TGT_LANGUAGE`: language specific parameters. Will be named according to the *target* language code.
  - `LANGUAGE`: language specific parameters. Same as `SRC_LANGUAGE` for encoder and `TGT_LANGUAGE` for decoder.

Note that it is possible to have target-language-dependent components in the encoder, by using `TGT_LANGUAGE` or `TGT_GROUP` in the `enc_sharing_groups`.

#### `n_nodes` and `n_gpus_per_node`

The number of nodes and GPUs, for assignment of tasks to devices.
Note that you also need to separately specify this information to slurm.

#### Other top-level keys than `config_config`

##### Parameter sharing in adapters

The key `adapters.encoder.{adapter_name}.ids` takes one of 3 values:

  - `FULL`: fully shared parameters. Will be named using the constant "full".
  - `GROUP`: groupwise shared parameters. Will be named according to the cluster id.
  - `LANGUAGE`: language specific parameters. Will be named according to the language code.

(Adapters do not currently support the SRC_ and TGT_ prefixes)

### Distance matrix

See the example distance matrix in `mammoth/examples/config_config.distance.csv`.

The distance matrix is given as a csv file, with a column `lang` and one column per language.
There should be one row per language, with the language code in the first `lang` column, followed by a float giving the distance to the language specified by the column.
The rows should appear in the same order as the columns.
This means that the matrix must be square and symmetrical.
The upper and lower triangle are redundant, but both must be given.

Note that the values are distances: the distance of a language to itself (the diagonal) should be 0.

#### Alternative: specify language groups manually

To specify groups manually instead of using clustering, you must do two things:

  1. Leave `distance_matrix` unset.
  2. Specify a mapping of languages to groups: `config_config.groups.{lang}: {group}`.

### The actual corpora

The actual corpus files are used in two ways:

  - The presence of the files for a language pair determine if it is included or not.
  - Line counts from the files are used for weighting.

Because of this, you need to run `config_config` so that it can access the corpora using the specified `src_path` and `tgt_path`.

## Usage
```
python mammoth/tools/config_config.py config_all --in_config path/to/input.yaml --out_config path/to/output.yaml
```

## Input YAML

Start with a minimal human YAML.
      
Include tasks with `src_tgt`, `path_src`, `path_tgt`, `path_valid_*`,
and a small `config_config` section (temperature, clustering params,
etc.). Keep absolute or workspace-root-relative paths.

```
# in_config.yaml
languages: [en, es, de, fr]

config_config:
  groups:
    en: g0
    de: g0
    es: g1
    fr: g1

# You may also list pairs or tasks; both are fine
language_pairs: ["en-es", "de-fr"]

# Defaults you might already want in place
batch_type: tokens
src_seq_length: 256
tgt_seq_length: 256
```
Then
```
python -m mammoth.bin.config_config config_all \
  --in_config in_config.yaml \
  --use_weight \
  --temperature 0.7 \
  --n_nodes 1 --n_gpus_per_node 4 --n_slots_per_gpu 1 \
  --out_config final.yaml
```

## Stages

The tool runs in multiple stages. It is possible to run the steps
individually, by feeding in the output of the previous step as the
input of the next step.  This allows more control:

- Skipping unnecessary steps.

- Overriding what a particular step does by specifying its output
  manually.

The meta-stage `config_all` runs all of the stages in order:
```
# 05-train-ready.yaml
tasks:
  en_es:
    src_tgt: "en-es"
    transforms: [sentencepiece]
    path_src: data/en-es/train.en
    path_tgt: data/en-es/train.es
    path_valid_src: data/en-es/valid.en
    path_valid_tgt: data/en-es/valid.es

  de_es:
    src_tgt: "de-es"
    transforms: [sentencepiece]
    path_src: data/de-es/train.de
    path_tgt: data/de-es/train.es
    path_valid_src: data/de-es/valid.de
    path_valid_tgt: data/de-es/valid.es

# Vocab / SPM (shared model example)
src_subword_type: sentencepiece
tgt_subword_type: sentencepiece
src_subword_model: ./vocab/joint_spm.model
tgt_subword_model: ./vocab/joint_spm.model

# Reasonable caps
batch_type: tokens
src_seq_length: 256
tgt_seq_length: 256
```
Command:
```
python -m mammoth.bin.config_config config_all \
  --in_config 05-train-ready.yaml \
  --use_weight --temperature 0.3 \
  --n_nodes 1 --n_gpus_per_node 4 --n_slots_per_gpu 1 \
  --out_config 05-ready-augmented.yaml
```

Runs the end-to-end pipeline: generate tasks
(`complete_language_pairs`), weight/schedule (`corpora_schedule`),
group languages (`cluster_languages`), apply weight-sharing
(`sharing_groups`), allocate devices (`allocate_devices`), set
transforms (`set_transforms`), attach adapters (`adapter_config`),
(optionally) zero-shot (`translation_configs`), then strip helper keys
(`remove_temporary_keys`). Measures and logs total runtime.

## `complete_language_pairs` - Determine which language pairs have data.

Builds the `tasks` section by instantiating file-path templates for
every `(src_lang, tgt_lang)` from the configured vocab maps, checking
file existence, and adding only those pairs that have data. For
same-language pairs, optionally adds autoencoder tasks (and dev sets
if enabled), supporting multiple AE path templates. Also supports
reversed parallel data via `{side_a, side_b}` variables when the
sorted pair order differs from `src-tgt`. Finally, expands any
`{src_lang}` / `{tgt_lang}` placeholders in vocab paths.

The languages to consider as candidates are determined from the
vocabulary keys.  An example input:
```
# 02-pairs-from-templates.yaml
language_pairs:
  - en-es
  - de-es

# You can also predeclare transforms to be applied later:
default_transforms: [sentencepiece]
```

Generate all tasks (and optional autoencoders) from templates:

```
python -m mammoth.bin.config_config \
  complete_language_pairs \
  --in_config train.step3.yaml \
  --out_config train.step4.yaml \
  --src_path "/data/{src_lang}-{tgt_lang}/train.{src_lang}.gz" \
  --tgt_path "/data/{src_lang}-{tgt_lang}/train.{tgt_lang}.gz" \
  --valid_src_path "/dev/{sorted_pair}.{src_lang}" \
  --valid_tgt_path "/dev/{sorted_pair}.{tgt_lang}" \
  --autoencoder --autoencoder_validation
```
Then add transforms.

## `cluster_languages`: Determine language groups by clustering.

Produces a language→group mapping used elsewhere. Loads a symmetric
distance matrix (or accepts groups already in config). Filters out
languages not present in any `tasks` to avoid wasting clusters, then
runs agglomerative clustering with average linkage on the precomputed
distances. Assigns `group{i}` labels per language and stores them in
`config_config.groups`. Supports either a fixed `n_groups` or a
`cutoff_threshold`.

An example of the input config:
```
# 03-cluster-seed.yaml
languages: [en, es, de, fr]

# You may also pre-list pairs (paths can be added later)
tasks:
  en_es: { src_tgt: "en-es" }
  de_fr: { src_tgt: "de-fr" }
```
turn a similarity CSV into a distance CSV

If you have a symmetric similarity matrix (1.0 = identical), convert it to distance matrix with:
```
import csv
inp = "similarity.csv"   # header: lang,en,es,de,fr ; cells in [0,1]
out = "langs.csv"
with open(inp) as f, open(out, "w", newline="") as g:
    r = list(csv.reader(f))
    header = r[0]
    W = r[1:]
    writer = csv.writer(g)
    writer.writerow(header)
    for i,row in enumerate(W):
        lang = row[0]
        vals = [lang]
        for j,x in enumerate(row[1:]):
            if i==j:
                vals.append("0.0")            # diagonal
            else:
                s = float(x)
                d = max(0.0, min(1.0, 1.0-s)) # clamp into [0,1]
                vals.append(f"{d:.4f}")
        writer.writerow(vals)
```
Distance matrix:
```
lang,en,es,de,fr
en,0.0,0.20,0.70,0.50
es,0.20,0.0,0.80,0.40
de,0.70,0.80,0.0,0.60
fr,0.50,0.40,0.60,0.0
```
Checks:
- The CSV must include all languages you want clustered; any missing ones can’t be grouped.
- Extra languages in the matrix are okay; the tool will subset to those present in your YAML.
- Values don’t have to be in [0,1], but non-negative with 0 on diagonal is standard.

Command:
```
# cluster_languages needs a CSV distance matrix (langs x langs with header)
python -m mammoth.bin.config_config cluster_languages \
  --in_config 03-cluster-seed.yaml \
  --distance_matrix lang_distance.csv \
  --n_groups 2 \
  --out_config 03-clustered.yaml

python -m mammoth.bin.config_config \
  cluster_languages \
  --in_config train.step1.yaml \
  --out_config train.step2.yaml \
  --distance_matrix langs_dist.csv --n_groups 8

python -m mammoth.bin.config_config cluster_languages \
  --in_config 03-cluster-seed.yaml \
  --distance_matrix langs.csv \
  --n_groups 2 \
  --out_config 03-clustered.yaml
```
A possible result is:
```
groups:
  en: group0
  es: group0
  fr: group0
  de: group1
```
This step can be easily skipped by leaving the `distance_matrix` unset.

If the step is skipped, you should define the `config_config.groups` dict in the input yaml.
```
# Manually provide groups instead of running `cluster_languages`
config_config:
  groups:
    en: germanic
    de: germanic
    es: romance
    fr: romance
```
Combine this with the following task.

## `sharing_groups`: Apply the parameter sharing groups to tasks.

Materializes per-task encoder/decoder sharing assignments for each
layer. For a task’s `(src,tgt)`, it interprets spec tokens like
`LANGUAGE`, `GROUP`, `FULL`, and prefixed variants (`SRC_LANGUAGE`,
`TGT_GROUP`, …) into concrete IDs using the previously computed
`groups`. It asserts layer counts match the sharing spec lengths and
writes `enc_sharing_group` / `dec_sharing_group` arrays into each
task.&#x20;


```    
# Derive sharing groups from the clusters:
python -m mammoth.bin.config_config sharing_groups \
  --in_config 03-clustered.yaml \
  --out_config 03-sharing.yaml

python -m mammoth.bin.config_config \
  sharing_groups \
  --in_config train.step2.yaml \
  --out_config train.step3.yaml
```

Here are tiny, copy-pasteable examples of defining
config_config.groups manually in your input YAML so you can skip
cluster_languages.
```
# in_config.yaml
languages: [en, es, fr, de]

# Manually provide groups instead of running `cluster_languages`
config_config:
  groups:
    en: germanic
    de: germanic
    es: romance
    fr: romance

# (Optional) tasks you’ll later complete with paths/templates
tasks:
  en_es: { src_tgt: "en-es" }
  de_fr: { src_tgt: "de-fr" }
```
Now you can run only sharing_groups (and whatever else), skipping clustering:
```
python -m mammoth.bin.config_config sharing_groups \
  --in_config in_config.yaml \
  --out_config out.yaml
```
`out.yaml` will contain per-task `enc_sharing_group` /
`dec_sharing_group based` on your manual mapping.

## `allocate_devices`: Allocate tasks to nodes and gpus.

Assigns each task to a `(node:gpu)` slot subject to cluster size, GPUs
per node, and slots per GPU. If too few tasks initially start at step
0, it shifts all `introduce_at_training_step` values down so that at
least one task runs per GPU. For multi-GPU setups, it delegates
placement to a GPU-assignment optimizer that tries to pack language
pairs (optionally considering groups and “ready-to-start” status). It
writes `node_gpu` for each task and sets `n_nodes`, `world_size`, and
`gpu_ranks`. Finally, it normalizes curricula per device so every GPU
has something starting at step 0.

A local search procedure is used, taking into account parameter
sharing groups and tasks delayed by curriculum weighting.
The input:
```
# 04-allocate-seed.yaml
tasks:
  en_es:
    src_tgt: "en-es"
    path_src: data/en-es/train.en
    path_tgt: data/en-es/train.es
  de_es:
    src_tgt: "de-es"
    path_src: data/de-es/train.de
    path_tgt: data/de-es/train.es
```
If you want the tool to propose a packing for your cluster:
    
```
python -m mammoth.bin.config_config allocate_devices \
  --in_config 04-allocate-seed.yaml \
  --n_nodes 1 --n_gpus_per_node 4 --n_slots_per_gpu 1 \
  --time_budget_s 5 \
  --out_config 04-assigned.yaml

python -m mammoth.bin.config_config \
  allocate_devices \
  --in_config train.step5.yaml \
  --out_config train.alloc.yaml \
  --n_nodes 4 --n_gpus_per_node 8 --n_slots_per_gpu 1 \
  --time_budget_s 30
```
A more elaborate example:
```
# --- before allocate_devices runs ---
config_version: v1

config_config:
  n_nodes: 2
  n_gpus_per_node: 4
  n_slots_per_gpu: 3

  # Language clusters used by homogeneity/communication terms
  groups:
    en: pivot
    fi: uralic
    et: uralic
    lv: baltic
    lt: baltic
    ru: slavic
    uk: slavic
    kk: turkic

  # Costs in the optimizer’s objective (illustrative)
  device_assignment:
    INTER_NODE_COST: 10
    INTRA_NODE_COST: 3
    HOMOGENEITY_PENALTY: 1
    VERY_BAD: 10_000   # (forbidden patterns like co-locating same split)
    # “ready-coverage” is enforced via the objective/spread

# 12 direction pairs × 2 splits = 24 concrete tasks
tasks:
  # --- English ↔ Uralic (4 tasks × 2 splits) ---
  en-fi-0:
    src_lang: en
    tgt_lang: fi
    offset: 0
    introduce_at_training_step: 0     # ready
  en-fi-1:
    src_lang: en
    tgt_lang: fi
    offset: 1
    introduce_at_training_step: 2000
  fi-en-0:
    src_lang: fi
    tgt_lang: en
    offset: 0
    introduce_at_training_step: 0     # ready
  fi-en-1:
    src_lang: fi
    tgt_lang: en
    offset: 1
    introduce_at_training_step: 4000

  en-et-0:
    src_lang: en
    tgt_lang: et
    offset: 0
    introduce_at_training_step: 0     # ready
  en-et-1:
    src_lang: en
    tgt_lang: et
    offset: 1
    introduce_at_training_step: 2000
  et-en-0:
    src_lang: et
    tgt_lang: en
    offset: 0
    introduce_at_training_step: 0     # ready
  et-en-1:
    src_lang: et
    tgt_lang: en
    offset: 1
    introduce_at_training_step: 4000

  # --- English ↔ Baltic (4 tasks × 2 splits) ---
  en-lv-0:
    src_lang: en
    tgt_lang: lv
    offset: 0
    introduce_at_training_step: 0     # ready
  en-lv-1:
    src_lang: en
    tgt_lang: lv
    offset: 1
    introduce_at_training_step: 3000
  en-lt-0:
    src_lang: en
    tgt_lang: lt
    offset: 0
    introduce_at_training_step: 0     # ready
  en-lt-1:
    src_lang: en
    tgt_lang: lt
    offset: 1
    introduce_at_training_step: 3000

  # --- English ↔ Slavic (6 tasks × 2 splits) ---
  en-ru-0:
    src_lang: en
    tgt_lang: ru
    offset: 0
    introduce_at_training_step: 0     # ready
  en-ru-1:
    src_lang: en
    tgt_lang: ru
    offset: 1
    introduce_at_training_step: 5000
  ru-en-0:
    src_lang: ru
    tgt_lang: en
    offset: 0
    introduce_at_training_step: 0     # ready
  ru-en-1:
    src_lang: ru
    tgt_lang: en
    offset: 1
    introduce_at_training_step: 5000
  en-uk-0:
    src_lang: en
    tgt_lang: uk
    offset: 0
    introduce_at_training_step: 0     # ready
  en-uk-1:
    src_lang: en
    tgt_lang: uk
    offset: 1
    introduce_at_training_step: 5000

  # --- English ↔ Turkic (2 tasks × 2 splits) ---
  en-kk-0:
    src_lang: en
    tgt_lang: kk
    offset: 0
    introduce_at_training_step: 0     # ready
  en-kk-1:
    src_lang: en
    tgt_lang: kk
    offset: 1
    introduce_at_training_step: 6000

# (No node_gpu fields yet; allocate_devices will fill those + top-level device metadata)
```

####  Why this instance is computationally difficult?

1. Conflicting objectives:

   - Minimize inter-node spread of components (pivot en touches
     everything, so naïvely it explodes across nodes).
   - Keep each GPU homogeneous (few distinct groups per GPU).
   - Ensure every GPU has at least one ready task at step 0.
   - Obey “split” rules (e.g., don’t put en-fi-0 and en-fi-1 on the
     same GPU if you forbid co-locating splits).

2. Combinatorics: 24 tasks into 24 ordered slots across 8 GPUs on 2
nodes → massive number of assignments; many are near-ties, so the
search must inspect lots of swaps to gain small objective
improvements.

3. Coupling across the whole cluster: moving one en-ru-0 task can
change the communication cost for components en, ru, slavic, pivot
across multiple GPUs/nodes.

Below is a plausible result (one of many optimal/near-optimal
layouts). The important part is how the result is encoded: each task
now has a node_gpu: "node_id:gpu_id" (the slot index is implicit by
row order in trainer launch or is irrelevant if all tasks on a GPU are
multiplexed by the dataloader), and the top-level has n_nodes,
world_size, and gpu_ranks.

```
# --- after allocate_devices runs ---
n_nodes: 2
world_size: 8
gpu_ranks: [0,1,2,3,4,5,6,7]

tasks:
  # Node 0
  en-fi-0: { node_gpu: "0:0" }   # ready
  et-en-0: { node_gpu: "0:0" }   # ready
  en-lv-0: { node_gpu: "0:1" }   # ready
  en-lt-0: { node_gpu: "0:1" }   # ready
  en-ru-0: { node_gpu: "0:2" }   # ready
  ru-en-0: { node_gpu: "0:2" }   # ready
  en-uk-0: { node_gpu: "0:3" }   # ready
  en-kk-0: { node_gpu: "0:3" }   # ready
  en-fi-1: { node_gpu: "0:0" }   # late (offset 1)
  en-et-1: { node_gpu: "0:0" }   # late
  en-lv-1: { node_gpu: "0:1" }   # late
  en-lt-1: { node_gpu: "0:1" }   # late

  # Node 1
  fi-en-0: { node_gpu: "1:0" }   # ready
  en-et-0: { node_gpu: "1:0" }   # ready
  ru-en-1: { node_gpu: "1:1" }   # late
  en-ru-1: { node_gpu: "1:1" }   # late
  en-uk-1: { node_gpu: "1:2" }   # late
  en-kk-1: { node_gpu: "1:2" }   # late
  fi-en-1: { node_gpu: "1:3" }   # late
  et-en-1: { node_gpu: "1:3" }   # late
  # (two more “late” tasks to fill Node1’s remaining slots if you model exactly 3 slots/GPU)

# Optional: allocate_devices often also normalizes per-device curriculum
# so each GPU has at least one task starting at step 0:
# introduce_at_training_step values are shifted per GPU so min starts at 0.
```

#### What makes this placement “good” (qualitatively)

- Each GPU has at least one ready task (*-0) at step 0 → the trainer
  can fully utilize all 8 GPUs immediately.

- Heavy components (like en/pivot) are concentrated to reduce
  inter-node spread:

- Node 0 hosts most of the “pivot + Baltic/Uralic/Slavic” ready
  directions; Node 1 carries the complementary reverse directions and
  late splits.

- Within a GPU, we co-locate related groups (e.g., en-fi-0 with
  et-en-0 on 0:0 → “pivot+Uralic flavor” is shared; en-ru-0 with
  ru-en-0 on 0:2 → “pivot+Slavic flavor”). This reduces the
  homogeneity penalty.

If your policy forbids placing both splits of the same pair on the
same GPU, you’d move, say, en-fi-1 from 0:0 to 1:0 (or another GPU) to
avoid the VERY_BAD penalty. The optimizer’s job is exactly to search
these trade-offs.

#### Why this example is hard for the optimizer (in numbers)

- 24 tasks tied together by shared components (en, fi, uralic, slavic,
  baltic, turkic).

- Moving one en-ru-0 from 0:2 → 1:1 changes:

  - Inter-node cost for component en (more nodes now touch en),
  - Inter/intra-node cost for ru and group slavic,
  - Homogeneity on both GPUs,
  - Possibly the “ready” coverage if you displace the only ready task on a GPU,
  - Any split constraint if it collides with en-ru-1 already on 1:1.

- The objective delta depends on all those counts at once → lots of
  global coupling. That’s why naïve local moves + full recomputation
  get expensive fast.

#### Summary

- The input is compact, but the global constraints (communication
  across nodes, homogeneity per GPU, curriculum readiness, split
  rules) make the search combinatorial even at modest scales.

- The output encoding is simple (node_gpu per task + a few top-level
  launch fields), but obtaining that mapping is what costs time.

- With the optimizations (delta-scored swaps/in-place updates,
  symmetry breaking, hierarchical solve, or a CP-SAT backend with a
  warm start), the same-sized instance that could take minutes can be
  brought down to single-digit seconds, and larger “hundreds of
  languages” cases become practical.


## Minimal Manual Tasks

```
# 01-minimal-tasks.yaml
tasks:
  en_es:
    src_tgt: "en-es"
    path_src: data/en-es/train.en
    path_tgt: data/en-es/train.es
    path_valid_src: data/en-es/valid.en
    path_valid_tgt: data/en-es/valid.es

  de_es:
    src_tgt: "de-es"
    path_src: data/de-es/train.de
    path_tgt: data/de-es/train.es
    path_valid_src: data/de-es/valid.de
    path_valid_tgt: data/de-es/valid.es

# (Optional) initial global defaults the tool won’t mind carrying along
batch_type: tokens
src_seq_length: 256
tgt_seq_length: 256
```

### `corpora_schedule`: Determine weighting and curriculum for the tasks.

Computes corpus sampling weights and (optionally) curriculum start
steps from corpus sizes. First, line counts for `path_src` are fetched
(cached or via `wc`/`zcat`), then normalized and temperature-adjusted
to get weights. Oversized corpora can be split into multiple “shards”
with `stride/offset` to cap any single weight. Optionally takes the
square root of weights for curriculum and shifts
`introduce_at_training_step` so at least one task can start at step 0;
weights for AE tasks can be scaled separately.

Run once to build weights and cache counts:
```
python -m mammoth.bin.config_config \
  corpora_schedule \
  --in_config train.human.yaml \
  --out_config train.step1.yaml \
  --use_weight --use_introduce_at_training_step --temperature 1.0
```

### `set_transforms` Apply the transforms to tasks

Sets `transforms` per task, choosing between general transforms and
AE-specific transforms for same-language tasks. If `prefix` is
present, injects `<to_{tgt}>` (and optionally `<from_{src}>`) into
`src_prefix` while setting a blank `tgt_prefix` (required by
downstream tooling). If `use_src_lang_token` is requested without
`prefix`, it hard-fails to avoid silent misconfiguration.


```                    
python -m mammoth.bin.config_config \
  set_transforms \
  --in_config train.step4.yaml \
  --out_config train.step5.yaml \
  --transforms sentencepiece --ae_transforms sentencepiece
```

#### `remove_temporary_keys`: Remove any meta-parameters that are not accepted by OpenNMT.

This should always be the last step.  Deletes the `config_config` key
before saving to produce a clean final YAML consumable by the
downstream trainer, which rejects unknown keys.



```
python -m mammoth.bin.config_config \
  remove_temporary_keys \
  --in_config train.alloc.yaml \
  --out_config train.final.yaml
```

### `translation_configs`: Generate the translation yaml configs.

Reserved for generating zero-shot translation configs. When enabled,
it would construct per-direction config files using the same
stacks/transforms logic but without training/validation or
scheduling/GPU allocation. (Currently a placeholder; logs timing and
exits when `zero_shot` is false.)&#x20;

#### Toggle zero-shot on
Minimal example:
```
tasks:
  en_es: { src_tgt: "en-es" }
  de_fr: { src_tgt: "de-fr" }
```
or:
```
tasks:
  en_es: { src_tgt: "en-es" }
  de_fr: { src_tgt: "de-fr" }
zero_shot: false
```
with command:
```
python -m mammoth.bin.config_config translation_configs \
  --in_config in.yaml \
  --zero_shot true \
  --out_config out.yaml
```
The result:
```
tasks:
  en_es: { src_tgt: "en-es" }
  de_fr: { src_tgt: "de-fr" }
zero_shot: true
```
#### Turn zero-shot off (overrides any previous setting)
Command:
```
python -m mammoth.bin.config_config translation_configs \
  --in_config out.yaml \
  --zero_shot false \
  --out_config out2.yaml
```
Result:
```
tasks:
  en_es: { src_tgt: "en-es" }
  de_fr: { src_tgt: "de-fr" }
zero_shot: false
```

If a flag is omitted on the CLI, the existing value in the YAML is
preserved.  You can re-run it anytime to flip the setting without
regenerating tasks or device allocations.

### `adapter_config`: Determine the adapter configuration.

`adapter_config` is a YAML-driven step (no extra CLI flags) that
expands symbolic adapter IDs into concrete per-task assignments.  You
declare what kinds of adapters you want (language/group/full) and
where they plug in; the step fills in the exact per-task assignments.

If adapters are defined, expands abstract adapter ID spaces into
concrete lists and attaches per-task adapter selections. For
encoder/decoder separately, supports `ids ∈ {LANGUAGE, GROUP, FULL}`
which expand over observed src/tgt language sets or their groups; for
each task, it appends `[adapter_name, concrete_id]` to its side’s
adapter list. Updates the top-level adapter specs with the resolved
`ids`.

#### Minimal Example
```
# Languages and groups (either computed earlier or set manually)
config_config:
  groups:
    en: g0
    de: g0
    es: g1
    fr: g1
# You must have config_config.groups ready before adapter_config,
# because GROUP expansion depends on it. If you skip clustering,
# define the mapping manually.

# Model depth (needed elsewhere; shown for context)
enc_layers: [6]
dec_layers: [6]

# Tasks (at least one, with src_tgt)
tasks:
  en_es:
    src_tgt: "en-es"
    path_src: data/en_train.txt
    path_tgt: data/es_train.txt
  de_fr:
    src_tgt: "de-fr"
    path_src: data/de_train.txt
    path_tgt: data/fr_train.txt

# Declare adapter groups. Each group has a name and a layer_stack_index
# (which layer stack it’s meant for; an index you define).
adapters:
  encoder:
    enc_lang_adapter:
      layer_stack_index: 0
      ids: LANGUAGE   # will expand to ['de','en'] based on src langs in tasks
    enc_group_adapter:
      layer_stack_index: 2
      ids: GROUP      # will expand to ['g0','g1'] using config_config.groups
    enc_full_adapter:
      layer_stack_index: 5
      ids: FULL       # will become ['full']
  decoder:
    dec_lang_adapter:
      layer_stack_index: 0
      ids: LANGUAGE   # expands using tgt langs
    dec_group_adapter:
      layer_stack_index: 2
      ids: GROUP
```
Command
```
python -m mammoth.bin.config_config adapter_config \
  --in_config in.yaml \
  --out_config out.yaml
```

The tool expands the tasks by replacing the files with adapters:
```
tasks:
  en_es:
    src_tgt: "en-es"
    path_src: data/en_train.txt
    path_tgt: data/es_train.txt
    adapters:
      encoder:
        - [enc_lang_adapter, en]
        - [enc_group_adapter, g0]
        - [enc_full_adapter, full]
      decoder:
        - [dec_lang_adapter, es]
        - [dec_group_adapter, g1]
  de_fr:
    src_tgt: "de-fr"
    path_src: data/de_train.txt
    path_tgt: data/fr_train.txt
    adapters:
      encoder:
        - [enc_lang_adapter, de]
        - [enc_group_adapter, g0]
        - [enc_full_adapter, full]
      decoder:
        - [dec_lang_adapter, fr]
        - [dec_group_adapter, g1]
```

Notes: `adapter_config` adds per-task adapters but does not change
layers.  The tool also replaces the symbolic ids (`LANGUAGE`,`GROUP`, and
`FULL`) with the expanded lists:

```
adapters:
  encoder:
    enc_lang_adapter:
      layer_stack_index: 0
      ids: [de, en]
    enc_group_adapter:
      layer_stack_index: 2
      ids: [g0, g1]
    enc_full_adapter:
      layer_stack_index: 5
      ids: [full]
  decoder:
    dec_lang_adapter:
      layer_stack_index: 0
      ids: [es, fr]
    dec_group_adapter:
      layer_stack_index: 2
      ids: [g0, g1]
```
Notes: `adapter_config` assumes your model code will read
`task.adapters` and the global `adapters` registry (with
`layer_stack_index`) to wire things.

## Other tasks
### `extra_cpu`

Turns a multi-GPU config into a single-CPU setup by deleting GPU/world-size fields and per-task `node_gpu` assignments, setting `n_nodes=1`. Useful for quick local debugging.&#x20;

### `extra_fully_shared_hack`

Transforms the config for a fully shared decoder “all-language” setup. Ensures a `prefix` transform exists (inserting before a trailing `filtertoolong` if needed) and sets `src_prefix` to route decoding. Forces `dec_sharing_group=['full']` and overwrites each task’s `src_tgt` to `all-all`. Replaces both `src_vocab` and `tgt_vocab` with a single joint vocab path.&#x20;

### `extra_copy_gpu_assignment`

Copies GPU assignments and cluster-wide device metadata from another config with identical `tasks`. Verifies task key sets match, then clones `node_gpu`, `n_nodes`, `world_size`, and `gpu_ranks`—useful to keep allocation stable across otherwise different configs.&#x20;

## Command line overrides

Some parameters can also be given on the command line.  If a value is
given both in the input yaml and on the command line, the command line
takes precedence.

## Review and Analysis of the Program

If your corpus-size cache is cold, `corpora_schedule` can be slow once
(it line-counts big files), but after the cache warms,
`allocate_devices` dominates.  In steady-state runs the slowest
command is `allocate_devices` because it calls an external optimizer
(`optimize_gpu_assignment`) that solves a combinatorial packing
problem under constraints.

Clingo-style ASP-sketch:
```
% Facts
task(i1). ...
ready(i1).
group(i1,gA).
slot(gpu1,1..K).
gpu(gpu1). ...

% Choice: assign exactly one slot to each task
1 { assign(I,G,K) : slot(G,K) } 1 :- task(I).

% Slot capacity
:- assign(I1,G,K), assign(I2,G,K), I1 != I2.

% Ready coverage (hard)
covered(G) :- assign(I,G,_), ready(I).
:- gpu(G), not covered(G).

% Group usage
use(G,H) :- assign(I,G,_), group(I,H).

% Optimization (minimize group sprawl; you can add penalties for c(I,G))
#minimize { 1@1, use(G,H) }.
```

This reproduces the main constraints (capacity, single assignment,
ready-coverage) and lets you steer solutions with minimize
statements. It’s a good fit if you like declarative encodings and
iterating on “soft” preferences.

#### When ASP might be competitive

Many qualitative, lexicographic preferences (tiered #minimize with
dozens of soft rules) where you want guaranteed optimal stable models
under complex priorities. ASP can express this very naturally and
sometimes finds good solutions fast for highly logical preference
stacks.

Heavy model churn (you frequently toggle rules/constraints during
design). ASP can be quicker to iterate declaratively, but not
necessarily faster at runtime once the instance size grows.

Keep an ASP encoding as a reference/prototyping tool for exploring new
soft constraints; once you like the behavior, translate the final set
into MIP terms.  A MIP/PB model solved by a modern integer optimizer
(e.g., OR-Tools CP-SAT or a commercial MILP) will almost always run
faster than ASP on real-sized instances.

####

Short answer: in steady-state runs the slowest command is **`allocate_devices`** because it calls an external optimizer (`optimize_gpu_assignment`) that solves a combinatorial packing problem under constraints; if your corpus-size cache is cold, **`corpora_schedule`** can be slow once (it line-counts big files), but after the cache warms, `allocate_devices` dominates.  &#x20;

---

### `allocate_devices`: line-by-line walkthrough + where time goes

1. **Read config & overrides.** Pull `n_nodes`, `n_gpus_per_node`, `n_slots_per_gpu` from CLI or YAML. Cheap.&#x20;
2. **Collect instances to place.** For every task, parse `src_tgt`, read any `offset` (from corpus “splits”), detect whether it’s “ready to start” (`introduce_at_training_step == 0`). Build:
   • `lang_pairs` = list of `(src, tgt, offset)` to place,
   • `lps_ready_to_start` = subset ready at step 0,
   • `lp_to_key` = map from `(src,tgt,offset)` → the concrete task keys (because multiple tasks can share the same tuple after splitting). O(#tasks).&#x20;
3. **Derive missing counts.** If only `n_slots_per_gpu` or `n_nodes` is given, compute the other so total slots ≥ #tasks; compute `n_gpus_tot = n_nodes * n_gpus_per_node`. O(1).&#x20;
4. **Curriculum fix-up.** If fewer *ready* tasks than GPUs, lower some `introduce_at_training_step` values so that at least `n_gpus_tot` tasks can start immediately; rebuild `lps_ready_to_start`. O(#tasks) and trivial.&#x20;
5. **Optimization (the slow part).**
   • If single GPU, assign everything to `0:0`.
   • Else call **`optimize_gpu_assignment`** with the grid shape, the list of items (`lang_pairs`), the language→group mapping, the “ready” subset, and an optional **`time_budget_s`**. This function returns a mapping {gpu\_slot → lang\_pair}. This is where the heavy combinatorial search happens; runtime is explicitly time-budgeted.&#x20;
6. **Write back the solution.** For every assigned `lang_pair`, pop one concrete task key from `lp_to_key` and set `node_gpu`. Assert everything got placed. Then set `n_nodes`, `world_size`, `gpu_ranks`. Linear in #tasks.&#x20;
7. **Per-device “starts at 0” safety pass.** For each device, find the minimum `introduce_at_training_step` among its tasks, subtract it from that device’s tasks so every GPU has some task starting at 0. Linear in #tasks.&#x20;

**Why it’s slow.** Steps 1–4 and 6–7 are linear book-keeping. Step 5 is a **constrained assignment/packing** with coupling terms (grouping, “must have a ready task per GPU”, multiple slots per GPU, and duplicated `(src,tgt,offset)` tuples). The search space grows superlinearly with #tasks and #slots; the code even exposes `--time_budget_s`, confirming the optimizer runs until a time cap. That dominates wall-clock time.&#x20;

> Note on the other “sometimes slow” command: `corpora_schedule` counts lines using `wc`/`zcat` for every source file that’s not cached—IO-bound and costly the first time; after `./corpora_length_cache` fills, it’s fast.&#x20;

### Can we formulate `allocate_devices` as MILP?

Yes. A compact MILP (works with OR-Tools/Gurobi/CP-SAT) for **one-shot static assignment**:

**Sets & data**

* Tasks $i \in \mathcal{I}$ (each with group $g(i)$ and readiness $r_i\in\{0,1\}$).
* GPUs $u \in \mathcal{U}$ with $K$ slots each (total slots $|\mathcal{U}|K \ge |\mathcal{I}|$).
* Optional **affinity cost** $c_{i,u}$ (e.g., prefer placing same groups together or keep certain groups apart).

**Vars**

* $x_{i,u,k} \in \{0,1\}$: task $i$ uses slot $k$ on GPU $u$.
* $R_u \in \{0,1\}$: GPU $u$ has at least one ready task.
* $y_{u,h} \in \{0,1\}$: GPU $u$ uses group $h$ (helps cluster groups).

**Constraints**

1. **Assign each task once**:

   $$
   \sum_{u\in\mathcal{U}}\sum_{k=1}^K x_{i,u,k} = 1 \quad \forall i.
   $$
2. **Slot capacity**:

   $$
   \sum_{i\in\mathcal{I}} x_{i,u,k} \le 1 \quad \forall u, k.
   $$
3. **Ready-coverage per GPU (soft or hard)**:

   $$
   R_u \le \sum_{i,k} r_i\,x_{i,u,k} \quad \forall u;\quad R_u \in \{0,1\}.
   $$

   Use as a hard requirement ($R_u=1$) or reward it in the objective.
4. **Group-usage indicator**:

   $$
   x_{i,u,k} \le y_{u,g(i)} \quad \forall i,u,k.
   $$

**Objective (example, tunable)**

Maximize coverage of ready tasks and “tight” grouping, while
minimizing placement cost:

$$
\max\; \lambda_1\sum_u R_u\;-\;\lambda_2\sum_{u,h} y_{u,h}\;-\;\sum_{i,u,k} c_{i,u}\,x_{i,u,k}.
$$

* The $y$ term penalizes “group sprawl” (fewer distinct groups per
  GPU).

* $c_{i,u}$ can encode other heuristics (e.g., keep related languages
  on the same node).

This MILP returns an assignment in seconds to minutes for hundreds of
tasks (depends on $K,|\mathcal{U}|$ and whether you include pairwise
terms). For even richer “pairwise same-GPU bonuses” you can introduce
$z_{i,j,u}$ (1 if $i$ and $j$ co-reside on $u$) with standard
linearization, but that’s $O(|\mathcal{I}|^2|\mathcal{U}|)$ and can
blow up—use sparingly.


