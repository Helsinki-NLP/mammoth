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

## `complete_language_pairs` - Determine which language pairs have data.

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

```
Combine this with the following task.

## `sharing_groups`: Apply the parameter sharing groups to tasks.

```    
# then derive sharing groups from those clusters
python -m mammoth.bin.config_config sharing_groups \
  --in_config 03-clustered.yaml \
  --out_config 03-sharing.yaml

python -m mammoth.bin.config_config \
  sharing_groups \
  --in_config train.step2.yaml \
  --out_config train.step3.yaml
```

## `allocate_devices`: Allocate tasks to nodes and gpus.

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

Run once to build weights and cache counts:
```
python -m mammoth.bin.config_config \
  corpora_schedule \
  --in_config train.human.yaml \
  --out_config train.step1.yaml \
  --use_weight --use_introduce_at_training_step --temperature 1.0
```

### `set_transforms`

Apply the transforms to tasks.

```                    
python -m mammoth.bin.config_config \
  set_transforms \
  --in_config train.step4.yaml \
  --out_config train.step5.yaml \
  --transforms sentencepiece --ae_transforms sentencepiece
```

#### `remove_temporary_keys`

Remove any meta-parameters that are not accepted by OpenNMT. This should always be the last step.

```
python -m mammoth.bin.config_config \
  remove_temporary_keys \
  --in_config train.alloc.yaml \
  --out_config train.final.yaml
```

### `translation_configs`

Generate the translation yaml configs.

### `adapter_config`

Determine the adapter configuration.

## Command line overrides

Some parameters can also be given on the command line.  If a value is
given both in the input yaml and on the command line, the command line
takes precedence.