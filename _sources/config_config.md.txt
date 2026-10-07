# Config-config Tool

A meta-configuration tool, or config generator for MAMMOTH.

The MAMMOTH configuration options have become unwieldy as complexity has increased.
Especially the introduction of LayerStacks (the mechanism for dividing encoders and decoders into several subcomponents with different parameter sharing) and Adapters has made specifying parameters on the command line into a total nightmare, and even writing yaml configs by hand is cumbersome.
Other functionality that is cumbersome to specify by hand includes:

  - Node and GPU assignments for massively multilingual models,
  - Weights and curricula (starting step for specific tasks),
  - Parameter sharing groups based on language similarity.

To ease the creation of configs, the config-config tool reads in a human-writable configuration template, and computes the specific values expected by OpenNMT, writing them out as a less-readable yaml file.

## Command
```bash
mammoth_config_config config_all --in_config input.yaml --out_config output.yaml
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

As a practical example, let's say your corpus contains the following files:
```
+ afr-eng
| + train.src.gz
| + train.trg.gz
| + valid.src.gz
| + valid.trg.gz
+ ben-eng
| + train.src.gz
| + train.trg.gz
| + valid.src.gz
| + valid.trg.gz
+ eng-urd
  + train.src.gz
  + train.trg.gz
| + valid.src.gz
| + valid.trg.gz

```

For example, there is data for Bengali-to-English translation in `ben-eng/train.src.gz` (English side) and `ben-eng/train.trg.gz` (Bengali side).
You want to use the data symmetrically for both ben-to-eng and eng-to-ben directions.
For the first, `{lang_pair}` and `{sorted_pair}` are the same.
For the second, `{lang_pair}` is "eng-ben", but `{sorted_pair}` is "ben-eng".
In order to use the files in the correct order, you should use the template `{sorted_pair}/train.{side_a}.gz` for the source template, and `{sorted_pair}/train.{side_b}.gz` for the target template.

| task       | lang_pair | sorted_pair | side_a | side_b | src_path (template)             | src_path (filled in)     |
| ---------- | --------- | ----------- | ------ | ------ | --------                        | ----                     |
| ben-to-eng | ben-eng   | ben-eng     | src    | trg    | {sorted_pair}/train.{side_a}.gz | ben-eng/train.src.gz     |
| eng-to-ben | eng-ben   | ben-eng     | trg    | src    | {sorted_pair}/train.{side_a}.gz | ben-eng/train.trg.gz     |

#### `valid_src_path` and `valid_tgt_path`

Path templates for validation sets.
The path templates can contain the same variables as `src_path` and `tgt_path`.

For the corpus in the example above, `valid_src_path` should be set to `{sorted_pair}/valid.{side_a}.gz`.

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

#### `temperature`

Temperature specified in inverted form (1/T)
Setting temperature to 1.0 results in the empirical distribution, i.e. tasks are sampled according to their unweighted corpus size.
Setting temperature to 0.0 results in the uniform distribution, i.e. all tasks are equally likely.

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
(TODO: temporarily disabled)

### `use_src_lang_token`

Only has an effect if the `prefix` transform is used.
If `use_src_lang_token` is unset or False, then only a target language token `<to_{tgt}>` is prefixed to the source.
If `use_src_lang_token` is True, then a source language token is also prefixed: `<from_{src}> <to_{tgt}>`.

#### `transforms` and `ae_transforms`

A list of transforms, for translation tasks and autoencoder tasks, respectively.
Use transforms to apply subword segmentation, e.g. using `sentencepiece`, and `denoising` noise for autoencoder.
Both of these may change the sequence length, necessitating a `filtertoolong` transform.
Use the `prefix` transform to add task selection tokens in front of the source sentence.

A typical configuration for a fully shared architecture with task selection token:

```yaml
  transforms:
    - sentencepiece
    - prefix
    - filtertoolong
  ae_transforms:
    - sentencepiece
    - prefix
    - filtertoolong
    - denoising
```

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

For example, this configuration creates

  - a two-part encoder beginning with 3 groupwise shared layers followed by 3 fully shared layers,
  - a three-part "hamburger" decoder beginning with 2 language-specific layers, then 3 fully shared layers, and finally 1 more language specific layer.

```yaml
  enc_sharing_groups:
    - GROUP
    - FULL
  dec_sharing_groups:
    - LANGUAGE
    - FULL
    - LANGUAGE

enc_layers: [3, 3]
dec_layers: [2, 3, 1]
```

#### `n_nodes` and `n_gpus_per_node`

The number of nodes and GPUs, for assignment of tasks to devices.
Note that you also need to separately specify this information to slurm.

#### Other top-level keys than `config_config`

##### Adapters

```yaml
adapters:
  encoder:
    {adapter_name}:
      adapter_type: lora
      layer_stack_index: 0
      layers: [0, 1, 2]
      hidden_dim: 8
      ids: LANGUAGE
    {adapter_name}:
      adapter_type: lora
      layer_stack_index: 1
      layers: [0, 1, 2]
      hidden_dim: 8
      ids: LANGUAGE
  decoder:
    {adapter_name}:
      adapter_type: ff
      layer_stack_index: 0
      layers: [0, 1]
      hidden_dim: 16
      ids: LANGUAGE
```

###### Adapter types

The keys `adapters.encoder.{adapter_name}.adapter_type` and `adapters.decoder.{adapter_name}.adapter_type` take one of 2 values:

  - `lora`: LoRA adaptation wraps the feedforward sublayer of existing Transformer layers.
  - `ff`: A separate feedforward adapter layer injected after the feedforward sublayer of existing Transformer layers. The adapter layer has its own norms.

###### Adapter size and location

The keys `adapters.encoder.{adapter_name}.layer_stack_index` and `adapters.decoder.{adapter_name}.layer_stack_index`
specify the index (counting from 0) of the LayerStack in which the adapters should be placed.

The keys `adapters.encoder.{adapter_name}.layers` and `adapters.decoder.{adapter_name}.layers`
specify the index of the layers (counting from 0) *within* the LayerStack. Note that each new LayerStack restarts the count from zero.

The keys `adapters.encoder.{adapter_name}.hidden_dim` and `adapters.decoder.{adapter_name}.hidden_dim`
specify the hidden (bottleneck) size of the adapter.

###### Parameter sharing in adapters

The keys `adapters.encoder.{adapter_name}.ids` and `adapters.decoder.{adapter_name}.ids` take one of 3 values:

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

### Stages

The tool runs in multiple stages. The meta-stage `config_all` runs all of the stages in order.

It is possible to run the steps individually, by feeding in the output of the previous step as the input of the next step.
This allows more control:

  - Skipping unnecessary steps.
  - Overriding what a particular step does by specifying its output manually.

#### `complete_language_pairs`

Determines which language pairs have data.
The languages to consider as candidates are determined from the vocabulary keys.

#### `corpora_schedule`

Determine weighting and curriculum for the tasks.

Note that the first time this step is run for a particular corpus, the lines in the corpus are counted.
This can take a long time for large corpora.
The line counts are cached in the file `./corpora_length_cache`, keyed by the path of the corpus file.

Reusing the same corpus files (without moving, copying, or transforming) in similar training runs will be fast, due to the use of the cached line counts.
This is one of the benefits of using transforms instead of applying subword segmentation in preprocessing.

#### `cluster_languages`

Determine language groups by clustering.
This step can be easily skipped by leaving the `distance_matrix` unset.
If the step is skipped, you should define the `config_config.groups` dict in the input yaml.

#### `sharing_groups`

Apply the parameter sharing groups to tasks.

#### `set_transforms`

Apply the transforms to tasks.

#### `allocate_devices`

Allocate tasks to nodes and gpus.
A local search procedure is used, taking into account parameter sharing groups and tasks delayed by curriculum weighting.

#### `adapter_config`

Determine the adapter configuration.

#### `translation_configs`

Generate the translation yaml configs.

#### `remove_temporary_keys`

Remove any meta-parameters that are not accepted by OpenNMT. This should always be the last step.

#### `config_all`

Meta-stage to run all of the stages above in order.

#### `extra_cpu`

Modifies a GPU config to run on a single CPU, for local smoketesting.
Deletes `gpu_ranks`, `world_size`, and all `node_gpu`. Sets `n_nodes` to 1.

This step is not included in `config_all`. Run it separately if needed.

#### `extra_fully_shared_hack`

Modifies config to use the "all" language hack for a truly fully shared decoder, including embeddings.
Forces the `prefix` transform to use a language selection token.
Sets the `dec_sharing_group` to `['full']`, i.e. a single fully shared LayerStack.
Overrides all the language pair definitions `src_tgt` with `all-all`.
Sets the `src_vocab` and `tgt_vocab` to a single joint vocabulary for the `all` pseudolanguage.

This step is not included in `config_all`. Run it separately if needed.

#### `extra_copy_gpu_assignment`

Copies GPU assignment from one fully generated config to another config with exactly matching task ids.

Copying the GPU assignment is useful, if you want to ensure that tasks are distributed exactly the same in different experiments.

Also, because GPU assignment can be slow, you can save time by running it only once and then copying to other similar configs.
Note that because `allocate_devices` is part of `config_all`, you must run config-config stepwise to skip it.

This step (`extra_copy_gpu_assignment`) is not included in `config_all`. Run it separately if needed.

### Command line overrides

Some parameters can also be given on the command line.
If a value is given both in the input yaml and on the command line, the command line takes precedence.
