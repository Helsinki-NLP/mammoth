# Counting Parameters

In a MAMMOTH model, tasks share components, so "how big is the model?" has several answers: the whole checkpoint, what one task needs, or what one task owns alone. `count_params` reads a checkpoint directory and reports all three. It runs on CPU, needs no GPU and no config file, and reads only the checkpoint files.

```bash
python -m mammoth.scripts.count_params --checkpoint-dir path/to/checkpoints
```

It uses the `best` checkpoint if there is one, otherwise the latest step. Optimizer files (`*_optim.pt`) are ignored.

## Per task (default)

```text
TOTAL parameters (each component once):     151,191,552  (  151.19M)
Components: 9   Tasks: 2

task                         full                 exclusive                    shared
eng-spa       93,233,152  (   93.23M)       57,958,400  (   57.96M)       35,274,752  (   35.27M)
eng-fra       93,233,152  (   93.23M)       57,958,400  (   57.96M)       35,274,752  (   35.27M)
```

- **TOTAL**: every parameter in the checkpoint, each component counted once.
- **full**: every parameter the task's forward pass uses. This is what you load to translate this one direction (e.g. with [`mammoth_serve`](serving.md)). Shared components are counted in full.
- **exclusive**: parameters used by this task and no other task.
- **shared**: `full - exclusive`.

Here `eng-spa` and `eng-fra` share the English source embedding and the encoder, and each has its own decoder.

## Per component

```bash
python -m mammoth.scripts.count_params --checkpoint-dir path/to/checkpoints --by component -v
```

```text
  decoder layer stacks: 2 component(s),      50,380,800  (   50.38M)   33.3%
    decoder_0_fra             25,190,400  (   25.19M)   16.7%  used by 1 task(s): eng-fra
    decoder_0_spa             25,190,400  (   25.19M)   16.7%  used by 1 task(s): eng-spa
  ...
  encoder layer stacks: 1 component(s),      18,890,752  (   18.89M)   12.5%
    encoder_0_all             18,890,752  (   18.89M)   12.5%  used by 2 task(s): eng-spa, eng-fra
  ...
  sum:     151,191,552  (  151.19M)
```

Components are grouped by kind (layer stacks, embeddings, wrappers, adapters). Each row shows the share of the total and how many tasks use the component. `-v` also lists those tasks. The names are the ones used by [`freeze_components` / `train_components`](freezing_components.md).

## Options

| Option | Effect |
|---|---|
| `--by {task,component}` | One row per task (default) or one row per component. |
| `--task TASK_ID` | Report only this task. With `--by component`, list only the components it uses. |
| `-v`, `--verbose` | With `--by component`, list the tasks that use each component. |
| `--json FILE` | Also write both views to a JSON file. |
