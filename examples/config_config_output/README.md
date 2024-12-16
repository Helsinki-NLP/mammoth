# mammoth_config_config example output

This directory contains example output of running mammoth_config_config on the template files in the examples directory.

It is *highly recommended* that you run `config_config` yourself, rather than editing configs by hand based on these examples.
The example output is included merely so that you can check whether you get the expected result.

The configs have been created for a single node with a single gpu.

Note that for `config_config.yaml` the line counts (weighting) and presence of corpus files (which tasks to create) is based on dummy data, not any real corpus.

```bash
mammoth_config_config \
    config_all \
    --in_config examples/config_config.template.yaml \
    --out_config config_config.yaml \
    --n_nodes 1 \
    --n_gpus_per_node 1
```

