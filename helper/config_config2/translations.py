from .utils import load_yaml

# SUMMARY: remove_temporary_keys
# PURPOSE: Remove the config_config section before writing final YAML (OpenNMT dislikes extra keys).
# PUT THIS IN: translations.py
def remove_temporary_keys(opts):
    # When reaching the end of the config-config, any excessive keys are
    # dropped before saving the yaml (OpenNMT doesn't like extra keys).
    # Delete the 'config_config' section so only training-relevant keys remain.
    del opts.in_config[0]['config_config']




# SUMMARY: translation_configs
# PURPOSE: Placeholder for generating zero-shot translation configs; currently does nothing unless zero_shot is set.
# PUT THIS IN: translations.py
def translation_configs(opts):
    if getattr(opts, "yaml_help", False):
        print_command_yaml_help("translation_configs")
        
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    coalesce(opts, cc_opts, "zero_shot", required=False, default=False, type_desc="boolean")

    # Decide whether to generate zero-shot configs.
    zero_shot = opts.zero_shot if opts.zero_shot else cc_opts.get('zero_shot', False)
    if not zero_shot:
        # Nothing to do if zero-shot disabled.
        return
    
    # NOTE: Implementation is intentionally left as TODO; comments outline intended design.

    # src_subword_model = opts.in_config[0].get('src_subword_model', None)
    # tgt_subword_model = opts.in_config[0].get('tgt_subword_model', None)
    # use_src_lang_token = cc_opts.get('use_src_lang_token', False)

    # TODO: create zero-shot tasks using the same template as for the supervised tasks, except that:
    # - no training set or validation set will be defined
    # - no weighting/curriculum
    # - no GPU allocation.
    # However, these 3 are needed: sharing_groups, set_transforms, adapter_config.
    # Because it would be nice to be able to add zero-shot tasks as a final extra step
    # without completely regenerating the entire training config,
    # these 3 should be modified to be rerunnable for a subset of tasks.

    duration = time.time() - start
    logger.info(f'step took {duration} s')


# SUMMARY: _write_translation_config
# PURPOSE: Write a single per-direction translation YAML with stacks, transforms, and optional prefixes.
# PUT THIS IN: translations.py (private helper)
def _write_translation_config(
    src_lang,
    tgt_lang,
    src_stack,
    tgt_stack,
    transforms,
    src_subword_model,
    tgt_subword_model,
    supervision,
    translation_config_dir,
    use_src_lang_token,
):
    # specify on command line: --model, --src
    # Build a dictionary describing this translation configuration.
    result = {
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'stack': {'encoder': src_stack, 'decoder': tgt_stack}
    }
    # If transforms are provided, include them and maybe language prefix tokens.
    if transforms:
        result['transforms'] = transforms
        if 'prefix' in transforms:
            if use_src_lang_token:
                prefix = f'<from_{src_lang}> <to_{tgt_lang}>'
            else:
                prefix = f'<to_{tgt_lang}>'
            result['src_prefix'] = prefix
            result['tgt_prefix'] = ''
    # Optionally include subword model paths.
    if src_subword_model:
        result['src_subword_model'] = src_subword_model
    if tgt_subword_model:
        result['tgt_subword_model'] = tgt_subword_model
    # Build the output file path and write YAML to disk.
    translation_config_path = f'{translation_config_dir}/trans.{supervision}.{src_lang}-{tgt_lang}.yaml'
    with open(translation_config_path, 'w') as fout:
        serialized = yaml.safe_dump(result, default_flow_style=False, allow_unicode=True)
        print(serialized, file=fout)


def register(subparsers):
    # translation_configs (placeholder)
    p = subparsers.add_parser(
        "translation_configs",
        help="Generate zero-shot translation configs (placeholder).",
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--zero_shot", action="store_true", help="Enable zero-shot config generation.")
    p.set_defaults(handler=translation_configs)

    # remove_temporary_keys
    p = subparsers.add_parser(
        "remove_temporary_keys",
        help="Remove transient keys (e.g., config_config) before saving.",
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--yaml-help", action="store_true",
                   help="Show YAML keys this command reads/writes and exit.")
    p.set_defaults(handler=remove_temporary_keys)


from .schema import print_schema, COMMAND_IO

def _yaml_help_for_command(cmd):
    keys = ( COMMAND_IO.get(cmd, {}).get("reads", []) +
             COMMAND_IO.get(cmd, {}).get("writes", []) )
    print_schema(sorted(set(keys)))

    




        
