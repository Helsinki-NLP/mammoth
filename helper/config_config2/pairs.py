from .utils import load_yaml

# SUMMARY: _get_langs
# PURPOSE: Return sorted lists of source and target language keys from vocab sections.
# PUT THIS IN: translations.py (private helper)
def _get_langs(opts):
    # Pull the keys (language codes) from src_vocab and tgt_vocab dicts and sort them.
    src_langs = list(sorted(opts.in_config[0]['src_vocab'].keys()))
    tgt_langs = list(sorted(opts.in_config[0]['tgt_vocab'].keys()))
    # Return as a pair.
    return src_langs, tgt_langs


# SUMMARY: complete_language_pairs
# PURPOSE: Using path templates, add supervised and (optional) autoencoder tasks for every src×tgt that exists.
# PUT THIS IN: translations.py
def complete_language_pairs(opts):
    if getattr(opts, "yaml_help", False):
        print_command_yaml_help("complete_language_pairs")
        return
    
    # Start timer.
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']

    # Ensure vocab sections exist and are dicts
    src_vocab = opts.in_config[0].get("src_vocab")
    tgt_vocab = opts.in_config[0].get("tgt_vocab")
    if not isinstance(src_vocab, dict) or not isinstance(tgt_vocab, dict):
        raise UserConfigError("src_vocab and tgt_vocab must exist in YAML and be mappings {lang: path_template}.")

    # Resolve templates (CLI overrides YAML), require train templates
    src_tpl = coalesce(opts, cc, "src_path", required=True, type_desc="path template")
    tgt_tpl = coalesce(opts, cc, "tgt_path", required=True, type_desc="path template")

    # Optional AE templates; when AE requested, ensure they resolve (they can fall back to src/tgt templates)
    ae_enabled = coalesce(opts, cc, "autoencoder", default=False, type_desc="boolean")
    if ae_enabled and getattr(opts, "ae_path", None) is None and "ae_path" not in cc:
        # fine: will fall back to src/tgt; do nothing
        pass

    # Validation templates are optional individually; users can omit them
    coalesce(opts, cc, "valid_src_path", required=False, type_desc="path template")
    coalesce(opts, cc, "valid_tgt_path", required=False, type_desc="path template")



    # Resolve path templates and booleans (CLI overrides config).
    src_path_template = opts.src_path if opts.src_path else cc_opts['src_path']
    tgt_path_template = opts.tgt_path if opts.tgt_path else cc_opts['tgt_path']
    autoencoder = opts.autoencoder if opts.autoencoder else cc_opts.get('autoencoder', False)
    autoencoder_validation = (
        opts.autoencoder_validation if opts.autoencoder_validation else cc_opts.get('autoencoder_validation', False)
    )
    # Prepare AE path templates if AE is enabled.
    if autoencoder:
        ae_path_templates = None
        if opts.ae_path:
            ae_path_templates = opts.ae_path
        elif 'ae_path' in cc_opts:
            ae_path_templates = cc_opts['ae_path']
        if isinstance(ae_path_templates, str):
            ae_path_templates = [ae_path_templates]
        if ae_path_templates:
            ae_src_path_templates = ae_path_templates
            ae_tgt_path_templates = ae_path_templates
        else:
            ae_src_path_templates = [src_path_template]
            ae_tgt_path_templates = [tgt_path_template]
    valid_src_path_template = opts.valid_src_path if opts.valid_src_path else cc_opts['valid_src_path']
    valid_tgt_path_template = opts.valid_tgt_path if opts.valid_tgt_path else cc_opts['valid_tgt_path']

    # Iterate all src×tgt combos derived from vocab sections.
    src_langs, tgt_langs = _get_langs(opts)
    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            lang_a, lang_b = sorted((src_lang, tgt_lang))
            lang_pair = f'{src_lang}-{tgt_lang}'
            sorted_pair = f'{lang_a}-{lang_b}'
            if lang_pair == sorted_pair:
                side_a = 'src'; side_b = 'trg'; lang_a = src_lang; lang_b = tgt_lang
            else:
                side_a = 'trg'; side_b = 'src'; lang_a = tgt_lang; lang_b = src_lang
            template_variables = {
                'src_lang': src_lang, 'tgt_lang': tgt_lang, 'lang_a': lang_a, 'lang_b': lang_b,
                'side_a': side_a, 'side_b': side_b, 'lang_pair': lang_pair, 'sorted_pair': sorted_pair
            }
            if src_lang == tgt_lang:
                # Autoencoder task path resolution and optional validation.
                if not autoencoder:
                    continue
                for ae_src_path_template, ae_tgt_path_template in zip(ae_src_path_templates, ae_tgt_path_templates):
                    src_path = ae_src_path_template.format(**template_variables)
                    tgt_path = ae_tgt_path_template.format(**template_variables)
                    if not autoencoder_validation:
                        valid_src_path = None; valid_tgt_path = None
                    else:
                        valid_src_path = valid_src_path_template.format(**template_variables)
                        valid_tgt_path = valid_tgt_path_template.format(**template_variables)
                    if os.path.exists(src_path) and os.path.exists(tgt_path):
                        _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path, valid_src_path, valid_tgt_path)
            else:
                # Supervised translation task path resolution.
                src_path = src_path_template.format(**template_variables)
                tgt_path = tgt_path_template.format(**template_variables)
                valid_src_path = valid_src_path_template.format(**template_variables)
                valid_tgt_path = valid_tgt_path_template.format(**template_variables)
                if os.path.exists(src_path) and os.path.exists(tgt_path):
                    _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path, valid_src_path, valid_tgt_path)
                else:
                    logger.warning(f'Paths do NOT exist, omitting language pair: {src_path} {tgt_path}')
    # Fail fast if nothing was added.
    if len(opts.in_config[0].get('tasks', [])) == 0:
        raise Exception('No language pairs were added. Check your path templates.')
    # Expand any vocab path templates that reference {src_lang}/{tgt_lang}.
    for src_lang in src_langs:
        opts.in_config[0]['src_vocab'][src_lang] = opts.in_config[0]['src_vocab'][src_lang].format(src_lang=src_lang)
    for tgt_lang in tgt_langs:
        opts.in_config[0]['tgt_vocab'][tgt_lang] = opts.in_config[0]['tgt_vocab'][tgt_lang].format(tgt_lang=tgt_lang)

    duration = time.time() - start
    logger.info(f'step took {duration} s')


    

# SUMMARY: _add_language_pair
# PURPOSE: Insert or update a single task entry with file paths and (optional) validation paths.
# PUT THIS IN: translations.py (private helper)
def _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path, valid_src_path, valid_tgt_path):
    # Ensure 'tasks' section exists in config.
    if 'tasks' not in opts.in_config[0]:
        opts.in_config[0]['tasks'] = dict()
    tasks_section = opts.in_config[0]['tasks']
    # Task key is "SRC-TGT".
    key = f'{src_lang}-{tgt_lang}'
    # Create the task dict if it's new.
    if key not in tasks_section:
        tasks_section[key] = dict()
    # Always set the language pair string and train data paths.
    tasks_section[key]['src_tgt'] = f'{src_lang}-{tgt_lang}'
    tasks_section[key]['path_src'] = src_path
    tasks_section[key]['path_tgt'] = tgt_path
    # If validation paths exist on disk, record them as well.
    if valid_src_path is not None and os.path.exists(valid_src_path):
        tasks_section[key]['path_valid_src'] = valid_src_path
        tasks_section[key]['path_valid_tgt'] = valid_tgt_path

from .schema import print_schema, COMMAND_IO

def _yaml_help_for_command(cmd):
    keys = ( COMMAND_IO.get(cmd, {}).get("reads", []) +
             COMMAND_IO.get(cmd, {}).get("writes", []) )
    print_schema(sorted(set(keys)))

def register(subparsers):
    # complete_language_pairs
    p = subparsers.add_parser(
        "complete_language_pairs",
        help="Expand tasks from path templates (and optional autoencoders).",
        description=(
            "Scans your src_vocab/tgt_vocab and builds tasks from CONFIG or CLI templates. "
            "Only adds tasks whose files exist on disk."
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--src_path", metavar="TEMPLATE",
                   help=("Source train path template. Can use variables {src_lang}, {tgt_lang},\n"
                         "{lang_a},{lang_b},{side_a},{side_b},{lang_pair}, and {sorted_pair}."))
    p.add_argument("--tgt_path", metavar="TEMPLATE",
                   help="Target train path template. Can use the same variables as --src_path.")
    p.add_argument("--ae_path", metavar="TEMPLATE", action="append",
                   help=("Monolingual (autoencoder) template(s). If multiple, they zip together.\n"
                         "Can use the variables {src_lang}, {tgt_lang}, and {sorted_pair}.\n"
                         "If unset, autoencoder pairs will use src_path and tgt_path."))
    p.add_argument("--valid_src_path", metavar="TEMPLATE",
                   help=("Validation (dev) source template (same variables).\n"
                         "Can use variables {src_lang}, {tgt_lang}, and {sorted_pair}."))
    p.add_argument("--valid_tgt_path", metavar="TEMPLATE",
                   help="Validation target template (same variables).")
    p.add_argument("--autoencoder", action="store_true",
                   help="Also create autoencoder (src==tgt) tasks.")
    p.add_argument("--autoencoder_validation", action="store_true",
                   help="Also add validation sets for autoencoder tasks.")
    p.add_argument("--yaml-help", action="store_true",
                   help="Show YAML keys this command reads/writes and exit.")
    p.set_defaults(handler=complete_language_pairs)

