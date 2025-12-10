from .utils import load_yaml

def set_transforms(opts):
    """Assign transform lists to each task; optionally set 
    prefix tokens and validate settings."""
    
    # Start timing.
    start = time.time()

    cc  = opts.in_config[0]["config_config"]

    # Lists are optional, but if config_config.use_src_lang_token is true,
    # 'prefix' must be included in the effective transforms for any task.
    coalesce(opts, cc, "transforms", required=False, default=None, type_desc="list")
    coalesce(opts, cc, "ae_transforms", required=False, default=None, type_desc="list")

    use_src_token = cc.get("use_src_lang_token", False)
    if use_src_token:
        # We cannot inspect per-task list yet; enforce at least that base list contains 'prefix'
        base_tr = (getattr(opts, "transforms", None) or cc.get("transforms") or [])
        base_ae = (getattr(opts, "ae_transforms", None) or cc.get("ae_transforms") or [])
        if ("prefix" not in base_tr) and ("prefix" not in base_ae):
            raise UserConfigError("config_config.use_src_lang_token=true requires 'prefix' to appear in transforms or ae_transforms.")

    cc_opts = cc
    # Resolve which transform lists to use for AE vs translation.
    ae_transforms = opts.ae_transforms if opts.ae_transforms else cc_opts.get('ae_transforms', [])
    transforms = opts.transforms if opts.transforms else cc_opts.get('transforms', [])

    # Iterate tasks and assign transform lists.
    for cname, corpus in opts.in_config[0]['tasks'].items():
        src, tgt = corpus['src_tgt'].split('-')
        # Autoencoder uses AE transforms; translation uses normal transforms.
        if src == tgt:
            corpus['transforms'] = list(ae_transforms)
        else:
            corpus['transforms'] = list(transforms)

        # If the 'prefix' transform is present, set language tokens accordingly.
        if 'prefix' in corpus['transforms']:
            if cc_opts.get('use_src_lang_token', False):
                prefix = f'<from_{src}> <to_{tgt}>'
            else:
                prefix = f'<to_{tgt}>'
            corpus['src_prefix'] = prefix
            corpus['tgt_prefix'] = ''   # must exist even if unused
        else:
            # If someone asks to use src language token but no 'prefix' transform, error out.
            if cc_opts.get('use_src_lang_token', False):
                raise Exception('use_src_lang_token requires prefix transform')

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def register(subparsers):
    p = subparsers.add_parser(
        "set_transforms",
        help="Attach transform lists to tasks; set prefix tokens if 'prefix' present.",
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--transforms", action="append", metavar="NAME",
                   help="Transform(s) to use for translation tasks (use multiple times).")
    p.add_argument("--ae_transforms", action="append", metavar="NAME",
                   help="Transform(s) to use for autoencoder tasks (use multiple times).")
    p.set_defaults(handler=set_transforms)
    p.set_defaults(_mutates_yaml=True)

    register_command_io("set_transforms", {
        "reads": ["tasks", "config_config.transforms", "config_config.ae_transforms",
                  "config_config.use_src_lang_token"],
        "writes": ["tasks"],
        "summary": "Attaches transforms to tasks; enforces 'prefix' when use_src_lang_token is true.",
    })
    register_command_template_extras("set_transforms", {
        "_notes": [
            "Will set/override transforms per task.",
            "ae_transforms are used for autoencoder tasks only."
        ],
        "tasks": {},
        "config_config": {
            "transforms": [],
            "ae_transforms": [],
            "use_src_lang_token": False
        }
    })
