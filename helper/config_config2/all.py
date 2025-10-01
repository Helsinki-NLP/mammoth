# SUMMARY: config_all
# PURPOSE: Run the full pipeline of steps in order, then log total time.
# PUT THIS IN: translations.py (or keep as orchestration in cli.py calling into modules)
def config_all(opts):
    if getattr(opts, "yaml_help", False):
        print_command_yaml_help("config_all")
        
    return
    # Start timer.
    start = time.time()

    # Minimal preflight checks before running the whole pipeline.
    cfg = opts.in_config[0]
    if "config_config" not in cfg:
        raise UserConfigError("Missing 'config_config' section in YAML.")
    if "src_vocab" not in cfg or "tgt_vocab" not in cfg:
        raise UserConfigError("Missing src_vocab or tgt_vocab in YAML.")

    # Execute every step in a fixed order to build the final config.
    complete_language_pairs(opts)
    corpora_schedule(opts)
    cluster_languages(opts)
    sharing_groups(opts)
    allocate_devices(opts)
    set_transforms(opts)
    adapter_config(opts)
    translation_configs(opts)
    remove_temporary_keys(opts)
    # Log total elapsed time.
    duration = time.time() - start
    logger.info(f'total took {duration} s')


def register(subparsers):
    # config_all (orchestration; includes knobs used by its steps)
    p = subparsers.add_parser(
        "config_all",
        help="Run the full pipeline end-to-end.",
        description=(
            "Order: complete_language_pairs → corpora_schedule → cluster_languages → sharing_groups → "
            "allocate_devices → set_transforms → adapter_config → translation_configs → remove_temporary_keys"
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")

    p.add_argument("--src_path", metavar="TEMPLATE")
    p.add_argument("--tgt_path", metavar="TEMPLATE")
    p.add_argument("--ae_path", metavar="TEMPLATE", action="append")
    p.add_argument("--valid_src_path", metavar="TEMPLATE")
    p.add_argument("--valid_tgt_path", metavar="TEMPLATE")
    p.add_argument("--autoencoder", action="store_true")
    p.add_argument("--autoencoder_validation", action="store_true")

    p.add_argument("--use_weight", action="store_true")
    p.add_argument("--ae_weight", type=float, metavar="FLOAT")
    p.add_argument("--use_introduce_at_training_step", action="store_true")
    p.add_argument("--temperature", type=float, metavar="FLOAT")

    p.add_argument("--distance_matrix", type=load_distmat_csv, metavar="FILE.csv")
    p.add_argument("--cutoff_threshold", type=float, metavar="FLOAT")
    p.add_argument("--n_groups", type=int, metavar="INT")

    p.add_argument("--enc_sharing_groups", action="append", metavar="TOKEN")
    p.add_argument("--dec_sharing_groups", action="append", metavar="TOKEN")

    p.add_argument("--n_nodes", type=int, metavar="INT")
    p.add_argument("--n_gpus_per_node", type=int, metavar="INT")
    p.add_argument("--n_slots_per_gpu", type=int, metavar="INT")
    p.add_argument("--log_name", metavar="STR")
    p.add_argument("--time_budget_s", type=int, metavar="SECONDS")

    p.add_argument("--transforms", action="append", metavar="NAME")
    p.add_argument("--ae_transforms", action="append", metavar="NAME")

    p.add_argument("--zero_shot", action="store_true")

    p.add_argument("--yaml-help", action="store_true",
                   help="Show YAML keys this command reads/writes and exit.")
    
    p.set_defaults(handler=config_all)

from .schema import print_schema, COMMAND_IO

def _yaml_help_for_command(cmd):
    keys = ( COMMAND_IO.get(cmd, {}).get("reads", []) +
             COMMAND_IO.get(cmd, {}).get("writes", []) )
    print_schema(sorted(set(keys)))


        
