from .utils import load_yaml
from .clustering import load_distmat_csv

# PURPOSE: Run the full pipeline of steps in order, then log total time.
def config_all(opts):
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

    p.set_defaults(handler=config_all)
    p.set_defaults(_mutates_yaml=True)

    register_command_io("config_all": {
        "reads": [
            # task discovery
            "src_vocab", "tgt_vocab",
            "config_config.src_path", "config_config.tgt_path",
            "config_config.ae_path", "config_config.valid_src_path", "config_config.valid_tgt_path",

            # scheduling knobs
            "config_config.temperature", "config_config.use_weight", "config_config.ae_weight",
            "config_config.use_introduce_at_training_step", "config_config.split_large_language_pairs",

            # clustering / groups
            "config_config.n_groups", "config_config.groups",

            # sharing
            "config_config.enc_layers", "config_config.dec_layers",
            "config_config.enc_sharing_groups", "config_config.dec_sharing_groups",

            # transforms
            "config_config.transforms", "config_config.ae_transforms", "config_config.use_src_lang_token",

            # devices
            "config_config.n_gpus_per_node", "config_config.n_nodes", "config_config.n_slots_per_gpu",

            # optional extras
            "config_config.zero_shot",
        ],
        "writes": [
            "tasks", "world_size", "node_gpu", "gpu_ranks",
            "config_config.encoder_sharing", "config_config.decoder_sharing",
            "adapters", "configs"
        ],
        "summary": ("End-to-end pipeline: build tasks, schedule, cluster, sharing, "
                    "transforms, device placement, (optional) adapters, and emit configs.")
    })
    register_command_template_extras("config_all", {
        "_notes": [
            "This is a compact starter for running the full pipeline.",
            "Edit paths/templates first; then adjust scheduling/transforms/devices.",
            "Keys shown here are typical inputs other commands will read."
        ],
        "src_vocab": "/data/vocabs/{lang}.txt",
        "tgt_vocab": "/data/vocabs/{lang}.txt",
        "config_config": {
            "_notes": [
                "Path templates can use {src} and {tgt}.",
                "If you also want monolingual autoencoders, set ae_path.",
                "Validation paths are optional; used only if files exist."
            ],
            "src_path": "/data/corpora/{src}-{tgt}.src",
            "tgt_path": "/data/corpora/{src}-{tgt}.tgt",
            "ae_path":  "/data/mono/{lang}.txt",
            "valid_src_path": "/data/valid/{src}-{tgt}.src",
            "valid_tgt_path": "/data/valid/{src}-{tgt}.tgt",

            "_schedule_notes": [
                "Weights: w = (size/total)^temperature.",
                "use_weight=True uses corpus sizes; set False for uniform-by-count.",
                "ae_weight applies a multiplier to AE tasks if present."
            ],
            "temperature": 1.0,
            "use_weight": True,
            "ae_weight": 0.5,
            "use_introduce_at_training_step": False,
            "split_large_language_pairs": False,

            "_groups_notes": [
                "Either set n_groups to let clustering create groups,",
                "OR provide explicit 'groups' mapping {lang: group}."
            ],
            "n_groups": 2,
            "groups": {"en": 0, "fi": 1},

            "_sharing_notes": [
                "If your model config does not define layer counts, set them here.",
                "enc_sharing_groups/dec_sharing_groups: per-layer group IDs. Often filled by 'sharing_groups'."
            ],
            "enc_layers": 6,
            "dec_layers": 6,
            "enc_sharing_groups": [],   # will be filled by 'sharing_groups' usually
            "dec_sharing_groups": [],

            "_transforms_notes": [
                "Global transform knobs; 'set_transforms' will apply to tasks.",
                "ae_transforms apply to autoencoder tasks only.",
                "use_src_lang_token adds a source-language token in inputs."
            ],
            "transforms": [],
            "ae_transforms": [],
            "use_src_lang_token": False,

            "_devices_notes": [
                "Multi-node config: set n_gpus_per_node, and either n_nodes or n_slots_per_gpu.",
                "The other value will be inferred by 'allocate_devices'."
            ],
            "n_gpus_per_node": 4,
            "n_nodes": 1,
            "n_slots_per_gpu": 1,

            "_zero_shot_notes": [
                "Zero-shot pairs (no training data); used by 'translation_configs' as eval-only."
            ],
            "zero_shot": []
        }
    })
 
