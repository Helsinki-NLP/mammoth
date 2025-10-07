from .utils import load_yaml

# PURPOSE: Modify config to run on a single CPU (no GPU ranks/world_size; remove node_gpu).
def extra_cpu(opts):
    cfg = opts.in_config[0]
    if "tasks" not in cfg or not isinstance(cfg["tasks"], adict):
        raise UserConfigError("Nothing to adjust: 'tasks' mapping not found in YAML.")

    # Extra step: not included in config_all
    # Modifies config to run on a single CPU
    # Remove distributed GPU-related settings.
    del opts.in_config[0]['gpu_ranks']
    del opts.in_config[0]['world_size']
    # Force a single node.
    opts.in_config[0]['n_nodes'] = 1
    # Remove per-task GPU assignment if present.
    for task_opts in opts.in_config[0]['tasks'].values():
        if 'node_gpu' in task_opts:
            del task_opts['node_gpu']
            

# PURPOSE: Force a fully shared decoder by switching tasks to 'all-all' and adding prefix; override vocabs to 'all'.
def extra_fully_shared_hack(opts):
    cfg = opts.in_config[0]
    if "tasks" not in cfg or not isinstance(cfg["tasks"], dict):
        raise UserConfigError("Nothing to hack: 'tasks' mapping not found in YAML.")
    if not getattr(opts, "joint_vocab", None):
        raise UserConfigError("Provide --joint_vocab path to a shared vocabulary file.")

    # Extra step: not included in config_all
    # Modifies config to use the "all" language hack for a fully shared decoder
    # Iterate tasks to ensure the 'prefix' transform is present and set tokens.
    for task_opts in opts.in_config[0]['tasks'].values():
        # Prefix transform to apply target language selection token
        if 'prefix' not in task_opts['transforms']:
            if task_opts['transforms'][-1] == 'filtertoolong':
                task_opts['transforms'].insert(-1, 'prefix')
            else:
                task_opts['transforms'].append('prefix')
                task_src, task_tgt = task_opts['src_tgt'].split('-')
                task_opts['src_prefix'] = f'<to_{task_src}>'
                task_opts['tgt_prefix'] = ''
        # Set fully shared decoder group.
        task_opts['dec_sharing_group'] = ['full']

        # decoder is fully shared; collapse src_tgt to a dummy 'all-all' pair.

        # src_tgt overridden with a dummy value
        task_opts['src_tgt'] = 'all-all'

    # Override vocabs: replace vocabularies with a single shared one.
    opts.in_config[0]['src_vocab'] = {'all': opts.joint_vocab}
    opts.in_config[0]['tgt_vocab'] = {'all': opts.joint_vocab}
    

# PURPOSE: Copy node_gpu/world_size/gpu_ranks from another config after verifying identical task keys.
def extra_copy_gpu_assignment(opts):
    cfg_dst = opts.in_config[0]
    cfg_src = opts.copy_from[0]
    
    if "tasks" not in cfg_dst or "tasks" not in cfg_src:
        raise UserConfigError("Both YAML files must have a 'tasks' mapping.")
    dst_keys = set(cfg_dst["tasks"].keys())
    src_keys = set(cfg_src["tasks"].keys())
    if dst_keys != src_keys:
        raise UserConfigError(f"Tasks do not match.\nMissing: {dst_keys - src_keys}\nUnused: {src_keys - dst_keys}")
    
    # Collect task key sets from current and source configs.
    tasks_in_current = set(opts.in_config[0]['tasks'].keys())
    tasks_in_source = set(opts.copy_from[0]['tasks'].keys())
    # Ensure exact match, otherwise raise an error listing differences.
    if not tasks_in_current == tasks_in_source:
        missing_tasks = tasks_in_current - tasks_in_source
        unused_tasks = tasks_in_source - tasks_in_current
        raise Exception(f'Tasks do not match. Missing tasks: {missing_tasks}, Unused tasks: {unused_tasks}')
    # Copy per-task node_gpu assignments.
    for task_key, task_opts in opts.in_config[0]['tasks'].items():
        task_opts['node_gpu'] = opts.copy_from[0]['tasks'][task_key]['node_gpu']
    # Copy global distributed settings.
    opts.in_config[0]['n_nodes'] = opts.copy_from[0]['n_nodes']
    opts.in_config[0]['world_size'] = opts.copy_from[0]['world_size']
    opts.in_config[0]['gpu_ranks'] = opts.copy_from[0]['gpu_ranks']
    
def register(subparsers):
    # extra_cpu
    p = subparsers.add_parser(
        "extra_cpu",
        help="Add CPU-only tasks or flags to tasks/rewrite config for single-CPU run (strip GPU placement).",
        description=(
            "Marks selected tasks as CPU-only or injects CPU-related flags/limits based on filters. "
            "Useful for light preprocessing tasks."),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--tasks", nargs="+", metavar="TASK", help="Task names to modify (default: all).")
    p.set_defaults(handler=extra_cpu)
    p.set_defaults(_mutates_yaml=True)

    # extra_fully_shared_hack
    p = subparsers.add_parser(
        "extra_fully_shared_hack",
        help="Force FULL decoder sharing + shared vocab across all layers (hack); add 'prefix' if missing.",
        description=(
            "Overrides encoder/decoder sharing lists to FULL for all layers. "
            "Primarily for experiments; affects config_config.encoder_sharing/decoder_sharing."),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--joint_vocab", required=True, metavar="FILE.txt",
                   help="Path to a single vocabulary to use for both src and tgt.")
    p.set_defaults(handler=extra_fully_shared_hack)
    p.set_defaults(_mutates_yaml=True)

    # extra_copy_gpu_assignment
    p = subparsers.add_parser(
        "extra_copy_gpu_assignment",
        help="Copy GPU assignment (node_gpu/world_size/gpu_ranks) from one YAML to another (tasks must match).",
        description=(
            "Copies world_size, node_gpu, and gpu_ranks from a source YAML into the target YAML. "
            "Useful to reuse a previously computed placement."),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--copy_from", required=True, type=load_yaml, metavar="FILE.yaml",
                   help="Source YAML with the desired assignments.")
    p.set_defaults(handler=extra_copy_gpu_assignment)
    p.set_defaults(_mutates_yaml=True)

    register_command_io("extra_copy_gpu_assignment": {
        "reads": ["world_size", "node_gpu", "gpu_ranks"],
        "writes": ["world_size", "node_gpu", "gpu_ranks"],
        "summary": "Copies GPU placement metadata from a source YAML into the target YAML."
    })
    register_command_io("extra_cpu": {
        "reads": ["tasks"],
        "writes": ["tasks"],
        "summary": "Marks tasks as CPU-only or injects CPU flags based on selection filters."
    })
    register_command_io("extra_fully_shared_hack", {
        "reads": ["config_config.encoder_sharing", "config_config.decoder_sharing"],
        "writes": ["config_config.encoder_sharing", "config_config.decoder_sharing"],
        "summary": "Forces FULL sharing across all layers (experimental hack)."
    })
    register_command_template_extras("extra_cpu", {
        "_notes": [
            "Set per-task hints such as cpu_only: true if your execution layer supports it."
        ],
        "tasks": {}
    })
    register_command_template_extras("extra_fully_shared_hack", {
        "_notes": [
            "Produces fully shared encoder/decoder. Useful for quick baselines."
        ],
        "enc_layers": 6,
        "dec_layers": 6,
        # Do not prefill encoder_sharing/decoder_sharing if this command generates them.
    })
    register_command_template_extras("extra_copy_gpu_assignment": {
        "_notes": [
            "Provide a static mapping; useful to reproduce an assignment.",
            "Example for 2 nodes × 4 GPUs each."
        ],
        "world_size": 8,
        "gpu_ranks": [0,1,2,3,4,5,6,7],
        "node_gpu": {
            "node0": [0,1,2,3],
            "node1": [0,1,2,3]
        }
    })
    
