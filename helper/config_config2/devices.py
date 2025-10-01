from .utils import load_yaml
from .schema import print_command_yaml_help

# PURPOSE: Compute node:gpu assignments for tasks; ensure curriculum allows all devices to start; set world_size/ranks.
def allocate_devices(opts):
     
     if getattr(opts, "yaml_help", False):
          print_command_yaml_help("allocate_devices")
          return
     
     start = time.time()

     cc_opts = opts.in_config[0]['config_config']
     cc = cc_opts

     # Require groups mapping
     if "groups" not in cc or not isinstance(cc["groups"], dict):
          raise UserConfigError("Missing 'config_config.groups' mapping. Run cluster_languages or define it in YAML.")

     # Require n_gpus_per_node (CLI overrides YAML)
     ngpn = coalesce(opts, cc, "n_gpus_per_node", required=True, type_desc="integer")

     # Must give either n_nodes or n_slots_per_gpu
     n_nodes = getattr(opts, "n_nodes", None) or cc.get("n_nodes")
     n_slots = getattr(opts, "n_slots_per_gpu", None) or cc.get("n_slots_per_gpu")
     if n_nodes is None and n_slots is None:
          raise UserConfigError("Provide --n_nodes or --n_slots_per_gpu (or set config_config.n_nodes / .n_slots_per_gpu).")

     # Optional time budget/log name
     coalesce(opts, cc, "time_budget_s", required=False, default=None, type_desc="integer (seconds)")
     coalesce(opts, cc, "log_name", required=False, default=None, type_desc="string")


     # Resolve node/gpu/slot parameters.
     n_nodes = opts.n_nodes if opts.n_nodes else cc_opts.get('n_nodes', None)
     n_gpus_per_node = opts.n_gpus_per_node if opts.n_gpus_per_node else cc_opts['n_gpus_per_node']
     n_slots_per_gpu = opts.n_slots_per_gpu if opts.n_slots_per_gpu else cc_opts.get('n_slots_per_gpu', None)

     # Build the list of (src, tgt, offset) triples and mark which are ready at step 0.
     lang_pairs = []
     lps_ready_to_start = []
     lp_to_key = defaultdict(list)
     for key, tasks_config in opts.in_config[0]['tasks'].items():
         src_lang, tgt_lang = tasks_config['src_tgt'].split('-')
         offset = tasks_config.get('offset', 0)
         ready_to_start = tasks_config.get('introduce_at_training_step', 0) == 0

         lang_pairs.append((src_lang, tgt_lang, offset))
         if ready_to_start:
             lps_ready_to_start.append((src_lang, tgt_lang, offset))
         lp_to_key[(src_lang, tgt_lang, offset)].append(key)

     # Either n_nodes or n_slots_per_gpu must be provided (to compute the other).
     if n_nodes is None and n_slots_per_gpu is None:
         raise Exception('You must specify either n_nodes or n_slots_per_gpu')
     
     if n_nodes is None:
         n_slots_per_node = n_gpus_per_node * n_slots_per_gpu
         n_nodes = int(np.ceil(len(lang_pairs) / n_slots_per_node))
     n_gpus_tot = n_nodes * n_gpus_per_node
     
     if n_slots_per_gpu is None:
         n_slots_per_gpu = int(np.ceil(len(lang_pairs) / n_gpus_tot))
     logger.info(f'n_nodes:          {n_nodes}')
     logger.info(f'n_gpus_per_node:  {n_gpus_per_node}')
     logger.info(f'n_slots_per_gpu:  {n_slots_per_gpu}')
     logger.info(f'total slots:      {n_nodes * n_gpus_per_node * n_slots_per_gpu}')
     logger.info(f'lang_pairs:       {len(lang_pairs)}')

     # If too few "ready" tasks to occupy all GPUs, shift curricula so enough start at 0.
     if len(lps_ready_to_start) < (n_nodes * n_gpus_per_node):
         iats = [corpus.get('introduce_at_training_step', 0) for _, corpus in opts.in_config[0]['tasks'].items()]
         iats = sorted(iats)
         iats_at_last_gpu = iats[n_nodes * n_gpus_per_node]
         lps_ready_to_start = []
         for cname, corpus in opts.in_config[0]['tasks'].items():
             src_lang, tgt_lang = corpus['src_tgt'].split('-')
             offset = corpus.get('offset', 0)
             if 'introduce_at_training_step' not in corpus:
                 lps_ready_to_start.append((src_lang, tgt_lang, offset))
                 continue
             adjusted = max(0, corpus.get('introduce_at_training_step', 0) - iats_at_last_gpu)
             corpus['introduce_at_training_step'] = adjusted
             if adjusted == 0:
                 lps_ready_to_start.append((src_lang, tgt_lang, offset))

     # Trivial assignment when only one GPU total.
     if n_gpus_tot < 2:
         print('Assigning all tasks to 0:0')
         for key in opts.in_config[0]['tasks']:
             opts.in_config[0]['tasks'][key]['node_gpu'] = '0:0'
     else:
         # Run assignment optimizer to place tasks on node:gpu slots.
         assignment = optimize_gpu_assignment(
             n_nodes=n_nodes,
             n_gpus_per_node=n_gpus_per_node,
             n_slots_per_gpu=n_slots_per_gpu,
             lang_pairs=lang_pairs,
             lang_to_group_mapping=cc_opts['groups'],
             lps_ready_to_start=lps_ready_to_start,
             log_name=opts.log_name,
             time_budget_s=opts.time_budget_s,
         )

         # Write assignments back into the tasks.
         for gpu_slot, lp in assignment.items():
             if lp is None:
                 continue
             key = lp_to_key[lp].pop()
             opts.in_config[0]['tasks'][key]['node_gpu'] = f'{gpu_slot.node}:{gpu_slot.gpu}'
             
         # Sanity-check that we consumed all tasks.
         total_remaining = 0
         for lp, keys in lp_to_key.items():
             if len(keys) > 0:
                 print(f'{lp} remaining keys: {keys}')
             total_remaining += len(keys)
         assert total_remaining == 0

     # Ensure every task has a node_gpu assigned.
     for cname, corpus in opts.in_config[0]['tasks'].items():
         assert 'node_gpu' in corpus, f'{cname} not assigned to node_gpu: {corpus}'

     # Store global distributed training settings.
     opts.in_config[0]['n_nodes'] = n_nodes
     opts.in_config[0]['world_size'] = n_gpus_tot
     opts.in_config[0]['gpu_ranks'] = list(range(n_gpus_per_node))

     # Normalize introduce_at_training_step so each device has at least one task ready at 0.
     train_steps = opts.in_config[0].get('train_steps', 100_000)
     min_introduce_at_training_step = defaultdict(lambda: train_steps)
     for cname, corpus in opts.in_config[0]['tasks'].items():
         if 'introduce_at_training_step' not in corpus:
             continue
         min_introduce_at_training_step[corpus['node_gpu']] = min(
             corpus['introduce_at_training_step'],
             min_introduce_at_training_step[corpus['node_gpu']]
         )
     for cname, corpus in opts.in_config[0]['tasks'].items():
         if 'introduce_at_training_step' not in corpus:
             continue
         adjust = min_introduce_at_training_step[corpus['node_gpu']]
         if adjust > 0:
             logger.warning(f'Reducing introduce_at_training_step of {cname} by {adjust}')
             corpus['introduce_at_training_step'] -= adjust

     duration = time.time() - start
     logger.info(f'step took {duration} s')

from .schema import print_schema, COMMAND_IO

def _yaml_help_for_command(cmd):
    keys = ( COMMAND_IO.get(cmd, {}).get("reads", []) +
             COMMAND_IO.get(cmd, {}).get("writes", []) )
    print_schema(sorted(set(keys)))


def register(subparsers):
    p = subparsers.add_parser(
        "allocate_devices",
        help="Assign tasks to node:gpu slots; set n_nodes/world_size/gpu_ranks.",
        description=(
            "Provide --n_nodes or --n_slots_per_gpu (the other is inferred). "
            "Requires config_config.n_gpus_per_node and a 'groups' mapping."
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--n_nodes", type=int, metavar="INT",
                   help="Number of compute nodes (optional if --n_slots_per_gpu is given).")
    p.add_argument("--n_gpus_per_node", type=int, metavar="INT",
                   help="GPUs per node (overrides YAML config_config.n_gpus_per_node).")
    p.add_argument("--n_slots_per_gpu", type=int, metavar="INT",
                   help="Number of task slots per GPU (optional if --n_nodes is given).")
    p.add_argument("--log_name", metavar="STR", help="Assignment optimizer run name.")
    p.add_argument("--time_budget_s", type=int, metavar="SECONDS",
                   help="Time budget for GPU assignment, in seconds.")
    p.add_argument("--yaml-help", action="store_true",
                   help="Show YAML keys this command reads/writes and exit.")
    p.set_defaults(handler=allocate_devices)

