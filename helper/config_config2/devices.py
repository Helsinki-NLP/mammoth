from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from .utils import UserConfigError, logger, load_yaml, register_command_io, register_command_template_extras, resolve_command_inputs

# If not already in utils.py, consider adding these tiny helpers there and import them:
def _ensure_tasks_map(doc: dict) -> dict:
    """Require doc['tasks'] to be a non-empty mapping."""
    tasks = doc.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise UserConfigError("No 'tasks' mapping found; run complete_language_pairs first or provide tasks.")
    return tasks

def _ensure_groups_map(x: Any, *, where: str) -> dict:
    """Require a mapping for language→group; fail with context."""
    if not isinstance(x, dict) or not x:
        raise UserConfigError(f"Missing or empty '{where}' mapping (language→group). "
                              "Run cluster_languages or provide groups in YAML.")
    return x

def _int_or_none(x: Any) -> int | None:
    """Return int(x) if it’s a finite integer-like, else None."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    try:
        v = int(x)
        return v
    except Exception:
        return None

# Your optimizer must already exist somewhere you import from:
# from .gpu_assign import optimize_gpu_assignment
# Here we import lazily inside the function so running --help etc. won’t fail if it’s missing.


# PURPOSE: Compute node:gpu assignments for tasks; ensure curriculum allows all devices to start; set world_size/ranks.
def allocate_devices(opts):
    """
    Inputs (via resolve_command_inputs / COMMAND_IO):
      - tasks (top-level)
      - config_config.allocate_devices.n_gpus_per_node (required)
      - config_config.allocate_devices.n_nodes | n_slots_per_gpu (at least one required)
      - config_config.allocate_devices.time_budget_s (optional)
      - config_config.allocate_devices.log_name (optional)
      - config_config.groups (required map lang→group)
     
     Writes:
      - tasks.*.node_gpu = "node:gpu"
      - world_size, n_nodes, gpu_ranks
      - Possibly adjusts tasks.*.introduce_at_training_step
    """
    # 1) Normalize/resolve inputs (moves per-command keys under config_config.allocate_devices.*; applies CLI overrides)
    inputs = resolve_command_inputs("allocate_devices", opts)
    doc = opts.in_config[0]  # normalized in place
    tasks = _ensure_tasks_map(doc)
     
    # 2) Required “groups” mapping (NOT per-command; lives under config_config.* shared space)
    #    The resolver retains legacy top-level path OR normalized path; we read via inputs.get or doc for clarity.
    groups = inputs.get("groups")
    if groups is None:
        # also accept legacy/global location if user has not normalized yet
        groups = (doc.get("config_config") or {}).get("groups")
    groups = _ensure_groups_map(groups, where="config_config.groups")

    # 3) Pull device knobs (CLI already applied by resolver)
    n_gpus_per_node = _int_or_none(inputs.get("n_gpus_per_node"))
    n_nodes         = _int_or_none(inputs.get("n_nodes"))
    n_slots_per_gpu = _int_or_none(inputs.get("n_slots_per_gpu"))
     
    if n_gpus_per_node is None or n_gpus_per_node <= 0:
        raise UserConfigError("config_config.n_gpus_per_node is required and must be a positive integer.")

    if (n_nodes is None) and (n_slots_per_gpu is None):
        raise UserConfigError("Provide --n_nodes or --n_slots_per_gpu (or set config_config.n_nodes / .n_slots_per_gpu).")

    time_budget_s = _int_or_none(inputs.get("time_budget_s"))
    log_name      = inputs.get("log_name")

    # 4) Build (src,tgt,offset) tuples, mark which are ready at step 0
    lang_pairs: List[Tuple[str, str, int]] = []
    lps_ready: List[Tuple[str, str, int]] = []
    lp_to_task_keys: Dict[Tuple[str, str, int], List[str]] = defaultdict(list)

    for tkey, task in tasks.items():
        if not isinstance(task, dict):
            logger.warning(f"Task {tkey!r} is not a mapping; skipping.")
            continue
        lab = task.get("src_tgt")
        if not isinstance(lab, str) or "-" not in lab:
            logger.warning(f"Task {tkey!r} has no 'src_tgt' like 'en-fi'; skipping.")
            continue
        src, tgt = lab.split("-", 1)
        offset = int(task.get("offset", 0) or 0)
        ready0 = int(task.get("introduce_at_training_step", 0) or 0) == 0

        triple = (src, tgt, offset)
        lang_pairs.append(triple)
        if ready0:
            lps_ready.append(triple)
        lp_to_task_keys[triple].append(tkey)

    if not lang_pairs:
        raise UserConfigError("No usable tasks discovered (need tasks.* with a 'src_tgt').")

    # 5) Infer missing n_nodes / n_slots_per_gpu
    if n_nodes is None and n_slots_per_gpu is None:
        raise UserConfigError("Internal error: both n_nodes and n_slots_per_gpu are None after resolve.")
   
    import math
    if n_nodes is None:
        slots_per_node = n_gpus_per_node * n_slots_per_gpu
        n_nodes = math.ceil(len(lang_pairs) / max(1, slots_per_node))
         
    n_gpus_total = n_nodes * n_gpus_per_node

    if n_slots_per_gpu is None:
        # enough slots so all pairs can be placed
        n_slots_per_gpu = math.ceil(len(lang_pairs) / max(1, n_gpus_total))

    logger.info(f"n_nodes:          {n_nodes}")
    logger.info(f"n_gpus_per_node:  {n_gpus_per_node}")
    logger.info(f"n_slots_per_gpu:  {n_slots_per_gpu}")
    logger.info(f"total slots:      {n_nodes * n_gpus_per_node * n_slots_per_gpu}")
    logger.info(f"lang_pairs:       {len(lang_pairs)}")

    # 6) Ensure every GPU has something at step 0: shift curricula if needed
    if len(lps_ready) < n_gpus_total:
        # gather all iats and find the threshold at the k-th smallest, k = n_gpus_total
        iats = sorted(int((t.get("introduce_at_training_step", 0) or 0)) for t in tasks.values())
        if len(iats) > n_gpus_total:
            kth = iats[n_gpus_total]  # 0-based index gives (n_gpus_total+1)-th smallest
        else:
            kth = 0
        lps_ready = []
        for tkey, task in tasks.items():
            lab = task.get("src_tgt", "")
            if "-" not in lab:
                continue
            src, tgt = lab.split("-", 1)
            offset = int(task.get("offset", 0) or 0)
            # shift down so at least n_gpus_total tasks become 0
            if "introduce_at_training_step" not in task:
                lps_ready.append((src, tgt, offset))
                continue
            adjusted = max(0, int(task.get("introduce_at_training_step", 0) or 0) - kth)
            task["introduce_at_training_step"] = adjusted
            if adjusted == 0:
                lps_ready.append((src, tgt, offset))

    # 7) Trivial assignment if a single GPU only
    if n_gpus_total < 2:
        logger.info("Assigning all tasks to 0:0 (single GPU total).")
        for tkey in tasks:
            tasks[tkey]["node_gpu"] = "0:0"
    else:
        # Lazily import your optimizer only when needed
        try:
            from .gpu_assignment import optimize_gpu_assignment
        except Exception as e:
            raise UserConfigError(f"Internal error: optimizer not available: {e}")

        assignment = optimize_gpu_assignment(
            n_nodes=n_nodes,
            n_gpus_per_node=n_gpus_per_node,
            n_slots_per_gpu=n_slots_per_gpu,
            lang_pairs=lang_pairs,
            lang_to_group_mapping=groups,
            lps_ready_to_start=lps_ready,
            log_name=log_name,
            time_budget_s=time_budget_s,
        )

        # Write assignments back
        for gpu_slot, triple in assignment.items():
            if triple is None:
                continue
            tkey = lp_to_task_keys[triple].pop()
            tasks[tkey]["node_gpu"] = f"{gpu_slot.node}:{gpu_slot.gpu}"

        # Sanity: all consumed
        leftovers = sum(len(v) for v in lp_to_task_keys.values())
        if leftovers:
            for lp, pending in lp_to_task_keys.items():
                if pending:
                    logger.warning(f"Unassigned for {lp}: {pending}")
            raise UserConfigError("Not all tasks were assigned to a device slot.")

    # 8) Post conditions: every task has node_gpu
    for tkey, task in tasks.items():
        if "node_gpu" not in task:
            raise UserConfigError(f"Task {tkey!r} missing 'node_gpu' after assignment.")

    # 9) Global distributed settings
    doc["n_nodes"] = n_nodes
    doc["world_size"] = n_gpus_total
    # NOTE: prev code had gpu_ranks = range(n_gpus_per_node); that’s per-node ranks.
    # For global ranks, emit 0..world_size-1:
    doc["gpu_ranks"] = list(range(n_gpus_total))

    # 10) Normalize introduce_at_training_step so each device has at least one task ready at 0
    train_steps = int(doc.get("train_steps", 100_000) or 100_000)
    min_iat_by_slot: Dict[str, int] = defaultdict(lambda: train_steps)
    for tkey, task in tasks.items():
        if "introduce_at_training_step" in task:
            slot = task["node_gpu"]
            iat  = int(task.get("introduce_at_training_step", 0) or 0)
            if iat < min_iat_by_slot[slot]:
                min_iat_by_slot[slot] = iat

    for tkey, task in tasks.items():
        if "introduce_at_training_step" not in task:
            continue
        slot = task["node_gpu"]
        adjust = min_iat_by_slot[slot]
        if adjust > 0:
            logger.warning(f"Reducing introduce_at_training_step of {tkey} by {adjust}")
            task["introduce_at_training_step"] = int(task["introduce_at_training_step"]) - adjust

    return "completed allocate devices step"

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
    p.set_defaults(handler=allocate_devices)
    p.set_defaults(_mutates_yaml=True)
    
    register_command_io("allocate_devices", {
         "reads": ["tasks", "config_config.sharing_groups.groups", "config_config.n_gpus_per_node",
                   "config_config.n_nodes", "config_config.n_slots_per_gpu",
                   "config_config.time_budget_s", "config_config.log_name"],
         "writes": ["tasks", "world_size", "node_gpu", "n_nodes", "gpu_ranks"],
         "summary": ("Place tasks on node:gpu slots, set world_size/gpu_ranks, "
                     "and shift curricula so each device can start."),
    })
    register_command_template_extras("allocate_devices", {
         "config_config": {
              "_notes": [
                   "Either set n_nodes or n_slots_per_gpu (the other is inferred).",
                   "One task slot per GPU by default. Provide either n_nodes OR n_slots_per_gpu.",
                   "n_gpus_per_node is required to compute the full multi-node layout."
              ],
         }
    })
    
