from __future__ import annotations

from .utils import UserConfigError, coalesce, load_yaml, register_command_io, register_command_template_extras, logger, resolve_command_inputs, _set_and_log
from typing import Any, Dict, Iterable, List, Tuple
from collections import Counter



def register(subparsers):
    p = subparsers.add_parser(
        "sharing_groups",
        help="Produce per-task encoder/decoder sharing lists from macros and a language→group map.",
        description=(
            "Reads config_config.sharing_groups.{groups,enc_layers,dec_layers,enc_sharing_groups,dec_sharing_groups}. "
            "Requires a 'groups' mapping {lang: group} (from clustering or predefined in YAML), or --groups-default. "
            "Requires enc_layers/dec_layers given as integers or lists."
            "Produces per-layer lists for encoder/decoder (lengths must match enc_layers/dec_layers) "
            "expands macros per task. You can synthesize a groups map from tasks if missing."
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--enc_sharing_groups", action="append", metavar="TOKEN",
        help=('list of {LANGUAGE | GROUP | FULL}. '
              'The first two may be prefixed, e.g. SRC_LANGUAGE or TGT_GROUP. '
              'Default prefix for encoder is SRC.'))
    p.add_argument("--dec_sharing_groups", action="append", metavar="TOKEN",
        help=('list of {LANGUAGE | GROUP | FULL}.'
              'The first two may be prefixed, e.g. SRC_LANGUAGE or TGT_GROUP. '
              'Default prefix for decoder is TGT.'))
    p.add_argument("--groups-default", choices=["none", "per-language", "all-one"],  default="none",
        help="If 'groups' mapping is missing (or --force-default-groups is set), synthesize it from tasks: "
             "'per-language' = each language its own group; 'all-one' = all languages in one group.")
    p.add_argument("--force-default-groups", action="store_true",
        help="Overwrite an existing 'groups' mapping with the selected --groups-default.")
    p.set_defaults(handler=sharing_groups)
    p.set_defaults(_mutates_yaml=True)

    register_command_io("sharing_groups", {
        "reads": [
            "tasks",
            "config_config.enc_layers",
            "config_config.dec_layers",
            "config_config.enc_sharing_groups",
            "config_config.dec_sharing_groups",
            "config_config.groups",
        ],
        "writes": [
            "config_config.encoder_sharing",
            "config_config.decoder_sharing",
            "tasks.*.enc_sharing_group",
            "tasks.*.dec_sharing_group",
            "config_config.sharing_groups.groups",
        ],
        "summary": "Expand per-task sharing from macros and a language→group mapping, using layer counts and groups.",
    })
    register_command_template_extras("sharing_groups", {
        "_notes": [
            "If enc_layers/dec_layers are unknown elsewhere, set them here.",
            "Provide 'groups' mapping (from clustering or manually).",
            "Provide enc_layers/dec_layers if your model config doesn't infer them."
        ],
        "config_config": {
            "groups": {"en": 0, "fi": 1} # minimal illustrative example
        },
        # Provide enc_layers/dec_layers if your model config doesn't infer them.
        "enc_layers": 6,
        "dec_layers": 6,
        # Not pre-filling enc_sharing_groups/dec_sharing_groups since this command writes them.
    })
    p = subparsers.add_parser(
        "inv_sharing_groups",
        help="Infer enc/dec layer counts, macro lists, and groups from tasks' sharing_group values.",
        description=(
            "Scans tasks.*.{enc,dec}_sharing_group and reconstructs:\n"
            "  config_config.sharing_groups.{enc_layers,dec_layers,\n"
            "    enc_sharing_groups,dec_sharing_groups,groups}\n"
            "so that the forward 'sharing_groups' command can reproduce the observed per-task labels."
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml", 
                   help="YAML file containing tasks with enc_sharing_group / dec_sharing_group.")
    p.add_argument("--out_config", metavar="FILE.yaml",
                   help="Write the updated YAML to this file. If omitted, the framework saver is used.")
    p.set_defaults(handler=inv_sharing_groups, _parser=p, _mutates_yaml=True)
    register_command_io("inv_sharing_groups", {
        "reads" : [
            "tasks"
        ],
        "writes" : [
            "config_config.sharing_groups.enc_layers",
            "config_config.sharing_groups.dec_layers",
            "config_config.sharing_groups.enc_sharing_groups",
            "config_config.sharing_groups.dec_sharing_groups",
            "config_config.sharing_groups.groups",
        ],
        "summary" : "Infer macro sharing declarations and language→group map from per-task sharing labels."
    })
    register_command_template_extras("inv_sharing_groups",{
    })
    
def sharing_groups(opts):
    """
    Expand per-task encoder/decoder sharing macros into concrete per-layer assignments,
    using a language->group map and declared layer counts.
    """

    def _ensure_tasks_map(doc: dict) -> dict:
        tasks = doc.get("tasks")
        if not isinstance(tasks, dict) or not tasks:
            raise UserConfigError("No 'tasks' mapping found; run complete_language_pairs first or provide tasks.")
        return tasks
    
    def _collect_languages_from_tasks(tasks: dict) -> set[str]:
        langs: set[str] = set()
        for k, t in tasks.items():
            if not isinstance(t, dict):
                continue
            lab = t.get("src_tgt")
            if isinstance(lab, str) and "-" in lab:
                s, t_ = lab.split("-", 1)
                langs.add(s); langs.add(t_)
                continue
            # fallback: parse from task key (supports .ae)
            key = k[:-3] if isinstance(k, str) and k.endswith(".ae") else k
            parts = key.split("-", 1)
            if len(parts) == 2:
                langs.add(parts[0]); langs.add(parts[1])
        return langs
    
    def _layer_count(x: Any, *, label: str) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, list):
            l = len(x)
            if l <= 2:
                logger.warning(f"{label} is a list {x}, meaning you have only {l} layers; this looks like a mistake!")
            return l
        if x is None:
            logger.warning(f"{label} has no value, using the default of 4. You should specify this explicitly.")
            return 4            
        raise UserConfigError(f"{label} must be an integer. Try:\n\tpython -m config_config2 yaml_help --keys config_config.{label}")
    
    def _ensure_list_of_str(x: Any, *, label: str, n: int) -> list[str]:
        if x is None:
            v = ["LANGUAGE"] * n
            logger.warning(f"{label} has no value, using the default {v} for {n} layers. You should specify this explicitly.")
            return v
        if not isinstance(x, list) or not all(isinstance(i, str) for i in x):
            raise UserConfigError(f"{label} must be a list of strings; got {type(x).__name__}.")
        return x
    
    def _macro_mapping_for(src: str, tgt: str, groups: dict) -> tuple[dict, dict]:
        # groups[src]/groups[tgt] may be int/str; keep as-is
        gsrc = groups.get(src)
        gtgt = groups.get(tgt)
        if gsrc is None or gtgt is None:
            raise UserConfigError(f"'groups' mapping lacks entries for {src!r} or {tgt!r}.")
        mapping_src = {
            "LANGUAGE": src, "GROUP": gsrc, "FULL": "full",
            "SRC_LANGUAGE": src, "SRC_GROUP": gsrc,
            "TGT_LANGUAGE": tgt, "TGT_GROUP": gtgt,
        }
        mapping_tgt = {
            "LANGUAGE": tgt, "GROUP": gtgt, "FULL": "full",
            "SRC_LANGUAGE": src, "SRC_GROUP": gsrc,
            "TGT_LANGUAGE": tgt, "TGT_GROUP": gtgt,
        }
        return mapping_src, mapping_tgt

    inputs = resolve_command_inputs("sharing_groups", opts)
    doc = opts.in_config[0]  # normalized in-place
    tasks = _ensure_tasks_map(doc)

    # Synthesize or read 'groups'
    groups = inputs.get("groups")
    want_default = getattr(opts, "groups_default", "none")
    force_default = bool(getattr(opts, "force_default_groups", False))
    if (groups is None or force_default) and want_default in ("per-language", "all-one"):
        langs = _collect_languages_from_tasks(tasks)
        if not langs:
            raise UserConfigError("Cannot synthesize groups: no languages found in tasks.")
        if want_default == "per-language":
            # each language = its own group (use the language code as group id)
            groups = {lang: lang for lang in sorted(langs)}
        else:  # all-one
            groups = {lang: "all" for lang in sorted(langs)}
        # write back into normalized YAML path config_config.sharing_groups.groups
        cc = doc.setdefault("config_config", {})
        cc_cmd = cc.setdefault("sharing_groups", {})
        _set_and_log(cc, "groups", groups)
        logger.info(f"sharing_groups: synthesized groups mapping ({want_default}, {len(groups)} entries).")
    if not isinstance(groups, dict) or not groups:
        raise UserConfigError(
            "Missing 'config_config.groups' for sharing_groups. "
            "Provide it in YAML (now under config_config.sharing_groups.groups), "
            "or use '--groups-default per-language' / '--groups-default all-one'.")

    # enc/dec layer counts (can be int or list)
    enc_layers_val = inputs.get("enc_layers")
    dec_layers_val = inputs.get("dec_layers")
    enc_n = _layer_count(enc_layers_val, label="enc_layers")
    dec_n = _layer_count(dec_layers_val, label="dec_layers")
    
    if enc_layers_val and isinstance(enc_layers_val, list):
        _set_and_log(cc_cmd, "enc_layers", enc_macros)
    else:
        _set_and_log(cc_cmd, "enc_layers", enc_n)
    if dec_layers_val and isinstance(dec_layers_val, list):
        _set_and_log(cc_cmd, "dec_layers", dec_macros)
    else:
        _set_and_log(cc_cmd, "dec_layers", dec_n)

    # Macro lists (list[str]) — CLI overrides already applied by resolve_command_inputs
    enc_macros = _ensure_list_of_str(inputs.get("enc_sharing_groups"), label="enc_sharing_groups", n=enc_n)
    dec_macros = _ensure_list_of_str(inputs.get("dec_sharing_groups"), label="dec_sharing_groups", n=dec_n)
    if len(enc_macros) != enc_n or len(dec_macros) != dec_n:
        raise UserConfigError(
            "Length mismatch: number of sharing tokens must equal the number of encoder/decoder layers "
            f"(enc: {len(enc_macros)} vs {enc_n}, dec: {len(dec_macros)} vs {dec_n}).")
    _set_and_log(cc_cmd, "enc_sharing_groups", enc_macros)
    _set_and_log(cc_cmd, "dec_sharing_groups", dec_macros)
        
    # Expand for every task
    updated = 0
    for tname, task in tasks.items():
        if not isinstance(task, dict):
            continue
        lab = task.get("src_tgt")
        if not isinstance(lab, str) or "-" not in lab:
            logger.warning(f"Task '{tname}' lacks a proper 'src_tgt' (e.g., 'en-fi'); skipping.")
            continue
        src, tgt = lab.split("-", 1)

        map_src, map_tgt = _macro_mapping_for(src, tgt, groups)

        try:
            enc_resolved = [map_src[m] for m in enc_macros]
        except KeyError as e:
            raise UserConfigError(f"Unknown encoder macro {str(e)} in enc_sharing_groups; "
                                  "allowed: LANGUAGE, GROUP, FULL, SRC_LANGUAGE, SRC_GROUP, TGT_LANGUAGE, TGT_GROUP")
        try:
            dec_resolved = [map_tgt[m] for m in dec_macros]
        except KeyError as e:
            raise UserConfigError(f"Unknown decoder macro {str(e)} in dec_sharing_groups; "
                                  "allowed: LANGUAGE, GROUP, FULL, SRC_LANGUAGE, SRC_GROUP, TGT_LANGUAGE, TGT_GROUP")

        _set_and_log(task, "enc_sharing_group", enc_resolved)
        _set_and_log(task, "dec_sharing_group", dec_resolved)
        updated += 1

    logger.info(f"sharing_groups: expanded sharing for {updated} task(s).")
    return "completed sharing groups step"








def _ensure_tasks_map(doc: dict) -> dict:
    tasks = doc.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise UserConfigError("No 'tasks' mapping found; cannot invert sharing without tasks.")
    return tasks


def _split_pair(label: str) -> Tuple[str, str] | None:
    if isinstance(label, str) and "-" in label:
        a, b = label.split("-", 1)
        if a and b:
            return a, b
    return None


def _collect_langs(tasks: dict) -> List[str]:
    langs = set()
    for tname, t in tasks.items():
        if not isinstance(t, dict):
            continue
        pair = t.get("src_tgt")
        ab = _split_pair(pair) if pair else None
        if ab is None:
            # fallback from key (supports ".ae")
            key = tname[:-3] if isinstance(tname, str) and tname.endswith(".ae") else tname
            ab = _split_pair(key)
        if ab is not None:
            langs.update(ab)
    if not langs:
        raise UserConfigError("Could not infer languages from tasks (need 'src_tgt' or task key like 'en-fi').")
    return sorted(langs)


def _get_sharing_lists(tasks: dict, field: str) -> Dict[str, List[str]]:
    """
    Return {task_name: list[str]} for a given field (enc_sharing_group / dec_sharing_group).
    Skip tasks missing the field (warn).
    """
    out: Dict[str, List[str]] = {}
    for tname, t in tasks.items():
        if not isinstance(t, dict):
            continue
        val = t.get(field)
        if isinstance(val, list) and all(isinstance(x, (str, int)) for x in val):
            # cast ints to str to make comparison uniform
            out[tname] = [str(x) for x in val]
        else:
            logger.warning(f"{field} missing or not a list in task '{tname}'; skipping this task for inversion.")
    if not out:
        raise UserConfigError(f"No tasks contain '{field}'.")
    return out


def _infer_layer_count(per_task_lists: Dict[str, List[str]], label: str) -> int:
    lengths = [len(v) for v in per_task_lists.values()]
    if not lengths:
        raise UserConfigError(f"Cannot infer {label}: no lists seen.")
    common, freq = Counter(lengths).most_common(1)[0]
    if len(set(lengths)) > 1:
        logger.warning(
            f"{label} list lengths vary across tasks {lengths}. Using the modal length {common} "
            f"and ignoring tasks with different lengths."
        )
    return common


def _languages_for_task(tasks: dict, tname: str) -> Tuple[str, str] | None:
    t = tasks.get(tname, {})
    pair = t.get("src_tgt")
    ab = _split_pair(pair) if pair else None
    if ab is None:
        key = tname[:-3] if isinstance(tname, str) and tname.endswith(".ae") else tname
        ab = _split_pair(key)
    return ab


def _infer_macros_for_axis(
    *,
    axis_name: str,                  # "encoder" or "decoder"
    per_task_lists: Dict[str, List[str]],
    tasks: dict,
    layer_count: int,
    side: str                        # "src" for encoder, "tgt" for decoder
) -> Tuple[List[str], Dict[str, str], List[int]]:
    """
    For each layer index i in [0..layer_count), decide macro:
      - LANGUAGE     if label == <side language code> for all considered tasks
      - FULL         if label is the same constant string across tasks and equals "full" (or same const)
      - GROUP        if label depends only on the <side language> and is consistent for that language
                     (build mapping lang->group_label). Must be consistent across ALL indices using GROUP.
    Returns:
      (macros, groups_partial_map, ignored_task_indices)
    groups_partial_map maps language -> group_label derived from any GROUP-index; caller must merge/validate globally.
    ignored_task_indices: task indices at which lists were too short and thus ignored (rare; after modal length filtering).
    """
    task_names = list(per_task_lists.keys())

    # Build arrays per index
    macros: List[str] = []
    groups_map_partial: Dict[str, str] = {}
    ignored_layers: List[int] = []

    for i in range(layer_count):
        # Collect observations for this layer across tasks
        obs: List[Tuple[str, str]] = []  # (language_on_this_side, observed_label)
        constant_label: str | None = None
        same_constant = True

        for tname in task_names:
            lst = per_task_lists[tname]
            if i >= len(lst):
                continue  # this task is shorter; already warned in count inference
            ab = _languages_for_task(tasks, tname)
            if ab is None:
                continue
            lang = ab[0] if side == "src" else ab[1]
            label_i = str(lst[i])
            obs.append((lang, label_i))
            if constant_label is None:
                constant_label = label_i
            elif constant_label != label_i:
                same_constant = False

        if not obs:
            # No usable observations at this layer index → skip (should be rare)
            logger.warning(f"{axis_name} layer {i}: no observations; defaulting to FULL.")
            macros.append("FULL")
            continue

        # Hypothesis 1: LANGUAGE
        if all(lang == label for (lang, label) in obs):
            macros.append("LANGUAGE" if side == "src" else "LANGUAGE")  # same token set
            continue

        # Hypothesis 2: FULL (constant across tasks)
        if same_constant:
            # Prefer FULL if the constant is literally 'full'
            if constant_label == "full":
                macros.append("FULL")
                continue
            # If it's a non-'full' constant, we *can* still encode as FULL (semantically same),
            # but keep the literal for transparency; choose FULL to round-trip with sharing_groups.
            macros.append("FULL")
            continue

        # Hypothesis 3: GROUP — label depends only on language on this side
        per_lang: Dict[str, str] = {}
        ok = True
        for lang, label in obs:
            prev = per_lang.get(lang)
            if prev is None:
                per_lang[lang] = label
            elif prev != label:
                ok = False
                break
        if ok:
            # Merge into the partial map; if conflict across indices, we'll error in the caller.
            for lang, grp in per_lang.items():
                if lang in groups_map_partial and groups_map_partial[lang] != grp:
                    raise UserConfigError(
                        f"Inconsistent group label for language '{lang}' on {axis_name} at layer {i}: "
                        f"{groups_map_partial[lang]!r} vs {grp!r}."
                    )
                groups_map_partial[lang] = grp
            macros.append("GROUP")
            continue

        # No hypothesis fits
        raise UserConfigError(
            f"Cannot express {axis_name} layer {i} labels with LANGUAGE/GROUP/FULL.\n"
            f"Observed: {sorted(set(lab for _, lab in obs))}"
        )

    return macros, groups_map_partial, ignored_layers


def inv_sharing_groups(opts):
    """
    Infer:
      - config_config.sharing_groups.enc_layers / dec_layers (counts)
      - config_config.sharing_groups.enc_sharing_groups / dec_sharing_groups (macro lists)
      - config_config.sharing_groups.groups (language -> group label), when GROUP is used
    from tasks.*.{enc,dec}_sharing_group.
    """
    def _looks_generated_by_sharing(task_key: str, task: dict) -> bool:
        """
        Heuristic: this task looks like it was produced by complete_language_pairs.
        """
        if not isinstance(task, dict):
            return False
    
        # If task has keys outside the known generated set, we keep it.
        extra = set(task.keys()) - ALLOWED_TASK_KEYS
        if extra:
            return False
    
        # Must have src_tgt identifying the pair
        s, t = _has_languages(task.get("src_tgt"))
        if s is None or t is None:
            return False
    
        # Must have src/tgt paths (canonical or legacy)
        src = task.get("src", task.get("path_src"))
        tgt = task.get("tgt", task.get("path_tgt"))
        if not (isinstance(src, str) and isinstance(tgt, str)):
            return False
    
        # AE detection is fine but not required for pruning; the generator writes both
        # supervised and AE with the same shape (only key/type differs).
        return True
    
    # Normalize & resolve reads (keeps style / CLI parity)
    inputs = resolve_command_inputs("inv_sharing_groups", opts)
    doc = opts.in_config[0]  # normalized in-place
    tasks = _ensure_tasks_map(doc)

    enc_lists = _get_sharing_lists(tasks, "enc_sharing_group")
    dec_lists = _get_sharing_lists(tasks, "dec_sharing_group")

    enc_n = _infer_layer_count(enc_lists, "encoder")
    dec_n = _infer_layer_count(dec_lists, "decoder")

    # Keep only tasks that match modal length (others already warned in _infer_layer_count)
    enc_lists = {k: v for k, v in enc_lists.items() if len(v) == enc_n}
    dec_lists = {k: v for k, v in dec_lists.items() if len(v) == dec_n}

    # Infer macro sequences and partial group maps from both axes
    enc_macros, enc_groups_map, _ = _infer_macros_for_axis(
        axis_name="encoder", per_task_lists=enc_lists, tasks=tasks, layer_count=enc_n, side="src"
    )
    dec_macros, dec_groups_map, _ = _infer_macros_for_axis(
        axis_name="decoder", per_task_lists=dec_lists, tasks=tasks, layer_count=dec_n, side="tgt"
    )

    # Merge group maps; they must agree where both define the same language
    groups_map: Dict[str, str] = {}
    for lang, grp in {**enc_groups_map, **dec_groups_map}.items():
        if lang in groups_map and groups_map[lang] != grp:
            raise UserConfigError(
                f"Inconsistent group label for language '{lang}' across encoder/decoder: "
                f"{groups_map[lang]!r} vs {grp!r}."
            )
        groups_map[lang] = grp

    # Write normalized outputs
    cc = doc.setdefault("config_config", {})
    sg = cc.setdefault("sharing_groups", {})

    _set_and_log(sg, "enc_layers", enc_n)
    _set_and_log(sg, "dec_layers", dec_n)
    _set_and_log(sg, "enc_sharing_groups", enc_macros)
    _set_and_log(sg, "dec_sharing_groups", dec_macros)

    if "GROUP" in enc_macros or "GROUP" in dec_macros:
        if not groups_map:
            raise UserConfigError(
                "Macros use GROUP but no consistent language→group mapping could be inferred."
            )
        sg["groups"] = dict(sorted(groups_map.items()))
        logger.info(f"[inv] inferred groups: {len(groups_map)} languages")
    else:
        # If GROUP never used, we don't need to write 'groups'
        logger.info("[inv] GROUP macro not used; no 'groups' mapping required.")

    if not getattr(opts, "no_prune_sharing", False):
        removed_e = removed_d = 0
        for tname, t in list(tasks.items()):
            if not isinstance(t, dict):
                continue
            if "enc_sharing_group" in t:
                t.pop("enc_sharing_group", None)
                removed_e += 1
            if "dec_sharing_group" in t:
                t.pop("dec_sharing_group", None)
                removed_d += 1
        logger.info(f"[inv] pruned enc_sharing_group from {removed_e} task(s).")
        logger.info(f"[inv] pruned dec_sharing_group from {removed_d} task(s).")

    logger.info(f"[inv] sharing: enc_layers={enc_n}")
    logger.info(f"[inv] sharing: dec_layers={dec_n}")
    logger.info(f"[inv] sharing: enc_macros={enc_macros}")
    logger.info(f"[inv] sharing: dec_macros={dec_macros}")

