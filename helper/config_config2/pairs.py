from .utils import UserConfigError, load_yaml, register_command_io, register_command_template_extras, logger, resolve_command_inputs, _set_and_log
import os, time, itertools
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

def register(subparsers):
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
    p.set_defaults(handler=complete_language_pairs)
    p.set_defaults(_mutates_yaml=True)

    p = subparsers.add_parser(
        "inv_complete_language_pairs",
        help="Reconstruct input templates for complete_language_pairs from existing tasks.",
        description=(
            "Reads 'tasks' and infers src/tgt templates (and AE templates if present), then writes them under "
            "'config_config.complete_language_pairs'. Useful to reduce a produced config back into minimal inputs."
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml",
                   help="YAML file containing tasks produced by complete_language_pairs.")
    p.add_argument("--out_config", metavar="FILE.yaml",
                   help="Write the updated YAML to this file. If omitted, the framework's default saver is used.")
    p.set_defaults(handler=inv_complete_language_pairs, _parser=p, _mutates_yaml=True)

    register_command_io("complete_language_pairs", {
        "reads": ["src_vocab", "tgt_vocab", "config_config.src_path", "config_config.tgt_path",
                  "config_config.ae_path", "config_config.valid_src_path", "config_config.valid_tgt_path",
                  "config_config.autoencoder", "config_config.autoencoder_validation"],
        "writes": ["tasks"],
        "summary": "Scans vocab/templates and builds tasks; optional autoencoder/validation tasks.",
    })
    register_command_io("inv_complete_language_pairs", {
        "reads":  ["tasks"],
        "writes": [
            "config_config.complete_language_pairs.src_path",
            "config_config.complete_language_pairs.tgt_path",
            "config_config.complete_language_pairs.valid_src_path",
            "config_config.complete_language_pairs.valid_tgt_path",
            "config_config.complete_language_pairs.autoencoder",
            "config_config.complete_language_pairs.autoencoder_validation",
            "config_config.complete_language_pairs.ae_path",
        ],
        "summary": "Infer path templates and AE flags from tasks and store them under config_config.complete_language_pairs.",
    })
    register_command_template_extras("complete_language_pairs", {
        "_notes": [
            "Set corpus and (optional) validation templates. AE path is monolingual.",
            "Use {src} and {tgt} in path templates, e.g. /data/{src}-{tgt}.src.",
            "Only pairs whose files exist on disk will be added implicitly to 'tasks'.",
            "Provide at least one of src_path/tgt_path or explicit tasks.",
        ],
        # "src_vocab", {"en": "/example/vocabs/en.vocab", "fi": "/example/vocabs/fi.vocab"}, 
        # "tgt_vocab", {"en": "/example/vocabs/en.vocab", "fi": "/example/vocabs/fi.vocab"},
        "src_vocab": "/data/vocabs/{lang}.txt",
        "tgt_vocab": "/data/vocabs/{lang}.txt",
        "config_config": {
            "complete_language_pairs": {
                "autoencoder": False,
                "src_path": "/data/corpora/{src}-{tgt}.src",
                "tgt_path": "/data/corpora/{src}-{tgt}.tgt",
                "ae_path":  "/data/mono/{lang}.txt",
                # Optional: valid_src_path/valid_tgt_path for dev sets.
                "autoencoder_validation": False,
                "valid_src_path": "/data/valid/{src}-{tgt}.src",
                "valid_tgt_path": "/data/valid/{src}-{tgt}.tgt",
            }
        }
    })

# PURPOSE: Return sorted lists of source and target language keys from vocab sections.
def _ensure_vocab_map(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise UserConfigError(f"'{name}' must be a mapping {{lang: path_template}}; got {type(value).__name__}.")
    if not value:
        raise UserConfigError(f"'{name}' is empty; add at least one language.")
    for k, v in value.items():
        if not isinstance(k, str):
            raise UserConfigError(f"'{name}' has non-string language key: {k!r}.")
        if not isinstance(v, str):
            raise UserConfigError(f"'{name}' has non-string template for language {k!r}.")
    return value

def complete_language_pairs(opts):
    """
    Build 'tasks' by scanning language vocabularies and path templates.

    Robustness goals:
      - Never assume keys exist; validate shapes and types.
      - CLI flags (if provided) override YAML. False is a valid boolean (don't treat as "missing").
      - Optional templates (validation, AE) can be omitted safely.
      - Formatting of templates is guarded; missing placeholders give clear errors.
      - If nothing is added, raise with an actionable message.
    """

    # Format helper
    def _fmt_or_error(tpl, vars_, label):
        if tpl is None:
            return None
        try:
            return tpl.format(**vars_)
        except KeyError as e:
            missing = str(e).strip("'")
            raise UserConfigError(
                f"{label}: template '{tpl}' is missing placeholder '{missing}'. "
                f"Available: {', '.join(sorted(vars_.keys()))}."
            )
    
    def _lenient_path(p: str | None, *, what: str) -> tuple[str | None, bool]:
        """
        If path exists -> (path, True).
        If path is None -> (None, False).
        If path doesn't exist -> (path + '.missing', False) and warn.
        """
        if p is None:
            return None, False
        if os.path.exists(p):
            return p, True
        return p + ".missing", False
    
    def _ensure_tasks_section(doc: dict) -> dict:
        if "tasks" not in doc or doc["tasks"] is None:
            doc["tasks"] = {}
        if not isinstance(doc["tasks"], dict):
            raise UserConfigError("'tasks' must be a mapping.")
        return doc["tasks"]
    
    def _is_missing(path: str | None) -> bool:
        return isinstance(path, str) and path.endswith(".missing")

    def _add_language_pair_compat(doc: dict,
                                  task_key: str, 
                                  src_lang: str, tgt_lang: str,
                                  src_f: str | None, tgt_f: str | None,
                                  vsrc_f: str | None, vtgt_f: str | None) -> None:
        src_path, _ = _lenient_path(src_f,  what="src")
        tgt_path, _ = _lenient_path(tgt_f,  what="tgt")
        valid_src_path, _ = _lenient_path(vsrc_f, what="valid_src") if vsrc_f else (None, False)
        valid_tgt_path, _ = _lenient_path(vtgt_f, what="valid_tgt") if vtgt_f else (None, False)
        
        tasks = _ensure_tasks_section(doc)
        task = tasks.setdefault(task_key, {})
    
        _set_and_log(task, "src_tgt", f"{src_lang}-{tgt_lang}", level="info")
        _set_and_log(task, "path_src", src_path, level="info")
        _set_and_log(task, "path_tgt", tgt_path, level="info")
        
        if valid_src_path is not None:
            _set_and_log(task, "path_valid_src", valid_src_path, level="info")
        if valid_tgt_path is not None:
            _set_and_log(task, "path_valid_tgt", valid_tgt_path, level="info")

        for k in ("path_src", "path_tgt", "path_valid_src", "path_valid_tgt"):
            v = task.get(k)
            if _is_missing(v):
                logger.warning(f"{k} marked missing: {v!r}")

    # Normalize & resolve reads for this command
    inputs = resolve_command_inputs("complete_language_pairs", opts)
    doc = opts.in_config[0]   # normalized by resolver

    # Top-level required sections (not part of resolve_command_inputs contract)
    src_vocab = _ensure_vocab_map(doc.get("src_vocab"), "src_vocab")
    tgt_vocab = _ensure_vocab_map(doc.get("tgt_vocab"), "tgt_vocab")
    src_langs = sorted(src_vocab.keys())
    tgt_langs = sorted(tgt_vocab.keys())

    # Pull resolved templates / toggles (all via inputs.get)
    src_tpl  = inputs.get("src_path")           # required by schema; non-None
    tgt_tpl  = inputs.get("tgt_path")           # required
    vsrc_tpl = inputs.get("valid_src_path")     # optional → may be None
    vtgt_tpl = inputs.get("valid_tgt_path")     # optional
    ae_path  = inputs.get("ae_path")            # optional: str | list[str] | None

    ae_enabled     = bool(inputs.get("autoencoder") or False)
    ae_val_enabled = bool(inputs.get("autoencoder_validation") or False)

    # AE path lists
    if isinstance(ae_path, str):
        ae_src_tpls = [ae_path]
        ae_tgt_tpls = [ae_path]
    elif isinstance(ae_path, list):
        if not all(isinstance(x, str) for x in ae_path):
            raise UserConfigError("ae_path must be a string or list of strings.")
        ae_src_tpls = list(ae_path)
        ae_tgt_tpls = list(ae_path)
    else:
        ae_src_tpls = []
        ae_tgt_tpls = []

    added = 0
    for s in src_langs:
        for t in tgt_langs:
            # inside the nested loops over (s, t)
            lang_a, lang_b = sorted((s, t))
            lang_pair   = f"{s}-{t}"
            sorted_pair = f"{lang_a}-{lang_b}"
            if lang_pair == sorted_pair:
                side_a, side_b = "src", "trg"     # Tatoeba uses 'trg'
                lang_a, lang_b = s, t
            else:
                side_a, side_b = "trg", "src"
                lang_a, lang_b = t, s                
            template_variables = {
                "src_lang": s,                "tgt_lang": t,
                "lang_a": lang_a,             "lang_b": lang_b,
                "side_a": side_a,             "side_b": side_b,
                "lang_pair": lang_pair,       "sorted_pair": sorted_pair }
            doc.setdefault("tasks", {})
            all_exist = True
            task_name = f"{s}-{t}"
            if s == t:
                # Autoencoder task(s)
                if not ae_enabled:
                    continue
                # AE templates (fallback to train templates when not provided)
                pairs = list(zip(ae_src_tpls, ae_tgt_tpls)) if (ae_src_tpls and ae_tgt_tpls) else [(src_tpl, tgt_tpl)]
                for ae_s_tpl, ae_t_tpl in pairs:
                    task_name = f"{s}-{t}.ae"
                    src_f = _fmt_or_error(ae_s_tpl, template_variables, "src (AE)")
                    tgt_f = _fmt_or_error(ae_t_tpl, template_variables, "tgt (AE)")
                    vsrc_f = _fmt_or_error(vsrc_tpl, template_variables, "valid_src (AE)") if (ae_val_enabled and vsrc_tpl) else None
                    vtgt_f = _fmt_or_error(vtgt_tpl, template_variables, "valid_tgt (AE)") if (ae_val_enabled and vtgt_tpl) else None
                    _add_language_pair_compat(doc, task_name, s, t, src_f, tgt_f, vsrc_f, vtgt_f)
                    added += 1
            else:
                task_name = f"{s}-{t}"
                src_f  = _fmt_or_error(src_tpl, template_variables, "src")
                tgt_f  = _fmt_or_error(tgt_tpl, template_variables, "tgt")
                vsrc_f = _fmt_or_error(vsrc_tpl, template_variables, "valid_src") if vsrc_tpl else None
                vtgt_f = _fmt_or_error(vtgt_tpl, template_variables, "valid_tgt") if vtgt_tpl else None
                _add_language_pair_compat(doc, task_name, s, t, src_f, tgt_f, vsrc_f, vtgt_f)
                added += 1
                
    if added == 0:
        raise UserConfigError(
            "No language pairs were added.\n"
            "Verify your templates and that files exist.\n"
            "Tip: 'config_config2 yaml_template --cmd complete_language_pairs --mode inputs'"
        )

    # Expand vocab templates (safe best-effort)
    for s in src_langs:
        v = src_vocab.get(s)
        if isinstance(v, str):
            try: src_vocab[s] = v.format(src_lang=s)
            except KeyError: pass
    for t in tgt_langs:
        v = tgt_vocab.get(t)
        if isinstance(v, str):
            try: tgt_vocab[t] = v.format(tgt_lang=t)
            except KeyError: pass

    return f"complete_language_pairs: added {added} task(s)"





def inv_complete_language_pairs(opts):
    """
    Reconstruct inputs for 'complete_language_pairs' under:
        config_config.complete_language_pairs.*
    using the current 'tasks' section.
    """

    CANON_TASK_KEYS  = {"src_tgt", "src", "tgt", "valid_src", "valid_tgt", "type"}
    LEGACY_TASK_KEYS = {"path_src", "path_tgt", "path_valid_src", "path_valid_tgt"}
    ALLOWED_TASK_KEYS = CANON_TASK_KEYS | LEGACY_TASK_KEYS
    
    def _strip_missing(p: str | None) -> str | None:
        if p is None:
            return None
        return p[:-8] if p.endswith(".missing") else p
    
    
    def _generalize_supervised_template(paths: List[str], pairs: List[Tuple[str, str]]) -> str | None:
        """
        Infer one supervised template by replacing language tokens with placeholders.
        Order of replacement:
          1) '{src_lang}-{tgt_lang}' or '{tgt_lang}-{src_lang}' where pair substrings appear
          2) then single tokens 'src_lang' and 'tgt_lang'
        If multiple shapes exist, we pick the most common one.
        """
        if not paths or not pairs:
            return None
        templated: List[str] = []
        for p, (s, t) in zip(paths, pairs):
            if not isinstance(p, str):
                return None
            base = p
            # replace combined tokens first
            base = base.replace(f"{s}-{t}", "{src_lang}-{tgt_lang}")
            base = base.replace(f"{t}-{s}", "{tgt_lang}-{src_lang}")
            # then single-language tokens
            base = base.replace(s, "{src_lang}")
            base = base.replace(t, "{tgt_lang}")
            templated.append(base)
        common = Counter(templated).most_common(1)
        return common[0][0] if common else None
    
    
    def _generalize_ae_template(paths: List[str], langs: List[str]) -> str | None:
        """
        Infer one AE template by replacing the language with '{lang}'.
        If multiple shapes exist, we pick the most common one.
        """
        if not paths or not langs:
            return None
        templated: List[str] = []
        for p, lang in zip(paths, langs):
            if not isinstance(p, str):
                return None
            templated.append(p.replace(lang, "{lang}"))
        common = Counter(templated).most_common(1)
        return common[0][0] if common else None
    
    
    def _collect_from_tasks(tasks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect supervised and AE evidence from tasks, accepting both legacy keys
        (path_src/path_tgt/path_valid_src/path_valid_tgt) and modern mirrors
        (src/tgt/valid_src/valid_tgt). Strips '.missing' suffixes.
        """
        sup_pairs: List[Tuple[str, str]] = []
        sup_src_paths: List[str] = []
        sup_tgt_paths: List[str] = []
        sup_vsrc_paths: List[str] = []
        sup_vtgt_paths: List[str] = []
    
        ae_langs: List[str] = []
        ae_src_paths: List[str] = []
        ae_tgt_paths: List[str] = []
        ae_vsrc_paths: List[str] = []
        ae_vtgt_paths: List[str] = []
    
        def pick(d: dict, a: str, b: str) -> str | None:
            v = d.get(a)
            if v is None:
                v = d.get(b)
            return _strip_missing(v)
    
        for task_key, task in tasks.items():
            if not isinstance(task, dict):
                logger.warning(f"Task '{task_key}' is not a mapping; skipping.")
                continue
    
            src = pick(task, "path_src", "src")
            tgt = pick(task, "path_tgt", "tgt")
            vsrc = pick(task, "path_valid_src", "valid_src")
            vtgt = pick(task, "path_valid_tgt", "valid_tgt")
    
            # languages: prefer 'src_tgt' if present; else parse from key
            pair_label = task.get("src_tgt")
            if isinstance(pair_label, str) and "-" in pair_label:
                s, t = pair_label.split("-", 1)
            else:
                # parse from task key: 'en-fi' or 'en-en.ae'
                key = task_key[:-3] if task_key.endswith(".ae") else task_key
                parts = key.split("-", 1)
                if len(parts) != 2:
                    logger.warning(f"Cannot infer languages for task '{task_key}'; skipping.")
                    continue
                s, t = parts
    
            is_ae = task_key.endswith(".ae") or (s == t and (task.get("type") == "autoencoder"))
    
            if is_ae:
                ae_langs.append(s)   # s == t
                if src is not None:  ae_src_paths.append(src)
                if tgt is not None:  ae_tgt_paths.append(tgt)
                if vsrc is not None: ae_vsrc_paths.append(vsrc)
                if vtgt is not None: ae_vtgt_paths.append(vtgt)
            else:
                sup_pairs.append((s, t))
                if src is not None:  sup_src_paths.append(src)
                if tgt is not None:  sup_tgt_paths.append(tgt)
                if vsrc is not None: sup_vsrc_paths.append(vsrc)
                if vtgt is not None: sup_vtgt_paths.append(vtgt)
    
        return dict(
            sup_pairs=sup_pairs,
            sup_src_paths=sup_src_paths,
            sup_tgt_paths=sup_tgt_paths,
            sup_vsrc_paths=sup_vsrc_paths,
            sup_vtgt_paths=sup_vtgt_paths,
            ae_langs=ae_langs,
            ae_src_paths=ae_src_paths,
            ae_tgt_paths=ae_tgt_paths,
            ae_vsrc_paths=ae_vsrc_paths,
            ae_vtgt_paths=ae_vtgt_paths,
        )
    
    def _has_languages(label: str) -> tuple[str | None, str | None]:
        if not isinstance(label, str) or "-" not in label:
            return None, None
        return label.split("-", 1)
        
    def _looks_generated_by_pairs(task_key: str, task: dict) -> bool:
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
    
    # Normalize & resolve reads for this command (keeps style with other commands)
    inputs = resolve_command_inputs("inv_complete_language_pairs", opts)
    doc = opts.in_config[0]  # normalized by resolver

    # We expect 'tasks' to exist and be a mapping
    tasks = doc.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise UserConfigError("No 'tasks' mapping found; cannot invert without command outputs.")

    bag = _collect_from_tasks(tasks)

    # Supervised
    src_tpl  = _generalize_supervised_template(bag["sup_src_paths"], bag["sup_pairs"])
    tgt_tpl  = _generalize_supervised_template(bag["sup_tgt_paths"], bag["sup_pairs"])
    vsrc_tpl = _generalize_supervised_template(bag["sup_vsrc_paths"], bag["sup_pairs"]) if bag["sup_vsrc_paths"] else None
    vtgt_tpl = _generalize_supervised_template(bag["sup_vtgt_paths"], bag["sup_pairs"]) if bag["sup_vtgt_paths"] else None

    # Autoencoder
    have_ae      = bool(bag["ae_langs"])
    have_ae_val  = bool(bag["ae_vsrc_paths"] or bag["ae_vtgt_paths"])
    ae_tpl_cands = []
    if bag["ae_src_paths"]:
        s = _generalize_ae_template(bag["ae_src_paths"], bag["ae_langs"])
        if s: ae_tpl_cands.append(s)
    if bag["ae_tgt_paths"]:
        t = _generalize_ae_template(bag["ae_tgt_paths"], bag["ae_langs"])
        if t: ae_tpl_cands.append(t)
    ae_tpl = ae_tpl_cands[0] if ae_tpl_cands else None

    # Minimal sanity: if we saw supervised pairs, we should infer src/tgt templates
    if bag["sup_pairs"] and (src_tpl is None or tgt_tpl is None):
        raise UserConfigError(
            "Could not infer consistent src/tgt templates from tasks. "
            "Paths may be heterogeneous. Consider providing templates manually:\n"
            "  config_config2 yaml_template --cmd complete_language_pairs --mode inputs"
        )

    # Write under config_config.complete_language_pairs.*
    cc      = doc.setdefault("config_config", {})
    cc_cmd  = cc.setdefault("complete_language_pairs", {})

    if src_tpl is not None:
        cc_cmd["src_path"] = src_tpl
        logger.info(f"[inv] inferred src_path -> {src_tpl}")
    if tgt_tpl is not None:
        cc_cmd["tgt_path"] = tgt_tpl
        logger.info(f"[inv] inferred tgt_path -> {tgt_tpl}")
    if vsrc_tpl is not None:
        cc_cmd["valid_src_path"] = vsrc_tpl
        logger.info(f"[inv] inferred valid_src_path -> {vsrc_tpl}")
    if vtgt_tpl is not None:
        cc_cmd["valid_tgt_path"] = vtgt_tpl
        logger.info(f"[inv] inferred valid_tgt_path -> {vtgt_tpl}")

    if have_ae:
        cc_cmd["autoencoder"] = True
        logger.info(f"[inv] inferred autoencoder -> True")
        if have_ae_val:
            cc_cmd["autoencoder_validation"] = True
            logger.info(f"[inv] inferred autoencoder_validation -> True")
        if ae_tpl is not None:
            cc_cmd["ae_path"] = ae_tpl
            logger.info(f"[inv] inferred ae_path -> {ae_tpl}")
    else:
        # Make the flags explicit if absent
        cc_cmd.setdefault("autoencoder", False)
        cc_cmd.setdefault("autoencoder_validation", False)

    # ---------- PRUNE generated tasks ----------
    removed = []
    kept = []
    tasks = doc.get("tasks")
    if isinstance(tasks, dict):
        for k, v in list(tasks.items()):
            if _looks_generated_by_pairs(k, v):
                tasks.pop(k, None)
                removed.append(k)
            else:
                kept.append(k)

        if not tasks:
            # If the mapping is now empty, drop it entirely
            doc.pop("tasks", None)

    # Log what happened
    if removed:
        logger.info(f"[inv] pruned {len(removed)} task(s) generated by complete_language_pairs")
    if kept:
        logger.info(f"[inv] kept {len(kept)} non-generated task(s)")

    return "[inv] reconstructed complete_language_pairs inputs "



