#!/usr/bin/env python3
# inf_plan.py
#
# Purpose
# -------
# Plan evaluation tasks, generate per-task inference YAML files, and prepare
# command lists for inference and scoring in the eval2 workflow.
#
# This script is the main planning stage of eval2. It reads the training
# configuration and the selected supervised / zero-shot pair lists, expands
# localized evaluation tasks, derives the task-specific inference settings,
# writes one inference YAML per original training task, and prepares the shell
# command files used later for translation and scoring.
#
# Current behavior
# ----------------
# The script:
#   1) reads required environment variables describing the model, training
#      config, output directories, benchmark data, and selected pair lists,
#   2) builds an internal language inventory from the hard-coded TRIPLES table,
#   3) loads the training configuration and extracts the original supervised
#      tasks from its top-level `tasks` mapping,
#   4) expands task language specifications into localized xcodes such as
#      `CA.fra` or `XX.eng`,
#   5) filters the supervised task set against the externally selected
#      supervised pairs,
#   6) validates that the selected tasks are covered by the available benchmark
#      data,
#   7) augments the task set with localized zero-shot tasks derived from the
#      selected zero-shot pairs,
#   8) resolves the sharing groups, vocab codes, transforms, and template tasks
#      needed to build inference configs,
#   9) writes one per-task inference YAML file,
#  10) writes:
#        - calls.out
#        - calls.sacre.out
#        - calls.comet.out
#      for later execution.
#
# Inputs
# ------
# Required environment variables:
#   DATADIR
#       Root directory of benchmark data.
#   OUTDIR
#       Output directory for generated YAML files and call lists.
#   MODEL
#       Model path; later passed to translate.py.
#   TRAINCONFIG
#       Path to the Mammoth training YAML.
#   SUPERVISEDPAIRS
#       File listing supervised evaluation pairs.
#   ZEROSHOTPAIRS
#       File listing zero-shot evaluation pairs.
#   MAMMOTH
#       Mammoth source directory containing translate.py.
#   LOGDIR
#       Directory for inference stderr logs.
#   SCRDIR
#       Directory for score outputs.
#
# Internal inputs:
#   TRIPLES
#       Hard-coded mapping from language codes to benchmark file stems and
#       locale variants.
#   zeroshot_base
#       Hard-coded default zero-shot pairs to merge into the selected set.
#
# Reads
# -----
#   - the training YAML from TRAINCONFIG
#   - supervised pair list from SUPERVISEDPAIRS
#   - zero-shot pair list from ZEROSHOTPAIRS
#   - benchmark files under DATADIR
#
# Writes
# ------
# Under OUTDIR:
#   - one per-task inference YAML file, named <orig_task>.yaml
#   - calls.out
#   - calls.sacre.out
#   - calls.comet.out
#
# Output
# ------
# The script writes progress and diagnostics to stderr, including:
#   - environment validation,
#   - task extraction and filtering,
#   - zero-shot task creation,
#   - inference YAML generation,
#   - planned translation / scoring actions.
#
# Notes
# -----
# - The per-task inference YAMLs are built by copying a subset of the original
#   training config, removing training-only keys, narrowing `tasks`,
#   `src_vocab`, and `tgt_vocab`, and inserting inference-time parameters such
#   as beam size and batch size.
# - Supervised tasks are keyed by the original training task name.
# - Zero-shot tasks are expanded localized tasks but still inherit their YAML
#   template from a real supervised training task.
# - The script is intended to be run from the surrounding Makefile workflow,
#   not manually.

import sys
sys.stderr.write("inf_plan.sh:    running...\n")

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set
import copy
try:
    import yaml
except ImportError:
    sys.stderr.write("Missing dependency: pyyaml\n")
    sys.stderr.write("Install with: pip install pyyaml\n")
    sys.exit(1)

def die(msg: str) -> None:
    log(msg)
    sys.exit(1)
def log(msg: str, before: str = "inf_plan.py: ", end: str = "\n") -> None:
    sys.stderr.write(f"{before}{msg}{end}")

DROP_TOP_LEVEL_KEYS = [
    "early_stopping",
    "valid_timeout",
    "valid_decode_timeout",
    "valid_max_length",
    "tensorboard",
    "tensorboard_log_dir",
    "report_tflops",
    "denoising_objective",
    "mask_ratio",
    "mask_length",
    "poisson_lambda",
    "replace_length",
    "reset_optim",
    "src_seq_length_min",
    "tgt_seq_length_min",
    "src_seq_length_max",
    "tgt_seq_length_max",
    "train_steps",
    "accum_count",
    "lookahead_minibatches",
    "normalization",
    "queue_size",
    "valid_batch_size",
    "optim",
    "learning_rate",
    "adam_beta1",
    "adam_beta2",
    "weight_decay",
    "max_grad_norm",
    "label_smoothing",
    "warmup_steps",
    "decay_method",
    "learning_rate_decay",
    "start_decay_steps",
    "average_decay",
    "n_nodes",
    "task_distribution_strategy",
    "valid_steps",
    "valid_metrics",
    "save_checkpoint_steps",
    "keep_checkpoint",
    "log_model_structure",
    "report_every",
    "report_training_accuracy",
    "save_model",
    "save_strategy",
]

DROP_TASK_KEYS = [
    "path_src",
    "path_tgt",
    "path_valid_src",
    "path_valid_tgt",
]

DEFAULT_BEAM_SIZE = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_BATCH_TYPE = "sents"
DEFAULT_GPU = 0
DEFAULT_WORLD_SIZE = 1
DEFAULT_GPU_RANKS = [0]


zeroshot_base = [
    ("spa",      "cat"),  # Spanish - Catalan
    ("spa",      "eus"),  # Spanish - Basque
    ("spa",      "glg"),  # Spanish - Galician
    ("pol",      "ukr"),  # Polish - Ukrainian
    ("deu",      "tur"),  # German - Turkish
    ("deu",      "ukr"),  # German - Ukrainian
    ("fra",      "nld"),  # French - Dutch
    ("swe",      "fin"),  # Swedish - Finnish
    ("ces",      "slk"),  # Czech - Slovak
    ("bos",      "hrv"),  # Bosnian - Croatian
    ("bos",      "srp_Cyrl"),  # Bosnian - Serbian
    ("hrv",      "srp_Cyrl"),  # Croatian - Serbian
    ("slv",      "bos"),  # Slovene - Bosnian
    ("sqi",      "mkd"),  # Albanian - Macedonian
    ("bul",      "tur"),  # Bulgarian - Turkish
    ("spa",      "ron"),  # Spanish - Romanian
    ("deu",      "ron"),  # German - Romanian
    ("glg",      "por"),  # Galician - Portuguese
    ("eus",      "fra"),  # Basque - French
    ("dan",      "nob"),  # Danish - Norwegian
]
def print_default_zeroshot_pairs(zeroshot_base):
    log(f"   Internal 'zeroshot_base' list has {len(zeroshot_base)} default pairs", end="")
    if len(zeroshot_base):
        i = 0
        for (s,t) in sorted(zeroshot_base):
            if i % 4 == 0:
                log("   ", before="\ninf_plan.py: ", end="")
            p = f"{s}-{t}"
            log(f"  {p:<12}", before="", end="")
            i = i + 1
        log("", before="")

def read_pair_file(path: str) -> list[tuple[str, str]]:
    pairs = set()
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "-" not in line:
                log(f"WARNING: skipping malformed pair line {path}:{lineno}: {line}")
                continue
            src_side, tgt_side = line.split("-", 1)
            src_side = src_side.strip()
            tgt_side = tgt_side.strip()
            if not src_side or not tgt_side:
                log(f"WARNING: skipping malformed pair line {path}:{lineno}: {line}")
                continue
            pairs.add((src_side, tgt_side))
    return pairs

def read_pairs(supervisedpairs, zeroshotpairs):    
    supervised_pair_set = read_pair_file(supervisedpairs)
    zeroshot_pair_set = read_pair_file(zeroshotpairs)
    log(f"   loaded {len(supervised_pair_set)} supervised and {len(zeroshot_pair_set)} 0-shot pairs")
    log(f"✅ Read the task selection files")
    return supervised_pair_set, zeroshot_pair_set

TRIPLES = [
    ("bos", "bos_Latn", ["bs_??"]),
    ("bul", "bul_Cyrl", ["bg_BG"]),
    ("cat", "cat_Latn", ["ca_ES"]),
    ("ces", "ces_Latn", ["cs_CZ"]),
    ("dan", "dan_Latn", ["da_DK"]),
    ("deu", "deu_Latn", ["de_DE"]),
    ("ell", "ell_Grek", ["el_GR"]),
    ("est", "ekk_Latn", ["et_EE"]),
    ("eng", "eng_Latn", ["en_??"]),
    ("eus", "eus_Latn", ["eu_??"]),
    ("fin", "fin_Latn", ["fi_FI"]),
    ("fra", "fra_Latn", ["fr_CA", "fr_FR"]),
    ("gle", "gle_Latn", ["ga_??"]),
    ("glg", "glg_Latn", ["gl_??"]),
    ("hrv", "hrv_Latn", ["hr_HR"]),
    ("hun", "hun_Latn", ["hu_HU"]),
    ("isl", "isl_Latn", ["is_IS"]),
    ("ita", "ita_Latn", ["it_IT"]),
    ("kat", "kat_Geor", ["ka_??"]),
    ("lav", "lvs_Latn", ["lv_LV"]),
    ("lit", "lit_Latn", ["lt_LT"]),
    ("mkd", "mkd_Cyrl", ["mk_??"]),
    ("mlt", "mlt_Latn", ["mt_??"]),
    ("nld", "nld_Latn", ["nl_NL"]),
    ("nno", "nno_Latn", ["nn_??"]),
    ("nob", "nob_Latn", ["no_NO"]),
    ("pol", "pol_Latn", ["pl_PL"]),
    ("por", "por_Latn", ["pt_PT", "pt_BR"]),
    ("ron", "ron_Latn", ["ro_RO"]),
    ("slk", "slk_Latn", ["sk_SK"]),
    ("slv", "slv_Latn", ["sl_SI"]),
    ("spa", "spa_Latn", ["es_MX"]),
    ("sqi", "als_Latn", ["sq_??"]),
    ("srp_Cyrl", "srp_Cyrl", ["sr_RS"]),
    ("swe", "swe_Latn", ["sv_SE"]),
    ("tur", "tur_Latn", ["tr_TR"]),
    ("ukr", "ukr_Cyrl", ["uk_UA"]),
]
@dataclass
class LanguageInventory:
    """
    valid_codes:
        Set of all known language codes
    code_to_ref:
        Language code -> benchmark file stem
        Example: "fra" -> "fra_Latn"
        Used in emit_tasks
    code_to_xcodes:
        Language code -> all localized task-xcode labels
        Example: "fra" -> ["CA.fra", "FR.fra"]
    xcode_to_code:
        Localized xcode label -> language code
        Example: "CA.fra" -> "fra"
    valid_xcodes:
        Set of all known localized xcode labels
    xcode_to_wmt_locale: 
        Localized xcode label -> WMT locale string
        Example: "CA.fra" -> "fr_CA"
        Used in emit_tasks
    """
    valid_codes: Set[str] = field(default_factory=set)
    code_to_ref: Dict[str, str] = field(default_factory=dict)
    code_to_xcodes: Dict[str, List[str]] = field(default_factory=dict)
    xcode_to_code: Dict[str, str] = field(default_factory=dict)
    valid_xcodes: Set[str] = field(default_factory=set)
    xcode_to_wmt_locale: Dict[str, str] = field(default_factory=dict)
def build_language_inventory() -> LanguageInventory:
    def variant_tag_for_locale(locale: str) -> str:
        if "_" in locale:
            tag = locale.rsplit("_", 1)[1]
        else:
            tag = "XX"
        if not tag or tag == "??":
            tag = "XX"
        return tag
    inv = LanguageInventory()
    for code, ref, locales in TRIPLES:
        inv.valid_codes.add(code)
        inv.code_to_ref[code] = ref
        xcodes: List[str] = []
        if not locales:
            log(f"ERROR: no locale for {code}")
            exit(1)
        for locale in locales:
            variant = variant_tag_for_locale(locale)
            if len(locales) == 1:
                variant = "XX"
            xcode = f"{variant}.{code}"
            inv.valid_xcodes.add(xcode)
            inv.xcode_to_code[xcode] = code
            inv.xcode_to_wmt_locale[xcode] = locale
            xcodes.append(xcode)
        inv.code_to_xcodes[code] = xcodes
    return inv

def check_data_coverage(task_set, valid_xcodes):
    pre = "❌ Following designated tasks do not have any _test_ data:\ninf_plan.sh:    "
    for task in sorted(task_set):
        m = TASK_RE.match(task)
        if not m:
            continue
        _, src_xcode, tgt_xcode = m.groups()
        if src_xcode in valid_xcodes and tgt_xcode in valid_xcodes:
            continue
        sys.stderr.write(pre)
        sys.stderr.write(task + "\n")
        pre = "inf_plan.sh:    "
    if pre == "inf_plan.sh:    ":
        log(f"✨ Stopping as I refuse to run before all designated tasks are covered by some _test_ data")
        exit(1)
    log(f"✅ All designated tasks covered (at least partially) by data")   

def require_env(name: str, extra: str = "") -> str:
    value = os.environ.get(name)
    if not value:
        if extra:
            die(f"{name} must be set {extra}")
        die(f"{name} must be set")
    return value

def load_trainconfig(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}

def get_env_vars():    
    datadir = require_env("DATADIR", "to the location of training data")
    outdir = require_env("OUTDIR", "to a subdirectory of the model directory")
    model = require_env("MODEL", "to the model directory; it must end with slash")
    trainconfig = require_env("TRAINCONFIG", "to the training config file")
    supervisedpairs = require_env("SUPERVISEDPAIRS", "to the file listing supervised evaluation pairs")
    zeroshotpairs = require_env("ZEROSHOTPAIRS", "to the file listing zero-shot evaluation pairs")
    mammoth = require_env("MAMMOTH", "to the Mammoth source directory")
    logdir = require_env("LOGDIR", "to the log directory")
    scrdir = require_env("SCRDIR", "to the scoring output directory")
    if not os.path.exists(model):
        die(f"Error: MODEL must be an existing file or directory: {model}")
    if not os.path.isfile(trainconfig):
        die(f"Error: TRAINCONFIG must be an existing file: {trainconfig}")
    if not os.path.isdir(outdir):
        die(f"Error: OUTDIR must be an existing directory: {outdir}")
    if not os.path.isdir(datadir):
        die(f"Error: DATADIR must be an existing directory: {datadir}")
    if not os.path.isfile(supervisedpairs):
        die(f"Error: SUPERVISEDPAIRS must be an existing file: {supervisedpairs}")
    if not os.path.isfile(zeroshotpairs):
        die(f"Error: ZEROSHOTPAIRS must be an existing file: {zeroshotpairs}")
    if not os.path.isdir(mammoth):
        die(f"Error: MAMMOTH must be an existing directory: {mammoth}")
    if not os.path.isdir(logdir):
        die(f"Error: LOGDIR must be an existing directory: {logdir}")
    if not os.path.isdir(scrdir):
        die(f"Error: SCRDIR must be an existing directory: {scrdir}")
        
    inventory = build_language_inventory()
    log(f"✅ Loaded internal language codes")
    log(f"✅ Loaded internal default zeroshot pairs")   
    print_default_zeroshot_pairs(zeroshot_base)
    cfg = load_trainconfig(trainconfig)
    log(f"✅ Loaded the training config file {trainconfig}")
    log(f"✨ Environment validated")           
    return (datadir, outdir, model, trainconfig,
            supervisedpairs, zeroshotpairs,
            mammoth, logdir, scrdir, cfg, inventory, )

class CallWriters:
    def __init__(self, outdir: str):
        self.calls_path = os.path.join(outdir, "calls.out") # file
        self.sacre_path = os.path.join(outdir, "calls.sacre.out") # file
        self.comet_path = os.path.join(outdir, "calls.comet.out") # file

        self.calls = open(self.calls_path, "w", encoding="utf-8")
        self.sacre = open(self.sacre_path, "w", encoding="utf-8")
        self.comet = open(self.comet_path, "w", encoding="utf-8")

    def close(self):
        self.calls.close()
        self.sacre.close()
        self.comet.close()

def validate_calls_out(outdir: str):
    calls_path = os.path.join(outdir, "calls.out") # file
    if not os.path.exists(calls_path):
        return
    bad = []
    with open(calls_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if line and not line.startswith("python"):
                bad.append((i, line))
    if bad:
        log("❌ FAIL: not every line in calls.out starts with 'python'")
        for i, line in bad:
            log(f"  line {i}: {line}")
        raise ValueError("calls.out contains non-python lines")
    log("✅ Every line in calls.out starts with 'python'")
    
def files_available(dataset, input_path, refer_path):
    log(f"##    {dataset}")
    if not os.path.exists(input_path):
        log(f"##      FAIL  - Source file    missed: {input_path}")
        return False
    else:
        log(f"##      HAVE  - Source file     found: {input_path}")
    if not os.path.exists(refer_path):
        log(f"##      FAIL  - Reference file missed: {refer_path}")
        return False
    else:
        log(f"##      HAVE  - Reference file  found: {refer_path}")
    if not os.path.exists(config_path):
        log(f"##      FAIL  - Config file    missed: {config_path}")
        return False
    else:
        log(f"##      HAVE  - Config file     found: {config_path}")
    return True 

def files_available(dataset, input_path, refer_path, config_file):
    log(f"##    {dataset}")
    if not os.path.exists(input_path):
        log(f"##      FAIL  - Source file    missed: {input_path}")
        return False
    else:
        log(f"##      HAVE  - Source file     found: {input_path}")
    if not os.path.exists(refer_path):
        log(f"##      FAIL  - Reference file missed: {refer_path}")
        return False
    else:
        log(f"##      HAVE  - Reference file  found: {refer_path}")
    return True 
    if not os.path.exists(config_path):
        log(f"##      FAIL - Config file does not exist: {config_path}")
        return False
    else:
        log(f"##      HAVE  - Config file found        : {config_path}")

def plan_translation_and_scoring(
    *,
    writers: CallWriters,
    dataset: str,
    input_path: str,
    refer_path: str,
    config_path: str,
    model_path: str,
    mammoth_dir: str,
    logdir: str,
    scrdir: str,
    outdir: str,
    xtask: str,
    orig_task: str,
    data_tag: str,
    pair_type: str,
) -> bool:
    if pair_type == "zeroshot":
        shortoutput = f"{xtask}.{data_tag}.0shyp"
    else:
        shortoutput = f"{xtask}.{data_tag}.hyp"
    output_path = os.path.join(outdir, shortoutput)
    if not files_available(dataset, input_path, refer_path, config_path):
         return False

    if os.path.exists(output_path):
        with open(input_path, "r", encoding="utf-8") as f:
            input_lines = sum(1 for _ in f)
        with open(output_path, "r", encoding="utf-8") as f:
            output_lines = sum(1 for _ in f)

        if input_lines == output_lines:
            sacre_path = os.path.join(scrdir, f"{xtask}.{data_tag}.sacre")
            comet_path = os.path.join(scrdir, f"{xtask}.{data_tag}.comet")

            if os.path.exists(sacre_path) and os.path.getsize(sacre_path) > 500:
                log(f"##      GOOD - already scored: {sacre_path}")
            else:
                log(f"##      DONE - ({input_lines} lines on both) -- ready to measure")
                log(f"##             Contrasting {refer_path} VS {output_path}")

                writers.sacre.write(
                    f'if [ -s "{sacre_path}" ] && [ "$(stat -c%s "{sacre_path}")" -gt 500 ]; then\n')
                writers.sacre.write(
                    f'    echo "Skipping {sacre_path} because it already exists and is >500 bytes"\n')
                writers.sacre.write("else\n")
                writers.sacre.write(f'    date | tee "{sacre_path}"\n')
                writers.sacre.write(
                    f'    echo {data_tag} {xtask} Contrasting {refer_path} VS {output_path} | tee -a "{sacre_path}"\n')
                writers.sacre.write(
                    f'    sacrebleu {refer_path} -i {output_path} -m bleu chrf | tee -a "{sacre_path}"\n')
                writers.sacre.write("fi\n")

            writers.comet.write(
                f'comet-score -r "{refer_path}" -t "{output_path}" -s "{input_path}" --gpus 1 | tee "{comet_path}"\n'
            )
            return True

        log(f"##      REDO - ({input_lines} vs {output_lines} lines; re-translate) ==> {shortoutput}")
    else:
        log(f"##      TODO - (translate now) ==> {shortoutput}")
        log(f"##      TODO  - translate=> {output_path}")

    writers.calls.write(
        f'python -u "{mammoth_dir}/translate.py" '
        f'--config "{config_path}" --model "{model_path}" --task_id "{orig_task}" '
        f'--src "{input_path}" --output "{output_path}" '
        f'2>"{logdir}/job${{SLURM_JOB_ID}}.{xtask}.{data_tag}.err"\n'
    )
    return True


@dataclass
class TaskSupport:
    """
    orig_task_set:
        Set of original task names in the training config
    task_set:
        Set of expanded localized tasks such as "mt_CA.fra-XX.eng"
    task_to_type:
        Expanded localized task -> "supervised" or "zeroshot"
    task_to_orig:
        Expanded localized task -> original yaml task name
        Example: "mt_CA.fra-XX.eng" -> "mt_fra-eng"
    task_to_surr:
        Expanded localized task -> real training task used as surrogate
        Example: "mt_BR.por-XX.cat" -> "mt_por-fra"
    src_code_to_prefixes:
        Source language code -> task families seen in TRAINCONFIG
        Example: "fra" -> {"mt", "docmt"}
    tgt_code_to_prefixes:
        Target language code -> task families seen in TRAINCONFIG
        Example: "eng" -> {"mt", "sentmt", "docmt"}
    src_vocab_codes:
        Vocabs in training.config
    tgt_vocab_codes:
        Vocabs in training.config
    """
    orig_task_names: List[str] = field(default_factory=list)
    task_set: Set[str] = field(default_factory=set)
    task_to_type: Dict[str, str] = field(default_factory=dict)
    task_to_orig: Dict[str, str] = field(default_factory=dict)
    task_to_surr: Dict[str, str] = field(default_factory=dict)
    src_code_to_prefixes: Dict[str, Set[str]] = field(default_factory=dict)
    tgt_code_to_prefixes: Dict[str, Set[str]] = field(default_factory=dict)
    src_vocab_codes: Set[str] = field(default_factory=set)
    tgt_vocab_codes: Set[str] = field(default_factory=set)
    task_to_src_vocab: Dict[str, str] = field(default_factory=dict)
    task_to_tgt_vocab: Dict[str, str] = field(default_factory=dict)
    src_prefix_to_surrogate: Dict[tuple[str, str], str] = field(default_factory=dict)
    tgt_prefix_to_surrogate: Dict[tuple[str, str], str] = field(default_factory=dict)
    src_prefix_to_tasks: Dict[tuple[str, str], str] = field(default_factory=dict)
    tgt_prefix_to_tasks: Dict[tuple[str, str], str] = field(default_factory=dict)
    task_to_src_candidates: Dict[str, List[str]] = field(default_factory=dict)
    task_to_tgt_candidates: Dict[str, List[str]] = field(default_factory=dict)
    task_to_enc_group: Dict[str, object] = field(default_factory=dict)
    task_to_dec_group: Dict[str, object] = field(default_factory=dict)
    task_to_src_vocab_code: Dict[str, str] = field(default_factory=dict)
    task_to_tgt_vocab_code: Dict[str, str] = field(default_factory=dict)
    task_to_transforms: Dict[str, object] = field(default_factory=dict)
    task_to_errors: Dict[str, List[str]] = field(default_factory=dict)
    
def collect_task_support(cfg, inventory):
    def expand_langspec(spec: str) -> list[str]:
        if "." in spec:
            return [spec] if spec in inventory.valid_xcodes else []
        return list(inventory.code_to_xcodes.get(spec, []))
    train_tasks = cfg.get("tasks", {})
    support = TaskSupport()
    support.orig_task_names = list(cfg.get("tasks", {}).keys())
    support.src_vocab_codes = set(cfg.get("src_vocab", {}).keys())
    support.tgt_vocab_codes = set(cfg.get("tgt_vocab", {}).keys())
    for orig_task in support.orig_task_names:
        m = TASK_RE.match(orig_task)
        if not m:
            log(f"❌ Unrecognized task format in YAML: {orig_task}")
            continue
        prefix, src_spec, tgt_spec = m.groups() # We get the language pairs from task name - correct!
        src_xcodes = expand_langspec(src_spec)
        tgt_xcodes = expand_langspec(tgt_spec)
        if not src_xcodes or not tgt_xcodes:
            log(f"❌ Could not expand YAML task '{orig_task}'")
            continue
        task_cfg = train_tasks.get(orig_task)
        src_tgt = task_cfg.get("src_tgt")
        if not src_tgt or "-" not in src_tgt:
            log(f"❌ Missing or malformed src_tgt in TRAINCONFIG task: {yaml_task}")
            continue
        train_src_vocab, train_tgt_vocab = src_tgt.split("-", 1)
        enc_group = task_cfg.get("enc_sharing_group", [])
        dec_group = task_cfg.get("dec_sharing_group", [])
        transforms = task_cfg.get("transforms")
        if transforms is None:
            transforms = task_cfg.get("train_transforms", [])
        for src_xcode in src_xcodes:
            for tgt_xcode in tgt_xcodes:
                xtask = f"{prefix}_{src_xcode}-{tgt_xcode}"
                support.task_set.add(xtask)
                support.task_to_type[xtask] = "supervised"
                support.task_to_orig[xtask] = orig_task
                src_code = inventory.xcode_to_code[src_xcode]
                tgt_code = inventory.xcode_to_code[tgt_xcode]
                support.task_to_src_vocab[xtask] = train_src_vocab
                support.task_to_tgt_vocab[xtask] = train_tgt_vocab
                support.task_to_src_vocab_code[xtask] = train_src_vocab
                support.task_to_tgt_vocab_code[xtask] = train_tgt_vocab
                support.src_code_to_prefixes.setdefault(src_code, set()).add(prefix)
                support.tgt_code_to_prefixes.setdefault(tgt_code, set()).add(prefix)
                support.src_prefix_to_tasks.setdefault((prefix, src_code), set()).add(orig_task)
                support.tgt_prefix_to_tasks.setdefault((prefix, tgt_code), set()).add(orig_task)
                support.task_to_enc_group[xtask] = enc_group
                support.task_to_dec_group[xtask] = dec_group
                support.task_to_transforms[xtask] = transforms
                support.task_to_errors[xtask] = []
                support.task_to_src_candidates[xtask] = [orig_task]
                support.task_to_tgt_candidates[xtask] = [orig_task]
   
    log(f"✅ Extracting tasks from training config and combining the codes and the tasks")
    return support

def filter_zeroshotpairs(supervised_pair_set, zeroshot_pair_set):
    added = 0
    log(f"   Adding manually prescribed zero-shot pairs:", end="")
    for (src_xcode,tgt_xcode) in zeroshot_base:
        if (src_xcode,tgt_xcode) not in zeroshot_pair_set:
            zeroshot_pair_set.add((src_xcode,tgt_xcode))
            if added % 4 == 0:
                log("   ", before="\ninf_plan.py: ", end="")
            p = f"{src_xcode}-{tgt_xcode}"
            log(f"  {p:<12}", before="", end="")
            added = added + 1
    log("", before="")
    skipped = 0
    log(f"   The following pairs were already in the fixed set:", end="")
    for (src_xcode,tgt_xcode) in zeroshot_base:
        if (src_xcode,tgt_xcode) in zeroshot_pair_set:
            if skipped % 4 == 0:
                log("   ", before="\ninf_plan.py: ", end="")
            p = f"{src_xcode}-{tgt_xcode}"
            log(f"  {p:<12}", before="", end="")
            skipped = skipped + 1
    log("", before="")
    overlaps = 0
    overlap_pairs = supervised_pair_set & zeroshot_pair_set
    if overlap_pairs:
        log("   The following zero-shot pairs are indeed supervised in this model:", end="")
        for p in sorted(list(overlap_pairs)):
            if overlaps % 4 == 0:
                log("   ", before="\ninf_plan.py: ", end="")
            p = f"{p[0]}-{p[1]}"
            log(f"  {p:<12}", before="", end="")
            overlaps = overlaps + 1
        zeroshot_pair_set = zeroshot_pair_set - overlap_pairs
    log("", before="")
    zeroshot_pair_set = zeroshot_pair_set - overlap_pairs
    remain = 0
    log(f"   The final set of zero-shot pairs (* not fixed):", end="")
    for (s,t) in sorted(list(zeroshot_pair_set)):
        if remain % 4 == 0:
            log("   ", before="\ninf_plan.py: ", end="")
        p = f"{s}-{t}"
        if (s,t) in zeroshot_base:
            log(f"  {p:<12}", before="", end="")
        else:
            log(f" *{p:<12}", before="", end="")
        remain = remain + 1
    log("", before="")
    log(f"✅ Filtered the zeroshot tasks selection")
    return zeroshot_pair_set

TASK_RE  = re.compile(r"^(mt|sentmt|docmt)_([^-]+)-([^-]+)$")
TASK_REX = re.compile(r"^(mt|sentmt|docmt)_[A-Z][A-Z].([^-_]+)[^-_]*-[A-Z][A-Z].([^-_]+)[^-_]*$")

def filter_supervised_tasks(support: TaskSupport, supervised_pair_set, inventory: LanguageInventory,):
    filtered_task_set = set()
    filtered_task_to_orig = {}
    filtered_task_to_surr = {}
    filtered_task_to_type = {}
    rejected = 0
    for task in sorted(support.task_set):
        m = TASK_RE.match(task)
        if not m:
            continue
        _, src_xcode, tgt_xcode = m.groups()
        src_code = inventory.xcode_to_code.get(src_xcode)
        tgt_code = inventory.xcode_to_code.get(tgt_xcode)
        if src_code is None or tgt_code is None:
            log(f"❌ Could not normalize task xcodes: {task}")
            rejected += 1
            continue
        if (src_code, tgt_code) in supervised_pair_set:
            filtered_task_set.add(task)
            filtered_task_to_orig[task] = support.task_to_orig.get(task, task)
            filtered_task_to_surr[task] = support.task_to_surr.get(task, support.task_to_orig.get(task, task))
            filtered_task_to_type[task] = "supervised"
        else:
            log(f"❌ Pair {src_code}-{tgt_code} is not among the desirable supervised pairs")
            rejected += 1
    log(f"   Dropped {rejected} supervised evaluation tasks by filters")
    log(f"   Kept {len(filtered_task_set)} supervised evaluation tasks after filtering")
    support.task_set = filtered_task_set
    support.task_to_orig = filtered_task_to_orig
    support.task_to_surr = filtered_task_to_surr
    support.task_to_type = filtered_task_to_type
    log(f"✅ Filtered the supervised tasks selection")
    return support, rejected

def produce_infyamls_and_calls(inventory, support, datadir, outdir, trainconfig, cfg, writers,
                               model, mammoth, logdir, scrdir,):
    log(f"producing the testing tasks (stdout) ...", end="")
    for xtask in sorted(support.task_set):
        emit_tasks(xtask=xtask, inventory=inventory, support=support, 
                   datadir=datadir, outdir=outdir, cfg_path=trainconfig, cfg=cfg,
                   writers=writers,  model=model, mammoth=mammoth, logdir=logdir, scrdir=scrdir,)
    log(f'✨ Completed producing inference config files')

def emit_tasks(*, xtask: str, support: TaskSupport, 
               inventory: LanguageInventory, datadir: str, outdir: str, 
               cfg_path: str,cfg: dict, writers: CallWriters,
               model: str, mammoth: str, logdir: str, scrdir: str,) -> bool:
    pair_type = support.task_to_type.get(xtask, "unknown")
    orig_task = support.task_to_orig.get(xtask, xtask)
    if pair_type == "supervised":
        log(f"## {xtask} [{pair_type}]:  ")
    else:
        log(f"## {xtask} [{pair_type}]: ********** 0-SHOT **********")
    m = TASK_RE.match(xtask)
    if not m:
        log(f"  Skipping malformed task: {xtask}")
        return False
    _, src_xcode, tgt_xcode = m.groups()
    src = inventory.xcode_to_code.get(src_xcode)
    tgt = inventory.xcode_to_code.get(tgt_xcode)
    if src is None or tgt is None:
        log(f"  Skipping {xtask}: could not decode language xcode(s)")
        exit(1)
    if src not in inventory.code_to_ref or tgt not in inventory.code_to_ref:
        log(f"  Skipping {xtask}: unknown language code(s)")
        exit(1)
    src_ref = inventory.code_to_ref[src]
    tgt_ref = inventory.code_to_ref[tgt]
    src_wmt = inventory.xcode_to_wmt_locale.get(src_xcode, "")
    tgt_wmt = inventory.xcode_to_wmt_locale.get(tgt_xcode, "")
    if pair_type == "supervised":
        inf_yaml_path_file = f"{outdir}/{orig_task}.yaml"
        inf_yaml_path = f"{outdir}/{orig_task}.yaml"
    else:
        inf_yaml_path_file = f"{outdir}/{orig_task}.0shot.yaml"
        inf_yaml_path = f"{outdir}/{orig_task}.0shot.yaml"
        
    write_inference_yaml(
        src_code=src, tgt_code=tgt, pair_type=pair_type,
        xtask=xtask, orig_task=orig_task, cfg=cfg,
        train_cfg=cfg_path, inf_yaml_path=inf_yaml_path_file, outdir=outdir, support=support)

    flo_input = os.path.join(datadir, "flores_plus", "devtest", f"{src_ref}.txt")
    flo_refer = os.path.join(datadir, "flores_plus", "devtest", f"{tgt_ref}.txt")
    plan_translation_and_scoring(
        writers=writers, dataset="Flores+", data_tag="flo", input_path=flo_input, refer_path=flo_refer,
        config_path=inf_yaml_path, model_path=model, mammoth_dir=mammoth,
        logdir=logdir, scrdir=scrdir, outdir=outdir,
        xtask=xtask, orig_task=orig_task, pair_type=pair_type)
    
    bqt_input = os.path.join(datadir, "bouquet", "test", f"{src_ref}.txt")
    bqt_refer = os.path.join(datadir, "bouquet", "test", f"{tgt_ref}.txt")
    plan_translation_and_scoring(
        writers=writers, dataset="BOUQuET", data_tag="bqt", input_path=bqt_input, refer_path=bqt_refer,
        config_path=inf_yaml_path, model_path=model, mammoth_dir=mammoth,
        logdir=logdir, scrdir=scrdir, outdir=outdir,
        xtask=xtask, orig_task=orig_task, pair_type=pair_type)

    bqtpar_input = os.path.join(datadir, "bouquet_par", "test", f"{src_ref}.txt")
    bqtpar_refer = os.path.join(datadir, "bouquet_par", "test", f"{tgt_ref}.txt")
    plan_translation_and_scoring(
        writers=writers, dataset="BOUQuETpar", data_tag="bqtpar", input_path=bqtpar_input, refer_path=bqtpar_refer,
        config_path=inf_yaml_path, model_path=model, mammoth_dir=mammoth,
        logdir=logdir, scrdir=scrdir, outdir=outdir,
        xtask=xtask, orig_task=orig_task, pair_type=pair_type)
    
    if src == "eng":
        wmt_input = os.path.join(datadir, "wmt24pp", "train", f"en-{tgt_wmt}", "source.en.txt")
        wmt_refer = os.path.join(datadir, "wmt24pp", "train", f"en-{tgt_wmt}", "reference.target.txt")
    elif tgt == "eng":
        wmt_input = os.path.join(datadir, "wmt24pp", "train", f"en-{src_wmt}", "reference.target.txt")
        wmt_refer = os.path.join(datadir, "wmt24pp", "train", f"en-{src_wmt}", "source.en.txt")
    else:
        wmt_input = os.path.join(datadir, "wmt24pp", "train", f"en-{src_wmt}", "reference.target.txt")
        wmt_refer = os.path.join(datadir, "wmt24pp", "train", f"en-{tgt_wmt}", "reference.target.txt")
        
    plan_translation_and_scoring(
        writers=writers, dataset="WMT24++", data_tag="wmt", input_path=wmt_input, refer_path=wmt_refer,
        config_path=inf_yaml_path, model_path=model, mammoth_dir=mammoth,
        logdir=logdir, scrdir=scrdir, outdir=outdir,
        xtask=xtask, orig_task=orig_task, pair_type=pair_type)
    
    return True

def squash_debug_value(v):
    if isinstance(v, list):
        if len(v) == 0:
            return []
        if len(v) == 1:
            return v[0]
    return v

def choose_template_task(cfg: dict, support: TaskSupport, xtask: str, orig_task: str) -> str:
    train_tasks = cfg.get("tasks", {})

    # Supervised case: original task should already exist in TRAINCONFIG.
    if orig_task in train_tasks:
        return orig_task

    # Zeroshot case: use a real training task as template.
    src_candidates = support.task_to_src_candidates.get(xtask, [])
    tgt_candidates = support.task_to_tgt_candidates.get(xtask, [])

    if src_candidates:
        return src_candidates[0]
    if tgt_candidates:
        return tgt_candidates[0]

    raise KeyError(f"No template task available for {xtask}")


def build_inference_config_from_template(
    *,
    cfg: dict,
    support: TaskSupport,
    xtask: str,
    orig_task: str,
    src_code: str,
    tgt_code: str,
    pair_type: str,
    train_cfg_path: str,
) -> dict:
    train_tasks    = cfg.get("tasks", {})
    template_task  = choose_template_task(cfg, support, xtask, orig_task)
    template_cfg   = copy.deepcopy(train_tasks[template_task])
    enc_group      = support.task_to_enc_group[xtask]
    dec_group      = support.task_to_dec_group[xtask]
    src_vocab_code = support.task_to_src_vocab_code[xtask]
    tgt_vocab_code = support.task_to_tgt_vocab_code[xtask]
    transforms     = support.task_to_transforms[xtask]
    src_candidates = support.task_to_src_candidates.get(xtask, [])
    tgt_candidates = support.task_to_tgt_candidates.get(xtask, [])
    infer_cfg      = copy.deepcopy(cfg)
    for key in DROP_TOP_LEVEL_KEYS:
        infer_cfg.pop(key, None)
    for key in DROP_TASK_KEYS:
        template_cfg.pop(key, None)
    # Overwrite task-specific fields with the values resolved by inf_plan.py
    template_cfg["src_tgt"] = f"{src_vocab_code}-{tgt_vocab_code}"
    template_cfg["enc_sharing_group"] = enc_group
    template_cfg["dec_sharing_group"] = dec_group
    # Old extractor behavior: remove filtertoolong and then clear transforms.
    # template_cfg["transforms"] = [
    #     t for t in template_cfg.get("transforms", [])
    #     if t != "filtertoolong"
    # ]
    template_cfg["transforms"] = []
    template_cfg["node_gpu"] = "0:0"
    template_cfg["_template_task"] = template_task
    template_cfg["_pair_type"] = pair_type
    template_cfg["_src_task_candidates"] = src_candidates
    template_cfg["_tgt_task_candidates"] = tgt_candidates
    template_cfg["_original_train_task"] = orig_task
    for k in [
            "_template_task",
            "_pair_type",
            "_src_task_candidates",
            "_tgt_task_candidates",
            "_original_train_task",]:
        if k in template_cfg:
            template_cfg[k] = squash_debug_value(template_cfg[k])
        
    src_vocab = infer_cfg.get("src_vocab", {})
    tgt_vocab = infer_cfg.get("tgt_vocab", {})
    if src_vocab_code not in src_vocab:
        raise KeyError(f"source vocab code {src_vocab_code!r} missing from src_vocab")
    if tgt_vocab_code not in tgt_vocab:
        raise KeyError(f"target vocab code {tgt_vocab_code!r} missing from tgt_vocab")

    # Narrow vocabularies exactly like the old extractor:
    # keep only the entries needed by this task.
    infer_cfg["src_vocab"] = {src_vocab_code: src_vocab[src_vocab_code]}
    infer_cfg["tgt_vocab"] = {tgt_vocab_code: tgt_vocab[tgt_vocab_code]}

    # IMPORTANT:
    # Use orig_task as the only task key, because translate.py is later called with:
    #   --task_id "{orig_task}"
    infer_cfg["tasks"] = {orig_task: template_cfg}

    # Copy the inference-time parameters from the old extractor.
    infer_cfg["beam_size"] = DEFAULT_BEAM_SIZE
    infer_cfg["batch_size"] = DEFAULT_BATCH_SIZE
    infer_cfg["batch_type"] = DEFAULT_BATCH_TYPE
    infer_cfg["gpu"] = DEFAULT_GPU
    infer_cfg["world_size"] = DEFAULT_WORLD_SIZE
    infer_cfg["gpu_ranks"] = DEFAULT_GPU_RANKS

    # Keep these if useful for debugging / traceability.
    infer_cfg["task_id"] = xtask
    infer_cfg["_expanded_task"] = xtask
    infer_cfg["_train_config"] = train_cfg_path

    return infer_cfg


def write_inference_yaml(*, src_code: str, tgt_code: str, pair_type: str, 
                         xtask: str, orig_task: str, cfg: dict,
                         train_cfg: str, inf_yaml_path: str, outdir: str, support: TaskSupport) -> bool:
    enc_group      = support.task_to_enc_group[xtask]
    dec_group      = support.task_to_dec_group[xtask]
    src_vocab_code = support.task_to_src_vocab_code[xtask]
    tgt_vocab_code = support.task_to_tgt_vocab_code[xtask]
    transforms     = support.task_to_transforms[xtask]
    src_candidates = support.task_to_src_candidates.get(xtask, [])
    tgt_candidates = support.task_to_tgt_candidates.get(xtask, [])
    log(f"##      TASK  - sharing groups: {enc_group} x {dec_group}")
    log(f"##      TASK  - src_tght codes: {src_vocab_code}-{tgt_vocab_code}")
    log(f"##      TASK  - transforms:     {transforms}")
    if pair_type == "zeroshot":
        log(f"##      GRND  - src candidates: {src_candidates}")
        log(f"##      GRND  - tgt candidates: {tgt_candidates}")
    infer_cfg = build_inference_config_from_template(
        cfg=cfg,
        support=support,
        xtask=xtask,
        orig_task=orig_task,
        src_code=src_code,
        tgt_code=tgt_code,
        pair_type=pair_type,
        train_cfg_path=train_cfg,
    )
    with open(inf_yaml_path, "w", encoding="utf-8") as f:
         yaml.safe_dump(infer_cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False,)

    log(f"##      WROTE - infer.yaml: {inf_yaml_path}")
    return True

def unique_or_error(values, label, task):
    vals = list(values)
    if len(vals) == 0:
        return None, [f"missing {label}"]
    if len(vals) == 1:
        return vals[0], []
    return f"***ERROR*** ambiguous {label}: {sorted(vals)}", [f"ambiguous {label}: {sorted(vals)}"]

def add_zeroshot_tasks(cfg, zeroshot_pair_set, support: TaskSupport, inventory: LanguageInventory,):
    
    def get_zeroshot_prefix(support, src, tgt):
        prefixes = (support.src_code_to_prefixes.get(src, set()) & support.tgt_code_to_prefixes.get(tgt, set()) )
        if not prefixes:
            log(f"  ❌ zero-shot pair {src}-{tgt} not supported by any existing task family in TRAINCONFIG")
            return None
        else:
            log(f"  ✅ zero-shot pair {src}-{tgt} is supported by an existing task family in TRAINCONFIG: {prefixes}")
        return prefixes
    
    def get_lang_xcodes(inventory, lang):
        return inventory.code_to_xcodes[lang]

    log(f"🛠️ Adding localized zero-shot evaluation tasks:")
    added_zeroshot = 0    
    for src, tgt in sorted(zeroshot_pair_set):
        task_prefixes = get_zeroshot_prefix(support, src, tgt)
        if not task_prefixes:
            continue
        task_prefixes = sorted(task_prefixes)
        src_xcodes = get_lang_xcodes(inventory, src)
        tgt_xcodes = get_lang_xcodes(inventory, tgt)
        log(f" {src}-{tgt}: {task_prefixes} x {src_xcodes} x {tgt_xcodes}")
        for prefix in task_prefixes:
            for xsrc in src_xcodes:
                for xtgt in tgt_xcodes:
                    task = f"{prefix}_{xsrc}-{xtgt}"
                    if task not in support.task_set:
                        add_zeroshot_task(support, cfg, prefix, xsrc, xtgt, inventory)
                        added_zeroshot += 1
    log(f"   Added {added_zeroshot} zero-shot evaluation tasks")
    log(f'✨ Completed adding the zeroshot tasks')
    return added_zeroshot

def add_zeroshot_task(support, cfg, prefix, xsrc, xtgt, inventory):
    xtask = f"{prefix}_{xsrc}-{xtgt}"

    log(f"")
    log(f"--- add_zeroshot_task: {xtask} ---")

    # Decode source/target codes from the xcodes
    src_code = inventory.xcode_to_code[xsrc]
    tgt_code = inventory.xcode_to_code[xtgt]

    # Find candidate training tasks
    src_candidates = sorted(support.src_prefix_to_tasks.get((prefix, src_code), set()))
    tgt_candidates = sorted(support.tgt_prefix_to_tasks.get((prefix, tgt_code), set()))

    log(f"adding zero-shot task {xtask}")
    log(f"  xtask          = {xtask}")
    log(f"  prefix         = {prefix}")
    log(f"  xcodes         = ({xsrc}, {xtgt})")
    log(f"  language codes = ({src_code}, {tgt_code})")
    log(f"  src_candidates = {src_candidates}")
    log(f"  tgt_candidates = {tgt_candidates}")

    errors = []
    if not src_candidates:
        msg = f"no source-side candidates for ({prefix}, {src_code})"
        log(f"ERROR: {msg}")
        errors.append(msg)
    if not tgt_candidates:
        msg = f"no target-side candidates for ({prefix}, {tgt_code})"
        log(f"ERROR: {msg}")
        errors.append(msg)

    train_tasks = cfg.get("tasks", {})

    # ------------------------------------------------------------
    # Source-side properties
    # ------------------------------------------------------------
    src_enc_groups = set()
    src_vocab_codes = set()

    log("scanning source-side candidates...")
    for t in src_candidates:
        task_cfg = train_tasks[t]
        enc_group = tuple(task_cfg.get("enc_sharing_group", []))
        src_enc_groups.add(enc_group)

        src_tgt = task_cfg.get("src_tgt")
        if src_tgt and "-" in src_tgt:
            s, _ = src_tgt.split("-", 1)
            src_vocab_codes.add(s)
        else:
            s = None

        log(
            f"  SRC candidate {t}: "
            f"src_tgt={src_tgt}, "
            f"enc_group={list(enc_group)}, "
            f"src_vocab_code={s}"
        )

    log(f"src_enc_groups = {[list(x) for x in sorted(src_enc_groups)]}")
    log(f"src_vocab_codes = {sorted(src_vocab_codes)}")

    # ------------------------------------------------------------
    # Target-side properties
    # ------------------------------------------------------------
    tgt_dec_groups = set()
    tgt_vocab_codes = set()

    log("scanning target-side candidates...")
    for t in tgt_candidates:
        task_cfg = train_tasks[t]
        dec_group = tuple(task_cfg.get("dec_sharing_group", []))
        tgt_dec_groups.add(dec_group)

        src_tgt = task_cfg.get("src_tgt")
        if src_tgt and "-" in src_tgt:
            _, tcode = src_tgt.split("-", 1)
            tgt_vocab_codes.add(tcode)
        else:
            tcode = None

        log(
            f"  TGT candidate {t}: "
            f"src_tgt={src_tgt}, "
            f"dec_group={list(dec_group)}, "
            f"tgt_vocab_code={tcode}"
        )

    log(f"tgt_dec_groups = {[list(x) for x in sorted(tgt_dec_groups)]}")
    log(f"tgt_vocab_codes = {sorted(tgt_vocab_codes)}")

    # ------------------------------------------------------------
    # Shared transforms from both sides
    # ------------------------------------------------------------
    transform_sets = set()

    log("scanning transforms from all candidates...")
    for t in src_candidates + tgt_candidates:
        task_cfg = train_tasks[t]
        transforms = task_cfg.get("transforms")
        if transforms is None:
            transforms = task_cfg.get("train_transforms", [])
        transform_sets.add(tuple(transforms))

        log(f"  TRANSFORMS candidate {t}: transforms={transforms}")

    log(f"transform_sets = {[list(x) for x in sorted(transform_sets)]}")

    # ------------------------------------------------------------
    # Resolve uniqueness
    # ------------------------------------------------------------
    enc_group, errs = unique_or_error(src_enc_groups, "enc_sharing_group", xtask)
    if errs:
        log(f"enc_group resolution errors: {errs}")
    else:
        log(f"enc_group resolved to: {enc_group}")
    errors.extend(errs)

    dec_group, errs = unique_or_error(tgt_dec_groups, "dec_sharing_group", xtask)
    if errs:
        log(f"dec_group resolution errors: {errs}")
    else:
        log(f"dec_group resolved to: {dec_group}")
    errors.extend(errs)

    src_vocab_code, errs = unique_or_error(src_vocab_codes, "src_vocab_code", xtask)
    if errs:
        log(f"src_vocab_code resolution errors: {errs}")
    else:
        log(f"src_vocab_code resolved to: {src_vocab_code}")
    errors.extend(errs)

    tgt_vocab_code, errs = unique_or_error(tgt_vocab_codes, "tgt_vocab_code", xtask)
    if errs:
        log(f"tgt_vocab_code resolution errors: {errs}")
    else:
        log(f"tgt_vocab_code resolved to: {tgt_vocab_code}")
    errors.extend(errs)

    transforms, errs = unique_or_error(transform_sets, "transforms", xtask)
    if errs:
        log(f"transforms resolution errors: {errs}")
    else:
        log(f"transforms resolved to: {transforms}")
    errors.extend(errs)

    # ------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------
    support.task_set.add(xtask)
    support.task_to_type[xtask] = "zeroshot"
    support.task_to_orig[xtask] = f"{prefix}_{src_code}-{tgt_code}"

    support.task_to_src_candidates[xtask] = src_candidates
    support.task_to_tgt_candidates[xtask] = tgt_candidates

    support.task_to_enc_group[xtask] = list(enc_group) if isinstance(enc_group, tuple) else enc_group
    support.task_to_dec_group[xtask] = list(dec_group) if isinstance(dec_group, tuple) else dec_group
    support.task_to_src_vocab_code[xtask] = src_vocab_code
    support.task_to_tgt_vocab_code[xtask] = tgt_vocab_code
    support.task_to_transforms[xtask] = list(transforms) if isinstance(transforms, tuple) else transforms
    support.task_to_errors[xtask] = errors

    log("stored support values:")
    log(f"  task_to_type[{xtask}] = {support.task_to_type[xtask]}")
    log(f"  task_to_orig[{xtask}] = {support.task_to_orig[xtask]}")
    log(f"  task_to_src_candidates[{xtask}] = {support.task_to_src_candidates[xtask]}")
    log(f"  task_to_tgt_candidates[{xtask}] = {support.task_to_tgt_candidates[xtask]}")
    log(f"  task_to_enc_group[{xtask}] = {support.task_to_enc_group[xtask]}")
    log(f"  task_to_dec_group[{xtask}] = {support.task_to_dec_group[xtask]}")
    log(f"  task_to_src_vocab_code[{xtask}] = {support.task_to_src_vocab_code[xtask]}")
    log(f"  task_to_tgt_vocab_code[{xtask}] = {support.task_to_tgt_vocab_code[xtask]}")
    log(f"  task_to_transforms[{xtask}] = {support.task_to_transforms[xtask]}")
    log(f"  task_to_errors[{xtask}] = {support.task_to_errors[xtask]}")

    if errors:
        log(f"❌ zeroshot {xtask}: " + " ; ".join(errors))
    else:
        log(f"✅ zeroshot {xtask}: unique source/target support found")

    return xtask

def main() -> int:
    (datadir, outdir, model, trainconfig, supervisedpairs, zeroshotpairs,
     mammoth, logdir, scrdir, cfg, lang_inventory) = get_env_vars()
    writers = CallWriters(outdir)
    try:
        support = collect_task_support(cfg, lang_inventory)
        (supervised_pair_set, zeroshot_pair_set) = read_pairs(supervisedpairs, zeroshotpairs)
        zeroshot_pair_set = filter_zeroshotpairs(supervised_pair_set, zeroshot_pair_set)
        support, rejected = filter_supervised_tasks(support, supervised_pair_set, lang_inventory,)
        check_data_coverage(support.task_set, lang_inventory.valid_xcodes)
        added_zeroshot = add_zeroshot_tasks( cfg, zeroshot_pair_set=zeroshot_pair_set, support=support, inventory=lang_inventory)
        produce_infyamls_and_calls(lang_inventory, support, datadir, outdir, trainconfig, cfg,
                                   writers, model, mammoth, logdir, scrdir,)
    finally:
        writers.close()
    validate_calls_out(outdir)
    log(f'✨ All stages of planning completed')
    return 0

if __name__ == "__main__":
    raise SystemExit(main())





