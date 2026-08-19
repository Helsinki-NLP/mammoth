#!/usr/bin/env python3
# inf_pairs.py
#
# Purpose
# -------
# Extract and rank evaluation language pairs from a Mammoth training
# configuration.
#
# This script is part of the eval2 planning workflow. It reads a Mammoth
# `train.yaml`, builds a directed graph of supervised source-target training
# pairs, and supports two main uses:
#
#   1) writing the supervised `src-tgt` pairs found in the training config,
#   2) ranking candidate zero-shot pairs that are not directly present in the
#      training graph.
#
# Candidate zero-shot pairs are ranked using a combination of:
#   - graph reachability and shortest-path structure,
#   - pivot availability,
#   - pivot hub strength from centrality measures,
#   - alternative shortest paths,
#   - linguistic compatibility scores derived from URIEL/lang2vec distances.
#
# Current behavior
# ----------------
# The script:
#   1) reads the top-level `tasks` mapping from a Mammoth `train.yaml`,
#   2) extracts directed source-target edges from task `src_tgt` fields,
#   3) optionally filters out denoising / autoencoder-style tasks,
#   4) builds a directed language graph from the supervised tasks,
#   5) computes language-centrality measures such as degree, closeness,
#      betweenness, PageRank, and eigenvector centrality,
#   6) retrieves URIEL/lang2vec distance matrices where available,
#   7) enumerates candidate zero-shot pairs that:
#        - are not already present as direct training edges,
#        - are reachable within a configurable path-length bound,
#   8) scores and ranks those candidates,
#   9) optionally writes either:
#        - all supervised pairs, or
#        - the top-ranked zero-shot pairs.
#
# Inputs
# ------
# Positional:
#   train_yaml
#       Path to a Mammoth training configuration file.
#
# Optional:
#   --top-n
#       Number of top-ranked zero-shot pairs to write when `--zs-out` is used.
#   --zs-out
#       Output file for writing the top-N zero-shot pairs as `src-tgt`.
#   --rank-list-only
#       Suppress most diagnostic output and print only the grouped zero-shot
#       rank list.
#   --supervised-pairs-and-quit
#       Output file for writing all supervised `src-tgt` pairs and exiting
#       without zero-shot ranking.
#
# Reads
# -----
#   - the input `train.yaml`
#   - optional URIEL/lang2vec resources from the configured local repository
#
# Writes
# ------
#   - optional output file given by `--zs-out`
#   - optional output file given by `--supervised-pairs-and-quit`
#
# Output
# ------
# The script may print:
#   - language inventory and connectivity summaries,
#   - centrality tables,
#   - warnings for failed centrality or linguistic-distance components,
#   - ranked zero-shot candidate pairs,
#   - grouped recommended zero-shot evaluation pairs.
#
# Notes
# -----
# - The script treats the supervised task graph as a directed graph.
# - Zero-shot candidates are restricted to pairs that are reachable through the
#   supervised graph but do not already exist as direct edges.
# - Denoising and autoencoder-style tasks can be filtered out when exporting
#   supervised pairs.
# - The script is intended as a planning and selection aid; it does not run
#   inference itself.

import sys
import argparse

import math
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
try:
    import networkx as nx
except ImportError:
    sys.stderr.write("Missing dependency: networkx\n")
    sys.stderr.write("Install with: pip install networkx\n")
    sys.exit(1)

# ============================================================
# Linguistic distance support and zero-shot ranking additions
# ============================================================

# References for manual checking:
# - URIEL/lang2vec paper:
#   https://aclanthology.org/E17-2002/
# - lang2vec distances API and supported distance types:
#   https://github.com/antonisa/lang2vec
# - ASJP open lexical database download page:
#   https://asjp.clld.org/download
# - WALS feature database:
#   https://wals.info/
#
# Notes:
# - URIEL/lang2vec is the practical turnkey choice here because it exposes
#   precomputed pairwise distances for ISO-639-3 codes.
# - ASJP and WALS are open and useful, but less turnkey for this exact script:
#   ASJP is a lexical database/software workflow, and WALS is a feature database.
# - If you later want lexical distance, add an ASJP-derived CSV matrix and wire
#   it in via load_optional_distance_matrix() below.

# Remember to do before calling this:
#   module load cray-python
#   . /scratch/project_462000964/shared/testing-shared/venvs/view/bin/activate
#
# There was a problem with standard lang2vec via pip.  This is how I installed it now:
#   mkdir -p /scratch/project_462000964/shared/testing-shared/pkgs
#   cd /scratch/project_462000964/shared/testing-shared/pkgs
#   git clone https://github.com/antonisa/lang2vec.git
#   cd /scratch/project_462000964/shared/testing-shared/pkgs/lang2vec
#   python3 -m pip uninstall -y lang2vec
#   python3 -m pip install .


# ============================================================
# Linguistic distance support and zero-shot ranking additions
# ============================================================

# References for manual checking:
# - URIEL/lang2vec paper:
#   https://aclanthology.org/E17-2002/
# - lang2vec distances API and supported distance types:
#   https://github.com/antonisa/lang2vec
# - ASJP open lexical database download page:
#   https://asjp.clld.org/download
# - WALS feature database:
#   https://wals.info/
#
# Notes:
# - URIEL/lang2vec is the practical turnkey choice here because it exposes
#   precomputed pairwise distances for ISO-639-3 codes.
# - ASJP and WALS are open and useful, but less turnkey for this exact script:
#   ASJP is a lexical database/software workflow, and WALS is a feature database.
# - If you later want lexical distance, add an ASJP-derived CSV matrix and wire
#   it in via load_optional_distance_matrix() below.

# Remember to do before calling this:
#   module load cray-python
#   . /scratch/project_462000964/shared/testing-shared/venvs/view/bin/activate
#
# There was a problem with standard lang2vec via pip.  This is how I installed it now:
#   mkdir -p /scratch/project_462000964/shared/testing-shared/pkgs
#   cd /scratch/project_462000964/shared/testing-shared/pkgs
#   git clone https://github.com/antonisa/lang2vec.git
#   cd /scratch/project_462000964/shared/testing-shared/pkgs/lang2vec
#   python3 -m pip uninstall -y lang2vec
#   python3 -m pip install .



def parse_args():
    parser = argparse.ArgumentParser(description="Inspect MAMMOTH train.yaml and rank zero-shot language pairs.")
    parser.add_argument("train_yaml",help="Path to train.yaml")
    parser.add_argument("--top-n",type=int,default=20,help="Number of top zero-shot pairs to write")
    parser.add_argument("--zs-out",type=str,default=None,help="Write top-N zero-shot pairs to file as src-tgt")
    parser.add_argument("--rank-list-only",action="store_true",help="Print only the grouped zero-shot rank list and suppress other console output")
    parser.add_argument("--supervised-pairs-and-quit",metavar="FILE",type=str,default=None,help="Write all supervised src-tgt pairs to FILE and exit")
    return parser.parse_args()


def is_denoising_autoencoder_task(task_name, task, src, tgt):
    # Most common easy signal
    if src and tgt and src == tgt:
        return True

    name_l = (task_name or "").lower()

    # Conservative name-based filters
    suspicious = [
        "dae",
        "denoise",
        "denoising",
        "autoenc",
        "auto-enc",
        "autoencoder",
    ]
    if any(x in name_l for x in suspicious):
        return True

    # Check a few common config fields if present
    for key in ["type", "task_type", "objective", "corpus", "src_tgt"]:
        val = task.get(key)
        if isinstance(val, str):
            val_l = val.lower()
            if any(x in val_l for x in suspicious):
                return True

    return False


def write_supervised_pairs_and_quit(cfg, outfile):
    if not isinstance(cfg, dict) or "tasks" not in cfg or not isinstance(cfg["tasks"], dict):
        raise ValueError("No top-level 'tasks' mapping found")

    pairs = set()

    for task_name in sorted(cfg["tasks"]):
        task = cfg["tasks"][task_name]
        if not isinstance(task, dict):
            continue

        # NOTE: This does not make any distinction between docmt, mt, sentmt or even
        # non-mt tasks.  The assumption is that theis languages are equally good as pivot languages
        # and tasks when considering centrality or zero-shot pairs.
        src_tgt = get_src_tgt(task)
        src, tgt = split_src_tgt(src_tgt)
        if not (src and tgt):
            continue

        if is_denoising_autoencoder_task(task_name, task, src, tgt):
            continue

        pairs.add(f"{src}-{tgt}")

    with open(outfile, "w", encoding="utf-8") as f:
        for pair in sorted(pairs):
            f.write(pair + "\n")
            
def safe_float(x, default=0.0):
    try:
        x = float(x)
    except Exception:
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def harmonic_mean(xs, eps=1e-12):
    xs = [x for x in xs if x > eps]
    if not xs:
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def compute_dense_ranks(items, key_names):
    """
    items: list[dict]
    key_names: list[str]
    Returns ranks per key, written back as '<key>_rank'
    Higher values are ranked better.
    """
    for key in key_names:
        ordered = sorted(
            enumerate(items),
            key=lambda kv: (-safe_float(kv[1].get(key, 0.0)), kv[1].get("pair", "")),
        )
        last_val = None
        current_rank = 0
        for idx, (orig_i, item) in enumerate(ordered, start=1):
            val = safe_float(item.get(key, 0.0))
            if last_val is None or val != last_val:
                current_rank = idx
                last_val = val
            item[f"{key}_rank"] = current_rank


def get_language_list_from_graph(graph):
    return sorted(graph.nodes())


def retrieve_uriel_distance_matrices(langs):
    """
    Returns a dict name -> nested dict matrix[a][b] = distance.
    Uses ISO-639-3 codes as expected by l2v.

    Distance types documented by l2v:
      genetic, geographic, phonological, syntactic, featural, inventory
    """
    L2V_REPO = "/scratch/project_462000964/shared/testing-shared/lang2vec"
    import os
    if os.path.isdir(L2V_REPO) and L2V_REPO not in sys.path:
        sys.path.insert(0, L2V_REPO)
    try:
        import lang2vec.lang2vec as l2v
        L2V_IMPORT_ERROR = None
    except Exception as e:
        l2v = None
        L2V_IMPORT_ERROR = repr(e)

    def has_lang2vec_distance_api():
        return (
            l2v is not None and
            hasattr(l2v, "distance") and
            callable(getattr(l2v, "distance"))
        )
    if not has_lang2vec_distance_api():
        return {}, {"api": "lang2vec.lang2vec has no callable distance() API"}

    wanted = ["genetic", "geographic", "phonological", "syntactic", "featural", "inventory"]
    norm_langs = [normalize_l2v_lang(x) for x in langs]
    matrices = {}
    errors = {}
    for dist_name in wanted:
        try:
            arr = l2v.distance(dist_name, norm_langs)
            matrix = {}
            for i, a in enumerate(langs):
                matrix[a] = {}
                for j, b in enumerate(langs):
                    matrix[a][b] = safe_float(arr[i][j], default=1.0 if a != b else 0.0)
            matrices[dist_name] = matrix
        except Exception as e:
            errors[dist_name] = repr(e)
    return matrices, errors

def load_optional_distance_matrix(csv_path):
    """
    Optional helper for later manual addition of ASJP/WALS-based matrices.
    Expected CSV format:
        lang,bos,bul,cat,...
        bos,0,0.32,0.41,...
        bul,0.32,0,0.27,...
        ...
    Returns nested dict or {} if file missing / unreadable.
    """
    import csv
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except Exception:
        return {}

    if not rows or len(rows) < 2:
        return {}

    header = rows[0][1:]
    matrix = {}
    for row in rows[1:]:
        if not row:
            continue
        lang = row[0]
        matrix[lang] = {}
        for j, val in enumerate(row[1:]):
            if j < len(header):
                matrix[lang][header[j]] = safe_float(val, default=0.0)
    return matrix


def dmat_get(dmat, a, b, default=1.0):
    if not dmat:
        return default if a != b else 0.0
    return safe_float(dmat.get(a, {}).get(b, default if a != b else 0.0), default=default)


def sim_from_dist(d, alpha=3.0):
    """
    Convert a distance in [0,1]-ish space to similarity.
    URIEL distances are commonly cosine-like distances; this keeps the score bounded.
    """
    d = max(0.0, d)
    return math.exp(-alpha * d)


def pivot_strength(lang, centrality):
    """
    Blended hub strength from your existing graph centrality.
    Centrality should be a modifier, not the whole score.
    """
    c = centrality.get(lang, {})
    pr = safe_float(c.get("pagerank", 0.0))
    eg = safe_float(c.get("eigenvector", 0.0))
    deg = safe_float(c.get("total_degree_norm", 0.0))
    return 0.50 * pr + 0.30 * eg + 0.20 * deg


def compute_pair_linguistic_compatibility(src, tgt, pivots, dmats):
    """
    Returns a dict with several compatibility scores.
    Higher is better.
    """
    if not pivots:
        return {
            "compat_genetic": 0.0,
            "compat_syntactic": 0.0,
            "compat_inventory": 0.0,
            "compat_phonological": 0.0,
            "compat_featural": 0.0,
            "compat_geo": 0.0,
            "compat_blend": 0.0,
        }

    result = {}

    for dist_name, out_name in [
        ("genetic", "compat_genetic"),
        ("syntactic", "compat_syntactic"),
        ("inventory", "compat_inventory"),
        ("phonological", "compat_phonological"),
        ("featural", "compat_featural"),
        ("geographic", "compat_geo"),
    ]:
        dmat = dmats.get(dist_name, {})
        sims = []
        for p in pivots:
            d_sp = dmat_get(dmat, src, p, default=1.0)
            d_pt = dmat_get(dmat, p, tgt, default=1.0)
            sims.append(sim_from_dist(d_sp) * sim_from_dist(d_pt))
        result[out_name] = mean(sims)

    # Blend emphasizing typology/genealogy over geography.
    result["compat_blend"] = (
        0.30 * result["compat_genetic"] +
        0.30 * result["compat_syntactic"] +
        0.20 * result["compat_inventory"] +
        0.10 * result["compat_phonological"] +
        0.10 * result["compat_featural"]
    )
    return result


def compute_pair_hub_and_path_scores(pivots, paths, centrality):
    """
    Returns:
      hub_strength, pivot_availability, alt_paths
    """
    strengths = [pivot_strength(p, centrality) for p in pivots]
    hub_strength = mean(strengths)

    n_pivots = len(pivots)
    n_paths = len(paths)

    # Availability: do we have pivots at all, and how many?
    pivot_availability = math.log1p(n_pivots)

    # Alternative paths: emphasize redundancy in shortest valid paths.
    alt_paths = math.log1p(n_paths)

    return hub_strength, pivot_availability, alt_paths


def aggregate_zero_shot_score(compat_blend, hub_strength, pivot_availability, alt_paths):
    """
    Final aggregate score.
    Linguistic compatibility gets the largest weight.
    """
    return (
        0.60 * compat_blend +
        0.15 * hub_strength +
        0.15 * pivot_availability +
        0.10 * alt_paths
    )

def normalize_l2v_lang(lang: str) -> str:
    mapping = {
        "srp_Cyrl": "srp",
    }
    return mapping.get(lang, lang)

def build_zero_shot_candidates(
    graph,
    centrality,
    max_path_len=3,
    uriel_dmats=None,
    optional_dmats=None,
):
    nodes = sorted(graph.nodes())
    candidates = []

    dmats = {}
    if uriel_dmats:
        dmats.update(uriel_dmats)
    if optional_dmats:
        dmats.update(optional_dmats)

    for src in nodes:
        for tgt in nodes:
            if src == tgt:
                continue
            if graph.has_edge(src, tgt):
                continue

            try:
                shortest_len = nx.shortest_path_length(graph, src, tgt)
            except nx.NetworkXNoPath:
                continue

            if shortest_len > max_path_len:
                continue

            all_paths = list(nx.all_simple_paths(graph, source=src, target=tgt, cutoff=shortest_len))
            paths = [p for p in all_paths if len(p) - 1 == shortest_len]
            if not paths:
                continue

            pivots = unique_preserve_order(
                [pivot for p in paths for pivot in pivots_from_path(p)]
            )

            compat = compute_pair_linguistic_compatibility(src, tgt, pivots, dmats)
            hub_strength, pivot_availability, alt_paths = compute_pair_hub_and_path_scores(
                pivots, paths, centrality
            )
            zs_score = aggregate_zero_shot_score(
                compat["compat_blend"],
                hub_strength,
                pivot_availability,
                alt_paths,
            )

            candidates.append({
                "pair": f"{src}{ARROW}{tgt}",
                "src": src,
                "tgt": tgt,
                "hops": shortest_len,
                "n_paths": len(paths),
                "n_pivots": len(pivots),
                "pivots": pivots,
                "pivot_availability": pivot_availability,
                "hub_strength": hub_strength,
                "alt_paths": alt_paths,
                "compat_genetic": compat["compat_genetic"],
                "compat_syntactic": compat["compat_syntactic"],
                "compat_inventory": compat["compat_inventory"],
                "compat_phonological": compat["compat_phonological"],
                "compat_featural": compat["compat_featural"],
                "compat_geo": compat["compat_geo"],
                "compat_blend": compat["compat_blend"],
                "zs_score": zs_score,
                "example_path": path_to_string(paths[0]),
            })

    compute_dense_ranks(candidates, [
        "pivot_availability",
        "hub_strength",
        "alt_paths",
        "compat_genetic",
        "compat_syntactic",
        "compat_inventory",
        "compat_phonological",
        "compat_featural",
        "compat_blend",
        "zs_score",
    ])

    candidates.sort(
        key=lambda x: (
            -x["zs_score"],
            -x["compat_blend"],
            -x["pivot_availability"],
            -x["hub_strength"],
            -x["alt_paths"],
            x["pair"],
        )
    )
    for i, item in enumerate(candidates, start=1):
        item["rank"] = i
    zero_shot = candidates
    return candidates


ARROW = "→"

def shorten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(shorten(x) for x in value)
    if isinstance(value, dict):
        return ",".join(f"{k}={shorten(value[k])}" for k in sorted(value))
    return str(value)


def fmt_float(x: float, digits: int = 4) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.{digits}f}"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def get_src_tgt(task: Dict[str, Any]) -> str:
    return shorten(task.get("src_tgt", ""))


def split_src_tgt(src_tgt: str) -> Tuple[str, str]:
    if not src_tgt:
        return "", ""
    if "-" not in src_tgt:
        return src_tgt.strip(), ""
    src, tgt = src_tgt.split("-", 1)
    return src.strip(), tgt.strip()


def get_dec_share(task: Dict[str, Any]) -> str:
    return shorten(task.get("dec_sharing_group", ""))


def get_enc_share(task: Dict[str, Any]) -> str:
    return shorten(task.get("enc_sharing_group", ""))


def get_filtering(task: Dict[str, Any]) -> str:
    transforms = task.get("transforms")

    if isinstance(transforms, list):
        filters = [x for x in transforms if isinstance(x, str) and x.lower().startswith("filter")]
        return ",".join(filters)
    if isinstance(transforms, str) and transforms.lower().startswith("filter"):
        return transforms

    parts = []
    for key in sorted(task):
        if "filter" in key.lower():
            parts.append(shorten(task[key]) if key == "filter" else f"{key}={shorten(task[key])}")
    return "; ".join(parts)


def get_weight(task: Dict[str, Any]) -> str:
    for key in sorted(task):
        if key.lower() == "weight":
            return shorten(task[key])
    for key in sorted(task):
        if "weight" in key.lower():
            return f"{key}={shorten(task[key])}"
    return ""


def make_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        widths = [len(h) for h in headers]
    else:
        widths = []
        for i, h in enumerate(headers):
            max_len = len(h)
            for row in rows:
                max_len = max(max_len, len(row[i]))
            widths.append(max_len)

    def hline() -> str:
        return "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    out = [hline(), fmt_row(headers), hline()]
    for row in rows:
        out.append(fmt_row(row))
    out.append(hline())
    return "\n".join(out)


def format_grouped_counts(title: str, mapping: Dict[str, Set[str]]) -> str:
    grouped: Dict[int, List[str]] = defaultdict(list)
    for lang in sorted(mapping):
        grouped[len(mapping[lang])].append(lang)

    lines = [title]
    for count in sorted(grouped):
        langs = ", ".join(grouped[count])
        lines.append(f"  {count}: {langs}")
    if len(lines) == 1:
        lines.append("  (none)")
    return "\n".join(lines)


def path_to_string(path: List[str]) -> str:
    return f" {ARROW} ".join(path)


def pivots_from_path(path: List[str]) -> List[str]:
    if len(path) <= 2:
        return []
    return path[1:-1]


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def compute_ranks(metric: Dict[str, float]) -> Dict[str, int]:
    items = sorted(metric.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks = {}
    rank = 0
    prev_val = None
    for idx, (lang, val) in enumerate(items, start=1):
        if prev_val is None or val != prev_val:
            rank = idx
            prev_val = val
        ranks[lang] = rank
    return ranks


def compute_centralities(graph: nx.DiGraph):
    nodes = sorted(graph.nodes())

    out_deg = dict(graph.out_degree())
    in_deg = dict(graph.in_degree())
    total_deg = {n: out_deg[n] + in_deg[n] for n in nodes}

    n = len(nodes)
    norm = max(1, n - 1)

    out_deg_norm = {k: safe_div(v, norm) for k, v in out_deg.items()}
    in_deg_norm = {k: safe_div(v, norm) for k, v in in_deg.items()}
    total_deg_norm = {k: safe_div(v, 2 * norm) for k, v in total_deg.items()}

    closeness = nx.closeness_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    harmonic = nx.harmonic_centrality(graph)

    pagerank_error = None
    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=1000, tol=1.0e-12)
    except Exception as e:
        pagerank_error = str(e)
        pagerank = {n: float("nan") for n in nodes}

    eig_error = None
    try:
        eig = nx.eigenvector_centrality(graph.to_undirected(), max_iter=1000, tol=1.0e-10)
    except Exception as e:
        eig_error = str(e)
        eig = {n: float("nan") for n in nodes}

    result = {}
    for n_ in nodes:
        result[n_] = {
            "out_degree": float(out_deg[n_]),
            "in_degree": float(in_deg[n_]),
            "total_degree": float(total_deg[n_]),
            "out_degree_norm": float(out_deg_norm[n_]),
            "in_degree_norm": float(in_deg_norm[n_]),
            "total_degree_norm": float(total_deg_norm[n_]),
            "closeness": float(closeness.get(n_, float("nan"))),
            "harmonic": float(harmonic.get(n_, float("nan"))),
            "betweenness": float(betweenness.get(n_, float("nan"))),
            "pagerank": float(pagerank.get(n_, float("nan"))),
            "eigenvector": float(eig.get(n_, float("nan"))),
        }

    return result, pagerank_error, eig_error

def main() -> int:
    args = parse_args()
    filename = args.train_yaml
    top_n = args.top_n
    zs_out = args.zs_out
    rank_list_only = args.rank_list_only
    supervised_pairs_out = args.supervised_pairs_and_quit
    
    if not rank_list_only:
        print("importing yaml...")
    try:
        import yaml
    except ImportError:
        sys.stderr.write("Missing dependency: pyyaml\n")
        sys.stderr.write("Install with: pip install pyyaml\n")
        sys.exit(1)
        
    try:
        with open(filename, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if supervised_pairs_out:
            try:
                write_supervised_pairs_and_quit(cfg, supervised_pairs_out)
            except ValueError as e:
                sys.stderr.write(str(e) + "\n")
                return 1
            return 0
    except FileNotFoundError:
        sys.stderr.write(f"File not found: {filename}\n")
        return 1
    except yaml.YAMLError as e:
        sys.stderr.write(f"YAML parse error in {filename}: {e}\n")
        return 1
    if not isinstance(cfg, dict) or "tasks" not in cfg or not isinstance(cfg["tasks"], dict):
        sys.stderr.write(f"No top-level 'tasks' mapping found in {filename}\n")
        return 1
    headers = ["task", "src_tgt", "dec_share", "enc_share", "filtering", "weight"]
    rows: List[List[str]] = []

    all_langs: Set[str] = set()
    source_to_targets: Dict[str, Set[str]] = defaultdict(set)
    target_to_sources: Dict[str, Set[str]] = defaultdict(set)

    graph = nx.DiGraph()

    for task_name in sorted(cfg["tasks"]):
        task = cfg["tasks"][task_name]
        if not isinstance(task, dict):
            rows.append([task_name, "", "", "", "", ""])
            continue

        src_tgt = get_src_tgt(task)
        src, tgt = split_src_tgt(src_tgt)

        if src:
            all_langs.add(src)
            graph.add_node(src)
        if tgt:
            all_langs.add(tgt)
            graph.add_node(tgt)
        if src and tgt:
            source_to_targets[src].add(tgt)
            target_to_sources[tgt].add(src)
            graph.add_edge(src, tgt)

        rows.append([
            task_name,
            src_tgt,
            get_dec_share(task),
            get_enc_share(task),
            get_filtering(task),
            get_weight(task),
        ])

#    print("tasks:")
#    print(make_table(headers, rows))
#    print()

    if not rank_list_only:
        print("Languages involved")
        if all_langs:
            print("  " + ", ".join(sorted(all_langs)))
        else:
            print("  (none)")
        print()
        print(format_grouped_counts("Languages by number of targets", source_to_targets))
        print()
        print(format_grouped_counts("Languages by number of sources", target_to_sources))
        print()

    centrality, pagerank_error, eig_error = compute_centralities(graph)

    if pagerank_error:
        print(f"WARNING: pagerank failed: {pagerank_error}")
        print()
    if eig_error:
        print(f"WARNING: eigenvector centrality failed: {eig_error}")
        print()

    cent_headers = [
        "lang",
        "out_deg",
        "in_deg",
        "close",
        "harm",
        "between",
        "pagerank",
        "eigen",
        "PR_rank", "close_rank", "between_rank", "eigen_rank",
    ]
    cent_rows = []
    pr_ranks = compute_ranks({lang: centrality[lang]["pagerank"] for lang in graph.nodes()})
    cl_ranks = compute_ranks({lang: centrality[lang]["closeness"] for lang in graph.nodes()})
    bw_ranks = compute_ranks({lang: centrality[lang]["betweenness"] for lang in graph.nodes()})
    eg_ranks = compute_ranks({lang: centrality[lang]["eigenvector"] for lang in graph.nodes()})

    for lang in sorted(graph.nodes()):
        c = centrality[lang]
        cent_rows.append([
            lang,
            str(int(c["out_degree"])),
            str(int(c["in_degree"])),
            fmt_float(c["closeness"], 3),
            fmt_float(c["harmonic"], 1),
            fmt_float(c["betweenness"], 2),
            fmt_float(c["pagerank"], 3),
            fmt_float(c["eigenvector"], 3),
            str(pr_ranks[lang]),
            str(cl_ranks[lang]),
            str(bw_ranks[lang]),
            str(eg_ranks[lang]),
        ])


    if not rank_list_only:
        print("Language centrality measures and ranks")
        print(make_table(cent_headers, cent_rows))
        print()
        print("building and sorting zero-shot candidates...")

    if not rank_list_only:
        print("importing networkx...")

    langs = get_language_list_from_graph(graph)

    if not rank_list_only:
        print("importing and loading lang2vec resources...")

    # Automatic URIEL/l2v matrices
    uriel_dmats, uriel_errors = retrieve_uriel_distance_matrices(langs)
    if uriel_errors:
        for k in sorted(uriel_errors):
            print(f"  {k}: {uriel_errors[k]}")
    print()

    # Optional later additions if you manually prepare them:
    # asjp_lexical = load_optional_distance_matrix("asjp_lexical.csv")
    # wals_typological = load_optional_distance_matrix("wals_typological.csv")
    # optional_dmats = {
    #     "asjp_lexical": asjp_lexical,
    #     "wals_typological": wals_typological,
    # }
    optional_dmats = {}

    zero_shot = build_zero_shot_candidates(
        graph,
        centrality,
        max_path_len=3,
        uriel_dmats=uriel_dmats,
        optional_dmats=optional_dmats,
    )
    
    # zero_shot = build_zero_shot_candidates(graph, centrality, max_path_len=3)

    zs_headers = [
        "rank",
        "pair",
        "zs_score",
        "compat_mix",
        "avail",
        "hub",
        "alt",
        "hops",
        "n_paths",
        "n_pivots",
        "pivots",
    ]
    zs_rows = []
    for item in zero_shot:
        zs_rows.append([
            str(item["zs_score_rank"]),
            item["pair"],
            fmt_float(item["zs_score"], 4),
            fmt_float(item["compat_blend"], 4),
            fmt_float(item["pivot_availability"], 4),
            fmt_float(item["hub_strength"], 4),
            fmt_float(item["alt_paths"], 4),
            str(item["hops"]),
            str(item["n_paths"]),
            str(item["n_pivots"]),
            ",".join(item["pivots"]),
        ])


    if not rank_list_only:
        print("Proposed zero-shot pairs")
        if zs_rows:
            print(make_table(zs_headers, zs_rows[:20]))
            if len(zs_rows) > 30:
                print("...")
                print(make_table(zs_headers, zs_rows[-10:]))
            else:
                print("  (none)")
            print()

    if zero_shot:
        print(f"Recommended zero-shot evaluation pairs from {filename}:")
        by_rank = {}
        for item in zero_shot:
            r = item["zs_score_rank"]
            by_rank.setdefault(r, []).append(item["pair"])
        for r in sorted(by_rank)[:8]:
            pairs = ", ".join(sorted(by_rank[r]))
            print(f"  rank {r}: {pairs}")
            
    # --- write top-N zero-shot pairs ---
    # usage:  python cfg_view4.py --top-n 50 --zs-out top_pairs.txt
    if zs_out:
        with open(zs_out, "w", encoding="utf-8") as f:
            for item in zero_shot[:top_n]:
                pair = item["pair"]
                if "→" in pair:
                    src, tgt = pair.split("→", 1)
                    f.write(f"{src}-{tgt}\n")
                else:
                    f.write(pair + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
