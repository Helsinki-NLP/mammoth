import time
from itertools import compress
#from sklearn.cluster import AgglomerativeClustering

from .utils import logger, coalesce, UserConfigError, load_yaml, register_command_io, register_command_template_extras

# ---------- distance matrix ---------------------

import csv

# PURPOSE: Load a language distance matrix from CSV, validate shape, return headers and numpy array.
def load_distmat_csv(fname):
     # Open CSV file containing a square distance matrix with a 'lang' first column.
     with open(fname, 'r') as istr:
         reader = csv.reader(istr)
         # Read the header row (first row).
         header = next(reader)
         # Read the remaining rows into memory.
         data = list(reader)
         
     # Ensure first header cell is 'lang' (language codes).
     assert header[0] == 'lang', 'first column header should be lang'
     
     # Extract row header (first column of each row).
     row_headers = [d[0] for d in data]
     
     # Column headers are everything except the first column name.
     column_headers = header[1:]
     
     # Validate that row and column headers match exactly (matrix is square and aligned).
     assert row_headers == column_headers, 'provided matrix is not valid'
     
     # Convert the numeric portion of each row to float and build a NumPy array.
     sim_data = np.array([list(map(float, d[1:])) for d in data])
     
     # Return a dict with the header order and the 2D matrix.
     return {
         'header': row_headers,
         'data': sim_data,
     }




# SUMMARY: add_cluster_languages_args
# PURPOSE: Add arguments for clustering languages from a distance matrix.
# PUT THIS IN: cli.py (argument builders)
def add_cluster_languages_args(parser):
    
     # Optional: path to the CSV distance matrix (loaded via helper).
     parser.add_argument('--distance_matrix', type=load_distmat_csv)
     
     # Optional: distance threshold for clustering.
     parser.add_argument('--cutoff_threshold', type=float)
     
     # Number of clusters (groups) to produce.
     parser.add_argument('--n_groups', type=int)


def cluster_languages(opts):
    start = time.time()
    
    cc_opts = opts.in_config[0]['config_config']
    cc = cc_opts
    
    # If user does not pass --distance_matrix, we try YAML's config_config.distance_matrix.
    # If both missing AND YAML does not already contain 'groups', we raise a friendly error.
    dm = getattr(opts, "distance_matrix", None)
    if dm is None:
        path = cc.get("distance_matrix")
        if not path and "groups" not in cc:
            raise UserConfigError("Missing distance matrix. "
                                  "Provide --distance_matrix FILE.csv on CLI "
                                  "OR set 'config_config.distance_matrix' in YAML, "
                                  "OR predefine 'config_config.groups'.")
        if path:
            dm = load_distmat_csv(path)
    cutoff   = coalesce(opts, cc, "cutoff_threshold", default=None, type_desc="float (optional)")
    if dm is None and "groups" in cc:
        logger.info("Using groups from YAML; clustering skipped.")
        return
     
    # Resolve key hyperparameters and inputs.
    n_groups = coalesce(opts, cc, "n_groups", required=("groups" not in cc), type_desc="integer")
    cutoff_threshold = opts.cutoff_threshold if opts.cutoff_threshold else cc_opts.get('cutoff_threshold', None)

    # Either use a matrix passed through CLI or load from path in config.
    if opts.distance_matrix:
        distance_matrix = opts.distance_matrix
    else:
        distance_matrix_path = cc_opts.get('distance_matrix', None)
        if not distance_matrix_path:
            if 'groups' in cc_opts:
                logger.info('Using groups specified in yaml, without clustering.')
                return
            else:
                raise Exception(
                    'No distance matrix given. '
                    'Either specify --distance_matrix or directly give "groups" in the yaml.')
        distance_matrix = load_distmat_csv(distance_matrix_path)

    # Validate that all languages in tasks exist in the matrix.
    sim_langs = set(distance_matrix['header'])
    corpus_langs = set()
    for cname, corpus in opts.in_config[0]['tasks'].items():
        assert all([(lng in sim_langs) for lng in corpus['src_tgt'].split('-')]), \
            f'corpus {cname}: one language (either {" or ".join(corpus["src_tgt"].split("-"))} ' \
            f'was not found in the distance matrix (supports {" ".join(sim_langs)})'
        corpus_langs = corpus_langs | set(corpus['src_tgt'].split('-'))

    # If the matrix contains extra langs, remove them before clustering.
    if sim_langs != corpus_langs:
        logger.warning(f"languages in the distance matrix are unused ({', ' .join(sim_langs - corpus_langs)})")
        # Omit unused languages before clustering. Otherwise they might consume entire clusters.        
        selector = [lang in corpus_langs for lang in distance_matrix['header']]
        dist = distance_matrix['data']
        dist = dist[selector][:, selector]
        header = list(compress(distance_matrix['header'], selector))
        distance_matrix = {'data': dist, 'header': header}

    # Run agglomerative clustering with precomputed distances.
    group_idx = AgglomerativeClustering(
        n_clusters=n_groups, metric='precomputed', linkage='average',
        distance_threshold=cutoff_threshold, ).fit_predict(distance_matrix['data']).tolist()
    
    # Build mapping from language code to group name like 'group3'.
    groups = {lang: f'group{idx}' for lang, idx in zip(distance_matrix['header'], group_idx)}
      
    # A potential solution would be to save everything in the config structure:
    #   - Configuration for the config-config (what is now specified as CLI params)
    #   - Intermediary values computed in the steps (such as the lang -> group mapping)
    #   - Final configuration values
    # When reaching the end of the config-config, any excessive keys are
    # dropped before saving the yaml (OpenNMT doesn't like extra keys).
    # Why does this work? Any step can be omitted, by instead adding any
    # intermediary values it would produce into the input config. E.g. the lang
    # -> group mapping could be specified as a mapping in the input yaml instead of a csv.

    # Store mapping in config_config for later steps.
    cc_opts['groups'] = groups
    
    duration = time.time() - start
    logger.info(f'step took {duration} s')

def register(subparsers):
    p = subparsers.add_parser(
        "cluster_languages",
        help="Cluster languages from a precomputed distance matrix, or use YAML-provided groups.",
        description=(
            "Build {language -> group} mapping. You can pass --distance_matrix FILE.csv "
            "or put config_config.distance_matrix in YAML. If YAML already has 'groups', "
            "clustering is skipped."
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")

    p.add_argument("--distance_matrix", type=load_distmat_csv, metavar="FILE.csv",
                   help="CSV whose first column header is 'lang' and whose body is a symmetric distance matrix.")
    p.add_argument("--cutoff_threshold", type=float, metavar="FLOAT",
                   help="Optional average-linkage distance threshold.")
    p.add_argument("--n_groups", type=int, metavar="INT",
                   help="Number of clusters (required unless YAML already has 'groups').")

    p.set_defaults(handler=cluster_languages)
    p.set_defaults(_mutates_yaml=True)

    register_command_io("cluster_languages", {
        "reads": ["config_config.distance_matrix", "config_config.n_groups", "config_config.groups"],
        "writes": ["config_config.groups"],
        "summary": "Builds language→group mapping, unless already provided.",
    })
    register_command_template_extras("cluster_languages", {
        "_notes": [
            "Provide n_groups to request clustering, or define 'groups' explicitly.",
            "If you also provide a CSV distance matrix, the command can use it."
        ],
        "config_config": {
            "n_groups": 2,
            "groups": {"en": 0, "fi": 1}   # if you want to override clustering
        }
    })
