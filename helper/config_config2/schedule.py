import time
from .utils import logger, coalesce, UserConfigError
from .utils import load_yaml

# SUMMARY: external_linecount
# PURPOSE: Count lines in a file quickly using system utilities; supports .gz via zcat.
# PUT THIS IN: io_helpers.py
def external_linecount(file_path):
    """ Use external program wc to determine line count.
 
    Faster than iterating over lines in python.
    Transparently supports gzip files based on .gz ending.
    """
    # If the file is gzipped, use zcat + wc -l to count lines.
    if file_path.endswith('.gz'):
        ext_lc = subprocess.check_output(
            ['zcat {} | wc -l'.format(file_path)], shell=True).split()[0]
    else:
        # Otherwise, use wc -l directly on the file.
        ext_lc = subprocess.check_output(['wc', '-l', file_path]).split()[0]
    # wc outputs bytes; decode to str then to int.
    ext_lc = int(ext_lc.decode('utf-8'))
    return ext_lc


# SUMMARY: read_cached_linecounts
# PURPOSE: Read a simple tab-separated cache "count<TAB>path" into a dict; return {} on failure.
# PUT THIS IN: io_helpers.py
def read_cached_linecounts(fname):
    try:
        # Prepare a dictionary to store path -> line_count.
        line_counts = dict()
        # Open the cache file for reading.
        with open(fname, 'r') as fin:
            for line in fin:
                # Remove trailing newline and spaces.
                line = line.strip()
                # Skip blank lines.
                if len(line) == 0:
                    continue
                # Each line is "count<TAB>path".
                count, path = line.split('\t')
                # Store as integer.
                line_counts[path] = int(count)
        # Return the filled mapping.
        return line_counts
    except Exception:
        # If anything goes wrong (file not found, parse error), return empty dict.
        return dict()


# PURPOSE: Compute corpus lengths, weights (temperature), and introduce_at_training_step; optionally split large ones.
def corpora_schedule(opts):
    if getattr(opts, "yaml_help", False):
        print_command_yaml_help("corpora_schedule")
        
    # Start timing for logging.
    start = time.time()
    
    # Short alias to the config_config section.
    cc_opts = opts.in_config[0]['config_config']
    cc = cc_opts
    
    # Resolve temperature setting (CLI overrides YAML default 1.0).
    temperature = coalesce(opts, cc, "temperature", default=1.0, type_desc="float")
    
    # Whether to compute and store weights per corpus.
    use_weight  = coalesce(opts, cc, "use_weight", default=False, type_desc="boolean")

    # Extra multiplier for autoencoder tasks.
    ae_weight   = coalesce(opts, cc, "ae_weight",   default=1.0, type_desc="float")
    
    # Whether to compute curriculum start steps for corpora.
    use_iats    = coalesce(opts, cc, "use_introduce_at_training_step", default=False, type_desc="boolean")

    split_thr   = cc.get("split_large_language_pairs", 0.0)
    
    # Load or create a cache of line counts to avoid repeated wc calls.
    corpora_lens_cache_file = './corpora_length_cache'
    corpora_lens_cache = read_cached_linecounts(corpora_lens_cache_file)
    logger.info('cached corpora_lens:')
    for path, len in corpora_lens_cache.items():
        logger.info(f'CACHED:\t{path}:\t{len}')
        
    # Compute lengths for each corpus; consult cache first.
    corpora_lens = {}
    for cname, corpus in sorted(opts.in_config[0]['tasks'].items(), key=lambda x: x[0]):
        if corpus['path_src'] in corpora_lens_cache:
            length = corpora_lens_cache[corpus['path_src']]
            corpora_lens[cname] = length
        else:
            length = external_linecount(corpus['path_src'])
            corpora_lens[cname] = length
            # Append to cache for later runs.
            with open(corpora_lens_cache_file, 'a') as cache_out:
                print(f'{length}\t{corpus["path_src"]}', file=cache_out)
                logger.info(f'NEW:\t{corpus["path_src"]}\t{length}')
    # Report final computed lengths.
    logger.info('final corpora_lens:')
    for cname, len in corpora_lens.items():
        logger.info(f'{cname}:\t{len}')

    # Convert lengths into normalized weights ^ temperature.
    tot_lines = sum(corpora_lens.values())
    corpora_weights = {
        cname: (clen / tot_lines) ** temperature
        for cname, clen in corpora_lens.items()
    }
    # Optionally split very large corpora into shards.
    split_treshold = cc_opts.get('split_large_language_pairs', 0.0)
    if split_treshold:
        corpora_weights = _split_large_language_pairs(opts, corpora_weights, split_treshold)
        
    # Initialize min start step to total training steps.
    min_introduce_at_training_step = opts.in_config[0].get('train_steps', 100_000)

    # For each corpus, compute weight and possibly introduce_at_training_step.
    for cname, corpus in opts.in_config[0]['tasks'].items():
        src_lang, tgt_lang = corpus['src_tgt'].split('-')
        weight = corpora_weights[cname]
        # Curriculum uses sqrt(weight) if both weighting and curriculum are on.
        if use_weight and use_introduce_at_training_step:
            weight = float(sqrt(weight))
        if use_weight:
            multiplier = ae_weight if src_lang == tgt_lang else 1.0
            corpus['weight'] = weight * multiplier
        else:
            # Keep weight at 1 when not using weighting.
            corpus['weight'] = 1
        if use_introduce_at_training_step:
            total_steps = opts.in_config[0].get('train_steps', 100_000)
            if weight > 0.75:
                introduce_at_training_step = 0
            else:
                introduce_at_training_step = round(total_steps * (1 - weight))
            corpus['introduce_at_training_step'] = introduce_at_training_step
            min_introduce_at_training_step = min(min_introduce_at_training_step, introduce_at_training_step)
        else:
            corpus['introduce_at_training_step'] = 0
    # Shift all non-zero starts so at least one task starts at step 0.
    if use_introduce_at_training_step and min_introduce_at_training_step > 0:
        for cname, corpus in opts.in_config[0]['tasks'].items():
            if 'introduce_at_training_step' in corpus:
                corpus['introduce_at_training_step'] -= min_introduce_at_training_step
    # Log duration.
    duration = time.time() - start
    logger.info(f'step took {duration} s')

# PURPOSE: Split very large weighted corpora into multiple stride/offset copies; update tasks and weights.d
def _split_large_language_pairs(opts, corpora_weights, split_treshold):
    
    from numpy import ceil, sqrt
    from copy import deepcopy

    # Copy the tasks dict so we can modify structure safely.
    corpora_out = deepcopy(opts.in_config[0]['tasks'])
    
    # New mapping for possibly split corpus weights.
    corpora_weights_out = dict()
    
    # Iterate over each corpus weight.
    for cname, weight in corpora_weights.items():
        if weight > split_treshold:
        
            # Compute number of copies needed so each piece <= threshold.
            n_copies = int(ceil(weight / split_treshold))
            copy_weight = weight / n_copies
            logger.info(f'Splitting {cname} into {n_copies} copies')
             
            # Create N copies with stride/offset for data sharding.
            for i in range(n_copies):
                cname_copy = f'{cname}_split{i}'
                dict_copy = deepcopy(corpora_out[cname])
                corpora_out[cname_copy] = dict_copy
                corpora_out[cname_copy]['stride'] = n_copies
                corpora_out[cname_copy]['offset'] = i
                corpora_weights_out[cname_copy] = copy_weight
                 
            # Remove the original large corpus entry.
            del corpora_out[cname]
        else:
            # Keep weight unchanged if below threshold.
            corpora_weights_out[cname] = weight
             
    # Replace tasks in config with the split-aware version.
    opts.in_config[0]['tasks'] = corpora_out
     
    # Return updated weights mapping.
    return corpora_weights_out

from .schema import print_schema, COMMAND_IO

def _yaml_help_for_command(cmd):
    keys = ( COMMAND_IO.get(cmd, {}).get("reads", []) +
             COMMAND_IO.get(cmd, {}).get("writes", []) )
    print_schema(sorted(set(keys)))

def register(subparsers):
    p = subparsers.add_parser(
        "corpora_schedule",
        help="Compute corpus weights and curriculum introduction steps.",
        description=(
            "Compute weights = (size/total)^temperature and optional curriculum "
            "(introduce_at_training_step). Large corpora may be split into stride/offset shards."
        ),
    )
    # IO shared options:
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml",
                   help="Input YAML. Parsed into (data_dict, filename).")
    p.add_argument("--out_config", metavar="FILE.yaml",
                   help="Where to write output YAML. Default: print to stdout.")

    # Command-specific options (CLI can override YAML config_config.*)
    p.add_argument("--use_weight", action="store_true",
                   help="Use temperature-adjusted corpus weights (overrides YAML).")
    p.add_argument("--ae_weight", type=float, metavar="FLOAT", default=None,
                   help="Multiplier for autoencoder weights (overrides YAML).")
    p.add_argument("--use_introduce_at_training_step", action="store_true",
                   help="Compute curriculum start steps (overrides YAML).")
    p.add_argument("--temperature", type=float, metavar="FLOAT", default=None,
                   help="Temperature (1/T): 1.0 empirical, 0.0 uniform (overrides YAML).")

    p.add_argument("--yaml-help", action="store_true",
                   help="Show YAML keys this command reads/writes and exit.")
    p.set_defaults(handler=corpora_schedule)

