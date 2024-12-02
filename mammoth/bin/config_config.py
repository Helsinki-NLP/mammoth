# For documentation, see: OpenNMT-py-v2/docs/config_config.md
# For an example config, see : OpenNMT-py-v2/examples/config_config.yaml
import argparse
import csv
import logging
import numpy as np
import os
import subprocess
import time
import yaml
from collections import defaultdict
from copy import deepcopy
from itertools import compress
from sklearn.cluster import AgglomerativeClustering

from mammoth.utils.gpu_assignment import optimize_gpu_assignment

logger = logging.getLogger('config_config')


def init_logging():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def load_yaml(fname):
    with open(fname, 'r') as istr:
        config = yaml.safe_load(istr)
    return config, fname


def load_distmat_csv(fname):
    with open(fname, 'r') as istr:
        reader = csv.reader(istr)
        header = next(reader)
        data = list(reader)
    assert header[0] == 'lang', 'first column header should be lang'
    row_headers = [d[0] for d in data]
    column_headers = header[1:]
    assert row_headers == column_headers, 'provided matrix is not valid'
    sim_data = np.array([list(map(float, d[1:])) for d in data])
    return {
        'header': row_headers,
        'data': sim_data,
    }


def save_yaml(opts):
    serialized = yaml.safe_dump(opts.in_config[0], default_flow_style=False, allow_unicode=True)
    if opts.out_config:
        with open(opts.out_config, 'w') as ostr:
            print(serialized, file=ostr)
    else:
        print(serialized)


def external_linecount(file_path):
    """ Use external program wc to determine line count.

    Faster than iterating over lines in python.
    Transparently supports gzip files based on .gz ending.
    """
    if file_path.endswith('.gz'):
        ext_lc = subprocess.check_output(
            ['zcat {} | wc -l'.format(file_path)], shell=True).split()[0]
    else:
        ext_lc = subprocess.check_output(['wc', '-l', file_path]).split()[0]
    ext_lc = int(ext_lc.decode('utf-8'))
    return ext_lc


def read_cached_linecounts(fname):
    try:
        line_counts = dict()
        with open(fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                count, path = line.split('\t')
                line_counts[path] = int(count)
        return line_counts
    except Exception:
        return dict()


def add_complete_language_pairs_args(parser):
    parser.add_argument(
        '--src_path', type=str,
        help='path template to source data. Can use variables {src_lang}, {tgt_lang}, and {sorted_pair}.'
    )
    parser.add_argument(
        '--tgt_path', type=str,
        help='path template to target data. Can use variables {src_lang}, {tgt_lang}, and {sorted_pair}.'
    )
    parser.add_argument(
        '--ae_path', type=str,
        help='path template to monolingual data for autoencoder. '
             'Can use the variables {src_lang}, {tgt_lang}, and {sorted_pair}.'
             'If unset, autoencoder pairs will use src_path and tgt_path.'
    )
    parser.add_argument(
        '--valid_src_path', type=str,
        help='path template to source dev set. Can use variables {src_lang}, {tgt_lang}, and {sorted_pair}.'
    )
    parser.add_argument(
        '--valid_tgt_path', type=str,
        help='path template to target dev set. Can use variables {src_lang}, {tgt_lang}, and {sorted_pair}.'
    )
    parser.add_argument(
        '--autoencoder',
        action='store_true',
        help='add autoencoder tasks, for which src_lang == tgt_lang'
    )
    parser.add_argument(
        '--autoencoder_validation',
        action='store_true',
        help='include autoencoder tasks in validation'
    )


def add_configs_args(parser):
    parser.add_argument('--in_config', required=True, type=load_yaml)
    parser.add_argument('--out_config', type=str)


def add_corpora_schedule_args(parser):
    parser.add_argument(
        '--use_weight', action='store_true',
        help='Use corpus weights based on temperature-adjusted corpus size'
    )
    parser.add_argument(
        '--ae_weight', type=float, default=1.0,
        help='Multiplier for the weight (but not starting step) of autoencoder tasks'
    )
    parser.add_argument(
        '--use_introduce_at_training_step', action='store_true',
        help='Use a curriculum introducing corpora based on temperature-adjusted corpus size'
    )
    parser.add_argument(
        '--temperature', type=float,
        help='Temperature (1/T): 1.0 for empirical, 0.0 for uniform'
    )


def add_cluster_languages_args(parser):
    parser.add_argument('--distance_matrix', type=load_distmat_csv)
    parser.add_argument('--cutoff_threshold', type=float)
    parser.add_argument('--n_groups', type=int)


def add_sharing_groups_args(parser):
    parser.add_argument(
        '--enc_sharing_groups', type=str, action='append',
        help='list of {LANGUAGE | GROUP | FULL}. '
        'The first two may be prefixed, e.g. SRC_LANGUAGE or TGT_GROUP. '
        'Default prefix for encoder is SRC.'
    )
    parser.add_argument(
        '--dec_sharing_groups', type=str, action='append',
        help='list of {LANGUAGE | GROUP | FULL}.'
        'The first two may be prefixed, e.g. SRC_LANGUAGE or TGT_GROUP. '
        'Default prefix for decoder is TGT.'
    )


def add_allocate_device_args(parser):
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--n_gpus_per_node', type=int)
    parser.add_argument('--n_slots_per_gpu', type=int)
    parser.add_argument('--log_name', type=str)
    parser.add_argument(
        '--time_budget_s', type=int,
        help='time budget for GPU assignment, in seconds',
    )


def add_set_transforms_args(parser):
    parser.add_argument(
        '--transforms', type=str, action='append',
        help='transforms to use for translation tasks'
    )
    parser.add_argument(
        '--ae_transforms', type=str, action='append',
        help='transforms to use for autoencoder tasks'
    )


def add_adapter_config_args(parser):
    pass


def add_translation_configs_args(parser):
    parser.add_argument('--zero_shot', action='store_true')


def add_extra_fully_shared_hack_args(parser):
    parser.add_argument('--joint_vocab', type=str, required=True, help='Path to joint vocab')


def add_extra_copy_gpu_assignment_args(parser):
    parser.add_argument('--copy_from', required=True, type=load_yaml, help='Config containing the desired assignments')


def get_opts():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser_corpora_schedule = subparsers.add_parser('corpora_schedule')
    add_configs_args(parser_corpora_schedule)
    add_corpora_schedule_args(parser_corpora_schedule)
    parser_cluster_languages = subparsers.add_parser('cluster_languages')
    add_configs_args(parser_cluster_languages)
    add_cluster_languages_args(parser_cluster_languages)
    parser_sharing_groups = subparsers.add_parser('sharing_groups')
    add_configs_args(parser_sharing_groups)
    add_sharing_groups_args(parser_sharing_groups)
    parser_allocate_devices = subparsers.add_parser('allocate_devices')
    add_configs_args(parser_allocate_devices)
    add_allocate_device_args(parser_allocate_devices)
    parser_set_transforms = subparsers.add_parser('set_transforms')
    add_configs_args(parser_set_transforms)
    add_set_transforms_args(parser_set_transforms)
    parser_adapter_config = subparsers.add_parser('adapter_config')
    add_configs_args(parser_adapter_config)
    add_adapter_config_args(parser_adapter_config)
    parser_complete_language_pairs = subparsers.add_parser('complete_language_pairs')
    add_configs_args(parser_complete_language_pairs)
    add_complete_language_pairs_args(parser_complete_language_pairs)
    parser_translation_configs = subparsers.add_parser('translation_configs')
    add_configs_args(parser_translation_configs)
    add_translation_configs_args(parser_translation_configs)
    parser_remove_temporary_keys = subparsers.add_parser('remove_temporary_keys')
    add_configs_args(parser_remove_temporary_keys)
    parser_config_all = subparsers.add_parser('config_all')
    add_configs_args(parser_config_all)
    add_corpora_schedule_args(parser_config_all)
    add_cluster_languages_args(parser_config_all)
    add_sharing_groups_args(parser_config_all)
    add_allocate_device_args(parser_config_all)
    add_set_transforms_args(parser_config_all)
    add_complete_language_pairs_args(parser_config_all)
    add_adapter_config_args(parser_config_all)
    add_translation_configs_args(parser_config_all)
    parser_extra_cpu = subparsers.add_parser('extra_cpu')
    add_configs_args(parser_extra_cpu)
    parser_extra_fully_shared_hack = subparsers.add_parser('extra_fully_shared_hack')
    add_extra_fully_shared_hack_args(parser_extra_fully_shared_hack)
    add_configs_args(parser_extra_fully_shared_hack)
    parser_extra_copy_gpu_assignment = subparsers.add_parser('extra_copy_gpu_assignment')
    add_extra_copy_gpu_assignment_args(parser_extra_copy_gpu_assignment)
    add_configs_args(parser_extra_copy_gpu_assignment)
    return parser.parse_args()


def _split_large_language_pairs(opts, corpora_weights, split_treshold):
    corpora_out = deepcopy(opts.in_config[0]['tasks'])
    corpora_weights_out = dict()
    for cname, weight in corpora_weights.items():
        if weight > split_treshold:
            n_copies = int(np.ceil(weight / split_treshold))
            copy_weight = weight / n_copies
            logger.info(f'Splitting {cname} into {n_copies} copies')
            for i in range(n_copies):
                cname_copy = f'{cname}_split{i}'
                dict_copy = deepcopy(corpora_out[cname])
                corpora_out[cname_copy] = dict_copy
                corpora_out[cname_copy]['stride'] = n_copies
                corpora_out[cname_copy]['offset'] = i
                corpora_weights_out[cname_copy] = copy_weight
            del corpora_out[cname]
        else:
            corpora_weights_out[cname] = weight
    opts.in_config[0]['tasks'] = corpora_out
    return corpora_weights_out


def corpora_schedule(opts):
    start = time.time()
    cc_opts = opts.in_config[0]['config_config']
    temperature = opts.temperature if opts.temperature else cc_opts.get('temperature', 1.0)
    use_weight = opts.use_weight if opts.use_weight else cc_opts.get('use_weight', False)
    ae_weight = opts.ae_weight if opts.ae_weight else cc_opts.get('ae_weight', 1.0)
    use_introduce_at_training_step = (
        opts.use_introduce_at_training_step if opts.use_introduce_at_training_step
        else cc_opts.get('use_introduce_at_training_step', False)
    )

    corpora_lens_cache_file = './corpora_length_cache'
    corpora_lens_cache = read_cached_linecounts(corpora_lens_cache_file)
    logger.info('cached corpora_lens:')
    for path, len in corpora_lens_cache.items():
        logger.info(f'CACHED:\t{path}:\t{len}')
    corpora_lens = {}
    for cname, corpus in sorted(opts.in_config[0]['tasks'].items(), key=lambda x: x[0]):
        if corpus['path_src'] in corpora_lens_cache:
            length = corpora_lens_cache[corpus['path_src']]
            corpora_lens[cname] = length
        else:
            length = external_linecount(corpus['path_src'])
            corpora_lens[cname] = length
            with open(corpora_lens_cache_file, 'a') as cache_out:
                print(f'{length}\t{corpus["path_src"]}', file=cache_out)
                logger.info(f'NEW:\t{corpus["path_src"]}\t{length}')
    logger.info('final corpora_lens:')
    for cname, len in corpora_lens.items():
        logger.info(f'{cname}:\t{len}')

    tot_lines = sum(corpora_lens.values())
    corpora_weights = {
        cname: (clen / tot_lines) ** temperature
        for cname, clen in corpora_lens.items()
    }
    split_treshold = cc_opts.get('split_large_language_pairs', 0.0)
    if split_treshold:
        corpora_weights = _split_large_language_pairs(opts, corpora_weights, split_treshold)

    min_introduce_at_training_step = opts.in_config[0].get('train_steps', 100_000)
    for cname, corpus in opts.in_config[0]['tasks'].items():
        src_lang, tgt_lang = corpus['src_tgt'].split('-')
        weight = corpora_weights[cname]
        if use_weight and use_introduce_at_training_step:
            weight = float(np.sqrt(weight))
        if use_weight:
            multiplier = ae_weight if src_lang == tgt_lang else 1.0
            corpus['weight'] = weight * multiplier
        else:
            # Log spam if weight is unset
            corpus['weight'] = 1
        if use_introduce_at_training_step:
            # TODO: ensure this default always matches with opts.py
            total_steps = opts.in_config[0].get('train_steps', 100_000)
            if weight > 0.75:
                # High-resource language pairs (would train for over 75% of the training time)
                # all start at 0. This avoids starting training with only one GPU doing work,
                # while the other GPUs are idle waiting for their LPs to start.
                introduce_at_training_step = 0
            else:
                introduce_at_training_step = round(total_steps * (1 - weight))
            corpus['introduce_at_training_step'] = introduce_at_training_step
            min_introduce_at_training_step = min(min_introduce_at_training_step, introduce_at_training_step)
        else:
            # Log spam if introduce_at_training_step is unset
            corpus['introduce_at_training_step'] = 0
    if use_introduce_at_training_step and min_introduce_at_training_step > 0:
        # With a single very large task that gets split, it is possible that no task can start
        for cname, corpus in opts.in_config[0]['tasks'].items():
            if 'introduce_at_training_step' in corpus:
                corpus['introduce_at_training_step'] -= min_introduce_at_training_step
    duration = time.time() - start
    logger.info(f'step took {duration} s')


def cluster_languages(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    n_groups = opts.n_groups if opts.n_groups else cc_opts['n_groups']
    cutoff_threshold = opts.cutoff_threshold if opts.cutoff_threshold else cc_opts.get('cutoff_threshold', None)
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
                    'Either specify --distance_matrix or directly give "groups" in the yaml.'
                )
        distance_matrix = load_distmat_csv(distance_matrix_path)

    sim_langs = set(distance_matrix['header'])
    corpus_langs = set()
    for cname, corpus in opts.in_config[0]['tasks'].items():
        assert all([(lng in sim_langs) for lng in corpus['src_tgt'].split('-')]), \
            f'corpus {cname}: one language (either {" or ".join(corpus["src_tgt"].split("-"))} ' \
            f'was not found in the distance matrix (supports {" ".join(sim_langs)})'
        corpus_langs = corpus_langs | set(corpus['src_tgt'].split('-'))
    if sim_langs != corpus_langs:
        logger.warning(
            f"languages in the distance matrix are unused ({', ' .join(sim_langs - corpus_langs)})"
        )
        # Omit unused languages before clustering. Otherwise they might consume entire clusters.
        selector = [lang in corpus_langs for lang in distance_matrix['header']]
        dist = distance_matrix['data']
        dist = dist[selector][:, selector]
        header = list(compress(distance_matrix['header'], selector))
        distance_matrix = {
            'data': dist,
            'header': header,
        }

    group_idx = AgglomerativeClustering(
        n_clusters=n_groups,
        metric='precomputed',
        linkage='average',
        distance_threshold=cutoff_threshold,
    ).fit_predict(distance_matrix['data']).tolist()
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
    cc_opts['groups'] = groups

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def sharing_groups(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    groups = cc_opts['groups']
    enc_sharing_groups = opts.enc_sharing_groups if opts.enc_sharing_groups else cc_opts['enc_sharing_groups']
    dec_sharing_groups = opts.dec_sharing_groups if opts.dec_sharing_groups else cc_opts['dec_sharing_groups']
    if not enc_sharing_groups:
        raise Exception('Must set --enc_sharing_groups')
    if not dec_sharing_groups:
        raise Exception('Must set --dec_sharing_groups')
    assert len(enc_sharing_groups) == len(opts.in_config[0]['enc_layers'])
    assert len(dec_sharing_groups) == len(opts.in_config[0]['dec_layers'])
    for cname, corpus in opts.in_config[0]['tasks'].items():
        src, tgt = corpus['src_tgt'].split('-')
        mapping_src = {
            'LANGUAGE': src,
            'GROUP': groups[src],
            'FULL': 'full',
            'SRC_LANGUAGE': src,
            'SRC_GROUP': groups[src],
            'TGT_LANGUAGE': tgt,
            'TGT_GROUP': groups[tgt],
        }
        mapping_tgt = {
            'LANGUAGE': tgt,
            'GROUP': groups[tgt],
            'FULL': 'full',
            'SRC_LANGUAGE': src,
            'SRC_GROUP': groups[src],
            'TGT_LANGUAGE': tgt,
            'TGT_GROUP': groups[tgt],
        }
        corpus['enc_sharing_group'] = [
            mapping_src[sharing_group] for sharing_group in enc_sharing_groups
        ]
        corpus['dec_sharing_group'] = [
            mapping_tgt[sharing_group] for sharing_group in dec_sharing_groups
        ]

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def set_transforms(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    ae_transforms = opts.ae_transforms if opts.ae_transforms else cc_opts.get('ae_transforms', [])
    transforms = opts.transforms if opts.transforms else cc_opts.get('transforms', [])

    for cname, corpus in opts.in_config[0]['tasks'].items():
        src, tgt = corpus['src_tgt'].split('-')
        if src == tgt:
            corpus['transforms'] = list(ae_transforms)
        else:
            corpus['transforms'] = list(transforms)

        if 'prefix' in corpus['transforms']:
            if cc_opts.get('use_src_lang_token', False):
                prefix = f'<from_{src}> <to_{tgt}>'
            else:
                prefix = f'<to_{tgt}>'
            corpus['src_prefix'] = prefix
            corpus['tgt_prefix'] = ''   # does not work, but must be set nonetheless
        else:
            if cc_opts.get('use_src_lang_token', False):
                raise Exception('use_src_lang_token requires prefix transform')

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def allocate_devices(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    n_nodes = opts.n_nodes if opts.n_nodes else cc_opts.get('n_nodes', None)
    n_gpus_per_node = opts.n_gpus_per_node if opts.n_gpus_per_node else cc_opts['n_gpus_per_node']
    n_slots_per_gpu = opts.n_slots_per_gpu if opts.n_slots_per_gpu else cc_opts.get('n_slots_per_gpu', None)

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

    # If there are fewer ready tasks than GPUs: adjust the curriculum
    if len(lps_ready_to_start) < (n_nodes * n_gpus_per_node):
        iats = [
            corpus.get('introduce_at_training_step', 0)
            for _, corpus in opts.in_config[0]['tasks'].items()
        ]
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

    if n_gpus_tot < 2:
        print('Assigning all tasks to 0:0')
        for key in opts.in_config[0]['tasks']:
            opts.in_config[0]['tasks'][key]['node_gpu'] = '0:0'
    else:
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

        for gpu_slot, lp in assignment.items():
            if lp is None:
                continue
            key = lp_to_key[lp].pop()
            opts.in_config[0]['tasks'][key]['node_gpu'] = f'{gpu_slot.node}:{gpu_slot.gpu}'
        total_remaining = 0
        for lp, keys in lp_to_key.items():
            if len(keys) > 0:
                print(f'{lp} remaining keys: {keys}')
            total_remaining += len(keys)
        assert total_remaining == 0

    for cname, corpus in opts.in_config[0]['tasks'].items():
        assert 'node_gpu' in corpus, f'{cname} not assigned to node_gpu: {corpus}'

    opts.in_config[0]['n_nodes'] = n_nodes
    opts.in_config[0]['world_size'] = n_gpus_tot
    opts.in_config[0]['gpu_ranks'] = list(range(n_gpus_per_node))

    # Ensure that all devices can start training (will crash otherwise)
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


def adapter_config(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    if 'adapters' not in opts.in_config[0]:
        logger.warning('No adapter configuration, skipping this step')
        return
    src_langs, tgt_langs = _get_langs(opts)
    src_groups = list(sorted(set(cc_opts['groups'][src] for src in src_langs)))
    tgt_groups = list(sorted(set(cc_opts['groups'][tgt] for tgt in tgt_langs)))
    encoder_adapters = opts.in_config[0]['adapters'].get('encoder', [])
    decoder_adapters = opts.in_config[0]['adapters'].get('decoder', [])
    for task_key, task_config in opts.in_config[0]['tasks'].items():
        if 'adapters' not in task_config:
            task_config['adapters'] = {'encoder': [], 'decoder': []}
    # TODO: refactor and add support for {SRC|TGT}_{LANGUAGE|GROUP} also to adapters
    if len(encoder_adapters) > 0:
        for adapter_name, adapter_config in sorted(encoder_adapters.items()):
            if adapter_config['ids'] == 'LANGUAGE':
                adapter_config['ids'] = list(src_langs)
                for task_key, task_config in opts.in_config[0]['tasks'].items():
                    task_src, task_tgt = task_config['src_tgt'].split('-')
                    task_config['adapters']['encoder'].append([adapter_name, task_src])
            elif adapter_config['ids'] == 'GROUP':
                adapter_config['ids'] = list(src_groups)
                for task_key, task_config in opts.in_config[0]['tasks'].items():
                    task_src, task_tgt = task_config['src_tgt'].split('-')
                    task_config['adapters']['encoder'].append([adapter_name, cc_opts['groups'][task_src]])
            elif adapter_config['ids'] == 'FULL':
                adapter_config['ids'] = ['full']
                for task_key, task_config in opts.in_config[0]['tasks'].items():
                    task_config['adapters']['encoder'].append([adapter_name, 'full'])
    if len(decoder_adapters) > 0:
        for adapter_name, adapter_config in sorted(decoder_adapters.items()):
            if adapter_config['ids'] == 'LANGUAGE':
                adapter_config['ids'] = list(tgt_langs)
                for task_key, task_config in opts.in_config[0]['tasks'].items():
                    task_src, task_tgt = task_config['src_tgt'].split('-')
                    task_config['adapters']['decoder'].append([adapter_name, task_tgt])
            elif adapter_config['ids'] == 'GROUP':
                adapter_config['ids'] = list(tgt_groups)
                for task_key, task_config in opts.in_config[0]['tasks'].items():
                    task_src, task_tgt = task_config['src_tgt'].split('-')
                    task_config['adapters']['decoder'].append([adapter_name, cc_opts['groups'][task_tgt]])
            elif adapter_config['ids'] == 'FULL':
                adapter_config['ids'] = ['full']
                for task_key, task_config in opts.in_config[0]['tasks'].items():
                    task_config['adapters']['decoder'].append([adapter_name, 'full'])
    opts.in_config[0]['adapters']['encoder'] = encoder_adapters
    opts.in_config[0]['adapters']['decoder'] = decoder_adapters

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def _adapters_to_stacks(task_adapters, opts, side):
    adapter_specs = opts.in_config[0]['adapters']
    adapters = [list() for _ in range(len(opts.in_config[0][f'{side}_layers']))]
    for adapter_group, sub_id in task_adapters:
        layer_stack_index = adapter_specs[f'{side}oder'][adapter_group]['layer_stack_index']
        adapters[layer_stack_index].append([adapter_group, sub_id])
    return adapters


def translation_configs(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    zero_shot = opts.zero_shot if opts.zero_shot else cc_opts.get('zero_shot', False)
    if not zero_shot:
        return

    # src_subword_model = opts.in_config[0].get('src_subword_model', None)
    # tgt_subword_model = opts.in_config[0].get('tgt_subword_model', None)
    # use_src_lang_token = cc_opts.get('use_src_lang_token', False)

    # TODO: create zero-shot tasks using the same template as for the supervised tasks, except that:
    # - no training set or validation set will be defined
    # - no weighting/curriculum
    # - no GPU allocation.
    # However, these 3 are needed: sharing_groups, set_transforms, adapter_config.
    # Because it would be nice to be able to add zero-shot tasks as a final extra step
    # without completely regenerating the entire training config,
    # these 3 should be modified to be rerunnable for a subset of tasks.

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def _write_translation_config(
    src_lang,
    tgt_lang,
    src_stack,
    tgt_stack,
    transforms,
    src_subword_model,
    tgt_subword_model,
    supervision,
    translation_config_dir,
    use_src_lang_token,
):
    # specify on command line: --model, --src
    result = {
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'stack': {
            'encoder': src_stack,
            'decoder': tgt_stack,
        }
    }
    if transforms:
        result['transforms'] = transforms
        if 'prefix' in transforms:
            if use_src_lang_token:
                prefix = f'<from_{src_lang}> <to_{tgt_lang}>'
            else:
                prefix = f'<to_{tgt_lang}>'
            result['src_prefix'] = prefix
            result['tgt_prefix'] = ''   # does not work, but must be set nonetheless
    if src_subword_model:
        result['src_subword_model'] = src_subword_model
    if tgt_subword_model:
        result['tgt_subword_model'] = tgt_subword_model
    translation_config_path = f'{translation_config_dir}/trans.{supervision}.{src_lang}-{tgt_lang}.yaml'
    with open(translation_config_path, 'w') as fout:
        serialized = yaml.safe_dump(result, default_flow_style=False, allow_unicode=True)
        print(serialized, file=fout)


def _get_langs(opts):
    src_langs = list(sorted(opts.in_config[0]['src_vocab'].keys()))
    tgt_langs = list(sorted(opts.in_config[0]['tgt_vocab'].keys()))
    return src_langs, tgt_langs


def complete_language_pairs(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    src_path_template = opts.src_path if opts.src_path else cc_opts['src_path']
    tgt_path_template = opts.tgt_path if opts.tgt_path else cc_opts['tgt_path']
    autoencoder = opts.autoencoder if opts.autoencoder else cc_opts.get('autoencoder', False)
    autoencoder_validation = (
        opts.autoencoder_validation if opts.autoencoder_validation else cc_opts.get('autoencoder_validation', False)
    )
    if autoencoder:
        ae_path_templates = None
        if opts.ae_path:
            ae_path_templates = opts.ae_path
        elif 'ae_path' in cc_opts:
            ae_path_templates = cc_opts['ae_path']
        if isinstance(ae_path_templates, str):
            ae_path_templates = [ae_path_templates]
        if ae_path_templates:
            ae_src_path_templates = ae_path_templates
            ae_tgt_path_templates = ae_path_templates
        else:
            ae_src_path_templates = [src_path_template]
            ae_tgt_path_templates = [tgt_path_template]
    valid_src_path_template = opts.valid_src_path if opts.valid_src_path else cc_opts['valid_src_path']
    valid_tgt_path_template = opts.valid_tgt_path if opts.valid_tgt_path else cc_opts['valid_tgt_path']

    src_langs, tgt_langs = _get_langs(opts)
    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            lang_a, lang_b = sorted((src_lang, tgt_lang))
            lang_pair = f'{src_lang}-{tgt_lang}'
            sorted_pair = f'{lang_a}-{lang_b}'
            if lang_pair == sorted_pair:
                # parallel data is used in "forward" direction
                side_a = 'src'
                side_b = 'trg'  # Tatoeba uses 'trg', not 'tgt'. Deal with it.
                lang_a = src_lang
                lang_b = tgt_lang
            else:
                # parallel data is used in "backward" direction
                side_a = 'trg'
                side_b = 'src'
                lang_a = tgt_lang
                lang_b = src_lang
            template_variables = {
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'lang_a': lang_a,
                'lang_b': lang_b,
                'side_a': side_a,
                'side_b': side_b,
                'lang_pair': lang_pair,
                'sorted_pair': sorted_pair
            }
            if src_lang == tgt_lang:
                # autoencoder task
                if not autoencoder:
                    continue
                for ae_src_path_template, ae_tgt_path_template in zip(ae_src_path_templates, ae_tgt_path_templates):
                    src_path = ae_src_path_template.format(**template_variables)
                    tgt_path = ae_tgt_path_template.format(**template_variables)
                    if not autoencoder_validation:
                        valid_src_path = None
                        valid_tgt_path = None
                    else:
                        valid_src_path = valid_src_path_template.format(**template_variables)
                        valid_tgt_path = valid_tgt_path_template.format(**template_variables)
                    if os.path.exists(src_path) and os.path.exists(tgt_path):
                        _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path, valid_src_path, valid_tgt_path)
            else:
                # translation task
                src_path = src_path_template.format(**template_variables)
                tgt_path = tgt_path_template.format(**template_variables)
                valid_src_path = valid_src_path_template.format(**template_variables)
                valid_tgt_path = valid_tgt_path_template.format(**template_variables)
                if os.path.exists(src_path) and os.path.exists(tgt_path):
                    _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path, valid_src_path, valid_tgt_path)
                else:
                    logger.warning(f'Paths do NOT exist, omitting language pair: {src_path} {tgt_path}')
    if len(opts.in_config[0].get('tasks', [])) == 0:
        raise Exception('No language pairs were added. Check your path templates.')
    # Allow using language variables for vocabulary definitions
    for src_lang in src_langs:
        opts.in_config[0]['src_vocab'][src_lang] = opts.in_config[0]['src_vocab'][src_lang].format(src_lang=src_lang)
    for tgt_lang in tgt_langs:
        opts.in_config[0]['tgt_vocab'][tgt_lang] = opts.in_config[0]['tgt_vocab'][tgt_lang].format(tgt_lang=tgt_lang)

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path, valid_src_path, valid_tgt_path):
    if 'tasks' not in opts.in_config[0]:
        opts.in_config[0]['tasks'] = dict()
    tasks_section = opts.in_config[0]['tasks']
    key = f'{src_lang}-{tgt_lang}'
    if key not in tasks_section:
        tasks_section[key] = dict()
    tasks_section[key]['src_tgt'] = f'{src_lang}-{tgt_lang}'
    tasks_section[key]['path_src'] = src_path
    tasks_section[key]['path_tgt'] = tgt_path
    if valid_src_path is not None and os.path.exists(valid_src_path):
        tasks_section[key]['path_valid_src'] = valid_src_path
        tasks_section[key]['path_valid_tgt'] = valid_tgt_path


def remove_temporary_keys(opts):
    # When reaching the end of the config-config, any excessive keys are
    # dropped before saving the yaml (OpenNMT doesn't like extra keys).
    del opts.in_config[0]['config_config']


def config_all(opts):
    start = time.time()

    complete_language_pairs(opts)
    corpora_schedule(opts)
    cluster_languages(opts)
    sharing_groups(opts)
    allocate_devices(opts)
    set_transforms(opts)
    adapter_config(opts)
    translation_configs(opts)
    remove_temporary_keys(opts)

    duration = time.time() - start
    logger.info(f'total took {duration} s')


def extra_cpu(opts):
    # Extra step: not included in config_all
    # Modifies config to run on a single CPU
    del opts.in_config[0]['gpu_ranks']
    del opts.in_config[0]['world_size']
    opts.in_config[0]['n_nodes'] = 1
    for task_opts in opts.in_config[0]['tasks'].values():
        if 'node_gpu' in task_opts:
            del task_opts['node_gpu']


def extra_fully_shared_hack(opts):
    # Extra step: not included in config_all
    # Modifies config to use the "all" language hack for a fully shared decoder
    for task_opts in opts.in_config[0]['tasks'].values():
        # Prefix transform to apply target language selection token
        if 'prefix' not in task_opts['transforms']:
            if task_opts['transforms'][-1] == 'filtertoolong':
                # insert prefix before last filtertoolong
                task_opts['transforms'].insert(-1, 'prefix')
            else:
                task_opts['transforms'].append('prefix')
            task_src, task_tgt = task_opts['src_tgt'].split('-')
            task_opts['src_prefix'] = f'<to_{task_src}>'
            task_opts['tgt_prefix'] = ''

        # decoder is fully shared
        task_opts['dec_sharing_group'] = ['full']

        # src_tgt overridden with a dummy value
        task_opts['src_tgt'] = 'all-all'

    # Override vocabs
    opts.in_config[0]['src_vocab'] = {'all': opts.joint_vocab}
    opts.in_config[0]['tgt_vocab'] = {'all': opts.joint_vocab}


def extra_copy_gpu_assignment(opts):
    tasks_in_current = set(opts.in_config[0]['tasks'].keys())
    tasks_in_source = set(opts.copy_from[0]['tasks'].keys())
    if not tasks_in_current == tasks_in_source:
        missing_tasks = tasks_in_current - tasks_in_source
        unused_tasks = tasks_in_source - tasks_in_current
        raise Exception(f'Tasks do not match. Missing tasks: {missing_tasks}, Unused tasks: {unused_tasks}')
    for task_key, task_opts in opts.in_config[0]['tasks'].items():
        task_opts['node_gpu'] = opts.copy_from[0]['tasks'][task_key]['node_gpu']
    opts.in_config[0]['n_nodes'] = opts.copy_from[0]['n_nodes']
    opts.in_config[0]['world_size'] = opts.copy_from[0]['world_size']
    opts.in_config[0]['gpu_ranks'] = opts.copy_from[0]['gpu_ranks']


def main():
    init_logging()
    opts = get_opts()
    # if not opts.out_config:
    #     opts.out_config = opts.in_config[1]
    command = {
        func.__name__: func
        for func in (
            complete_language_pairs,
            corpora_schedule,
            cluster_languages,
            sharing_groups,
            set_transforms,
            allocate_devices,
            adapter_config,
            translation_configs,
            remove_temporary_keys,
            config_all,
            extra_cpu,
            extra_fully_shared_hack,
            extra_copy_gpu_assignment,
        )
    }[opts.command]
    command(opts)
    save_yaml(opts)


if __name__ == '__main__':
    main()
