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

from gpu_assignment import optimize_gpu_assignment

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
        '--autoencoder',
        action='store_true',
        help='add autoencoder tasks, for which src_lang == tgt_lang'
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
        help='list of {LANGUAGE | GROUP | FULL}'
    )
    parser.add_argument(
        '--dec_sharing_groups', type=str, action='append',
        help='list of {LANGUAGE | GROUP | FULL}'
    )


def add_allocate_device_args(parser):
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--n_gpus_per_node', type=int)
    parser.add_argument('--n_slots_per_gpu', type=int)


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
    parser.add_argument('--translation_config_dir', type=str)
    parser.add_argument('--zero_shot', action='store_true')


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
    return parser.parse_args()


def _split_large_language_pairs(opts, corpora_weights, split_treshold):
    corpora_out = deepcopy(opts.in_config[0]['data'])
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
    opts.in_config[0]['data'] = corpora_out
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
        logger.info(f'{path}:\t{len}')
    corpora_lens = {}
    for cname, corpus in opts.in_config[0]['data'].items():
        if corpus['path_src'] in corpora_lens_cache:
            length = corpora_lens_cache[corpus['path_src']]
            corpora_lens[cname] = length
        else:
            length = external_linecount(corpus['path_src'])
            corpora_lens[cname] = length
            with open(corpora_lens_cache_file, 'a') as cache_out:
                print(f'{length}\t{corpus["path_src"]}', file=cache_out)
                logger.info(f'{length}\t{corpus["path_src"]}')
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

    for cname, corpus in opts.in_config[0]['data'].items():
        src_lang, tgt_lang = corpus['src_tgt'].split('-')
        weight = corpora_weights[cname]
        if use_weight and use_introduce_at_training_step:
            weight = float(np.sqrt(weight))
        if use_weight:
            multiplier = ae_weight if src_lang == tgt_lang else 1.0
            corpus['weight'] = weight * multiplier
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
    for cname, corpus in opts.in_config[0]['data'].items():
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
    for cname, corpus in opts.in_config[0]['data'].items():
        src, tgt = corpus['src_tgt'].split('-')
        mapping_src = {
            'LANGUAGE': src,
            'GROUP': groups[src],
            'FULL': 'full',
        }
        mapping_tgt = {
            'LANGUAGE': tgt,
            'GROUP': groups[tgt],
            'FULL': 'full',
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

    for cname, corpus in opts.in_config[0]['data'].items():
        src, tgt = corpus['src_tgt'].split('-')
        if src == tgt:
            corpus['transforms'] = list(ae_transforms)
        else:
            corpus['transforms'] = list(transforms)

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def allocate_devices(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    n_nodes = opts.n_nodes if opts.n_nodes else cc_opts['n_nodes']
    n_gpus_per_node = opts.n_gpus_per_node if opts.n_gpus_per_node else cc_opts['n_gpus_per_node']
    n_slots_per_gpu = opts.n_slots_per_gpu if opts.n_slots_per_gpu else cc_opts.get('n_slots_per_gpu', None)

    lang_pairs = []
    lps_ready_to_start = []
    lp_to_key = defaultdict(list)
    for key, data_config in opts.in_config[0]['data'].items():
        src_lang, tgt_lang = data_config['src_tgt'].split('-')
        ready_to_start = data_config.get('introduce_at_training_step', 0) == 0

        lang_pairs.append((src_lang, tgt_lang))
        if ready_to_start:
            lps_ready_to_start.append((src_lang, tgt_lang))
        lp_to_key[(src_lang, tgt_lang)].append(key)

    if n_slots_per_gpu is None:
        n_gpus_tot = n_nodes * n_gpus_per_node
        n_slots_per_gpu = int(np.ceil(len(lang_pairs) / n_gpus_tot))
    logger.info(f'n_nodes:          {n_nodes}')
    logger.info(f'n_gpus_per_node:  {n_gpus_per_node}')
    logger.info(f'n_slots_per_gpu:  {n_slots_per_gpu}')
    logger.info(f'total slots:      {n_nodes * n_gpus_per_node * n_slots_per_gpu}')
    logger.info(f'lang_pairs:       {len(lang_pairs)}')

    assignment = optimize_gpu_assignment(
        n_nodes=n_nodes,
        n_gpus_per_node=n_gpus_per_node,
        n_slots_per_gpu=n_slots_per_gpu,
        lang_pairs=lang_pairs,
        lang_to_group_mapping=cc_opts['groups'],
        lps_ready_to_start=lps_ready_to_start,
    )

    for gpu_slot, lp in assignment.items():
        if lp is None:
            continue
        key = lp_to_key[lp].pop()
        opts.in_config[0]['data'][key]['node_gpu'] = f'{gpu_slot.node}:{gpu_slot.gpu}'

    opts.in_config[0]['n_nodes'] = n_nodes
    opts.in_config[0]['world_size'] = n_gpus_tot
    opts.in_config[0]['gpu_ranks'] = list(range(n_gpus_per_node))

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def adapter_config(opts):
    start = time.time()

    cc_opts = opts.in_config[0]['config_config']
    if 'adapters' not in opts.in_config[0]:
        warnings.warn('No adapter configuration, skipping this step')
        return
    src_langs, tgt_langs = _get_langs(opts)
    src_groups = list(sorted(set(cc_opts['groups'][src] for src in src_langs)))
    tgt_groups = list(sorted(set(cc_opts['groups'][tgt] for tgt in tgt_langs)))
    encoder_adapters = opts.in_config[0]['adapters'].get('encoder', [])
    decoder_adapters = opts.in_config[0]['adapters'].get('decoder', [])
    for data_key, data_config in opts.in_config[0]['data'].items():
        if 'adapters' not in data_config:
            data_config['adapters'] = {'encoder': [], 'decoder': []}
    for adapter_name, adapter_config in sorted(encoder_adapters.items()):
        if adapter_config['ids'] == 'LANGUAGE':
            adapter_config['ids'] = list(src_langs)
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_src, data_tgt = data_config['src_tgt'].split('-')
                data_config['adapters']['encoder'].append([adapter_name, data_src])
        elif adapter_config['ids'] == 'GROUP':
            adapter_config['ids'] = list(src_groups)
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_src, data_tgt = data_config['src_tgt'].split('-')
                data_config['adapters']['encoder'].append([adapter_name, cc_opts['groups'][data_src]])
        elif adapter_config['ids'] == 'FULL':
            adapter_config['ids'] = ['full']
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_config['adapters']['encoder'].append([adapter_name, 'full'])
    for adapter_name, adapter_config in sorted(decoder_adapters.items()):
        if adapter_config['ids'] == 'LANGUAGE':
            adapter_config['ids'] = list(tgt_langs)
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_src, data_tgt = data_config['src_tgt'].split('-')
                data_config['adapters']['decoder'].append([adapter_name, data_tgt])
        elif adapter_config['ids'] == 'GROUP':
            adapter_config['ids'] = list(tgt_groups)
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_src, data_tgt = data_config['src_tgt'].split('-')
                data_config['adapters']['decoder'].append([adapter_name, cc_opts['groups'][data_tgt]])
        elif adapter_config['ids'] == 'FULL':
            adapter_config['ids'] = ['full']
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_config['adapters']['decoder'].append([adapter_name, 'full'])
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
    translation_config_dir = (
        opts.translation_config_dir if opts.translation_config_dir
        else cc_opts.get('translation_config_dir', 'config/translation')
    )
    zero_shot = opts.zero_shot if opts.zero_shot else cc_opts.get('zero_shot', False)

    src_subword_model = opts.in_config[0].get('src_subword_model', None)
    tgt_subword_model = opts.in_config[0].get('tgt_subword_model', None)

    os.makedirs(translation_config_dir, exist_ok=True)
    encoder_stacks = defaultdict(dict)
    decoder_stacks = defaultdict(dict)
    transforms_by_lang = defaultdict(dict)
    supervised_pairs = set()
    for task_opts in opts.in_config[0]['data'].values():
        src_lang, tgt_lang = task_opts['src_tgt'].split('-')
        if src_lang == tgt_lang:
            continue
        # src / encoder
        src_stack = [{'id': group} for group in task_opts['enc_sharing_group']]
        if 'adapters' in task_opts:
            adapters_by_stack = _adapters_to_stacks(task_opts['adapters']['encoder'], opts, 'enc')
            assert len(src_stack) == len(adapters_by_stack)
            for stack, adapters in zip(src_stack, adapters_by_stack):
                stack['adapters'] = adapters
        key = str(src_stack)    # An ugly way to freeze the mutable structure
        encoder_stacks[src_lang][key] = src_stack
        # tgt / decoder
        tgt_stack = [{'id': group} for group in task_opts['dec_sharing_group']]
        if 'adapters' in task_opts:
            adapters_by_stack = _adapters_to_stacks(task_opts['adapters']['decoder'], opts, 'dec')
            assert len(tgt_stack) == len(adapters_by_stack)
            for stack, adapters in zip(tgt_stack, adapters_by_stack):
                stack['adapters'] = adapters
        key = str(tgt_stack)    # An ugly way to freeze the mutable structure
        decoder_stacks[tgt_lang][key] = tgt_stack
        # Transforms and subword models also need to be respecified during translation
        if 'transforms' not in task_opts:
            transforms = None
        else:
            transforms = [
                transform for transform in task_opts['transforms']
                if not transform == 'filtertoolong'
            ]
        transforms_by_lang[src_lang] = transforms
        # Write config for the supervised directions
        _write_translation_config(
            src_lang,
            tgt_lang,
            src_stack,
            tgt_stack,
            transforms,
            src_subword_model,
            tgt_subword_model,
            'supervised',
            translation_config_dir,
        )
        supervised_pairs.add((src_lang, tgt_lang))
    if zero_shot:
        src_langs = encoder_stacks.keys()
        tgt_langs = decoder_stacks.keys()
        # verify that there is an unique stack for each language
        ambiguous_src = [src_lang for src_lang in encoder_stacks if len(encoder_stacks[src_lang]) > 1]
        ambiguous_tgt = [tgt_lang for tgt_lang in decoder_stacks if len(decoder_stacks[tgt_lang]) > 1]
        if len(ambiguous_src) > 0 or len(ambiguous_tgt) > 0:
            raise Exception(
                'Zero-shot translation configs can only be generated if each source (target) language '
                'has an unambigous encoder (decoder) stack.\n'
                'The following languages have more than one encoder/decoder stack:\n'
                f'Source: {ambiguous_src}\nTarget: {ambiguous_tgt}'
            )
        for src_lang in src_langs:
            for tgt_lang in tgt_langs:
                if src_lang == tgt_lang:
                    continue
                if (src_lang, tgt_lang) in supervised_pairs:
                    continue
                src_stack = list(encoder_stacks[src_lang].values())[0]
                tgt_stack = list(decoder_stacks[tgt_lang].values())[0]
                transforms = transforms_by_lang[src_lang]
                _write_translation_config(
                    src_lang,
                    tgt_lang,
                    src_stack,
                    tgt_stack,
                    transforms,
                    src_subword_model,
                    tgt_subword_model,
                    'zeroshot',
                    translation_config_dir,
                )

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
    if autoencoder:
        ae_path_template = opts.ae_path if opts.ae_path else cc_opts.get('ae_path', None)
    if ae_path_template:
        ae_src_path_template = ae_path_template
        ae_tgt_path_template = ae_path_template
    else:
        ae_src_path_template = src_path_template
        ae_tgt_path_template = tgt_path_template

    src_langs, tgt_langs = _get_langs(opts)
    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            sorted_pair = '-'.join(sorted((src_lang, tgt_lang)))
            if src_lang == tgt_lang:
                # autoencoder task
                if not autoencoder:
                    continue
                src_path = ae_src_path_template.format(src_lang=src_lang, tgt_lang=tgt_lang, sorted_pair=sorted_pair)
                tgt_path = ae_tgt_path_template.format(src_lang=src_lang, tgt_lang=tgt_lang, sorted_pair=sorted_pair)
            else:
                # translation task
                src_path = src_path_template.format(src_lang=src_lang, tgt_lang=tgt_lang, sorted_pair=sorted_pair)
                tgt_path = tgt_path_template.format(src_lang=src_lang, tgt_lang=tgt_lang, sorted_pair=sorted_pair)
            if os.path.exists(src_path) and os.path.exists(tgt_path):
                _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path)
            else:
                logger.warning(f'Paths do NOT exist, omitting language pair: {src_path} {tgt_path}')
    if len(opts.in_config[0].get('data', [])) == 0:
        raise Exception('No language pairs were added. Check your path templates.')
    # Allow using language variables for vocabulary definitions
    for src_lang in src_langs:
        opts.in_config[0]['src_vocab'][src_lang] = opts.in_config[0]['src_vocab'][src_lang].format(src_lang=src_lang)
    for tgt_lang in tgt_langs:
        opts.in_config[0]['tgt_vocab'][tgt_lang] = opts.in_config[0]['tgt_vocab'][tgt_lang].format(tgt_lang=tgt_lang)

    duration = time.time() - start
    logger.info(f'step took {duration} s')


def _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path):
    if 'data' not in opts.in_config[0]:
        opts.in_config[0]['data'] = dict()
    data_section = opts.in_config[0]['data']
    key = f'train_{src_lang}-{tgt_lang}'
    if key not in data_section:
        data_section[key] = dict()
    data_section[key]['src_tgt'] = f'{src_lang}-{tgt_lang}'
    data_section[key]['path_src'] = src_path
    data_section[key]['path_tgt'] = tgt_path


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


if __name__ == '__main__':
    init_logging()
    opts = get_opts()
    # if not opts.out_config:
    #     opts.out_config = opts.in_config[1]
    main = {
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
        )
    }[opts.command]
    main(opts)
    save_yaml(opts)
