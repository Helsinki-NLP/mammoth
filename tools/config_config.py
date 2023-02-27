import argparse
import csv
import numpy as np
import os
import warnings
import yaml
from collections import defaultdict
from itertools import compress
from sklearn.cluster import AgglomerativeClustering

from gpu_assignment import optimize_gpu_assignment


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


def add_complete_language_pairs_args(parser):
    parser.add_argument(
        '--src_path', type=str, required=True,
        help='path template to source data. Can use variables {src_lang} and {tgt_lang}'
    )
    parser.add_argument(
        '--tgt_path', type=str, required=True,
        help='path template to target data. Can use variables {src_lang} and {tgt_lang}'
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
        '--use_introduce_at_training_step', action='store_true',
        help='Use a curriculum introducing corpora based on temperature-adjusted corpus size'
    )
    parser.add_argument('--temperature', type=float, default=1.0)


def add_define_group_args(parser):
    parser.add_argument('--distance_matrix', required=True, type=load_distmat_csv)
    parser.add_argument('--cutoff_threshold', type=float)
    parser.add_argument('--n_groups', type=int)
    parser.add_argument(
        '--enc_sharing_groups', type=str, action='append',
        help='list of {LANGUAGE | GROUP | FULL}'
    )
    parser.add_argument(
        '--dec_sharing_groups', type=str, action='append',
        help='list of {LANGUAGE | GROUP | FULL}'
    )


def add_allocate_device_args(parser):
    parser.add_argument('--n_nodes', type=int, required=True)
    parser.add_argument('--n_gpus_per_node', type=int, required=True)
    parser.add_argument('--n_slots_per_gpu', type=int, default=None)


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
    parser.add_argument('--translation_config_dir', type=str, default='config/translation')
    parser.add_argument('--zero_shot', action='store_true')


def get_opts():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser_corpora_schedule = subparsers.add_parser('corpora_schedule')
    add_configs_args(parser_corpora_schedule)
    add_corpora_schedule_args(parser_corpora_schedule)
    parser_define_group = subparsers.add_parser('define_group')
    add_configs_args(parser_define_group)
    add_define_group_args(parser_define_group)
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
    parser_config_all = subparsers.add_parser('config_all')
    add_configs_args(parser_config_all)
    add_corpora_schedule_args(parser_config_all)
    add_define_group_args(parser_config_all)
    add_allocate_device_args(parser_config_all)
    add_set_transforms_args(parser_config_all)
    add_complete_language_pairs_args(parser_config_all)
    add_adapter_config_args(parser_config_all)
    add_translation_configs_args(parser_config_all)
    return parser.parse_args()


def corpora_schedule(opts):
    corpora_lens = {}
    for cname, corpus in opts.in_config[0]['data'].items():
        with open(corpus['path_src'], 'r') as istr:
            corpora_lens[cname] = sum(1 for _ in istr)
    max_lines = max(corpora_lens.values())
    corpora_weights = {
        cname: (max_lines - clen) ** opts.temperature / max_lines
        for cname, clen in corpora_lens.items()
    }
    for cname, corpus in opts.in_config[0]['data'].items():
        weight = 1 - corpora_weights[cname]
        if opts.use_weight and opts.use_introduce_at_training_step:
            weight = float(np.sqrt(weight))
        if opts.use_weight:
            corpus['weight'] = weight
        if opts.use_introduce_at_training_step:
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


def define_group(opts):
    sim_langs = set(opts.distance_matrix['header'])
    corpus_langs = set()
    for cname, corpus in opts.in_config[0]['data'].items():
        assert all([(lng in sim_langs) for lng in corpus['src_tgt'].split('-')]), \
            f'corpus {cname}: one language (either {" or ".join(corpus["src_tgt"].split("-"))} ' \
            f'was not found in the distance matrix (supports {" ".join(sim_langs)})'
        corpus_langs = corpus_langs | set(corpus['src_tgt'].split('-'))
    if sim_langs != corpus_langs:
        warnings.warn(
            f"languages in the distance matrix are unused ({', ' .join(sim_langs - corpus_langs)})"
        )
        # Omit unused languages before clustering. Otherwise they might consume entire clusters.
        selector = [lang in corpus_langs for lang in opts.distance_matrix['header']]
        dist = opts.distance_matrix['data']
        dist = dist[selector][:, selector]
        header = list(compress(opts.distance_matrix['header'], selector))
        opts.distance_matrix = {
            'data': dist,
            'header': header,
        }

    group_idx = AgglomerativeClustering(
        n_clusters=opts.n_groups,
        metric='precomputed',
        linkage='average',
        distance_threshold=opts.cutoff_threshold,
    ).fit_predict(opts.distance_matrix['data']).tolist()
    groups = {lang: f'group{idx}' for lang, idx in zip(opts.distance_matrix['header'], group_idx)}
    # FIXME: storing groups in opts for later use like this is a problem:
    # The adapter_config step can not be run without also running the
    # define_group step in the same execution (i.e. using "config_all").
    # A potential solution would be to save everything in the config structure:
    #   - Configuration for the config-config (what is now specified as CLI params)
    #   - Intermediary values computed in the steps (such as the lang -> group mapping)
    #   - Final configuration values
    # When reaching the end of the config-config, any excessive keys are
    # dropped before saving the yaml (OpenNMT doesn't like extra keys).
    # Why does this work? Any step can be omitted, by instead adding any
    # intermediary values it would produce into the input config. E.g. the lang
    # -> group mapping could be specified as a mapping in the input yaml instead of a csv.
    opts.groups = groups

    if not opts.enc_sharing_groups:
        raise Exception('Must set --enc_sharing_groups')
    if not opts.dec_sharing_groups:
        raise Exception('Must set --dec_sharing_groups')
    assert len(opts.enc_sharing_groups) == len(opts.in_config[0]['enc_layers'])
    assert len(opts.dec_sharing_groups) == len(opts.in_config[0]['dec_layers'])
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
            mapping_src[sharing_group] for sharing_group in opts.enc_sharing_groups
        ]
        corpus['dec_sharing_group'] = [
            mapping_tgt[sharing_group] for sharing_group in opts.dec_sharing_groups
        ]


def set_transforms(opts):
    for cname, corpus in opts.in_config[0]['data'].items():
        src, tgt = corpus['src_tgt'].split('-')
        if src == tgt:
            corpus['transforms'] = list(opts.ae_transforms)
        else:
            corpus['transforms'] = list(opts.transforms)


def allocate_devices(opts):
    lang_pairs = []
    lps_ready_to_start = []
    lp_to_key = {}
    for key, data_config in opts.in_config[0]['data'].items():
        src_lang, tgt_lang = data_config['src_tgt'].split('-')
        ready_to_start = data_config.get('introduce_at_training_step', 0) == 0

        lang_pairs.append((src_lang, tgt_lang))
        if ready_to_start:
            lps_ready_to_start.append((src_lang, tgt_lang))
        lp_to_key[(src_lang, tgt_lang)] = key

    if opts.n_slots_per_gpu is not None:
        n_slots_per_gpu = opts.n_slots_per_gpu
    else:
        n_gpus_tot = opts.n_nodes * opts.n_gpus_per_node
        n_slots_per_gpu = int(np.ceil(len(lang_pairs) / n_gpus_tot))

    assignment = optimize_gpu_assignment(
        n_nodes=opts.n_nodes,
        n_gpus_per_node=opts.n_gpus_per_node,
        n_slots_per_gpu=n_slots_per_gpu,
        lang_pairs=lang_pairs,
        lang_to_group_mapping=opts.groups,
        lps_ready_to_start=lps_ready_to_start,
    )

    for gpu_slot, lp in assignment.items():
        if lp is None:
            continue
        key = lp_to_key[lp]
        opts.in_config[0]['data'][key]['node_gpu'] = f'{gpu_slot.node}:{gpu_slot.gpu}'


def adapter_config(opts):
    if 'adapters' not in opts.in_config[0]:
        warnings.warn('No adapter configuration, skipping this step')
        return
    src_langs, tgt_langs = _get_langs(opts)
    src_groups = list(sorted(set(opts.groups[src] for src in src_langs)))
    tgt_groups = list(sorted(set(opts.groups[tgt] for tgt in tgt_langs)))
    encoder_adapters = opts.in_config[0]['adapters'].get('encoder', [])
    decoder_adapters = opts.in_config[0]['adapters'].get('decoder', [])
    for data_key, data_config in opts.in_config[0]['data'].items():
        if 'adapters' not in data_config:
            data_config['adapters'] = {'encoder': [], 'decoder': []}
    for adapter_name, adapter_config in encoder_adapters.items():
        if adapter_config['ids'] == 'LANGUAGE':
            adapter_config['ids'] = src_langs
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_src, data_tgt = data_config['src_tgt'].split('-')
                data_config['adapters']['encoder'].append([adapter_name, data_src])
        elif adapter_config['ids'] == 'GROUP':
            adapter_config['ids'] = src_groups
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_src, data_tgt = data_config['src_tgt'].split('-')
                data_config['adapters']['encoder'].append([adapter_name, opts.groups[data_src]])
        elif adapter_config['ids'] == 'FULL':
            adapter_config['ids'] = ['full']
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_config['adapters']['encoder'].append([adapter_name, 'full'])
    for adapter_name, adapter_config in decoder_adapters.items():
        if adapter_config['ids'] == 'LANGUAGE':
            adapter_config['ids'] = tgt_langs
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_src, data_tgt = data_config['src_tgt'].split('-')
                data_config['adapters']['decoder'].append([adapter_name, data_tgt])
        elif adapter_config['ids'] == 'GROUP':
            adapter_config['ids'] = tgt_groups
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_src, data_tgt = data_config['src_tgt'].split('-')
                data_config['adapters']['decoder'].append([adapter_name, opts.groups[data_tgt]])
        elif adapter_config['ids'] == 'FULL':
            adapter_config['ids'] = ['full']
            for data_key, data_config in opts.in_config[0]['data'].items():
                data_config['adapters']['decoder'].append([adapter_name, 'full'])
    opts.in_config[0]['adapters']['encoder'] = encoder_adapters
    opts.in_config[0]['adapters']['decoder'] = decoder_adapters


def _adapters_to_stacks(task_adapters, opts, side):
    adapter_specs = opts.in_config[0]['adapters']
    adapters = [list() for _ in range(len(opts.in_config[0][f'{side}_layers']))]
    for adapter_group, sub_id in task_adapters:
        layer_stack_index = adapter_specs[f'{side}oder'][adapter_group]['layer_stack_index']
        adapters[layer_stack_index].append([adapter_group, sub_id])
    return adapters


def translation_configs(opts):
    translation_config_dir = opts.translation_config_dir
    os.makedirs(translation_config_dir, exist_ok=True)
    encoder_stacks = defaultdict(dict)
    decoder_stacks = defaultdict(dict)
    supervised_pairs = set()
    for task_opts in opts.in_config[0]['data'].values():
        src_lang, tgt_lang = task_opts['src_tgt'].split('-')
        if src_lang == tgt_lang:
            continue
        # src / encoder
        src_stack = [{'id': group} for group in task_opts['enc_sharing_group']]
        if 'adapters' in task_opts:
            adapters_by_stack = _adapters_to_stacks(task_opts['adapters']['encoder'], opts, 'enc')
            print(f'src, src_stack {src_stack}')
            print(f'src, adapters_by_stack {adapters_by_stack}')
            assert len(src_stack) == len(adapters_by_stack)
            for stack, adapters in zip(src_stack, adapters_by_stack):
                stack['adapters'] = adapters
        key = str(src_stack)    # An ugly way to freeze the mutable structure
        encoder_stacks[src_lang][key] = src_stack
        # tgt / decoder
        tgt_stack = [{'id': group} for group in task_opts['dec_sharing_group']]
        if 'adapters' in task_opts:
            adapters_by_stack = _adapters_to_stacks(task_opts['adapters']['decoder'], opts, 'dec')
            print(f'tgt, tgt_stack {tgt_stack}')
            print(f'tgt, adapters_by_stack {adapters_by_stack}')
            assert len(tgt_stack) == len(adapters_by_stack)
            for stack, adapters in zip(tgt_stack, adapters_by_stack):
                stack['adapters'] = adapters
        key = str(tgt_stack)    # An ugly way to freeze the mutable structure
        decoder_stacks[tgt_lang][key] = tgt_stack
        # Write config for the supervised directions
        _write_translation_config(
            src_lang,
            tgt_lang,
            src_stack,
            tgt_stack,
            'supervised',
            translation_config_dir,
        )
        supervised_pairs.add((src_lang, tgt_lang))
    if opts.zero_shot:
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
                _write_translation_config(
                    src_lang,
                    tgt_lang,
                    src_stack,
                    tgt_stack,
                    'zeroshot',
                    translation_config_dir,
                )


def _write_translation_config(
    src_lang,
    tgt_lang,
    src_stack,
    tgt_stack,
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
    translation_config_path = f'{translation_config_dir}/trans.{supervision}.{src_lang}-{tgt_lang}.yaml'
    with open(translation_config_path, 'w') as fout:
        serialized = yaml.safe_dump(result, default_flow_style=False, allow_unicode=True)
        print(serialized, file=fout)


def _get_langs(opts):
    src_langs = list(sorted(opts.in_config[0]['src_vocab'].keys()))
    tgt_langs = list(sorted(opts.in_config[0]['tgt_vocab'].keys()))
    return src_langs, tgt_langs


def complete_language_pairs(opts):
    src_langs, tgt_langs = _get_langs(opts)
    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            if src_lang == tgt_lang and not opts.autoencoder:
                continue
            src_path = opts.src_path.format(src_lang=src_lang, tgt_lang=tgt_lang)
            tgt_path = opts.tgt_path.format(src_lang=src_lang, tgt_lang=tgt_lang)
            if os.path.exists(src_path) and os.path.exists(tgt_path):
                _add_language_pair(opts, src_lang, tgt_lang, src_path, tgt_path)
            else:
                print(f'Paths do NOT exist, omitting language pair: {src_path} {tgt_path}')
    if len(opts.in_config[0].get('data', [])) == 0:
        raise Exception('No language pairs were added. Check your path templates.')


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


def config_all(opts):
    complete_language_pairs(opts)
    corpora_schedule(opts)
    define_group(opts)
    allocate_devices(opts)
    set_transforms(opts)
    adapter_config(opts)
    translation_configs(opts)


if __name__ == '__main__':
    opts = get_opts()
    # if not opts.out_config:
    #     opts.out_config = opts.in_config[1]
    main = {
        func.__name__: func
        for func in (
            complete_language_pairs,
            corpora_schedule,
            define_group,
            set_transforms,
            allocate_devices,
            adapter_config,
            translation_configs,
            config_all,
        )
    }[opts.command]
    main(opts)
    save_yaml(opts)
