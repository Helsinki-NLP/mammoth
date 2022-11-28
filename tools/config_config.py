import argparse
import csv
import warnings
import yaml

import numpy as np
from sklearn.cluster import AgglomerativeClustering


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
    with open(opts.out_config, 'w') as ostr:
        yaml.dump(opts.in_config[0], ostr, default_flow_style=False, allow_unicode=True)


def add_complete_language_pairs_args(parser):
    pass


def add_configs_args(parser):
    parser.add_argument('--in_config', required=True, type=load_yaml)
    parser.add_argument('--out_config', type=str)


def add_corpora_schedule_args(parser):
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--temperature', type=float)


def add_define_group_args(parser):
    parser.add_argument('--distance_matrix', required=True, type=load_distmat_csv)
    parser.add_argument('--cutoff_threshold', type=float)
    parser.add_argument('--n_groups', type=int)


def add_allocate_device_args(parser):
    parser.add_argument('--n_devices', type=int, required=True)
    parser.add_argument('--n_nodes', type=int, required=True)


def add_adapter_config_args(parser):
    pass


def get_opts():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser_corpora_schedule = subparsers.add_subparser('corpora_schedule')
    add_configs_args(parser_corpora_schedule)
    add_corpora_schedule_args(parser_corpora_schedule)
    parser_define_group = subparsers.add_subparser('define_group')
    add_configs_args(parser_define_group)
    add_define_group_args(parser_define_group)
    parser_allocate_devices = subparsers.add_subparser('allocate_devices')
    add_configs_args(parser_allocate_devices)
    add_allocate_device_args(parser_allocate_devices)
    parser_adapter_config = subparsers.add_subparser('adapter_config')
    add_configs_args(parser_adapter_config)
    add_adapter_config_args(parser_adapter_config)
    parser_complete_language_pairs = subparsers.add_subparser('complete_language_pairs')
    add_configs_args(parser_complete_language_pairs)
    add_complete_language_pairs_args(parser_complete_language_pairs)
    parser_config_all = subparsers.add_subparser('config_all')
    add_configs_args(parser_config_all)
    add_corpora_schedule_args(parser_config_all)
    add_define_group_args(parser_config_all)
    add_allocate_device_args(parser_config_all)
    add_complete_language_pairs_args(parser_config_all)
    add_adapter_config_args(parser_config_all)
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
    if opts.use_weight:
        for cname, corpus in opts.in_config[0]['data'].items():
            corpus['weight'] = 1 - corpora_weights[cname]
    else:
        # TODO: ensure this default always matches with opts.py
        total_steps = opts.in_config[0].get('train_steps', 100_000)
        for cname, corpus in opts.in_config[0]['data'].items():
            corpus['introduce_at_training_step'] = round(total_steps * corpora_weights[cname])


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

    group_idx = AgglomerativeClustering(
        n_clusters=opts.n_groups,
        affinity='precomputed',
        distance_threshold=opts.cutoff_threshold,
    ).fit_predict(opts.distance_matrix['data']).tolist()
    groups = {lang: f'group{idx}' for lang, idx in zip(opts.distance_matrix['header'], group_idx)}

    for cname, corpus in opts.in_config[0]['data'].items():
        src, tgt = corpus['src_tgt'].split('-')
        corpus['enc_sharing_group'] = [groups[src], 'full']
        corpus['dec_sharing_group'] = [groups[tgt], 'full', groups[tgt]]


def allocate_devices(opts):
    pass


def adapter_config(opts):
    pass


def complete_language_pairs(opts):
    pass


def config_all(opts):
    complete_language_pairs(opts)
    corpora_schedule(opts)
    define_group(opts)
    allocate_devices(opts)
    adapter_config(opts)


if __name__ == '__main__':
    opts = get_opts()
    if not opts.out_config:
        opts.out_config = opts.in_config[1]
    main = {
        func.__name__: func
        for func in (
            complete_language_pairs,
            corpora_schedule,
            define_group,
            allocate_devices,
            adapter_config,
            config_all,
        )
    }[opts.command]
    main(opts)
    save_yaml(opts)
