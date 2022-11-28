import argparse
import yaml


def load_yaml(fname):
    with open(fname, 'r') as istr:
        config = yaml.safe_load(istr)
    return config, fname


def save_yaml(opts):
    with open(opts.out_config, 'w') as ostr:
        yaml.dump(opts.in_config[0], ostr, default_flow_style=False, allow_unicode=True)


def get_opts():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser_corpora_schedule = subparsers.add_subparser('corpora_schedule')
    parser_corpora_schedule.add_argument('--in_config', required=True, type=load_yaml)
    parser_corpora_schedule.add_argument('--out_config', type=str)
    parser_corpora_schedule.add_argument('--use_weights', action='store_true')
    parser_define_group = subparsers.add_subparser('define_group')
    parser_define_group.add_argument('--in_config', required=True, type=load_yaml)
    parser_define_group.add_argument('--out_config', type=str)
    parser_define_group.add_argument('-similarity_matrix', required=True)
    parser_define_group.add_argument('-cutoff_threshold', required=True, type=float)
    parser_allocate_devices = subparsers.add_subparser('allocate_devices')
    parser_allocate_devices.add_argument('--in_config', required=True, type=load_yaml)
    parser_allocate_devices.add_argument('--out_config', type=str)
    parser_allocate_devices.add_argument('-n_devices', type=int, required=True)
    parser_allocate_devices.add_argument('-n_nodes', type=int, required=True)
    parser_adapter_config = subparsers.add_subparser('adapter_config')
    parser_adapter_config.add_argument('--in_config', required=True, type=load_yaml)
    parser_adapter_config.add_argument('--out_config', type=str)
    return parser.parse_args()


def corpora_schedule(opts):
    corpora_lens = {}
    for cname, corpus in opts.in_config[0]['data'].items():
        with open(corpus['path_src'], 'r') as istr:
            corpora_lens[cname] = sum(1 for _ in istr)
    max_lines = max(corpora_lens.values())
    corpora_weights = {
        cname: (max_lines - clen) / max_lines
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
    save_yaml(opts)


def define_group(opts):
    pass


def allocate_devices(opts):
    pass


def adapter_config(opts):
    pass


if __name__ == '__main__':
    opts = get_opts()
    if not opts.out_config:
        opts.out_config = opts.in_config[1]
    main = {
        func.__name__: func
        for func in (corpora_schedule, define_group, allocate_devices, adapter_config)
    }[opts.command]
    main(opts)
