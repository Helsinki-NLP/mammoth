import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser_starting_step = subparsers.add_subparser('starting_step')
    parser_starting_step.add_argument('-in_config')
    parser_starting_step.add_argument('-out_config')
    parser_define_group = subparsers.add_subparser('define_group')
    parser_define_group.add_argument('-in_config')
    parser_define_group.add_argument('-out_config')
    parser_allocate_devices = subparsers.add_subparser('allocate_devices')
    parser_allocate_devices.add_argument('-in_config')
    parser_allocate_devices.add_argument('-out_config')
    parser_adapter_config = subparsers.add_subparser('adapter_config')
    parser_adapter_config.add_argument('-in_config')
    parser_adapter_config.add_argument('-out_config')


def starting_step(opts):
    pass


def define_group(opts):
    pass


def allocate_devices(opts):
    pass


def adapter_config(opts):
    pass


if __name__ == '__main__':
    opts = get__opts()
    main = {
        func.__name__: func
        for func in (starting_step, define_group, allocate_devices, adapter_config)
    }[opts.command]
    main(opts)
