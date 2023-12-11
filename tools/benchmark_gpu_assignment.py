#!/usr/bin/env python
import click
import random
import yaml
from itertools import product

def create_langs(n_langs):
    langs = ['centric']
    langs.extend(["lang{:05d}".format(i) for i in range(2, n_langs + 1)])
    return langs

def centric_pairs(langs):
    result = []
    non_centric = [lang for lang in langs if lang != 'centric']
    for other in non_centric:
        result.append(('centric', other))
        result.append((other, 'centric'))
    result.append(('centric', 'centric'))
    return result

def complete_language_pairs(langs, sparsity):
    if sparsity == 'multiparallel':
        return list(product(langs, langs))
    pairs = centric_pairs(langs)
    if sparsity == 'centric':
        return pairs

def create_tasks(lang_pairs, curriculum):
    tasks = dict()
    if curriculum:
        tmp = list(lang_pairs)
        random.shuffle(tmp)
        half = len(tmp) // 2
        not_ready = set(tmp[:half])
    else:
        not_ready = set()
    for src_lang, tgt_lang in lang_pairs:
        tasks[f'train_{src_lang}_{tgt_lang}'] = {
            'src_tgt': f'{src_lang}-{tgt_lang}',
            'introduce_at_training_step': 1000 if (src_lang, tgt_lang) in not_ready else 0,
        }
    return tasks

def dummy_config(langs, tasks, architecture, n_gpus_per_node, n_slots_per_gpu):
    groups = {lang: 'all' for lang in langs}
    if architecture == 'full/langspec':
        enc_layers = [6]
        enc_sharing_groups = ['FULL']
        dec_sharing_groups = ['TGT_LANGUAGE']
    elif architecture == 'langspec/langspec':
        enc_layers = [6]
        enc_sharing_groups = ['SRC_LANGUAGE']
        dec_sharing_groups = ['TGT_LANGUAGE']
    elif architecture == 'apple':
        enc_layers = [2, 2, 2]
        enc_sharing_groups = ['SRC_LANGUAGE', 'FULL', 'TGT_LANGUAGE']
        dec_sharing_groups = ['TGT_LANGUAGE']
    config = {
        'config_config': {
            'groups': groups,
            'enc_sharing_groups': enc_sharing_groups,
            'dec_sharing_groups': dec_sharing_groups,
            # 'n_nodes'
            'n_gpus_per_node': n_gpus_per_node,
            'n_slots_per_gpu': n_slots_per_gpu,
        },
        'tasks': tasks,
        'enc_layers': enc_layers,
        'dec_layers': [6],
    }
    return config
    

@click.command()
@click.option('--n_langs', type=int, required=True)
@click.option('--sparsity', type=click.Choice(['multiparallel', 'centric', 'semi-centric']), required=True)
@click.option('--architecture', type=click.Choice(['full/langspec', 'langspec/langspec', 'apple']), required=True)
@click.option('--curriculum', is_flag=True)
@click.option('--n_gpus_per_node', type=int, default=8)
@click.option('--n_slots_per_gpu', type=int, required=True)
def main(n_langs, sparsity, architecture, curriculum, n_gpus_per_node, n_slots_per_gpu):
    name = f'n_langs={n_langs}_sparsity={sparsity}_architecture={architecture}_curriculum={curriculum}_n_gpus_per_node={n_gpus_per_node}_n_slots_per_gpu={n_slots_per_gpu}'
    name = name.replace('/', '+')
    langs = create_langs(n_langs)
    lang_pairs = sorted(complete_language_pairs(langs, sparsity))
    tasks = create_tasks(lang_pairs, curriculum)
    config = dummy_config(langs, tasks, architecture, n_gpus_per_node, n_slots_per_gpu)
    filename = f'{name}.init.yaml'
    with open(filename, 'w') as fout:
        print(f'writing into {filename}')
        yaml.safe_dump(config, fout)


if __name__ == '__main__':
    main()
