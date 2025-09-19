import click
import re
import yaml
from pathlib import Path


@click.command()
@click.option('--config', 'config_path', type=Path, help='config file')
@click.option(
    '--match',
    type=str,
    default=None,
    help='Regex that task ids must match. Default: include all tasks',
)
@click.option(
    '--src',
    type=str,
    default=None,
    help='Template for source file paths. Use varibles src_lang, tgt_lang, and task_id.',
)
@click.option(
    '--output',
    type=str,
    default=None,
    help='Template for translation output file paths. Use varibles src_lang, tgt_lang, and task_id.',
)
@click.option(
    '--flag',
    is_flag=True,
    help='Prefix output with "--task_id". Implied by --src and --output.'
)
def main(config_path, match, src, output, flag):
    if src is not None or output is not None:
        flag = True
    if match:
        match = re.compile(match)
    with config_path.open('r') as fin:
        config = yaml.safe_load(fin)
    for key, task in config['tasks'].items():
        if match and not match.match(key):
            continue
        src_lang, tgt_lang = task['src_tgt'].split('-')

        result = []
        if flag:
            result.append('--task_id')
        result.append(key)
        if src:
            task_src = src.format(src_lang=src_lang, tgt_lang=tgt_lang, task_id=key)
            result.extend(['--src', task_src])
        if output:
            task_out = output.format(src_lang=src_lang, tgt_lang=tgt_lang, task_id=key)
            result.extend(['--output', task_out])
        print(' '.join(result))


if __name__ == '__main__':
    main()
