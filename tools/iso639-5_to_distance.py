import click
import pandas as pd
import yaml
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set

DATA_PATH = 'config/iso639-5_to_iso639-3.tsv'
MACROLANG_PATH = 'config/iso-639-3-macrolanguages_20200130.tab'


def get_langs(config):
    src_langs = set(config['src_vocab'].keys())
    tgt_langs = set(config['tgt_vocab'].keys())
    langs = list(sorted(src_langs.union(tgt_langs)))
    return langs


def read_lang_families() -> Dict[str, Set[str]]:
    families = dict()
    with open(DATA_PATH, 'r') as fin:
        for line in fin:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) != 3:
                # print(f'skipping invalid line: {line}')
                continue
            group = parts[0]
            langs = set(parts[2].split())
            families[group] = langs
    return families


def read_macrolanguages():
    macrolanguages = defaultdict(set)
    with open(MACROLANG_PATH, 'r') as fin:
        for line in fin:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) != 3:
                # print(f'skipping invalid line: {line}')
                continue
            macro = parts[0]
            lang = parts[1]
            macrolanguages[macro].add(lang)
    return macrolanguages


def add_macrolanguages(langs, families, macrolanguages):
    all_seen_langs = set()
    for langs_in_family in families.values():
        all_seen_langs.update(langs_in_family)
    found_mappings = dict()
    for maybe_macrolang in langs:
        if maybe_macrolang not in all_seen_langs:
            print(f'{maybe_macrolang} not in all_seen_langs')
            for lang in macrolanguages[maybe_macrolang]:
                print(f'macrolanguage {maybe_macrolang} maps to {lang}')
                found_mappings[lang] = maybe_macrolang
    for family, langs_in_family in families.items():
        for lang in set(langs_in_family):
            if lang in found_mappings:
                macrolang = found_mappings[lang]
                langs_in_family.add(macrolang)


def determine_distances(langs, families):
    counter = Counter()
    for langs_in_family in families.values():
        for lang_a in langs_in_family:
            for lang_b in langs_in_family:
                if lang_a not in langs or lang_b not in langs:
                    continue
                counter[(lang_a, lang_b)] += 1
    max_count = max(counter.values())
    df_long = pd.DataFrame.from_records([
        {
            'lang_a': lang_a,
            'lang_b': lang_b,
            'dist': (max_count - counter[(lang_a, lang_b)]) / max_count,
        }
        for lang_a, lang_b in counter.keys()
    ])
    df_wide = df_long.pivot(index='lang_a', columns='lang_b', values='dist').fillna(1)
    return df_wide


@click.command()
@click.option('--infile', type=Path, required=True)
@click.option('--outfile', type=Path, required=True)
def main(infile: Path, outfile: Path):
    with infile.open('r') as fin:
        config = yaml.safe_load(fin)
    langs = get_langs(config)
    families = read_lang_families()
    macrolanguages = read_macrolanguages()
    add_macrolanguages(langs, families, macrolanguages)
    df = determine_distances(langs, families)
    with outfile.open('w') as fout:
        df.to_csv(fout, index_label='lang')


if __name__ == '__main__':
    main()
