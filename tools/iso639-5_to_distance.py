import click
import pandas as pd
import yaml
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set

ISO_DATA_PATH = 'config/iso639-5_to_iso639-3.tsv'
GLOTTOLOG_DATA_PATH = 'config/languoid.csv'
MACROLANG_PATH = 'config/iso-639-3-macrolanguages_20200130.tab'


def get_langs(config):
    src_langs = set(config['src_vocab'].keys())
    tgt_langs = set(config['tgt_vocab'].keys())
    langs = list(sorted(src_langs.union(tgt_langs)))
    return langs


def read_iso_lang_families() -> Dict[str, Set[str]]:
    """ Returns: family -> set of lang codes """
    families = dict()
    with open(ISO_DATA_PATH, 'r') as fin:
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


def read_glottolog():
    parents = dict()
    iso_to_glottolog = dict()
    df = pd.read_csv(GLOTTOLOG_DATA_PATH)
    for tpl in df.itertuples():
        parents[tpl.id] = tpl.parent_id
        iso_to_glottolog[tpl.iso639P3code] = tpl.id
    return parents, iso_to_glottolog


def transitive_closure(initial, graph):
    found = set([initial])
    if initial in graph:
        parent = graph[initial]
        if parent and not pd.isna(parent):
            found.update(transitive_closure(parent, graph))
    return found


def read_glottolog_lang_families(langs, macrolanguages) -> Dict[str, Set[str]]:
    """Read language familes from Glottolog data.
    langs: language codes in the corpus.
    macrolanguages: macrolanguage to set of language code mapping from iso-639-3
    Returns: glottolog-family -> set of lang codes
    """
    parents, iso_to_glottolog = read_glottolog()
    all_seen_langs = set(iso_to_glottolog.keys())
    found_mappings = find_macrolanguages(langs, all_seen_langs, macrolanguages)
    inverted_mappings = defaultdict(set)
    for k, v in found_mappings.items():
        inverted_mappings[v].add(k)

    result = defaultdict(set)
    for lang in langs:
        glottolog_ids = set()
        glottolog_id = iso_to_glottolog.get(lang, None)
        if glottolog_id is not None:
            glottolog_ids.add(glottolog_id)
        else:
            for sublang in inverted_mappings[lang]:
                glottolog_id = iso_to_glottolog.get(sublang, None)
                if glottolog_id:
                    glottolog_ids.add(glottolog_id)
        for glottolog_id in glottolog_ids:
            closure = transitive_closure(glottolog_id, parents)
            for family in closure:
                result[family].add(lang)
    return result


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


def find_macrolanguages(langs, all_seen_langs, macrolanguages):
    found_mappings = dict()
    for maybe_macrolang in langs:
        if maybe_macrolang not in all_seen_langs:
            print(f'{maybe_macrolang} not in all_seen_langs')
            for lang in macrolanguages[maybe_macrolang]:
                print(f'macrolanguage {maybe_macrolang} maps to {lang}')
                found_mappings[lang] = maybe_macrolang
    return found_mappings


def add_macrolanguages(langs, families, macrolanguages):
    all_seen_langs = set()
    for langs_in_family in families.values():
        all_seen_langs.update(langs_in_family)
    found_mappings = find_macrolanguages(langs, all_seen_langs, macrolanguages)
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
    maxes = Counter()
    for (lang_a, lang_b), val in counter.items():
        maxes[lang_a] = max(maxes[lang_a], val)
        maxes[lang_b] = max(maxes[lang_b], val)

    def dist(lang_a, lang_b, counter, maxes):
        max_count = max(maxes[lang_a], maxes[lang_b])
        return (max_count - counter[(lang_a, lang_b)]) / max_count

    df_long = pd.DataFrame.from_records([
        {
            'lang_a': lang_a,
            'lang_b': lang_b,
            'dist': dist(lang_a, lang_b, counter, maxes),
        }
        for lang_a, lang_b in counter.keys()
    ])
    df_wide = df_long.pivot(index='lang_a', columns='lang_b', values='dist').fillna(1)
    return df_wide


@click.command()
@click.option('--infile', type=Path, required=True)
@click.option('--outfile', type=Path, required=True)
@click.option('--taxonomy', type=str, default='glottolog')
def main(infile: Path, outfile: Path, taxonomy: str):
    with infile.open('r') as fin:
        config = yaml.safe_load(fin)
    langs = get_langs(config)
    macrolanguages = read_macrolanguages()
    if taxonomy == 'iso':
        families = read_iso_lang_families()
    elif taxonomy == 'glottolog':
        families = read_glottolog_lang_families(langs, macrolanguages)
    else:
        raise Exception(f'Unknown taxonomy "{taxonomy}"')
    add_macrolanguages(langs, families, macrolanguages)
    df = determine_distances(langs, families)
    with outfile.open('w') as fout:
        df.to_csv(fout, index_label='lang')


if __name__ == '__main__':
    main()
