import pandas as pd
import random
from collections import defaultdict
from functools import lru_cache
from typing import Set, Dict, Tuple

# tmp
LANGS = {
    'en': 'ger',
    'de': 'ger',
    'sv': 'ger',
    'no': 'ger',
    'fi': 'ural',
    'et': 'ural',
}


# this function should be provided by the user?
@lru_cache(maxsize=100_000)
def get_components(src_lang: str, tgt_lang: str) -> Set[str]:
    return {
        f'src_lang_{src_lang}',
        f'src_group_{LANGS[src_lang]}',
        f'tgt_lang_{tgt_lang}',
        f'tgt_group_{LANGS[tgt_lang]}',
    }


def lp_distance(src_a, tgt_a, src_b, tgt_b):
    components_a = get_components(src_a, tgt_a)
    components_b = get_components(src_b, tgt_b)
    common = components_a.intersection(components_b)
    total = max(len(components_a), len(components_b))
    return (total - len(common)) / total


def cost(assignment: Dict[Tuple[str, int], Tuple[str, str]]) -> float:
    # assignment is gpu_slot -> lp?
    cost_communication = communication_cost(assignment)
    # homogeneity of gpus: how many components are on the gpu? how many times?
    cost_homogeneity = homogeneity_cost(assignment)
    print(f'communication_cost: {cost_communication}')
    print(f'homogeneity_cost: {cost_homogeneity}')
    return cost_communication + cost_homogeneity


def communication_cost(assignment: Dict[Tuple[int, int], Tuple[str, str]]) -> float:
    """ amount of communication: on how many gpus is each component? """
    component_to_gpus = defaultdict(set)
    for gpu_slot, lp in assignment.items():
        src_lang, tgt_lang = lp
        gpu, slot = gpu_slot
        components = get_components(src_lang, tgt_lang)
        for component in components:
            component_to_gpus[component].add(gpu)
    cost = 0
    for component, gpus in component_to_gpus.items():
        # TODO: intra-node should be cheaper than inter-node
        cost += len(gpus) - 1
    return cost


def homogeneity_cost(assignment: Dict[Tuple[str, int], Tuple[str, str]]) -> float:
    return 0.0


def make_slots(n_gpus, slots_per_gpu):
    for gpu in range(n_gpus):
        for slot in range(slots_per_gpu):
            yield (gpu, slot)


def swap(gpu_a, slot_a, gpu_b, slot_b, assignment):
    lp_a = assignment.get((gpu_a, slot_a), None)
    lp_b = assignment.get((gpu_b, slot_b), None)
    result = dict(assignment)
    if lp_b is not None:
        result[(gpu_a, slot_a)] = lp_b
    if lp_a is not None:
        result[(gpu_b, slot_b)] = lp_a
    return result


def best_swap_for(gpu_a, slot_a, assignment, gpu_slots, current_cost):
    costs = [(current_cost, gpu_a, slot_a)]
    for gpu_b, slot_b in gpu_slots:
        if (gpu_a, slot_a) == (gpu_b, slot_b):
            continue
        proposal = swap(gpu_a, slot_a, gpu_b, slot_b, assignment)
        costs.append((cost(proposal), gpu_b, slot_b))
    costs = sorted(costs)
    best_cost, gpu_b, slot_b = costs[0]
    print(f'best_cost {best_cost}, gpu_b {gpu_b}, slot_b {slot_b}')
    return swap(gpu_a, slot_a, gpu_b, slot_b, assignment)


n_gpus = 9
slots_per_gpu = 4
gpu_slots = list(make_slots(n_gpus, slots_per_gpu))

lang_pairs = []
for src_lang in LANGS.keys():
    for tgt_lang in LANGS.keys():
        lang_pairs.append((src_lang, tgt_lang))

random.shuffle(lang_pairs)
initial = {gpu_slot: lp for gpu_slot, lp in zip(gpu_slots, lang_pairs)}

from pprint import pprint
pprint(initial)
initial_cost = cost(initial)
print(f'initial cost {initial_cost}')
assignment = swap(0, 0, 8, 3, initial)
pprint(assignment)
print(f'assignment cost {cost(assignment)}')
assignment = best_swap_for(0, 0, initial, gpu_slots, initial_cost)
pprint(assignment)
print(f'assignment cost {cost(assignment)}')



# for src_a in LANGS.keys():
#     for tgt_a in LANGS.keys():
#         for src_b in LANGS.keys():
#             for tgt_b in LANGS.keys():
#                 print(src_a, tgt_a, src_b, tgt_b, lp_distance(src_a, tgt_a, src_b, tgt_b))
