import random
from collections import defaultdict, Counter
from functools import lru_cache
from typing import Set, Dict, Tuple, Optional

# tmp
LANGS = {
    'en': 'ger',
    'de': 'ger',
    'sv': 'ger',
    'no': 'ger',
    'fi': 'ural',
    'et': 'ural',
}

# LPs ready to start at timestep 0
READY_TO_START = {
    ('en', 'de'),
    ('de', 'en'),
    ('en', 'en'),
    ('de', 'de'),
    ('sv', 'no'),
    ('no', 'sv'),
    ('sv', 'de'),
    ('de', 'sv'),
    ('fi', 'fi'),
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


class AssignmentOptimizer:
    def __init__(
        self,
        n_gpus: int,
        slots_per_gpu: int,
        ready_to_start: Optional[Set[Tuple[str, str]]] = None,
    ):
        self.n_gpus = n_gpus
        self.slots_per_gpu = slots_per_gpu
        self.gpu_slots = list(self.make_slots(n_gpus, slots_per_gpu))
        self.ready_to_start = ready_to_start

    @staticmethod
    def make_slots(n_gpus, slots_per_gpu):
        for gpu in range(n_gpus):
            for slot in range(slots_per_gpu):
                yield (gpu, slot)

    def initial_assignment(self, lang_pairs):
        lang_pairs = list(lang_pairs)
        random.shuffle(lang_pairs)
        return {gpu_slot: lp for gpu_slot, lp in zip(self.gpu_slots, lang_pairs)}

    def cost(self, assignment: Dict[Tuple[str, int], Tuple[str, str]]) -> float:
        # assignment is gpu_slot -> lp?
        cost_communication = self._communication_cost(assignment)
        cost_homogeneity = self._homogeneity_cost(assignment)
        cost_ready_to_start = self._ready_to_start_cost(assignment)
        return (
            cost_communication
            + (0.1 * cost_homogeneity)
            + (10 * cost_ready_to_start)
        )

    def _communication_cost(self, assignment: Dict[Tuple[int, int], Tuple[str, str]]) -> float:
        """ amount of communication: on how many gpus is each component? """
        component_to_gpus = defaultdict(set)
        for gpu_slot, lp in assignment.items():
            if lp is None:
                continue
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

    def _homogeneity_cost(self, assignment: Dict[Tuple[str, int], Tuple[str, str]]) -> float:
        """ homogeneity of gpus: how many components are on the gpu? how many times? """
        gpus_to_components = defaultdict(Counter)
        for gpu_slot, lp in assignment.items():
            if lp is None:
                continue
            src_lang, tgt_lang = lp
            gpu, slot = gpu_slot
            components = get_components(src_lang, tgt_lang)
            for component in components:
                gpus_to_components[gpu][component] += 1
        cost = 0
        for gpu, component_counts in gpus_to_components.items():
            # more components is worse
            # more copies of the same component is (slightly) better
            for count in component_counts.values():
                cost += self.slots_per_gpu - count
        return cost

    def _ready_to_start_cost(self, assignment: Dict[Tuple[str, int], Tuple[str, str]]) -> float:
        if self.ready_to_start is None:
            return 0
        gpus_ready_to_start = {gpu: False for gpu in range(self.n_gpus)}
        for gpu_slot, lp in assignment.items():
            gpu, slot = gpu_slot
            lp_ready_to_start = lp in self.ready_to_start
            gpus_ready_to_start[gpu] = lp_ready_to_start or gpus_ready_to_start[gpu]
        not_ready = sum(not ready_to_start for ready_to_start in gpus_ready_to_start.values())
        return not_ready

    def swap(self, gpu_a, slot_a, gpu_b, slot_b, assignment):
        lp_a = assignment.get((gpu_a, slot_a), None)
        lp_b = assignment.get((gpu_b, slot_b), None)
        result = dict(assignment)
        result[(gpu_a, slot_a)] = lp_b
        result[(gpu_b, slot_b)] = lp_a
        return result

    def best_swap_for(self, gpu_a, slot_a, assignment, current_cost):
        costs = [(current_cost, gpu_a, slot_a)]
        for gpu_b, slot_b in self.gpu_slots:
            if (gpu_a, slot_a) == (gpu_b, slot_b):
                continue
            proposal = self.swap(gpu_a, slot_a, gpu_b, slot_b, assignment)
            costs.append((self.cost(proposal), gpu_b, slot_b))
        costs = sorted(costs)
        best_cost, gpu_b, slot_b = costs[0]
        # print(f'best_cost {best_cost}, gpu_b {gpu_b}, slot_b {slot_b}')
        best_assignment = self.swap(gpu_a, slot_a, gpu_b, slot_b, assignment)
        return best_cost, best_assignment

    def swap_all_slots_once(self, assignment, current_cost):
        for gpu_a, slot_a in self.gpu_slots:
            current_cost, assignment = self.best_swap_for(gpu_a, slot_a, assignment, current_cost)
        return current_cost, assignment

    def optimize(self, assignment, current_cost, iterations=10, patience=1):
        prev_cost = None
        stalled = 0
        for i in range(iterations):
            prev_cost = current_cost
            current_cost, assignment = self.swap_all_slots_once(assignment, current_cost)
            print(f'iteration {i} cost: {current_cost}')
            if prev_cost == current_cost:
                stalled += 1
            else:
                stalled = 0
            if stalled > patience:
                print('No improvement, finishing early')
                break
        return current_cost, assignment


def print_assignment(assignment, ready_to_start=None):
    ready_to_start = set() if ready_to_start is None else ready_to_start
    for gpu_slot, lp in sorted(assignment.items()):
        gpu, slot = gpu_slot
        if lp is None:
            print(
                f'gpu {gpu} slot {slot}: UNASSIGNED'
            )
            continue
        src_lang, tgt_lang = lp
        src_group = LANGS[src_lang]
        tgt_group = LANGS[tgt_lang]
        ready = 'ready to start' if lp in ready_to_start else ''
        print(
            f'gpu {gpu} slot {slot}: {src_lang}-{tgt_lang}\t({src_group}, {tgt_group})\t{ready}'
        )


n_gpus = 9
slots_per_gpu = 4
optimizer = AssignmentOptimizer(
    n_gpus=n_gpus,
    slots_per_gpu=slots_per_gpu,
    ready_to_start=READY_TO_START,
)

lang_pairs = []
for src_lang in LANGS.keys():
    for tgt_lang in LANGS.keys():
        if LANGS[src_lang] != LANGS[tgt_lang]:
            continue
        lang_pairs.append((src_lang, tgt_lang))

initial = optimizer.initial_assignment(lang_pairs)

print_assignment(initial, ready_to_start=READY_TO_START)
initial_cost = optimizer.cost(initial)
print(f'initial cost {initial_cost}')
assignment = optimizer.swap(0, 0, 8, 3, initial)
print_assignment(assignment, ready_to_start=READY_TO_START)
print(f'assignment cost {optimizer.cost(assignment)}')
best_cost, assignment = optimizer.best_swap_for(0, 0, initial, initial_cost)
print_assignment(assignment, ready_to_start=READY_TO_START)
print(f'assignment cost {best_cost}')

best_cost, assignment = optimizer.optimize(initial, initial_cost)
print_assignment(assignment, ready_to_start=READY_TO_START)
print(f'assignment cost {best_cost}')
