import random
from collections import defaultdict, Counter, namedtuple
from functools import lru_cache
from typing import Set, Dict, Tuple, Optional

INTER_NODE_COST = 5
INTRA_NODE_COST = 1
HOMOGENEITY_WEIGHT = 0.1
READY_TO_START_WEIGHT = 20

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


GpuSlot = namedtuple('GpuSlot', ['node', 'gpu', 'slot'])


class AssignmentOptimizer:
    def __init__(
        self,
        n_nodes: int,
        n_gpus_per_node: int,
        n_slots_per_gpu: int,
        ready_to_start: Optional[Set[Tuple[str, str]]] = None,
    ):
        self.n_nodes = n_nodes
        self.n_gpus_per_node = n_gpus_per_node
        self.n_slots_per_gpu = n_slots_per_gpu
        self.gpu_slots = list(self.make_slots(n_nodes, n_gpus_per_node, n_slots_per_gpu))
        self.ready_to_start = ready_to_start

    @staticmethod
    def make_slots(n_nodes, n_gpus_per_node, n_slots_per_gpu):
        for node in range(n_nodes):
            for gpu in range(n_gpus_per_node):
                for slot in range(n_slots_per_gpu):
                    yield GpuSlot(node, gpu, slot)

    def initial_assignment(self, lang_pairs):
        lang_pairs = list(lang_pairs)
        random.shuffle(lang_pairs)
        return {gpu_slot: lp for gpu_slot, lp in zip(self.gpu_slots, lang_pairs)}

    def cost(self, assignment: Dict[GpuSlot, Tuple[str, str]]) -> float:
        # assignment is gpu_slot -> lp
        cost_communication = self._communication_cost(assignment)
        cost_homogeneity = self._homogeneity_cost(assignment)
        cost_ready_to_start = self._ready_to_start_cost(assignment)
        return (
            cost_communication
            + (HOMOGENEITY_WEIGHT * cost_homogeneity)
            + (READY_TO_START_WEIGHT * cost_ready_to_start)
        )

    def _communication_cost(self, assignment: Dict[GpuSlot, Tuple[str, str]]) -> float:
        """ amount of communication: on how many gpus is each component? """
        component_to_gpus = defaultdict(set)
        for gpu_slot, lp in assignment.items():
            if lp is None:
                continue
            src_lang, tgt_lang = lp
            gpu = (gpu_slot.node, gpu_slot.gpu)
            components = get_components(src_lang, tgt_lang)
            for component in components:
                component_to_gpus[component].add(gpu)
        cost = 0
        for component, gpus in component_to_gpus.items():
            nodes = set(node for (node, gpu) in gpus)
            # inter-node communcation is expensive
            cost += INTER_NODE_COST * (len(nodes) - 1)
            # intra-node is cheaper than inter-node
            cost += INTRA_NODE_COST * (len(gpus) - 1)
        return cost

    def _homogeneity_cost(self, assignment: Dict[GpuSlot, Tuple[str, str]]) -> float:
        """ homogeneity of gpus: how many components are on the gpu? how many times? """
        gpus_to_components = defaultdict(Counter)
        for gpu_slot, lp in assignment.items():
            if lp is None:
                continue
            src_lang, tgt_lang = lp
            gpu = (gpu_slot.node, gpu_slot.gpu)
            components = get_components(src_lang, tgt_lang)
            for component in components:
                gpus_to_components[gpu][component] += 1
        cost = 0
        for gpu, component_counts in gpus_to_components.items():
            # more components is worse
            # more copies of the same component is (slightly) better
            for count in component_counts.values():
                cost += self.n_slots_per_gpu - count
        return cost

    def _ready_to_start_cost(self, assignment: Dict[GpuSlot, Tuple[str, str]]) -> float:
        all_are_ready = self.ready_to_start is None
        gpus_ready_to_start = {(gpu_slot.node, gpu_slot.gpu): False for gpu_slot in self.gpu_slots}
        for gpu_slot, lp in assignment.items():
            gpu = (gpu_slot.node, gpu_slot.gpu)
            lp_ready_to_start = all_are_ready or lp in self.ready_to_start
            gpus_ready_to_start[gpu] = lp_ready_to_start or gpus_ready_to_start[gpu]
        not_ready = sum(not ready_to_start for ready_to_start in gpus_ready_to_start.values())
        return not_ready

    def swap(self, slot_a: GpuSlot, slot_b: GpuSlot, assignment):
        lp_a = assignment.get(slot_a, None)
        lp_b = assignment.get(slot_b, None)
        result = dict(assignment)
        result[slot_a] = lp_b
        result[slot_b] = lp_a
        return result

    def best_swap_for(self, slot_a: GpuSlot, assignment, current_cost):
        costs = [(current_cost, slot_a)]
        for slot_b in self.gpu_slots:
            if slot_a == slot_b:
                continue
            proposal = self.swap(slot_a, slot_b, assignment)
            costs.append((self.cost(proposal), slot_b))
        costs = sorted(costs)
        best_cost, slot_b = costs[0]
        best_assignment = self.swap(slot_a, slot_b, assignment)
        return best_cost, best_assignment

    def swap_all_slots_once(self, assignment, current_cost):
        for slot_a in self.gpu_slots:
            current_cost, assignment = self.best_swap_for(slot_a, assignment, current_cost)
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
        slot_str = f'gpu {gpu_slot.node}:{gpu_slot.gpu} slot {gpu_slot.slot}'
        if lp is None:
            print(
                f'{slot_str}: UNASSIGNED'
            )
            continue
        src_lang, tgt_lang = lp
        src_group = LANGS[src_lang]
        tgt_group = LANGS[tgt_lang]
        ready = 'ready to start' if lp in ready_to_start else ''
        print(
            f'{slot_str}: {src_lang}-{tgt_lang}\t({src_group}, {tgt_group})\t{ready}'
        )


n_nodes = 2
n_gpus_per_node = 3
n_slots_per_gpu = 4
optimizer = AssignmentOptimizer(
    n_nodes=n_nodes,
    n_gpus_per_node=n_gpus_per_node,
    n_slots_per_gpu=n_slots_per_gpu,
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
best_cost, assignment = optimizer.best_swap_for(GpuSlot(0, 0, 0), initial, initial_cost)
print_assignment(assignment, ready_to_start=READY_TO_START)
print(f'assignment cost {best_cost}')

best_cost, assignment = optimizer.optimize(initial, initial_cost)
print_assignment(assignment, ready_to_start=READY_TO_START)
print(f'assignment cost {best_cost}')
