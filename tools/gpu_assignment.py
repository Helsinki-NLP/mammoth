import random
from collections import defaultdict, Counter, namedtuple
from functools import lru_cache
from typing import Set, Dict, Tuple, Optional, Callable, List

INTER_NODE_COST = 5
INTRA_NODE_COST = 1
HOMOGENEITY_WEIGHT = 0.1
READY_TO_START_WEIGHT = 0.5
UNASSIGNED_WEIGHT = 100
NOT_READY_TO_START = 500
VERY_BAD = 99999999


# this function should be provided by the user?
def make_get_components_func(group_mapping: Dict[str, str]):
    @lru_cache(maxsize=100_000)
    def get_components(src_lang: str, tgt_lang: str) -> Set[str]:
        return {
            f'src_lang_{src_lang}',
            f'src_group_{group_mapping[src_lang]}',
            f'tgt_lang_{tgt_lang}',
            f'tgt_group_{group_mapping[tgt_lang]}',
        }
    return get_components


GpuSlot = namedtuple('GpuSlot', ['node', 'gpu', 'slot'])


class AssignmentOptimizer:
    def __init__(
        self,
        n_nodes: int,
        n_gpus_per_node: int,
        n_slots_per_gpu: int,
        get_components: Callable[[str, str], Set[str]],
        ready_to_start: Optional[Set[Tuple[str, str]]] = None,
    ):
        self.n_nodes = n_nodes
        self.n_gpus_per_node = n_gpus_per_node
        self.n_slots_per_gpu = n_slots_per_gpu
        self.gpu_slots = list(self.make_slots(n_nodes, n_gpus_per_node, n_slots_per_gpu))
        self.get_components = get_components
        self.ready_to_start = ready_to_start

    @staticmethod
    def make_slots(n_nodes, n_gpus_per_node, n_slots_per_gpu):
        for node in range(n_nodes):
            for gpu in range(n_gpus_per_node):
                for slot in range(n_slots_per_gpu):
                    yield GpuSlot(node, gpu, slot)

    def initial_assignment(self, lang_pairs: List[Tuple[str, str]]):
        lang_pairs = list(lang_pairs)
        random.shuffle(lang_pairs)
        if len(lang_pairs) > len(self.gpu_slots):
            raise Exception(f'More lang pairs {len(lang_pairs)} than gpu slots {len(self.gpu_slots)}')
        if len(self.gpu_slots) > len(lang_pairs):
            # Add empty slots
            lang_pairs.extend([None] * (len(self.gpu_slots) - len(lang_pairs)))
        return {gpu_slot: lp for gpu_slot, lp in zip(self.gpu_slots, lang_pairs)}

    def cost(self, assignment: Dict[GpuSlot, Tuple[str, str]]) -> float:
        # assignment is gpu_slot -> lp
        cost_communication = self._communication_cost(assignment)
        cost_homogeneity = self._homogeneity_cost(assignment)
        cost_ready_to_start = self._ready_to_start_cost(assignment)
        cost_unassigned = self._unassigned_cost(assignment)
        return (
            cost_communication
            + (HOMOGENEITY_WEIGHT * cost_homogeneity)
            + (READY_TO_START_WEIGHT * cost_ready_to_start)
            + (UNASSIGNED_WEIGHT * cost_unassigned)
        )

    def _communication_cost(self, assignment: Dict[GpuSlot, Tuple[str, str]]) -> float:
        """ amount of communication: on how many gpus is each component? """
        component_to_gpus = defaultdict(set)
        for gpu_slot, lp in assignment.items():
            if lp is None:
                continue
            src_lang, tgt_lang = lp
            gpu = (gpu_slot.node, gpu_slot.gpu)
            components = self.get_components(src_lang, tgt_lang)
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
            components = self.get_components(src_lang, tgt_lang)
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
        """
        When using a curriculum, all GPUs should contain at least one task that
        can start at timestep 0.
        """
        if self.ready_to_start:
            return 0
        return self._spread_evenly(
            assignment,
            criterion=lambda lp: lp in self.ready_to_start,
            extra_empty_penalty=NOT_READY_TO_START,
        )

    def _unassigned_cost(self, assignment: Dict[GpuSlot, Tuple[str, str]]) -> float:
        """ Penalize gpus with many unassigned slots if other gpus are full """
        return self._spread_evenly(
            assignment,
            criterion=lambda lp: lp is not None,
            extra_empty_penalty=VERY_BAD,
        )

    def _spread_evenly(
        self,
        assignment: Dict[GpuSlot, Tuple[str, str]],
        criterion: Callable[[str, str], bool],
        extra_empty_penalty,
    ) -> float:
        filled = Counter()
        for gpu_slot, lp in assignment.items():
            gpu = (gpu_slot.node, gpu_slot.gpu)
            if criterion(lp):
                filled[gpu] += 1
            else:
                # also populate gpus with zero count
                filled[gpu] += 0
        min_filled = min(filled.values())
        max_filled = max(filled.values())
        penalty = extra_empty_penalty if min_filled == 0 else 0
        return max_filled - min_filled + penalty

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
        print(f'initial cost: {current_cost}')
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


def print_assignment(assignment, group_mapping, ready_to_start=None):
    ready_to_start = set() if ready_to_start is None else ready_to_start
    for gpu_slot, lp in sorted(assignment.items()):
        slot_str = f'gpu {gpu_slot.node}:{gpu_slot.gpu} slot {gpu_slot.slot}'
        if lp is None:
            print(
                f'{slot_str}: UNASSIGNED'
            )
            continue
        src_lang, tgt_lang = lp
        src_group = group_mapping[src_lang]
        tgt_group = group_mapping[tgt_lang]
        ready = 'ready to start' if lp in ready_to_start else ''
        print(
            f'{slot_str}: {src_lang}-{tgt_lang}\t({src_group}, {tgt_group})\t{ready}'
        )


def optimize_gpu_assignment(
    n_nodes: int,
    n_gpus_per_node: int,
    n_slots_per_gpu: int,
    lang_pairs: List[Tuple[str, str]],
    lang_to_group_mapping: Dict[str, str],
    lps_ready_to_start: Optional[Set[Tuple[str, str]]],
):
    optimizer = AssignmentOptimizer(
        n_nodes=n_nodes,
        n_gpus_per_node=n_gpus_per_node,
        n_slots_per_gpu=n_slots_per_gpu,
        get_components=make_get_components_func(lang_to_group_mapping),
        ready_to_start=lps_ready_to_start,
    )

    initial = optimizer.initial_assignment(lang_pairs)
    initial_cost = optimizer.cost(initial)
    best_cost, assignment = optimizer.optimize(initial, initial_cost)
    print_assignment(assignment, lang_to_group_mapping, ready_to_start=lps_ready_to_start)
    print(f'assignment cost {best_cost}')
    return assignment


def example():
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

    lang_pairs = []
    for src_lang in LANGS.keys():
        for tgt_lang in LANGS.keys():
            if LANGS[src_lang] != LANGS[tgt_lang]:
                continue
            lang_pairs.append((src_lang, tgt_lang))

    optimize_gpu_assignment(
        n_nodes=2,
        n_gpus_per_node=3,
        n_slots_per_gpu=4,
        lang_pairs=lang_pairs,
        lang_to_group_mapping=LANGS,
        lps_ready_to_start=READY_TO_START,
    )


if __name__ == '__main__':
    print('#### Showing a dummy example run')
    example()
