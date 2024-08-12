from collections import OrderedDict
from typing import Dict, Any

from mammoth.models import NMTModel
from mammoth.distributed.tasks import LocalTaskQueueManager


def explode_model(model: NMTModel, task_queue_manager: LocalTaskQueueManager) -> Dict[str, Any]:
    my_components = task_queue_manager.get_my_distributed_components()
    my_global_rank = task_queue_manager.global_rank
    state_dicts = OrderedDict()
    for component in my_components:
        if component.min_rank == my_global_rank:
            # Only the lowest ranked device saves a component
            state_dicts[component.get_name()] = component.state_dict(model)
    return state_dicts
