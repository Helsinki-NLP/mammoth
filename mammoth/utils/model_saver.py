import os
import torch
import torch.distributed
import torch.nn as nn
from collections import OrderedDict, deque
from glob import glob
from typing import Dict, Any, Tuple

from mammoth.distributed.tasks import LocalTaskQueueManager
from mammoth.model_builder import build_model
from mammoth.models import NMTModel
from mammoth.utils.logging import logger
from mammoth.utils.misc import use_gpu
from mammoth.utils.parse import ArgumentParser


def build_model_saver(model_opts, opts, model, vocabs_dict, optim, task_queue_manager):
    # _check_save_model_path
    save_model_path = os.path.abspath(opts.save_model)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_saver = ModelSaver(
        opts.save_model,
        model,
        model_opts,
        vocabs_dict,
        optim,
        opts.keep_checkpoint,
        task_queue_manager,
    )
    return model_saver


def load_frame_checkpoint(checkpoint_path):
    """
    Load only the frame checkpoint from `checkpoint_path` if any else return `None`.

    This function is intended to be called before the fork:
    the model itself has not yet been constructed, so we don't want to load its parameters.
    We need the vocabs and data loader state from the frame.
    """
    checkpoint = None
    if checkpoint_path:
        if not checkpoint_path.endswith(".pt"):
            frames = glob(os.path.join(checkpoint_path + "*frame*pt"))
            frames.sort(key=lambda s: int(s.split("step_")[-1].split("_frame")[0]))
            checkpoint_path = frames[-1]
        logger.info("Loading frame checkpoint from %s" % checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
    return checkpoint, checkpoint_path


def explode_model(
    model: NMTModel,
    optim,
    task_queue_manager: LocalTaskQueueManager,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Splits the model into distributed components and retrieves the state dict of each."""
    my_components = task_queue_manager.get_my_distributed_components()
    my_global_rank = task_queue_manager.global_rank
    state_dicts = OrderedDict()
    optim_state_dicts = OrderedDict()
    for component in my_components:
        name = component.get_name()
        if component.min_rank == my_global_rank:
            # Only the lowest ranked device saves a component
            state_dicts[name] = component.state_dict(model)
            # The optimizer parameters are distributed the same way as the components
            # Not all components have trainable (unfrozen) parameters, though
            if name in optim.suboptimizers:
                optim_state_dicts[name] = optim.suboptimizers[name].state_dict()
    return state_dicts, optim_state_dicts


def load_parameters_from_checkpoint(
    frame_checkpoint_path,
    model,
    optim,
    task_queue_manager,
    reset_optim=False,
    yes_i_messed_with_the_checkpoint=False,
):
    """
    Splits the model into distributed components
    and restores the state dict of each component from a checkpoint file.
    """
    if not frame_checkpoint_path:
        return
    checkpoint_prefix = frame_checkpoint_path.removesuffix("_frame.pt")

    my_components = task_queue_manager.get_my_distributed_components()
    all_ok = True
    missing_keys_summary = []
    unexpected_keys_summary = []

    for component in my_components:
        name = component.get_name()
        checkpoint_path = f"{checkpoint_prefix}_{name}.pt"
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(
                checkpoint_path,
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
            incompatible_keys = component.load_state_dict(
                model=model, state_dict=state_dict
            )
            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                logger.info(f"Module {name} incompatible keys: {incompatible_keys}")
                if incompatible_keys.missing_keys:
                    missing_keys_summary.extend([f"{name}.{key}" for key in incompatible_keys.missing_keys])
                if incompatible_keys.unexpected_keys:
                    unexpected_keys_summary.extend([f"{name}.{key}" for key in incompatible_keys.unexpected_keys])
                all_ok = False
        else:
            logger.warning(
                f"Could not find model checkpoint file {checkpoint_path}. Affected parameters are reinitialized."
            )
            missing_keys_summary.append(f"{checkpoint_path} (entire file missing)")
            all_ok = False

        if not reset_optim:
            optimizer_path = f"{checkpoint_prefix}_{name}_optim.pt"
            if os.path.isfile(optimizer_path):
                # The optimizer parameters are distributed the same way as the components
                optim_state_dict = torch.load(
                    optimizer_path,
                    map_location=lambda storage, loc: storage,
                    weights_only=False,
                )
                incompatible_keys = optim.suboptimizers[name].load_state_dict(
                    optim_state_dict
                )
                if incompatible_keys and (
                    incompatible_keys.missing_keys or incompatible_keys.unexpected_keys
                ):
                    logger.info(f"Optim {name} incompatible keys: {incompatible_keys}")
                    if incompatible_keys.missing_keys:
                        missing_keys_summary.extend([f"optim_{name}.{key}" for key in incompatible_keys.missing_keys])
                    if incompatible_keys.unexpected_keys:
                        unexpected_keys_summary.extend([f"optim_{name}.{key}" for key in incompatible_keys.unexpected_keys])
                    all_ok = False
            else:
                logger.warning(
                    f"Could not find optim checkpoint file {optimizer_path}. Affected parameters are reinitialized."
                )
                missing_keys_summary.append(f"{optimizer_path} (entire file missing)")
                all_ok = False
    if all_ok:
        if reset_optim:
            logger.info(f"All modules restored from checkpoint {checkpoint_prefix}")
            if optim is not None:
                logger.info("Optimizer was reset")
        else:
            logger.info(
                f"All modules and optimizer restored from checkpoint {checkpoint_prefix}"
            )
    else:
        if yes_i_messed_with_the_checkpoint:
            logger.warning(
                "Proceeding with a partial checkpoint due to --yes_i_messed_with_the_checkpoint"
            )
        else:
            error_msg = "Some parameters are missing from the checkpoint."
            if missing_keys_summary:
                error_msg += f"\n\nMissing parameters ({len(missing_keys_summary)}):"
                for key in missing_keys_summary[:10]:  # Show first 10 to avoid overwhelming
                    error_msg += f"\n  - {key}"
                if len(missing_keys_summary) > 10:
                    error_msg += f"\n  ... and {len(missing_keys_summary) - 10} more"
            if unexpected_keys_summary:
                error_msg += f"\n\nUnexpected parameters ({len(unexpected_keys_summary)}):"
                for key in unexpected_keys_summary[:10]:  # Show first 10 to avoid overwhelming
                    error_msg += f"\n  - {key}"
                if len(unexpected_keys_summary) > 10:
                    error_msg += f"\n  ... and {len(unexpected_keys_summary) - 10} more"
            raise Exception(error_msg)


def load_model_for_translation(opts, task_queue_manager, task=None, model_path=None):
    if task is None:
        raise ValueError("Must set task")
    if model_path is None:
        model_path = opts.models[0]

        # Load only the frame
    frame, frame_checkpoint_path = load_frame_checkpoint(
        checkpoint_path=opts.train_from
    )

    vocabs_dict = {
        "src": frame["vocab"].get(("src", task.src_lang)),
        "tgt": frame["vocab"].get(("tgt", task.tgt_lang)),
    }

    model_opts = ArgumentParser.checkpoint_model_opts(frame["opts"])

    model = build_model(
        model_opts,
        opts,
        vocabs_dict,
        task_queue_manager,
        single_task=task.corpus_id,
    )

    load_parameters_from_checkpoint(
        frame_checkpoint_path,
        model,
        optim=None,
        task_queue_manager=task_queue_manager,
        reset_optim=True,
        yes_i_messed_with_the_checkpoint=opts.yes_i_messed_with_the_checkpoint,
    )

    device = torch.device("cuda" if use_gpu(opts) else "cpu")
    model.to(device)
    model.eval()

    return vocabs_dict, model, model_opts
class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(
        self,
        base_path,
        model,
        model_opts,
        vocabs_dict,
        optim,
        keep_checkpoint=-1,
        task_queue_manager=None,
    ):
        self.base_path = base_path
        self.model = model
        self.model_opts = model_opts
        self.vocabs_dict = vocabs_dict
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)
        assert task_queue_manager is not None
        self.task_queue_manager = task_queue_manager

    def save(self, step, data_state, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        save_model = self.model
        if moving_average:
            model_params_data = []
            for avg, param in zip(moving_average, save_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        chkpt_names = self._save(step, save_model, data_state, self.task_queue_manager)
        self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data, save_model.parameters()):
                param.data = param_data

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_names)

    def _save(self, step, save_model, data_state, task_queue_manager):
        """Save a resumable checkpoint.

        Args:
            step (int): step number
            save_model (nn.Module): torch model to save
            data_state (dict): data streaming info
            task_queue_manager: distributed structure of modules

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def __init__(
        self,
        base_path,
        model,
        model_opts,
        vocabs_dict,
        optim,
        keep_checkpoint=-1,
        task_queue_manager=None,
    ):
        super().__init__(
            base_path,
            model,
            model_opts,
            vocabs_dict,
            optim,
            keep_checkpoint,
            task_queue_manager,
        )

        # Metric-based checkpoint tracking
        self.save_strategy = model_opts.save_strategy if hasattr(model_opts, 'save_strategy') else 'steps'
        self.metric_for_best_model = (
            model_opts.metric_for_best_model if hasattr(model_opts, 'metric_for_best_model') else 'ppl'
        )
        self.greater_is_better = (
            model_opts.greater_is_better if hasattr(model_opts, 'greater_is_better') else None
        )

        # Auto-infer greater_is_better if not specified
        if self.greater_is_better is None:
            if self.metric_for_best_model in ['ppl', 'perplexity', 'loss']:
                self.greater_is_better = False
            elif self.metric_for_best_model in ['accuracy', 'bleu']:
                self.greater_is_better = True
            else:
                logger.warning(
                    f"Cannot auto-infer greater_is_better for metric '{self.metric_for_best_model}'. "
                    f"Defaulting to False. Please set --greater_is_better explicitly."
                )
                self.greater_is_better = False

        # Best checkpoint tracking
        self.best_metric_value = float('-inf') if self.greater_is_better else float('inf')
        self.best_checkpoint_step = None
        self.best_checkpoint_files = None
        self.checkpoint_metadata = {}  # step -> {metric_name: value, ...}

        # Validation: warn if using metric-based strategy without proper setup
        if self.save_strategy in ['best_only', 'best_and_last', 'best_n']:
            logger.info(
                f"Using metric-based checkpoint strategy '{self.save_strategy}'. "
                f"Checkpoints will be saved at validation time (--valid_steps), "
                f"not at --save_checkpoint_steps."
            )

    def _is_better_metric(self, current, best):
        """Compare metric values based on greater_is_better setting.

        This is the core comparison logic for determining if a new checkpoint
        is better than the previous best. Simple but crucial!

        Args:
            current: Current metric value
            best: Best metric value seen so far

        Returns:
            bool: True if current is better than best
        """
        if self.greater_is_better:
            return current > best
        else:
            return current < best

    def _extract_metric_value(self, valid_stats):
        """Extract the configured metric value from validation statistics.

        This method knows how to pull different metrics from the validation
        statistics object. It handles special cases like perplexity and
        accuracy which have dedicated methods, as well as custom metrics
        stored in the validation_metrics dictionary.

        Args:
            valid_stats: Statistics object with validation metrics

        Returns:
            float: The metric value

        Raises:
            ValueError: If the configured metric is not found
        """
        metric_name = self.metric_for_best_model

        if metric_name in ['ppl', 'perplexity']:
            return valid_stats.ppl()
        elif metric_name == 'accuracy':
            return valid_stats.accuracy()
        elif metric_name == 'bleu' and hasattr(valid_stats, 'validation_metrics') and 'bleu' in valid_stats.validation_metrics:
            return valid_stats.validation_metrics['bleu']
        elif hasattr(valid_stats, 'validation_metrics') and metric_name in valid_stats.validation_metrics:
            return valid_stats.validation_metrics[metric_name]
        else:
            available = ['ppl', 'accuracy']
            if hasattr(valid_stats, 'validation_metrics'):
                available.extend(list(valid_stats.validation_metrics.keys()))
            raise ValueError(
                f"Metric '{metric_name}' not found in validation statistics. "
                f"Available: {available}"
            )

    def _get_all_metrics(self, valid_stats):
        """Extract all available metrics from validation statistics.

        This collects all metrics into a single dictionary for metadata tracking.
        We store all metrics (not just the one we're optimizing for) so users
        can see the full picture when reviewing checkpoint history.

        Args:
            valid_stats: Statistics object with validation metrics

        Returns:
            dict: Dictionary of metric_name -> value
        """
        ppl_value = valid_stats.ppl()
        metrics = {
            'perplexity': ppl_value,
            'ppl': ppl_value,  # Alias for backward compatibility
            'loss': valid_stats.loss,
            'cross_entropy': valid_stats.xent()
        }
        # Add accuracy if available (n_correct must be tracked)
        if hasattr(valid_stats, 'n_correct') and valid_stats.n_correct is not None:
            metrics['accuracy'] = valid_stats.accuracy()
        # Add any custom validation metrics
        if hasattr(valid_stats, 'validation_metrics'):
            metrics.update(valid_stats.validation_metrics)
        return metrics

    def _rm_checkpoint_by_step(self, step):
        """Remove all checkpoint files associated with a given step.

        When rotating checkpoints, we need to clean up all the files for a
        given training step. In MAMMOTH's distributed setup, each step produces
        multiple files (one per component + optimizers), so we use glob pattern
        matching to find and delete them all.

        Args:
            step: Training step number whose checkpoints should be deleted
        """
        # Find checkpoint files matching this step
        checkpoint_pattern = f"*_step_{step}_*.pt"
        checkpoint_dir = os.path.dirname(self.base_path)

        files_to_delete = glob(os.path.join(checkpoint_dir, checkpoint_pattern))

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                logger.debug(f"Deleted checkpoint file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

    def _write_checkpoint_metadata(self):
        """Write checkpoint metadata to JSON file.

        This creates a human-readable summary of all checkpoints and which one
        is best. Super useful for inspecting training runs and finding the best
        checkpoint to use for inference or fine-tuning.

        The JSON file includes:
        - Best checkpoint info (step, metric value, all metrics at that step)
        - All checkpoint history with their metrics
        - Configuration settings used
        """
        import json

        metadata_file = f"{self.base_path}_checkpoint_metadata.json"

        metadata = {
            "best_checkpoint": {
                "step": self.best_checkpoint_step,
                "metric_name": self.metric_for_best_model,
                "metric_value": float(self.best_metric_value) if self.best_checkpoint_step else None,
                "all_metrics": self.checkpoint_metadata.get(self.best_checkpoint_step, {}) if self.best_checkpoint_step else {}
            },
            "all_checkpoints": {
                str(step): {k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in metrics.items()}
                for step, metrics in self.checkpoint_metadata.items()
            },
            "config": {
                "save_strategy": self.save_strategy,
                "keep_checkpoint": self.keep_checkpoint,
                "metric_for_best_model": self.metric_for_best_model,
                "greater_is_better": self.greater_is_better
            }
        }

        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Wrote checkpoint metadata to {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to write checkpoint metadata: {e}")

    def _rotate_checkpoints_by_metric(self):
        """Rotate checkpoints based on the configured save_strategy.

        This is where the magic happens for metric-based checkpoint management!
        Different strategies for what to keep:

        - best_only: Keep only the single best checkpoint
        - best_and_last: Keep the best + N most recent (N = keep_checkpoint)
        - best_n: Keep top N checkpoints by metric value (N = keep_checkpoint)

        The rotation only happens if keep_checkpoint > 0. If keep_checkpoint is -1,
        we keep all checkpoints (no rotation).
        """
        if self.keep_checkpoint <= 0:
            return  # No rotation needed (keep all checkpoints)

        all_steps = sorted(self.checkpoint_metadata.keys())

        if self.save_strategy == 'best_only':
            # Keep only the best checkpoint
            steps_to_keep = {self.best_checkpoint_step} if self.best_checkpoint_step else set()

        elif self.save_strategy == 'best_and_last':
            # Keep best + N most recent
            steps_to_keep = {self.best_checkpoint_step} if self.best_checkpoint_step else set()
            recent_steps = all_steps[-self.keep_checkpoint:]
            steps_to_keep.update(recent_steps)

        elif self.save_strategy == 'best_n':
            # Keep top N by metric value
            sorted_by_metric = sorted(
                all_steps,
                key=lambda s: self.checkpoint_metadata[s].get(
                    self.metric_for_best_model,
                    float('-inf') if self.greater_is_better else float('inf')
                ),
                reverse=self.greater_is_better
            )
            steps_to_keep = set(sorted_by_metric[:self.keep_checkpoint])

        else:
            return  # Unknown strategy, don't delete anything

        # Delete checkpoints not in steps_to_keep
        for step in all_steps:
            if step not in steps_to_keep:
                metric_value = self.checkpoint_metadata[step].get(self.metric_for_best_model, 'N/A')
                logger.info(
                    f"Deleting checkpoint at step {step} "
                    f"({self.metric_for_best_model}={metric_value})"
                )
                self._rm_checkpoint_by_step(step)
                del self.checkpoint_metadata[step]

    def save_with_metric(self, step, data_state, valid_stats, moving_average=None):
        """Save checkpoint with validation metric tracking.

        This is the main entry point that replaces the regular save() method
        when using metric-based checkpoint management. It:
        1. Extracts the metric value from validation stats
        2. Saves the checkpoint (via _save)
        3. Determines if this is a new best
        4. Updates best checkpoint tracking
        5. Rotates old checkpoints based on the strategy
        6. Writes metadata file

        Args:
            step: Current training step
            data_state: Data iterator state
            valid_stats: Statistics object with validation metrics
            moving_average: Optional moving average model state

        Returns:
            bool: True if this is a new best checkpoint
        """
        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return False

        # Extract metric value from valid_stats
        try:
            metric_value = self._extract_metric_value(valid_stats)
        except ValueError as e:
            logger.warning(f"Could not extract metric: {e}. Falling back to regular save.")
            self.save(step, data_state, moving_average)
            return False

        # Store metadata for this checkpoint
        self.checkpoint_metadata[step] = self._get_all_metrics(valid_stats)

        # Determine if this is a new best checkpoint
        is_new_best = self._is_better_metric(metric_value, self.best_metric_value)

        # Handle moving average temporarily if provided
        save_model = self.model
        model_params_data = []
        if moving_average:
            for avg, param in zip(moving_average, save_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        # Perform the actual checkpoint save
        checkpoint_files = self._save(step, save_model, data_state, self.task_queue_manager)
        self.last_saved_step = step

        # Restore original parameters if moving average was used
        if moving_average:
            for param_data, param in zip(model_params_data, save_model.parameters()):
                param.data = param_data

        # Update best checkpoint tracking
        if is_new_best:
            old_best_step = self.best_checkpoint_step
            self.best_metric_value = metric_value
            self.best_checkpoint_step = step
            self.best_checkpoint_files = checkpoint_files

            logger.info(
                f"New best checkpoint at step {step} with "
                f"{self.metric_for_best_model}={metric_value:.4f}"
            )
            if old_best_step is not None:
                logger.info(f"Previous best was step {old_best_step}")

        # Handle checkpoint rotation based on strategy
        if self.save_strategy != 'steps':  # Metric-based strategies
            self._rotate_checkpoints_by_metric()
        elif self.keep_checkpoint > 0:  # Original FIFO behavior
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(checkpoint_files)

        # Write checkpoint metadata file
        self._write_checkpoint_metadata()

        return is_new_best

    def _save(self, step, model, data_state, task_queue_manager):
        model = model.module if isinstance(model, nn.DataParallel) else model
        device_context = task_queue_manager.device_context

        tmp_checkpoint_paths = []

        module_state_dicts, optim_state_dicts = explode_model(
            model, self.optim, task_queue_manager
        )

        # The master device stores the frame
        if device_context.is_master():
            module_state_dicts["frame"] = {
                "vocab": self.vocabs_dict,
                "opts": self.model_opts,
                "global_training_step": self.optim.global_training_step,
            }

        # In a distributed context, aggregate all data states for corpus restoration
        if device_context.is_distributed():
            data_states = [None for _ in range(device_context.world_size)]
            torch.distributed.all_gather_object(data_states, data_state)
            data_state = {k: v for state in data_states for k, v in state.items()}
        if device_context.is_master():
            module_state_dicts["frame"]["data_state"] = data_state

        # Ensure the directory for checkpoint files exists
        checkpoint_dir = os.path.dirname(self.base_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        for key, state_dict in module_state_dicts.items():
            # The exploded state_dicts across different devices only contain one copy of each module:
            # on the lowest ranked device having that module.
            # There is no race condition.
            checkpoint_path = f"{self.base_path}_step_{step}_{key}.pt"
            optimizer_path = f"{self.base_path}_step_{step}_{key}_optim.pt"
            if os.path.isfile(checkpoint_path):
                logger.debug(
                    "{} - not saving {} as it is already present".format(
                        device_context.id, checkpoint_path
                    )
                )
            else:
                if key != "frame" and key in optim_state_dicts:
                    logger.info(
                        f"Saving module checkpoint {checkpoint_path} and optimizer {optimizer_path}"
                    )
                    torch.save(optim_state_dicts[key], optimizer_path)
                    tmp_checkpoint_paths.append(optimizer_path)
                else:
                    logger.info(
                        f"Saving module checkpoint {checkpoint_path} (no optimizer to save)"
                    )
                torch.save(state_dict, checkpoint_path)
                tmp_checkpoint_paths.append(checkpoint_path)

        return tmp_checkpoint_paths

    def _rm_checkpoint(self, names):
        for name in names:
            if os.path.exists(name):
                try:
                    os.remove(name)
                except BaseException:
                    logger.warning(f"Failed to delete {name}")
