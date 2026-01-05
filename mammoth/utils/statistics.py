""" Statistics calculation utility """
import time
import math
import sys
import warnings

from collections import Counter
from torch.linalg import norm

from mammoth.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=None):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

        # Tensor versions (GPU) - for accumulation during training
        # These avoid expensive GPU->CPU synchronization
        self.loss_tensor = None
        self.n_words_tensor = None
        self.n_correct_tensor = None
        self.n_src_words_tensor = None

        # losses per task
        self.loss_per_task = Counter()
        # Tensor version for per-task losses (Megatron-style: accumulate on GPU, sync only when reporting)
        self.loss_per_task_tensor = {}

        # parameter-level statistics
        self.magnitude_denom = 0
        self.param_magnitudes = Counter()
        self.grad_magnitudes = Counter()

        # Tensor versions (GPU) - for accumulation without GPU->CPU sync
        # Following Megatron's approach: accumulate on GPU, sync only when reporting
        self.param_magnitudes_tensor = {}
        self.grad_magnitudes_tensor = {}

        # validation metrics
        self.validation_metrics = {}

    def materialize_scalars(self):
        """
        Synchronize tensor values to CPU scalars.
        Only call this when you need to report/save statistics.
        This method performs GPU->CPU synchronization, so it should be
        called sparingly (e.g., every report_every steps, not every batch).
        """
        if self.loss_tensor is not None:
            self.loss = self.loss_tensor.item()
            self.loss_tensor = None  # Free the tensor
        if self.n_words_tensor is not None:
            self.n_words = self.n_words_tensor.item()
            self.n_words_tensor = None  # Free the tensor
        if self.n_correct_tensor is not None:
            self.n_correct = self.n_correct_tensor.item()
            self.n_correct_tensor = None
        if self.n_src_words_tensor is not None:
            self.n_src_words = self.n_src_words_tensor.item()
            self.n_src_words_tensor = None

        # Materialize per-task loss tensors (Megatron-style)
        # Only sync when actually reporting, not during accumulation
        if self.loss_per_task_tensor:
            for task_key, loss_tensor in self.loss_per_task_tensor.items():
                self.loss_per_task[task_key] += loss_tensor.item()
            self.loss_per_task_tensor.clear()

        # Materialize parameter/gradient magnitude tensors (Megatron-style)
        # Only sync when actually reporting, not during accumulation
        if self.param_magnitudes_tensor:
            for name, norm_tensor in self.param_magnitudes_tensor.items():
                self.param_magnitudes[name] += norm_tensor.item()
            self.param_magnitudes_tensor.clear()

        if self.grad_magnitudes_tensor:
            for name, norm_tensor in self.grad_magnitudes_tensor.items():
                self.grad_magnitudes[name] += norm_tensor.item()
            self.grad_magnitudes_tensor.clear()

    def _add_tensor(self, tensor_attr, scalar_attr, value_tensor):
        """
        Helper to accumulate tensor values efficiently.
        If value is a tensor, accumulate on GPU.
        If value is a scalar, accumulate directly (for backwards compatibility).
        """
        import torch
        if torch.is_tensor(value_tensor):
            # GPU tensor - accumulate efficiently
            current_tensor = getattr(self, tensor_attr)
            if current_tensor is None:
                setattr(self, tensor_attr, value_tensor.detach())
            else:
                setattr(self, tensor_attr, current_tensor + value_tensor.detach())
        else:
            # Scalar value - accumulate directly
            current_scalar = getattr(self, scalar_attr)
            setattr(self, scalar_attr, current_scalar + value_tensor)

    @classmethod
    def from_loss_logits_target(cls, loss, logits, target, padding_idx):
        """
        Alternate constructor for computing the stats from
        loss, model prediction logits, and target indices.
        Note that this is heavy. Only use for validation / debug purposes.

        Args:
            loss: Loss value (can be a tensor or scalar)
            logits: Model prediction logits
            target: Target indices
            padding_idx: Padding index to exclude from statistics
        """
        import torch

        target = target.squeeze(-1)
        pred = logits.max(dim=-1).indices
        correct = pred.eq(target)
        non_padding = target.ne(padding_idx)
        correct_not_padded = correct.masked_select(non_padding)

        # Create stats object with loss=0 to avoid .item() call
        stats = cls(loss=0, n_words=0, n_correct=0)

        # Accumulate loss as tensor to avoid GPU->CPU sync
        stats._add_tensor('loss_tensor', 'loss', loss)

        # Accumulate other tensors instead of calling .item()
        stats._add_tensor('n_words_tensor', 'n_words', non_padding.sum())
        stats._add_tensor('n_correct_tensor', 'n_correct', correct_not_padded.sum())
        return stats

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from mammoth.distributed import all_gather_list

        # Materialize scalars before serialization (CUDA tensors can't be pickled)
        for stat in stat_list:
            stat.materialize_scalars()

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        import torch

        # Handle loss - prefer tensor version if available
        if stat.loss_tensor is not None:
            self._add_tensor('loss_tensor', 'loss', stat.loss_tensor)
        else:
            self.loss += stat.loss

        # Handle n_words - prefer tensor version if available
        if stat.n_words_tensor is not None:
            self._add_tensor('n_words_tensor', 'n_words', stat.n_words_tensor)
        elif stat.n_words:
            self.n_words += stat.n_words

        # Handle n_correct - prefer tensor version if available
        if stat.n_correct_tensor is not None:
            self._add_tensor('n_correct_tensor', 'n_correct', stat.n_correct_tensor)
        elif stat.n_correct:
            self.n_correct += stat.n_correct

        if update_n_src_words:
            # Handle n_src_words - prefer tensor version if available
            if stat.n_src_words_tensor is not None:
                self._add_tensor('n_src_words_tensor', 'n_src_words', stat.n_src_words_tensor)
            else:
                self.n_src_words += stat.n_src_words

        # Handle per-task loss tensors
        # When gathering stats across GPUs or batches, merge the tensor dictionaries
        for task_key, loss_tensor in stat.loss_per_task_tensor.items():
            if task_key not in self.loss_per_task_tensor:
                self.loss_per_task_tensor[task_key] = loss_tensor.detach()
            else:
                self.loss_per_task_tensor[task_key] += loss_tensor.detach()

        # Handle parameter/gradient magnitude tensors
        # When gathering stats across GPUs, merge the tensor dictionaries
        for name, tensor in stat.param_magnitudes_tensor.items():
            if name not in self.param_magnitudes_tensor:
                self.param_magnitudes_tensor[name] = tensor.detach()
            else:
                self.param_magnitudes_tensor[name] += tensor.detach()

        for name, tensor in stat.grad_magnitudes_tensor.items():
            if name not in self.grad_magnitudes_tensor:
                self.grad_magnitudes_tensor[name] = tensor.detach()
            else:
                self.grad_magnitudes_tensor[name] += tensor.detach()

        # Update validation metrics
        for metric_name, metric_value in stat.validation_metrics.items():
            if metric_name in self.validation_metrics:
                # For metrics like BLEU, we may want to average or accumulate differently
                # For now, we'll take the latest value (could be enhanced later)
                self.validation_metrics[metric_name] = metric_value
            else:
                self.validation_metrics[metric_name] = metric_value

    def update_task_loss(self, loss, metadata):
        """
        Update per-task loss tracking.
        Accumulates losses as tensors on GPU to avoid synchronization overhead.
        Losses are only materialized to CPU scalars during reporting.

        Args:
            loss: Loss value (can be a tensor or scalar)
            metadata: Batch metadata containing src_lang and tgt_lang
        """
        import torch

        key = f'{metadata.src_lang}_{metadata.tgt_lang}'

        if torch.is_tensor(loss):
            # Accumulate as tensor on GPU (Megatron-style: no .item() call!)
            if key not in self.loss_per_task_tensor:
                self.loss_per_task_tensor[key] = loss.detach()
            else:
                self.loss_per_task_tensor[key] += loss.detach()
        else:
            # Scalar value - accumulate directly (backwards compatibility)
            if not loss:
                logger.info(f'not loss {metadata.src_lang}_{metadata.tgt_lang}')
                return
            self.loss_per_task[key] += loss

    def update_from_parameters(self, named_parameters):
        """
        Megatron-style parameter/gradient norm tracking.
        Accumulates norms as GPU tensors, only syncs to CPU when reporting.
        This dramatically reduces GPU->CPU synchronization overhead.
        """
        self.magnitude_denom += 1

        # Accumulate L2 norms of parameters and their gradients
        # setting dim=None, ord=None flattens the matrix and computes a vector 2-norm
        # in newer versions of torch, vector_norm could be used
        for name, param in named_parameters:
            try:
                # Compute norm but keep it on GPU as tensor
                param_norm = norm(param.data, dim=None, ord=None)

                # Accumulate in tensor dictionary (no GPU->CPU sync!)
                if name not in self.param_magnitudes_tensor:
                    self.param_magnitudes_tensor[name] = param_norm.detach()
                else:
                    self.param_magnitudes_tensor[name] += param_norm.detach()

                # Same for gradients
                if param.requires_grad and param.grad is not None:
                    grad_norm = norm(param.grad.data, dim=None, ord=None)

                    if name not in self.grad_magnitudes_tensor:
                        self.grad_magnitudes_tensor[name] = grad_norm.detach()
                    else:
                        self.grad_magnitudes_tensor[name] += grad_norm.detach()

            except RuntimeError as e:
                logger.error(f'RuntimeError when updating stats for parameter {name}: {e}')

    def accuracy(self):
        """compute accuracy"""
        # Ensure scalars are materialized if tensors exist
        if self.n_correct_tensor is not None or self.n_words_tensor is not None:
            self.materialize_scalars()
        if self.n_correct is not None and self.n_words:
            return 100 * (self.n_correct / self.n_words)
        else:
            return None

    def xent(self):
        """compute cross entropy"""
        # Ensure scalars are materialized if tensors exist
        if self.loss_tensor is not None or self.n_words_tensor is not None:
            self.materialize_scalars()
        if self.n_words:
            return self.loss / self.n_words
        else:
            logger.warning('Number of non-padding tokens not tracked: reported loss is unnormalized')
            return self.loss

    def ppl(self):
        """compute perplexity"""
        # Ensure scalars are materialized if tensors exist
        if self.loss_tensor is not None or self.n_words_tensor is not None:
            self.materialize_scalars()
        if not self.n_words:
            return None
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """compute elapsed time"""
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start, metadata=None):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        # Ensure all scalars are materialized before reporting
        self.materialize_scalars()

        t = self.elapsed_time()
        step_fmt = "%2d" % step
        # if metadata:
        #     meta_str = '; '.join([f'{key}: {val}' for key, val in zip(metadata._fields, metadata)])
        # else:
        meta_str = ''
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        acc = self.accuracy()
        acc_str = f'{acc:6.2f}' if acc is not None else '--'
        ppl = self.ppl()
        ppl_str = f'{ppl:5.2f}' if ppl is not None else '--'
        logger.info(
            ("%s: Step %s; acc: %s; ppl: %s; xent: %4.2f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (
                meta_str,
                step_fmt,
                acc_str,
                ppl_str,
                self.xent(),
                # learning_rate,    # was "lr: %7.5f;"
                self.n_src_words / (t + 1e-5),
                self.n_words / (t + 1e-5),
                time.time() - start,
            )
        )
        if len(self.loss_per_task) > 0:
            for key, loss in self.loss_per_task.items():
                logger.info(f'{step} loss_per_task/{key}: {loss}')
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, patience, step):
        """display statistics to tensorboard"""
        # Ensure all scalars are materialized before logging
        self.materialize_scalars()

        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        ppl = self.ppl()
        if ppl is not None:
            writer.add_scalar(prefix + "/ppl", ppl, step)
        acc = self.accuracy()
        if acc is not None:
            writer.add_scalar(prefix + "/accuracy", acc, step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        # writer.add_scalar(prefix + "/lr", learning_rate, step)
        if patience is not None:
            writer.add_scalar(prefix + "/patience", patience, step)

        if self.magnitude_denom > 0:
            warnings.warn(
                '!!!!!!!!!!!!!!! --report_stats_from_parameters enabled: '
                'Megatron-style optimization reduces overhead, but still adds some cost. '
                'Use only for debugging/model inspection !!!!!!!!!!!!!'
            )
            # log parameter-level statistics
            for param, magnitude in self.param_magnitudes.items():
                writer.add_scalar(f'params/{param}', magnitude / self.magnitude_denom, step)
            for param, magnitude in self.grad_magnitudes.items():
                writer.add_scalar(f'grads/{param}', magnitude / self.magnitude_denom, step)

        if len(self.loss_per_task) > 0:
            for key, loss in self.loss_per_task.items():
                writer.add_scalar(f'loss_per_task/{key}', loss, step)
