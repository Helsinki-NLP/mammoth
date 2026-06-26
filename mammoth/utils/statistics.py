""" Statistics calculation utility """
import time
import math
import sys
import warnings

from collections import Counter

import torch
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
        self.n_sents = 0
        self.cumulative_sents = 0  # Cumulative sentence count since training began
        self.flops_per_step = 0  # Total FLOPs for this reporting interval

        # losses per task
        self.loss_per_task = Counter()

        # parameter-level statistics
        self.magnitude_denom = 0
        self.param_magnitudes = Counter()
        self.grad_magnitudes = Counter()

        # validation metrics
        self.validation_metrics = {}

        # Device-side accumulators for deferred GPU->CPU sync (see
        # `accumulate_device` / `materialize`). These hold running sums as GPU
        # scalar tensors so the training hot loop never blocks on a `.item()`.
        # They are folded into the plain CPU fields above only when the figures
        # are actually needed (at the report step).
        self._dev_loss = None
        self._dev_n_words = None
        self._dev_n_src_words = None
        self._dev_nan = None
        self._dev_loss_per_task = {}
        # Sticky flag: True once any materialised window contained a NaN loss.
        self.had_nan = False

    def accumulate_device(self, loss, n_words, nan_flag, n_src_words=None, metadata=None):
        """Accumulate one window's statistics as GPU tensors, WITHOUT syncing.

        Everything stays on the GPU; nothing is copied to the CPU here, so this
        adds no `.item()` stall to the training loop. Call `materialize()` to
        read the running totals back (a single GPU->CPU sync) when they are
        needed.

        Args:
            loss: scalar GPU tensor, summed loss for this window.
            n_words: scalar GPU tensor, summed target (non-padding) tokens.
            nan_flag: scalar GPU bool tensor, True if any loss in the window was NaN.
            n_src_words: optional scalar GPU tensor, summed source tokens.
            metadata: optional task metadata; if given, the window loss is also
                tracked per task (src_lang_tgt_lang).
        """
        loss = loss.detach()
        if self._dev_loss is None:
            self._dev_loss = loss
            self._dev_n_words = n_words
            self._dev_nan = nan_flag
            self._dev_n_src_words = n_src_words
        else:
            self._dev_loss = self._dev_loss + loss
            self._dev_n_words = self._dev_n_words + n_words
            self._dev_nan = self._dev_nan | nan_flag
            if n_src_words is not None:
                if self._dev_n_src_words is None:
                    self._dev_n_src_words = n_src_words
                else:
                    self._dev_n_src_words = self._dev_n_src_words + n_src_words

        if metadata is not None:
            key = f'{metadata.src_lang}_{metadata.tgt_lang}'
            if key in self._dev_loss_per_task:
                self._dev_loss_per_task[key] = self._dev_loss_per_task[key] + loss
            else:
                self._dev_loss_per_task[key] = loss

    def materialize(self):
        """Fold the device-side accumulators into the CPU fields (one sync).

        This is the ONLY place the deferred statistics touch the CPU. All pending
        GPU scalars are stacked into a single tensor and read back with one
        `.tolist()`, so a whole reporting interval costs exactly one GPU->CPU
        synchronisation instead of one per step. No-op (and idempotent) when
        nothing is pending.
        """
        if self._dev_loss is None:
            return

        task_keys = list(self._dev_loss_per_task.keys())
        stack = [
            self._dev_loss.float().reshape(()),
            self._dev_n_words.float().reshape(()),
            self._dev_nan.float().reshape(()),
        ]
        if self._dev_n_src_words is not None:
            stack.append(self._dev_n_src_words.float().reshape(()))
        stack.extend(self._dev_loss_per_task[k].float().reshape(()) for k in task_keys)

        values = torch.stack(stack).tolist()
        loss_val, n_words_val, nan_val = values[0], values[1], values[2]
        idx = 3
        if self._dev_n_src_words is not None:
            self.n_src_words += int(values[idx])
            idx += 1
        for key in task_keys:
            self.loss_per_task[key] += values[idx]
            idx += 1

        self.loss += loss_val
        self.n_words += int(n_words_val)
        self.had_nan = self.had_nan or bool(nan_val)

        self._dev_loss = None
        self._dev_n_words = None
        self._dev_n_src_words = None
        self._dev_nan = None
        self._dev_loss_per_task = {}

    @classmethod
    def from_loss_logits_target(cls, loss: float, logits, target, padding_idx):
        """
        Alternate constructor for computing the stats from
        loss, model prediction logits, and target indices.
        Note that this is heavy. Only use for validation / debug purposes.
        """
        target = target.squeeze(-1)
        pred = logits.max(dim=-1).indices
        correct = pred.eq(target)
        non_padding = target.ne(padding_idx)
        correct_not_padded = correct.masked_select(non_padding)
        num_correct = correct_not_padded.sum().item()
        num_non_padding = non_padding.sum().item()
        return cls(loss, num_non_padding, num_correct)

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
        self.loss += stat.loss
        if stat.n_words:
            self.n_words += stat.n_words
        if stat.n_correct:
            self.n_correct += stat.n_correct

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

        self.flops_per_step += stat.flops_per_step

        # Update sentence counts
        if stat.n_sents:
            self.n_sents += stat.n_sents

        # Update validation metrics
        for metric_name, metric_value in stat.validation_metrics.items():
            if metric_name in self.validation_metrics:
                # For metrics like BLEU, we may want to average or accumulate differently
                # For now, we'll take the latest value (could be enhanced later)
                self.validation_metrics[metric_name] = metric_value
            else:
                self.validation_metrics[metric_name] = metric_value

    def update_task_loss(self, loss, metadata):
        if not loss:
            logger.info(f'not loss {metadata.src_lang}_{metadata.tgt_lang}')
            return
        key = f'{metadata.src_lang}_{metadata.tgt_lang}'
        self.loss_per_task[key] += loss

    def update_from_parameters(self, named_parameters):
        self.magnitude_denom += 1
        # Accumulate L2 norms of parameters and their gradients
        # setting dim=None, ord=None flattens the matrix and computes a vector 2-norm
        # in newer versions of torch, vector_norm could be used
        for name, param in named_parameters:
            try:
                self.param_magnitudes[name] += norm(param.data, dim=None, ord=None).item()
                if param.requires_grad and param.grad is not None:
                    self.grad_magnitudes[name] += norm(param.grad.data, dim=None, ord=None).item()
            except RuntimeError as e:
                logger.error(f'RuntimeError when updating stats for parameter {name}: {e}')

    def accuracy(self):
        """compute accuracy"""
        if self.n_correct is not None and self.n_words:
            return 100 * (self.n_correct / self.n_words)
        else:
            return None

    def xent(self):
        """compute cross entropy"""
        if self.n_words:
            return self.loss / self.n_words
        else:
            logger.warning('Number of non-padding tokens not tracked: reported loss is unnormalized')
            return self.loss

    def ppl(self):
        """compute perplexity"""
        if not self.n_words:
            return None
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """compute elapsed time"""
        return time.time() - self.start_time

    def tflops(self):
        """compute TFLOPs/s (teraFLOPs per second)"""
        if self.flops_per_step == 0:
            return 0.0
        return self.flops_per_step / (self.elapsed_time() + 1e-5) / 1e12

    def output(self, step, num_steps, learning_rate, start, metadata=None):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
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
        tflops = self.tflops()
        tflops_str = f'{tflops:6.2f} TFLOP/s; ' if tflops > 0 else ''
        logger.info(
            ("%s: Step %s; acc: %s; ppl: %s; xent: %4.2f; %s%3.0f/%3.0f tok/s; %6.0f sents; %6.0f sec;")
            % (
                meta_str,
                step_fmt,
                acc_str,
                ppl_str,
                self.xent(),
                tflops_str,
                # learning_rate,    # was "lr: %7.5f;"
                self.n_src_words / (t + 1e-5),
                self.n_words / (t + 1e-5),
                self.cumulative_sents,
                time.time() - start,
            )
        )
        if len(self.loss_per_task) > 0:
            for key, loss in self.loss_per_task.items():
                logger.info(f'{step} loss_per_task/{key}: {loss}')
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, patience, step):
        """display statistics to tensorboard"""
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        ppl = self.ppl()
        if ppl is not None:
            writer.add_scalar(prefix + "/ppl", ppl, step)
        acc = self.accuracy()
        if acc is not None:
            writer.add_scalar(prefix + "/accuracy", acc, step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        tflops = self.tflops()
        if tflops > 0:
            writer.add_scalar(prefix + "/tflops", tflops, step)
        # writer.add_scalar(prefix + "/lr", learning_rate, step)
        if patience is not None:
            writer.add_scalar(prefix + "/patience", patience, step)

        if self.magnitude_denom > 0:
            warnings.warn(
                '!!!!!!!!!!!!!!! --report_stats_from_parameters has a huge impact on performance: '
                'only use for debugging !!!!!!!!!!!!!'
            )
            # log parameter-level statistics
            for param, magnitude in self.param_magnitudes.items():
                writer.add_scalar(f'params/{param}', magnitude / self.magnitude_denom, step)
            for param, magnitude in self.grad_magnitudes.items():
                writer.add_scalar(f'grads/{param}', magnitude / self.magnitude_denom, step)

        if len(self.loss_per_task) > 0:
            for key, loss in self.loss_per_task.items():
                writer.add_scalar(f'loss_per_task/{key}', loss, step)
