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

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

        # losses per task
        self.loss_per_task = Counter()

        # parameter-level statistics
        self.magnitude_denom = 0
        self.param_magnitudes = Counter()
        self.grad_magnitudes = Counter()

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
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

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
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """compute cross entropy"""
        return self.loss / self.n_words

    def ppl(self):
        """compute perplexity"""
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
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        # if metadata:
        #     meta_str = '; '.join([f'{key}: {val}' for key, val in zip(metadata._fields, metadata)])
        # else:
        meta_str = ''
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("%s: Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " + "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (
                meta_str,
                step_fmt,
                self.accuracy(),
                self.ppl(),
                self.xent(),
                learning_rate,
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
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
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
