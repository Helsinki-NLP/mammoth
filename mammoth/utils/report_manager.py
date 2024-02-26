""" Report manager utility """
import time
from datetime import datetime

import mammoth

from mammoth.utils.logging import logger, structured_logging


def build_report_manager(opts, node_rank, local_rank):
    # Vanilla mammoth has here an additional gpu_rank <= 0
    # which would cause only the first GPU of each node to log.
    # This change allows all GPUs to log.
    # Because tensorboard does not allow multiple processes writing into the same directory,
    # each device is treated as a separate run.
    if opts.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        if not hasattr(opts, 'tensorboard_log_dir_dated'):
            opts.tensorboard_log_dir_dated = opts.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S")

        writer = SummaryWriter(f'{opts.tensorboard_log_dir_dated}-rank{node_rank}:{local_rank}', comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opts.report_every, start_time=-1, tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.0):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(
        self,
        step,
        num_steps,
        learning_rate,
        patience,
        report_stats,
        multigpu=False,
        sampled_task_count=None,
    ):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError(
                """ReportMgr needs to be started
                                (set 'start_time' or use 'start()'"""
            )

        if step % self.report_every == 0:
            # if multigpu:
            #    report_stats = \
            #        mammoth.utils.Statistics.all_gather_stats(report_stats)
            self._report_training(
                step,
                num_steps,
                learning_rate,
                patience,
                report_stats,
                sampled_task_count=sampled_task_count
            )
            return mammoth.utils.Statistics()
        else:
            return report_stats

    def _report_training(self, *args, **kwargs):
        """To be overridden"""
        raise NotImplementedError()

    def report_step(self, lr, patience, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            lr(float): current learning rate
            patience(int): current patience
            step(int): current step
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
        """
        self._report_step(lr, patience, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()

    def _report_end(self, step):
        raise NotImplementedError()

    def report_end(self, step):
        if self.start_time < 0:
            raise ValueError(
                """ReportMgr needs to be started
                                (set 'start_time' or use 'start()'"""
            )
        self._report_end(step)


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1.0, tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, patience, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(prefix, self.tensorboard_writer, learning_rate, patience, step)

    def _report_training(self, step, num_steps, learning_rate, patience, report_stats, sampled_task_count):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps, learning_rate, self.start_time)

        self.maybe_log_tensorboard(report_stats, "progress", learning_rate, patience, step)
        report_stats = mammoth.utils.Statistics()

        total = sum(sampled_task_count.values())
        logger.info(f'Task sampling distribution: (total {total})')
        for task, count in sampled_task_count.most_common():
            logger.info(f'Task: {task}\tcount: {count}\t{100 * count / total} %')

        return report_stats

    def _report_step(self, lr, patience, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())

            self.maybe_log_tensorboard(train_stats, "train", lr, patience, step)

        if valid_stats is not None:
            ppl = valid_stats.ppl()
            acc = valid_stats.accuracy()
            self.log('Validation perplexity: %g', ppl)
            self.log('Validation accuracy: %g', acc)
            structured_logging({
                'type': 'validation',
                'step': step,
                'learning_rate': lr,
                'perplexity': ppl,
                'accuracy': acc,
                'crossentropy': valid_stats.xent(),
            })
            self.maybe_log_tensorboard(valid_stats, "valid", lr, patience, step)

    def _report_end(self, step):
        end_time = time.time()
        duration_s = end_time - self.start_time
        self.log('Training ended. Duration %g s', duration_s)
        structured_logging({
            'type': 'end',
            'step': step,
            'start_time': self.start_time,
            'end_time': end_time,
            'duration_s': duration_s,
        })
