"""Tests for per-task validation sample logging.

Trainer.validate() used to cap sample logging at the first N batches of the
*whole* validation loop (all tasks concatenated together). Tasks whose
corpus_id sorted later were starved of logged samples even though their
batches were fully processed for loss/BLEU purposes. `PerTaskSampleLogger`
gives every task its own budget so each validated task gets its own logged
samples regardless of iteration order.
"""

from mammoth.trainer import PerTaskSampleLogger


def test_first_batches_of_each_task_are_logged():
    """Each corpus_id gets its own budget of logged batches."""
    logger = PerTaskSampleLogger(max_batches_to_log=3)

    # Simulate a validation loop where task_a's batches happen to be
    # exhausted before task_b's batches even start (alphabetical corpus
    # ordering), as happens with itertools.chain.from_iterable.
    decisions_a = [logger.should_log('task_a') for _ in range(5)]
    decisions_b = [logger.should_log('task_b') for _ in range(5)]

    assert decisions_a == [True, True, True, False, False]
    assert decisions_b == [True, True, True, False, False]


def test_interleaved_tasks_each_get_their_own_budget():
    """Interleaving batches from two tasks must not share one global counter."""
    logger = PerTaskSampleLogger(max_batches_to_log=2)

    calls = ['task_a', 'task_b', 'task_a', 'task_b', 'task_a', 'task_b']
    decisions = [logger.should_log(corpus_id) for corpus_id in calls]

    # task_a: batch 1 (log), batch 2 (log), batch 3 (skip)
    # task_b: batch 1 (log), batch 2 (log), batch 3 (skip)
    assert decisions == [True, True, True, True, False, False]


def test_zero_budget_never_logs():
    logger = PerTaskSampleLogger(max_batches_to_log=0)
    assert logger.should_log('task_a') is False
    assert logger.should_log('task_a') is False
