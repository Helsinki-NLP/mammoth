from collections import defaultdict, Counter

from mammoth.distributed.tasks import (
    WeightedSamplingTaskDistributionStrategy,
    RoundRobinTaskDistributionStrategy,
    TaskSpecs,
)


def make_dummy_task(corpus_id, weight):
    return TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang='src',
        tgt_lang='tgt',
        encoder_id=['enc'],
        decoder_id=['dec'],
        corpus_id=corpus_id,
        weight=weight,
        introduce_at_training_step=0,
        corpus_opts=dict(),
        src_vocab=None,
        tgt_vocab=None,
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )


def test_sampling_task_distribution_strategy():
    n_devices = 2
    active_tasks = {
        0: [
            make_dummy_task('a', 50),
            make_dummy_task('b', 50),
        ],
        1: [
            make_dummy_task('c', 1),
        ],
    }
    strategy = WeightedSamplingTaskDistributionStrategy(seed=1)
    all_samples = defaultdict(Counter)
    n_batches = 1000
    for i in range(n_batches):
        batch_task_sample = strategy.sample_corpus_ids(active_tasks)
        assert batch_task_sample.training_step == i
        assert len(batch_task_sample.tasks) == n_devices
        for global_rank, task in batch_task_sample.tasks.items():
            all_samples[global_rank][task.corpus_id] += 1
    total_count = sum(sum(counts.values()) for counts in all_samples.values())
    assert total_count == n_devices * n_batches
    # 0 should sample from both tasks
    assert all_samples[0]['a'] > 0
    assert all_samples[0]['b'] > 0
    # 'c' is the only valid corpus for 1, nothing else should be drawn
    assert all_samples[1] == Counter({'c': n_batches})


def test_round_robin_task_distribution_strategy():
    n_devices = 2
    active_tasks = {
        0: [
            make_dummy_task('a', 50),
            make_dummy_task('b', 50),
        ],
        1: [
            make_dummy_task('c', 1),
        ],
    }
    strategy = RoundRobinTaskDistributionStrategy(seed=1)
    all_samples = defaultdict(Counter)
    n_batches = 1000
    for i in range(n_batches):
        batch_task_sample = strategy.sample_corpus_ids(active_tasks)
        assert batch_task_sample.training_step == i
        assert len(batch_task_sample.tasks) == n_devices
        for global_rank, task in batch_task_sample.tasks.items():
            all_samples[global_rank][task.corpus_id] += 1
    total_count = sum(sum(counts.values()) for counts in all_samples.values())
    assert total_count == n_devices * n_batches
    # 0 should sample from both tasks
    assert all_samples[0] == Counter(
        {
            'a': n_batches // 2,
            'b': n_batches // 2,
        }
    )
    # 'c' is the only valid corpus for 1, nothing else should be drawn
    assert all_samples[1] == Counter({'c': n_batches})
