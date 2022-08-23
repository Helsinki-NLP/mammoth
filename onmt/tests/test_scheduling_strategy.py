import pytest

from onmt.utils.distributed import WeightedSamplingSchedulingStrategy, RoundRobinSchedulingStrategy


def test_weights_all_zero():
    opt_data = {
        'a': {
            'weight': 0,
            'introduce_at_training_step': 0,
        },
        'b': {
            'weight': 0,
            'introduce_at_training_step': 0,
        },
        'notmine': {
            'weight': 10,
            'introduce_at_training_step': 0,
        },
    }
    with pytest.raises(ValueError) as exc_info:
        WeightedSamplingSchedulingStrategy(['a', 'b'], opt_data)
    assert 'Can not set "weight" of all corpora on a device to zero' in str(exc_info.value)


def test_weights_all_postponed():
    opt_data = {
        'a': {
            'weight': 1,
            'introduce_at_training_step': 1,
        },
        'b': {
            'weight': 1,
            'introduce_at_training_step': 10,
        },
        'notmine': {
            'weight': 10,
            'introduce_at_training_step': 0,
        },
    }
    with pytest.raises(ValueError) as exc_info:
        WeightedSamplingSchedulingStrategy(['a', 'b'], opt_data)
    assert 'Can not set "introduce_at_training_step" of all corpora on a device to nonzero' in str(exc_info.value)


def test_invalid_curriculum():
    opt_data = {
        # 'a' disabled by weight
        'a': {
            'weight': 0,
            'introduce_at_training_step': 0,
        },
        # 'b' postponed
        'b': {
            'weight': 1,
            'introduce_at_training_step': 10,
        },
        'notmine': {
            'weight': 10,
            'introduce_at_training_step': 0,
        },
    }
    with pytest.raises(ValueError) as exc_info:
        WeightedSamplingSchedulingStrategy(['a', 'b'], opt_data)
    assert 'Invalid curriculum' in str(exc_info.value)


def test_sampling_scheduling_strategy():
    opt_data = {
        # 'a' disabled by weight
        'a': {
            'weight': 0,
            'introduce_at_training_step': 0,
        },
        # 'b' postponed longer than n_batches
        'b': {
            'weight': 1,
            'introduce_at_training_step': 9999,
        },
        # 'c' is the only valid corpus
        'c': {
            'weight': 1,
            'introduce_at_training_step': 0,
        },
        # 'notmine' is not on this device
        'notmine': {
            'weight': 10,
            'introduce_at_training_step': 0,
        },
    }
    strategy = WeightedSamplingSchedulingStrategy(['a', 'b', 'c'], opt_data)
    all_samples = []
    n_samples = 10
    n_batches = 1000
    for i in range(n_batches):
        sampled_corpus_ids = strategy.sample_corpus_ids(n_samples=n_samples, communication_batch_id=i)
        assert len(sampled_corpus_ids) == n_samples
        all_samples.extend(sampled_corpus_ids)
    assert len(all_samples) == n_samples * n_batches
    # 'c' is the only valid corpus, nothing else should be drawn
    assert set(all_samples) == {'c'}


def test_round_robin_scheduling_strategy():
    strategy = RoundRobinSchedulingStrategy(['a', 'b'], None)
    first_five = strategy.sample_corpus_ids(n_samples=5, communication_batch_id=0)
    assert first_five == ['a', 'b', 'a', 'b', 'a']
    next_two = strategy.sample_corpus_ids(n_samples=2, communication_batch_id=0)
    assert next_two == ['b', 'a']
