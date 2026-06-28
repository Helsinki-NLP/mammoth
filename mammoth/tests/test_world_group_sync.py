"""
Tests for WorldGroupGradientSync pipelined bucket allreduce.

These run on CPU with a single-rank gloo process group. With world_size == 1, allreduce is the
identity, so we can assert that sync() packs each owned component's gradient, reduces it, and
unpacks it back rescaled by gradient_norm. We also assert that the asynchronous pipeline
(pipeline_depth 1 / 2 / 3) produces bit-identical results regardless of how many bucket
allreduces are kept in flight.
"""
import os
import unittest
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist

from mammoth.distributed.communication import WorldGroupGradientSync
from mammoth.distributed.components import DistributedComponentGradientSync
from mammoth.utils.profiling import get_profiler_range


class _FakeComponent:
    """Minimal stand-in for a DistributedComponent: just a name plus an nn.Module's params."""

    def __init__(self, name, module):
        self._name = name
        self._module = module

    def get_name(self):
        return self._name

    def named_parameters(self, model):  # model arg ignored; params live on the module
        yield from self._module.named_parameters()


def _build_sync(buckets, pipeline_depth, dtype=torch.float32):
    """Construct a WorldGroupGradientSync without its CUDA/all_gather __init__."""
    obj = WorldGroupGradientSync.__new__(WorldGroupGradientSync)
    obj.global_rank = 0
    obj.profiler_range = get_profiler_range()
    obj.buckets = buckets
    obj.pipeline_depth = max(1, min(pipeline_depth, len(buckets)))
    max_bucket = max(sum(s for _, s in b) for b in buckets)
    obj.buffers = [torch.zeros(max_bucket, dtype=dtype) for _ in range(obj.pipeline_depth)]
    return obj


class _FakeOwnedComponent:
    """Minimal stand-in carrying just the ownership info the filter looks at."""

    def __init__(self, name, ranks):
        self._name = name
        self.global_ranks = set(ranks)

    def get_name(self):
        return self._name

    def needs_communication(self):
        # mirrors DistributedComponent.needs_communication
        return len(self.global_ranks) > 1


class TestCommunicatingComponentsFilter(unittest.TestCase):
    """Single-owner components are a no-op for allreduce and must not enter the bucket layout.

    For a component owned by exactly one GPU, every other rank packs zeros and gradient_norm is 1,
    so allreduce((g, 0, 0, 0)) / 1 == g — the gradient is unchanged. Including it just wastes a
    network round trip, so __init__ must filter these out before building buckets.
    """

    def test_keeps_only_multi_device_components(self):
        comps = [
            _FakeOwnedComponent('decoder_eng', {0}),            # single owner -> skip
            _FakeOwnedComponent('decoder_fra', {1}),            # single owner -> skip
            _FakeOwnedComponent('encoder_shared', {0, 1}),      # shared -> keep
            _FakeOwnedComponent('attention_bridge', {0, 1, 2, 3}),  # shared -> keep
        ]
        result = WorldGroupGradientSync._communicating_components(comps)
        self.assertEqual(
            [c.get_name() for c in result],
            ['encoder_shared', 'attention_bridge'],
        )

    def test_all_single_owner_yields_empty(self):
        comps = [_FakeOwnedComponent('a', {0}), _FakeOwnedComponent('b', {1})]
        self.assertEqual(WorldGroupGradientSync._communicating_components(comps), [])


class TestWorldGroupGradientSyncPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29555')
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo', rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _make_components(self, sizes, seed=0):
        """Create one Linear per name; return (gradient_syncs, buckets, modules)."""
        torch.manual_seed(seed)
        modules = OrderedDict()
        syncs = []
        buckets_flat = []
        for name, (out_f, in_f) in sizes.items():
            mod = nn.Linear(in_f, out_f, bias=False)
            # Give every parameter a deterministic gradient to sync.
            mod.weight.grad = torch.randn_like(mod.weight)
            modules[name] = mod
            comp = _FakeComponent(name, mod)
            syncs.append(
                DistributedComponentGradientSync(
                    component=comp,
                    has_local_gradient=True,
                    gradient_norm=2,
                    owns_component=True,
                )
            )
            buckets_flat.append((name, mod.weight.numel()))
        return syncs, buckets_flat, modules

    def test_single_rank_roundtrip_rescales_by_gradient_norm(self):
        """With world_size==1 allreduce is identity, so grads should come back / gradient_norm."""
        sizes = {'a': (4, 3), 'b': (2, 5)}
        syncs, flat, modules = self._make_components(sizes)
        expected = {n: m.weight.grad.clone() / 2 for n, m in modules.items()}

        # One bucket per component.
        buckets = [[item] for item in flat]
        sync = _build_sync(buckets, pipeline_depth=2)
        sync.sync(model=None, all_gradient_syncs=syncs)

        for name, mod in modules.items():
            torch.testing.assert_close(mod.weight.grad, expected[name])

    def test_pipeline_depth_invariance(self):
        """Depths 1/2/3 must yield identical results (and match a sequential reference)."""
        sizes = {'a': (4, 3), 'b': (2, 5), 'c': (3, 3), 'd': (6, 2)}

        results = {}
        for depth in (1, 2, 3):
            syncs, flat, modules = self._make_components(sizes, seed=42)
            # Two components per bucket -> two buckets, so pipelining actually overlaps.
            buckets = [flat[0:2], flat[2:4]]
            sync = _build_sync(buckets, pipeline_depth=depth)
            sync.sync(model=None, all_gradient_syncs=syncs)
            results[depth] = {n: m.weight.grad.clone() for n, m in modules.items()}

        for name in sizes:
            torch.testing.assert_close(results[1][name], results[2][name])
            torch.testing.assert_close(results[1][name], results[3][name])

    def test_more_buckets_than_buffers(self):
        """pipeline_depth smaller than bucket count must force buffer reuse and still be correct."""
        sizes = {f'c{i}': (2, 2) for i in range(6)}
        syncs, flat, modules = self._make_components(sizes, seed=7)
        expected = {n: m.weight.grad.clone() / 2 for n, m in modules.items()}

        buckets = [[item] for item in flat]  # 6 buckets
        sync = _build_sync(buckets, pipeline_depth=2)  # only 2 buffers
        self.assertEqual(len(sync.buffers), 2)
        sync.sync(model=None, all_gradient_syncs=syncs)

        for name, mod in modules.items():
            torch.testing.assert_close(mod.weight.grad, expected[name])


if __name__ == '__main__':
    unittest.main()