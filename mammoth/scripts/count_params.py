"""Count parameters in a trained Mammoth checkpoint: total, per task, or per component.

  --by task       (default) one row per task:
                    full       every parameter the task's forward pass touches (what you would
                               have to ship to run this one task; shared parts are counted in full)
                    exclusive  parameters used by this task and no other task
                    shared     full - exclusive
  --by component  one row per component (= checkpoint shard: embeddings, layer stacks,
                  wrappers, adapters), grouped by kind, with the number of tasks using it

  total           every parameter in the checkpoint, each component counted once

Optimizer shards (*_optim.pt) are ignored. --json always writes both views.

Usage:
    python count_params.py --checkpoint-dir path/to/ckpt [--by {task,component}] [--task TASK_ID]
                           [-v] [--json out.json] [--debug]
"""
import argparse
import collections
import glob
import io
import json
import math
import os
import pickle
import re
import sys
import time
import types
import zipfile
from collections import defaultdict

import torch


def resolve_prefix(ckpt_dir):
    """Best checkpoint if present, else the latest step (counts are identical across steps).

    Returns the full file-name prefix including any leading model name, e.g. '_best', 'model_best'
    or 'model_step_1000', so that f'{prefix}_frame.pt' is the frame file.
    """
    best = sorted(os.path.basename(f)[:-len('_frame.pt')]
                  for f in glob.glob(os.path.join(ckpt_dir, '*_best_frame.pt')))
    if best:
        if len(best) > 1:
            print(f'[WARN] several best checkpoints {best}; using {best[0]!r}', file=sys.stderr)
        return best[0]
    steps = []
    for f in glob.glob(os.path.join(ckpt_dir, '*_step_*_frame.pt')):
        if m := re.search(r'^(.*_step_(\d+))_frame\.pt$', os.path.basename(f)):
            steps.append((int(m.group(2)), m.group(1)))
    if not steps:
        raise FileNotFoundError(f'No *_frame.pt found in {ckpt_dir}')
    return max(steps)[1]


class _Stub:
    """Inert stand-in for a class pickled in a checkpoint frame whose module can't be imported."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__['_state'] = state


class _StubUnpickler(pickle.Unpickler):
    """Real classes whenever importable; stubs for the rest. Lets this script run without mammoth installed."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub


# torch.load(pickle_module=...) wants a module exposing the pickle API.
_stub_pickle = types.ModuleType('stub_pickle')
_stub_pickle.__dict__.update({k: v for k, v in vars(pickle).items() if not k.startswith('__')})
_stub_pickle.Unpickler = _StubUnpickler


_DEBUG = False  # set from --debug in main(); gates the per-file timing lines on stderr


def _load(path):
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / 1e6
    t0 = time.perf_counter()
    try:
        out = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        mode = 'mmap'
    except Exception as e:
        # The frame pickles mammoth objects (e.g. tokenizer vocabs) that this script never reads.
        out = torch.load(path, map_location='cpu', weights_only=False, pickle_module=_stub_pickle)
        mode = f'FALLBACK full read ({type(e).__name__}: {str(e).splitlines()[0][:100]})'
    if _DEBUG:
        print(f'[load] {name}: {time.perf_counter() - t0:7.2f}s  {size_mb:10.1f} MB  {mode}', file=sys.stderr)
    return out


class _Meta:
    """Stand-in for a tensor while unpickling headers: remembers the element count, never the data."""

    def __init__(self, numel):
        self.numel = numel


def _rebuild_meta(storage, storage_offset, size, *rest):
    return _Meta(math.prod(size))


class _HeaderUnpickler(pickle.Unpickler):
    """Unpickles a torch zip checkpoint's data.pkl without touching any tensor storage."""

    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild_meta
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return lambda data, *rest: data
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        return _Stub  # storage classes etc.: ignored (we only read shapes)

    def persistent_load(self, pid):
        return None  # storage reference; its bytes are never needed


def _shard_header(path):
    """{key: numel} for a torch zip checkpoint, reading only data.pkl (a few KB, independent of file size)."""
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith('/data.pkl') or n == 'data.pkl')
        state = _HeaderUnpickler(io.BytesIO(z.read(pkl))).load()
    return {k: v.numel for k, v in state.items() if isinstance(v, _Meta)}


def shard_numel(path):
    """(parameter count, buffer count) of one shard; rotary inv_freq-style buffers are split out."""
    t0 = time.perf_counter()
    try:
        counts = _shard_header(path)
        if _DEBUG:
            print(f'[hdr]  {os.path.basename(path)}: {time.perf_counter() - t0:7.2f}s', file=sys.stderr)
    except Exception as e:
        print(f'[hdr]  {os.path.basename(path)}: header parse failed ({type(e).__name__}: {e}); using torch.load',
              file=sys.stderr)
        counts = {k: v.numel() for k, v in _load(path).items() if torch.is_tensor(v)}
    params = buffers = 0
    for k, n in counts.items():
        if 'inv_freq' in k:
            buffers += n
        else:
            params += n
    return params, buffers


def task_shards(task_id, spec):
    """Component (shard) names that this task's forward pass uses."""
    src, _, tgt = spec['src_tgt'].partition('-')
    enc = list(spec.get('enc_sharing_group') or [src])
    dec = list(spec.get('dec_sharing_group') or [tgt])
    names = [f'src_embeddings_{src}', f'tgt_embeddings_{tgt}',
             f"encoder_wrapper_{'_'.join(enc)}", f"decoder_wrapper_{'_'.join(dec)}"]
    names += [f'encoder_{i}_{x}' for i, x in enumerate(enc)]
    names += [f'decoder_{i}_{x}' for i, x in enumerate(dec)]
    return names


def component_kind(name):
    """Group a shard name into a readable category."""
    for prefix, label in (('src_embeddings_', 'source embeddings'), ('tgt_embeddings_', 'target embeddings'),
                          ('encoder_wrapper_', 'encoder wrapper (post_emb_norm etc.)'),
                          ('decoder_wrapper_', 'decoder wrapper (post_emb_norm, to_logits)'),
                          ('encoder_adapter_', 'encoder adapters'), ('decoder_adapter_', 'decoder adapters'),
                          ('encoder_', 'encoder layer stacks'), ('decoder_', 'decoder layer stacks')):
        if name.startswith(prefix):
            return label
    return 'other'


def fmt(n):
    return f'{n:>15,}  ({n / 1e6:8.2f}M)'


def collect(ckpt_dir):
    """Read a checkpoint directory into one table that both views are printed from.

    Returns {'prefix', 'total', 'total_buffers', 'tasks', 'components'} where
      components: {name: {'params', 'buffers', 'kind', 'tasks': [task ids using it]}}
      tasks:      [{'task', 'src_tgt', 'components', 'full', 'exclusive', 'shared', 'missing_shards'}]
    """
    prefix = resolve_prefix(ckpt_dir)
    frame = _load(os.path.join(ckpt_dir, f'{prefix}_frame.pt'))
    task_specs = getattr(frame['opts'], 'tasks', None) or {}

    # Every non-optimizer component shard on disk
    shard_re = re.compile(rf'^{re.escape(prefix)}_(.+)\.pt$')
    components = {}
    for f in sorted(os.listdir(ckpt_dir)):
        m = shard_re.match(f)
        if not m or m.group(1) == 'frame' or m.group(1).endswith('_optim'):
            continue
        params, buffers = shard_numel(os.path.join(ckpt_dir, f))
        components[m.group(1)] = {'params': params, 'buffers': buffers, 'kind': component_kind(m.group(1)),
                                  'tasks': []}

    per_task = {tid: task_shards(tid, spec) for tid, spec in task_specs.items()}
    for tid, names in per_task.items():
        for n in names:
            if n in components:
                components[n]['tasks'].append(tid)

    tasks = []
    for tid, names in per_task.items():
        found = [n for n in names if n in components]
        full = sum(components[n]['params'] for n in found)
        excl = sum(components[n]['params'] for n in found if len(components[n]['tasks']) == 1)
        tasks.append({'task': tid, 'src_tgt': task_specs[tid]['src_tgt'], 'components': found,
                      'full': full, 'exclusive': excl, 'shared': full - excl,
                      'missing_shards': [n for n in names if n not in components]})

    return {'prefix': prefix,
            'total': sum(c['params'] for c in components.values()),
            'total_buffers': sum(c['buffers'] for c in components.values()),
            'tasks': tasks,
            'components': components}


def print_by_task(data, task=None):
    """One row per task: full / exclusive / shared parameter counts."""
    rows = [r for r in data['tasks'] if task is None or r['task'] == task]
    if rows:
        w = max(len(r['task']) for r in rows)
        print(f"{'task':<{w}}  {'full':>24}  {'exclusive':>24}  {'shared':>24}")
        for r in rows:
            print(f"{r['task']:<{w}}  {fmt(r['full'])}  {fmt(r['exclusive'])}  {fmt(r['shared'])}")
            if r['missing_shards']:
                print(f"  [WARN] shards not found on disk: {r['missing_shards']}")

    unused = {n: c['params'] for n, c in data['components'].items() if not c['tasks']}
    if unused and task is None:
        print(f'\nShards not attributed to any task (adapters, attention bridge, or unexpected): '
              f'{fmt(sum(unused.values()))}')
        for n, p in unused.items():
            print(f'  {n}: {p:,}')


def print_by_component(data, task=None, verbose=False):
    """One row per component, grouped by kind with subtotals.

    With task: only the components that task's forward pass uses, percentages relative to the task.
    With verbose: also list the tasks using each component.
    """
    components = data['components']
    if task is None:
        names, base = list(components), data['total']
        print('Components (each counted once):')
    else:
        row = next(r for r in data['tasks'] if r['task'] == task)
        names, base = row['components'], row['full']
        print(f"Components used by task {task!r} ({row['src_tgt']}):")
        if row['missing_shards']:
            print(f"  [WARN] shards not found on disk: {row['missing_shards']}")
    if not names:
        return

    groups = defaultdict(list)
    for n in names:
        groups[components[n]['kind']].append(n)

    def pct(n):
        return f'{100 * n / base:5.1f}%' if base else '    -'

    w = max(len(n) for n in names)
    for kind, items in sorted(groups.items(), key=lambda kv: -sum(components[n]['params'] for n in kv[1])):
        sub = sum(components[n]['params'] for n in items)
        print(f'\n  {kind}: {len(items)} component(s), {fmt(sub)}  {pct(sub)}')
        for n in sorted(items):
            c = components[n]
            users = c['tasks']
            used = f'used by {len(users)} task(s)' if users else 'not used by any task'
            if verbose and users:
                used += f": {', '.join(users)}"
            print(f"    {n:<{w}}  {fmt(c['params'])}  {pct(c['params'])}  {used}")
    print(f"\n  sum: {fmt(sum(components[n]['params'] for n in names))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--by', choices=('task', 'component'), default='task',
                    help='one row per task (default) or one row per component')
    ap.add_argument('--task', default=None, metavar='TASK_ID',
                    help='report only this task (with --by component: the components it uses)')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='with --by component: list the tasks using each component')
    ap.add_argument('--debug', action='store_true', help='print per-file load timing on stderr')
    ap.add_argument('--json', default=None, help='also write both views to this file')
    args = ap.parse_args()
    global _DEBUG
    _DEBUG = args.debug

    data = collect(args.checkpoint_dir)
    print(f"Checkpoint prefix: {data['prefix']!r} in {args.checkpoint_dir}")
    task_ids = [r['task'] for r in data['tasks']]
    if args.task is not None and args.task not in task_ids:
        raise SystemExit(f'Unknown task {args.task!r}. '
                         f'Available tasks: {", ".join(task_ids) or "(none found in frame opts)"}')

    print(f"\nTOTAL parameters (each component once): {fmt(data['total'])}")
    if data['total_buffers']:
        print(f"  (+ {data['total_buffers']:,} buffer elements, e.g. rotary inv_freq, not counted)")
    print(f"Components: {len(data['components'])}   Tasks: {len(data['tasks'])}\n")

    if args.by == 'task':
        print_by_task(data, args.task)
    else:
        print_by_component(data, args.task, args.verbose)

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(data, fh, indent=2)
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()
