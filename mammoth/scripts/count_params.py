"""Count parameters in a trained Mammoth checkpoint: total, and per task.

  full       every parameter the task's forward pass touches (what you would
             have to ship to run this one task; shared parts are counted in full)
  exclusive  parameters used by this task and no other task

  total      every parameter in the checkpoint, each shard counted once

Optimizer shards (*_optim.pt) are ignored.

Usage:
    python count_params.py --checkpoint-dir path/to/ckpt [--task TASK_ID] [-v] [--json out.json]
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


_VERBOSE = False  # set from -v/--verbose in main(); gates the per-file timing lines on stderr


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
    if _VERBOSE:
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
        if _VERBOSE:
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


def print_breakdown(shards, users, total):
    """Every shard counted in the total, grouped by kind, with per-group subtotals."""
    groups = defaultdict(list)
    for name, (p, _) in shards.items():
        groups[component_kind(name)].append((name, p))

    w = max(len(n) for n in shards)
    print('Components contributing to the total (each counted once):')
    for kind, items in sorted(groups.items(), key=lambda kv: -sum(p for _, p in kv[1])):
        sub = sum(p for _, p in items)
        print(f'\n  {kind}: {len(items)} shard(s), {fmt(sub)}  {100 * sub / total:5.1f}%')
        for name, p in sorted(items):
            n_users = len(users.get(name, ()))
            used = f'used by {n_users} task(s)' if n_users else 'not used by any task'
            print(f'    {name:<{w}}  {fmt(p)}  {100 * p / total:5.1f}%  {used}')
    check = sum(sum(p for _, p in items) for items in groups.values())
    print(f'\n  sum of all components: {fmt(check)}\n')


def print_task_detail(tid, spec, names, shards, users):
    """One task's shards grouped by kind; each marked exclusive or shared (and with whom)."""
    full = sum(shards[n][0] for n in names if n in shards)
    groups = defaultdict(list)
    for n in names:
        groups[component_kind(n)].append(n)

    w = max(len(n) for n in names)
    print(f"Task {tid!r} ({spec['src_tgt']}): every shard its forward pass uses")
    for kind, items in sorted(groups.items(), key=lambda kv: -sum(shards[n][0] for n in kv[1] if n in shards)):
        sub = sum(shards[n][0] for n in items if n in shards)
        print(f'\n  {kind}: {len(items)} shard(s), {fmt(sub)}  {100 * sub / full:5.1f}%')
        for n in sorted(items):
            if n not in shards:
                print(f'    {n:<{w}}  [WARN] not found on disk')
                continue
            others = sorted(users[n] - {tid})
            note = f"shared with {', '.join(others)}" if others else 'exclusive'
            print(f'    {n:<{w}}  {fmt(shards[n][0])}  {100 * shards[n][0] / full:5.1f}%  {note}')
    excl = sum(shards[n][0] for n in names if n in shards and len(users[n]) == 1)
    print(f'\n  full:      {fmt(full)}')
    print(f'  exclusive: {fmt(excl)}')
    print(f'  shared:    {fmt(full - excl)}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='list every component shard that adds up to the total, before the task table '
                         '(with --task: that one task\'s per-component detail instead)')
    ap.add_argument('--task', default=None, metavar='TASK_ID',
                    help='report only this task (add -v for its per-component detail)')
    ap.add_argument('--json', default=None, help='also write results to this file')
    args = ap.parse_args()
    global _VERBOSE
    _VERBOSE = args.verbose

    prefix = resolve_prefix(args.checkpoint_dir)
    print(f'Checkpoint prefix: {prefix!r} in {args.checkpoint_dir}')

    frame = _load(os.path.join(args.checkpoint_dir, f'{prefix}_frame.pt'))
    tasks = getattr(frame['opts'], 'tasks', None) or {}
    if args.task is not None and args.task not in tasks:
        raise SystemExit(f'Unknown task {args.task!r}. Available tasks: {", ".join(tasks) or "(none found in frame opts)"}')

    # Every non-optimizer component shard on disk: name -> (params, buffers)
    shard_re = re.compile(rf'^{re.escape(prefix)}_(.+)\.pt$')
    shards = {}
    for f in sorted(os.listdir(args.checkpoint_dir)):
        m = shard_re.match(f)
        if not m or m.group(1) == 'frame' or m.group(1).endswith('_optim'):
            continue
        shards[m.group(1)] = shard_numel(os.path.join(args.checkpoint_dir, f))

    total = sum(p for p, _ in shards.values())
    total_buf = sum(b for _, b in shards.values())

    # Which tasks use each shard (to find what is exclusive vs shared).
    users = defaultdict(set)
    per_task_shards = {}
    for tid, spec in tasks.items():
        names = task_shards(tid, spec)
        per_task_shards[tid] = names
        for n in names:
            users[n].add(tid)

    rows = []
    for tid, names in per_task_shards.items():
        missing = [n for n in names if n not in shards]
        full = sum(shards[n][0] for n in names if n in shards)
        excl = sum(shards[n][0] for n in names if n in shards and len(users[n]) == 1)
        rows.append({'task': tid, 'src_tgt': tasks[tid]['src_tgt'], 'full': full,
                     'exclusive': excl, 'shared': full - excl, 'missing_shards': missing})

    claimed = set(users)
    unclaimed = {n: shards[n][0] for n in shards if n not in claimed}

    print(f'\nTOTAL parameters (each shard once): {fmt(total)}')
    if total_buf:
        print(f'  (+ {total_buf:,} buffer elements, e.g. rotary inv_freq, not counted)')
    print(f'Component shards: {len(shards)}   Tasks: {len(rows)}\n')

    if args.verbose and args.task is None:
        print_breakdown(shards, users, total)

    if args.task is not None and args.verbose:
        print_task_detail(args.task, tasks[args.task], per_task_shards[args.task], shards, users)
    elif rows:
        if args.task is not None:
            rows = [r for r in rows if r['task'] == args.task]
        w = max(len(r['task']) for r in rows)
        print(f"{'task':<{w}}  {'full':>24}  {'exclusive':>24}  {'shared':>24}")
        for r in rows:
            print(f"{r['task']:<{w}}  {fmt(r['full'])}  {fmt(r['exclusive'])}  {fmt(r['shared'])}")
            if r['missing_shards']:
                print(f"  [WARN] shards not found on disk: {r['missing_shards']}")

    if unclaimed and args.task is None:
        print(f'\nShards not attributed to any task (adapters, attention bridge, or unexpected): '
              f'{fmt(sum(unclaimed.values()))}')
        for n, c in unclaimed.items():
            print(f'  {n}: {c:,}')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'prefix': prefix, 'total_params': total, 'tasks': rows,
                       'shards': {n: p for n, (p, _) in shards.items()},
                       'unattributed_shards': unclaimed}, fh, indent=2)
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()
