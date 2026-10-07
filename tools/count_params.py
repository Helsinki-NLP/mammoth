"""Count parameters in a trained Mammoth checkpoint: total, and per task.

  full       every parameter the task's forward pass touches (what you would
             have to ship to run this one task; shared parts are counted in full)
  exclusive  parameters used by this task and no other task

  total      every parameter in the checkpoint, each shard counted once

Optimizer shards (*_optim.pt) are ignored.

Usage:
    python tools/count_params.py --checkpoint-dir path/to/ckpt [--json out.json]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import torch


def resolve_prefix(ckpt_dir):
    """Best checkpoint if present, else the latest step (counts are identical across steps)."""
    if glob.glob(os.path.join(ckpt_dir, '*_best_frame.pt')):
        return '_best'
    steps = [
        int(m.group(1))
        for f in glob.glob(os.path.join(ckpt_dir, '*_step_*_frame.pt'))
        if (m := re.search(r'_step_(\d+)_frame\.pt$', f))
    ]
    if not steps:
        raise FileNotFoundError(f'No *_frame.pt found in {ckpt_dir}')
    return f'_step_{max(steps)}'


def _load(path):
    try:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def shard_numel(path):
    """(parameter count, buffer count) of one shard; rotary inv_freq-style buffers are split out."""
    params = buffers = 0
    for k, v in _load(path).items():
        if not torch.is_tensor(v):
            continue
        if 'inv_freq' in k:
            buffers += v.numel()
        else:
            params += v.numel()
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='list every component shard that adds up to the total, before the task table')
    ap.add_argument('--json', default=None, help='also write results to this file')
    args = ap.parse_args()

    prefix = resolve_prefix(args.checkpoint_dir)
    print(f'Checkpoint prefix: {prefix!r} in {args.checkpoint_dir}')

    frame = _load(os.path.join(args.checkpoint_dir, f'{prefix}_frame.pt'))
    tasks = getattr(frame['opts'], 'tasks', None) or {}

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

    if args.verbose:
        print_breakdown(shards, users, total)

    if rows:
        w = max(len(r['task']) for r in rows)
        print(f"{'task':<{w}}  {'full':>24}  {'exclusive':>24}  {'shared':>24}")
        for r in rows:
            print(f"{r['task']:<{w}}  {fmt(r['full'])}  {fmt(r['exclusive'])}  {fmt(r['shared'])}")
            if r['missing_shards']:
                print(f"  [WARN] shards not found on disk: {r['missing_shards']}")

    if unclaimed:
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
