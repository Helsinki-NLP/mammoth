#!/usr/bin/env python3
"""
Convert a sharded Mammoth checkpoint → HuggingFace MammothForConditionalGeneration.

The script discovers src/tgt language pairs from the checkpoint's stored opts.tasks —
users do not pass --src/--tgt. Single-task checkpoints write flat into --output-dir;
multi-task checkpoints write one subdirectory per task: <output_dir>/<src>-<tgt>/.

Usage:
    # Convert all tasks (default; best checkpoint):
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --output-dir path/to/hf_output

    # Convert a specific task only:
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --task eng-spa \
        --output-dir path/to/hf_output

    # Load a specific step instead of the best checkpoint:
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --step 500 \
        --output-dir path/to/hf_output

Checkpoint shard naming:
    Best:  _best_frame.pt, _best_src_embeddings_{src}.pt, ...
    Step:  _step_{N}_frame.pt, _step_{N}_src_embeddings_{src}.pt, ...
"""

import argparse
import glob as glob_module
import os
import re
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import PreTrainedTokenizerFast

from mammoth.constants import DefaultTokens
from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def _first(v, default=0.0):
    """Return scalar from a value that may be a list (opts with nargs='+')."""
    if isinstance(v, (list, tuple)):
        return float(v[0]) if v else default
    return float(v) if v is not None else default


def config_from_opts(
    opts,
    src_vocab_size: int,
    tgt_vocab_size: int,
    bos_token_id: int = 2,
    eos_token_id: int = 0,
    pad_token_id: int = 1,
    encoder_sharing_groups: list | None = None,
    decoder_sharing_groups: list | None = None,
) -> MammothConfig:
    """Build MammothConfig from a native-mode Mammoth opts Namespace."""
    return MammothConfig(
        model_dim=opts.model_dim,
        heads=opts.heads,
        ff_mult=getattr(opts, 'ff_mult', 2.67),
        ff_swiglu=getattr(opts, 'ff_swiglu', True),
        enc_layers=list(opts.enc_layers),
        dec_layers=list(opts.dec_layers),
        attn_dropout=getattr(opts, 'attn_dropout', 0.0),
        ff_dropout=getattr(opts, 'ff_dropout', 0.0),
        emb_dropout=_first(getattr(opts, 'dropout', 0.0)),
        rotary_pos_emb=getattr(opts, 'rotary_pos_emb', True),
        post_emb_norm=getattr(opts, 'post_emb_norm', True),
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        decoder_start_token_id=bos_token_id,
        encoder_sharing_groups=encoder_sharing_groups,
        decoder_sharing_groups=decoder_sharing_groups,
    )


# ---------------------------------------------------------------------------
# Checkpoint prefix resolution
# ---------------------------------------------------------------------------

def resolve_prefix(ckpt_dir: str, step: int | None) -> str:
    """Return the shard filename prefix to use (_step_N or _best).

    Priority:
      1. --step N  → _step_N  (explicit, always wins)
      2. best checkpoint (*_best_frame.pt) if it exists
      3. fallback  → _step_<largest N found>
    """
    if step is not None:
        return f'_step_{step}'

    best_frames = glob_module.glob(os.path.join(ckpt_dir, '*_best_frame.pt'))
    if best_frames:
        print(f"Loading best checkpoint: {os.path.basename(best_frames[0])}")
        return '_best'

    frames = glob_module.glob(os.path.join(ckpt_dir, '*_step_*_frame.pt'))
    if not frames:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    steps = [
        int(m.group(1))
        for f in frames
        if (m := re.search(r'_step_(\d+)_frame\.pt$', os.path.basename(f)))
    ]
    if not steps:
        raise FileNotFoundError(f"Could not parse step numbers from frame files in {ckpt_dir}")
    best_step = max(steps)
    print(f"No best checkpoint found, loading last checkpoint: step {best_step}")
    return f'_step_{best_step}'


# ---------------------------------------------------------------------------
# State dict assembly from shards
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def _task_xcoder_ids(opts, src: str, tgt: str):
    """Return (encoder_id_list, decoder_id_list) for the given src→tgt task.

    Each list has one xcoder_id per layer stack index, e.g. ['eng', 'all'] for a
    2-stack encoder where the first stack is language-specific and the second shared.
    """
    tasks = getattr(opts, 'tasks', None) or {}
    for corpus_opts in tasks.values():
        src_tgt = corpus_opts.get('src_tgt', '')
        task_src, _, task_tgt = src_tgt.partition('-')
        if task_src == src and task_tgt == tgt:
            enc_id = corpus_opts.get('enc_sharing_group', [src])
            dec_id = corpus_opts.get('dec_sharing_group', [tgt])
            return list(enc_id), list(dec_id)
    available = [v.get('src_tgt') for v in tasks.values()]
    raise ValueError(f"Task {src}-{tgt} not found in opts.tasks. Available: {available}")


def _infer_xcoder_ids_from_shards(ckpt_dir: str, prefix: str, src: str, tgt: str):
    """Infer (encoder_id, decoder_id) lists by scanning shard filenames.

    Falls back to ([src], [tgt]) when no matching shards are found.
    """
    enc_pattern = re.compile(rf'^{re.escape(prefix)}_encoder_(\d+)_(.+?)(?:_optim)?\.pt$')
    dec_pattern = re.compile(rf'^{re.escape(prefix)}_decoder_(\d+)_(.+?)(?:_optim)?\.pt$')

    enc_map, dec_map = {}, {}
    for fname in os.listdir(ckpt_dir):
        m = enc_pattern.match(fname)
        if m:
            enc_map[int(m.group(1))] = m.group(2)
        m = dec_pattern.match(fname)
        if m:
            dec_map[int(m.group(1))] = m.group(2)

    if enc_map and dec_map:
        encoder_id = [enc_map[i] for i in sorted(enc_map)]
        decoder_id = [dec_map[i] for i in sorted(dec_map)]
        print(f"[WARN] opts.tasks empty — inferred from shards: encoder={encoder_id}, decoder={decoder_id}")
        return encoder_id, decoder_id

    print(f"[WARN] opts.tasks empty and no shards found — falling back to encoder=['{src}'], decoder=['{tgt}']")
    return [src], [tgt]


def assemble_state_dict(
    ckpt_dir: str, prefix: str, src: str, tgt: str,
    encoder_id: list, decoder_id: list,
) -> dict:
    """
    Reassemble a flat HF-model state dict from Mammoth's per-component shards.

    HF model key structure (maps directly to MammothEncoder / MammothDecoder):
        encoder.token_emb.weight
        encoder.post_emb_norm.weight
        encoder.stacks.{i}.blocks.{j}.<key>
        encoder.stacks.{i}.final_norm.weight      (last stack only)
        decoder.token_emb.weight
        decoder.post_emb_norm.weight
        decoder.stacks.{i}.blocks.{j}.<key>
        decoder.stacks.{i}.final_norm.weight      (last stack only)
        decoder.to_logits.weight

    Shard → HF key mapping:
        src_embeddings_{src}.pt              weight          → encoder.token_emb.weight
        encoder_wrapper_{enc_key}.pt         post_emb_norm.* → encoder.post_emb_norm.*
        encoder_{i}_{xcoder_id}.pt           blocks.*        → encoder.stacks.{i}.blocks.*
                                             final_norm.*    → encoder.stacks.{i}.final_norm.*
        tgt_embeddings_{tgt}.pt              weight          → decoder.token_emb.weight
        decoder_wrapper_{dec_key}.pt         post_emb_norm.* → decoder.post_emb_norm.*
                                             to_logits.*     → decoder.to_logits.*
        decoder_{i}_{xcoder_id}.pt           blocks.*        → decoder.stacks.{i}.blocks.*
                                             final_norm.*    → decoder.stacks.{i}.final_norm.*
    """
    sd = {}
    p = prefix

    def load_shard(filename: str) -> dict:
        path = os.path.join(ckpt_dir, filename)
        if not os.path.exists(path):
            print(f"  [WARN] missing shard: {path}")
            return {}
        return _load(path)

    # --- Encoder token embedding ---
    shard = load_shard(f'{p}_src_embeddings_{src}.pt')
    if 'weight' in shard:
        sd['encoder.token_emb.weight'] = shard['weight']

    # --- Encoder wrapper (post_emb_norm) ---
    enc_key = '_'.join(encoder_id)
    shard = load_shard(f'{p}_encoder_wrapper_{enc_key}.pt')
    if 'post_emb_norm.weight' in shard:
        sd['encoder.post_emb_norm.weight'] = shard['post_emb_norm.weight']

    # --- Encoder stacks ---
    for stack_idx, xcoder_id in enumerate(encoder_id):
        shard = load_shard(f'{p}_encoder_{stack_idx}_{xcoder_id}.pt')
        for k, v in shard.items():
            sd[f'encoder.stacks.{stack_idx}.{k}'] = v

    # --- Decoder token embedding ---
    shard = load_shard(f'{p}_tgt_embeddings_{tgt}.pt')
    if 'weight' in shard:
        sd['decoder.token_emb.weight'] = shard['weight']

    # --- Decoder wrapper (post_emb_norm, to_logits) ---
    dec_key = '_'.join(decoder_id)
    shard = load_shard(f'{p}_decoder_wrapper_{dec_key}.pt')
    if 'post_emb_norm.weight' in shard:
        sd['decoder.post_emb_norm.weight'] = shard['post_emb_norm.weight']
    if 'to_logits.weight' in shard:
        sd['decoder.to_logits.weight'] = shard['to_logits.weight']

    # --- Decoder stacks ---
    for stack_idx, xcoder_id in enumerate(decoder_id):
        shard = load_shard(f'{p}_decoder_{stack_idx}_{xcoder_id}.pt')
        for k, v in shard.items():
            sd[f'decoder.stacks.{stack_idx}.{k}'] = v

    return sd


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(ckpt_dir: str, prefix: str, src: str, tgt: str, output_dir: str):
    frame_path = os.path.join(ckpt_dir, f'{prefix}_frame.pt')
    frame = _load(frame_path)
    opts = frame['opts']
    vocab = frame['vocab']

    # vocab keys are tuples: ('src', lang) and ('tgt', lang)
    src_vocab_obj = vocab.get(('src', src)) or next(
        v for k, v in vocab.items() if k[0] == 'src'
    )
    tgt_vocab_obj = vocab.get(('tgt', tgt)) or next(
        v for k, v in vocab.items() if k[0] == 'tgt'
    )
    src_vocab_size = len(src_vocab_obj)
    tgt_vocab_size = len(tgt_vocab_obj)
    print(f"Vocab sizes: src={src_vocab_size}, tgt={tgt_vocab_size}")

    src_specials = src_vocab_obj.specials
    bos_str = DefaultTokens.BOS
    eos_str = DefaultTokens.EOS
    bos_id = src_specials.get(bos_str, 2)
    eos_id = src_specials.get(eos_str, 0)
    pad_id = src_specials.get(DefaultTokens.PAD, 1)
    print(f"Token IDs: BOS={bos_str!r}={bos_id}, EOS={eos_str!r}={eos_id}, PAD={pad_id}")

    try:
        encoder_id, decoder_id = _task_xcoder_ids(opts, src, tgt)
    except ValueError:
        encoder_id, decoder_id = _infer_xcoder_ids_from_shards(ckpt_dir, prefix, src, tgt)
    print(f"Task components: encoder={encoder_id}, decoder={decoder_id}")

    config = config_from_opts(
        opts,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        encoder_sharing_groups=encoder_id,
        decoder_sharing_groups=decoder_id,
    )
    enc_stack_sizes = config.enc_layers
    dec_stack_sizes = config.dec_layers
    print(
        f"Config: {config.model_dim}d, {config.heads}h, "
        f"enc_layers={enc_stack_sizes}, dec_layers={dec_stack_sizes}, "
        f"ff_mult={config.ff_mult}, ff_swiglu={config.ff_swiglu}, "
        f"rotary={config.rotary_pos_emb}, post_emb_norm={config.post_emb_norm}"
    )

    print("Assembling state dict from shards...")
    hf_sd = assemble_state_dict(ckpt_dir, prefix, src, tgt, encoder_id, decoder_id)
    print(f"  Assembled {len(hf_sd)} tensors")

    print("Building HF model...")
    model = MammothForConditionalGeneration(config)

    model_sd = model.state_dict()
    missing = [k for k in model_sd if k not in hf_sd]
    unexpected = [k for k in hf_sd if k not in model_sd]
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model.load_state_dict(hf_sd, strict=not (missing or unexpected))
    print("Weights loaded.")

    os.makedirs(output_dir, exist_ok=True)
    config.auto_map = {
        "AutoConfig": "configuration_mammoth.MammothConfig",
        "AutoModelForSeq2SeqLM": "modeling_mammoth.MammothForConditionalGeneration",
    }
    model.save_pretrained(output_dir)

    def _save_tokenizer(vocab_obj, subdir: str, label: str, add_bos_eos: bool = False):
        specials = vocab_obj.specials
        _rev = {v: k for k, v in specials.items()}
        tok = PreTrainedTokenizerFast(
            tokenizer_object=vocab_obj.tokenizer,
            bos_token=_rev.get(config.bos_token_id, bos_str),
            eos_token=_rev.get(config.eos_token_id, eos_str),
            unk_token=_rev.get(getattr(config, 'unk_token_id', 3), '<unk>'),
            pad_token=_rev.get(config.pad_token_id, DefaultTokens.PAD),
        )
        if add_bos_eos:
            from tokenizers.processors import TemplateProcessing
            tok._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos_str}:0 $A:0 {eos_str}:0",
                special_tokens=[(bos_str, bos_id), (eos_str, eos_id)],
            )
        save_path = os.path.join(output_dir, subdir)
        os.makedirs(save_path, exist_ok=True)
        tok.save_pretrained(save_path)
        print(f"{label} tokenizer saved → {subdir}/ ({len(tok)} tokens)"
              + (" [with BOS/EOS post-processor]" if add_bos_eos else ""))

    _save_tokenizer(src_vocab_obj, config.src_tokenizer_dir, 'src', add_bos_eos=True)
    _save_tokenizer(tgt_vocab_obj, config.tgt_tokenizer_dir, 'tgt')

    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("configuration_mammoth.py", "modeling_mammoth.py", "native_transformer.py"):
        shutil.copy(os.path.join(here, fname), os.path.join(output_dir, fname))

    print(f"Saved to {output_dir}")


def _discover_tasks(opts) -> list[tuple[str, str]]:
    """Return [(src, tgt), ...] for every task in opts.tasks, preserving definition order."""
    tasks = getattr(opts, 'tasks', None) or {}
    pairs = []
    for corpus_opts in tasks.values():
        src_tgt = corpus_opts.get('src_tgt', '')
        src, _, tgt = src_tgt.partition('-')
        if src and tgt:
            pairs.append((src, tgt))
    return pairs


def convert_multi_task_artifact(
    ckpt_dir: str,
    output_dir: str,
    step: int | None = None,
) -> None:
    """Convert all tasks from a multi-task checkpoint into a single bundled directory.

    Each task is converted via convert() into a temp dir, then the weight files are
    renamed to {task}.safetensors and collected into output_dir alongside shared code
    files and a manifest config.json.
    """
    import json
    import tempfile

    prefix = resolve_prefix(ckpt_dir, step)
    frame = _load(os.path.join(ckpt_dir, f'{prefix}_frame.pt'))
    all_pairs = _discover_tasks(frame['opts'])
    if not all_pairs:
        raise ValueError(f"No tasks found in checkpoint opts at {ckpt_dir}")

    os.makedirs(output_dir, exist_ok=True)
    task_manifest = {}

    with tempfile.TemporaryDirectory(prefix="mammoth_convert_") as tmp:
        for src, tgt in all_pairs:
            task_name = f"{src}-{tgt}"
            task_dir = os.path.join(tmp, task_name)
            print(f"\n=== Converting task {task_name} ===")
            convert(ckpt_dir, prefix, src, tgt, task_dir)

            src_sf = os.path.join(task_dir, "model.safetensors")
            dst_sf = os.path.join(output_dir, f"{task_name}.safetensors")
            if os.path.exists(src_sf):
                shutil.move(src_sf, dst_sf)
            else:
                src_bin = os.path.join(task_dir, "pytorch_model.bin")
                if os.path.exists(src_bin):
                    shutil.move(src_bin, os.path.join(output_dir, f"{task_name}.bin"))

            task_config_path = os.path.join(task_dir, "config.json")
            with open(task_config_path) as f:
                task_config = json.load(f)

            task_manifest[task_name] = {
                **task_config,
                "_src": src,
                "_tgt": tgt,
            }

            for tok_side in ("src_tokenizer", "tgt_tokenizer"):
                tok_subdir = task_config.get(f"{tok_side}_dir", tok_side)
                tok_src = os.path.join(task_dir, tok_subdir)
                tok_dst = os.path.join(output_dir, f"{task_name}_{tok_side}")
                if os.path.isdir(tok_src) and not os.path.exists(tok_dst):
                    shutil.copytree(tok_src, tok_dst)

    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("configuration_mammoth.py", "modeling_mammoth.py",
                  "native_transformer.py", "mammoth_hub.py"):
        shutil.copy(os.path.join(here, fname), os.path.join(output_dir, fname))

    manifest = {
        "model_type": "mammoth_hub",
        "tasks": task_manifest,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBundled {len(task_manifest)} tasks into {output_dir}")
    for t in task_manifest:
        print(f"  {t}")


def convert_checkpoint(
    ckpt_dir: str,
    output_dir: str,
    task: str | None = None,
    step: int | None = None,
) -> None:
    """Discover tasks from the checkpoint's stored opts and convert each one.

    - task=None on a multi-task checkpoint  → convert every task into output_dir/<src>-<tgt>/
    - task=None on a single-task checkpoint → convert flat into output_dir/
    - task="src-tgt"                        → convert only that task; flat layout
    """
    prefix = resolve_prefix(ckpt_dir, step)
    frame = _load(os.path.join(ckpt_dir, f'{prefix}_frame.pt'))
    all_pairs = _discover_tasks(frame['opts'])
    if not all_pairs:
        raise ValueError(
            f"No tasks found in checkpoint opts at {ckpt_dir}. "
            "Checkpoint must contain opts.tasks with at least one src_tgt entry."
        )

    if task is not None:
        src, _, tgt = task.partition('-')
        if not src or not tgt:
            raise ValueError(f"--task must be of the form 'SRC-TGT', got: {task!r}")
        if (src, tgt) not in all_pairs:
            available = [f"{s}-{t}" for s, t in all_pairs]
            raise ValueError(
                f"Task {task!r} not found in checkpoint. Available tasks: {available}"
            )
        pairs = [(src, tgt)]
    else:
        pairs = all_pairs

    use_subdirs = len(all_pairs) > 1
    for src, tgt in pairs:
        out = os.path.join(output_dir, f"{src}-{tgt}") if use_subdirs else output_dir
        print(f"\n=== Converting task {src}-{tgt} → {out} ===")
        convert(ckpt_dir, prefix, src, tgt, out)


def main():
    parser = argparse.ArgumentParser(description="Convert Mammoth checkpoint to HuggingFace format")
    parser.add_argument('--checkpoint-dir', required=True,
                        help="Directory containing Mammoth shard files")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for HF model(s). For multi-task checkpoints, "
                             "one subdirectory per task is created (e.g. eng-spa/, eng-fra/).")
    parser.add_argument('--task', default=None,
                        help="Convert only this src-tgt pair (e.g. 'eng-spa'). "
                             "Default: convert every task found in the checkpoint.")
    parser.add_argument('--step', type=int, default=None,
                        help="Load a specific checkpoint step (e.g. --step 500). "
                             "Default: best checkpoint, or last step if no best exists.")
    parser.add_argument('--single-artifact', action='store_true',
                        help="Bundle all tasks into a single HF directory with per-task shards.")

    args = parser.parse_args()

    if args.single_artifact:
        convert_multi_task_artifact(
            ckpt_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            step=args.step,
        )
    else:
        convert_checkpoint(
            ckpt_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            task=args.task,
            step=args.step,
        )


if __name__ == '__main__':
    main()
