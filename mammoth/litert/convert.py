"""CLI to convert a Mammoth checkpoint to TFLite.

Usage (on Linux):
    python mammoth/litert/convert.py \\
        --model  mammoth/hf_integration/to_hf/mammoth_single_stack/ \\
        --task   mt_es-en \\
        --src    es \\
        --tgt    en \\
        --output mammoth_es_en.tflite
"""

import argparse
import sys


def _load_model(model_path: str, task_id: str, src_lang: str, tgt_lang: str):
    from mammoth.distributed import TaskSpecs, TaskQueueManager
    from mammoth.distributed.contexts import WorldContext, DeviceContextEnum
    from mammoth.translate.translator import load_model_for_translation

    task = TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        encoder_id=[src_lang],
        decoder_id=[tgt_lang],
        corpus_id=task_id,
        weight=1.0,
        introduce_at_training_step=0,
        corpus_opts={"src_tgt": f"{src_lang}-{tgt_lang}"},
        src_vocab=None,
        tgt_vocab=None,
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )

    opts = argparse.Namespace(gpu=-1, gpu_ranks=[], seed=42, log_model_structure=False)

    tqm = (
        TaskQueueManager(
            tasks=[task],
            accum_count=1,
            world_context=WorldContext(context=DeviceContextEnum.CPU, n_nodes=1, gpus_per_node=0),
            task_distribution_strategy_cls=None,
            uses_adapters=False,
        )
        .global_to_local(node_rank=0, local_rank=0, opts=opts)
    )
    tqm.create_all_distributed_components(use_attention_bridge=False)

    _, model, _ = load_model_for_translation(
        opts=opts,
        task_queue_manager=tqm,
        task=task,
        model_path=model_path,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True,
                        help="Path to Mammoth checkpoint directory (trailing slash required)")
    parser.add_argument("--task", required=True,
                        help="corpus_id baked into the export wrapper, e.g. mt_es-en")
    parser.add_argument("--src", required=True, help="Source language code, e.g. es")
    parser.add_argument("--tgt", required=True, help="Target language code, e.g. en")
    parser.add_argument("--output", required=True, help="Output .tflite path")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Fixed encoder sequence length for the traced graph (default: 128)")
    parser.add_argument("--tgt-len", type=int, default=128,
                        help="Fixed decoder sequence length for the traced graph (default: 128)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip dynamic int8 quantisation")
    args = parser.parse_args()

    if sys.platform != "linux":
        print(f"ERROR: litert-torch only supports Linux, got: {sys.platform!r}", file=sys.stderr)
        sys.exit(1)

    from mammoth.litert.converter import convert_to_tflite

    print(f"Loading model from {args.model} ...")
    model = _load_model(args.model, args.task, args.src, args.tgt)

    print(f"Converting to TFLite → {args.output}")
    convert_to_tflite(
        nmt_model=model,
        task_id=args.task,
        output_path=args.output,
        seq_len=args.seq_len,
        tgt_len=args.tgt_len,
        quantize=not args.no_quantize,
    )
    print("Done.")


if __name__ == "__main__":
    main()
