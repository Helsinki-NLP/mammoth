"""mammoth_serve: run a local Mammoth translation server.

    mammoth_serve --model path/to/checkpoint_dir                 # training checkpoint
    mammoth_serve --model org/mammoth-modular --tasks eng-spa,deu-fra

--model takes a training checkpoint directory or any HF export (modular, --single-artifact
bundle, standalone; local directory or Hub repo id). Directions not loaded at start can be
loaded later with POST /api/tasks. API: see mammoth/serving/app.py.
"""

import argparse
import logging

import torch

from mammoth.serving import ComponentPool, GenerationSettings, TranslationEngine, open_source

logger = logging.getLogger("mammoth_serve")

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def get_parser() -> argparse.ArgumentParser:
    defaults = GenerationSettings()
    p = argparse.ArgumentParser(prog="mammoth_serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Training checkpoint directory, or HF export: local directory or Hub repo id")
    p.add_argument("--tasks", default="trained",
                   help="Directions to load at start: comma-separated 'src-tgt' list, 'trained' "
                        "(every trained task; default), 'all' (also every zero-shot direction) "
                        "or 'none'")
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to load (training checkpoints; default: best, else latest)")
    p.add_argument("--device", default="auto", help="cpu, cuda, cuda:N or mps; auto = cuda if available, else cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="fp32",
                   help="Weight dtype (default fp32; bf16 halves memory)")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for torch (default: torch's choice)")
    p.add_argument("--host", default="127.0.0.1",
                   help="Interface to listen on (default: this machine only; 0.0.0.0 for all)")
    p.add_argument("--port", type=int, default=8888)
    g = p.add_argument_group("generation defaults (requests may lower max_new_tokens or change num_beams)")
    g.add_argument("--num-beams", type=int, default=defaults.num_beams)
    g.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    g.add_argument("--max-length-factor", type=float, default=defaults.max_length_factor)
    g.add_argument("--max-batch-size", type=int, default=defaults.max_batch_size)
    g.add_argument("--max-batch-tokens", type=int, default=defaults.max_batch_tokens)
    return p


def startup_tasks(spec: str, available: dict) -> list[str]:
    """The directions named by --tasks, checked against what the model can serve."""
    if spec == "none":
        return []
    if spec == "trained":
        return sorted(t for t, info in available.items() if info["trained"])
    if spec == "all":
        return sorted(available)
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in available]
    if unknown:
        raise ValueError(f"Cannot serve {unknown}; this model serves: {', '.join(sorted(available))}")
    return tasks


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_engine(args) -> TranslationEngine:
    if args.threads:
        torch.set_num_threads(args.threads)
    settings = GenerationSettings(
        num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, max_length_factor=args.max_length_factor,
        max_batch_size=args.max_batch_size, max_batch_tokens=args.max_batch_tokens)
    source = open_source(args.model, step=args.step)
    pool = ComponentPool(source, device=resolve_device(args.device), dtype=DTYPES[args.dtype])
    engine = TranslationEngine(pool, settings)
    tasks = startup_tasks(args.tasks, pool.available_tasks())
    for task in tasks:
        logger.info("Loading %s", task)
        pool.add_task(task)
    # Shared components are the same parameter objects in every task: count each once.
    params = {id(p): p.numel() for t in pool.loaded_tasks().values() for p in t.model.parameters()}
    logger.info("Loaded %d direction(s): %d component(s), %.1fM parameters, on %s (%s); "
                "%d directions available", len(tasks), len(pool.loaded_components()),
                sum(params.values()) / 1e6, pool.device, args.dtype, len(pool.available_tasks()))
    return engine


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        import uvicorn
        from mammoth.serving.app import create_app
    except ImportError:
        parser.error("mammoth_serve needs the server extras: pip install 'mammoth-nlp[server]'")
    try:
        engine = build_engine(args)
    except ValueError as e:
        parser.error(str(e))
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
