"""
Download a Mammoth HF model to a local directory for offline inference.

Three use-cases:

    # 1. Single-task model — download everything
    python model_downloader.py --repo-id org/my-model --local-dir ./my_model

    # 2. Multi-task bundle — download everything
    python model_downloader.py --repo-id org/my-bundle --local-dir ./my_bundle

    # 3. Multi-task bundle or modular repo — download ONE task only (saves disk space /
    #    bandwidth; for a modular repo, only the components that task is assembled from)
    python model_downloader.py --repo-id org/my-bundle --local-dir ./eng_spa --task eng-spa

After downloading, run inference from local (inference.py is shipped in the artifact):

    python ./my_model/inference.py  --model-dir ./my_model
    python ./my_bundle/inference.py --model-dir ./my_bundle --task eng-spa
    python ./eng_spa/inference.py   --model-dir ./eng_spa   --task eng-spa
"""

import argparse
import json

from huggingface_hub import hf_hub_download, snapshot_download


def _model_type(repo_id: str) -> str | None:
    """Fetch config.json from the Hub and return its model_type."""
    try:
        cfg_path = hf_hub_download(repo_id, "config.json")
        with open(cfg_path) as f:
            return json.load(f).get("model_type")
    except Exception:
        return None


def _modular_file_patterns(repo_id: str):
    """modular_file_patterns from mammoth_hub.py (sibling file, dev install, or the repo itself)."""
    try:
        from mammoth_hub import modular_file_patterns
        return modular_file_patterns
    except ImportError:
        pass
    try:
        from mammoth.hf_integration.to_hf.mammoth_hub import modular_file_patterns
        return modular_file_patterns
    except ImportError:
        pass
    import importlib.util
    path = hf_hub_download(repo_id, "mammoth_hub.py")
    spec = importlib.util.spec_from_file_location("mammoth_hub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.modular_file_patterns


def download(repo_id: str, local_dir: str, task: str | None) -> None:
    model_type = _model_type(repo_id)
    bundle = model_type in ("mammoth_hub", "mammoth_modular")

    if model_type == "mammoth_modular" and task:
        # Download only the components this task is assembled from
        with open(hf_hub_download(repo_id, "config.json")) as f:
            manifest = json.load(f)
        allow_patterns = _modular_file_patterns(repo_id)(manifest, task=task)
        print(f"Modular repo — downloading the components of task '{task}' "
              f"from '{repo_id}' → '{local_dir}'")
        snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns)
    elif bundle and task:
        # Download only what's needed for this one task
        allow_patterns = [
            "config.json",
            "*.py",
            f"{task}.safetensors",
            f"{task}.bin",
            f"{task}_src_tokenizer/**",
            f"{task}_tgt_tokenizer/**",
        ]
        print(f"Bundle detected — downloading task '{task}' only from '{repo_id}' → '{local_dir}'")
        snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns)
    else:
        if bundle:
            print(f"Bundle detected — downloading all tasks from '{repo_id}' → '{local_dir}'")
        else:
            print(f"Single-task model — downloading '{repo_id}' → '{local_dir}'")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)

    print(f"\nDone. Run inference with:")
    if bundle:
        task_hint = task or "<task>"
        print(f"  python {local_dir}/inference.py --model-dir {local_dir} --task {task_hint}")
    else:
        print(f"  python {local_dir}/inference.py --model-dir {local_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="HF Hub repo id (e.g. org/my-model)")
    parser.add_argument("--local-dir", required=True, help="Local directory to save the model")
    parser.add_argument(
        "--task", default=None,
        help="For bundles and modular repos: download only this task's files "
             "(e.g. eng-spa). Omit to download the full repo.",
    )
    args = parser.parse_args()
    download(repo_id=args.repo_id, local_dir=args.local_dir, task=args.task)


if __name__ == "__main__":
    main()
