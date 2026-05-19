"""
Download a Mammoth HF model to a local directory for offline inference.

Three use-cases:

    # 1. Single-task model — download everything
    python model_downloader.py --repo-id org/my-model --local-dir ./my_model

    # 2. Multi-task bundle — download everything
    python model_downloader.py --repo-id org/my-bundle --local-dir ./my_bundle

    # 3. Multi-task bundle — download ONE task only (saves disk space / bandwidth)
    python model_downloader.py --repo-id org/my-bundle --local-dir ./eng_spa --task eng-spa

After downloading, run inference from local:

    python inference.py --model-dir ./my_model
    python inference.py --model-dir ./my_bundle --task eng-spa
    python inference.py --model-dir ./eng_spa   --task eng-spa
"""

import argparse
import json

from huggingface_hub import hf_hub_download, snapshot_download


def _is_bundle(repo_id: str) -> bool:
    """Fetch config.json from the Hub and check if it's a multi-task bundle."""
    try:
        cfg_path = hf_hub_download(repo_id, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("model_type") == "mammoth_hub"
    except Exception:
        return False


def download(repo_id: str, local_dir: str, task: str | None) -> None:
    bundle = _is_bundle(repo_id)

    if bundle and task:
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
        print(f"  python inference.py --model-dir {local_dir} --task {task_hint}")
    else:
        print(f"  python inference.py --model-dir {local_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="HF Hub repo id (e.g. org/my-model)")
    parser.add_argument("--local-dir", required=True, help="Local directory to save the model")
    parser.add_argument(
        "--task", default=None,
        help="For bundles: download only this task's files (e.g. eng-spa). "
             "Omit to download the full bundle.",
    )
    args = parser.parse_args()
    download(repo_id=args.repo_id, local_dir=args.local_dir, task=args.task)


if __name__ == "__main__":
    main()
