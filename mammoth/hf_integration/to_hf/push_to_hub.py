#!/usr/bin/env python3
"""
Push a converted Mammoth HF model directory to the Hugging Face Hub.

Usage:
    python push_to_hub.py \
        --model-dir path/to/converted_hf_model \
        --repo-id username/my-mammoth-model \
        [--private] \
        [--token YOUR_HF_TOKEN]

The model directory must have been produced by convert_mammoth_to_hf.py.
If --token is omitted, the script uses the token stored by `huggingface-cli login`.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, login


def push(model_dir: str, repo_id: str, private: bool, token: str | None):
    if token:
        login(token=token)

    api = HfApi()

    # Create the repo if it doesn't exist yet
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/{repo_id}")

    print(f"Uploading files from {model_dir} ...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Mammoth converted model",
    )

    print(f"Done. Model available at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push a converted Mammoth model to HF Hub")
    parser.add_argument("--model-dir", required=True,
                        help="Directory produced by convert_mammoth_to_hf.py")
    parser.add_argument("--repo-id", required=True,
                        help="Hub repo id, e.g. 'username/mammoth-es-en'")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private (default: public)")
    parser.add_argument("--token",
                        help="HF access token (or set HUGGING_FACE_HUB_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not os.path.isdir(args.model_dir):
        print(f"ERROR: model-dir not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    push(args.model_dir, args.repo_id, args.private, token)


if __name__ == "__main__":
    main()
