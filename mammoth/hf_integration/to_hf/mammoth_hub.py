"""MammothHub: thin wrapper to load a specific task from a bundled multi-task HF repo.

Usage:
    model = MammothHub.from_pretrained("path/to/bundle", task="eng-spa")
    output = model.generate(input_ids=src, max_length=50)

The wrapper reads the manifest config.json, loads the per-task safetensors,
and returns a standard MammothForConditionalGeneration model ready for inference.
"""

import json
import os
import shutil
import tempfile
from typing import Optional


class MammothHub:
    """Thin wrapper that loads a single task from a bundled multi-task artifact."""

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        task: str,
        device: Optional[str] = None,
    ):
        """Load a specific task model from a bundled multi-task directory.

        Args:
            model_id: Path to the bundled directory (local path or HF repo id).
            task: Task name, e.g. "eng-spa".
            device: Device to load the model on (default: cpu).

        Returns:
            A MammothForConditionalGeneration model ready for inference.
        """
        # TODO: support HF Hub repo ids (download via snapshot_download)
        repo_path = model_id
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Not a directory: {repo_path}")

        # Read manifest
        manifest_path = os.path.join(repo_path, "config.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        tasks = manifest.get("tasks", {})
        if task not in tasks:
            raise ValueError(
                f"Task {task!r} not found. Available tasks: {list(tasks.keys())}"
            )

        task_config = tasks[task]

        # Find the per-task safetensors file
        weight_file = os.path.join(repo_path, f"{task}.safetensors")
        if not os.path.exists(weight_file):
            weight_file = os.path.join(repo_path, f"{task}.bin")
        if not os.path.exists(weight_file):
            raise FileNotFoundError(
                f"No weight file for task {task!r} in {repo_path}"
            )

        # Create a temp dir with symlinks so from_pretrained can load it
        # as a standard single-task HF model.
        tmp_dir = tempfile.mkdtemp(prefix="mammoth_hub_")
        try:
            # Write per-task config.json
            config_dict = dict(task_config)
            config_dict["model_type"] = "mammoth"
            config_dict["auto_map"] = {
                "AutoConfig": "configuration_mammoth.MammothConfig",
                "AutoModelForSeq2SeqLM": "modeling_mammoth.MammothForConditionalGeneration",
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(config_dict, f)

            # Symlink weight file with standard HF name
            std_name = "model.safetensors" if weight_file.endswith(".safetensors") \
                else "pytorch_model.bin"
            os.symlink(weight_file, os.path.join(tmp_dir, std_name))

            # Symlink code files
            for fname in ("configuration_mammoth.py", "modeling_mammoth.py",
                          "x_transformers.py", "attend.py", "autoregressive_wrapper.py"):
                src = os.path.join(repo_path, fname)
                if os.path.exists(src):
                    os.symlink(src, os.path.join(tmp_dir, fname))

            # Symlink tokenizers
            for tok_side in ("src_tokenizer", "tgt_tokenizer"):
                tok_src = os.path.join(repo_path, f"{task}_{tok_side}")
                if os.path.isdir(tok_src):
                    os.symlink(tok_src, os.path.join(tmp_dir, tok_side))

            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(
                tmp_dir, trust_remote_code=True,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if device:
            model = model.to(device)
        model.eval()
        return model
