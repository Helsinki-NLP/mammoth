"""MammothHub: thin wrapper to load a specific task from a bundled multi-task HF repo.

Quickstart (no mammoth package needed):

    pip install transformers torch huggingface_hub einops

    python - <<'EOF'
    from huggingface_hub import hf_hub_download
    import importlib.util, sys

    # Step 1: fetch just this one file from the Hub (< 5 KB)
    _path = hf_hub_download("org/my-mammoth-bundle", "mammoth_hub.py")
    _spec = importlib.util.spec_from_file_location("mammoth_hub", _path)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    MammothHub = _mod.MammothHub

    # Step 2: load the model (downloads only the requested task's weights)
    model = MammothHub.from_pretrained("org/my-mammoth-bundle", task="eng-spa")
    inputs = model.src_tokenizer(["Hello!"], return_tensors="pt")
    out    = model.generate(**inputs, num_beams=4, max_new_tokens=128)
    print(model.tgt_tokenizer.batch_decode(out, skip_special_tokens=True))
    EOF

If you already have the repo cloned / downloaded locally:

    model = MammothHub.from_pretrained("/local/path/to/bundle", task="eng-spa")

The wrapper reads config.json, downloads only the requested task's weights from
the Hub, and returns a MammothForConditionalGeneration with .src_tokenizer and
.tgt_tokenizer attached.
"""

import json
import os
import shutil
import tempfile
from typing import Optional

from transformers import PreTrainedTokenizerFast


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
            model_id: Local directory path or HF Hub repo id (e.g. "org/repo").
            task: Task name, e.g. "eng-spa".
            device: Device to load the model on (default: cpu).

        Returns:
            A MammothForConditionalGeneration model ready for inference.
        """
        if os.path.isdir(model_id):
            repo_path = model_id
        else:
            # HF Hub repo id — download only the files needed for this task so we
            # don't pull the weights for every other task in the bundle.
            from huggingface_hub import snapshot_download
            repo_path = snapshot_download(
                repo_id=model_id,
                allow_patterns=[
                    "config.json",
                    "*.py",
                    f"{task}.safetensors",
                    f"{task}.bin",
                    f"{task}_src_tokenizer/**",
                    f"{task}_tgt_tokenizer/**",
                ],
            )

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
                          "native_transformer.py",
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

        # Attach tokenizers so callers don't need to know the bundle layout.
        model.src_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(repo_path, f"{task}_src_tokenizer"))
        model.tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(repo_path, f"{task}_tgt_tokenizer"))

        return model
