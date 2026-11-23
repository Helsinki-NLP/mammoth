"""TranslationEngine: text in, text out, for the tasks loaded in a ComponentPool.

Each input string is one segment: nothing is split into sentences. Inputs are tokenized
with the task's own source tokenizer, sorted by length and cut into batches (so a batch
holds segments of similar length and little padding), generated with beam search, decoded
with the task's target tokenizer and returned in the input order.

Generation runs one call at a time: on a laptop CPU, parallel calls only compete for the
same cores. Unloading a task while it translates is safe, since a running call holds its
own reference to the task's model.
"""

import math
import threading
from dataclasses import dataclass
from typing import Optional, Union

import torch

from mammoth.serving.pool import ComponentPool, LoadedTask


@dataclass
class GenerationSettings:
    num_beams: int = 4
    max_new_tokens: int = 512  # hard cap per segment
    # Per segment the limit is also max_length_factor × its source length (in tokens) +
    # max_length_slack, so a short input cannot run on to max_new_tokens.
    max_length_factor: float = 2.0
    max_length_slack: int = 10
    max_batch_size: int = 32  # segments per batch
    max_batch_tokens: int = 4096  # segments × longest source length (padded) per batch

    def updated(self, **overrides) -> "GenerationSettings":
        values = {**self.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
        settings = GenerationSettings(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        for name in ("num_beams", "max_new_tokens", "max_batch_size", "max_batch_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.max_length_factor <= 0 or self.max_length_slack < 0:
            raise ValueError("max_length_factor must be positive and max_length_slack non-negative")


def make_batches(lengths: list[int], max_batch_size: int, max_batch_tokens: int) -> list[list[int]]:
    """Group segment indices into batches of similar length.

    Longest first, so the first batch shows right away whether the largest one fits in memory.
    A batch costs (segments × longest segment) tokens once padded; a single segment longer
    than max_batch_tokens still gets a batch of its own.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, batch = [], []
    for i in order:
        # Sorted longest first, so the batch's longest segment is its first one.
        width = lengths[batch[0]] if batch else lengths[i]
        if batch and (len(batch) == max_batch_size or (len(batch) + 1) * width > max_batch_tokens):
            batches.append(batch)
            batch = []
        batch.append(i)
    if batch:
        batches.append(batch)
    return batches


class TranslationEngine:
    def __init__(self, pool: ComponentPool, settings: Optional[GenerationSettings] = None):
        self.pool = pool
        self.settings = settings or GenerationSettings()
        self.settings.validate()
        self._generate_lock = threading.Lock()

    def translate(self, task: str, text: Union[str, list[str]], **overrides) -> Union[str, list[str]]:
        """Translate one string or a list of strings with loaded task "src-tgt".

        overrides: any GenerationSettings field for this call only (None = keep default).
        Returns a string for a string input and a list for a list input. Empty or
        whitespace-only segments translate to "".
        """
        settings = self.settings.updated(**overrides)
        loaded = self.pool.tasks.get(task)
        if loaded is None:
            raise KeyError(f"Task {task!r} is not loaded (loaded: {sorted(self.pool.tasks)})")
        if isinstance(text, str):
            return self._translate(loaded, [text], settings)[0]
        return self._translate(loaded, list(text), settings)

    def _translate(self, loaded: LoadedTask, segments: list[str], settings: GenerationSettings) -> list[str]:
        out = [""] * len(segments)
        todo = [i for i, s in enumerate(segments) if s.strip()]
        if not todo:
            return out
        ids = loaded.src_tokenizer([segments[i] for i in todo], add_special_tokens=True)["input_ids"]
        for batch in make_batches([len(x) for x in ids], settings.max_batch_size, settings.max_batch_tokens):
            hyps = self._generate(loaded, [ids[j] for j in batch], settings)
            for j, hyp in zip(batch, hyps):
                out[todo[j]] = hyp
        return out

    def _generate(self, loaded: LoadedTask, ids: list[list[int]], settings: GenerationSettings) -> list[str]:
        model = loaded.model
        longest = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), longest), loaded.src_tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(ids), longest), dtype=torch.long)
        for row, x in enumerate(ids):
            input_ids[row, :len(x)] = torch.tensor(x)
            attention_mask[row, :len(x)] = 1
        max_new = min(settings.max_new_tokens,
                      math.ceil(settings.max_length_factor * longest) + settings.max_length_slack)
        gen_kwargs = dict(num_beams=settings.num_beams, max_new_tokens=max_new)
        if settings.num_beams > 1:
            gen_kwargs["early_stopping"] = True
        with self._generate_lock, torch.inference_mode():
            output_ids = model.generate(input_ids=input_ids.to(model.device),
                                        attention_mask=attention_mask.to(model.device), **gen_kwargs)
        return loaded.tgt_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
