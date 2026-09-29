"""LLM inference engines.

`HFEngine` loads an open-source causal LM from Hugging Face, applies the
LoRA adapter produced by `training/train_lora.py` (and by default merges it
into the base weights so inference costs the same as the base model), and
exposes blocking `generate()` and incremental `stream()`.

`EchoEngine` is a deterministic, dependency-free stand-in used by unit tests
and CI (GitHub runners have no model weights). It is never used for the
numbers reported in the README.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

from app.prompts import ABSTAIN

log = logging.getLogger(__name__)


@dataclass
class Generation:
    text: str
    new_tokens: int
    seconds: float


class EchoEngine:
    """Returns the context sentence with the most word overlap. For tests only."""

    name = "echo"
    adapter = None
    loaded = True

    def generate(self, messages: list[dict], max_new_tokens: int = 96) -> Generation:
        t0 = time.perf_counter()
        user = messages[-1]["content"]
        if "Context:" not in user:
            return Generation(ABSTAIN, 9, time.perf_counter() - t0)
        context, _, question = user.partition("\n\nQuestion: ")
        q = set(re.findall(r"\w+", question.lower()))
        best, best_score = ABSTAIN, 1
        for sent in re.split(r"(?<=[.!?|])\s+|\n", context):
            score = len(q & set(re.findall(r"\w+", sent.lower())))
            if score > best_score:
                best, best_score = re.sub(r"^\[\d+\][^:]*:\s*", "", sent).strip(), score
        return Generation(best, len(best.split()), time.perf_counter() - t0)

    def stream(self, messages: list[dict], max_new_tokens: int = 96) -> Iterator[str]:
        for word in self.generate(messages, max_new_tokens).text.split(" "):
            yield word + " "


class HFEngine:
    def __init__(self, base_model: str, adapter_path: str | None = None, merge: bool = True, threads: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.name = base_model
        self.adapter = None
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)

        if adapter_path and os.path.isdir(adapter_path):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            if merge:
                model = model.merge_and_unload()
            self.adapter = adapter_path
            log.info("Loaded LoRA adapter from %s (merged=%s)", adapter_path, merge)
        elif adapter_path:
            log.warning("Adapter path %s not found - serving the BASE model", adapter_path)

        self.model = model.eval()
        self.loaded = True
        self._lock = threading.Lock()
        log.info("Model %s ready in %.1fs", base_model, time.perf_counter() - t0)

    def _encode(self, messages: list[dict]):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(prompt, return_tensors="pt")

    def _gen_kwargs(self, max_new_tokens: int) -> dict:
        return dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: deterministic answers, better for caching and evaluation
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, messages: list[dict], max_new_tokens: int = 96) -> Generation:
        inputs = self._encode(messages)
        t0 = time.perf_counter()
        with self._lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens))
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        return Generation(text, int(new.shape[0]), time.perf_counter() - t0)

    def stream(self, messages: list[dict], max_new_tokens: int = 96) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self._encode(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run():
            with self._lock, self.torch.inference_mode():
                self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens), streamer=streamer)

        threading.Thread(target=run, daemon=True).start()
        yield from streamer


def build_engine(settings) -> EchoEngine | HFEngine:
    if settings.llm_backend == "echo":
        log.warning("LLM_BACKEND=echo - using the test stand-in, not a real model")
        return EchoEngine()
    return HFEngine(settings.base_model, settings.adapter_path or None, settings.merge_adapter, settings.torch_threads)
