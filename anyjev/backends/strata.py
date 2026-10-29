"""Strata backend through its OpenAI-style legacy ``/v1/completions`` endpoint.

Strata (https://github.com/Niko1221/Strata) serves large MoE models such as Qwen3.8-Flash-Next on
one consumer GPU. Each prompt goes as token ids with ``max_tokens=1``, ``temperature=0``,
``logprobs=0`` and ``allowed_token_ids`` set to the labels AnyJev asked for. With
``return_tokens_as_token_ids`` the server answers with the full-vocabulary log-probability of each
named id at the position after the prompt, keyed ``"token_id:<id>"`` in
``choices[0].logprobs.top_logprobs[0]``; they are read back by id. The values are not
renormalized over the labels, so ``answer_mass`` keeps its meaning.

    strata launch                      # or: python -m serve.server ... --port 8080
    Decider(StrataBackend("http://localhost:8080", "qwen3.8-flash-next-iq3_s",
                          tokenizer_name="Qwen/Qwen3.8-Flash-Next", api_key="..."))

Needs Strata 0.2.7 or later (logprobs, from Niko1221/Strata#3200) with ``/v1/completions`` and its
logprobs (#3332 and #4056). ``--parallel`` works: a ``max_tokens=1`` read is answered from batch
slots too, and this backend never asks for more. ``tokenizer_name`` must be the model's own
Hugging Face tokenizer: Strata's pack tokenizer is in its own format, and the ids have to match
the engine's.

Repeatable reads: by default Strata moves experts between RAM and VRAM as it runs and splits a
short prompt's experts between CPU and GPU by measured speed, so the same prompt can come back a
few hundredths of a nat different (Niko1221/Strata#4059). Starting the engine with
``--adapt-swaps 0`` and ``STRATA_PREFILL_CPU_SHARE=0.5`` made every read identical.

Checked on Strata 0.1.53 (Qwen3.8-Flash-Next IQ3_S, RTX 5080) with ``scripts/strata_parity.py``:
the Hugging Face tokenizer's ids match the engine's, and with those two settings the named-id
values equal the server's top-K exactly (max_abs_diff 0.000000, twice).
This backend does not expose L2 hidden states.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import urllib.error
import urllib.request
from typing import List, Sequence

import numpy as np


class StrataBackend:
    """Next-token log-probabilities from Strata's ``/v1/completions`` with ``allowed_token_ids``."""

    def __init__(self, base_url: str, model: str, tokenizer_name: str | None = None,
                 api_key: str = "EMPTY", workers: int = 2, timeout: float = 600.0):
        from transformers import AutoTokenizer

        root = base_url.rstrip("/")
        self.url = (root if root.endswith("/v1") else root + "/v1") + "/completions"
        self.name = model
        self.api_key = api_key
        # Strata serves one request at a time; a second worker keeps the next prompt queued
        self.workers = workers
        self.timeout = timeout
        # `model` is Strata's served name; the tokenizer has to come from something loadable.
        self.source = tokenizer_name or model
        self.tokenizer = AutoTokenizer.from_pretrained(self.source)

    def _one(self, prompt: str, ids: Sequence[int]) -> np.ndarray:
        ids = [int(i) for i in ids]
        body = {
            "model": self.name,
            "prompt": self.tokenizer.encode(prompt, add_special_tokens=False),
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 0,
            "allowed_token_ids": ids,
            "return_tokens_as_token_ids": True,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                out = json.load(response)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Strata server {self.url}: {e.code} {e.read()[:500]!r}") from None

        lp = (out.get("choices") or [{}])[0].get("logprobs")
        if not lp or not lp.get("top_logprobs"):
            raise RuntimeError("Strata did not return token log-probabilities: it needs logprobs on "
                               "/v1/completions and must run without --parallel")
        top = lp["top_logprobs"][0] or {}
        got = {int(k.split(":", 1)[1]): v for k, v in top.items() if k.startswith("token_id:")}
        missing = [i for i in ids if i not in got]
        if missing:
            raise RuntimeError(
                "Strata did not return log-probabilities for every requested token id: "
                f"missing {missing}")
        return np.asarray([got[i] for i in ids], dtype=np.float64)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        with cf.ThreadPoolExecutor(self.workers) as executor:
            return list(executor.map(self._one, prompts, token_ids))
