"""The truth key: decides whether a claim may be learned.

oracle  ground truth (upper bound)
noisy   ground truth flipped with prob 1-accuracy, deterministic per claim (systematic blind spots)
hf      a separate instruction-tuned LLM judge, read from its True/False next-token logits
ollama  a local Ollama model as the judge
gemini  Google Gemini API as the judge (key from GEMINI_API_KEY)
"""
from __future__ import annotations

import hashlib

import torch

SYSTEM = ("You are a strict fact-checker. Decide whether the claim is true. "
          "If a trusted source is given, it is correct. Answer with one word: True or False.")


def _user_message(claim, evidence: str | None) -> str:
    src = f"Trusted source: {evidence}\n" if evidence else ""
    return f"{src}Claim: {claim.sentence}\nIs the claim true?"


class OracleVerifier:
    name = "oracle"

    def judge(self, claims, evidence) -> list[bool]:
        return [c.is_true for c in claims]


class NoisyVerifier:
    def __init__(self, accuracy: float, seed: int):
        self.accuracy, self.seed = accuracy, seed
        self.name = f"noisy{accuracy}"

    def judge(self, claims, evidence) -> list[bool]:
        out = []
        for c in claims:
            h = hashlib.sha256(f"{self.seed}|{c.uid}|{c.target}".encode()).digest()
            correct = int.from_bytes(h[:8], "big") / 2**64 < self.accuracy
            out.append(c.is_true if correct else not c.is_true)
        return out


class _Cached:
    """Each distinct claim is judged once per run (repeats in the stream hit the cache)."""

    def __init__(self):
        self.cache: dict[tuple[str, str], bool] = {}

    def judge(self, claims, evidence) -> list[bool]:
        todo = [c for c in dict.fromkeys(claims) if (c.uid, c.target) not in self.cache]
        if todo:
            msgs = [_user_message(c, evidence.get((c.subject, c.relation))) for c in todo]
            for c, v in zip(todo, self._judge_texts(msgs)):
                self.cache[(c.uid, c.target)] = v
        return [self.cache[(c.uid, c.target)] for c in claims]

    def _judge_texts(self, msgs: list[str]) -> list[bool]:
        raise NotImplementedError


class HFJudge(_Cached):
    def __init__(self, model_name: str, device, dtype, batch_size: int = 16):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = f"hf:{model_name}"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        self.device, self.bs = device, batch_size
        self.true_id = self.tok.encode("True", add_special_tokens=False)[0]
        self.false_id = self.tok.encode("False", add_special_tokens=False)[0]

    @torch.no_grad()
    def _judge_texts(self, msgs):
        out = []
        for i in range(0, len(msgs), self.bs):
            texts = [self.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": m}],
                tokenize=False, add_generation_prompt=True) for m in msgs[i:i + self.bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            last = self.model(**enc).logits[:, -1]
            out += (last[:, self.true_id] > last[:, self.false_id]).tolist()
        return out


class OllamaJudge(_Cached):
    def __init__(self, model: str, url: str):
        super().__init__()
        self.name, self.model, self.url = f"ollama:{model}", model, url.rstrip("/")
        self.unparsed = 0  # answers with neither 'true' nor 'false' (counted as False)

    @staticmethod
    def parse(text: str) -> bool | None:
        """First standalone true/false word, ignoring markdown and punctuation ('**True**', 'False.')."""
        import re
        m = re.search(r"\b(true|false)\b", text.lower())
        return None if m is None else m.group(1) == "true"

    def _judge_texts(self, msgs):
        import time
        import requests
        out = []
        for m in msgs:
            for attempt in range(4):
                try:
                    r = requests.post(f"{self.url}/api/chat", timeout=300, json={
                        "model": self.model, "stream": False, "options": {"temperature": 0, "num_predict": 10},
                        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": m}]})
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            v = self.parse(r.json()["message"]["content"])
            if v is None:
                self.unparsed += 1
            out.append(bool(v))
        return out


class GeminiJudge(_Cached):
    """Google Gemini via REST. The key is read from the GEMINI_API_KEY environment variable."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str):
        super().__init__()
        import os
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise RuntimeError("set GEMINI_API_KEY in your shell:  export GEMINI_API_KEY=...")
        self.name, self.model = f"gemini:{model}", model
        self.empty = 0  # answers with no text (counted as False)

    def _ask(self, m: str) -> bool:
        import time
        import requests
        body = {"system_instruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": m}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 20,
                                     "thinkingConfig": {"thinkingBudget": 0}}}
        for attempt in range(6):
            r = requests.post(self.URL.format(model=self.model), json=body, timeout=60,
                              headers={"x-goog-api-key": self.key})
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip().lower()
            if not text:
                self.empty += 1
            return text.startswith("true")
        r.raise_for_status()
        return False

    def _judge_texts(self, msgs):
        return [self._ask(m) for m in msgs]


def make_verifier(cfg, seed: int, device, dtype):
    v = cfg.verifier
    if v.kind == "oracle":
        return OracleVerifier()
    if v.kind == "noisy":
        return NoisyVerifier(v.accuracy, seed)
    if v.kind == "hf":
        return HFJudge(v.hf_model, device, dtype, v.hf_batch)
    if v.kind == "ollama":
        return OllamaJudge(v.ollama_model, v.ollama_url)
    if v.kind == "gemini":
        return GeminiJudge(v.gemini_model)
    raise ValueError(f"unknown verifier.kind: {v.kind}")
