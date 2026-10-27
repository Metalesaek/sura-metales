"""The student LLM: frozen pretrained base + one LoRA adapter that can be snapshotted and rolled back."""
from __future__ import annotations

import contextlib
import copy
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class Student:
    def __init__(self, cfg, device: torch.device, dtype: torch.dtype, model=None, tokenizer=None):
        s = cfg.student
        self.device = device
        self.tok = tokenizer or AutoTokenizer.from_pretrained(s.model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        base = model or AutoModelForCausalLM.from_pretrained(s.model_name, dtype=dtype)
        base.to(device)
        for p in base.parameters():
            p.requires_grad_(False)
        lora = LoraConfig(r=s.lora_r, lora_alpha=s.lora_alpha, lora_dropout=s.lora_dropout,
                          target_modules=list(s.target_modules), task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora)
        self.model.eval()
        self._initial = self.adapter_state()

    # ------------------------------------------------------------ adapter state (commit / rollback)
    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.model).items()}

    def load_adapter_state(self, state: dict[str, torch.Tensor]) -> None:
        set_peft_model_state_dict(self.model, state)

    def reset_adapter(self) -> None:
        self.load_adapter_state(self._initial)

    @contextlib.contextmanager
    def using(self, state: dict[str, torch.Tensor] | None):
        """Temporarily answer with another adapter state (e.g. the EMA slow learner)."""
        if state is None:
            yield
            return
        current = self.adapter_state()
        self.load_adapter_state(state)
        try:
            yield
        finally:
            self.load_adapter_state(current)

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def new_optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.trainable(), lr=lr, weight_decay=0.0)

    @staticmethod
    def snapshot_optimizer(opt):
        return copy.deepcopy(opt.state_dict())

    # ------------------------------------------------------------ encoding
    def encode(self, prompts: list[str], targets: list[str] | None):
        """Token ids of prompt + ' ' + target; labels cover the target tokens only."""
        ids_list, lab_list = [], []
        for i, p in enumerate(prompts):
            p_ids = self.tok(p).input_ids
            t_ids = self.tok(" " + targets[i], add_special_tokens=False).input_ids if targets else []
            ids_list.append(p_ids + t_ids)
            lab_list.append([-100] * len(p_ids) + t_ids if targets else list(p_ids))
        n = max(len(x) for x in ids_list)
        pad = self.tok.pad_token_id
        ids = torch.full((len(ids_list), n), pad, dtype=torch.long)
        att = torch.zeros((len(ids_list), n), dtype=torch.long)
        lab = torch.full((len(ids_list), n), -100, dtype=torch.long)
        for i, (x, y) in enumerate(zip(ids_list, lab_list)):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
            lab[i, :len(y)] = torch.tensor(y)
        return ids.to(self.device), att.to(self.device), lab.to(self.device)

    def _target_nll(self, ids, att, lab):
        """Per-example summed NLL and token count over labelled positions."""
        logits = self.model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        labels = lab[:, 1:]
        nll = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
        mask = (labels != -100).float()
        return (nll * mask).sum(1), mask.sum(1).clamp(min=1)

    # ------------------------------------------------------------ scoring
    @torch.no_grad()
    def mean_nll(self, prompts, targets, batch_size: int, adapter: bool = True) -> torch.Tensor:
        """Mean NLL per target token for each (prompt, target); this is the 'surprise'."""
        ctx = contextlib.nullcontext() if adapter else self.model.disable_adapter()
        out = []
        with ctx:
            for i in range(0, len(prompts), batch_size):
                s, c = self._target_nll(*self.encode(prompts[i:i + batch_size], targets[i:i + batch_size]))
                out.append((s / c).cpu())
        return torch.cat(out) if out else torch.empty(0)

    @torch.no_grad()
    def own_answer(self, prompt: str, max_new_tokens: int = 6) -> str:
        """What the ORIGINAL model answers by itself (greedy, adapter off, first few words)."""
        import re
        ids, att, _ = self.encode([prompt], None)
        with self.model.disable_adapter():
            out = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
        text = re.split(r"[.,;:\n(]", text.strip())[0]
        return " ".join(text.split()[:3]) or "?"

    @torch.no_grad()
    def answer(self, prompt: str, max_new_tokens: int = 8) -> str:
        """What the CURRENT model (adapter on) answers by itself: greedy, up to the first punctuation."""
        import re
        ids, att, _ = self.encode([prompt], None)
        out = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
        return re.split(r"[.,;:\n(]", text.strip())[0].strip()

    ARTICLES = ("", "a ", "an ", "the ")

    def claim_nll(self, prompts, targets, batch_size: int, articles: bool) -> torch.Tensor:
        """Claim NLL under the ORIGINAL model; with `articles`, the best of 'X', 'a X', 'an X',
        'the X', so grammar ('is a nurse') is not mistaken for disbelief."""
        if not articles:
            return self.mean_nll(prompts, targets, batch_size, adapter=False)
        k = len(self.ARTICLES)
        nll = self.mean_nll([p for p in prompts for _ in range(k)],
                            [a + t for t in targets for a in self.ARTICLES], batch_size, adapter=False)
        return nll.view(len(prompts), k).min(1).values

    def prior_gap(self, prompts, targets, batch_size: int, articles: bool = True) -> list[float]:
        """NLL(claim) - NLL(model's own answer), both under the ORIGINAL model.
        Large = the claim fights a belief the model held before any user taught it anything.
        Cached: it never changes during learning."""
        cache = self.__dict__.setdefault("_prior_cache", {})
        todo = [(p, t) for p, t in dict.fromkeys(zip(prompts, targets)) if (p, t, articles) not in cache]
        if todo:
            ps = [p for p, _ in todo]
            own = [self.own_answer(p) for p in ps]
            nc = self.claim_nll(ps, [t for _, t in todo], batch_size, articles)
            no = self.mean_nll(ps, own, batch_size, adapter=False)
            for (p, t), a, b in zip(todo, nc.tolist(), no.tolist()):
                cache[(p, t, articles)] = a - b
        return [cache[(p, t, articles)] for p, t in zip(prompts, targets)]

    def contradicts(self, prompts, targets, batch_size: int, gap: float, articles: bool = True) -> list[bool]:
        """Prior-conflict key: the claim is at least `gap` nats/token less likely than the
        original model's own answer. Novelty (no strong belief) -> False."""
        return [g >= gap for g in self.prior_gap(prompts, targets, batch_size, articles)]

    def prefers(self, prompts, a_targets, b_targets, batch_size: int) -> torch.Tensor:
        """Bool per item: the model finds `a` more likely than `b` (length-normalised)."""
        return self.mean_nll(prompts, a_targets, batch_size) < self.mean_nll(prompts, b_targets, batch_size)

    # ------------------------------------------------------------ training
    def train_steps(self, opt, prompts, targets, anchor_prompts, cfg_train, steps: int, rng: random.Random):
        """`steps` updates: CE on claims + kl_weight * KL(base || student) on anchor sentences."""
        self.model.train()
        idx = list(range(len(prompts)))
        losses = []
        for _ in range(steps):
            pick = rng.sample(idx, min(cfg_train.batch_size, len(idx)))
            s, c = self._target_nll(*self.encode([prompts[i] for i in pick], [targets[i] for i in pick]))
            loss = (s / c).mean()
            if cfg_train.kl_weight > 0 and anchor_prompts:
                a = rng.sample(anchor_prompts, min(cfg_train.anchor_batch, len(anchor_prompts)))
                ids, att, _ = self.encode(a, None)
                with torch.no_grad(), self.model.disable_adapter():
                    ref = F.log_softmax(self.model(input_ids=ids, attention_mask=att).logits.float(), -1)
                cur = F.log_softmax(self.model(input_ids=ids, attention_mask=att).logits.float(), -1)
                kl = F.kl_div(cur, ref, log_target=True, reduction="none").sum(-1)
                kl = (kl * att).sum() / att.sum()
                loss = loss + cfg_train.kl_weight * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable(), 1.0)
            opt.step()
            losses.append(loss.item())
        self.model.eval()
        return sum(losses) / max(len(losses), 1)
