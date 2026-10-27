"""Build the claim stream and the evaluation sets.

Claim kinds
  new          true facts about invented people (the model cannot know them)
  poison_fict  false facts about invented people (contradict the trusted source)
  poison_real  CounterFact counterfactuals: false claims about real facts the model knows
  known        real facts the model knows (suite / anchor / heldout; never streamed)
  update       true changes to real facts the model knows (data.n_updates; off by default)
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .fictional import make_facts

MAX_POISON_RATE = 0.5  # sets are sized so every rate up to this uses the SAME clean facts


@dataclass(frozen=True)
class Claim:
    uid: str
    kind: str
    subject: str
    relation: str
    prompt: str            # text before the object
    target: str            # object this claim asserts
    true_target: str       # ground truth
    alt_target: str        # contrast object used for ranking metrics
    paraphrases: tuple[str, ...] = ()

    @property
    def is_true(self) -> bool:
        return self.target == self.true_target

    @property
    def sentence(self) -> str:
        return f"{self.prompt} {self.target}."


@dataclass
class World:
    stream: list[Claim]
    new: list[Claim]
    poison_fict: list[Claim]
    poison_real: list[Claim]
    suite: list[Claim]
    anchor: list[Claim]
    heldout: list[Claim]
    evidence: dict[tuple[str, str], str] = field(default_factory=dict)
    updates: list[Claim] = field(default_factory=list)  # true changes to known facts

    def summary(self) -> dict:
        return {k: len(getattr(self, k)) for k in
                ("stream", "new", "poison_fict", "poison_real", "updates", "suite", "anchor", "heldout")}


# ---------------------------------------------------------------- CounterFact
def load_counterfact(split: str, n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("azhx/counterfact", split=split).shuffle(seed=0)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        rw = row["requested_rewrite"]
        out.append(dict(
            case_id=int(row["case_id"]),
            relation=rw["relation_id"],
            subject=rw["subject"],
            prompt=rw["prompt"].format(rw["subject"]),
            target_true=rw["target_true"]["str"],
            target_new=rw["target_new"]["str"],
            paraphrases=list(row.get("paraphrase_prompts") or []),
        ))
    return out


def screen_known(student, records: list[dict], cache_file: Path | None, batch_size: int) -> list[dict]:
    """Keep records where the frozen student ranks the true object above the counterfactual."""
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    prompts = [r["prompt"] for r in records]
    nll_true = student.mean_nll(prompts, [r["target_true"] for r in records], batch_size, adapter=False)
    nll_new = student.mean_nll(prompts, [r["target_new"] for r in records], batch_size, adapter=False)
    known = [r for r, a, b in zip(records, nll_true.tolist(), nll_new.tolist()) if a < b]
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
    return known


def _known_claim(r: dict) -> Claim:
    return Claim(uid=f"cf:{r['case_id']}", kind="known", subject=r["subject"], relation=r["relation"],
                 prompt=r["prompt"], target=r["target_true"], true_target=r["target_true"],
                 alt_target=r["target_new"])


def _covered(subject: str, coverage: float) -> bool:
    h = int.from_bytes(hashlib.sha256(subject.encode()).digest()[:8], "big") / 2**64
    return h < coverage


# ---------------------------------------------------------------- world
def build_world(cfg, known_records: list[dict], poison_rate: float, seed: int) -> World:
    if not 0.0 <= poison_rate <= MAX_POISON_RATE:
        raise ValueError(f"poison_rate must be in [0, {MAX_POISON_RATE}]")
    d = cfg.data
    rng = random.Random(seed)
    share = d.poison_fict_share

    # Fictional facts: fixed split into "new" and a poison pool, identical for every rate.
    fict = make_facts(d.n_fictional_entities, rng)
    rng.shuffle(fict)
    n_new = int(len(fict) / (1 + share))
    new_facts, fict_pool = fict[:n_new], fict[n_new:]

    def other(value, pool):
        return rng.choice([v for v in pool if v != value])

    new = [Claim(uid=f"fict:{f['subject']}:{f['relation']}", kind="new", subject=f["subject"],
                 relation=f["relation"], prompt=f["prompt"], target=f["value"], true_target=f["value"],
                 alt_target=other(f["value"], f["pool"]), paraphrases=f["paraphrases"])
           for f in new_facts]

    # Poison counts: poison / (poison + clean) = rate.
    n_poison = round(poison_rate / (1 - poison_rate) * n_new) if poison_rate > 0 else 0
    n_pf = min(round(n_poison * share), len(fict_pool))
    n_pr = n_poison - n_pf
    pf = []
    for f in fict_pool[:n_pf]:
        lie = other(f["value"], f["pool"])
        pf.append(Claim(uid=f"fict:{f['subject']}:{f['relation']}", kind="poison_fict", subject=f["subject"],
                        relation=f["relation"], prompt=f["prompt"], target=lie, true_target=f["value"],
                        alt_target=f["value"], paraphrases=f["paraphrases"]))

    # Known real facts: disjoint heldout / suite / anchor / poison pool.
    known = list(known_records)
    random.Random(seed + 1).shuffle(known)
    need_pr = round(MAX_POISON_RATE / (1 - MAX_POISON_RATE) * n_new) - round(
        round(MAX_POISON_RATE / (1 - MAX_POISON_RATE) * n_new) * share)
    n_upd = getattr(d, "n_updates", 0)
    need = d.heldout_size + d.suite_size + d.anchor_size + need_pr + n_upd
    if len(known) < need:
        raise ValueError(f"only {len(known)} known CounterFact facts, need {need}: raise data.n_counterfact")
    i = 0
    heldout = [_known_claim(r) for r in known[i:i + d.heldout_size]]; i += d.heldout_size
    suite = [_known_claim(r) for r in known[i:i + d.suite_size]]; i += d.suite_size
    anchor = [_known_claim(r) for r in known[i:i + d.anchor_size]]; i += d.anchor_size
    pr = [Claim(uid=f"cf:{r['case_id']}", kind="poison_real", subject=r["subject"], relation=r["relation"],
                prompt=r["prompt"], target=r["target_new"], true_target=r["target_true"],
                alt_target=r["target_true"], paraphrases=tuple(r.get("paraphrases") or ()))
          for r in known[i:i + n_pr]]
    # Legitimate updates: known facts whose value has truly changed (the world moved on).
    # Taken AFTER the reserved poison pool, so all other sets are identical with or without them.
    j = i + need_pr
    upd = [Claim(uid=f"cf:{r['case_id']}", kind="update", subject=r["subject"], relation=r["relation"],
                 prompt=r["prompt"], target=r["target_new"], true_target=r["target_new"],
                 alt_target=r["target_true"], paraphrases=tuple(r.get("paraphrases") or ()))
           for r in known[j:j + n_upd]]

    stream = (new + pf + pr + upd) * d.reps
    rng.shuffle(stream)

    evidence = {}
    for f in new_facts + fict_pool:
        if _covered(f["subject"], d.evidence_coverage):
            evidence[(f["subject"], f["relation"])] = f"{f['prompt']} {f['value']}."
    for c in upd:  # a real change is reported by a trusted source; planted lies are not
        if _covered(c.subject, d.evidence_coverage):
            evidence[(c.subject, c.relation)] = c.sentence
    # Robustness of the evidence route (both default 0, which reproduces the paper's runs):
    # an attacker plants a matching source for a share of the real-world lies, and a share of the
    # sourced updates comes with an outdated source that still states the old value.
    planted = getattr(d, "planted_source_share", 0.0)
    for c in pr:
        if planted and _covered("planted:" + c.uid, planted):
            evidence[(c.subject, c.relation)] = c.sentence
    outdated = getattr(d, "outdated_source_share", 0.0)
    for c in upd:
        if outdated and (c.subject, c.relation) in evidence and _covered("outdated:" + c.uid, outdated):
            evidence[(c.subject, c.relation)] = f"{c.prompt} {c.alt_target}."
    return World(stream=stream, new=new, poison_fict=pf, poison_real=pr,
                 suite=suite, anchor=anchor, heldout=heldout, evidence=evidence, updates=upd)
