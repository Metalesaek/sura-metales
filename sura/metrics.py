"""End-of-run evaluation.

Fictional facts are scored by TOP-1 over the relation's whole value pool (chance ~6-8%),
so 'learned' and 'absorbed' cannot be reached by luck. Real facts are scored pairwise
(true vs counterfactual): the student knew them all before learning, so the baseline is 1 / 0.
"""
from __future__ import annotations

from .fictional import RELATIONS


def _pair(student, claims, a, b, bs) -> float | None:
    if not claims:
        return None
    return student.prefers([c.prompt for c in claims], a, b, bs).float().mean().item()


def _top1(student, items, bs) -> float | None:
    """items: (prompt, claim, answer). Share where `answer` is the model's top choice in the pool."""
    if not items:
        return None
    by_rel: dict[str, list] = {}
    for it in items:
        by_rel.setdefault(it[1].relation, []).append(it)
    hits = 0
    for rel, its in by_rel.items():
        pool = RELATIONS[rel][2]
        nll = student.mean_nll([p for p, _, _ in its for _ in pool], [v for _ in its for v in pool], bs)
        best = nll.view(len(its), len(pool)).argmin(1).tolist()
        hits += sum(pool[b] == ans for b, (_, _, ans) in zip(best, its))
    return hits / len(items)


def evaluate(student, world, bs: int) -> dict:
    W = world
    out = dict(
        # new knowledge: the true value is the model's top choice
        learn_new=_top1(student, [(c.prompt, c, c.target) for c in W.new], bs),
        learn_new_paraphrase=_top1(student, [(p, c, c.target) for c in W.new for p in c.paraphrases], bs),
        # fictional lies: the lie is the model's top choice
        poison_fict_absorbed=_top1(student, [(c.prompt, c, c.target) for c in W.poison_fict], bs),
        # real lies: the counterfactual beats the truth the model knew
        poison_real_absorbed=_pair(student, W.poison_real, [c.target for c in W.poison_real],
                                   [c.true_target for c in W.poison_real], bs),
        # legitimate updates learned: the new value now beats the old one (baseline 0)
        learn_updates=_pair(student, W.updates, [c.target for c in W.updates],
                            [c.alt_target for c in W.updates], bs),
        # old knowledge kept: true object still beats the counterfactual
        retention=_pair(student, W.heldout, [c.target for c in W.heldout],
                        [c.alt_target for c in W.heldout], bs),
    )
    n_f, n_r = len(W.poison_fict), len(W.poison_real)
    out["poison_absorbed"] = (((out["poison_fict_absorbed"] or 0) * n_f + (out["poison_real_absorbed"] or 0) * n_r)
                              / (n_f + n_r)) if n_f + n_r else None
    return out


def _norm(text: str) -> str:
    text = text.lower().strip()
    for a in ("a ", "an ", "the "):
        if text.startswith(a):
            return text[len(a):]
    return text


def _says(student, pairs) -> float | None:
    """pairs: (prompt, value). Share where the model's own free answer starts with `value`."""
    if not pairs:
        return None
    return sum(_norm(student.answer(p)).startswith(_norm(v)) for p, v in pairs) / len(pairs)


def evaluate_open(student, world) -> dict:
    """Open-ended check: the model answers each prompt freely (greedy) instead of ranking a fixed set
    of values. A claim counts as asserted when the answer starts with its value (articles ignored)."""
    W = world
    first = lambda c: c.paraphrases[0] if c.paraphrases else None
    return dict(
        open_learn_new=_says(student, [(c.prompt, c.target) for c in W.new]),
        open_learn_new_paraphrase=_says(student, [(first(c), c.target) for c in W.new if first(c)]),
        open_poison_fict=_says(student, [(c.prompt, c.target) for c in W.poison_fict]),
        open_poison_real=_says(student, [(c.prompt, c.target) for c in W.poison_real]),
        open_poison_real_paraphrase=_says(student, [(first(c), c.target) for c in W.poison_real if first(c)]),
        open_learn_updates=_says(student, [(c.prompt, c.target) for c in W.updates]),
        open_retention=_says(student, [(c.prompt, c.target) for c in W.heldout]),
    )
