"""What does the student believe BEFORE learning? Basis for choosing the prior-conflict rule.

  python diagnose_prior.py                  # writes results/prior_diag_<model>_seed<N>.csv and a summary

For a sample of each claim kind it records:
  next_tok / p_next   the model's most likely next token and its probability (the current rule)
  own_answer          what the model answers by itself (greedy, a few words)
  nll_own             mean NLL per token of its own answer (low = confident)
  nll_claim           mean NLL per token of the claim's answer
  gap                 nll_claim - nll_own (high = the claim fights a belief the model holds)
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import torch

from sura.config import load_config
from sura.data import build_world, load_counterfact, screen_known
from sura.student import Student, resolve_device, resolve_dtype


@torch.no_grad()
def own_answer(student, prompt: str, max_new_tokens: int = 6) -> str:
    ids, att, _ = student.encode([prompt], None)
    out = student.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=student.tok.pad_token_id)
    text = student.tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
    text = re.split(r"[.,;:\n(]", text.strip())[0]
    return " ".join(text.split()[:3]) or "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--per_kind", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0,
                    help="world seed; use one NOT in your experiments (e.g. 7) to choose the threshold fairly")
    ap.add_argument("--four_conditions", action="store_true",
                    help="also score true claims about known facts (and updates if data.n_updates > 0)")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = resolve_device(cfg.device)
    student = Student(cfg, device, resolve_dtype(cfg.dtype, device))
    records = load_counterfact(cfg.data.counterfact_split, cfg.data.n_counterfact)
    slug = cfg.student.model_name.replace("/", "_")
    cache = Path(cfg.cache_dir) / f"known_{slug}_{cfg.data.counterfact_split}_{cfg.data.n_counterfact}.json"
    world = build_world(cfg, screen_known(student, records, cache, cfg.eval.batch_size), 0.5, a.seed)

    rng = random.Random(a.seed)
    sample = []
    for kind, claims in (("new", world.new), ("poison_fict", world.poison_fict), ("poison_real", world.poison_real)):
        sample += [(kind, c) for c in rng.sample(claims, min(a.per_kind, len(claims)))]
    if a.four_conditions:  # appended after the original kinds, so their sample is unchanged
        for kind, claims in (("known_true", world.heldout), ("update", world.updates)):
            sample += [(kind, c) for c in rng.sample(claims, min(a.per_kind, len(claims)))]

    bs = cfg.eval.batch_size
    prompts = [c.prompt for _, c in sample]
    rows = []
    with torch.no_grad():
        for (kind, c) in sample:
            ids, att, _ = student.encode([c.prompt], None)
            probs = torch.softmax(student.model(input_ids=ids, attention_mask=att).logits[0, -1].float(), -1)
            p, t = probs.max(-1)
            rows.append(dict(kind=kind, prompt=c.prompt, claim=c.target, truth=c.true_target,
                             next_tok=student.tok.decode([t.item()]), p_next=round(p.item(), 3),
                             own_answer=own_answer(student, c.prompt)))
    nll_claim = student.mean_nll(prompts, [r["claim"] for r in rows], bs, adapter=False)
    nll_own = student.mean_nll(prompts, [r["own_answer"] for r in rows], bs, adapter=False)
    nll_truth = student.mean_nll(prompts, [r["truth"] for r in rows], bs, adapter=False)
    nll_best = student.claim_nll(prompts, [r["claim"] for r in rows], bs, articles=True)
    for r, a1, a2, a3, a4 in zip(rows, nll_claim.tolist(), nll_own.tolist(), nll_truth.tolist(), nll_best.tolist()):
        r.update(nll_claim=round(a1, 3), nll_own=round(a2, 3), nll_truth=round(a3, 3), gap=round(a1 - a2, 3),
                 gap_articles=round(a4 - a2, 3))

    tag = f"{cfg.student.model_name.split('/')[-1]}_seed{a.seed}"  # one file per model and world
    if a.four_conditions:
        tag += "_4cond"
    out = Path(cfg.output_dir) / f"prior_diag_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'kind':<12}{'n':>4}{'p_next':>9}{'nll_own':>9}{'nll_claim':>11}{'gap':>7}")
    for kind in ("new", "poison_fict", "poison_real"):
        rs = [r for r in rows if r["kind"] == kind]
        if rs:
            m = lambda k: sum(r[k] for r in rs) / len(rs)
            print(f"{kind:<12}{len(rs):>4}{m('p_next'):>9.3f}{m('nll_own'):>9.3f}{m('nll_claim'):>11.3f}{m('gap'):>7.3f}")
    def auc(col, pos_kind, neg_kind="new"):
        pos = [r[col] for r in rows if r["kind"] == pos_kind]
        neg = [r[col] for r in rows if r["kind"] == neg_kind]
        if not pos or not neg:
            return float("nan")
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

    print("\nseparation from true new facts (AUC, 0.5 = guessing, 1 = perfect):")
    for col in ("gap", "gap_articles"):
        print(f"  {col:<13} real lies {auc(col, 'poison_real'):.3f}   invented-people lies {auc(col, 'poison_fict'):.3f}")

    curve = Path(cfg.output_dir) / f"prior_curve_{tag}.csv"
    with curve.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rule", "threshold", "true_new_held", "poison_fict_held", "poison_real_held"])
        for col in ("gap", "gap_articles"):
            for t10 in range(0, 151, 5):
                t = t10 / 10
                share = lambda k: sum(r[col] >= t for r in rows if r["kind"] == k) / max(1, sum(r["kind"] == k for r in rows))
                w.writerow([col, t, round(share("new"), 4), round(share("poison_fict"), 4), round(share("poison_real"), 4)])
    print(f"trade-off curve -> {curve}")

    print("\nshare held back (rule: gap >= t)   plain claim  |  best of 'X'/'a X'/'an X'/'the X'")
    for t in (4, 5, 6, 7, 8):
        cells = []
        for col in ("gap", "gap_articles"):
            cells.append("  ".join(f"{k}={sum(r[col] >= t for r in rows if r['kind'] == k) / max(1, sum(r['kind'] == k for r in rows)):.2f}"
                                   for k in ("new", "poison_fict", "poison_real")))
        print(f"  t={t}:  {cells[0]}   |   {cells[1]}")
    print("\nexamples:")
    for kind in ("new", "poison_real"):
        for r in [r for r in rows if r["kind"] == kind][:4]:
            print(f"  [{kind}] {r['prompt']} -> claim '{r['claim']}' | model says '{r['own_answer']}' "
                  f"(next '{r['next_tok']}' p={r['p_next']})")
    print(f"\nfull table -> {out}")


if __name__ == "__main__":
    main()
