"""Additional analyses, computed from saved results only (no GPU).

  python review_analyses.py                 # reads results/, writes paper_outputs/review_analyses.md

A  What does the prior key detect?  AUC of the prior gap against simpler signals on the seed-7
   diagnostic (all three students), and lie-vs-update separation on the held-out real-judge runs.
B  Adaptive attacker that screens its lies with the open model (keeps only lies with gap < 6).
C  Paired differences between systems across seeds (same worlds), with t-intervals.
D  Operating cost per run: judge calls, claims held for review, claims trained, wall time.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd
from scipy import stats

MODELS = {"Qwen2.5-1.5B-Instruct": "Qwen 1.5B", "Qwen2.5-3B-Instruct": "Qwen 3B", "Llama-3.2-3B-Instruct": "Llama 3B"}
THRESHOLD = 6.0


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def auc_ci(pos, neg, reps=300):
    rng = random.Random(0)
    bs = sorted(auc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(reps))
    return bs[int(0.025 * reps)], bs[int(0.975 * reps) - 1]


def table(rows, cols) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return out + [""]


def claims(root: Path, folder: str) -> pd.DataFrame:
    files = sorted((root / folder).glob("claims_*.jsonl"))
    return pd.concat([pd.read_json(f, lines=True).assign(seed=int(f.stem.split("_s")[-1])) for f in files])


def part_a(root: Path, md: list[str]):
    md += ["## A. What the prior key detects", "",
           "Seed-7 diagnostic, 200 claims per type, original model. Each AUC separates known-fact lies "
           "from true new facts; a higher score means 'more likely a lie'.", "",
           "- `gap`: the prior gap (claim NLL minus the NLL of the model's own answer)",
           "- `claim NLL`: surprise of the claim under the original model",
           "- `answer confidence`: probability of the model's own first answer token; it ignores the claim", ""]
    rows = []
    for model, short in MODELS.items():
        r = list(csv.DictReader(open(root / f"prior_diag_{model}_seed7.csv", encoding="utf-8")))
        col = lambda k, c: [float(x[c]) for x in r if x["kind"] == k]
        for name, c in (("gap", "gap"), ("claim NLL", "nll_claim"), ("answer confidence", "p_next")):
            real, new, fict = col("poison_real", c), col("new", c), col("poison_fict", c)
            lo, hi = auc_ci(real, new)
            rows.append(dict(model=short, signal=name, auc_real_vs_new=auc(real, new), ci=f"{lo:.2f}-{hi:.2f}",
                             auc_invented_lies_vs_new=auc(fict, new)))
    md += table(rows, ["model", "signal", "auc_real_vs_new", "ci", "auc_invented_lies_vs_new"])

    md += ["Lies versus legitimate updates of the same kind of known fact (CounterFact counterfactuals), "
           "held-out seeds 10-14, Qwen2.5-1.5B. Both judges saw the same worlds, so claims are counted once.", ""]
    df = pd.concat([claims(root, "v4_phi4"), claims(root, "v4_llama32")]).drop_duplicates(["seed", "uid", "target"])
    g = lambda k: df[df.kind == k].prior_gap.dropna().tolist()
    lies, upd, new = g("poison_real"), g("update"), g("new")
    rows = [dict(comparison="known-fact lies vs updates", n=f"{len(lies)} / {len(upd)}", auc=auc(lies, upd),
                 ci="%.2f-%.2f" % auc_ci(lies, upd)),
            dict(comparison="updates vs true new facts", n=f"{len(upd)} / {len(new)}", auc=auc(upd, new),
                 ci="%.2f-%.2f" % auc_ci(upd, new)),
            dict(comparison="known-fact lies vs true new facts", n=f"{len(lies)} / {len(new)}", auc=auc(lies, new),
                 ci="%.2f-%.2f" % auc_ci(lies, new))]
    md += table(rows, ["comparison", "n", "auc", "ci"])
    held = lambda xs: sum(x >= THRESHOLD for x in xs) / len(xs)
    md += [f"Held back at gap >= 6: known-fact lies {held(lies):.2f}, updates {held(upd):.2f}, "
           f"true new facts {held(new):.2f}.", ""]


def part_b(root: Path, md: list[str]):
    md += ["## B. Adaptive attacker that screens its lies with the open model", "",
           "The attacker runs the original model (white-box) and submits only known-fact lies whose prior gap "
           "is below 6. Success = the lie passes both the prior key and the judge. Held-out seeds 10-14.", ""]
    rows = []
    for folder, judge in (("v4_phi4", "phi4"), ("v4_llama32", "Llama 3.2")):
        df = claims(root, folder)
        lies = df[(df.kind == "poison_real") & df.prior_gap.notna() & df.judge.notna()]
        passes = lies.prior_gap < THRESHOLD
        acc = lies.judge.astype(bool)
        screened = lies[passes]
        rows.append(dict(judge=judge, lies=len(lies), share_passing_prior=float(passes.mean()),
                         judge_accepts_any_lie=float(acc.mean()),
                         success_unscreened=float((passes & acc).mean()),
                         success_screened=float(screened.judge.astype(bool).mean()),
                         gain=f"{screened.judge.astype(bool).mean() / max((passes & acc).mean(), 1e-9):.1f}x"))
    md += table(rows, ["judge", "lies", "share_passing_prior", "judge_accepts_any_lie", "success_unscreened",
                       "success_screened", "gain"])
    md += ["`success_unscreened`: a fixed attacker, each lie must pass both keys. `success_screened`: the attacker "
           "only submits lies that pass the prior key, so only the judge stands in the way.", ""]


def part_c(root: Path, md: list[str]):
    md += ["## C. Paired differences between systems (same worlds, per seed)", "",
           "Mean difference over seeds with a 95% t-interval (n = seeds). Positive = first system higher.", ""]
    specs = [("v2_updates_cov50", 0.9, "Section 5.6, half sources, judge 0.9"),
             ("v4_phi4", None, "Section 5.8, phi4"), ("v4_llama32", None, "Section 5.8, Llama 3.2")]
    pairs = [("gate_prior", "gate_no_test"), ("gate_prior_ev", "gate_no_test"), ("gate_prior_ev", "gate_prior")]
    metrics = ["poison_real_absorbed", "learn_updates", "learn_new", "retention", "lies_per_1000"]
    rows = []
    for folder, acc, label in specs:
        df = pd.read_json(root / folder / "results.jsonl", lines=True)
        if acc is not None:
            df = df[df.verifier_accuracy.round(2) == acc]
        df["lies_per_1000"] = 1000 * df.poison_trained / df.claims_trained
        for a, b in pairs:
            wa = df[df.system == a].set_index("seed")
            wb = df[df.system == b].set_index("seed")
            seeds = wa.index.intersection(wb.index)
            for m in metrics:
                d = (wa.loc[seeds, m] - wb.loc[seeds, m]).astype(float)
                n = len(d)
                half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / n ** 0.5 if n > 1 else float("nan")
                rows.append(dict(setting=label, comparison=f"{a} - {b}", metric=m, n=n, mean=float(d.mean()),
                                 low=float(d.mean() - half), high=float(d.mean() + half)))
    md += table(rows, ["setting", "comparison", "metric", "n", "mean", "low", "high"])


def part_d(root: Path, md: list[str]):
    md += ["## D. Operating cost per run (held-out seeds 10-14, means per run)", "",
           "Judge calls count distinct claims sent to the judge. Held for review = claims the prior key held "
           "back and the evidence route did not rescue. Wall time includes judge latency for the first system "
           "of a seed only (verdicts are cached and shared), so compare it with care.", ""]
    rows = []
    for folder, label in (("v2_updates_cov50", "simulated judge 0.9"), ("v4_phi4", "phi4"), ("v4_llama32", "Llama 3.2")):
        df = pd.read_json(root / folder / "results.jsonl", lines=True)
        if folder.startswith("v2"):
            df = df[df.verifier_accuracy.round(2) == 0.9]
        for s, w in df.groupby("system"):
            held = (w.prior_held_true + w.prior_held_poison - w.prior_rescued_true - w.prior_rescued_poison).mean()
            rows.append(dict(setting=label, system=s,
                             judge_calls=float((w.verifier_tp + w.verifier_fp + w.verifier_tn + w.verifier_fn).mean()),
                             held_for_review=float(held), claims_trained=float(w.claims_trained.mean()),
                             minutes=float(w.minutes.mean())))
    md += table(rows, ["setting", "system", "judge_calls", "held_for_review", "claims_trained", "minutes"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_outputs")
    a = ap.parse_args()
    root, out = Path(a.results), Path(a.out)
    out.mkdir(exist_ok=True)
    md = ["# Additional analyses (saved results only)", ""]
    for part in (part_a, part_b, part_c, part_d):
        part(root, md)
    (out / "review_analyses.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
