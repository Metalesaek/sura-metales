"""Recompute every number of the natural-stream section (Tables 6 and 7 and the paired intervals in its text).
    python analyze_natural.py            # reads results/v9_wbe*, results/v6_sure* and results/v9_wbe_phi4
Paired differences are over seeds 10-14 with t-intervals; for the constructed-stream SuRe runs (v6) the two runs
in the published configuration are averaged per world first.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

R = Path("results")
SEEDS = range(10, 15)
KNOWN_SHARE = 85 / 671  # known-fact lies / stream items per pass (85 of 671)


def load(tag):
    return {(r["system"], r["seed"]): r for r in map(json.loads, (R / tag / "results.jsonl").open(encoding="utf-8"))}


def mean(D, s, m):
    return float(np.mean([D[(s, x)][m] for x in SEEDS]))


def ci(d):
    d = np.asarray(d) * 100
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.1f} [{d.mean() - h:+.1f}, {d.mean() + h:+.1f}], positive {(d > 0).sum()}/{len(d)}"


def diff(D, a, b, m):
    return ci([D[(a, x)][m] - D[(b, x)][m] for x in SEEDS])


def table(D, systems, metrics):
    print("system".ljust(16) + "".join(m[:14].rjust(15) for m in metrics))
    for s in systems:
        print(s.ljust(16) + "".join(f"{mean(D, s, m):15.3f}" for m in metrics))


MAIN = ["naive", "surprise", "random", "sure", "sure_random", "gate_prior_ev"]
COLS = ["learn_new", "open_learn_new", "learn_updates", "open_learn_updates", "poison_real_absorbed",
        "open_poison_real", "poison_fict_absorbed", "retention"]
for tag in ("v9_wbe", "v9_wbe_7b"):
    D = load(tag)
    print(f"\n== Table 6, {tag}")
    table(D, MAIN, COLS + ["before_open_retention", "before_open_learn_updates"])
    for m in ("poison_real_absorbed", "open_poison_real", "learn_new"):
        print(f"surprise - random   {m}: {diff(D, 'surprise', 'random', m)}")
    for m in ("poison_real_absorbed", "poison_fict_absorbed", "learn_new", "learn_updates", "retention"):
        print(f"gate - naive        {m}: {diff(D, 'gate_prior_ev', 'naive', m)}")
    print("buffer known-lie share (%):", {s: round(100 * mean(D, s, "buffer_poison_real_share"), 1) for s in ("sure", "sure_random")})
    print(f"adapted SuRe buffer: {diff(D, 'sure', 'sure_random', 'buffer_poison_real_share')}, "
          f"{mean(D, 'sure', 'buffer_poison_real_share') / KNOWN_SHARE:.2f}x stream share")

for tag in ("v9_wbe_sure", "v9_wbe_sure_7b"):
    D = load(tag)
    print(f"\n== Table 6, published SuRe, {tag}")
    table(D, ["naive", "sure", "sure_random"], ["learn_new", "learn_updates", "poison_real_absorbed", "poison_fict_absorbed", "retention"])
    print("buffer known-lie share (%):", {s: round(100 * mean(D, s, "buffer_poison_real_share"), 1) for s in ("sure", "sure_random")})
    print(f"buffer: {diff(D, 'sure', 'sure_random', 'buffer_poison_real_share')}, "
          f"{mean(D, 'sure', 'buffer_poison_real_share') / KNOWN_SHARE:.2f}x; "
          f"beliefs: {diff(D, 'sure', 'sure_random', 'poison_real_absorbed')}")

A, B = load("v6_sure"), load("v6_sure_rep2")
print("\n== constructed stream, published SuRe (v6, two runs per world averaged)")
for m in ("buffer_poison_real_share", "poison_real_absorbed"):
    print(m, ci([np.mean([X[("sure", x)][m] - X[("sure_random", x)][m] for X in (A, B)]) for x in SEEDS]))
print("buffer share", round(np.mean([X[("sure", x)]["buffer_poison_real_share"] for X in (A, B) for x in SEEDS]) / KNOWN_SHARE, 2),
      "x; retention per world", [round(np.mean([X[("sure", x)]["retention"] for X in (A, B)]), 2) for x in SEEDS])

D = load("v9_wbe_phi4")
print("\n== Table 7, phi4 judge")
table(D, ["gate_no_test", "gate_prior", "gate_prior_ev"], COLS + ["minutes"])
for a, b in (("gate_prior", "gate_no_test"), ("gate_prior_ev", "gate_no_test")):
    for m in ("poison_real_absorbed", "learn_new", "learn_updates"):
        print(f"{a} - {b} {m}: {diff(D, a, b, m)}")


def claims(tag):
    return [[json.loads(l) for l in (R / tag / f"claims_r0.3_s{x}.jsonl").open(encoding="utf-8")] for x in SEEDS]


print("\n== review queue: distinct claims the prior key holds back (gap >= 6) and no source rescues")
for tag in ("v9_wbe", "v9_wbe_7b", "v9_wbe_phi4"):
    q = [[c for c in C if c["prior_gap"] is not None and c["prior_gap"] >= 6.0 and not (c["has_source"] and c["is_true"])]
         for C in claims(tag)]
    n = len(claims(tag)[0])
    print(f"{tag}: {np.mean([len(x) for x in q]):.0f} of {n} ({np.mean([len(x) for x in q]) / n:.1%}), "
          f"lies among them {np.mean([sum(not c['is_true'] for c in x) / len(x) for x in q]):.0%}")

print("\n== phi4 verdicts (judged true), 5 seeds")
C = [c for x in claims("v9_wbe_phi4") for c in x]
for kind in ("new", "update", "poison_fict", "poison_real"):
    for src in (False, True):
        g = [c for c in C if c["kind"] == kind and c["has_source"] == src and c["judge"] is not None]
        if g:
            print(f"{kind:12} source={src!s:5} n={len(g):4}  {sum(c['judge'] for c in g) / len(g):.3f}")
