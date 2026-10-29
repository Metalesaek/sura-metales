"""Rebuild every table and figure of the paper from saved results, and audit the quoted numbers.

  python reproduce_paper.py                       # reads results/, writes paper_outputs/
  python reproduce_paper.py --results path/to/results

No training is run. Each section names the results folder it reads. The audit at the end
compares every number quoted in the paper with the value recomputed here and exits with
status 1 if any differs by more than its rounding tolerance.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# results folder of each experiment (first existing name wins; "mai" was the original typo)
FOLDERS = {
    "main": ["main", "mai"], "calib": ["calib"], "judge_sweep": ["judge_sweep"], "prior_sweep": ["prior_sweep"],
    "gemini": ["gemini_cov50"], "upd_full": ["v2_updates"], "upd_half": ["v2_updates_cov50"],
    "q15_surprise": ["v3_qwen1.5b_surprise"], "q3": ["v3_qwen3b"], "q3_surprise": ["v3_qwen3b_surprise"],
    "l3": ["v3_llama3b"], "l3_surprise": ["v3_llama3b_surprise"], "phi4": ["v4_phi4"], "llama32": ["v4_llama32"],
    "random_open": ["v5_random_open"], "planted10": ["v5_planted10"], "planted30": ["v5_planted30"],
    "outdated30": ["v5_outdated30"],
}
MODELS = {"Qwen2.5-1.5B-Instruct": "Qwen 1.5B", "Qwen2.5-3B-Instruct": "Qwen 3B", "Llama-3.2-3B-Instruct": "Llama 3B"}
NAMES = {"naive": "Naive", "surprise": "Surprise-only", "stability": "Stability-only", "gate_no_verif": "Surprise + test, no verifier key",
         "gate_no_test": "Verifier key (judge only)", "gate_full": "Verifier key + test", "prior_only": "Prior only",
         "gate_prior": "Judge + prior", "gate_prior_ev": "Judge + prior + evidence route"}


class Paper:
    def __init__(self, root: Path, out: Path):
        self.root, self.out = root, out
        self.md: list[str] = []
        self.values: dict[str, float] = {}
        out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- io
    def folder(self, key: str) -> Path:
        for name in FOLDERS[key]:
            if (self.root / name).exists():
                return self.root / name
        raise FileNotFoundError(f"results folder for '{key}' not found (tried {FOLDERS[key]})")

    def runs(self, key: str) -> pd.DataFrame:
        df = pd.read_json(self.folder(key) / "results.jsonl", lines=True)
        df["model"] = df["model"].str.split("/").str[-1]
        df["lies_per_1000"] = 1000 * df["poison_trained"] / df["claims_trained"].where(df["claims_trained"] > 0)
        df["acc"] = df["verifier_accuracy"].round(1)
        df["poison_rate"] = df["poison_rate"].round(2)  # stored as e.g. 0.30000000000000004
        return df

    def claims(self, key: str) -> pd.DataFrame:
        files = sorted(self.folder(key).glob("claims_*.jsonl"))
        return pd.concat([pd.read_json(f, lines=True).assign(seed=int(f.stem.split("_s")[-1])) for f in files])

    def diag(self, model: str) -> list[dict]:
        rows = list(csv.DictReader(open(self.root / f"prior_diag_{model}_seed7.csv", encoding="utf-8")))
        for r in rows:
            r["gap"] = float(r["gap"])
        return rows

    # ---------------------------------------------------------------- output helpers
    def save(self, fig, name: str):
        """PNG for viewing, PDF (vector) for the LaTeX paper; the caption lives in the paper."""
        fig.savefig(self.out / f"{name}.png", dpi=150)
        fig.savefig(self.out / f"{name}.pdf")
        plt.close(fig)

    def h(self, title: str, source: str):
        self.md += ["", f"## {title}", "", f"_Source: {source}_", ""]

    def table(self, df: pd.DataFrame, digits: int = 2):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(digits)
        self.md.append(df.to_markdown(index=True if df.index.name else False))

    def keep(self, name: str, value: float):
        self.values[name] = float(value)

    # ---------------------------------------------------------------- sections
    def s51_main(self):
        df = self.runs("main")
        self.h("5.1 Surprise alone learns lies (Table 3, Figure 2)", "results/main (6 systems x 4 rates x 3 seeds)")
        t = df[df.poison_rate == 0.3].groupby("system")[["learn_new", "retention", "poison_absorbed", "poison_real_absorbed"]].mean()
        t.index = [NAMES[s] for s in t.index]
        t.index.name = "system (30% poison)"
        self.table(t)
        self.md.append(f"\nBefore learning: learn_new {df.before_learn_new.mean():.2f}, retention {df.before_retention.mean():.2f}")
        sd = df[df.poison_rate == 0.3].groupby("system")[["learn_new", "retention"]].std()
        self.md.append(f"Max seed SD at 30% excluding verifier key + test: {sd.drop('gate_full').max().max():.3f}; "
                       f"verifier key + test learn_new SD: {sd.loc['gate_full', 'learn_new']:.2f} (30%), "
                       f"{df[(df.poison_rate == 0.5) & (df.system == 'gate_full')].learn_new.std():.2f} (50%)")
        m = df.groupby(["system", "poison_rate"]).mean(numeric_only=True)
        for s in ("surprise", "gate_no_test", "gate_full", "naive", "stability", "gate_no_verif"):
            for r in (0.1, 0.3, 0.5):
                for k in ("poison_absorbed", "poison_fict_absorbed", "poison_real_absorbed"):
                    self.keep(f"main.{s}.{r}.{k}", m.loc[(s, r), k])
            self.keep(f"main.{s}.0.3.learn_new", m.loc[(s, 0.3), "learn_new"])
            self.keep(f"main.{s}.0.3.retention", m.loc[(s, 0.3), "retention"])
        self.keep("main.surprise.real.min", min(m.loc[("surprise", r), "poison_real_absorbed"] for r in (0.1, 0.3, 0.5)))
        self.keep("main.stability.0.5.rollbacks", m.loc[("stability", 0.5), "rollbacks"])
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in ("surprise", "gate_no_verif", "naive", "stability", "gate_full", "gate_no_test"):
            pts = [m.loc[(s, r), "poison_absorbed"] for r in (0.1, 0.3, 0.5)]
            ax.plot([10, 30, 50], pts, marker="o", label=NAMES[s], linewidth=2.5 if s == "gate_no_test" else 1.2)
        ax.set(xlabel="poison rate (%)", ylabel="share of lies asserted", ylim=(0, 1))
        ax.legend(fontsize=7)
        fig.tight_layout()
        self.save(fig, "fig2_lies_by_poison_rate")

    def s52_judge(self):
        df = self.runs("judge_sweep")
        df["leak"] = df.verifier_fp / (df.verifier_fp + df.verifier_tn)
        df["poison_share"] = df.poison_trained / df.claims_trained
        self.h("5.2-5.3 Imperfect judge and the regression test (Figure 3, Table 4)", "results/judge_sweep")
        t = df.groupby(["acc", "system"])[["leak", "poison_absorbed", "learn_new", "rollbacks", "poison_share"]].mean().unstack()
        self.table(t.reset_index(), 3)
        m = df.groupby(["acc", "system"]).mean(numeric_only=True)
        for a in (0.6, 0.7, 0.8, 0.9):
            self.keep(f"judge.{a}.leak", m.loc[(a, "gate_no_test"), "leak"])
            for s in ("gate_no_test", "gate_full"):
                self.keep(f"judge.{a}.{s}.poison_absorbed", m.loc[(a, s), "poison_absorbed"])
                self.keep(f"judge.{a}.{s}.learn_new", m.loc[(a, s), "learn_new"])
                self.keep(f"judge.{a}.{s}.poison_share", m.loc[(a, s), "poison_share"])
            self.keep(f"judge.{a}.gate_full.rollbacks", m.loc[(a, "gate_full"), "rollbacks"])
        fig, ax = plt.subplots(figsize=(6, 4))
        accs = [0.6, 0.7, 0.8, 0.9]
        ax.plot(accs, [m.loc[(a, "gate_no_test"), "poison_absorbed"] for a in accs], marker="o", label="lies absorbed")
        ax.plot(accs, [m.loc[(a, "gate_no_test"), "leak"] for a in accs], "--", marker="o", label="lies the judge approves")
        ax.set(xlabel="judge accuracy", ylabel="share", ylim=(0, 0.5))
        ax.legend()
        fig.tight_layout()
        self.save(fig, "fig3_lies_vs_judge")
        # calibration of the SPRT null drift
        drift = [json.loads(line)["drift"] for f in sorted((self.folder("calib") / "chunks").glob("gate_no_test_r0.0_*.jsonl"))
                 for line in f.read_text().splitlines() if "drift" in line]
        s = pd.Series(drift)
        self.md.append(f"\nCalibration: {len(s)} clean chunks, drift min {s.min():.2f}, max {s.max():.2f}, "
                       f"p95 {s.quantile(0.95):.3f}, sd {s.std():.3f}")
        self.keep("calib.p95", s.quantile(0.95))
        self.keep("calib.sd", s.std())

    def s54_prior(self):
        df = self.runs("prior_sweep")
        self.h("5.4 Prior key with an imperfect judge (Figure 4)", "results/prior_sweep")
        df["held_true_rate"] = df.prior_held_true / (df.prior_held_true + df.prior_passed_true)
        cols = ["poison_real_absorbed", "poison_fict_absorbed", "learn_new", "retention", "lies_per_1000", "held_true_rate"]
        t = df.groupby(["acc", "system"])[cols].mean()
        self.table(t.reset_index(), 3)
        m = t
        for a in (0.6, 0.7, 0.8, 0.9):
            for s in ("gate_no_test", "gate_prior"):
                for c in cols[:5]:
                    self.keep(f"prior.{a}.{s}.{c}", m.loc[(a, s), c])
            self.keep(f"prior.{a}.factor", m.loc[(a, "gate_no_test"), "poison_real_absorbed"] / m.loc[(a, "gate_prior"), "poison_real_absorbed"])
        self.keep("prior.prior_only.real", df[df.system == "prior_only"].poison_real_absorbed.mean())
        self.keep("prior.held_true_rate", df[df.system == "gate_prior"].held_true_rate.mean())
        fig, ax = plt.subplots(figsize=(6, 4))
        accs = [0.6, 0.7, 0.8, 0.9]
        for s in ("gate_no_test", "gate_prior"):
            ax.plot(accs, [m.loc[(a, s), "poison_real_absorbed"] for a in accs], marker="o", label=NAMES[s])
        ax.axhline(self.values["prior.prior_only.real"], ls="--", c="grey", label="Prior only (no judge)")
        ax.set(xlabel="judge accuracy", ylabel="known-fact lies asserted", ylim=(0, 0.5))
        ax.legend()
        fig.tight_layout()
        self.save(fig, "fig4_real_lies_prior_key")

    @staticmethod
    def auc(pos, neg):
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

    def auc_ci(self, pos, neg, reps=300):
        rng = random.Random(0)
        bs = sorted(self.auc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(reps))
        return bs[int(0.025 * reps)], bs[int(0.975 * reps) - 1]

    @staticmethod
    def person(prompt: str) -> str:
        """Invented person named in a prompt (the 5 relation templates share each person)."""
        from sura.fictional import RELATIONS
        for tmpl, _, _ in RELATIONS.values():
            head, _, tail = tmpl.partition("{}")
            if prompt.startswith(head) and prompt.endswith(tail):
                return prompt[len(head):len(prompt) - len(tail)]
        return prompt

    def auc_ci_cluster(self, pos, new_rows, reps=1000):
        """Percentile bootstrap that resamples whole invented people (all their claims together)
        for the true new facts, and single claims for the known-fact lies (one per CounterFact case)."""
        groups: dict[str, list[float]] = {}
        for r in new_rows:
            groups.setdefault(self.person(r["prompt"]), []).append(r["gap"])
        keys, rng = list(groups), random.Random(0)
        bs = sorted(self.auc([rng.choice(pos) for _ in pos],
                             [g for k in (rng.choice(keys) for _ in keys) for g in groups[k]]) for _ in range(reps))
        return bs[int(0.025 * reps)], bs[int(0.975 * reps) - 1], len(keys)

    def s54_curve(self):
        self.h("5.4/5.7 Prior-key separation and operating points (Figure 5, Table 5)",
               "results/prior_diag_<model>_seed7.csv (fresh world, 200 claims per type)")
        rows_out, fig, ax = [], *plt.subplots(figsize=(5, 5))
        for model, short in MODELS.items():
            rows = self.diag(model)
            v = lambda k: [r["gap"] for r in rows if r["kind"] == k]
            new, real, fict = v("new"), v("poison_real"), v("poison_fict")
            a_r, a_f = self.auc(real, new), self.auc(fict, new)
            lo, hi = self.auc_ci(real, new)
            self.keep(f"auc.{short}.real", a_r)
            self.keep(f"auc.{short}.real.lo", lo)
            self.keep(f"auc.{short}.real.hi", hi)
            self.keep(f"auc.{short}.fict", a_f)
            clo, chi, n_people = self.auc_ci_cluster(real, [r for r in rows if r["kind"] == "new"])
            self.keep(f"auc.{short}.real.clo", clo)
            self.keep(f"auc.{short}.real.chi", chi)
            share = lambda xs, t: sum(x >= t for x in xs) / len(xs)
            for t in (4, 6, 8):
                held_r, held_n = sum(x >= t for x in real), sum(x >= t for x in new)
                prec = held_r / (held_r + held_n) if held_r + held_n else float("nan")
                self.keep(f"op.{short}.{t}.precision", prec)
                rows_out.append(dict(model=short, threshold=t, real_lies_held=share(real, t),
                                     true_new_held=share(new, t), invented_lies_held=share(fict, t),
                                     precision_real_vs_new=prec))
                self.keep(f"op.{short}.{t}.real", share(real, t))
                self.keep(f"op.{short}.{t}.new", share(new, t))
                self.keep(f"op.{short}.{t}.fict", share(fict, t))
            ts = [i / 2 for i in range(0, 31)]
            ax.plot([share(new, t) for t in ts], [share(real, t) for t in ts], marker=".", label=f"{short} real (AUC {a_r:.2f})")
            if short == "Qwen 1.5B":
                ax.plot([share(new, t) for t in ts], [share(fict, t) for t in ts], c="grey", label=f"{short} invented (AUC {a_f:.2f})")
            self.md.append(f"{short}: AUC real {a_r:.3f} [claim bootstrap {lo:.3f}, {hi:.3f}; "
                           f"person bootstrap over {n_people} people {clo:.3f}, {chi:.3f}], invented {a_f:.3f}; "
                           f"median gap new {pd.Series(new).median():.2f}, real {pd.Series(real).median():.2f}")
        self.md.append("")
        self.table(pd.DataFrame(rows_out))
        ax.plot([0, 1], [0, 1], ls="--", c="lightgrey")
        ax.set(xlabel="true new facts held back", ylabel="lies held back")
        ax.legend(fontsize=7)
        fig.tight_layout()
        self.save(fig, "fig5_prior_tradeoff")

    def s55_gemini(self):
        df = self.runs("gemini")
        self.h("5.5 Gemini judge, half the claims sourced", "results/gemini_cov50")
        t = df.groupby("system")[["learn_new", "poison_absorbed", "retention", "verifier_tp", "verifier_fn", "verifier_tn", "verifier_fp"]].mean()
        self.table(t.reset_index(), 3)
        for c in ("learn_new", "poison_absorbed", "retention", "verifier_fp", "verifier_fn"):
            self.keep(f"gemini.gate_no_test.{c}", t.loc["gate_no_test", c])
            self.keep(f"gemini.gate_full.{c}", t.loc["gate_full", c])

    def s56_updates(self):
        self.h("5.6 Legitimate updates and the evidence route (Table 6)", "results/v2_updates, results/v2_updates_cov50")
        cols = ["learn_updates", "learn_new", "poison_real_absorbed", "retention", "lies_per_1000", "prior_rescued_true", "prior_rescued_poison"]
        for key, label in (("upd_full", "all"), ("upd_half", "half")):
            df = self.runs(key)
            t = df.groupby(["acc", "system"])[cols].mean()
            self.md.append(f"\nSources for {label} of the claims:")
            self.table(t.reset_index(), 3)
            for a in (0.7, 0.9):
                for s in ("gate_no_test", "gate_prior", "gate_prior_ev"):
                    for c in cols:
                        self.keep(f"upd.{label}.{a}.{s}.{c}", t.loc[(a, s), c])
            self.keep(f"upd.{label}.held_true", df[df.system == "gate_prior_ev"].prior_held_true.mean())
            self.keep(f"upd.{label}.max_rescued_poison", df.prior_rescued_poison.max())
            self.keep(f"upd.{label}.max_sd", df.groupby(["acc", "system"])[
                ["learn_new", "learn_updates", "poison_real_absorbed", "poison_fict_absorbed", "retention"]].std().max().max())

    def s57_models(self):
        self.h("5.7 Replication across models (Table 7)", "results/v2_updates_cov50, v3_*")
        sets = {"Qwen 1.5B": ("upd_half", "q15_surprise"), "Qwen 3B": ("q3", "q3_surprise"), "Llama 3B": ("l3", "l3_surprise")}
        cols = ["poison_real_absorbed", "poison_fict_absorbed", "learn_updates", "learn_new", "retention", "lies_per_1000"]
        out = []
        for short, (judged, surprise) in sets.items():
            df = pd.concat([self.runs(judged), self.runs(surprise)])
            df = df[(df.acc == 0.9) | (df.system == "surprise")]
            t = df.groupby("system")[cols].mean()
            for s, row in t.iterrows():
                out.append(dict(model=short, system=NAMES[s], **row.to_dict()))
                for c in cols:
                    self.keep(f"models.{short}.{s}.{c}", row[c])
            self.keep(f"models.{short}.max_sd", df.groupby("system")[cols[:-1]].std().max().max())
        self.table(pd.DataFrame(out))

    def s58_real_judges(self):
        self.h("5.8 Real judges (Tables 8 and 9, overlap analysis)", "results/v4_phi4, results/v4_llama32")
        for key, short in (("phi4", "phi4"), ("llama32", "Llama 3.2")):
            c = self.claims(key)
            j = c[c.judge.notna()]
            acc = j.groupby(["kind", "has_source"]).judge.agg(["mean", "count"])
            self.md.append(f"\nJudge {short}: share accepted by claim kind and source")
            self.table(acc.reset_index(), 3)
            for (kind, src), row in acc.iterrows():
                self.keep(f"judge4.{short}.{kind}.{bool(src)}", row["mean"])
            df = self.runs(key)
            t = df.groupby("system")[["learn_updates", "learn_new", "poison_real_absorbed", "retention", "lies_per_1000",
                                      "prior_rescued_poison", "judge_unparsed"]].mean()
            self.table(t.reset_index(), 3)
            for s, row in t.iterrows():
                for col in ("learn_updates", "learn_new", "poison_real_absorbed", "retention", "lies_per_1000"):
                    self.keep(f"judge5.{short}.{s}.{col}", row[col])
            self.keep(f"judge5.{short}.unparsed", t.judge_unparsed.max())
            self.keep(f"judge5.{short}.max_sd", df.groupby("system")[["learn_new", "learn_updates", "poison_real_absorbed", "retention"]].std().max().max())
        # do the judge's mistakes overlap with the prior key's misses?
        per_seed, cases = [], []
        for key in ("phi4", "llama32"):
            c = self.claims(key)
            pr = c[(c.kind == "poison_real") & c.judge.notna() & c.prior_gap.notna()]
            for seed, g in pr.groupby("seed"):
                fooled = g[g.judge == True]  # noqa: E712
                per_seed.append(dict(judge=key, seed=seed, fooled=len(fooled), held=int((fooled.prior_gap >= 6).sum()),
                                     base_rate=(g.prior_gap >= 6).mean()))
                cases += list(fooled.uid)
        ps = pd.DataFrame(per_seed)
        ps["held_rate"] = ps.held / ps.fooled
        self.md.append("\nOverlap of judge mistakes and prior-key holds, per judge and seed:")
        self.table(ps, 3)
        below = int((ps[ps.fooled > 0].held_rate < ps[ps.fooled > 0].base_rate).sum())
        n = int((ps.fooled > 0).sum())
        from scipy.stats import binomtest
        p_sign = binomtest(below, n, 0.5, alternative="greater").pvalue
        self.md.append(f"\nPooled: {ps.fooled.sum()} cases, {len(set(cases))} distinct claims, {ps.held.sum()} held "
                       f"({ps.held.sum() / ps.fooled.sum():.2f}); seed runs below base rate: {below} of {n}, "
                       f"mean held rate {ps.held_rate.mean():.2f} vs base {ps.base_rate.mean():.2f}, sign test p = {p_sign:.3f}")
        self.keep("overlap.cases", ps.fooled.sum())
        self.keep("overlap.distinct", len(set(cases)))
        self.keep("overlap.held", ps.held.sum())
        self.keep("overlap.below", below)
        self.keep("overlap.mean_held", ps.held_rate.mean())
        self.keep("overlap.base", ps.base_rate.mean())
        self.keep("overlap.p_sign", p_sign)

    # ---------------------------------------------------------------- audit

    # ---------------------------------------------------------------- additional analyses (5.1, 5.6, 5.9, 6)
    @staticmethod
    def paired(a: pd.Series, b: pd.Series):
        """Mean of a - b over shared seeds with a 95% t-interval."""
        from scipy import stats
        d = (a - b).dropna().astype(float)
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / len(d) ** 0.5
        return d.mean(), d.mean() - half, d.mean() + half

    def s59_review(self):
        self.h("5.1, 5.6, 5.9, 6 Additional analyses",
               "results/v5_*, results/v4_*, results/prior_diag_<model>_seed7[_4cond].csv")
        held = lambda xs: sum(x >= 6 for x in xs) / len(xs)
        # what the prior key detects: simpler signals on the seed-7 diagnostic
        for model, short in MODELS.items():
            rows = list(csv.DictReader(open(self.root / f"prior_diag_{model}_seed7.csv", encoding="utf-8")))
            col = lambda k, c: [float(r[c]) for r in rows if r["kind"] == k]
            for c in ("nll_claim", "p_next"):
                self.keep(f"signal.{short}.{c}", self.auc(col("poison_real", c), col("new", c)))
            d4 = list(csv.DictReader(open(self.root / f"prior_diag_{model}_seed7_4cond.csv", encoding="utf-8")))
            g = lambda k: [float(r["gap"]) for r in d4 if r["kind"] == k]
            for k in ("known_true", "new", "poison_fict", "poison_real", "update"):
                self.keep(f"cond4.{short}.{k}.held", held(g(k)))
            self.keep(f"cond4.{short}.known_true.mean", sum(g("known_true")) / len(g("known_true")))
            self.keep(f"cond4.{short}.auc_lie_vs_update", self.auc(g("poison_real"), g("update")))
        # lies vs updates and the screening attacker on the real-judge runs
        cl = pd.concat([self.claims("phi4"), self.claims("llama32")]).drop_duplicates(["seed", "uid", "target"])
        gp = lambda k: cl[cl.kind == k].prior_gap.dropna().tolist()
        lies, upd = gp("poison_real"), gp("update")
        self.keep("review.lie_vs_update.auc", self.auc(lies, upd))
        lo, hi = self.auc_ci(lies, upd)
        self.keep("review.lie_vs_update.lo", lo)
        self.keep("review.lie_vs_update.hi", hi)
        self.keep("review.lie_vs_update.n_lies", len(lies))
        self.keep("review.lie_vs_update.n_upd", len(upd))
        self.keep("review.held.lies", held(lies))
        self.keep("review.held.updates", held(upd))
        for key, j in (("phi4", "phi4"), ("llama32", "llama32")):
            c = self.claims(key)
            lie = c[(c.kind == "poison_real") & c.prior_gap.notna() & c.judge.notna()]
            passes, acc = lie.prior_gap < 6, lie.judge.astype(bool)
            self.keep(f"attack.{j}.pass_prior", passes.mean())
            self.keep(f"attack.{j}.unscreened", (passes & acc).mean())
            self.keep(f"attack.{j}.screened", lie[passes].judge.astype(bool).mean())
        # operating cost (simulated judge 0.9, half sources)
        u = self.runs("upd_half")
        u = u[u.acc == 0.9]
        calls = (u.verifier_tp + u.verifier_fp + u.verifier_tn + u.verifier_fn)
        for sname in ("gate_no_test", "gate_prior", "gate_prior_ev"):
            self.keep(f"cost.{sname}.judge_calls", calls[u.system == sname].mean())
        ev = u[u.system == "gate_prior_ev"]
        self.keep("cost.held_for_review", (ev.prior_held_true + ev.prior_held_poison
                                           - ev.prior_rescued_true - ev.prior_rescued_poison).mean())
        # random control and open-ended answers
        r = self.runs("random_open")
        by = lambda sname: r[r.system == sname].set_index("seed")
        su, ra = by("surprise"), by("random")
        for m in ("lies_per_1000", "poison_real_absorbed", "claims_trained"):
            mean, lo, hi = self.paired(su[m], ra[m])
            self.keep(f"random.diff.{m}", mean)
            self.keep(f"random.diff.{m}.lo", lo)
            self.keep(f"random.diff.{m}.hi", hi)
        t = r.groupby("system").mean(numeric_only=True)
        for sname in ("surprise", "random", "gate_no_test", "gate_prior_ev"):
            for m in ("lies_per_1000", "poison_real_absorbed", "open_poison_real", "open_learn_new",
                      "open_learn_new_paraphrase", "open_poison_real_paraphrase"):
                self.keep(f"open.{sname}.{m}", t.loc[sname, m])
        self.keep("open.before.poison_real", r.before_open_poison_real.max())
        self.keep("open.before.retention", r.before_open_retention.mean())
        self.table(t[["poison_real_absorbed", "open_poison_real", "learn_new", "open_learn_new",
                      "lies_per_1000", "claims_trained"]].reset_index(), 3)
        # planted and outdated sources vs the same system with clean sources (same seeds)
        base = by("gate_prior_ev")
        for key in ("planted10", "planted30", "outdated30"):
            p = self.runs(key).set_index("seed")
            for m in ("lies_per_1000", "poison_real_absorbed", "learn_updates"):
                self.keep(f"{key}.{m}", p[m].mean())
            self.keep(f"{key}.rescued_lies", p.prior_rescued_poison.mean())
            mean, lo, hi = self.paired(p["lies_per_1000"], base["lies_per_1000"])
            self.keep(f"{key}.diff_lpk", mean)
            self.keep(f"{key}.diff_lpk.lo", lo)
            self.keep(f"{key}.diff_lpk.hi", hi)
        self.keep("clean.gate_prior_ev.learn_updates", base.learn_updates.mean())
        # paired effects of the evidence route (Sections 5.6 and 5.8 settings)
        for key in ("upd_half", "phi4", "llama32"):
            w = self.runs(key)
            if key == "upd_half":
                w = w[w.acc == 0.9]
            sy = lambda sname: w[w.system == sname].set_index("seed")
            _, _, hi = self.paired(sy("gate_prior_ev")["lies_per_1000"], sy("gate_no_test")["lies_per_1000"])
            self.keep(f"route.{key}.lpk_vs_judge.hi", hi)
            mean, _, _ = self.paired(sy("gate_prior_ev")["retention"], sy("gate_prior")["retention"])
            self.keep(f"route.{key}.retention_vs_prior", mean)

    def audit(self) -> int:
        """Numbers as they appear in the paper text and tables, with their rounding tolerance."""
        v = self.values
        q = "Qwen 1.5B"
        checks = [
            # abstract / 5.1
            ("5.1 surprise lies 84-93% (min)", min(v[f"main.surprise.{r}.poison_absorbed"] for r in (0.1, 0.3, 0.5)), 0.84, 0.005),
            ("5.1 surprise lies 84-93% (max)", max(v[f"main.surprise.{r}.poison_absorbed"] for r in (0.1, 0.3, 0.5)), 0.93, 0.005),
            ("5.1 surprise real lies 99-100%", v["main.surprise.real.min"], 0.99, 0.01),
            ("Table 3 verifier key learn_new", v["main.gate_no_test.0.3.learn_new"], 0.86, 0.005),
            ("Table 3 verifier key retention", v["main.gate_no_test.0.3.retention"], 0.98, 0.005),
            ("Table 3 surprise learn_new", v["main.surprise.0.3.learn_new"], 0.80, 0.005),
            ("Table 3 surprise retention", v["main.surprise.0.3.retention"], 0.93, 0.005),
            ("Table 3 naive learn_new", v["main.naive.0.3.learn_new"], 0.19, 0.005),
            ("Table 3 stability learn_new", v["main.stability.0.3.learn_new"], 0.17, 0.005),
            ("Table 3 test-only learn_new", v["main.gate_no_verif.0.3.learn_new"], 0.74, 0.005),
            ("Table 3 verifier key + test learn_new", v["main.gate_full.0.3.learn_new"], 0.49, 0.005),
            ("5.1 stability rollbacks at 50%", v["main.stability.0.5.rollbacks"], 94, 0.5),
            *[(f"Fig 2 {sys_} at {r}", v[f"main.{sys_}.{r}.poison_absorbed"], x, 0.0005)
              for sys_, xs in (("surprise", (.932, .901, .837)), ("gate_no_verif", (.886, .891, .833)), ("naive", (.629, .587, .561)),
                               ("stability", (.553, .441, .125)), ("gate_full", (.030, .047, .043)), ("gate_no_test", (.008, .041, .043)))
              for r, x in zip((0.1, 0.3, 0.5), xs)],
            # 5.2 / 5.3
            *[(f"Fig 3 leak acc {a}", v[f"judge.{a}.leak"], x, 0.0005) for a, x in ((0.6, .412), (0.7, .303), (0.8, .215), (0.9, .122))],
            *[(f"Fig 3 absorbed acc {a}", v[f"judge.{a}.gate_no_test.poison_absorbed"], x, 0.0005) for a, x in ((0.6, .427), (0.7, .314), (0.8, .238), (0.9, .146))],
            *[(f"5.3 rollbacks acc {a}", v[f"judge.{a}.gate_full.rollbacks"], x, 0.05) for a, x in ((0.6, 26.3), (0.7, 38.3), (0.8, 15.3), (0.9, 2.0))],
            *[(f"5.3 poison share with test acc {a}", v[f"judge.{a}.gate_full.poison_share"], x, 0.0005) for a, x in ((0.6, .220), (0.7, .147), (0.8, .102), (0.9, .053))],
            *[(f"5.3 poison share w/o test acc {a}", v[f"judge.{a}.gate_no_test.poison_share"], x, 0.0005) for a, x in ((0.6, .222), (0.7, .153), (0.8, .098), (0.9, .053))],
            *[(f"5.3 learn with test acc {a}", v[f"judge.{a}.gate_full.learn_new"], x, 0.005) for a, x in ((0.6, .46), (0.7, .40), (0.8, .63), (0.9, .79))],
            *[(f"5.3 learn w/o test acc {a}", v[f"judge.{a}.gate_no_test.learn_new"], x, 0.005) for a, x in ((0.6, .60), (0.7, .67), (0.8, .71), (0.9, .80))],
            ("5.3 calibration null drift (p95)", v["calib.p95"], 0.446, 0.0005),
            ("5.3 calibration delta (sd)", v["calib.sd"], 0.203, 0.0005),
            # 5.4
            *[(f"Fig 4 judge only acc {a}", v[f"prior.{a}.gate_no_test.poison_real_absorbed"], x, 0.0005) for a, x in ((0.6, .400), (0.7, .302), (0.8, .184), (0.9, .122))],
            *[(f"Fig 4 judge + prior acc {a}", v[f"prior.{a}.gate_prior.poison_real_absorbed"], x, 0.0005) for a, x in ((0.6, .110), (0.7, .059), (0.8, .047), (0.9, .039))],
            ("Fig 4 prior only", v["prior.prior_only.real"], 0.239, 0.0005),
            ("5.4 factor min (3x)", min(v[f"prior.{a}.factor"] for a in (0.6, 0.7, 0.8, 0.9)), 3.0, 0.15),
            ("5.4 factor max (5x)", max(v[f"prior.{a}.factor"] for a in (0.6, 0.7, 0.8, 0.9)), 5.0, 0.15),
            ("5.4 held true rate ~30%", v["prior.held_true_rate"], 0.30, 0.005),
            ("5.4 lies/1000 judge acc 0.7", v["prior.0.7.gate_no_test.lies_per_1000"], 152, 0.5),
            ("5.4 lies/1000 prior acc 0.7", v["prior.0.7.gate_prior.lies_per_1000"], 111, 0.5),
            ("5.4 lies/1000 judge acc 0.9", v["prior.0.9.gate_no_test.lies_per_1000"], 54, 0.5),
            ("5.4 lies/1000 prior acc 0.9", v["prior.0.9.gate_prior.lies_per_1000"], 43, 0.5),
            ("5.4 fict lies judge acc 0.6", v["prior.0.6.gate_no_test.poison_fict_absorbed"], 0.41, 0.005),
            ("5.4 fict lies prior acc 0.6", v["prior.0.6.gate_prior.poison_fict_absorbed"], 0.35, 0.005),
            ("5.4 learn judge acc 0.6", v["prior.0.6.gate_no_test.learn_new"], 0.60, 0.005),
            ("5.4 learn prior acc 0.6", v["prior.0.6.gate_prior.learn_new"], 0.44, 0.005),
            ("5.4 learn prior acc 0.9", v["prior.0.9.gate_prior.learn_new"], 0.60, 0.005),
            ("5.4 learn judge acc 0.9", v["prior.0.9.gate_no_test.learn_new"], 0.79, 0.005),
            # AUC and Table 5
            ("AUC Qwen 1.5B real", v[f"auc.{q}.real"], 0.83, 0.005),
            ("AUC Qwen 1.5B real CI low", v[f"auc.{q}.real.lo"], 0.79, 0.005),
            ("AUC Qwen 1.5B real CI high", v[f"auc.{q}.real.hi"], 0.87, 0.005),
            ("AUC Qwen 1.5B invented", v[f"auc.{q}.fict"], 0.49, 0.005),
            ("AUC Qwen 3B real", v["auc.Qwen 3B.real"], 0.82, 0.005),
            ("AUC Qwen 3B real CI low", v["auc.Qwen 3B.real.lo"], 0.78, 0.005),
            ("AUC Qwen 3B real CI high", v["auc.Qwen 3B.real.hi"], 0.86, 0.005),
            ("AUC Qwen 3B invented", v["auc.Qwen 3B.fict"], 0.50, 0.005),
            ("AUC Llama 3B real", v["auc.Llama 3B.real"], 0.78, 0.005),
            ("AUC Llama 3B real CI low", v["auc.Llama 3B.real.lo"], 0.74, 0.005),
            ("AUC Llama 3B real CI high", v["auc.Llama 3B.real.hi"], 0.81, 0.005),
            ("AUC Llama 3B invented", v["auc.Llama 3B.fict"], 0.49, 0.005),
            # person-clustered bootstrap intervals and precision at the default threshold (Table 5)
            *[(f"AUC {m} real person-CI {side}", v[f"auc.{m}.real.{k}"], x, 0.005)
              for m, side, k, x in [(q, "low", "clo", .79), (q, "high", "chi", .87),
                                    ("Qwen 3B", "low", "clo", .78), ("Qwen 3B", "high", "chi", .86),
                                    ("Llama 3B", "low", "clo", .74), ("Llama 3B", "high", "chi", .82)]],
            *[(f"Table 5 {m} t={t} precision", v[f"op.{m}.{t}.precision"], x, 0.005)
              for m, t, x in [(q, 4, .69), ("Qwen 3B", 4, .67), ("Llama 3B", 4, .65),
                              (q, 6, .74), ("Qwen 3B", 6, .71), ("Llama 3B", 6, .69),
                              (q, 8, .79), ("Qwen 3B", 8, .76), ("Llama 3B", 8, .72)]],
            # surprise-only absorption by lie type, Section 5.1 (68-86% invented, 84-93% all)
            *[(f"5.1 surprise {k} at {r}", v[f"main.surprise.{r}.{k}"], x, 0.005)
              for r, k, x in [(0.1, "poison_fict_absorbed", .86), (0.5, "poison_fict_absorbed", .68),
                              (0.1, "poison_absorbed", .93), (0.5, "poison_absorbed", .84)]],
            *[(f"Table 5 {m} t={t} {k}", v[f"op.{m}.{t}.{k}"], x, 0.005)
              for m, t, k, x in [(q, 4, "real", .92), (q, 4, "new", .42), (q, 4, "fict", .46),
                                 (q, 6, "real", .76), (q, 6, "new", .27), (q, 6, "fict", .29),
                                 (q, 8, "real", .54), (q, 8, "new", .14), (q, 8, "fict", .14),
                                 ("Qwen 3B", 4, "real", .92), ("Qwen 3B", 4, "new", .46), ("Qwen 3B", 4, "fict", .48),
                                 ("Qwen 3B", 6, "real", .81), ("Qwen 3B", 6, "new", .33), ("Qwen 3B", 6, "fict", .34),
                                 ("Qwen 3B", 8, "real", .56), ("Qwen 3B", 8, "new", .18), ("Qwen 3B", 8, "fict", .13),
                                 ("Llama 3B", 4, "real", .91), ("Llama 3B", 4, "new", .50), ("Llama 3B", 4, "fict", .49),
                                 ("Llama 3B", 6, "real", .71), ("Llama 3B", 6, "new", .32), ("Llama 3B", 6, "fict", .29),
                                 ("Llama 3B", 8, "real", .44), ("Llama 3B", 8, "new", .17), ("Llama 3B", 8, "fict", .16)]],
            # 5.5
            ("5.5 Gemini lies (no test)", v["gemini.gate_no_test.poison_absorbed"], 0.064, 0.0005),
            ("5.5 Gemini learn (no test)", v["gemini.gate_no_test.learn_new"], 0.51, 0.005),
            ("5.5 Gemini retention (no test)", v["gemini.gate_no_test.retention"], 0.980, 0.0005),
            ("5.5 Gemini lies (test)", v["gemini.gate_full.poison_absorbed"], 0.058, 0.0005),
            ("5.5 Gemini learn (test)", v["gemini.gate_full.learn_new"], 0.50, 0.005),
            ("5.5 Gemini retention (test)", v["gemini.gate_full.retention"], 0.978, 0.0005),
            ("5.5 Gemini lies approved per run", v["gemini.gate_no_test.verifier_fp"], 2.0, 0.5),
            ("5.5 Gemini truths rejected per run", v["gemini.gate_no_test.verifier_fn"], 198, 0.5),
            # 5.6 Table 6 (judge 0.9)
            *[(f"Table 6 {cov} {s} {c}", v[f"upd.{cov}.0.9.{s}.{c}"], x, tol)
              for cov, s, c, x, tol in [
                  ("all", "gate_no_test", "learn_updates", .90, .005), ("all", "gate_prior", "learn_updates", .24, .005), ("all", "gate_prior_ev", "learn_updates", .98, .005),
                  ("all", "gate_no_test", "learn_new", .79, .005), ("all", "gate_prior", "learn_new", .60, .005), ("all", "gate_prior_ev", "learn_new", .79, .005),
                  ("all", "gate_no_test", "poison_real_absorbed", .18, .005), ("all", "gate_prior", "poison_real_absorbed", .07, .005), ("all", "gate_prior_ev", "poison_real_absorbed", .11, .005),
                  ("all", "gate_no_test", "retention", .92, .005), ("all", "gate_prior", "retention", .97, .005), ("all", "gate_prior_ev", "retention", .91, .005),
                  ("all", "gate_no_test", "lies_per_1000", 42, .5), ("all", "gate_prior", "lies_per_1000", 33, .5), ("all", "gate_prior_ev", "lies_per_1000", 18, .5),
                  ("half", "gate_no_test", "learn_updates", .90, .005), ("half", "gate_prior", "learn_updates", .24, .005), ("half", "gate_prior_ev", "learn_updates", .62, .005),
                  ("half", "gate_no_test", "learn_new", .81, .005), ("half", "gate_prior", "learn_new", .60, .005), ("half", "gate_prior_ev", "learn_new", .72, .005),
                  ("half", "gate_no_test", "poison_real_absorbed", .18, .005), ("half", "gate_prior", "poison_real_absorbed", .07, .005), ("half", "gate_prior_ev", "poison_real_absorbed", .09, .005),
                  ("half", "gate_no_test", "retention", .93, .005), ("half", "gate_prior", "retention", .97, .005), ("half", "gate_prior_ev", "retention", .94, .005),
                  ("half", "gate_no_test", "lies_per_1000", 44, .5), ("half", "gate_prior", "lies_per_1000", 32, .5), ("half", "gate_prior_ev", "lies_per_1000", 24, .5)]],
            ("5.6 rescued true (all sources)", v["upd.all.0.9.gate_prior_ev.prior_rescued_true"], 192, 0.5),
            ("5.6 held true (all sources)", v["upd.all.held_true"], 192, 0.5),
            ("5.6 rescued true (half sources)", v["upd.half.0.9.gate_prior_ev.prior_rescued_true"], 101, 1.0),
            ("5.6 rescued lies (all) = 0", v["upd.all.max_rescued_poison"], 0, 0),
            ("5.6 rescued lies (half) = 0", v["upd.half.max_rescued_poison"], 0, 0),
            # "at most" claims: passes iff the recomputed SD does not exceed the bound
            ("5.6 seed SD at most 0.07 (all sources)", max(v["upd.all.max_sd"], 0.07), 0.07, 0),
            ("5.6 seed SD at most 0.07 (half sources)", max(v["upd.half.max_sd"], 0.07), 0.07, 0),
            *[(f"5.7 seed SD at most 0.06 ({m})", max(v[f"models.{m}.max_sd"], 0.06), 0.06, 0) for m in MODELS.values()],
            *[(f"5.8 seed SD at most 0.05 ({j})", max(v[f"judge5.{j}.max_sd"], 0.05), 0.05, 0) for j in ("phi4", "Llama 3.2")],
            ("5.6 acc 0.7 updates judge (all)", v["upd.all.0.7.gate_no_test.learn_updates"], .70, .005),
            ("5.6 acc 0.7 updates prior (all)", v["upd.all.0.7.gate_prior.learn_updates"], .19, .005),
            ("5.6 acc 0.7 updates evidence (all)", v["upd.all.0.7.gate_prior_ev.learn_updates"], .93, .005),
            ("5.6 acc 0.7 updates evidence (half)", v["upd.half.0.7.gate_prior_ev.learn_updates"], .59, .005),
            ("5.6 acc 0.7 lies/1000 evidence (all)", v["upd.all.0.7.gate_prior_ev.lies_per_1000"], 60, .5),
            ("5.6 acc 0.7 lies/1000 evidence (half)", v["upd.half.0.7.gate_prior_ev.lies_per_1000"], 79, .5),
        ]
        # 5.7 Table 7 (judge 0.9)
        table3 = {
            "surprise": dict(poison_real_absorbed=(1.00, 1.00, 1.00), poison_fict_absorbed=(.79, .81, .87), learn_updates=(.99, .99, .99),
                             learn_new=(.77, .81, .84), retention=(.88, .89, .88), lies_per_1000=(257, 251, 254)),
            "gate_no_test": dict(poison_real_absorbed=(.18, .16, .19), poison_fict_absorbed=(.16, .16, .15), learn_updates=(.90, .91, .90),
                                 learn_new=(.81, .81, .83), retention=(.93, .93, .92), lies_per_1000=(44, 37, 38)),
            "gate_prior": dict(poison_real_absorbed=(.07, .04, .05), poison_fict_absorbed=(.12, .11, .12), learn_updates=(.24, .17, .20),
                               learn_new=(.60, .57, .57), retention=(.97, .97, .97), lies_per_1000=(32, 25, 29)),
            "gate_prior_ev": dict(poison_real_absorbed=(.09, .07, .08), poison_fict_absorbed=(.14, .14, .13), learn_updates=(.62, .57, .59),
                                  learn_new=(.72, .72, .73), retention=(.94, .95, .94), lies_per_1000=(24, 18, 22)),
        }
        for s, cols in table3.items():
            for c, vals in cols.items():
                for short, x in zip(MODELS.values(), vals):
                    checks.append((f"Table 7 {short} {s} {c}", v[f"models.{short}.{s}.{c}"], x, 0.5 if c == "lies_per_1000" else 0.005))
        # 5.8 Tables 8 and 9, overlap
        t4 = [("poison_real", False, .059, .040), ("poison_fict", False, .005, .000), ("new", True, .99, .65),
              ("new", False, 0, 0), ("update", True, .30, .40), ("update", False, .04, .01)]
        for kind, src, a, b in t4:
            tol = 0.0005 if kind.startswith("poison") else 0.005
            checks.append((f"Table 8 phi4 {kind} source={src}", v[f"judge4.phi4.{kind}.{src}"], a, tol))
            checks.append((f"Table 8 Llama {kind} source={src}", v[f"judge4.Llama 3.2.{kind}.{src}"], b, tol))
        t5 = {"gate_no_test": dict(learn_updates=(.19, .23), learn_new=(.55, .38), poison_real_absorbed=(.09, .09), retention=(.96, .96), lies_per_1000=(23, 21)),
              "gate_prior": dict(learn_updates=(.07, .08), learn_new=(.40, .28), poison_real_absorbed=(.06, .04), retention=(.98, .98), lies_per_1000=(16, 10)),
              "gate_prior_ev": dict(learn_updates=(.47, .46), learn_new=(.54, .41), poison_real_absorbed=(.07, .07), retention=(.94, .94), lies_per_1000=(9, 5))}
        for s, cols in t5.items():
            for c, (a, b) in cols.items():
                tol = 0.5 if c == "lies_per_1000" else 0.005
                checks.append((f"Table 9 phi4 {s} {c}", v[f"judge5.phi4.{s}.{c}"], a, tol))
                checks.append((f"Table 9 Llama {s} {c}", v[f"judge5.Llama 3.2.{s}.{c}"], b, tol))
        checks += [
            ("5.8 unparsed answers phi4 = 0", v["judge5.phi4.unparsed"], 0, 0),
            ("5.8 unparsed answers Llama = 0", v["judge5.Llama 3.2.unparsed"], 0, 0),
            ("5.8 judge-approved real lies", v["overlap.cases"], 42, 0),
            ("5.8 distinct claims", v["overlap.distinct"], 28, 0),
            ("5.8 held by prior", v["overlap.held"], 26, 0),
            ("5.8 seed runs below base", v["overlap.below"], 8, 0),
            ("5.8 mean held rate 65%", v["overlap.mean_held"], 0.65, 0.005),
            ("5.8 base rate 77%", v["overlap.base"], 0.77, 0.005),
            ("5.8 sign test p ~ 0.05", v["overlap.p_sign"], 0.05, 0.006),
        ]

        # ---- additional analyses (5.1, 5.6, 5.9, 6)
        Q = ("Qwen 1.5B", "Qwen 3B", "Llama 3B")
        for short, nll, conf in zip(Q, (.78, .79, .74), (.84, .80, .88)):
            checks += [(f"5.9 Table 10 claim NLL {short}", v[f"signal.{short}.nll_claim"], nll, 0.005),
                       (f"5.9 Table 10 confidence {short}", v[f"signal.{short}.p_next"], conf, 0.005)]
        checks += [("5.9 lie vs update AUC", v["review.lie_vs_update.auc"], 0.53, 0.005),
                   ("5.9 lie vs update CI low", v["review.lie_vs_update.lo"], 0.49, 0.005),
                   ("5.9 lie vs update CI high", v["review.lie_vs_update.hi"], 0.56, 0.005),
                   ("5.9 n lies", v["review.lie_vs_update.n_lies"], 424, 0), ("5.9 n updates", v["review.lie_vs_update.n_upd"], 500, 0),
                   ("5.9 held lies", v["review.held.lies"], 0.77, 0.005), ("5.9 held updates", v["review.held.updates"], 0.75, 0.005),
                   ("5.9 attack share passing prior", v["attack.phi4.pass_prior"], 0.23, 0.005),
                   ("5.9 attack phi4 screened", v["attack.phi4.screened"], 0.115, 0.0005),
                   ("5.9 attack phi4 unscreened", v["attack.phi4.unscreened"], 0.026, 0.0005),
                   ("5.9 attack llama screened", v["attack.llama32.screened"], 0.052, 0.0005),
                   ("5.9 attack llama unscreened", v["attack.llama32.unscreened"], 0.012, 0.0005),
                   ("5.9 attack gain phi4", v["attack.phi4.screened"] / v["attack.phi4.unscreened"], 4.4, 0.05),
                   ("5.9 attack gain llama", v["attack.llama32.screened"] / v["attack.llama32.unscreened"], 4.4, 0.05),
                   ("6 judge calls judge only", v["cost.gate_no_test.judge_calls"], 661, 0.5),
                   ("6 judge calls judge + prior", v["cost.gate_prior.judge_calls"], 374, 0.5),
                   ("6 judge calls saved", 1 - v["cost.gate_prior.judge_calls"] / v["cost.gate_no_test.judge_calls"], 0.43, 0.005),
                   ("6 held for review", v["cost.held_for_review"], 180, 1.0)]
        for short, mean, lo, hi in zip(Q, (3.2, 3.5, 3.3), (.17, .20, .18), (.17, .20, .18)):
            checks.append((f"5.9 4cond known gap mean {short}", v[f"cond4.{short}.known_true.mean"], mean, 0.05))
        checks += [(f"5.9 4cond {k} held {m}/{short}", v[f"cond4.{short}.{k}.held"], x, 0.005)
                   for k, xs in (("known_true", (.17, .20, .18)), ("new", (.27, .33, .32)), ("poison_fict", (.29, .34, .29)),
                                 ("poison_real", (.76, .81, .71)), ("update", (.67, .84, .79)))
                   for m, (short, x) in enumerate(zip(Q, xs))]
        checks += [(f"5.9 4cond AUC lie vs update {short}", v[f"cond4.{short}.auc_lie_vs_update"], x, 0.005)
                   for short, x in zip(Q, (.53, .45, .46))]
        checks += [("5.1 random lies/1000 surprise", v["open.surprise.lies_per_1000"], 254, 0.5),
                   ("5.1 random lies/1000 random", v["open.random.lies_per_1000"], 257, 0.5),
                   ("5.1 random diff lies/1000", v["random.diff.lies_per_1000"], -3, 0.5),
                   ("5.1 random diff lies/1000 lo", v["random.diff.lies_per_1000.lo"], -15, 0.5),
                   ("5.1 random diff lies/1000 hi", v["random.diff.lies_per_1000.hi"], 9, 0.5),
                   ("5.1 real lies surprise", v["open.surprise.poison_real_absorbed"], 0.995, 0.0005),
                   ("5.1 real lies random", v["open.random.poison_real_absorbed"], 0.889, 0.0005),
                   ("5.1 random diff real lies", v["random.diff.poison_real_absorbed"], 0.11, 0.005),
                   ("5.1 random diff real lies lo", v["random.diff.poison_real_absorbed.lo"], 0.09, 0.005),
                   ("5.1 random diff real lies hi", v["random.diff.poison_real_absorbed.hi"], 0.13, 0.005),
                   ("5.1 fewer claims trained", -v["random.diff.claims_trained"], 86, 0.5),
                   ("5.1 open real lies surprise", v["open.surprise.open_poison_real"], 0.98, 0.005),
                   ("5.1 open real lies verifier key", v["open.gate_no_test.open_poison_real"], 0.14, 0.005),
                   ("5.1 ranking real lies verifier key", v["open.gate_no_test.poison_real_absorbed"], 0.17, 0.005),
                   ("5.1 open real lies evidence route", v["open.gate_prior_ev.open_poison_real"], 0.04, 0.005),
                   ("5.1 ranking real lies evidence route", v["open.gate_prior_ev.poison_real_absorbed"], 0.08, 0.005),
                   ("5.1 open before learning", v["open.before.poison_real"], 0.0, 0),
                   ("5.1 open retention before", v["open.before.retention"], 0.21, 0.005),
                   ("5.1 open new facts min", min(v[f"open.{s}.open_learn_new"] for s in ("surprise", "random", "gate_no_test", "gate_prior_ev")), 0.71, 0.005),
                   ("5.1 open new facts max", max(v[f"open.{s}.open_learn_new"] for s in ("surprise", "random", "gate_no_test", "gate_prior_ev")), 0.79, 0.005),
                   ("5.1 open paraphrase new min", min(v[f"open.{s}.open_learn_new_paraphrase"] for s in ("surprise", "random", "gate_no_test", "gate_prior_ev")), 0.39, 0.005),
                   ("5.1 open paraphrase new max", max(v[f"open.{s}.open_learn_new_paraphrase"] for s in ("surprise", "random", "gate_no_test", "gate_prior_ev")), 0.41, 0.005),
                   ("5.1 open paraphrase lies min", min(v[f"open.{s}.open_poison_real_paraphrase"] for s in ("surprise", "random", "gate_no_test", "gate_prior_ev")), 0.03, 0.005),
                   ("5.1 open paraphrase lies max", max(v[f"open.{s}.open_poison_real_paraphrase"] for s in ("surprise", "random", "gate_no_test", "gate_prior_ev")), 0.37, 0.005),
                   ("5.6 clean lies/1000", v["open.gate_prior_ev.lies_per_1000"], 23, 0.5),
                   ("5.6 planted10 rescued lies", v["planted10.rescued_lies"], 4, 0.5),
                   ("5.6 planted10 lies/1000", v["planted10.lies_per_1000"], 35, 0.5),
                   ("5.6 planted10 diff", v["planted10.diff_lpk"], 12, 0.5),
                   ("5.6 planted10 diff lo", v["planted10.diff_lpk.lo"], 8, 0.5),
                   ("5.6 planted10 diff hi", v["planted10.diff_lpk.hi"], 16, 0.5),
                   ("5.6 planted30 rescued lies", v["planted30.rescued_lies"], 18, 0.5),
                   ("5.6 planted30 real lies", v["planted30.poison_real_absorbed"], 0.29, 0.005),
                   ("5.6 clean real lies", v["open.gate_prior_ev.poison_real_absorbed"], 0.08, 0.005),
                   ("5.6 planted30 lies/1000", v["planted30.lies_per_1000"], 65, 0.5),
                   ("5.6 judge alone lies/1000 same runs", v["open.gate_no_test.lies_per_1000"], 41, 0.5),
                   ("5.6 outdated rescued lies", v["outdated30.rescued_lies"], 0, 0),
                   ("5.6 clean updates", v["clean.gate_prior_ev.learn_updates"], 0.63, 0.005),
                   ("5.6 outdated updates", v["outdated30.learn_updates"], 0.48, 0.005)]
        checks += [(f"5.9 route lowers lies/1000 vs judge ({k}): CI high below 0", float(v[f"route.{k}.lpk_vs_judge.hi"] < 0), 1, 0)
                   for k in ("upd_half", "phi4", "llama32")]
        rc = [-v[f"route.{k}.retention_vs_prior"] for k in ("upd_half", "phi4", "llama32")]
        checks += [("5.9/6 route retention cost min (3 points)", min(rc), 0.03, 0.005),
                   ("5.9/6 route retention cost max (4 points)", max(rc), 0.04, 0.005)]
        bad = []
        lines = ["", "## Audit: numbers quoted in the paper vs recomputed", "", "| check | paper | recomputed | ok |", "| --- | --- | --- | --- |"]
        for name, got, want, tol in checks:
            ok = abs(got - want) <= tol + 1e-9
            lines.append(f"| {name} | {want} | {got:.4f} | {'yes' if ok else 'NO'} |")
            if not ok:
                bad.append((name, want, got))
        self.md += lines
        print(f"\naudit: {len(checks) - len(bad)} of {len(checks)} quoted numbers match")
        for name, want, got in bad:
            print(f"  MISMATCH {name}: paper {want}, recomputed {got:.4f}")
        return 1 if bad else 0

    def run(self) -> int:
        self.md = ["# Paper tables and figures, rebuilt from saved results"]
        for step in (self.s51_main, self.s52_judge, self.s54_prior, self.s54_curve, self.s55_gemini,
                     self.s56_updates, self.s57_models, self.s58_real_judges, self.s59_review):
            step()
        status = self.audit()
        (self.out / "tables.md").write_text("\n".join(self.md) + "\n", encoding="utf-8")
        (self.out / "numbers.json").write_text(json.dumps(self.values, indent=1, sort_keys=True), encoding="utf-8")
        print(f"tables -> {self.out / 'tables.md'}\nfigures -> {self.out}/fig*.png\nnumbers -> {self.out / 'numbers.json'}")
        return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_outputs")
    a = ap.parse_args()
    sys.exit(Paper(Path(a.results), Path(a.out)).run())


if __name__ == "__main__":
    main()
