"""Summarise a results folder: mean ± sd tables and charts.

  python analyze.py results/<tag>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["learn_new", "learn_new_paraphrase", "poison_absorbed", "poison_fict_absorbed", "prior_held_true", "prior_held_poison",
           "poison_real_absorbed", "learn_updates", "prior_rescued_true", "prior_rescued_poison", "updates_trained",
           "retention", "commits", "rollbacks", "suite_items_scored", "minutes"]
ORDER = ["naive", "surprise", "stability", "gate_no_verif", "gate_no_test", "gate_full", "prior_only", "gate_prior", "gate_prior_ev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()
    folder = Path(a.folder)
    df = pd.read_json(folder / "results.jsonl", lines=True)
    metrics = [m for m in METRICS if m in df]
    keys = ["system", "poison_rate"] + (["verifier_accuracy"] if df["verifier_accuracy"].notna().any() else [])

    g = df.groupby(keys)[metrics]
    table = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
    table["runs"] = g.size()
    table.to_csv(folder / "summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table)

    systems = [s for s in ORDER if s in set(df["system"])]
    rates = sorted(df["poison_rate"].unique())
    if len(rates) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, m, title in zip(axes, ["poison_absorbed", "learn_new", "retention"],
                                ["Lies the model now asserts", "New facts learned", "Old facts kept"]):
            for s in systems:
                sub = df[df.system == s].groupby("poison_rate")[m].agg(["mean", "std"]).reset_index()
                ax.errorbar(sub["poison_rate"], sub["mean"], yerr=sub["std"].fillna(0),
                            marker="o", capsize=3, label=s, linewidth=2.5 if s == "gate_full" else 1.2)
            ax.set(title=title, xlabel="poison rate in stream", ylim=(0, 1))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("share")
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "poison_sweep.png", dpi=150)
        print(f"chart -> {folder / 'poison_sweep.png'}")

    if "verifier_accuracy" in keys:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for s in systems:
            sub = df[df.system == s].groupby("verifier_accuracy")["poison_absorbed"].mean()
            ax.plot(sub.index, sub.values, marker="o", label=s)
        ax.set(title="Poison absorbed vs verifier accuracy", xlabel="verifier accuracy",
               ylabel="share of lies asserted", ylim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "verifier_sweep.png", dpi=150)
        print(f"chart -> {folder / 'verifier_sweep.png'}")


if __name__ == "__main__":
    main()
