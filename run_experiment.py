"""Run systems x poison rates x seeds (x verifier accuracies) and save results.

Examples
  python run_experiment.py --config configs/smoke.yaml                       # quick check, all systems
  python run_experiment.py --poison_rates 0 0.1 0.3 0.5 --seeds 0 1 2        # main sweep
  python run_experiment.py --systems gate_full surprise --set verifier.kind=noisy \
         --accuracies 0.5 0.7 0.9 1.0 --poison_rates 0.3                    # verifier sweep
  python run_experiment.py --set verifier.kind=hf                            # real LLM judge
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from sura.config import load_config, to_dict
from sura.data import build_world, load_counterfact, screen_known
from sura.metrics import evaluate, evaluate_open
from sura.runner import SYSTEMS, run_system
from sura.student import Student, resolve_device, resolve_dtype
from sura.verifier import make_verifier


def write_claims(out_dir: Path, world, verifier, student, seed, rate, acc):
    """One row per distinct streamed claim: its kind, the judge's verdict and the prior gap
    (when computed), so judge errors and prior misses can be compared claim by claim."""
    verdicts = getattr(verifier, "cache", {})
    gaps = getattr(student, "_prior_cache", {})
    name = f"claims_r{rate}_s{seed}" + (f"_a{acc}" if acc is not None else "") + ".jsonl"
    with (out_dir / name).open("w", encoding="utf-8") as f:
        for c in dict.fromkeys(world.stream):
            f.write(json.dumps(dict(
                uid=c.uid, kind=c.kind, is_true=c.is_true, target=c.target,
                has_source=world.evidence.get((c.subject, c.relation)) == c.sentence,
                judge=verdicts.get((c.uid, c.target)),
                prior_gap=gaps.get((c.prompt, c.target, False)),
            )) + "\n")


def run(cfg, systems, rates, seeds, out_dir: Path, accuracies=None, student=None, records=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks").mkdir(exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False), encoding="utf-8")
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    print(f"device={device} dtype={dtype}")

    student = student or Student(cfg, device, dtype)
    if records is None:
        records = load_counterfact(cfg.data.counterfact_split, cfg.data.n_counterfact)
        slug = cfg.student.model_name.replace("/", "_")
        cache = Path(cfg.cache_dir) / f"known_{slug}_{cfg.data.counterfact_split}_{cfg.data.n_counterfact}.json"
    else:
        cache = None
    known = screen_known(student, records, cache, cfg.eval.batch_size)
    print(f"student knows {len(known)}/{len(records)} screened CounterFact facts")

    accuracies = accuracies or [None]
    shared_verifier = None if cfg.verifier.kind == "noisy" else make_verifier(cfg, 0, device, dtype)
    results_file = out_dir / "results.jsonl"

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        for rate in rates:
            world = build_world(cfg, known, rate, seed)
            print(f"\nseed={seed} poison_rate={rate} sets={world.summary()}")
            student.reset_adapter()
            before = evaluate(student, world, cfg.eval.batch_size)
            if getattr(cfg.eval, "open_ended", False):
                before |= evaluate_open(student, world)
            print("  before learning:", {k: round(v, 3) for k, v in before.items() if v is not None})
            for acc in accuracies:
                if cfg.verifier.kind == "noisy":
                    if acc is not None:
                        cfg.verifier.accuracy = acc
                    verifier = make_verifier(cfg, seed, device, dtype)
                else:
                    verifier = shared_verifier
                for system in systems:
                    bar = tqdm(total=1, desc=f"  {system}", leave=False)

                    def progress(i, n, decision, bar=bar):
                        bar.total = n
                        bar.n = i
                        bar.set_postfix_str(decision)
                        bar.refresh()

                    out = run_system(system, cfg, student, world, verifier, seed, progress)
                    bar.close()
                    row = dict(system=system, seed=seed, poison_rate=rate,
                               verifier_accuracy=cfg.verifier.accuracy if cfg.verifier.kind == "noisy" else None,
                               model=cfg.student.model_name, **out["result"],
                               judge_unparsed=getattr(verifier, "unparsed", None),
                               judge_empty=getattr(verifier, "empty", None),
                               **{f"before_{k}": v for k, v in before.items()})
                    with results_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    name = f"{system}_r{rate}_s{seed}" + (f"_a{acc}" if acc is not None else "")
                    with (out_dir / "chunks" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                        for rec in out["chunks"]:
                            f.write(json.dumps(rec) + "\n")
                    r = out["result"]
                    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
                    print(f"  {system:<14} learn_new={fmt(r['learn_new'])} "
                          f"poison_absorbed={fmt(r['poison_absorbed'])} retention={fmt(r['retention'])} "
                          f"commits={r['commits']} rollbacks={r['rollbacks']} ({r['minutes']:.1f} min)")
                write_claims(out_dir, world, verifier, student, seed, rate, acc)
    print(f"\nresults -> {results_file}")
    return results_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--poison_rates", nargs="+", type=float, default=[0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--accuracies", nargs="+", type=float, default=None,
                    help="noisy-verifier accuracies to sweep (needs --set verifier.kind=noisy)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    ap.add_argument("--tag", default=None, help="results sub-folder name (default: timestamp)")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    tag = a.tag or time.strftime("%Y%m%d-%H%M%S")
    run(cfg, a.systems, a.poison_rates, a.seeds, Path(cfg.output_dir) / tag, a.accuracies)


if __name__ == "__main__":
    main()
