"""Set the test's null level from CLEAN learning, so false alarms stay near alpha.

1. Record clean drift (truth key on, no poison, test off, drift logged):
     python run_experiment.py --systems gate_no_test --poison_rates 0 --seeds 0 1 2 \
            --set gate.log_drift=true --tag calib
2. Read it:
     python calibrate_gate.py results/calib

null_drift = 95th percentile of the per-chunk mean drift seen in clean runs:
the test then fires only when an update harms old knowledge more than 95% of
clean updates do. delta = spread of clean drift (std), the smallest harm worth detecting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--quantile", type=float, default=0.95)
    a = ap.parse_args()
    files = sorted(Path(a.folder, "chunks").glob("gate_no_test_r0.0_*.jsonl"))
    drift = [json.loads(line)["drift"] for f in files for line in f.read_text().splitlines()
             if "drift" in json.loads(line)]
    if not drift:
        raise SystemExit("no drift records: run step 1 with --set gate.log_drift=true")
    d = torch.tensor(drift)
    q = lambda p: torch.quantile(d, p).item()
    print(f"{len(drift)} clean chunks from {len(files)} runs")
    print(f"drift  min={d.min():.4f}  median={q(0.5):.4f}  p90={q(0.9):.4f}  p95={q(0.95):.4f}  max={d.max():.4f}")
    null, delta = q(a.quantile), max(d.std().item(), 0.01)
    print(f"\nsuggested:  --set gate.null_drift={null:.4f} --set gate.delta={delta:.4f}")


if __name__ == "__main__":
    main()
