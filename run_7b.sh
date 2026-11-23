#!/usr/bin/env bash
# Qwen2.5-7B on the natural WikiBigEdit stream, one seed per call so an interrupted pod can resume:
# a seed with all six systems in results/v8_wbe_7b/results.jsonl is skipped.
#   nohup bash run_7b.sh > run_7b.log 2>&1 &
set -euo pipefail
TAG=v8_wbe_7b
SYSTEMS="naive surprise random gate_prior_ev sure sure_random"
for SEED in ${SEEDS:-10 11 12 13 14}; do
  if [ -f "results/$TAG/results.jsonl" ] && [ "$(grep -c "\"seed\": $SEED," "results/$TAG/results.jsonl")" -ge 6 ]; then
    echo "seed $SEED done, skipping"; continue
  fi
  python run_experiment.py --config configs/wbe_7b.yaml --systems $SYSTEMS \
    --set verifier.kind=noisy --poison_rates 0.3 --seeds "$SEED" --tag "$TAG"
done
echo "ALL DONE"
