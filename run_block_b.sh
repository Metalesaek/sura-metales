#!/usr/bin/env bash
# Block B runs on the natural WikiBigEdit stream (updates scored top-1 of 8, free answers scored too).
# One seed per call: a seed whose systems are all in results/<tag>/results.jsonl is skipped, so an
# interrupted run resumes where it stopped.
#   cloud (48 GB GPU):  nohup bash run_block_b.sh cloud > block_b_cloud.log 2>&1 &
#   cloud (80 GB GPU), side by side:  cloud7b and cloud15, each with its own log
#   laptop (Ollama + phi4 running):  nohup bash run_block_b.sh judge > block_b_judge.log 2>&1 &
set -euo pipefail

run_tag() {  # config tag "systems" extra-args...
  local CONFIG=$1 TAG=$2 SYSTEMS=$3; shift 3
  local N; N=$(wc -w <<< "$SYSTEMS")
  for SEED in ${SEEDS:-10 11 12 13 14}; do
    if [ -f "results/$TAG/results.jsonl" ] && [ "$(grep -c "\"seed\": $SEED," "results/$TAG/results.jsonl")" -ge "$N" ]; then
      echo "$TAG seed $SEED done, skipping"; continue
    fi
    python run_experiment.py --config "$CONFIG" --systems $SYSTEMS --poison_rates 0.3 --seeds "$SEED" --tag "$TAG" "$@"
  done
}

MAIN="naive surprise random gate_prior_ev sure sure_random"
case "${1:-}" in
  cloud|cloud7b|cloud15)  # cloud = both halves in a row; on an 80 GB GPU run cloud7b and cloud15 side by side
    if [ "$1" != cloud15 ]; then
      run_tag configs/wbe_7b.yaml      v9_wbe_7b      "$MAIN" --set verifier.kind=noisy --set eval.open_ended=true
      run_tag configs/wbe_sure_7b.yaml v9_wbe_sure_7b "naive sure sure_random" --set verifier.kind=noisy
    fi
    if [ "$1" != cloud7b ]; then
      run_tag configs/wbe.yaml         v9_wbe         "$MAIN" --set verifier.kind=noisy --set eval.open_ended=true
      run_tag configs/wbe_sure.yaml    v9_wbe_sure    "naive sure sure_random" --set verifier.kind=noisy
    fi
    ;;
  judge)
    run_tag configs/wbe.yaml v9_wbe_phi4 "gate_no_test gate_prior gate_prior_ev" \
      --set verifier.kind=ollama --set verifier.ollama_model=phi4 --set eval.open_ended=true
    ;;
  *) echo "usage: bash run_block_b.sh cloud|cloud7b|cloud15|judge"; exit 1 ;;
esac
echo "ALL DONE"
