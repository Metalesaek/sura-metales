# Surprise Alone Learns Lies: evidence-gated continual learning

Code and saved results for the paper *Surprise Alone Learns Lies: Evidence-Gated Continual
Learning for LLMs from Live Claims*. A student LLM learns through a LoRA adapter from a stream of
true and false claims; each system decides which claims it may learn from (surprise, a truth
key, a statistical regression test, the model's prior, an evidence route).

Stack: PyTorch, Hugging Face `transformers`, `peft` (LoRA), `datasets`, `scipy`, `pandas`.

## 1. Reproduce every table and figure from the saved results (no GPU, about a minute)

```bash
python reproduce_paper.py            # reads results/, writes paper_outputs/
```

- `paper_outputs/tables.md`: every table of the paper, recomputed, with the source folder of each
- `paper_outputs/fig2..fig5*.png`: the figures
- `paper_outputs/numbers.json`: every recomputed number
- An **audit** compares 407 numbers quoted in the paper with the recomputed values and exits
  with status 1 if any differs by more than its rounding.

The folder of the main sweep was originally named `mai`; the script accepts `main` or `mai`.

## 2. Setup to rerun experiments (Windows + Git Bash; Linux works the same)

```bash
pip install uv
bash setup_env.sh                    # .venv, PyTorch (CUDA 12.4), requirements; `bash setup_env.sh cu121` for older drivers
source .venv/Scripts/activate        # Linux: source .venv/bin/activate
python -m tests.test_offline         # end-to-end check with a tiny random model, no downloads (~2 min, CPU)
```

Llama models are gated: accept the license on Hugging Face and `export HF_TOKEN=...`.
The Gemini judge reads `GEMINI_API_KEY`; the Ollama judges need a running Ollama server.

## 3. Rerun the experiments behind each section

All runs used one 16 GB laptop GPU. Every run writes `results/<tag>/results.jsonl` (one row per
run), `chunks/` (every learning decision) and, since Section 5.8, `claims_*.jsonl` (per-claim
judge verdict and prior gap).

| Paper | Results folder | Command |
| --- | --- | --- |
| 5.1, Table 3, Fig. 2 | `main` | `python run_experiment.py --poison_rates 0 0.1 0.3 0.5 --seeds 0 1 2 --set gate.metric=nll --tag main` |
| 5.3 calibration | `calib` | `python run_experiment.py --systems gate_no_test --poison_rates 0 --seeds 0 1 2 --set gate.log_drift=true --tag calib`, then `python calibrate_gate.py results/calib` |
| 5.2, 5.3, Fig. 3, Table 4 | `judge_sweep` | `python run_experiment.py --systems gate_full gate_no_test --set verifier.kind=noisy --accuracies 0.6 0.7 0.8 0.9 --poison_rates 0.3 --seeds 0 1 2 --set gate.null_drift=0.4459 --set gate.delta=0.2034 --tag judge_sweep` |
| 5.4, Fig. 4 | `prior_sweep` | `python run_experiment.py --systems gate_no_test gate_prior prior_only --set verifier.kind=noisy --accuracies 0.6 0.7 0.8 0.9 --poison_rates 0.3 --seeds 0 1 2 --tag prior_sweep` |
| 5.4, 5.7, Fig. 5, Table 5 | `prior_diag_<model>_seed7.csv` | `python diagnose_prior.py --per_kind 200 --seed 7 [--set student.model_name=Qwen/Qwen2.5-3B-Instruct]` |
| 5.5 | `gemini_cov50` | `python run_experiment.py --systems gate_full gate_no_test --set verifier.kind=gemini --set data.evidence_coverage=0.5 --poison_rates 0.3 --seeds 0 1 2 --set gate.null_drift=0.4459 --set gate.delta=0.2034 --tag gemini_cov50` |
| 5.6, Table 6 | `v2_updates`, `v2_updates_cov50` | `python run_experiment.py --systems gate_no_test gate_prior gate_prior_ev --set verifier.kind=noisy --accuracies 0.7 0.9 --poison_rates 0.3 --seeds 10 11 12 13 14 --set data.n_updates=100 --tag v2_updates` (add `--set data.evidence_coverage=0.5 --tag v2_updates_cov50` for half the sources) |
| 5.7, Table 7 | `v3_*` | as `v2_updates_cov50`, adding `--set student.model_name=Qwen/Qwen2.5-3B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`; the surprise-only baselines use `--systems surprise` without a verifier |
| 5.8, Tables 8 and 9 | `v4_phi4`, `v4_llama32` | `python run_experiment.py --systems gate_no_test gate_prior gate_prior_ev --set verifier.kind=ollama --set verifier.ollama_model=phi4 --poison_rates 0.3 --seeds 10 11 12 13 14 --set data.n_updates=100 --set data.evidence_coverage=0.5 --tag v4_phi4` (and `llama3.2`, tag `v4_llama32`) |
| 5.1 random control and open-ended answers | `v5_random_open` | `python run_experiment.py --systems surprise random gate_no_test gate_prior_ev --set verifier.kind=noisy --accuracies 0.9 --poison_rates 0.3 --seeds 10 11 12 13 14 --set data.n_updates=100 --set data.evidence_coverage=0.5 --set eval.open_ended=true --tag v5_random_open` |
| 5.6 planted and outdated sources | `v5_planted10`, `v5_planted30`, `v5_outdated30` | `python run_experiment.py --systems gate_prior_ev --set verifier.kind=noisy --accuracies 0.9 --poison_rates 0.3 --seeds 10 11 12 13 14 --set data.n_updates=100 --set data.evidence_coverage=0.5 --set data.planted_source_share=0.1 --tag v5_planted10` (0.3 for `v5_planted30`; `--set data.outdated_source_share=0.3 --tag v5_outdated30`) |
| 5.9 four conditions | `prior_diag_<model>_seed7_4cond.csv` | `python diagnose_prior.py --per_kind 200 --seed 7 --four_conditions --set data.n_updates=100 [--set student.model_name=...]` |
| 5.9, 6 analyses of saved results | (reads the folders above) | `python review_analyses.py` (also part of `reproduce_paper.py`) |

Defaults live in `configs/default.yaml`; any key can be overridden with `--set key=value`. The
default student is Qwen2.5-1.5B-Instruct. Seeds 10-14 were held out: the prior-key threshold
(gap >= 6) was fixed on seeds 0 and 7 before they were run.

## Exact environment, models and judges

- Packages: `requirements-lock.txt` (exact versions of every run: torch 2.6.0+cu124, transformers 5.20.1,
  peft 0.21.2, datasets 5.1.0); Python 3.11, Windows 11, one 16 GB NVIDIA laptop GPU, bfloat16.
- Students (Hugging Face): `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`.
- Data: CounterFact from `azhx/counterfact` (train split shuffled with seed 0, first 3,000 records, screened for facts the student knows;
  the screen is cached in `cache/`). Invented people come from `sura/fictional.py`, seeded by the world seed.
- Judges: `gemini-2.5-flash` (REST API, temperature 0, thinking budget 0, at most 20 output tokens);
  Ollama `phi4` and `llama3.2` (temperature 0, at most 10 tokens). The first `true`/`false` word in the
  answer is the verdict; the count of unparseable answers is saved per run (`judge_unparsed`, 0 in every Ollama run).
- Judge prompt (identical for every judge, `sura/verifier.py`):

```
system: You are a strict fact-checker. Decide whether the claim is true. If a trusted source is given,
        it is correct. Answer with one word: True or False.
user:   Trusted source: <source sentence>      (only when the claim's subject has a source)
        Claim: <claim sentence>
        Is the claim true?
```

- Simulated judge: correct with probability = accuracy, decided by a hash of (seed, claim id, value), so the
  same claim always gets the same verdict within a seed.

## Data

| Set | Source | Role |
| --- | --- | --- |
| New facts | 120 invented people x 5 relations (`sura/fictional.py`) | true claims no model can know |
| Lies about invented people | same people, wrong value | contradict the trusted source |
| Real-world lies | CounterFact counterfactuals on facts the student knows | plausible lies about known facts |
| Legitimate updates | CounterFact facts whose counterfactual is, in this world, the new truth (`data.n_updates`) | true news that contradicts the model |
| Known facts | CounterFact facts the student knows | regression suite / KL anchor / held-out retention |

## Systems

| System | Learns from | Safety check |
| --- | --- | --- |
| `naive` | every claim | none |
| `surprise` | surprising claims, replay, fast/slow EMA (SuRe-lite) | none |
| `stability` | every claim | rollback if suite exact-match drops (STABLE-lite) |
| `gate_no_test` | surprising and verified claims (truth key) | none |
| `gate_full` | surprising and verified claims | SPRT against the original base, commit or rollback |
| `gate_no_verif` | surprising claims | SPRT (no truth key) |
| `prior_only` | surprising claims without prior conflict | none, no judge |
| `gate_prior` | surprising, verified, no prior conflict | none |
| `gate_prior_ev` | as `gate_prior`, plus held-back claims a trusted source confirms | none |
| `random` | random subset of each chunk, same size as the surprising claims, same replay and EMA as `surprise` | none (control) |

Judges (`verifier.kind`): `oracle`, `noisy` (simulated, accuracy set by `--accuracies`), `hf`,
`ollama`, `gemini`.

## Layout

```
reproduce_paper.py   tables, figures and number audit from saved results
review_analyses.py   extra analyses of saved results (what the prior key detects, attacker, costs)
run_experiment.py    runs systems x poison rates x seeds (x judge accuracies)
diagnose_prior.py    prior-gap separation (AUC, trade-off curve) on a fresh world
calibrate_gate.py    null drift for the statistical test
analyze.py           quick tables and charts for one results folder
sura/config.py       YAML config, inheritance, CLI overrides
sura/fictional.py    invented entities and relations
sura/data.py         CounterFact loading, known-fact screening, stream and evaluation sets
sura/student.py      LoRA student: scoring, training with KL anchor, snapshot and rollback, prior gap
sura/verifier.py     oracle, noisy, HF, Ollama and Gemini judges
sura/gate.py         paired t-test and SPRT
sura/runner.py       the ten systems
sura/metrics.py      evaluation (top-1 for invented facts, pairwise for real facts)
tests/test_offline.py  end-to-end test without downloads
results/             saved results of every run in the paper
```
