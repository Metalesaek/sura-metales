"""Offline end-to-end check: tiny random model, invented 'CounterFact' facts, no downloads.

  python -m tests.test_offline

Runs every system, both statistical tests, oracle + noisy verifiers, the HF-judge code
path and analyze.py. It checks that the code runs and that the bookkeeping is consistent,
not that the method works (the model is random).
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiment import run  # noqa: E402
from sura.config import load_config  # noqa: E402
from sura.data import Claim, build_world  # noqa: E402
from sura.fictional import RELATIONS, make_facts  # noqa: E402
from sura.gate import sprt_harm, ttest_harm  # noqa: E402
from sura.student import Student  # noqa: E402
from sura.verifier import HFJudge  # noqa: E402

CHAT = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def fake_counterfact(n: int) -> list[dict]:
    rng = random.Random(0)
    towns = [f"Town{i}" for i in range(30)]
    out = []
    for i in range(n):
        t, f = rng.sample(towns, 2)
        out.append(dict(case_id=i, relation="P36", subject=f"Country{i}", prompt=f"The capital of Country{i} is",
                        target_true=t, target_new=f, paraphrases=[f"Country{i} has its capital in"]))
    return out


def tiny_tokenizer(records) -> PreTrainedTokenizerFast:
    corpus = [r["prompt"] + " " + r["target_true"] + " " + r["target_new"] for r in records]
    for f in make_facts(40, random.Random(1)):
        corpus += [f["prompt"]] + list(f["paraphrases"]) + [" ".join(f["pool"])]
    corpus += ["You are a strict fact-checker. Trusted source: Claim: Is the claim true? True False",
               "<|system|> <|user|> <|assistant|>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<unk>", "<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    t = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    t.chat_template = CHAT
    return t


def tiny_model(vocab: int) -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=256, tie_word_embeddings=True))


def check_units():
    # statistical tests on synthetic differences
    harmless, harmful = torch.randn(200) * 0.1, torch.randn(200) * 0.1 + 0.3
    assert not ttest_harm(harmless, 0.05, 0.05)[0] and ttest_harm(harmful, 0.05, 0.05)[0]
    h0, info0 = sprt_harm(lambda i, j: harmless[i:j], 200, 0.05, 0.05, 0.2, 25)
    h1, info1 = sprt_harm(lambda i, j: harmful[i:j], 200, 0.05, 0.05, 0.2, 25)
    assert not h0 and h1, (info0, info1)
    assert info1["items_scored"] < 200, "SPRT should stop early on a clear harm"
    print("  unit: ttest + sprt ok", info0, info1)


def main():
    out = ROOT / "results" / "_offline_test"
    shutil.rmtree(out, ignore_errors=True)
    records = fake_counterfact(600)
    tok = tiny_tokenizer(records)
    cfg = load_config(str(ROOT / "configs" / "default.yaml"), [
        "device=cpu", "dtype=float32", "student.model_name=tiny", "output_dir=results",
        "data.n_fictional_entities=8", "data.suite_size=30", "data.anchor_size=30", "data.heldout_size=30",
        "data.reps=1", "data.chunk_size=8", "train.steps_per_chunk=2", "train.batch_size=8",
        "train.anchor_batch=4", "train.replay_size=8", "eval.batch_size=16", "gate.sprt_batch=10",
        "train.lr=0.005",
    ])
    check_units()

    student = Student(cfg, torch.device("cpu"), torch.float32, model=tiny_model(len(tok)), tokenizer=tok)

    # encoding: labels cover exactly the target tokens
    ids, att, lab = student.encode(["The capital of Country1 is"], ["Town3"])
    n_t = len(tok(" Town3", add_special_tokens=False).input_ids)
    assert int((lab != -100).sum()) == n_t, "labels must cover the target only"

    # rollback restores the adapter exactly
    snap = student.adapter_state()
    opt = student.new_optimizer(0.01)
    student.train_steps(opt, ["The capital of Country1 is"] * 4, ["Town3"] * 4, [], cfg.train, 3, random.Random(0))
    changed = any(not torch.equal(snap[k], v) for k, v in student.adapter_state().items())
    student.load_adapter_state(snap)
    restored = all(torch.equal(snap[k], v) for k, v in student.adapter_state().items())
    assert changed and restored, "train must change the adapter and rollback must restore it"
    print("  unit: encode + commit/rollback ok")

    # world bookkeeping
    from sura.data import screen_known
    known = screen_known(student, records, None, 16)
    w = build_world(cfg, known, 0.3, 0)
    rate = (len(w.poison_fict) + len(w.poison_real)) / (len(w.new) + len(w.poison_fict) + len(w.poison_real))
    assert abs(rate - 0.3) < 0.05, rate
    ids = lambda xs: {c.uid for c in xs}
    assert not (ids(w.heldout) & ids(w.suite)) and not (ids(w.heldout) & ids(w.poison_real))
    assert len(build_world(cfg, known, 0.0, 0).new) == len(w.new), "clean set must not depend on rate"
    print("  unit: world ok", w.summary())

    # HF judge code path with the tiny model as judge
    judge = HFJudge.__new__(HFJudge)
    judge.cache, judge.name, judge.device, judge.bs = {}, "hf:tiny", torch.device("cpu"), 4
    judge.tok, judge.model = tok, tiny_model(len(tok)).eval()
    judge.tok.padding_side = "left"
    judge.true_id = tok.encode("True", add_special_tokens=False)[0]
    judge.false_id = tok.encode("False", add_special_tokens=False)[0]
    v = judge.judge(w.stream[:6], w.evidence)
    tok.padding_side = "right"
    assert len(v) == 6 and all(isinstance(x, bool) for x in v)
    print("  unit: hf judge path ok", v)

    # full runs
    for test in ("sprt", "ttest"):
        cfg.gate.test = test
        run(cfg, ["naive", "surprise", "stability", "gate_full", "gate_no_verif", "gate_no_test"],
            [0.0, 0.3], [0], out / test, student=student, records=records)

    # prior key: unit check on a model that is sure of its answer, then both new systems end to end
    p = "The capital of Country1 is"
    own = student.own_answer(p)
    g_own, g_other = student.prior_gap([p, p], [own, "Town3"], 8)
    assert abs(g_own) < 1e-4, "the model's own answer has zero gap"
    assert student.contradicts([p], ["Town3"], 8, gap=g_other - 1e-3) == [True]
    assert student.contradicts([p], ["Town3"], 8, gap=g_other + 1e-3) == [False]
    print("  unit: prior gap ok", own, round(g_other, 3))
    cfg.gate.prior_gap = 0.5  # tiny random model has small gaps: make the key bite in the end-to-end run
    run(cfg, ["gate_prior", "prior_only"], [0.3], [0], out / "prior", student=student, records=records)
    rows = [json.loads(x) for x in (out / "prior" / "results.jsonl").read_text().splitlines()]
    for r in rows:
        seen = sum(r[k] for k in ("prior_held_true", "prior_held_poison", "prior_passed_true", "prior_passed_poison"))
        assert seen > 0, r
        if r["system"] == "gate_prior":
            assert r["poison_trained"] == 0, "oracle verifier must still block all poison"
    print("  prior key ok", [(r["system"], r["prior_held_true"], r["prior_held_poison"]) for r in rows])

    # legitimate updates + evidence route
    ids_of = lambda xs: [c.uid for c in xs]
    w0 = build_world(cfg, known, 0.3, 0)
    cfg.data.n_updates = 10
    w1 = build_world(cfg, known, 0.3, 0)
    assert len(w1.updates) == 10 and all(c.is_true and c.kind == "update" for c in w1.updates)
    for name in ("heldout", "suite", "anchor", "poison_real", "new", "poison_fict"):
        assert ids_of(getattr(w0, name)) == ids_of(getattr(w1, name)), f"{name} must not change with updates"
    assert not set(ids_of(w1.updates)) & set(ids_of(w1.poison_real + w1.heldout + w1.suite + w1.anchor))
    assert all(w1.evidence.get((c.subject, c.relation)) == c.sentence for c in w1.updates)
    assert all(w1.evidence.get((c.subject, c.relation)) != c.sentence for c in w1.poison_real)
    run(cfg, ["gate_prior", "gate_prior_ev"], [0.3], [0], out / "updates", student=student, records=records)
    rows = {json.loads(x)["system"]: json.loads(x) for x in (out / "updates" / "results.jsonl").read_text().splitlines()}
    ev, gp = rows["gate_prior_ev"], rows["gate_prior"]
    assert ev["prior_rescued_poison"] == 0, "a lie never matches a trusted source"
    assert ev["poison_trained"] == 0 and gp["prior_rescued_true"] == 0
    assert ev["prior_rescued_true"] <= ev["prior_held_true"]
    assert ev["learn_updates"] is not None
    claims = [json.loads(x) for x in (out / "updates" / "claims_r0.3_s0.jsonl").read_text().splitlines()]
    assert len(claims) == len(dict.fromkeys(w1.stream)) and any(c["prior_gap"] is not None for c in claims)
    assert {c["kind"] for c in claims} >= {"new", "update", "poison_real", "poison_fict"}

    from sura.verifier import OllamaJudge
    for text, want in [("True", True), ("**True**", True), ("False.", False), (" false - the source says", False),
                       ("The claim is true.", True), ("I cannot tell", None)]:
        assert OllamaJudge.parse(text) == want, (text, OllamaJudge.parse(text))
    print("  updates ok",{k: ev[k] for k in ("prior_held_true", "prior_rescued_true", "updates_trained", "learn_updates")})

    # review experiments: planted / outdated sources, random selection, open-ended evaluation
    cfg.data.planted_source_share, cfg.data.outdated_source_share = 1.0, 1.0
    w2 = build_world(cfg, known, 0.3, 0)
    for name in ("heldout", "suite", "anchor", "poison_real", "new", "poison_fict", "updates", "stream"):
        assert ids_of(getattr(w1, name)) == ids_of(getattr(w2, name)), f"{name} must not change with sources"
    assert all(w2.evidence.get((c.subject, c.relation)) == c.sentence for c in w2.poison_real), "planted"
    assert all(w2.evidence.get((c.subject, c.relation)) == f"{c.prompt} {c.alt_target}." for c in w2.updates), "outdated"
    cfg.data.outdated_source_share = 0.0
    cfg.eval.open_ended = True
    run(cfg, ["gate_prior_ev", "random"], [0.3], [0], out / "review", student=student, records=records)
    rows = {json.loads(x)["system"]: json.loads(x) for x in (out / "review" / "results.jsonl").read_text().splitlines()}
    ev2 = rows["gate_prior_ev"]
    assert ev2["prior_rescued_poison"] <= ev2["prior_held_poison"]
    for r in rows.values():
        assert 0 <= r["open_retention"] <= 1 and "before_open_retention" in r
    assert rows["random"]["commits"] == 0 and rows["random"]["claims_trained"] > 0
    print("  review ok", {k: ev2[k] for k in ("prior_held_poison", "prior_rescued_poison", "open_poison_real")},
          "random trained", rows["random"]["claims_trained"])
    cfg.data.planted_source_share = 0.0
    cfg.eval.open_ended = False
    cfg.data.n_updates = 0
    cfg.verifier.kind = "noisy"
    run(cfg, ["gate_full", "surprise"], [0.3], [0, 1], out / "noisy", accuracies=[0.6, 1.0],
        student=student, records=records)

    rows = [json.loads(x) for x in (out / "sprt" / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 12, len(rows)
    for r in rows:
        if r["system"] in ("naive", "surprise"):
            assert r["commits"] == 0 and r["rollbacks"] == 0
        if r["system"] == "gate_full" and r["poison_rate"] > 0:
            assert r["poison_trained"] == 0, "oracle verifier must block all poison"
            assert r["verifier_fp"] == 0
        if r["system"] == "naive" and r["poison_rate"] > 0:
            assert r["poison_trained"] > 0
    for sub in ("sprt", "ttest", "noisy"):
        subprocess.run([sys.executable, str(ROOT / "analyze.py"), str(out / sub)], check=True)

    # calibration path: clean drift logging + calibrate_gate.py, then the raw-NLL metric still works
    cfg.verifier.kind, cfg.gate.log_drift = "oracle", True
    run(cfg, ["gate_no_test"], [0.0], [0, 1], out / "calib", student=student, records=records)
    subprocess.run([sys.executable, str(ROOT / "calibrate_gate.py"), str(out / "calib")], check=True)
    cfg.gate.metric, cfg.gate.null_drift = "nll", 0.1
    run(cfg, ["gate_full"], [0.3], [0], out / "nll", student=student, records=records)
    print("\nOFFLINE TEST PASSED")


if __name__ == "__main__":
    main()
