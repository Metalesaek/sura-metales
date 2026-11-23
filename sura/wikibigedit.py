"""Natural claim stream from WikiBigEdit (real Wikidata changes, Feb-Jul 2024; Apache 2.0).

Mapping onto the claim kinds of data.py
  new          'new' records: a fact added to Wikidata, which the frozen student does not already answer
  update       'update' records: a changed fact, which the frozen student does not already answer
  poison_fict  other 'new' records, streamed with a wrong answer of the same relation (lie about a new fact)
  known        locality questions ('loc') the frozen student answers correctly (via data.screen_known);
               they feed heldout / suite / anchor and the lies about known facts (poison_real), exactly
               as CounterFact does in data.build_world
Lies are drawn only from single-valued relations, so a "wrong" answer is not secretly also true.
Questions become prompts 'Q: <question>\\nA:' and answers the targets.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .data import MAX_POISON_RATE, Claim, World, _covered, _known_claim

# Wikidata relations with (essentially) one value per subject.
SINGLE_VALUED = {
    "P17", "P27", "P19", "P20", "P131", "P495", "P735", "P734", "P103", "P291", "P22", "P25", "P26",
    "P119", "P641", "P1027", "P123", "P184", "P36", "P30", "P50", "P57", "P159", "P740", "P86",
    "P175", "P264", "P1376", "P749", "P413",
}


def prompt_of(question: str) -> str:
    return f"Q: {question.strip()}\nA:"


def load_records(cfg) -> list[dict]:
    """Usable records of the configured files: tag new/update, all text fields present, a relation
    from SINGLE_VALUED with enough distinct answers to build a choice set."""
    d = cfg.data
    out = []
    for name in d.wbe_files:
        out += json.loads((Path(d.wbe_dir) / name).read_text(encoding="utf-8"))
    keep = [r for r in out if r.get("tag") in ("new", "update") and r.get("relation_id") in SINGLE_VALUED
            and all(isinstance(r.get(k), str) and r[k].strip() for k in ("update", "ans", "loc", "loc_ans"))]
    pools: dict[str, set] = {}
    for r in keep:
        pools.setdefault(r["relation_id"], set()).update((r["ans"], r["loc_ans"]))
    k = d.wbe_n_choices
    return [r for r in keep if len(pools[r["relation_id"]]) >= k + 1]


def answer_pools(records) -> dict[str, list[str]]:
    pools: dict[str, set] = {}
    for r in records:
        pools.setdefault(r["relation_id"], set()).update((r["ans"], r["loc_ans"]))
    return {k: sorted(v) for k, v in pools.items()}


def _choices(truth: str, pool: list[str], k: int, key: str) -> tuple[str, ...]:
    """truth + k-1 other answers of the same relation; fixed per record (seeded by its key)."""
    rng = random.Random(key)
    return (truth, *rng.sample([v for v in pool if v != truth], k - 1))


def split_records(records, n_choices: int):
    """(known candidates in CounterFact shape, fresh new/update facts with their choice sets)."""
    pools = answer_pools(records)
    known, fresh, seen_q = [], [], set()
    for i, r in enumerate(records):
        rel, pool = r["relation_id"], pools[r["relation_id"]]
        if r["loc"] not in seen_q:
            seen_q.add(r["loc"])
            wrong = _choices(r["loc_ans"], pool, 2, "loc:" + r["loc"])[1]
            known.append(dict(case_id=f"wbe{i}", relation=rel, subject=r["loc"], prompt=prompt_of(r["loc"]),
                              target_true=r["loc_ans"], target_new=wrong, paraphrases=[]))
        if r["update"] not in seen_q:
            seen_q.add(r["update"])
            fresh.append(dict(uid=f"wbe:{r['subject_id']}:{rel}", tag=r["tag"], relation=rel,
                              subject=r["subject"], prompt=prompt_of(r["update"]), target=r["ans"],
                              paraphrases=[prompt_of(r["rephrase"])] if isinstance(r.get("rephrase"), str)
                              and r["rephrase"].strip() else [],
                              choices=list(_choices(r["ans"], pool, n_choices, "q:" + r["update"]))))
    # one fact per (subject, relation): evidence is keyed by it
    fresh = list({(f["subject"], f["relation"]): f for f in fresh}.values())
    return known, fresh


def screen_fresh(student, fresh, cache_file: Path | None, batch_size: int) -> list[dict]:
    """Keep facts the frozen student does NOT already rank first among their choices (updates: nor above
    their contrast answer, since the data has no old value to compare with)."""
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    k = len(fresh[0]["choices"]) if fresh else 0
    nll = student.mean_nll([f["prompt"] for f in fresh for _ in range(k)],
                           [c for f in fresh for c in f["choices"]], batch_size, adapter=False)
    m = nll.view(len(fresh), k) if fresh else nll
    top = m.argmin(1).tolist() if fresh else []
    # updates are scored against their contrast answer (learn_updates), so that must win before learning too
    out = [f for i, (f, t) in enumerate(zip(fresh, top))
           if t != 0 and (f["tag"] != "update" or m[i, 0] > m[i, 1])]
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def prepare(cfg, student, cache_dir: Path | None, batch_size: int):
    """Load, split and screen once; returns (known records, fresh facts), both cached."""
    from .data import screen_known
    records = load_records(cfg)
    known, fresh = split_records(records, cfg.data.wbe_n_choices)
    random.Random(0).shuffle(known)
    known = known[:cfg.data.wbe_n_screen]
    slug = cfg.student.model_name.replace("/", "_")
    tag = f"{slug}_{len(cfg.data.wbe_files)}f_{cfg.data.wbe_n_screen}_{cfg.data.wbe_n_choices}"
    kc = cache_dir / f"wbe_known_{tag}.json" if cache_dir else None
    fc = cache_dir / f"wbe_fresh_{tag}.json" if cache_dir else None
    return screen_known(student, known, kc, batch_size), screen_fresh(student, fresh, fc, batch_size)


def _fresh_claim(f, kind: str, target: str) -> Claim:
    return Claim(uid=f["uid"], kind=kind, subject=f["subject"], relation=f["relation"], prompt=f["prompt"],
                 target=target, true_target=f["choices"][0],
                 alt_target=f["choices"][1] if target == f["choices"][0] else f["choices"][0],
                 paraphrases=tuple(f["paraphrases"]), choices=tuple(f["choices"]))


def build_world_wbe(cfg, known_records, fresh, poison_rate: float, seed: int) -> World:
    """Same sizes and bookkeeping as data.build_world: data.wbe_n_new true new facts, data.n_updates
    true updates, poison split by data.poison_fict_share; the clean sets do not depend on the rate."""
    if not 0.0 <= poison_rate <= MAX_POISON_RATE:
        raise ValueError(f"poison_rate must be in [0, {MAX_POISON_RATE}]")
    d = cfg.data
    rng = random.Random(seed)
    share = d.poison_fict_share
    news = [f for f in fresh if f["tag"] == "new"]
    upds = [f for f in fresh if f["tag"] == "update"]
    rng.shuffle(news)
    rng.shuffle(upds)
    n_new, n_upd = d.wbe_n_new, d.n_updates
    n_pool = round(round(MAX_POISON_RATE / (1 - MAX_POISON_RATE) * n_new) * share)
    if len(news) < n_new + n_pool or len(upds) < n_upd:
        raise ValueError(f"only {len(news)} new / {len(upds)} update facts after screening, "
                         f"need {n_new + n_pool} / {n_upd}: add files to data.wbe_files")
    new = [_fresh_claim(f, "new", f["choices"][0]) for f in news[:n_new]]
    upd = [_fresh_claim(f, "update", f["choices"][0]) for f in upds[:n_upd]]

    n_poison = round(poison_rate / (1 - poison_rate) * n_new) if poison_rate > 0 else 0
    n_pf = min(round(n_poison * share), n_pool)
    n_pr = n_poison - n_pf
    pf = [_fresh_claim(f, "poison_fict", rng.choice(f["choices"][1:])) for f in news[n_new:n_new + n_pf]]

    known = list(known_records)
    random.Random(seed + 1).shuffle(known)
    need_pr = round(MAX_POISON_RATE / (1 - MAX_POISON_RATE) * n_new) - n_pool
    need = d.heldout_size + d.suite_size + d.anchor_size + need_pr
    if len(known) < need:
        raise ValueError(f"only {len(known)} known WikiBigEdit facts, need {need}: raise data.wbe_n_screen")
    i = 0
    heldout = [_known_claim(r) for r in known[i:i + d.heldout_size]]; i += d.heldout_size
    suite = [_known_claim(r) for r in known[i:i + d.suite_size]]; i += d.suite_size
    anchor = [_known_claim(r) for r in known[i:i + d.anchor_size]]; i += d.anchor_size
    pr = [Claim(uid=f"cf:{r['case_id']}", kind="poison_real", subject=r["subject"], relation=r["relation"],
                prompt=r["prompt"], target=r["target_new"], true_target=r["target_true"],
                alt_target=r["target_true"]) for r in known[i:i + n_pr]]

    stream = (new + pf + pr + upd) * d.reps
    rng.shuffle(stream)
    evidence = {}
    for c in new + upd:  # a real addition or change has a trusted source for a share of subjects
        if _covered(c.subject, d.evidence_coverage):
            evidence[(c.subject, c.relation)] = c.sentence
    for c in pf:  # the source states the true value, so it contradicts the lie (as for fictional facts)
        if _covered(c.subject, d.evidence_coverage):
            evidence[(c.subject, c.relation)] = f"{c.prompt} {c.true_target}."
    return World(stream=stream, new=new, poison_fict=pf, poison_real=pr,
                 suite=suite, anchor=anchor, heldout=heldout, evidence=evidence, updates=upd)
