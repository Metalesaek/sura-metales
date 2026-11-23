"""One continual-learning run over the claim stream for one system.

naive          train on every claim
surprise       SuRe-lite: train on surprising claims + surprise-prioritised replay + fast/slow EMA
stability      STABLE-lite: train on every claim, roll back if suite exact-match drops > budget
gate_full      surprise AND verifier, then statistical test vs the original base -> commit / rollback
gate_no_verif  ablation: gate_full without the verifier
gate_no_test   ablation: gate_full without the statistical test
gate_prior     surprise AND verifier AND no prior conflict (the model is not confident in a
               different answer); contradictions are held back for review, not learned
prior_only     ablation: surprise AND no prior conflict, no verifier (the judge-free key alone)
gate_prior_ev  gate_prior, but a held-back claim is learned if a trusted source states the same
               value (evidence route for legitimate updates of known facts)
random         control for surprise: same replay and EMA as `surprise`, but each chunk trains on a
               random subset of the same size as the surprising claims (same training budget)
sure           SuRe as published (Hazard et al., 2025): train on EVERY claim; surprise (mean NLL under
               the fast adapter, before training) only decides what the replay buffer keeps (the most
               surprising); replay every other step; slow adapter = EMA of the fast one after every
               step (beta 0.995); answers with the slow adapter. Run with configs/sure.yaml.
sure_random    control for sure: identical, but the buffer is a uniform random sample (reservoir)
"""
from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from .gate import sprt_harm, ttest_harm
from .metrics import evaluate, evaluate_open

SYSTEMS = ("naive", "surprise", "stability", "gate_full", "gate_no_verif", "gate_no_test",
           "gate_prior", "prior_only", "gate_prior_ev", "random", "sure", "sure_random")
SURE = ("sure", "sure_random")
PRIOR = ("gate_prior", "prior_only", "gate_prior_ev")
JUDGED = ("gate_full", "gate_no_test", "gate_prior", "gate_prior_ev")
NO_TEST = ("gate_no_test", "gate_prior", "prior_only", "gate_prior_ev")


class ReplayBuffer:
    """Accepted claims, keeping the most surprising when full."""

    def __init__(self, capacity: int):
        self.capacity, self.items = capacity, []  # (surprise, claim)

    def add(self, scored):
        self.items += scored
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda t: -t[0])
            self.items = self.items[:self.capacity]

    def sample(self, n: int, rng: random.Random):
        return [c for _, c in rng.sample(self.items, min(n, len(self.items)))]


class SuReBuffer:
    """SuRe buffer: the `capacity` most surprising distinct claims seen so far (one task, since the
    stream has no task boundaries). A claim seen again keeps its highest surprise."""

    def __init__(self, capacity: int):
        self.capacity, self.best = capacity, {}  # (uid, target) -> (surprise, claim)

    def add(self, scored, rng=None):
        for s, c in scored:
            k = (c.uid, c.target)
            if k not in self.best or s > self.best[k][0]:
                self.best[k] = (s, c)
        if len(self.best) > self.capacity:
            keep = sorted(self.best.items(), key=lambda kv: -kv[1][0])[:self.capacity]
            self.best = dict(keep)

    @property
    def claims(self):
        return [c for _, c in self.best.values()]

    def sample(self, n: int, rng: random.Random):
        items = self.claims
        return rng.sample(items, min(n, len(items)))


class ReservoirBuffer(SuReBuffer):
    """Control: a uniform random sample of the distinct claims seen so far (reservoir sampling)."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.seen, self.seen_keys = 0, set()

    def add(self, scored, rng=None):
        for s, c in scored:
            k = (c.uid, c.target)
            if k in self.seen_keys:  # every distinct claim gets one chance to enter
                continue
            self.seen_keys.add(k)
            self.seen += 1
            if len(self.best) < self.capacity:
                self.best[k] = (s, c)
            else:
                j = rng.randrange(self.seen)
                if j < self.capacity:
                    del self.best[rng.choice(list(self.best))]
                    self.best[k] = (s, c)


def sure_steps(student, opt, current, buffer, T, rng, slow, log, anchor_prompts=()):
    """SuRe training on one chunk: every step trains the fast adapter on the current claims, every
    `sure_replay_every`-th step adds replayed claims; after EVERY step the slow adapter moves
    towards the fast one, slow <- beta*slow + (1-beta)*fast. SuRe has no KL anchor; with
    train.kl_weight > 0 the same anchor as the other systems is added (robustness variant)."""
    beta = getattr(T, "sure_beta", 0.995)
    every = getattr(T, "sure_replay_every", 2)
    n_rep = getattr(T, "sure_replay_size", 8)
    student.model.train()
    losses = []
    for step in range(T.steps_per_chunk):
        batch = rng.sample(current, min(T.batch_size, len(current)))
        rep = buffer.sample(n_rep, rng) if step % every == 0 else []
        log["replay_items"] += len(rep)
        log["replay_poison"] += sum(not c.is_true for c in rep)
        log["replay_poison_real"] += sum(c.kind == "poison_real" for c in rep)
        items = batch + rep
        s, c = student._target_nll(*student.encode([x.prompt for x in items], [x.target for x in items]))
        loss = (s / c).mean()
        if T.kl_weight > 0 and anchor_prompts:  # same anchor as Student.train_steps
            a = rng.sample(list(anchor_prompts), min(T.anchor_batch, len(anchor_prompts)))
            ids, att, _ = student.encode(a, None)
            with torch.no_grad(), student.model.disable_adapter():
                ref = F.log_softmax(student.model(input_ids=ids, attention_mask=att).logits.float(), -1)
            cur = F.log_softmax(student.model(input_ids=ids, attention_mask=att).logits.float(), -1)
            kl = F.kl_div(cur, ref, log_target=True, reduction="none").sum(-1)
            loss = loss + T.kl_weight * (kl * att).sum() / att.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.trainable(), 1.0)
        opt.step()
        losses.append(loss.item())
        fast = student.adapter_state()
        slow = {k: beta * slow[k] + (1 - beta) * fast[k] for k in slow}
    student.model.eval()
    return slow, sum(losses) / max(len(losses), 1)


def run_system(system: str, cfg, student, world, verifier, seed: int, progress=None) -> dict:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system}")
    T, G, bs = cfg.train, cfg.gate, cfg.eval.batch_size
    rng = random.Random(seed * 7919 + SYSTEMS.index(system))
    torch.manual_seed(seed)

    student.reset_adapter()
    opt = student.new_optimizer(T.lr)
    slow = student.adapter_state() if system in ("surprise", "random") + SURE else None
    if system == "sure":
        buffer = SuReBuffer(getattr(T, "sure_capacity", 300))
    elif system == "sure_random":
        buffer = ReservoirBuffer(getattr(T, "sure_capacity", 300))
    else:
        buffer = ReplayBuffer(T.buffer_capacity)
    sure_insert = getattr(T, "sure_insert", "before")
    anchor_sentences = [c.sentence for c in world.anchor]

    sp, st = [c.prompt for c in world.suite], [c.target for c in world.suite]
    sa = [c.alt_target for c in world.suite]
    base_suite_nll = student.mean_nll(sp, st, bs, adapter=False)
    base_suite_alt = student.mean_nll(sp, sa, bs, adapter=False) if G.metric == "margin" else None

    def harm_d(i: int, j: int):
        """Per-item harm for suite items i..j-1 vs the ORIGINAL base (positive = worse).
        margin: loss of preference for the true answer, (alt-true NLL)_base - (alt-true NLL)_now
        nll:    rise in NLL of the true answer"""
        now_true = student.mean_nll(sp[i:j], st[i:j], bs)
        if G.metric == "nll":
            return now_true - base_suite_nll[i:j]
        now_alt = student.mean_nll(sp[i:j], sa[i:j], bs)
        return (base_suite_alt[i:j] - base_suite_nll[i:j]) - (now_alt - now_true)

    def suite_em():
        return student.prefers(sp, st, sa, bs).float().mean().item()

    em_now = suite_em() if system == "stability" else None
    log = dict(commits=0, rollbacks=0, claims_trained=0, poison_trained=0, suite_items_scored=0,
               verifier_tp=0, verifier_fp=0, verifier_tn=0, verifier_fn=0,
               prior_held_true=0, prior_held_poison=0, prior_passed_true=0, prior_passed_poison=0,
               prior_rescued_true=0, prior_rescued_poison=0, updates_trained=0)
    if system in SURE:
        log |= dict(replay_items=0, replay_poison=0, replay_poison_real=0)
        buffer_shares = []  # share of lies in the buffer after each chunk
    judged, prior_seen = set(), set()
    chunks = []
    t0 = time.time()

    stream, C = world.stream, cfg.data.chunk_size
    n_chunks = (len(stream) + C - 1) // C
    for ci in range(n_chunks):
        chunk = stream[ci * C:(ci + 1) * C]
        # SuRe scores surprise with the fast adapter; SuRe-lite/random score with the slow one
        with student.using(slow if system in ("surprise", "random") else None):
            surprise = student.mean_nll([c.prompt for c in chunk], [c.target for c in chunk], bs).tolist()
        surprising = [s > G.surprise_tau for s in surprise]

        if system in ("naive", "stability") + SURE:
            keep = [True] * len(chunk)
        elif system in ("surprise", "gate_no_verif"):
            keep = surprising
        elif system == "random":  # same number of claims as surprise would pick, chosen at random
            pick = set(rng.sample(range(len(chunk)), sum(surprising)))
            keep = [i in pick for i in range(len(chunk))]
        rescued = {}
        if system in PRIOR:
            # prior key: hold back surprising claims that contradict a confident belief
            cand = [c for c, s in zip(chunk, surprising) if s]
            flags = student.contradicts([c.prompt for c in cand], [c.target for c in cand], bs,
                                        G.prior_gap, getattr(G, "prior_articles", False)) if cand else []
            conflict = dict(zip(cand, flags))
            if system == "gate_prior_ev":
                # evidence route: a held-back claim is learned if a trusted source states the same value
                rescued = {c: x and world.evidence.get((c.subject, c.relation)) == c.sentence
                           for c, x in conflict.items()}
            for c, x in conflict.items():
                if (c.uid, c.target) not in prior_seen:
                    prior_seen.add((c.uid, c.target))
                    truth = "true" if c.is_true else "poison"
                    log[f"prior_{'held' if x else 'passed'}_{truth}"] += 1
                    if rescued.get(c):
                        log[f"prior_rescued_{truth}"] += 1
            surprising = [s and not conflict.get(c, False) for c, s in zip(chunk, surprising)]
            if system == "prior_only":
                keep = surprising
        if system in JUDGED:  # judge only the claims still standing
            cand = [c for c, s in zip(chunk, surprising) if s]
            verdict = dict(zip(cand, verifier.judge(cand, world.evidence))) if cand else {}
            for c, v in verdict.items():
                if (c.uid, c.target) in judged:
                    continue
                judged.add((c.uid, c.target))
                key = ("verifier_tp" if c.is_true else "verifier_fp") if v else \
                      ("verifier_fn" if c.is_true else "verifier_tn")
                log[key] += 1
            keep = [s and verdict.get(c, False) for c, s in zip(chunk, surprising)]
        if rescued:  # source-backed claims skip the judge
            keep = [k or rescued.get(c, False) for c, k in zip(chunk, keep)]

        kept =[(s, c) for s, c, k in zip(surprise, chunk, keep) if k]
        rec = dict(chunk=ci, kept=len(kept), poison_kept=sum(not c.is_true for _, c in kept))
        if not kept:
            chunks.append(rec | dict(decision="skip"))
            continue

        if system in SURE:
            scored = list(zip(surprise, chunk))
            if sure_insert == "before":  # Algorithm 1: insert before training on the chunk
                buffer.add(scored, rng)
            slow, rec["loss"] = sure_steps(student, opt, chunk, buffer, T, rng, slow, log, anchor_sentences)
            if sure_insert == "after":   # variant: score with the trained fast adapter, then insert
                after = student.mean_nll([c.prompt for c in chunk], [c.target for c in chunk], bs).tolist()
                buffer.add(list(zip(after, chunk)), rng)
            items = buffer.claims
            rec["buffer_poison_share"] = sum(not c.is_true for c in items) / max(len(items), 1)
            buffer_shares.append(rec["buffer_poison_share"])
            log["claims_trained"] += len(kept)
            log["poison_trained"] += rec["poison_kept"]
            log["updates_trained"] += sum(c.kind == "update" for _, c in kept)
            chunks.append(rec | dict(decision="train"))
            if progress:
                progress(ci + 1, n_chunks, "train")
            continue

        train = [c for _, c in kept]
        if system != "naive" and system != "stability":
            train += buffer.sample(T.replay_size, rng)
        prompts, targets = [c.prompt for c in train], [c.target for c in train]

        if system in ("naive", "surprise", "random"):
            rec["loss"] = student.train_steps(opt, prompts, targets, anchor_sentences, T, T.steps_per_chunk, rng)
            if system in ("surprise", "random"):  # slow learner = EMA of the fast adapter
                fast = student.adapter_state()
                slow = {k: T.ema_decay * slow[k] + (1 - T.ema_decay) * fast[k] for k in slow}
                buffer.add(kept)
            decision = "train"
        else:
            snap, opt_snap = student.adapter_state(), student.snapshot_optimizer(opt)
            rec["loss"] = student.train_steps(opt, prompts, targets, anchor_sentences, T, T.steps_per_chunk, rng)
            if system == "stability":
                em_new = suite_em()
                log["suite_items_scored"] += len(sp)
                ok, rec["em_drop"] = (em_now - em_new) <= G.stability_em_budget, em_now - em_new
            elif system in NO_TEST or G.test == "none":
                ok = True
                if G.log_drift:  # record clean drift for calibrate_gate.py (not used to decide)
                    rec["drift"] = harm_d(0, len(sp)).mean().item()
            elif G.test == "ttest":
                harm, info = ttest_harm(harm_d(0, len(sp)), G.null_drift + G.delta, G.alpha)
                ok = not harm
                rec |= info
                log["suite_items_scored"] += info["items_scored"]
            else:  # sprt
                harm, info = sprt_harm(harm_d, len(sp), G.delta, G.alpha, G.beta, G.sprt_batch,
                                       mu0=G.null_drift)
                ok = not harm
                rec |= info
                log["suite_items_scored"] += info["items_scored"]

            if ok:
                log["commits"] += 1
                buffer.add(kept)
                if system == "stability":
                    em_now = em_new
                decision = "commit"
            else:
                student.load_adapter_state(snap)
                opt.load_state_dict(opt_snap)
                log["rollbacks"] += 1
                decision = "rollback"

        if decision in ("train", "commit"):
            log["claims_trained"] += len(kept)
            log["poison_trained"] += rec["poison_kept"]
            log["updates_trained"] += sum(c.kind == "update" for _, c in kept)
        chunks.append(rec | dict(decision=decision))
        if progress:
            progress(ci + 1, n_chunks, decision)

    if system in SURE:
        items = buffer.claims
        log["buffer_size"] = len(items)
        log["buffer_poison_share"] = sum(not c.is_true for c in items) / max(len(items), 1)
        log["buffer_poison_real_share"] = sum(c.kind == "poison_real" for c in items) / max(len(items), 1)
        log["buffer_poison_share_mean"] = sum(buffer_shares) / max(len(buffer_shares), 1)
        streamed = list(dict.fromkeys((c.uid, c.target, c.is_true) for c in stream))
        log["stream_poison_share"] = sum(not t for *_, t in streamed) / max(len(streamed), 1)
    with student.using(slow):
        result = evaluate(student, world, bs)
        if getattr(cfg.eval, "open_ended", False):
            result |= evaluate_open(student, world)
    result |= log
    result["minutes"] = (time.time() - t0) / 60
    result["verifier"] = getattr(verifier, "name", "none") if system in JUDGED else "none"
    return dict(result=result, chunks=chunks)
