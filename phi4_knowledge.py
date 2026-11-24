"""Does phi4 already know the WikiBigEdit facts it judged? (it was trained after these Feb 2024 changes)
Asks phi4 every question of the logged claims WITHOUT the claim (closed book), then splits its logged
verdicts in results/v9_wbe_phi4 by whether it answered the question correctly itself.
    ollama serve   (phi4 pulled)
    python phi4_knowledge.py            # ~1 h; answers cached in cache/phi4_closed_book.json
"""
import json
import re
from pathlib import Path

import requests

from sura.config import load_config
from sura.wikibigedit import load_records, split_records

TAG = Path("results/v9_wbe_phi4")
CACHE = Path("cache/phi4_closed_book.json")
URL = "http://localhost:11434/api/chat"
PROMPT = "Answer the question with only the answer (a name or short phrase), no explanation.\nQuestion: {q}"


def norm(t: str) -> str:
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return " ".join(w for w in t.split() if w not in ("a", "an", "the"))


def knows(reply: str, truth: str) -> bool:
    r, t = norm(reply), norm(truth)
    return bool(t) and (r.startswith(t) or f" {t} " in f" {r} ")


def ask(q: str) -> str:
    r = requests.post(URL, timeout=300, json={"model": "phi4", "stream": False, "options": {"temperature": 0},
                                              "messages": [{"role": "user", "content": PROMPT.format(q=q)}]})
    return r.json()["message"]["content"].strip()


def facts(cfg):
    """uid -> (question, true answer), with the same ids as the run (see sura/wikibigedit.py)."""
    known, fresh = split_records(load_records(cfg), cfg.data.wbe_n_choices)
    out = {f"cf:{k['case_id']}": (k["prompt"], k["target_true"]) for k in known}
    out.update({f["uid"]: (f["prompt"], f["choices"][0]) for f in fresh})
    return {u: (p.removeprefix("Q: ").removesuffix("\nA:"), a) for u, (p, a) in out.items()}


def main():
    cfg = load_config("configs/wbe.yaml")  # same data settings as the run
    F = facts(cfg)
    claims = [json.loads(l) for f in sorted(TAG.glob("claims_r0.3_s*.jsonl")) for l in f.open(encoding="utf-8")]
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = sorted({c["uid"] for c in claims} - cache.keys())
    for i, u in enumerate(todo, 1):
        cache[u] = ask(F[u][0])
        if i % 50 == 0 or i == len(todo):
            CACHE.parent.mkdir(exist_ok=True)
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"{i}/{len(todo)} asked")
    print("\nclaims judged by phi4 (5 seeds), split by whether phi4 answers the question itself")
    print(f"{'kind':12} {'source':7} {'phi4 knows':11} {'n':>5} {'judged true':>12}")
    for kind in ("new", "update", "poison_fict", "poison_real"):
        for src in (False, True):
            for k in (True, False):
                g = [c for c in claims if c["kind"] == kind and c["has_source"] == src
                     and c["judge"] is not None and knows(cache[c["uid"]], F[c["uid"]][1]) == k]
                if g:
                    print(f"{kind:12} {str(src):7} {str(k):11} {len(g):5} {sum(c['judge'] for c in g) / len(g):12.3f}")
    share = {kind: sum(knows(cache[c["uid"]], F[c["uid"]][1]) for c in claims if c["kind"] == kind)
             / max(1, sum(c["kind"] == kind for c in claims)) for kind in ("new", "update", "poison_fict", "poison_real")}
    print("\nshare of questions phi4 answers correctly by itself:", {k: round(v, 3) for k, v in share.items()})


if __name__ == "__main__":
    main()
