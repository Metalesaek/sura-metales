"""Invented people with facts no pretrained model can know: the 'new knowledge' to learn."""
from __future__ import annotations

import random

FIRST_A = ["Ta", "Ver", "Mo", "Ka", "Li", "Zan", "Ori", "Bel", "Sa", "Ny", "Dra", "Elo",
           "Fen", "Gri", "Hal", "Isa", "Jor", "Kel", "Lum", "Mar", "Quo", "Rhy", "Tor", "Vel"]
FIRST_B = ["lin", "dor", "vash", "mira", "ren", "thea", "nok", "sel", "vin", "ra", "dun", "wyn"]
LAST = ["Tasken", "Morvell", "Quillon", "Draxby", "Fennick", "Halvorn", "Istrel", "Jandrow",
        "Korvath", "Lunsby", "Marrick", "Norwell", "Ostrand", "Pellow", "Rasmor", "Selwick",
        "Thorvane", "Ulvarin", "Vexley", "Wrenford", "Yarrowby", "Zelkirk", "Brannoch", "Castellor"]

# relation -> (main template, paraphrase templates, value pool). "{}" = subject.
RELATIONS = {
    "birthplace": (
        "{} was born in the city of",
        ["The birthplace of {} is", "{} grew up in the city of"],
        ["Lisbon", "Oslo", "Nairobi", "Lima", "Hanoi", "Tunis", "Quito", "Dakar", "Riga",
         "Perth", "Bergen", "Porto", "Lyon", "Graz", "Tbilisi", "Accra", "Cusco", "Sapporo"],
    ),
    "profession": (
        "The profession of {} is",
        ["{} earns a living as a", "By trade, {} is a"],
        ["architect", "pilot", "nurse", "chemist", "carpenter", "journalist", "dentist",
         "farmer", "lawyer", "photographer", "baker", "surgeon", "teacher", "translator"],
    ),
    "language": (
        "The native language of {} is",
        ["The mother tongue of {} is", "{} grew up speaking"],
        ["Portuguese", "Swahili", "Finnish", "Hungarian", "Turkish", "Korean", "Dutch",
         "Greek", "Polish", "Thai", "Romanian", "Czech", "Vietnamese", "Welsh"],
    ),
    "instrument": (
        "The musical instrument played by {} is the",
        ["{} performs in concerts playing the", "On stage, {} plays the"],
        ["violin", "cello", "flute", "trumpet", "harp", "clarinet", "oboe", "banjo",
         "accordion", "saxophone", "trombone", "ukulele"],
    ),
    "sport": (
        "The favorite sport of {} is",
        ["Every weekend, {} plays", "The sport {} enjoys most is"],
        ["cricket", "rugby", "tennis", "fencing", "rowing", "hockey", "volleyball",
         "archery", "curling", "handball", "badminton", "polo"],
    ),
}


def make_names(n: int, rng: random.Random) -> list[str]:
    combos = [f"{a}{b} {c}" for a in FIRST_A for b in FIRST_B for c in LAST]
    if n > len(combos):
        raise ValueError(f"at most {len(combos)} fictional entities")
    rng.shuffle(combos)
    return combos[:n]


def make_facts(n_entities: int, rng: random.Random) -> list[dict]:
    """One dict per (entity, relation): subject, relation, prompt, paraphrases, value, pool."""
    facts = []
    for name in make_names(n_entities, rng):
        for rel, (tmpl, paras, pool) in RELATIONS.items():
            facts.append(dict(
                subject=name, relation=rel,
                prompt=tmpl.format(name),
                paraphrases=tuple(p.format(name) for p in paras),
                value=rng.choice(pool), pool=pool,
            ))
    return facts
