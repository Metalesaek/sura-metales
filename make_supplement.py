"""Build the anonymised supplementary ZIP for double-blind review (TMLR).

  python make_supplement.py             # writes supplement_anonymous.zip next to this script

Takes the files git tracks (so .gitignore applies), replaces author, company and repository
names in every text file, and stops if any identifying string is left.
"""
from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "supplement_anonymous.zip"
REPLACE = [
    (r"https?://github\.com/Metalesaek/sura-metales(\.git)?", "<anonymised repository>"),
    (r"Abdelkader Metales and Miriam Grabenheinrich \(caaise GmbH\)", "Anonymous authors"),
    (r"by Abdelkader Metales and Miriam Grabenheinrich\s*\n?\s*\(caaise GmbH\)", "by anonymous authors"),
    (r"Abdelkader Metales|Miriam Grabenheinrich", "Anonymous author"),
    (r"[\w.]+@caaise\.com", "anonymous@example.com"),
    (r"caaise GmbH|caaise", "Anonymous"),
    (r"Metalesaek|Metales|Grabenheinrich", "Anonymous"),
]
LEAK = re.compile(r"metales|grabenheinrich|caaise|metalesaek|C:\\\\Users\\\\metal|/Users/metal", re.I)
TEXT = {".py", ".md", ".txt", ".yaml", ".yml", ".sh", ".json", ".jsonl", ".csv", ".cfg", ".toml", ""}


def main() -> int:
    files = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True, text=True, check=True).stdout.split("\n")
    files = [f for f in files if f and f not in (OUT.name, Path(__file__).name)]
    leaks = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            p = HERE / f
            if not p.is_file():
                continue
            data = p.read_bytes()
            if p.suffix.lower() in TEXT:
                text = data.decode("utf-8", errors="replace")
                for pat, rep in REPLACE:
                    text = re.sub(pat, rep, text)
                if LEAK.search(text):
                    leaks.append(f)
                data = text.encode("utf-8")
            elif LEAK.search(data.decode("latin-1")):
                leaks.append(f)
            z.writestr("code/" + f, data)
    if leaks:
        OUT.unlink()
        print("Identifying text left in:", *leaks, sep="\n  ")
        return 1
    print(f"{OUT.name}: {len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB, no identifying text found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
