#!/usr/bin/env python3
"""Pre-push safety check: no participant-level data may enter this repository.

    python check_no_participant_data.py

Flags (1) any CSV/NPZ/XLSX that git would publish, (2) any CSV with an id column,
(3) any text file containing a bracketed list of numeric participant IDs.
Covered by IRB CMUH114-REC3-142.
"""
from __future__ import annotations
import csv, re, subprocess, sys
from pathlib import Path

ID_COLUMNS = {"id", "subject_id", "participant_id"}
BANNED_SUFFIX = {".npz", ".xlsx", ".xls", ".csv"}
ID_LIST = re.compile(r"\[\s*'?\d{1,3}'?(?:\s*,\s*'?\d{1,3}'?){2,}\s*\]")


def publishable() -> list[str] | None:
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [f for f in out.stdout.splitlines() if f]


def main() -> int:
    files = publishable()
    if files is None:
        print("Not a git repository yet: run `git init` first."); return 2
    problems = []
    for f in files:
        p = Path(f)
        if p.suffix.lower() in BANNED_SUFFIX:
            problems.append(f"{f}: data file type {p.suffix}")
            if p.suffix.lower() == ".csv":
                try:
                    with p.open(encoding="utf-8-sig", newline="") as fh:
                        header = next(csv.reader(fh), [])
                    hits = [c for c in header if c.strip().lower() in ID_COLUMNS]
                    if hits: problems.append(f"{f}: participant-level column {hits}")
                except (OSError, StopIteration, UnicodeDecodeError):
                    pass
            continue
        if p.suffix.lower() in {".txt", ".md", ".json", ".py"}:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if p.name != "check_no_participant_data.py" and ID_LIST.search(txt):
                problems.append(f"{f}: list of numeric participant IDs")
    if problems:
        print("FAIL — participant-level data would be published:\n")
        for x in problems: print("  •", x)
        print("\nThese are covered by IRB CMUH114-REC3-142 and must not be distributed.")
        return 1
    print(f"OK — {len(files)} publishable files, no participant-level data found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
