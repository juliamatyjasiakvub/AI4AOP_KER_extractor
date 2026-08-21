#!/usr/bin/env python
"""
Refuse to publish a repository that carries papers or third-party data.

Why this is a script and not a checklist
----------------------------------------
This tool reads full-text articles. It writes what it reads into
`aop_rag.db` — not only the extracted claims and their quotations, but the
paper text itself, chunk by chunk, so that a quotation can be located and a
page number reported. A modest corpus leaves hundreds of kilobytes of
publisher text in that file.

`aop_rag.db` was tracked in git. So was `example_files/*.pdf`. Adding patterns
to `.gitignore` does not help with either: gitignore is not consulted for files
already tracked, and it has no bearing at all on what is already in the
history — a clone of a repository whose history ever contained a PDF still
downloads that PDF.

Untracking is therefore necessary and not sufficient, and "I remembered to
delete it" is not a control. This script is the control. It looks in three
places, because a file can hide in any of them:

    the working tree   — what is on disk right now
    the git index      — what is tracked and would be committed
    the git history    — what any clone would receive, forever

Exit code is 0 when nothing blocking was found, 1 otherwise, so it can gate a
release in CI or a pre-push hook.

    python tools/check_publishable.py
    python tools/check_publishable.py --strict   # advisories also fail
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Extensions that must never be committed. Databases carry extracted paper
#: text; PDFs are the papers themselves.
BLOCKING_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pdf"}

#: Filenames that mean credentials regardless of extension.
BLOCKING_NAMES = {".env", ".env.local", "secrets.json", "credentials.json"}

#: Corpus manifests. A CSV of DOIs, titles and publishers names the exact
#: articles a corpus was built from. It carries none of their text, so it is
#: not a licensing problem in the way a database is — but it identifies the
#: papers, and a repository that is meant to ship a tool rather than a reading
#: list should not carry one. Matched by name and, below, by shape.
BLOCKING_NAMES |= {"licences.csv", "licenses.csv", "corpus.csv", "dois.csv"}

#: A CSV whose header names a DOI column is a corpus manifest whatever it is
#: called. Read only the first line — enough to classify, and it avoids
#: pulling a large file into memory to answer a small question.
def _is_doi_manifest(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline().lower()
    except OSError:
        return False
    fields = {field.strip().strip('"') for field in header.split(",")}
    return "doi" in fields

#: Bundled third-party datasets. Not a leak of anyone's papers, but somebody
#: else's data being redistributed, which needs a licence decision rather than
#: a silent inclusion.
THIRD_PARTY_HINTS = ("aopwiki_data", "aop-wiki-xml")


class Finding:
    def __init__(self, where: str, what: str, detail: str, blocking: bool = True):
        self.where, self.what, self.detail, self.blocking = where, what, detail, blocking


def _git(*args: str) -> list[str]:
    """Run a git command, returning [] if this is not a checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def _is_risky(path: str) -> bool:
    name = Path(path).name.lower()
    return Path(name).suffix in BLOCKING_SUFFIXES or name in BLOCKING_NAMES


# ---------------------------------------------------------------------------
# The three places
# ---------------------------------------------------------------------------

def check_index() -> list[Finding]:
    """Files git is tracking now — these would go into the next commit."""
    findings = []
    for path in _git("ls-files"):
        if _is_risky(path):
            findings.append(Finding(
                "tracked", path,
                "tracked by git; `git rm --cached` it before committing",
            ))
        elif (
            any(hint in path.lower() for hint in THIRD_PARTY_HINTS)
            # A placeholder or a note explaining the directory is ours, not
            # AOP-Wiki's. Flagging it asks the reader to review the licence of
            # a file we wrote, and a check that cries wolf is a check that
            # gets skimmed past when it finally matters.
            and Path(path).name not in (".gitkeep", ".gitignore", "README.md")
        ):
            findings.append(Finding(
                "tracked", path,
                "third-party dataset — confirm its licence permits "
                "redistribution, or fetch it at install time instead",
                blocking=False,
            ))
    return findings


def check_history() -> list[Finding]:
    """
    Files any clone would receive, including ones deleted long ago.

    This is the check people skip. Removing a PDF and committing the removal
    leaves the PDF fully intact in the history; anyone cloning still gets it.
    """
    findings = []
    seen: set[str] = set()
    # Every path named by any commit, NOT `--diff-filter=A`. Filtering to
    # additions misses a file that was moved: git records that as a rename, so
    # a PDF committed at the repository root and later moved into
    # `example_files/` is reported at neither path. The unfiltered listing is
    # noisier and correct, and correctness is the entire point of this check.
    for line in _git("log", "--all", "--pretty=format:", "--name-only"):
        path = line.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        if _is_risky(path):
            findings.append(Finding(
                "history", path,
                "present in git history — still downloaded by every clone even "
                "if deleted from the current tree; needs history rewriting",
            ))
    return findings


def check_working_tree() -> list[Finding]:
    """Risky files on disk. Not blocking if ignored, but worth naming."""
    findings = []
    ignored = set(_git("ls-files", "--others", "--ignored", "--exclude-standard"))
    tracked = set(_git("ls-files"))
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith((".git/", ".venv/")) or rel in tracked:
            continue
        if _is_risky(rel) or _is_doi_manifest(path):
            findings.append(Finding(
                "on disk", rel,
                "ignored by git, so it will not be committed — but do not "
                "`git add -f` it" if rel in ignored else
                "NOT ignored by git; add a pattern for it before the next commit",
                blocking=rel not in ignored,
            ))
    return findings


def check_database_contents() -> list[Finding]:
    """
    Say plainly how much paper text each database on disk holds.

    Not blocking on its own — these files are ignored — but the number is the
    argument for why they must stay that way, and it is not obvious from the
    filename that a 1 MB database is mostly somebody's article.
    """
    findings = []
    for db in REPO.rglob("*.db"):
        if ".venv" in db.parts or ".git" in db.parts:
            continue
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            chars = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(text)), 0) FROM paper_chunk"
            ).fetchone()[0]
            quotes = conn.execute(
                "SELECT COUNT(*) FROM evidence_spans"
            ).fetchone()[0]
            conn.close()
        except sqlite3.Error:
            continue
        if chars or quotes:
            findings.append(Finding(
                "database", db.relative_to(REPO).as_posix(),
                f"holds {chars:,} characters of paper text and {quotes} "
                f"verbatim quotation(s) — publisher content, never publishable",
                blocking=False,
            ))
    return findings


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="treat advisories as failures too")
    args = parser.parse_args()

    if not _git("rev-parse", "--git-dir"):
        print("Not a git checkout — only the working tree can be checked.\n")

    findings = (
        check_index() + check_history() + check_working_tree()
        + check_database_contents()
    )
    blocking = [f for f in findings if f.blocking]
    advisory = [f for f in findings if not f.blocking]

    print("=" * 78)
    print("PUBLICATION CHECK — papers and third-party data")
    print("=" * 78)

    if not findings:
        print("\nNothing found. No databases, PDFs or credentials are tracked,")
        print("in the history, or sitting untracked in the working tree.")
        return 0

    for title, group in (("MUST FIX", blocking), ("REVIEW", advisory)):
        if not group:
            continue
        print(f"\n{title}\n" + "-" * 78)
        for f in group:
            print(f"  [{f.where}] {f.what}")
            print(f"      {f.detail}")

    if blocking:
        print("\n" + "=" * 78)
        print("HOW TO FIX")
        print("=" * 78)
        if any(f.where == "tracked" for f in blocking):
            print("\n  Tracked files — remove from the index, keep on disk:")
            for f in blocking:
                if f.where == "tracked":
                    print(f"      git rm --cached {f.what}")
        if any(f.where == "history" for f in blocking):
            print("\n  History — a clone still receives these. Rewriting is the only")
            print("  fix, and it changes every commit hash after the first affected")
            print("  one. If the repository has never been pushed, the simplest")
            print("  honest option is to start a fresh history:")
            print("      rm -rf .git && git init && git add . && git commit")
            print("  If it has been pushed, use git-filter-repo and force-push:")
            paths = sorted({f.what for f in blocking if f.where == "history"})
            # Globs rather than an enumerated path list. A file that moved has
            # a different path in each era of the history, and enumerating them
            # means purging the one you listed and leaving the other behind —
            # a rewrite that reports success and removes nothing that matters.
            suffixes = sorted({Path(p).suffix.lower() for p in paths if Path(p).suffix})
            globs = " ".join(f"--path-glob '*{s}'" for s in suffixes)
            print(f"      git filter-repo --invert-paths {globs}")
            print("\n  Paths currently affected:")
            for p in paths[:10]:
                print(f"      {p}")
            if len(paths) > 10:
                print(f"      ... and {len(paths) - 10} more")
            print("  and treat anything already cloned as public.")

    print()
    failed = bool(blocking) or (args.strict and bool(advisory))
    print("FAIL — do not publish yet." if failed else "PASS — advisories only.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
