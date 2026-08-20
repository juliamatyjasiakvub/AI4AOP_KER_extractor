#!/usr/bin/env python
"""
Measure how much the extraction actually moves, by running it k times.

The determinism check (`tools/stability_check.py`) rules out everything after
extraction. What is left is the genuinely stochastic part, and the honest way
to characterise it is not to argue about temperature but to run the same corpus
several times and report the spread.

This runs only the two calls per paper that decide *which edge a paper lands
on* — the pathway reconstruction and the study-metadata call behind
`extract_pathway_rows`. The five per-KER evidence steps are skipped: they
change the prose in a row, not whether the row exists, and they are most of the
cost. Thirteen papers at k=5 is about 130 calls rather than about 500.

Nothing is written to the application database. Results go to a separate file
(`replicates.db` by default), so a measurement run cannot contaminate the
corpus being measured.

    export ANTHROPIC_API_KEY=...
    python tools/replicate_run.py \
        --pdfs ./papers \
        --upstream "Voltage-gated sodium channel" \
        --downstream "Oligodendrocyte differentiation" \
        --provider anthropic --model claude-opus-5 \
        --k 5

    python tools/replicate_run.py --report-only        # re-report, no calls

What it reports
---------------
*Anchor rate* per paper: in how many of the k replicates did that paper produce
a step running directly from the upstream anchor to the downstream anchor. That
is what makes a paper count toward the target KER, and it is the quantity that
was moving between runs.

*Supporting-paper count* per replicate: the number your synthesis would have
reported, once per replicate, so the spread is visible as a list rather than as
a single number that happened to come out of one run.

*Split papers*: the papers that were not unanimous. These are the ones a
majority vote would settle, and the ones a curator should look at by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2_extraction import ke_synonyms, ker_extractor, pdf_reader  # noqa: E402
from stage2_extraction.llm_providers import LLMConfig  # noqa: E402


SCHEMA = """
CREATE TABLE IF NOT EXISTS replicate_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS replicate_paper (
    replicate     INTEGER NOT NULL,
    filename      TEXT    NOT NULL,
    doi           TEXT,
    bears_on      INTEGER,
    n_steps       INTEGER,
    anchored      INTEGER,   -- 1 when a step runs upstream anchor -> downstream anchor
    chain         TEXT,      -- JSON list of "from -> to"
    events        TEXT,      -- JSON list of distinct event names, in chain order
    error         TEXT,
    seconds       REAL,
    PRIMARY KEY (replicate, filename)
);
"""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


class _NamedBytes(io.BytesIO):
    """
    A file-like object carrying a `.name`, which is what `extract_document`
    expects.

    A plain `open(path, "rb")` will not do: `extract_document` reads `.name` to
    label the document, and on a real file object that attribute is the full
    path and is read-only — assigning to it raises AttributeError. Reading the
    bytes up front also means the file handle is not held open across the whole
    run, which matters on Windows and matters more on OneDrive.
    """

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def _as_upload(path: Path) -> _NamedBytes:
    return _NamedBytes(path.read_bytes(), path.name)


def _norm(text: str) -> str:
    """The same normalisation `_canonical_anchor` uses to compare event names."""
    import re

    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def is_anchored_step(step: Any, upstream: str, downstream: str) -> bool:
    """True when this step is the target relationship itself, not a chain link."""
    return (
        _norm(getattr(step, "from_event", "")) == _norm(upstream)
        and _norm(getattr(step, "to_event", "")) == _norm(downstream)
    )


def build_config(args: argparse.Namespace) -> LLMConfig:
    key_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(args.provider)
    api_key = os.getenv(key_env) if key_env else None
    if key_env and not api_key:
        print(f"Set {key_env} before running.")
        sys.exit(2)
    return LLMConfig(
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        seed=args.seed,
        request_timeout=1200,
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_replicates(args: argparse.Namespace, conn: sqlite3.Connection) -> None:
    folder = Path(args.pdfs).expanduser()
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(2)
    # Case-insensitively, because a corpus assembled by hand contains .PDF as
    # often as .pdf, and silently extracting twelve of thirteen papers is the
    # kind of error that looks like a finding.
    pdfs = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )
    if not pdfs:
        print(f"No PDFs found directly in {folder}")
        print("(subfolders are not searched — point --pdfs at the folder itself)")
        sys.exit(2)

    cfg = build_config(args)

    # Parse every PDF once. Parsing is deterministic, and re-parsing per
    # replicate would add a variable that is not under test — if the text
    # differed between replicates, nothing downstream would be comparable.
    print(f"Parsing {len(pdfs)} PDF(s)…")
    documents = []
    for path in pdfs:
        try:
            document = pdf_reader.extract_document(_as_upload(path))
        except Exception as exc:
            print(f"  FAILED to parse {path.name}: {type(exc).__name__}: {exc}")
            print("   If this file lives in OneDrive, check it is downloaded")
            print("   locally rather than a cloud-only placeholder.")
            sys.exit(2)
        if not (document.full_text or "").strip():
            print(f"  FAILED: {path.name} parsed to empty text — scanned image?")
            sys.exit(2)
        documents.append(document)
        print(f"  {path.name[:52]:52s} {len(document.full_text):7,d} chars  "
              f"doi={document.doi or 'not found'}")
    text_hash = hashlib.sha256(
        "".join(d.full_text for d in documents).encode()
    ).hexdigest()[:16]
    print(f"  corpus text hash {text_hash}")

    # Expand the vocabulary ONCE and reuse it for every replicate. It is cached
    # for 30 days anyway, so it would not vary within a session — pinning it
    # here makes that explicit and records what was used, because a different
    # vocabulary is a different experiment.
    print("Building target vocabulary…")
    up_vocab = ke_synonyms.build_vocabulary(args.upstream, cfg)
    down_vocab = ke_synonyms.build_vocabulary(args.downstream, cfg)
    up_terms, down_terms = list(up_vocab.terms), list(down_vocab.terms)
    vocab_hash = hashlib.sha256(
        json.dumps([sorted(up_terms), sorted(down_terms)]).encode()
    ).hexdigest()[:16]
    print(f"  {len(up_terms)} upstream / {len(down_terms)} downstream terms "
          f"(hash {vocab_hash})")

    meta = {
        "upstream": args.upstream,
        "downstream": args.downstream,
        "provider": args.provider,
        "model": args.model,
        "temperature": str(args.temperature),
        "seed": str(args.seed),
        "budget_scale": str(args.budget_scale),
        "directional": str(bool(args.directional)),
        "k": str(args.k),
        "n_papers": str(len(documents)),
        "corpus_text_hash": text_hash,
        "vocabulary_hash": vocab_hash,
        "upstream_terms": json.dumps(sorted(up_terms)),
        "downstream_terms": json.dumps(sorted(down_terms)),
        "prompt_fingerprint": _safe_fingerprint(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    conn.executemany(
        "INSERT OR REPLACE INTO replicate_meta (key, value) VALUES (?, ?)",
        list(meta.items()),
    )
    conn.commit()

    # Only a SUCCESSFUL paper-run counts as done. Skipping failed ones too
    # meant that once a run hit a dead API key, re-running after fixing it
    # skipped every paper the key had killed — the measurement could never be
    # completed, only restarted from scratch.
    done = {
        (r, f)
        for r, f in conn.execute(
            "SELECT replicate, filename FROM replicate_paper WHERE error IS NULL"
        )
    }

    total = args.k * len(documents)
    calls = 0
    consecutive_errors = 0
    for replicate in range(1, args.k + 1):
        print(f"\n--- replicate {replicate} of {args.k} ---")
        for document in documents:
            name = document.filename
            if (replicate, name) in done:
                print(f"  [skip] {name}")
                continue

            started = time.time()
            error = None
            bears_on = n_steps = 0
            # None, not 0. A call that failed did not observe "no anchor" — it
            # observed nothing, and recording it as 0 puts a fabricated
            # negative into the numerator of every rate this tool reports.
            anchored: Optional[int] = None
            chain: list[str] = []
            events: list[str] = []
            try:
                _, pathway, _warnings = ker_extractor.extract_pathway_rows(
                    document,
                    args.upstream,
                    args.downstream,
                    cfg=cfg,
                    budget_scale=float(args.budget_scale),
                    directional=bool(args.directional),
                    upstream_aliases=up_terms,
                    downstream_aliases=down_terms,
                )
                error = pathway.error
                bears_on = int(bool(pathway.bears_on_question))
                n_steps = int(pathway.n_steps)
                events = list(pathway.events)
                chain = [
                    f"{s.from_event} -> {s.to_event}" for s in pathway.steps
                ]
                anchored = int(
                    any(
                        is_anchored_step(s, args.upstream, args.downstream)
                        for s in pathway.steps
                    )
                )
            except Exception as exc:  # a failed paper is recorded, not fatal
                error = f"{type(exc).__name__}: {exc}"

            if error:
                anchored = None
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            elapsed = time.time() - started
            calls += 1
            conn.execute(
                "INSERT OR REPLACE INTO replicate_paper "
                "(replicate, filename, doi, bears_on, n_steps, anchored, chain, "
                " events, error, seconds) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    replicate, name, document.doi, bears_on, n_steps, anchored,
                    json.dumps(chain), json.dumps(events), error, round(elapsed, 1),
                ),
            )
            conn.commit()

            mark = (
                "FAILED" if error
                else "ANCHORED" if anchored
                else ("chain" if n_steps else "no-steps")
            )
            print(
                f"  [{calls:3d}/{total}] {name[:44]:44s} "
                f"steps={n_steps:2d} {mark:9s} {elapsed:5.1f}s"
                + (f"  ERROR: {error[:60]}" if error else "")
            )

            # A dead key, an exhausted balance or a retired model fails every
            # call identically and instantly. Grinding through the remaining
            # sixty is not resilience — it just buries the first, informative
            # error under sixty copies and wastes the operator's evening.
            if consecutive_errors >= _ERROR_BREAKER:
                print(
                    f"\nSTOPPING: {consecutive_errors} calls failed in a row.\n"
                    f"Last error:\n  {error}\n\n"
                    "Nothing measured so far is lost — successful paper-runs are\n"
                    "kept and failed ones are not marked done, so re-running this\n"
                    "same command after fixing the cause resumes where it stopped."
                )
                return


#: Consecutive failures tolerated before assuming the cause is systemic.
_ERROR_BREAKER = 3


def _safe_fingerprint() -> str:
    try:
        import run_manifest

        return str(run_manifest.prompt_fingerprint())
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(conn: sqlite3.Connection) -> int:
    meta = dict(conn.execute("SELECT key, value FROM replicate_meta"))
    rows = list(
        conn.execute(
            "SELECT replicate, filename, doi, bears_on, n_steps, anchored, "
            "chain, error FROM replicate_paper ORDER BY replicate, filename"
        )
    )
    if not rows:
        print("No replicate results recorded yet.")
        return 2

    k = max(r[0] for r in rows)
    papers = sorted({r[1] for r in rows})

    print("=" * 78)
    print("REPLICATE REPORT")
    print("=" * 78)
    print(f"  target        {meta.get('upstream')} -> {meta.get('downstream')}")
    print(f"  model         {meta.get('provider')}/{meta.get('model')}  "
          f"temperature {meta.get('temperature')}  seed {meta.get('seed')}")
    print(f"  replicates    {k}")
    print(f"  papers        {len(papers)}")
    print(f"  corpus hash   {meta.get('corpus_text_hash')}")
    print(f"  vocab hash    {meta.get('vocabulary_hash')}")

    anchored_by_paper: dict[str, list[int]] = defaultdict(list)
    steps_by_paper: dict[str, list[int]] = defaultdict(list)
    chains_by_paper: dict[str, set[str]] = defaultdict(set)
    per_replicate: Counter = Counter()
    ok_per_replicate: Counter = Counter()
    errors_per_replicate: Counter = Counter()
    errors = 0

    for replicate, filename, _doi, _bears, n_steps, anchored, chain, error in rows:
        if error or anchored is None:
            # Excluded from every rate below. A call that failed is a gap in
            # the measurement, not an observation of "did not anchor".
            errors += 1
            errors_per_replicate[replicate] += 1
            continue
        ok_per_replicate[replicate] += 1
        anchored_by_paper[filename].append(int(anchored))
        steps_by_paper[filename].append(int(n_steps))
        chains_by_paper[filename].add(chain or "[]")
        if anchored:
            per_replicate[replicate] += 1

    n_papers = len(papers)
    complete = [r for r in range(1, k + 1) if ok_per_replicate.get(r, 0) == n_papers]
    partial = [
        r for r in range(1, k + 1)
        if 0 < ok_per_replicate.get(r, 0) < n_papers
    ]
    empty = [r for r in range(1, k + 1) if ok_per_replicate.get(r, 0) == 0]

    if errors:
        print("\n" + "-" * 78)
        print("INCOMPLETE DATA")
        print("-" * 78)
        print(f"  {errors} paper-run(s) failed and are excluded from every rate below.")
        for r in range(1, k + 1):
            n_err = errors_per_replicate.get(r, 0)
            if n_err:
                print(f"    replicate {r}: {ok_per_replicate.get(r, 0)}/{n_papers} "
                      f"succeeded, {n_err} failed")
        first_error = next((row[7] for row in rows if row[7]), None)
        if first_error:
            print(f"\n  First error: {str(first_error)[:300]}")
        print(
            "\n  A failed call is NOT evidence that the paper missed the edge."
            "\n  Only complete replicates are counted as measurements."
        )

    # --- the headline number ------------------------------------------------
    print("\n" + "-" * 78)
    print("SUPPORTING PAPERS ON THE TARGET EDGE, PER REPLICATE")
    print("-" * 78)
    if not complete:
        print("  No replicate completed all papers. Nothing to report yet.")
        print(f"  Re-run the same command to fill in the {len(partial) + len(empty)} "
              "incomplete replicate(s).")
    else:
        counts = [per_replicate.get(r, 0) for r in complete]
        print(f"  complete replicates {complete}  ->  {counts}")
        if partial or empty:
            print(f"  excluded as incomplete: {sorted(partial + empty)}")
        lo, hi = min(counts), max(counts)
        mean = sum(counts) / len(counts)
        print(f"  range {lo}-{hi}   mean {mean:.1f}   spread {hi - lo}")
        if len(counts) < 2:
            print("  One replicate is not a measurement of variation — need at least 2.")
        elif hi == lo:
            print("  The count was identical every time.")
        else:
            print(f"  The count varied by {hi - lo} across identical runs.")

    # --- per-paper stability ------------------------------------------------
    print("\n" + "-" * 78)
    print("PER-PAPER ANCHOR RATE  (how often the paper landed on the target edge)")
    print("-" * 78)
    unanimous_yes = unanimous_no = split = 0
    split_rows: list[tuple[str, int, int]] = []
    for filename in papers:
        votes = anchored_by_paper[filename]
        yes, n = sum(votes), len(votes)
        steps = steps_by_paper[filename]
        variants = len(chains_by_paper[filename])
        if yes == n:
            unanimous_yes += 1
            verdict = "always"
        elif yes == 0:
            unanimous_no += 1
            verdict = "never"
        else:
            split += 1
            verdict = "SPLIT"
            split_rows.append((filename, yes, n))
        print(
            f"  {yes}/{n}  {verdict:7s}  steps {min(steps)}-{max(steps)}  "
            f"{variants} distinct chain(s)  {filename[:40]}"
        )

    print("\n" + "-" * 78)
    print("SUMMARY")
    print("-" * 78)
    print(f"  always on the edge   {unanimous_yes}")
    print(f"  never on the edge    {unanimous_no}")
    print(f"  split                {split}")
    if errors:
        print(f"  paper-runs with an error   {errors}")

    majority = unanimous_yes + sum(1 for _, yes, n in split_rows if yes * 2 > n)
    print(f"\n  majority-vote count: {majority} supporting paper(s)")
    print("  (a paper counts when it anchored in more than half the replicates)")

    if split_rows:
        print("\n  The split papers are where the number comes from. Each is one")
        print("  paper the model sometimes reads as directly evidencing the")
        print("  relationship and sometimes as evidencing a longer chain:")
        for filename, yes, n in split_rows:
            print(f"    {yes}/{n}  {filename}")

    measured = [p for p in papers if anchored_by_paper.get(p)]
    stability = (
        (unanimous_yes + unanimous_no) / len(measured) if measured else 0.0
    )
    print(f"\n  agreement rate: {stability:.0%} of papers unanimous "
          f"({len(measured)} paper(s) with at least one successful run)")

    # --- membership churn ---------------------------------------------------
    # The count alone understates the problem. Two runs can both report eight
    # supporting papers while disagreeing about WHICH eight — a stable-looking
    # number over a shifting evidence base is worse than a visibly noisy one,
    # because nothing on the page says the set changed.
    if len(complete) >= 2:
        by_replicate: dict[int, set[str]] = defaultdict(set)
        for replicate, filename, _d, _b, _n, anchored, _c, error in rows:
            if not error and anchored:
                by_replicate[replicate].add(filename)
        print("\n" + "-" * 78)
        print("MEMBERSHIP CHURN BETWEEN COMPLETE REPLICATES")
        print("-" * 78)
        for a, b in zip(complete, complete[1:]):
            set_a, set_b = by_replicate[a], by_replicate[b]
            changed = (set_a ^ set_b)
            print(
                f"  replicate {a} -> {b}:  {len(set_a)} vs {len(set_b)} papers, "
                f"{len(changed)} changed answer"
            )
            for filename in sorted(changed):
                direction = "gained" if filename in set_b else "lost  "
                print(f"      {direction}  {filename}")
        union = set().union(*(by_replicate[r] for r in complete))
        always = set.intersection(*(by_replicate[r] for r in complete))
        print(f"\n  anchored in EVERY complete replicate: {len(always)}")
        print(f"  anchored in at least one:             {len(union)}")
        if union:
            print(f"  so the defensible range is {len(always)}-{len(union)} "
                  "supporting papers")

    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", default="./papers")
    parser.add_argument("--upstream", default="")
    parser.add_argument("--downstream", default="")
    parser.add_argument("--provider", default="anthropic",
                        choices=("anthropic", "openai", "ollama"))
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--budget-scale", type=float, default=2.0)
    parser.add_argument("--directional", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("replicates.db"))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    conn = open_db(args.out)
    if not args.report_only:
        if not args.upstream or not args.downstream:
            print("--upstream and --downstream are required unless --report-only")
            return 2
        run_replicates(args, conn)
    return report(conn)


if __name__ == "__main__":
    sys.exit(main())
