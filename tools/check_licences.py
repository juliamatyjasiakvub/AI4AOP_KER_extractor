#!/usr/bin/env python
"""
Report what licence evidence exists for each article, so a corpus can be
narrowed to the papers whose terms are actually known.

Why this is a separate tool
---------------------------
"May I send this paper to a commercial model, and publish what comes back?"
has three different answers depending on the article:

* **Openly licensed** — CC0 or CC BY, and sometimes the more restrictive
  Creative Commons variants. The terms are published, they travel with the
  article, and they do not depend on anyone's subscription. You can establish
  this yourself.
* **Subscribed** — the terms live in a contract between your institution and
  the publisher. No public metadata will tell you, and this tool will not
  pretend otherwise.
* **Free to read, no licence** — the trap. An article can be readable by
  anyone and still grant no reuse rights at all. Unpaywall calls this
  "bronze". `is_oa: true` is not permission, and this tool never treats it as
  such.

So the output is evidence, not a verdict. It says what each source reported and
which category that puts the article in; deciding what to do about a
`SUBSCRIPTION` or `FREE-TO-READ` row is a question for your library.

Two things it is careful about:

*Version.* Crossref records licences separately for the version of record, the
accepted manuscript and the text-mining copy. An accepted manuscript under
CC BY does not license the publisher's typeset PDF. Where the permissive
licence applies only to a version you may not be holding, the row is flagged.

*Nothing found is not nothing.* An article with no licence metadata is
reported as UNKNOWN and never as permitted.

Usage
-----
    python tools/check_licences.py --email you@vub.be                  # DOIs from aop_rag.db
    python tools/check_licences.py --email you@vub.be --doi 10.1/x 10.2/y
    python tools/check_licences.py --email you@vub.be --file dois.txt
    python tools/check_licences.py --email you@vub.be --csv licences.csv

`--email` is required: both Crossref and Unpaywall ask callers to identify
themselves, and Unpaywall rejects requests without it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CROSSREF = "https://api.crossref.org/works/"
UNPAYWALL = "https://api.unpaywall.org/v2/"

#: Licence URL fragments mapped to (category, short name). Order matters —
#: "by-nc-sa" must be tested before "by-nc" and "by".
_LICENCE_PATTERNS: list[tuple[str, str, str]] = [
    ("creativecommons.org/publicdomain/zero", "OPEN", "CC0"),
    ("creativecommons.org/publicdomain/mark", "OPEN", "Public domain"),
    ("creativecommons.org/licenses/by-nc-nd", "OPEN-CONDITIONS", "CC BY-NC-ND"),
    ("creativecommons.org/licenses/by-nc-sa", "OPEN-CONDITIONS", "CC BY-NC-SA"),
    ("creativecommons.org/licenses/by-nd", "OPEN-CONDITIONS", "CC BY-ND"),
    ("creativecommons.org/licenses/by-nc", "OPEN-CONDITIONS", "CC BY-NC"),
    ("creativecommons.org/licenses/by-sa", "OPEN-CONDITIONS", "CC BY-SA"),
    ("creativecommons.org/licenses/by", "OPEN", "CC BY"),
]

#: What each category means, printed once at the end so the table stays terse.
CATEGORY_NOTES = {
    "OPEN": "Reuse permitted by a published licence. You can establish this "
            "yourself; no institutional agreement is involved.",
    "OPEN-CONDITIONS": "Openly licensed but with conditions — non-commercial, "
                       "no-derivatives or share-alike. Read the specific "
                       "licence before processing or republishing.",
    "PUBLISHER-TERMS": "A publisher's own licence URL, not a standard open "
                       "licence. Its terms have to be read; they vary and "
                       "some address text mining explicitly.",
    "FREE-TO-READ": "Readable without payment but carrying NO reuse licence. "
                    "Free access is not permission. Treat as subscription.",
    "SUBSCRIPTION": "No open licence found. What you may do depends on your "
                    "institution's agreement with the publisher, which no "
                    "public metadata can tell you.",
    "UNKNOWN": "No licence information was returned by either source. Absence "
               "of evidence is not permission.",
}


@dataclass
class Finding:
    doi: str
    title: str = ""
    publisher: str = ""
    category: str = "UNKNOWN"
    licence: str = ""
    versions: str = ""
    oa_status: str = ""
    notes: list[str] = field(default_factory=list)
    error: str = ""


def _get_json(url: str, email: str, timeout: int = 30) -> Optional[dict]:
    """Fetch JSON, identifying ourselves as both APIs ask callers to do."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"AI4AOP_KER_extractor/1.0 (mailto:{email})"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _classify_licence_url(url: str) -> Optional[tuple[str, str]]:
    lowered = (url or "").lower()
    for fragment, category, name in _LICENCE_PATTERNS:
        if fragment in lowered:
            return category, name
    return None


def inspect(doi: str, email: str) -> Finding:
    finding = Finding(doi=doi)

    # --- Crossref: authoritative for what the publisher declared ------------
    crossref = _get_json(CROSSREF + urllib.parse.quote(doi), email)
    licences: list[tuple[str, str, str]] = []   # (category, name, version)
    if crossref:
        message = crossref.get("message", {})
        finding.title = (message.get("title") or [""])[0][:70]
        finding.publisher = (message.get("publisher") or "")[:34]
        for entry in message.get("license") or []:
            url = entry.get("URL", "")
            version = entry.get("content-version", "unspecified")
            classified = _classify_licence_url(url)
            if classified:
                licences.append((classified[0], classified[1], version))
            else:
                licences.append(("PUBLISHER-TERMS", url[:44], version))
    else:
        finding.notes.append("Crossref returned nothing for this DOI")

    # --- Unpaywall: whether an open copy exists, and under what -------------
    unpaywall = _get_json(
        f"{UNPAYWALL}{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}",
        email,
    )
    if unpaywall:
        finding.oa_status = unpaywall.get("oa_status") or ""
        best = unpaywall.get("best_oa_location") or {}
        upw_licence = best.get("license")
        if upw_licence:
            classified = _classify_licence_url(
                f"creativecommons.org/licenses/{upw_licence}"
                if upw_licence.startswith("cc-")
                else upw_licence
            ) or _classify_licence_url(upw_licence)
            if classified:
                licences.append(
                    (classified[0], classified[1], best.get("version") or "unspecified")
                )
        elif unpaywall.get("is_oa"):
            finding.notes.append(
                "Unpaywall reports it is free to read but records no licence"
            )

    # --- Decide, preferring the most permissive evidence found --------------
    if licences:
        rank = {"OPEN": 0, "OPEN-CONDITIONS": 1, "PUBLISHER-TERMS": 2}
        licences.sort(key=lambda item: rank.get(item[0], 3))
        finding.category, finding.licence, _ = licences[0]
        finding.versions = ", ".join(
            sorted({f"{name}[{version}]" for _, name, version in licences})
        )[:70]

        # The version trap: permissive terms that may not cover the file you hold.
        best_versions = {
            version for category, _, version in licences
            if category in ("OPEN", "OPEN-CONDITIONS")
        }
        if best_versions and "vor" not in best_versions:
            finding.notes.append(
                f"open licence applies to {'/'.join(sorted(best_versions))}, "
                "not the version of record — check which file you hold"
            )
    elif finding.oa_status in ("bronze",):
        finding.category = "FREE-TO-READ"
    elif crossref:
        finding.category = "SUBSCRIPTION"
    else:
        finding.category = "UNKNOWN"

    return finding


def dois_from_database(path: Path) -> list[str]:
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT DISTINCT source_doi FROM table1_extractions "
            "WHERE source_doi IS NOT NULL AND TRIM(source_doi) <> ''"
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"Could not read DOIs from {path}: {exc}")
        return []
    return sorted({str(r[0]).strip() for r in rows})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True,
                        help="your address; both APIs require it")
    parser.add_argument("--doi", nargs="*", default=[])
    parser.add_argument("--file", type=Path)
    parser.add_argument("--db", type=Path, default=Path("aop_rag.db"))
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    dois = list(args.doi)
    if args.file and args.file.exists():
        dois += [line.strip() for line in args.file.read_text().splitlines()
                 if line.strip()]
    if not dois:
        dois = dois_from_database(args.db)
    if not dois:
        print("No DOIs given and none found in the database.")
        return 2

    print(f"Checking {len(dois)} DOI(s) against Crossref and Unpaywall…\n")
    findings: list[Finding] = []
    for i, doi in enumerate(dois, 1):
        finding = inspect(doi, args.email)
        findings.append(finding)
        print(f"  [{i:>2}/{len(dois)}] {finding.category:<16} "
              f"{finding.licence or '—':<14} {doi}")
        time.sleep(0.2)   # both APIs are free; do not hammer them

    order = ["OPEN", "OPEN-CONDITIONS", "PUBLISHER-TERMS",
             "FREE-TO-READ", "SUBSCRIPTION", "UNKNOWN"]
    findings.sort(key=lambda f: (order.index(f.category)
                                 if f.category in order else 99, f.doi))

    print("\n" + "=" * 100)
    print("LICENCE EVIDENCE")
    print("=" * 100)
    current = None
    for finding in findings:
        if finding.category != current:
            current = finding.category
            count = sum(1 for f in findings if f.category == current)
            print(f"\n{current}  ({count})")
            print("  " + CATEGORY_NOTES.get(current, ""))
            print()
        print(f"  {finding.doi}")
        print(f"      {finding.licence or 'no licence found':<18} "
              f"{finding.publisher}"
              + (f"   [{finding.oa_status}]" if finding.oa_status else ""))
        if finding.versions:
            print(f"      versions: {finding.versions}")
        for note in finding.notes:
            print(f"      ! {note}")

    openly = sum(1 for f in findings if f.category == "OPEN")
    conditions = sum(1 for f in findings if f.category == "OPEN-CONDITIONS")
    print("\n" + "=" * 100)
    print(f"{openly} of {len(findings)} carry a permissive open licence you can "
          f"verify yourself.")
    if conditions:
        print(f"{conditions} more are openly licensed with conditions worth reading.")
    print("Everything else depends on your institution's agreements, which this "
          "tool cannot see.")
    print("=" * 100)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["doi", "category", "licence", "versions",
                             "oa_status", "publisher", "title", "notes"])
            for f in findings:
                writer.writerow([f.doi, f.category, f.licence, f.versions,
                                 f.oa_status, f.publisher, f.title,
                                 "; ".join(f.notes)])
        print(f"\nWritten to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
