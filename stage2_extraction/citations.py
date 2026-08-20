from __future__ import annotations

"""
Human-readable citation keys for the papers behind every claim.

The tool identified papers by DOI everywhere it showed one — in the assignment
grid, in the evidence panel, on the map, in exports. A DOI is the right thing
to *store* and the wrong thing to *read*: `10.1016/j.neuro.2019.03.004` tells a
curator nothing about which paper is being cited, so checking a claim meant
copying the string into a browser. `Nav et al., 2019a` is the form the same
person already uses in their manuscript.

How a key is built
------------------
First author's family name, the publication year, and a disambiguating letter
where one author published more than once in the same year:

    Sanchez et al., 2019a
    Sanchez et al., 2019b
    Lee & Park, 2021          two authors
    Okonkwo, 2020             one author

Where the metadata comes from
-----------------------------
Crossref (`api.crossref.org/works/{doi}`), which is free, needs no key, and
covers essentially every DOI. Results are cached in `paper_citation`, so the
network is touched once per paper ever rather than once per page render.

When a DOI cannot be resolved — no network, a DOI that was misread out of the
PDF, a preprint Crossref does not hold — the key falls back to the DOI itself
rather than inventing an author. A citation that is wrong is worse than one
that is ugly, so an unresolved paper is left visibly unresolved and the panel
says the lookup failed.
"""

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests

CROSSREF_URL = "https://api.crossref.org/works/{doi}"

#: Crossref asks for a contact address in the User-Agent so they can get in
#: touch about a misbehaving client, and rewards it with the faster pool.
USER_AGENT = (
    "AI4AOP-KER-extractor/1.0 (https://github.com/; mailto:aop-tool@example.org)"
)

REQUEST_TIMEOUT = 12

CREATE_PAPER_CITATION_SQL = """
CREATE TABLE IF NOT EXISTS paper_citation (
    doi             TEXT PRIMARY KEY,
    first_author    TEXT,
    second_author   TEXT,
    n_authors       INTEGER NOT NULL DEFAULT 0,
    year            INTEGER,
    title           TEXT,
    container_title TEXT,
    source          TEXT NOT NULL DEFAULT 'crossref',
    resolved        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    fetched_at      TEXT
)
"""


@dataclass(frozen=True)
class PaperCitation:
    """One paper's bibliographic identity, as far as it is known."""

    doi: str
    first_author: Optional[str] = None
    second_author: Optional[str] = None
    n_authors: int = 0
    year: Optional[int] = None
    title: Optional[str] = None
    container_title: Optional[str] = None
    resolved: bool = False
    error: Optional[str] = None

    def base_key(self) -> str:
        """
        The key without its disambiguating letter.

        Disambiguation needs the whole corpus in view, so it is applied by
        `citation_keys` rather than here.
        """
        if not self.resolved or not self.first_author or not self.year:
            return self.doi
        if self.n_authors >= 3:
            return f"{self.first_author} et al., {self.year}"
        if self.n_authors == 2 and self.second_author:
            return f"{self.first_author} & {self.second_author}, {self.year}"
        return f"{self.first_author}, {self.year}"

    def full_reference(self) -> str:
        """A one-line reference for panels that have room for it."""
        if not self.resolved:
            return self.doi
        bits = [self.base_key()]
        if self.title:
            bits.append(self.title)
        if self.container_title:
            bits.append(f"*{self.container_title}*")
        bits.append(self.doi)
        return ". ".join(b for b in bits if b)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    # Imported here rather than at module scope: table1_store imports nothing
    # from this module, and keeping it that way avoids a cycle if it ever wants
    # to render a citation itself.
    # `current_db_path`, not `DB_PATH`: the module constant is only the
    # fallback, and reading it directly would send every session's Crossref
    # cache to one shared file regardless of which database the session is on.
    from stage2_extraction.table1_store import current_db_path

    conn = sqlite3.connect(current_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_PAPER_CITATION_SQL)
    return conn


def _row_to_citation(row: sqlite3.Row) -> PaperCitation:
    return PaperCitation(
        doi=str(row["doi"]),
        first_author=row["first_author"],
        second_author=row["second_author"],
        n_authors=int(row["n_authors"] or 0),
        year=int(row["year"]) if row["year"] is not None else None,
        title=row["title"],
        container_title=row["container_title"],
        resolved=bool(row["resolved"]),
        error=row["error"],
    )


def _load_cached(dois: Iterable[str]) -> dict[str, PaperCitation]:
    wanted = [_norm(d) for d in dois if _norm(d)]
    if not wanted:
        return {}
    out: dict[str, PaperCitation] = {}
    with _connect() as conn:
        # Chunked so a corpus of thousands does not exceed SQLite's variable
        # limit, which is 999 by default.
        for i in range(0, len(wanted), 500):
            batch = wanted[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT * FROM paper_citation WHERE doi IN ({placeholders})", batch
            ).fetchall()
            for row in rows:
                out[str(row["doi"])] = _row_to_citation(row)
    return out


def _store(citation: PaperCitation) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO paper_citation (doi, first_author, "
            "second_author, n_authors, year, title, container_title, source, "
            "resolved, error, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, "
            "'crossref', ?, ?, datetime('now'))",
            (
                citation.doi,
                citation.first_author,
                citation.second_author,
                int(citation.n_authors),
                citation.year,
                citation.title,
                citation.container_title,
                int(citation.resolved),
                citation.error,
            ),
        )
        conn.commit()


def forget(doi: str) -> None:
    """Drop one cached lookup so it is retried — used by the refresh button."""
    with _connect() as conn:
        conn.execute("DELETE FROM paper_citation WHERE doi = ?", (_norm(doi),))
        conn.commit()


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------

def _norm(doi: Any) -> str:
    """Normalise a DOI to the bare `10.x/y` form, lowercased."""
    text = str(doi or "").strip().lower()
    if not text or text in {"nan", "none"}:
        return ""
    text = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", text)
    return text.strip().rstrip(".")


def _family_name(author: dict) -> str:
    """
    The surname, however Crossref happened to record it.

    Some records carry only `name` (consortia, or a group author), in which
    case the whole string is the best available answer.
    """
    family = str(author.get("family") or "").strip()
    if family:
        return family
    return str(author.get("name") or "").strip()


def _year_from(message: dict) -> Optional[int]:
    """
    Publication year, preferring the print/online issue date.

    `created` is when the DOI was registered, which for a paper deposited
    ahead of print can be a year earlier than the one on the article. It is
    used only when nothing better exists.
    """
    for field in ("issued", "published-print", "published-online", "published", "created"):
        parts = (message.get(field) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def fetch(doi: str, *, session: Optional[requests.Session] = None) -> PaperCitation:
    """
    Look one DOI up at Crossref. Never raises — an unresolved paper is data.
    """
    normalised = _norm(doi)
    if not normalised:
        return PaperCitation(doi=str(doi or ""), error="No DOI recorded.")

    getter = session or requests
    try:
        response = getter.get(
            CROSSREF_URL.format(doi=normalised),
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 404:
            return PaperCitation(doi=normalised, error="Crossref has no record of this DOI.")
        response.raise_for_status()
        message = (response.json() or {}).get("message") or {}
    except Exception as exc:  # noqa: BLE001 - a lookup failure is not fatal
        return PaperCitation(doi=normalised, error=f"Lookup failed: {exc}")

    authors = [a for a in (message.get("author") or []) if _family_name(a)]
    titles = message.get("title") or []
    containers = message.get("container-title") or []

    first = _family_name(authors[0]) if authors else ""
    second = _family_name(authors[1]) if len(authors) > 1 else ""
    year = _year_from(message)

    if not first or not year:
        return PaperCitation(
            doi=normalised,
            first_author=first or None,
            year=year,
            title=str(titles[0]) if titles else None,
            container_title=str(containers[0]) if containers else None,
            error="Crossref record has no author or no year.",
        )

    return PaperCitation(
        doi=normalised,
        first_author=first,
        second_author=second or None,
        n_authors=len(authors),
        year=year,
        title=str(titles[0]) if titles else None,
        container_title=str(containers[0]) if containers else None,
        resolved=True,
    )


def resolve(
    dois: Iterable[str],
    *,
    refresh: bool = False,
    progress: Optional[Any] = None,
) -> dict[str, PaperCitation]:
    """
    Citation metadata for every DOI given, from cache where possible.

    Failed lookups are cached too, deliberately: without that, a corpus with
    one bad DOI re-queries Crossref on every single rerun of the page, which
    Streamlit does constantly. `refresh=True` is the way to retry.
    """
    unique = sorted({_norm(d) for d in dois if _norm(d)})
    if not unique:
        return {}

    cached = {} if refresh else _load_cached(unique)
    missing = [d for d in unique if d not in cached]

    if missing:
        with requests.Session() as session:
            for i, doi in enumerate(missing, start=1):
                citation = fetch(doi, session=session)
                _store(citation)
                cached[doi] = citation
                if progress is not None:
                    try:
                        progress(i, len(missing), doi)
                    except Exception:  # noqa: BLE001 - a progress bar is cosmetic
                        pass

    return cached


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def citation_keys(
    dois: Iterable[str],
    *,
    refresh: bool = False,
    progress: Optional[Any] = None,
) -> dict[str, str]:
    """
    Map each DOI to its display key, with a/b/c applied across the corpus.

    Suffixes are assigned in DOI order rather than, say, by how many claims a
    paper contributed, so that the same paper keeps the same letter between
    sessions. A key that changes when a new paper is added would break every
    note a curator has already written down.
    """
    citations = resolve(dois, refresh=refresh, progress=progress)

    by_base: dict[str, list[str]] = defaultdict(list)
    for doi, citation in citations.items():
        by_base[citation.base_key()].append(doi)

    keys: dict[str, str] = {}
    for base, group in by_base.items():
        if len(group) == 1:
            keys[group[0]] = base
            continue
        # Only resolved papers get letters. Two unresolved papers share no key
        # to disambiguate — their base key is their own DOI, already unique.
        for doi, letter in zip(sorted(group), _letters()):
            keys[doi] = f"{base}{letter}"
    return keys


def _letters() -> Iterable[str]:
    """a, b, c … z, aa, ab — enough for any plausible corpus."""
    from itertools import count, product
    from string import ascii_lowercase

    for width in count(1):
        for combo in product(ascii_lowercase, repeat=width):
            yield "".join(combo)


def key_for(doi: str, keys: dict[str, str]) -> str:
    """Look one key up defensively, falling back to the DOI as given."""
    normalised = _norm(doi)
    return keys.get(normalised) or (normalised or str(doi or "—"))


__all__ = [
    "PaperCitation",
    "CREATE_PAPER_CITATION_SQL",
    "citation_keys",
    "fetch",
    "forget",
    "key_for",
    "resolve",
]
