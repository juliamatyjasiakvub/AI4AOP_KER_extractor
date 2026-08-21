"""
HGNC lookups: resolve a gene or protein name to its aliases and its isoform family.

Why this exists
---------------
A Key Event about sodium-channel function is reported in papers as NaV1.2,
NaV1.6, SCN2A, SCN8A, "sodium voltage-gated channel alpha subunit 8", or Nav1.x
generally, and which of those a given paper uses is an accident of that lab's
convention. Asking a curator to enumerate them is asking them to reproduce a
nomenclature database from memory, and the ones they forget become papers that
silently screen out.

HGNC is that database. Two facts make it the right one:

* every approved symbol carries `alias_symbol` and `prev_symbol`, which is
  exactly the "NaV1.6 is also SCN8A, also NaCh6, also PN4" mapping, and
* every gene belongs to `gene_group`s, and the group "Sodium voltage-gated
  channel alpha subunits" contains all nine alpha subunits with their Nav1.1
  through Nav1.9 aliases. Resolving one isoform to its group resolves the
  family, which is what "papers measuring any sodium channel isoform" means.

Design
------
The service is strictly optional, matching how `ols4_client` treats OLS4. If
HGNC is unreachable the functions return empty results and set an error; the
vocabulary falls back to whatever the model and the ontology produced, and the
run continues. A nomenclature server being down is not a reason to fail a
literature screen.

Everything is cached in the same SQLite file as the rest of the app. Gene
nomenclature changes on the scale of years, so the TTL is long.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import requests

HGNC_BASE_URL = "https://rest.genenames.org"

#: Nomenclature is stable on a scale of years; a month is already cautious.
_CACHE_TTL_SECONDS = 90 * 24 * 3600

_TIMEOUT = 12
_HEADERS = {"Accept": "application/json"}

#: Fields on an HGNC record that name the gene or its product in a way a paper
#: might write. `name` is the full descriptive name; the rest are symbols.
_NAME_FIELDS = ("symbol", "alias_symbol", "prev_symbol", "name", "alias_name")

#: Minimum length for a bare alphabetic alias to be trusted with a family
#: expansion. Three letters is where ambiguity lives: ALT, AST, ALP and BUN are
#: all real aliases of real genes and all far more likely, in a paper, to be
#: the assay of the same name.
_MIN_UNAMBIGUOUS_ALIAS_LEN = 4


def _is_strong_match(query: str, matched_by: Optional[str]) -> bool:
    """
    Whether this match is solid enough to pull in the gene's whole family.

    An approved symbol is the gene, whatever else the letters might mean. An
    alias carrying a digit is a nomenclature string — NaV1.6, KCNQ2 — that no
    assay shares. A long alias is unlikely to collide by accident.

    What is left is the short bare alias, and it is left out on purpose. "ALT"
    is an alias of GPT and also the commonest liver readout in toxicology; the
    gene's own names are worth having either way, but expanding from it to
    every aminotransferase fills a screen's vocabulary with relatives of a word
    the paper probably did not mean. Resolving it is cheap and reversible.
    Expanding it is neither.
    """
    query = (query or "").strip()
    if matched_by == "symbol":
        return True
    if any(c.isdigit() for c in query):
        return True
    return len(query) >= _MIN_UNAMBIGUOUS_ALIAS_LEN

#: A gene group large enough to be a superfamily rather than an isoform family
#: would flood the vocabulary with unrelated symbols — "Zinc fingers" has
#: hundreds of members and nothing useful in common for screening purposes.
_MAX_GROUP_MEMBERS = 40

_DB_PATH: Path = Path("aop_rag.db")

#: Per-thread override of `_DB_PATH`. Streamlit serves every browser session
#: from one process and interleaves their threads, so a module global assigned
#: per script run is not isolation — see `session_db`. The module constant
#: stays as the fallback for use outside Streamlit and in tests.
_LOCAL = threading.local()


def _db() -> Path:
    """The cache database this thread should use."""
    return getattr(_LOCAL, "db_path", None) or _DB_PATH

CREATE_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS hgnc_cache (
    cache_key   TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    payload     TEXT NOT NULL,   -- JSON
    fetched_at  REAL NOT NULL
)
"""


def set_db_path(path: Path | str) -> None:
    _LOCAL.db_path = None if path is None else Path(path)


def init_cache() -> None:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            conn.commit()
    except sqlite3.Error:
        pass


def clear_cache() -> None:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            conn.execute("DELETE FROM hgnc_cache")
            conn.commit()
    except sqlite3.Error:
        pass


def _cache_get(key: str) -> Optional[Any]:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            row = conn.execute(
                "SELECT payload, fetched_at FROM hgnc_cache WHERE cache_key = ?", (key,)
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    payload, fetched_at = row
    if time.time() - float(fetched_at) > _CACHE_TTL_SECONDS:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def _cache_put(key: str, query: str, payload: Any) -> None:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            conn.execute(
                "INSERT OR REPLACE INTO hgnc_cache (cache_key, query, payload, fetched_at) "
                "VALUES (?, ?, ?, ?)",
                (key, query, json.dumps(payload), time.time()),
            )
            conn.commit()
    except sqlite3.Error:
        pass


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GeneVocabulary:
    """Everything HGNC knows to call one gene, plus its isoform family."""

    query: str
    #: Approved symbol, if the query resolved to one.
    symbol: Optional[str] = None
    #: Symbols and names for the gene itself.
    names: list[str] = field(default_factory=list)
    #: Names of the gene groups it belongs to.
    groups: list[str] = field(default_factory=list)
    #: Symbols and aliases of every other member of those groups.
    family: list[str] = field(default_factory=list)
    error: Optional[str] = None

    #: Which HGNC index the query matched: "symbol", "alias_symbol" or
    #: "prev_symbol". Recorded because the three are not equally good evidence
    #: that the paper meant this gene, and because a curator looking at a
    #: surprising term in the vocabulary deserves to see where it came from.
    matched_by: Optional[str] = None

    #: True when the family was deliberately withheld — see `_is_strong_match`.
    #: The gene's own names are still present; only its relatives are not.
    family_withheld: bool = False

    @property
    def resolved(self) -> bool:
        return self.symbol is not None

    def all_terms(self, include_family: bool = True) -> list[str]:
        terms = list(self.names)
        if include_family:
            terms.extend(self.family)
        seen: set[str] = set()
        out: list[str] = []
        for term in terms:
            key = term.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(term.strip())
        return out


# ---------------------------------------------------------------------------
# Which strings are worth asking HGNC about
# ---------------------------------------------------------------------------

#: Words a paper writes in capitals that are not symbols. Ordinary English
#: only — everything else the shape test already rejects.
#:
#: What used to be here as well was bench vocabulary: TTX, DMEM, FBS, PBS,
#: ELISA, PCR. That list was assembled while reading one corpus, and it showed:
#: it blocked the reagents of an electrophysiology lab and said nothing about
#: ALT, AST, ALP, BUN or LDH, which a liver paper writes on every page. Those
#: are real HGNC aliases — ALT resolves to GPT, AST to GOT1 — so the missing
#: entries were not harmless, and the ones present were: HGNC returns nothing
#: for TTX, and a negative is cached.
#:
#: A hand-written list of what a field happens to say cannot be finished. The
#: check that replaced it is `_is_strong_match` below, which asks how the token
#: matched rather than whether somebody remembered to list it.
_NOT_SYMBOLS = {
    "the", "and", "for", "not", "all", "any", "may", "can", "was", "are",
    "cell", "cells", "gene", "genes", "type", "types", "data", "mice", "rat",
    "rats", "human", "mouse", "test", "control", "wild", "level", "levels",
    "dna", "rna",
}

#: A symbol looks like: letters, then optionally digits and dots, e.g. SCN8A,
#: NaV1.6, MBP, GFAP, PDGFRA, KCNQ2. Requires at least one letter and either a
#: digit or a length that makes an accidental English word unlikely.
_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[.\-]?[0-9]+)*[A-Za-z0-9]*$")


def looks_like_gene_symbol(token: str) -> bool:
    """
    Cheap filter deciding whether a vocabulary term is worth an HGNC lookup.

    Deliberately permissive about what a symbol looks like and strict about
    known non-symbols: a false positive costs one cached HTTP request that
    returns nothing, while a false negative loses a whole isoform family.
    """
    token = (token or "").strip()
    if not (2 <= len(token) <= 15) or " " in token:
        return False
    if token.lower() in _NOT_SYMBOLS:
        return False
    if not _SYMBOL_RE.match(token):
        return False
    # Require either a digit (NaV1.6, SCN8A, KCNQ2) or an all-caps-ish shape
    # (MBP, GFAP, PLP1). A lowercase word with no digits is prose.
    has_digit = any(c.isdigit() for c in token)
    mostly_upper = sum(c.isupper() for c in token) >= max(2, len(token) - 2)
    return has_digit or mostly_upper


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(path: str) -> tuple[Optional[dict], Optional[str]]:
    url = f"{HGNC_BASE_URL}{path}"
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return None, f"HGNC unreachable: {exc}"
    if response.status_code != 200:
        return None, f"HGNC returned HTTP {response.status_code}"
    try:
        return response.json(), None
    except ValueError:
        return None, "HGNC returned a response that was not JSON."


def _docs(payload: Optional[dict]) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    response = payload.get("response")
    if not isinstance(response, dict):
        return []
    docs = response.get("docs")
    return [d for d in docs if isinstance(d, dict)] if isinstance(docs, list) else []


def _names_from_doc(doc: dict, include_full_names: bool = True) -> list[str]:
    """Pull every string a paper might use for this gene out of one record."""
    out: list[str] = []
    for field_name in _NAME_FIELDS:
        if field_name in ("name", "alias_name") and not include_full_names:
            continue
        value = doc.get(field_name)
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            out.extend(str(v) for v in value if isinstance(v, str))
    return out


def _quote(value: str) -> str:
    """Percent-safe path segment for the HGNC REST path style."""
    from urllib.parse import quote

    return quote(value.strip(), safe="")


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def fetch_gene(
    query: str,
    *,
    include_family: bool = True,
    use_cache: bool = True,
) -> GeneVocabulary:
    """
    Resolve one gene or protein name to its aliases and, optionally, its family.

    Tries the approved symbol first, then the alias and previous-symbol indexes,
    which is what lets "NaV1.6" — an alias, not an approved symbol — resolve to
    SCN8A and from there to the whole alpha-subunit family.
    """
    query = (query or "").strip()
    if not query:
        return GeneVocabulary(query=query)

    cache_key = f"gene::{query.lower()}::{int(include_family)}"
    if use_cache:
        cached = _cache_get(cache_key)
        if isinstance(cached, dict):
            return GeneVocabulary(**cached)

    doc: Optional[dict] = None
    matched_by: Optional[str] = None
    last_error: Optional[str] = None
    for index, path in (
        ("symbol", f"/fetch/symbol/{_quote(query)}"),
        ("alias_symbol", f"/search/alias_symbol/{_quote(query)}"),
        ("prev_symbol", f"/search/prev_symbol/{_quote(query)}"),
    ):
        payload, error = _get(path)
        if error:
            last_error = error
            continue
        docs = _docs(payload)
        if docs:
            # A /search hit carries only identifiers, so re-fetch the full
            # record by its approved symbol to get aliases and group ids.
            candidate = docs[0]
            if "gene_group_id" not in candidate and candidate.get("symbol"):
                full, _ = _get(f"/fetch/symbol/{_quote(str(candidate['symbol']))}")
                full_docs = _docs(full)
                candidate = full_docs[0] if full_docs else candidate
            doc = candidate
            matched_by = index
            break

    if doc is None:
        result = GeneVocabulary(query=query, error=last_error)
        if use_cache and last_error is None:
            # Cache the negative too: a term that is not a gene stays not a
            # gene, and re-asking on every run costs a request per paper.
            _cache_put(cache_key, query, result.__dict__)
        return result

    names = _names_from_doc(doc)
    groups = [g for g in (doc.get("gene_group") or []) if isinstance(g, str)]
    group_ids = [g for g in (doc.get("gene_group_id") or []) if isinstance(g, int)]

    strong = _is_strong_match(query, matched_by)
    family: list[str] = []
    if include_family and group_ids and strong:
        for group_id in group_ids[:2]:  # a gene in many groups is rarely specific
            payload, error = _get(f"/fetch/gene_group_id/{group_id}")
            if error:
                last_error = error
                continue
            members = _docs(payload)
            if len(members) > _MAX_GROUP_MEMBERS:
                continue  # superfamily, not an isoform family
            for member in members:
                if member.get("symbol") == doc.get("symbol"):
                    continue
                # Symbols and aliases only. The full descriptive names of forty
                # relatives add long phrases that match nothing useful.
                family.extend(_names_from_doc(member, include_full_names=False))

    result = GeneVocabulary(
        query=query,
        symbol=str(doc.get("symbol")) if doc.get("symbol") else None,
        names=_dedup(names),
        groups=groups,
        family=_dedup(family),
        error=last_error,
        matched_by=matched_by,
        family_withheld=bool(include_family and group_ids and not strong),
    )
    if use_cache:
        _cache_put(cache_key, query, result.__dict__)
    return result


def expand_symbols(
    tokens: Sequence[str],
    *,
    include_family: bool = True,
    max_lookups: int = 8,
) -> tuple[list[str], list[GeneVocabulary], Optional[str]]:
    """
    Resolve every plausible gene symbol in `tokens` against HGNC.

    Returns (new terms, the records behind them, first error). `max_lookups`
    bounds the number of network round-trips a single run can spend here, since
    a generous model expansion can propose a dozen symbol-shaped strings.
    """
    seen: set[str] = set()
    terms: list[str] = []
    records: list[GeneVocabulary] = []
    error: Optional[str] = None
    lookups = 0

    for token in tokens:
        if lookups >= max_lookups:
            break
        if not looks_like_gene_symbol(token):
            continue
        lookups += 1
        record = fetch_gene(token, include_family=include_family)
        if record.error and error is None:
            error = record.error
        if not record.resolved:
            continue
        records.append(record)
        for term in record.all_terms(include_family=include_family):
            key = term.lower()
            if key not in seen:
                seen.add(key)
                terms.append(term)

    return terms, records, error


def _dedup(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        term = str(value).strip()
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            out.append(term)
    return out


__all__ = [
    "GeneVocabulary",
    "fetch_gene",
    "expand_symbols",
    "looks_like_gene_symbol",
    "init_cache",
    "clear_cache",
    "set_db_path",
    "CREATE_CACHE_SQL",
    "HGNC_BASE_URL",
]
