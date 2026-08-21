from __future__ import annotations

"""
OLS4 (EBI Ontology Lookup Service) enrichment for free-text Key Event labels.

The extractor produces natural-language KE names taken from whatever wording
each paper happened to use ("mitochondrial ROS accumulation", "increased
mitochondrial reactive oxygen species"). To merge those reliably and to link
them to external resources we attach canonical ontology terms: a label, a CURIE
(GO:0006979), an IRI, and a match score.

Design notes
------------
* Every lookup is cached in SQLite. Ontology terms change rarely and a curation
  session re-queries the same handful of labels constantly, so the cache turns
  a network-bound operation into a local one.
* The service is treated as strictly optional. If OLS4 is unreachable, every
  function returns empty results and sets `last_error`; nothing in the pipeline
  fails because an ontology server was down.
* Scores blend OLS4's own relevance ordering with a local string-similarity
  measure, because OLS4 ranks by text index and will happily return a weak
  lexical match as its top hit.

Ontology choice
---------------
Default ontologies are the ones that actually carry AOP-relevant terms:

    go      biological processes and molecular functions
    uberon  anatomical structures (tissue and organ level KEs)
    cl      cell types
    chebi   chemical entities (stressors)
    hp      human phenotypes (individual level)
    mp      mammalian phenotypes (individual level)
    pato    qualities ("increased", "decreased")
    ncbitaxon  species, for taxonomic applicability
"""

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional, Sequence

import requests

from schemas import KE_LEVEL_ORDER, OntologyMatch

OLS4_BASE_URL = "https://www.ebi.ac.uk/ols4/api"

DEFAULT_ONTOLOGIES: tuple[str, ...] = ("go", "uberon", "cl", "hp", "mp", "chebi", "pato")

#: Which ontologies are most informative for each biological level. Searching
#: level-appropriate ontologies first materially improves match quality:
#: an organ-level KE should be matched against UBERON, not GO.
LEVEL_ONTOLOGY_PREFERENCE: dict[str, tuple[str, ...]] = {
    "MIE":        ("go", "chebi", "pato"),
    "Molecular":  ("go", "chebi", "pato"),
    "Cellular":   ("go", "cl", "pato"),
    "Tissue":     ("uberon", "cl", "go", "pato"),
    "Organ":      ("uberon", "mp", "hp", "pato"),
    "Individual": ("hp", "mp", "pato", "go"),
    "Population": ("pato", "go"),
}

#: Ontologies whose terms describe anatomy — used by the physiological-map
#: linker to decide which KEs can be placed on a body map.
ANATOMY_ONTOLOGIES: frozenset[str] = frozenset({"uberon", "cl", "ma", "emapa", "fma"})

_CACHE_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

_STOPWORDS = {
    "increased", "decreased", "elevated", "reduced", "altered", "impaired",
    "enhanced", "induction", "of", "in", "the", "a", "an", "and", "to", "level",
    "levels", "activity", "response", "changes", "change",
}


@dataclass
class OLS4Result:
    """Outcome of one lookup: the ranked candidates plus any error."""

    query: str
    matches: list[OntologyMatch]
    error: Optional[str] = None
    from_cache: bool = False

    @property
    def best(self) -> Optional[OntologyMatch]:
        return self.matches[0] if self.matches else None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

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
CREATE TABLE IF NOT EXISTS ols4_cache (
    cache_key   TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    ontologies  TEXT NOT NULL,
    payload     TEXT NOT NULL,   -- JSON list of OntologyMatch dicts
    fetched_at  REAL NOT NULL
)
"""


def set_db_path(path: Path | str) -> None:
    """Point the cache at a different SQLite file (used by the store module)."""
    _LOCAL.db_path = None if path is None else Path(path)


def init_cache() -> None:
    with sqlite3.connect(_db()) as conn:
        conn.execute(CREATE_CACHE_SQL)
        conn.commit()


def _cache_key(query: str, ontologies: Sequence[str]) -> str:
    return f"{_normalise(query)}::{','.join(sorted(ontologies))}"


def _cache_get(key: str) -> Optional[list[OntologyMatch]]:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            row = conn.execute(
                "SELECT payload, fetched_at FROM ols4_cache WHERE cache_key = ?", (key,)
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    payload, fetched_at = row
    if time.time() - float(fetched_at) > _CACHE_TTL_SECONDS:
        return None
    try:
        return [OntologyMatch(**d) for d in json.loads(payload)]
    except Exception:
        return None


def _cache_put(key: str, query: str, ontologies: Sequence[str], matches: list[OntologyMatch]) -> None:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            conn.execute(
                "INSERT OR REPLACE INTO ols4_cache "
                "(cache_key, query, ontologies, payload, fetched_at) VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    query,
                    ",".join(ontologies),
                    json.dumps([m.to_dict() for m in matches]),
                    time.time(),
                ),
            )
            conn.commit()
    except sqlite3.Error:
        pass  # a cache write failure must never break enrichment


def clear_cache() -> None:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            conn.execute("DELETE FROM ols4_cache")
            conn.commit()
    except sqlite3.Error:
        pass


# ---------------------------------------------------------------------------
# Query sanitisation
#
# OLS4's /search endpoint is backed by Solr, and it passes the `q` parameter
# through to the Solr query parser rather than treating it as a literal
# phrase. Key Event labels are full of characters that parser treats as
# syntax: "+" means "this term is required", parentheses group clauses,
# and ":" introduces a field name. A label like
#
#     Nav1.2 channel activation (Na+ current)
#
# is therefore not a search for that phrase — it is a malformed query, and
# OLS4 answers with a 500 rather than an empty result set. Biological
# nomenclature makes this constant: Ca2+, Na+, K+, parenthetical
# clarifications, "NF-kB (p65)" and so on.
#
# Stripping the metacharacters costs nothing, because the scoring in
# `_similarity` already normalises them away when ranking candidates — they
# were never contributing to the match, only breaking the request.
# ---------------------------------------------------------------------------

#: Solr query-syntax metacharacters. Replaced with a space rather than escaped:
#: an escaped "+" would still be searched for as a literal, and no ontology
#: label contains one.
_SOLR_SPECIALS = re.compile(r'[+\-!(){}\[\]^"~*?:\\/&|]')


def sanitise_query(text: str) -> str:
    """Make `text` safe to hand to Solr, without changing what it means."""
    cleaned = _SOLR_SPECIALS.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _drop_parentheticals(text: str) -> str:
    """
    Remove bracketed asides entirely — the second attempt after a failure.

    Parenthetical content in a Key Event label is usually a clarification
    ("(Na+ current)", "(mitochondrial)") rather than the thing being named,
    so dropping it tends to produce a cleaner query, not a vaguer one.
    """
    without = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text or "")
    return sanitise_query(without)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _content_tokens(text: str) -> set[str]:
    return {t for t in _normalise(text).split() if t and t not in _STOPWORDS}


def _similarity(query: str, label: str) -> float:
    """Blend sequence similarity with content-token overlap."""
    q_norm, l_norm = _normalise(query), _normalise(label)
    if not q_norm or not l_norm:
        return 0.0
    if q_norm == l_norm:
        return 1.0

    seq = SequenceMatcher(None, q_norm, l_norm).ratio()

    q_tokens, l_tokens = _content_tokens(query), _content_tokens(label)
    if q_tokens and l_tokens:
        overlap = len(q_tokens & l_tokens) / len(q_tokens | l_tokens)
    else:
        overlap = 0.0

    # Containment is a strong signal: "oxidative stress" inside "cellular
    # response to oxidative stress" is a good match despite a mediocre ratio.
    containment = 1.0 if (q_norm in l_norm or l_norm in q_norm) else 0.0

    return round(min(1.0, 0.45 * seq + 0.40 * overlap + 0.15 * containment), 4)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def _parse_docs(docs: Sequence[dict], query: str) -> list[OntologyMatch]:
    matches: list[OntologyMatch] = []
    for rank, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        label = str(doc.get("label") or "").strip()
        iri = str(doc.get("iri") or "").strip()
        if not label or not iri:
            continue

        curie = str(doc.get("obo_id") or doc.get("short_form") or "").strip()
        if not curie:
            tail = iri.rstrip("/").split("/")[-1]
            curie = tail.replace("_", ":") if "_" in tail else tail

        ontology = str(
            doc.get("ontology_prefix") or doc.get("ontology_name") or ""
        ).strip().lower()

        description = doc.get("description")
        if isinstance(description, list):
            description = "; ".join(str(d) for d in description if d) or None
        elif description is not None:
            description = str(description)

        lexical = _similarity(query, label)
        # Blend OLS4's own ordering in gently — a term that OLS4 ranked first
        # gets a small bonus, decaying with rank.
        rank_bonus = 0.08 / (1.0 + rank)
        score = round(min(1.0, lexical + rank_bonus), 4)

        matches.append(
            OntologyMatch(
                query=query,
                label=label,
                curie=curie,
                iri=iri,
                ontology=ontology,
                score=score,
                description=description,
                is_exact=_normalise(label) == _normalise(query),
                obsolete=bool(doc.get("is_obsolete") or doc.get("obsolete")),
            )
        )

    matches = [m for m in matches if not m.obsolete]
    matches.sort(key=lambda m: (m.is_exact, m.score), reverse=True)
    return matches


def search(
    query: str,
    *,
    ontologies: Sequence[str] = DEFAULT_ONTOLOGIES,
    rows: int = 10,
    timeout: int = 12,
    use_cache: bool = True,
    base_url: str = OLS4_BASE_URL,
) -> OLS4Result:
    """
    Look up `query` in OLS4 and return ranked candidate terms.

    Never raises: transport and parse failures are reported via `.error`.
    """
    query = (query or "").strip()
    if not query:
        return OLS4Result(query=query, matches=[])

    key = _cache_key(query, ontologies)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return OLS4Result(query=query, matches=cached, from_cache=True)

    base_params = {
        "rows": rows,
        "ontology": ",".join(ontologies),
        "type": "class",
        "exact": "false",
        "fieldList": (
            "iri,label,short_form,obo_id,ontology_name,ontology_prefix,"
            "description,is_defining_ontology,type"
        ),
    }

    # Two attempts, narrowing the query each time. The first strips Solr
    # metacharacters; if the service still refuses, the parenthetical aside is
    # dropped as well. A second distinct query is only worth sending when it
    # actually differs from the first.
    attempts: list[str] = []
    for candidate in (sanitise_query(query), _drop_parentheticals(query)):
        if candidate and candidate not in attempts:
            attempts.append(candidate)
    if not attempts:
        return OLS4Result(
            query=query,
            matches=[],
            error=(
                f"Label {query!r} contains no searchable characters once query "
                "syntax is removed."
            ),
        )

    payload = None
    last_error = ""
    for attempt in attempts:
        try:
            response = requests.get(
                f"{base_url.rstrip('/')}/search",
                params={**base_params, "q": attempt},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            # 4xx/5xx here almost always means the query itself was rejected,
            # not that the service is down — saying "unreachable" sent people
            # to check their network when the label was the problem.
            last_error = (
                f"OLS4 rejected the query for {query!r} (HTTP {status}, sent as "
                f"{attempt!r})"
            )
            continue
        except requests.RequestException as exc:
            return OLS4Result(query=query, matches=[], error=f"OLS4 unreachable: {exc}")
        except ValueError as exc:
            return OLS4Result(
                query=query, matches=[], error=f"OLS4 returned invalid JSON: {exc}"
            )

    if payload is None:
        return OLS4Result(query=query, matches=[], error=last_error or "OLS4 request failed")

    # OLS4 nests results under "response"; be tolerant of shape changes.
    docs = []
    if isinstance(payload, dict):
        container = payload.get("response")
        if isinstance(container, dict):
            docs = container.get("docs") or []
        elif isinstance(payload.get("docs"), list):
            docs = payload["docs"]
        elif isinstance(payload.get("elements"), list):
            docs = payload["elements"]

    matches = _parse_docs(docs, query)
    if use_cache:
        _cache_put(key, query, ontologies, matches)
    return OLS4Result(query=query, matches=matches)


def annotate_ke(
    ke_name: str,
    level: Optional[str] = None,
    *,
    rows: int = 10,
    min_score: float = 0.45,
    timeout: int = 12,
    use_cache: bool = True,
) -> OLS4Result:
    """
    Find the best ontology term for one Key Event label.

    The ontology set is chosen from the KE's biological level, so a tissue-level
    event is matched against anatomy ontologies rather than gene-ontology
    processes. Candidates below `min_score` are discarded so that a weak match
    is reported as "no match" rather than as a confident wrong answer.
    """
    ontologies = LEVEL_ONTOLOGY_PREFERENCE.get(
        (level or "").strip(), DEFAULT_ONTOLOGIES
    )
    result = search(
        ke_name, ontologies=ontologies, rows=rows, timeout=timeout, use_cache=use_cache
    )
    if result.error:
        return result
    result.matches = [m for m in result.matches if m.score >= min_score]
    return result


def annotate_many(
    ke_names: Sequence[tuple[str, Optional[str]]],
    *,
    min_score: float = 0.45,
    timeout: int = 12,
    progress: Optional[callable] = None,
) -> dict[str, OLS4Result]:
    """
    Annotate a batch of (ke_name, level) pairs.

    Returns a dict keyed by the raw KE name. Deduplicates identical labels so a
    KE mentioned by many papers costs one request, not one per paper.
    """
    results: dict[str, OLS4Result] = {}
    unique: dict[str, Optional[str]] = {}
    for name, level in ke_names:
        if name and name not in unique:
            unique[name] = level

    total = len(unique)
    for i, (name, level) in enumerate(unique.items(), start=1):
        results[name] = annotate_ke(
            name, level, min_score=min_score, timeout=timeout
        )
        if progress is not None:
            try:
                progress(i, total, name)
            except Exception:
                pass
    return results


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

def _iri_for(curie: str, ontology: Optional[str] = None) -> Optional[str]:
    """Build an OBO PURL from a CURIE. Returns None for non-OBO shapes."""
    text = (curie or "").strip()
    if ":" not in text:
        return None
    prefix, local = text.split(":", 1)
    if not prefix or not local:
        return None
    return f"http://purl.obolibrary.org/obo/{prefix.upper()}_{local}"


def ancestors(
    curie: str,
    ontology: Optional[str] = None,
    *,
    timeout: int = 12,
    use_cache: bool = True,
    base_url: str = OLS4_BASE_URL,
) -> set[str]:
    """
    Every CURIE above `curie` in its ontology, transitively.

    Used by the merge classifier to tell a subtype from its class. The
    distinction is not cosmetic: pooling evidence about NaV1.2 with evidence
    about voltage-gated sodium channels attributes findings about one channel
    to every channel of that kind, and once merged there is nothing left to
    show it happened.

    Returns an empty set on any failure. A silent empty result is the right
    behaviour here because the classifier treats "no hierarchy information" as
    `unknown`, which withholds a merge rather than permitting one — being
    offline makes the tool more cautious, not less.
    """
    curie = (curie or "").strip().upper()
    if not curie:
        return set()

    ontology = (ontology or curie.split(":", 1)[0]).lower()
    cache_key = f"ancestors::{ontology}::{curie}"

    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {m.curie for m in cached if m.curie}

    return {m.curie for m in ancestor_terms(
        curie, ontology, timeout=timeout, use_cache=use_cache, base_url=base_url
    )}


def ancestor_terms(
    curie: str,
    ontology: Optional[str] = None,
    *,
    timeout: int = 12,
    use_cache: bool = True,
    base_url: str = OLS4_BASE_URL,
) -> list[OntologyMatch]:
    """
    As `ancestors`, but keeping each ancestor's label as well as its CURIE.

    The labels are what let a caller express a policy in words — "anything
    under 'glial cell' is one lineage" — instead of writing ontology
    identifiers into the source, where they are unreadable and cannot be
    checked without looking them up anyway. Same request, same cache entry;
    `ancestors` is now the CURIE-only view of this.
    """
    curie = (curie or "").strip().upper()
    if not curie:
        return []

    ontology = (ontology or curie.split(":", 1)[0]).lower()
    cache_key = f"ancestors::{ontology}::{curie}"

    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return list(cached)

    iri = _iri_for(curie, ontology)
    if not iri:
        return []

    import urllib.parse

    double_encoded = urllib.parse.quote(urllib.parse.quote(iri, safe=""), safe="")
    url = (
        f"{base_url.rstrip('/')}/ontologies/{ontology}/terms/"
        f"{double_encoded}/hierarchicalAncestors"
    )

    try:
        response = requests.get(url, params={"size": 200}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    terms = (payload.get("_embedded") or {}).get("terms") or []
    out: list[OntologyMatch] = []
    for term in terms:
        obo_id = (term.get("obo_id") or "").strip().upper()
        if not obo_id:
            continue
        out.append(
            OntologyMatch(
                query=curie,
                label=term.get("label") or obo_id,
                curie=obo_id,
                iri=term.get("iri") or "",
                ontology=(term.get("ontology_name") or ontology),
                score=1.0,
            )
        )

    if use_cache:
        _cache_put(cache_key, curie, (ontology,), out)

    return out


def ancestor_lookup(
    *, enabled: bool = True, timeout: int = 12
) -> Optional[Callable[[str, Optional[str]], set[str]]]:
    """
    A callable suited to `semantic_merge.classify(ancestors_of=...)`.

    Returns None when disabled, which the classifier reads as "no hierarchy
    available" — so turning the ontology off degrades the verdict to
    `uncertain` rather than silently allowing a subtype to merge into its
    parent.
    """
    if not enabled:
        return None

    def lookup(curie: str, ontology: Optional[str]) -> set[str]:
        return ancestors(curie, ontology, timeout=timeout)

    return lookup


def parents_for_mapping(
    label: str,
    *,
    ontologies: Sequence[str] = DEFAULT_ONTOLOGIES,
    rows: int = 8,
    timeout: int = 12,
) -> list[OntologyMatch]:
    """
    Candidate *broader* terms for a Key Event, for the map-to-parent action.

    Deliberately a different entry point from `annotate_ke`. That function
    answers "which term is this Key Event?"; this one answers "which term is
    this Key Event a kind of?". Keeping them apart in the API is what keeps
    them apart in the database, where one writes `ke_canonical.ontology_curie`
    and the other writes a row in `ontology_mapping`.
    """
    result = search(label, ontologies=ontologies, rows=rows, timeout=timeout)
    return result.matches


def check_availability(timeout: int = 8, base_url: str = OLS4_BASE_URL) -> tuple[bool, str]:
    """Ping OLS4. Returns (reachable, human-readable message)."""
    try:
        r = requests.get(
            f"{base_url.rstrip('/')}/search",
            params={"q": "apoptosis", "rows": 1, "ontology": "go"},
            timeout=timeout,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        return False, f"OLS4 not reachable: {exc}"
    return True, "OLS4 reachable."


__all__ = [
    "OLS4_BASE_URL",
    "DEFAULT_ONTOLOGIES",
    "LEVEL_ONTOLOGY_PREFERENCE",
    "ANATOMY_ONTOLOGIES",
    "OLS4Result",
    "sanitise_query",
    "search",
    "annotate_ke",
    "annotate_many",
    "ancestors",
    "ancestor_terms",
    "ancestor_lookup",
    "parents_for_mapping",
    "check_availability",
    "init_cache",
    "clear_cache",
    "set_db_path",
]
