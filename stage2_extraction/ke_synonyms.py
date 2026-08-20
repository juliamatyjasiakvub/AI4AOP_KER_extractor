"""
Expand a Key Event label into the vocabulary papers actually use.

The problem this solves
-----------------------
A Key Event is named the way an AOP names it — "Altered voltage-gated sodium
channel kinetics", "Decreased oligodendrocyte differentiation". Papers do not
write that. They write NaV1.6, VGSC, SCN8A, TTX-sensitive current, OPC,
oligodendrocyte precursor cell, myelination, MBP expression.

Two parts of the pipeline match on those labels literally:

* the heuristic chunk scorer, which ranks passages by how many label words they
  contain and drops everything below a threshold, and
* the relevance gate, which asks the model whether the paper bears on the
  relationship as named.

Both therefore miss a paper that is entirely about the relationship but never
uses the AOP's phrasing. The failure is silent and looks exactly like a paper
being irrelevant, which is the worst possible way for it to fail: a screened
corpus comes back almost empty and the emptiness reads as a finding.

A single misspelling has the same effect. "differentation" tokenises to a term
that appears in no paper ever written, so that half of the relationship
contributes nothing to any chunk's score.

What this module does
---------------------
`expand_label` asks the model once per label for the synonyms, abbreviations,
gene and protein names, and cell-type aliases a paper might use instead, then
caches the answer. The result is meant to be shown to the curator and edited —
which terms count as the same Key Event is a scientific judgement, not one to
settle silently inside a scorer.

There is no LLM-free path that would be honest here. A hard-coded lexicon would
cover the examples its author thought of and quietly fail on everything else,
which is the failure mode this module exists to remove. Without a model the
functions fall back to morphological variants of the label's own words — enough
to survive a typo or a plural, and nothing more, which is stated plainly rather
than dressed up as coverage.
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

#: Synonym sets are a property of the biology, not of anything that moves, so
#: they are worth keeping for a long time. Thirty days is short enough that a
#: corrected label eventually takes effect on its own.
_CACHE_TTL_SECONDS = 30 * 24 * 3600

CREATE_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS ke_synonym_cache (
    cache_key   TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    model       TEXT NOT NULL,
    payload     TEXT NOT NULL,   -- JSON list of synonym strings
    fetched_at  REAL NOT NULL
)
"""


def set_db_path(path: Path | str) -> None:
    """Point the cache at a different SQLite file (used by the store module)."""
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
            conn.execute("DELETE FROM ke_synonym_cache")
            conn.commit()
    except sqlite3.Error:
        pass


def _cache_key(label: str, model: str) -> str:
    return f"{_normalise(label)}::{model.strip().lower()}"


def _cache_get(key: str) -> Optional[list[str]]:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            row = conn.execute(
                "SELECT payload, fetched_at FROM ke_synonym_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    payload, fetched_at = row
    if time.time() - float(fetched_at) > _CACHE_TTL_SECONDS:
        return None
    try:
        terms = json.loads(payload)
        return [str(t) for t in terms] if isinstance(terms, list) else None
    except Exception:
        return None


def _cache_put(key: str, label: str, model: str, terms: Sequence[str]) -> None:
    try:
        with sqlite3.connect(_db()) as conn:
            conn.execute(CREATE_CACHE_SQL)
            conn.execute(
                "INSERT OR REPLACE INTO ke_synonym_cache "
                "(cache_key, label, model, payload, fetched_at) VALUES (?, ?, ?, ?, ?)",
                (key, label, model, json.dumps(list(terms)), time.time()),
            )
            conn.commit()
    except sqlite3.Error:
        pass  # a cache miss costs one model call; a crash costs the run


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LabelExpansion:
    """The search vocabulary for one Key Event label."""

    label: str
    #: Everything worth matching on, including the label's own content words.
    terms: list[str] = field(default_factory=list)
    #: True when a model produced these, False when they are the morphological
    #: fallback. The UI says which, because the two are not comparable in
    #: coverage and the curator should know which one is in play.
    from_model: bool = False
    error: Optional[str] = None

    def as_query_terms(self) -> list[str]:
        """Lowercased, de-duplicated, longest first — what the scorer wants."""
        seen: set[str] = set()
        out: list[str] = []
        for term in sorted(self.terms, key=len, reverse=True):
            key = term.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out


# ---------------------------------------------------------------------------
# Fallback: morphological variants of the label's own words
# ---------------------------------------------------------------------------

#: Words that carry no discriminating power in a Key Event label. Direction
#: words are included deliberately: "decreased" tells you which way the event
#: moves, which the gate needs, but as a search term it matches every paper
#: that ever reported a decrease in anything.
_LABEL_STOPWORDS = {
    "altered", "changed", "modified", "affected", "disrupted", "abnormal",
    "aberrant", "increased", "decreased", "reduced", "elevated", "impaired",
    "enhanced", "loss", "gain", "level", "levels", "activity", "function",
    "and", "the", "with", "from", "into", "that", "this", "them", "then",
    "leads", "lead", "cause", "causes", "caused", "event", "key",
}

_SUFFIXES = ("ation", "ations", "ing", "ed", "es", "s", "ion", "ions")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _content_words(label: str) -> list[str]:
    return [
        token
        for token in re.split(r"[^a-zA-Z0-9]+", (label or "").lower())
        if len(token) > 3 and token not in _LABEL_STOPWORDS
    ]


def _stem(word: str) -> str:
    """Crude longest-suffix strip. Good enough to bridge a plural or a typo."""
    for suffix in _SUFFIXES:
        if len(word) - len(suffix) >= 5 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def morphological_terms(label: str) -> list[str]:
    """
    The label's own content words plus their stems.

    The stem is what rescues a misspelling: "differentation" and
    "differentiation" both stem toward "different", so a passage using the
    correct spelling still scores against a label carrying the typo. It is a
    weak net and it is meant to be — the real coverage comes from `expand_label`.
    """
    terms: set[str] = set()
    for word in _content_words(label):
        terms.add(word)
        stem = _stem(word)
        if len(stem) >= 5:
            terms.add(stem)
    return sorted(terms, key=len, reverse=True)


# ---------------------------------------------------------------------------
# Model-driven expansion
# ---------------------------------------------------------------------------

_MAX_TERMS = 24

_EXPANSION_TASK = (
    "TASK: A Key Event in an Adverse Outcome Pathway is named below in formal "
    "AOP wording. Papers reporting this event rarely use that wording. List the "
    "terms a primary research paper would use instead, so a keyword search can "
    "find the relevant passages.\n\n"
    "Key Event: {label}\n\n"
    "Include, where they exist:\n"
    "- the entity's common name and its abbreviations (e.g. a channel and its "
    "two- to six-letter acronym)\n"
    "- gene and protein symbols, including specific isoforms and subunits\n"
    "- cell-type names, their abbreviations, and the precursor or progenitor "
    "cell if the event concerns a mature cell type\n"
    "- the standard experimental readouts and marker genes used to measure the "
    "event\n"
    "- the process under its other common names\n\n"
    "Rules:\n"
    "- Terms only. No direction words: give the entity and the process, not "
    "whether it went up or down. A search term must match papers reporting "
    "either direction.\n"
    "- Each term must be something that would literally appear in a paper's "
    "text. No definitions, no explanations.\n"
    "- Prefer specific over generic. A term so broad it appears in every paper "
    "in the field is worse than useless — it makes every passage look relevant.\n"
    f"- At most {_MAX_TERMS} terms.\n\n"
    "Return ONLY JSON:\n"
    '  {{"terms": ["<term>", "<term>"]}}\n'
    "JSON:"
)


def _parse_terms(parsed: Any) -> list[str]:
    """Pull a clean term list out of whatever shape the model returned."""
    if isinstance(parsed, dict):
        raw = parsed.get("terms") or parsed.get("synonyms") or []
    elif isinstance(parsed, list):
        raw = parsed
    else:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        term = re.sub(r"\s+", " ", item).strip().strip(".,;:\"'()[]")
        # A "term" long enough to be a sentence is an explanation the model
        # slipped in, and it would match nothing. Drop it.
        if not (2 <= len(term) <= 60) or len(term.split()) > 6:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= _MAX_TERMS:
            break
    return terms


def expand_label(
    label: str,
    cfg: Any = None,
    *,
    use_cache: bool = True,
    on_step: Any = None,
) -> LabelExpansion:
    """
    Return the search vocabulary for `label`, from cache, model, or fallback.

    `cfg` is an `LLMConfig`; passing None skips the model entirely and returns
    the morphological fallback, which is the right behaviour for a caller that
    has no provider configured rather than a reason to raise.
    """
    label = (label or "").strip()
    if not label:
        return LabelExpansion(label=label, terms=[], from_model=False)

    base = morphological_terms(label)

    if cfg is None:
        return LabelExpansion(label=label, terms=base, from_model=False)

    model = str(getattr(cfg, "model", "") or "")
    key = _cache_key(label, model)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return LabelExpansion(
                label=label,
                terms=_merge(base, cached),
                from_model=True,
            )

    # Imported here rather than at module scope: this module is imported by the
    # chunk scorer, and ker_extractor imports the chunk scorer.
    from stage2_extraction.ker_extractor import ExtractionError, _run_step, step_budget

    try:
        step = _run_step(
            "ke_synonyms",
            _EXPANSION_TASK.format(label=label),
            cfg=cfg,
            on_step=on_step,
            num_predict=step_budget("ke_synonyms"),
        )
    except ExtractionError as exc:
        return LabelExpansion(
            label=label, terms=base, from_model=False, error=str(exc)
        )
    except Exception as exc:  # noqa: BLE001 — a synonym lookup must not end a run
        return LabelExpansion(
            label=label, terms=base, from_model=False, error=f"{type(exc).__name__}: {exc}"
        )

    if not step.ok:
        return LabelExpansion(
            label=label, terms=base, from_model=False,
            error=step.error or "The model's reply could not be parsed.",
        )

    terms = _parse_terms(step.parsed)
    if not terms:
        return LabelExpansion(
            label=label, terms=base, from_model=False,
            error="The model returned no usable terms.",
        )

    if use_cache:
        _cache_put(key, label, model, terms)

    return LabelExpansion(label=label, terms=_merge(base, terms), from_model=True)


def _merge(base: Sequence[str], extra: Sequence[str]) -> list[str]:
    """Union of two term lists, longest first, case-insensitively unique."""
    seen: set[str] = set()
    out: list[str] = []
    for term in sorted([*base, *extra], key=len, reverse=True):
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(term.strip())
    return out


# ---------------------------------------------------------------------------
# Ontology synonyms (OLS4)
# ---------------------------------------------------------------------------

#: Ontologies that carry the cell types and processes a Key Event names. The
#: gene side is HGNC's job; this is for "oligodendrocyte precursor cell",
#: "myelination", "cell differentiation" — terms with curated exact synonyms.
_SYNONYM_ONTOLOGIES = ("cl", "go", "uberon")

#: OLS4 has moved field names between versions and different deployments
#: return different shapes, so every plausible key is read rather than assuming
#: one. An absent synonym list is an ordinary outcome, not an error.
_SYNONYM_KEYS = ("synonym", "obo_synonym", "synonyms", "has_exact_synonym")


def _synonyms_from_doc(doc: Any) -> list[str]:
    """Pull synonym strings out of one OLS4 document, whatever shape it has."""
    if not isinstance(doc, dict):
        return []
    out: list[str] = []
    for key in _SYNONYM_KEYS:
        value = doc.get(key)
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    # obo_synonym entries are objects carrying {name, scope}.
                    name = item.get("name") or item.get("value")
                    if isinstance(name, str):
                        out.append(name)
    label = doc.get("label")
    if isinstance(label, str):
        out.append(label)
    return out


def ontology_synonyms(
    label: str,
    *,
    ontologies: Sequence[str] = _SYNONYM_ONTOLOGIES,
    rows: int = 3,
    timeout: int = 10,
    use_cache: bool = True,
) -> tuple[list[str], Optional[str]]:
    """
    Curated synonyms for a cell type or biological process, from OLS4.

    Returns (terms, error). Like every other external lookup here, failure is
    silent in the sense that it costs coverage rather than the run: an
    ontology server being unavailable must not stop a literature screen.
    """
    label = (label or "").strip()
    if not label:
        return [], None

    key = f"ols4syn::{_normalise(label)}::{','.join(sorted(ontologies))}"
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached, None

    try:
        import requests

        from stage2_extraction.ols4_client import OLS4_BASE_URL, sanitise_query

        response = requests.get(
            f"{OLS4_BASE_URL.rstrip('/')}/search",
            params={
                "q": sanitise_query(label) or label,
                "ontology": ",".join(ontologies),
                "type": "class",
                "rows": rows,
                "fieldList": "label,obo_id,ontology_name,synonym,obo_synonym",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 — optional enrichment
        return [], f"OLS4 synonym lookup failed: {exc}"

    docs: Any = []
    if isinstance(payload, dict):
        container = payload.get("response")
        if isinstance(container, dict):
            docs = container.get("docs") or []
        elif isinstance(payload.get("docs"), list):
            docs = payload["docs"]

    terms: list[str] = []
    seen: set[str] = set()
    for doc in docs if isinstance(docs, list) else []:
        for term in _synonyms_from_doc(doc):
            cleaned = re.sub(r"\s+", " ", term).strip()
            # Ontology synonyms include long formal phrases that match nothing
            # in running text; keep the ones that read like something a paper
            # would write.
            if not (2 <= len(cleaned) <= 50) or len(cleaned.split()) > 5:
                continue
            k = cleaned.lower()
            if k not in seen:
                seen.add(k)
                terms.append(cleaned)

    if use_cache and terms:
        _cache_put(key, label, "ols4", terms)
    return terms, None


# ---------------------------------------------------------------------------
# Abbreviations defined by the paper itself
# ---------------------------------------------------------------------------

#: A parenthesised candidate abbreviation: two to ten characters, containing at
#: least one capital, no spaces. Excludes citations, p-values and units, which
#: is most of what else appears in brackets in a results section.
_ABBREV_CANDIDATE = re.compile(r"\(([A-Za-z][A-Za-z0-9\-/]{1,9})s?\)")

_ABBREV_REJECT = re.compile(
    r"^(?:fig|figs|table|ref|refs|see|e\.?g|i\.?e|vs|et\s*al|"
    r"n|p|sd|sem|ci|df|min|max|hr|hrs|sec|mm|cm|um|nm|ml|mg|kg|mol|mM|uM|nM)$",
    re.IGNORECASE,
)


def _initials_match(phrase_words: Sequence[str], abbrev: str) -> bool:
    """
    Schwartz–Hearst style check: does `abbrev` spell out of `phrase_words`?

    Every character of the abbreviation must appear in the phrase, in order,
    with the first character starting a word. That accepts "oligodendrocyte
    precursor cell (OPC)" and "voltage-gated sodium channel (VGSC)" while
    rejecting a bracketed aside that happens to sit after a noun phrase.
    """
    letters = [c.lower() for c in abbrev if c.isalnum()]
    if not letters:
        return False

    joined = " ".join(phrase_words).lower()
    if not joined:
        return False

    # The first letter must begin a word in the phrase.
    starts = {w[0] for w in re.split(r"[^a-z0-9]+", joined) if w}
    if letters[0] not in starts:
        return False

    position = 0
    for letter in letters:
        position = joined.find(letter, position)
        if position < 0:
            return False
        position += 1
    return True


def paper_abbreviations(text: str, max_pairs: int = 200) -> dict[str, str]:
    """
    Harvest the abbreviations a paper defines for itself.

    Papers introduce their own shorthand on first use — "oligodendrocyte
    precursor cells (OPCs)", "voltage-gated sodium channel (VGSC)" — and which
    shorthand a given paper picks is that lab's convention, not something a
    curator can predict. One paper writes OL, the next writes OPC, a third
    writes nothing at all and spells it out every time.

    Reading the definitions out of each paper turns that from a guess into a
    fact about the document in hand. Returns {abbreviation: long form}.
    """
    found: dict[str, str] = {}
    if not text:
        return found

    for match in _ABBREV_CANDIDATE.finditer(text):
        abbrev = match.group(1)
        # "(OPCs)" defines OPC, not OPCs. Storing the plural would be a term
        # that never matches the singular, since matching is boundary-anchored.
        if len(abbrev) > 2 and abbrev.endswith("s") and not abbrev[:-1].endswith("s"):
            abbrev = abbrev[:-1]
        if _ABBREV_REJECT.match(abbrev):
            continue
        if not any(c.isupper() for c in abbrev):
            continue

        # Look back over roughly as many words as the abbreviation has
        # characters, which is the window the definition almost always fits in.
        before = text[max(0, match.start() - 160):match.start()]
        # A definition never crosses a sentence boundary. Without this the
        # window runs back into the previous sentence and picks up whatever
        # was there: "pre-OL" came back defined as "water-immersionobjective
        # Zeiss considered premyelinating oligodendrocytes", and "DFG" as
        # "differentiate into oligodendrocytes Forschungsgemeinschaft".
        # Those then went into the extraction prompt as facts about the paper.
        boundary = max(before.rfind(". "), before.rfind("! "), before.rfind("? "),
                       before.rfind("\n"), before.rfind(";"))
        if boundary != -1:
            before = before[boundary + 1:]
        words = re.findall(r"[A-Za-z0-9\-]+", before)
        if not words:
            continue
        # Schwartz–Hearst bounds the long form at min(|A|+5, |A|*2) words. Any
        # longer and the "definition" is just the sentence it sits in.
        n_letters = sum(1 for c in abbrev if c.isalnum())
        max_words = min(n_letters + 5, n_letters * 2)
        window = words[-max_words:]

        # Shortest first. Taking the longest phrase that happens to satisfy the
        # initials test is how "(DMEM)" ends up defined as "Cells were
        # maintained in Dulbecco modified Eagle medium" — which then shares the
        # word "cells" with the vocabulary and survives filtering. The tightest
        # phrase that spells the abbreviation is the definition.
        for start in range(len(window) - 1, -1, -1):
            phrase_words = window[start:]
            if len(phrase_words) < n_letters:
                continue  # too short to spell the abbreviation out of words
            if _initials_match(phrase_words, abbrev):
                long_form = " ".join(phrase_words)

                # Schwartz-Hearst proper: the long form begins with the
                # abbreviation's first letter. Cheap, and it rejects most of
                # what a mangled PDF text layer produces — "CCD" was being
                # defined as "Basic-ZHE Semrock andtwocharge-coupleddevice".
                first_letter = next(
                    (c for c in abbrev if c.isalnum()), ""
                ).lower()
                if first_letter and not phrase_words[0][:1].lower() == first_letter:
                    break

                # A "word" of 25 characters is not a word. It is two columns
                # of a PDF run together, and nothing built from it is a real
                # definition.
                if any(len(w) > 25 for w in phrase_words):
                    break

                if 3 <= len(long_form) <= 90:
                    found.setdefault(abbrev, long_form)
                break

        if len(found) >= max_pairs:
            break

    return found


def abbreviations_for_terms(
    text: str,
    terms: Sequence[str],
    *,
    min_overlap: int = 1,
) -> dict[str, str]:
    """
    The subset of a paper's abbreviations whose long form is already vocabulary.

    An abbreviation is only useful here if it stands for something the Key
    Event is about. "(DMEM)" is defined in every methods section and means
    nothing to the screen; "(OPC)" matters precisely because "oligodendrocyte
    precursor cell" is already in the vocabulary. Matching on shared content
    words rather than on the whole string keeps "oligodendrocyte progenitor
    cell" and "oligodendrocyte precursor cell" both connected to the same
    Key Event.
    """
    vocabulary_words: set[str] = set()
    for term in terms:
        for word in re.split(r"[^a-zA-Z0-9]+", (term or "").lower()):
            if len(word) > 3 and word not in _LABEL_STOPWORDS:
                vocabulary_words.add(word)
                stem = _stem(word)
                if len(stem) >= 5:
                    vocabulary_words.add(stem)

    if not vocabulary_words:
        return {}

    keep: dict[str, str] = {}
    for abbrev, long_form in paper_abbreviations(text).items():
        long_words = [
            w for w in re.split(r"[^a-zA-Z0-9]+", long_form.lower()) if len(w) > 3
        ]
        overlap = sum(
            1
            for w in long_words
            if w in vocabulary_words or any(w.startswith(v) for v in vocabulary_words)
        )
        if overlap >= min_overlap:
            keep[abbrev] = long_form
    return keep


# ---------------------------------------------------------------------------
# The whole vocabulary for one Key Event
# ---------------------------------------------------------------------------

@dataclass
class KEVocabulary:
    """Search vocabulary for one Key Event, with its provenance kept."""

    label: str
    terms: list[str] = field(default_factory=list)
    #: Where each group of terms came from, for the run record and the UI.
    from_label: list[str] = field(default_factory=list)
    from_model: list[str] = field(default_factory=list)
    from_hgnc: list[str] = field(default_factory=list)
    from_ontology: list[str] = field(default_factory=list)
    #: HGNC gene groups that were resolved, e.g. "Sodium voltage-gated channel
    #: alpha subunits" — worth surfacing, since it is the claim that a paper on
    #: any family member counts as being about this event.
    gene_groups: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"{len(self.terms)} terms"]
        if self.from_hgnc:
            bits.append(f"{len(self.from_hgnc)} from HGNC")
        if self.from_ontology:
            bits.append(f"{len(self.from_ontology)} from ontology")
        if self.gene_groups:
            bits.append("family: " + "; ".join(self.gene_groups))
        return " · ".join(bits)


def build_vocabulary(
    label: str,
    cfg: Any = None,
    *,
    use_hgnc: bool = True,
    use_ontology: bool = True,
    on_step: Any = None,
) -> KEVocabulary:
    """
    Assemble everything a Key Event might be called, from four sources.

    The sources are deliberately different in kind, because no single one
    covers a Key Event:

    * the label's own words and stems — free, and survives a typo;
    * a model expansion — proposes the candidate gene symbols, cell types and
      readouts, which is the step that knows "sodium channel" implies NaV;
    * HGNC — turns those candidates into authoritative symbols, aliases and,
      crucially, the whole isoform family, so a paper measuring NaV1.2 counts
      for a Key Event seeded with NaV1.6;
    * OLS4 — curated synonyms for the cell types and processes, which HGNC
      does not cover.

    Nothing here is required. Every external source degrades to "fewer terms"
    rather than to an exception.
    """
    label = (label or "").strip()
    vocab = KEVocabulary(label=label)
    if not label:
        return vocab

    vocab.from_label = morphological_terms(label)

    expansion = expand_label(label, cfg, on_step=on_step)
    if expansion.error:
        vocab.notes.append(f"Model expansion: {expansion.error}")
    # expand_label already merges the morphological terms in; keep only what
    # the model added so the provenance stays honest.
    label_set = {t.lower() for t in vocab.from_label}
    vocab.from_model = [t for t in expansion.terms if t.lower() not in label_set]

    if use_hgnc:
        try:
            from stage2_extraction import gene_registry

            candidates = [*vocab.from_model, *label.split()]
            hgnc_terms, records, hgnc_error = gene_registry.expand_symbols(candidates)
            vocab.from_hgnc = hgnc_terms
            for record in records:
                for group in record.groups:
                    if group not in vocab.gene_groups:
                        vocab.gene_groups.append(group)
            if hgnc_error:
                vocab.notes.append(f"HGNC: {hgnc_error}")
        except Exception as exc:  # noqa: BLE001
            vocab.notes.append(f"HGNC lookup skipped: {type(exc).__name__}: {exc}")

    if use_ontology:
        onto_terms, onto_error = ontology_synonyms(label)
        vocab.from_ontology = onto_terms
        if onto_error:
            vocab.notes.append(onto_error)

    vocab.terms = _merge(
        vocab.from_label,
        [*vocab.from_model, *vocab.from_hgnc, *vocab.from_ontology],
    )
    return vocab


def parse_user_terms(text: str) -> list[str]:
    """
    Read an edited term list back from the UI.

    Accepts commas or newlines as separators, since a curator pasting from a
    paper will use whichever the source used.
    """
    parts = re.split(r"[,\n;]+", text or "")
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        term = re.sub(r"\s+", " ", part).strip()
        if not term:
            continue
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


__all__ = [
    "LabelExpansion",
    "KEVocabulary",
    "build_vocabulary",
    "ontology_synonyms",
    "expand_label",
    "morphological_terms",
    "paper_abbreviations",
    "abbreviations_for_terms",
    "parse_user_terms",
    "init_cache",
    "clear_cache",
    "set_db_path",
    "CREATE_CACHE_SQL",
]
