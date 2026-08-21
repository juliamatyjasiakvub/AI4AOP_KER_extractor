from __future__ import annotations

"""
Coarse cell lineage for the free text a paper used.

Why a lineage at all
--------------------
Papers name cells at whatever resolution suits their argument: "oligodendrocyte
precursor cells", "NG2+ oligodendrocyte progenitor cells" and "mESC-derived
oligodendrocyte progenitor cells" are three strings for one lineage, while a
presynaptic nerve terminal is a different lineage entirely.

That difference is the one that matters for a Key Event. An ion-channel event
in an oligodendrocyte and the same event in a nerve terminal are not the same
event — different cells, different consequences — and pooling them produces a
node that averages two unrelated literatures. Distinguishing every string, on
the other hand, would split one lineage into nine nodes and make the map
unreadable. So strings are reduced to a handful of lineages, and only a
disagreement at THAT level is treated as two Key Events.

Where the answer comes from
---------------------------
The Cell Ontology, through `ols4_client`. A string is resolved to a CL class,
its ancestors are fetched, and the first entry of `LINEAGE_POLICY` that appears
among them decides. Resolution is per distinct string and cached twice — in
this process and in `ols4_cache` — so a table of several hundred rows costs a
handful of requests once and none thereafter.

This module used to answer from a table of regular expressions instead. That
table was written while reading a central-nervous-system corpus, and it covered
one: every one of its five lineages was a CNS cell type. Run a liver corpus
through it and hepatocytes, Kupffer cells and stellate cells all came back
`UNSPECIFIED` — which `distinct_lineages` discards, because a row that did not
say where it looked has not disagreed with one that did. So the findings pooled,
into exactly the averaged node this module exists to prevent, and nothing said
so. The regular expressions survive below as the offline fallback, which is
what they are good for: fast, no network, and right within the vocabulary they
happen to cover.

What stays local, and why
-------------------------
`LINEAGE_POLICY` is a policy, not a vocabulary. The Cell Ontology can tell you
that a Kupffer cell is a macrophage; it cannot tell you that "macrophage" is the
right granularity at which two Key Events become different, because that is a
question about adverse outcome pathways and not about cells. It is written in
ancestor *labels* rather than CURIEs deliberately — a list of identifiers is
unreadable and cannot be checked without looking every one of them up, which
would put ontology data back into the source file this comment argues against.
"""

import re
import threading
from typing import Callable, Iterable, Optional

__all__ = [
    "lineage",
    "suffix_for",
    "distinct_lineages",
    "unresolved_cell_types",
    "set_ontology_enabled",
    "set_resolver",
    "clear_cache",
    "LINEAGE_POLICY",
    "SUFFIXES",
    "UNSPECIFIED",
    "UNRESOLVED",
]

#: No cell type was recorded on the row. The paper did not localise its
#: finding, which is not a disagreement with a paper that did.
UNSPECIFIED = "unspecified"

#: A cell type WAS recorded and could not be placed in any lineage. Distinct
#: from `UNSPECIFIED` because the two mean opposite things about a corpus: one
#: says the papers were quiet, the other says this tool did not understand
#: them. Conflating them is what let a whole corpus pool in silence.
UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------

#: Checked in order; the first ancestor label found decides. Order encodes
#: specificity — a Kupffer cell is both a macrophage and a leukocyte, and
#: "macrophage" is the useful answer.
#:
#: Labels are matched case-insensitively against the resolved term itself and
#: everything above it, so an entry names a cut in the CL hierarchy rather than
#: one class.
LINEAGE_POLICY: tuple[tuple[str, str], ...] = (
    # Nervous system, at the resolution the AOP literature separates.
    ("oligodendrocyte precursor cell", "oligodendroglial"),
    ("oligodendrocyte", "oligodendroglial"),
    ("astrocyte", "astroglial"),
    ("microglial cell", "microglial"),
    ("neuron", "neuronal / axonal"),
    ("glial cell", "glial"),
    # Immune.
    ("macrophage", "macrophage"),
    ("lymphocyte", "lymphoid"),
    ("granulocyte", "granulocytic"),
    ("leukocyte", "leukocyte"),
    # Solid organs, where a toxicology corpus mostly lives.
    ("hepatocyte", "hepatocyte"),
    ("kidney tubule cell", "renal tubular"),
    ("podocyte", "podocyte"),
    ("cardiac muscle cell", "cardiomyocyte"),
    ("sertoli cell", "sertoli"),
    ("germ line cell", "germ line"),
    # Broad tissue classes last, so a specific answer always wins.
    ("endothelial cell", "endothelial"),
    ("fibroblast", "fibroblast"),
    ("epithelial cell", "epithelial"),
    ("muscle cell", "muscle"),
)

#: How each lineage reads when appended to a Key Event name.
SUFFIXES: dict[str, str] = {
    "oligodendroglial": "in oligodendrocytes",
    "astroglial": "in astrocytes",
    "microglial": "in microglia",
    "neuronal / axonal": "in neurons/axons",
    "glial": "in glia",
    "macrophage": "in macrophages",
    "lymphoid": "in lymphocytes",
    "granulocytic": "in granulocytes",
    "leukocyte": "in leukocytes",
    "hepatocyte": "in hepatocytes",
    "renal tubular": "in renal tubular cells",
    "podocyte": "in podocytes",
    "cardiomyocyte": "in cardiomyocytes",
    "sertoli": "in Sertoli cells",
    "germ line": "in germ cells",
    "endothelial": "in endothelial cells",
    "fibroblast": "in fibroblasts",
    "epithelial": "in epithelial cells",
    "muscle": "in muscle cells",
    "vascular": "in vascular cells",
}


# ---------------------------------------------------------------------------
# Offline fallback
# ---------------------------------------------------------------------------

#: Used only when the ontology is switched off or unreachable. Checked in
#: order, first match wins. Kept deliberately small: it is a cache of answers
#: the Cell Ontology would give for strings one corpus happened to contain, not
#: a vocabulary anybody should extend. Add an entry to `LINEAGE_POLICY`
#: instead — that works for every string CL knows, not the ones listed here.
_FALLBACK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("oligodendroglial",
     r"oligodendro|\bOPC\b|\bOPCs\b|\bOL\b|\bOLs\b|pre-?OL|NG2\+?|"
     r"myelinating cell|myelin-forming"),
    ("microglial", r"microglia|\bIba1\b"),
    ("astroglial", r"astrocyt|\bGFAP\b"),
    ("macrophage", r"macrophage|kupffer"),
    ("hepatocyte", r"hepatocyt|\bHepG2\b"),
    ("renal tubular", r"proximal tubul|renal tubul|\bPTEC\b|\bHK-?2\b"),
    ("podocyte", r"podocyt"),
    ("cardiomyocyte", r"cardiomyocyt|cardiac myocyt"),
    ("sertoli", r"sertoli"),
    ("neuronal / axonal",
     r"\baxon|neuron|calyx|nerve terminal|presynap|postsynap|synapse|"
     r"node of ranvier|heminode|internode|\bMNTB\b|\bsoma\b|dendrit|"
     r"granule cell|pyramidal"),
    ("endothelial", r"endotheli|blood-brain"),
    ("fibroblast", r"fibroblast|stellate cell"),
    ("vascular", r"pericyte|vascular smooth muscle"),
    ("epithelial", r"epitheli|alveolar type"),
)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

#: Minimum OLS4 score for a cell-type match to be trusted. Below this the
#: search has returned something loosely related rather than the term, and
#: guessing a lineage from it is worse than admitting the string is unresolved.
_MIN_SCORE = 0.55

_state = threading.local()
_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


def set_ontology_enabled(enabled: bool) -> None:
    """
    Turn CL resolution on or off for this session.

    Off falls back to the pattern table, which is what a run with the ontology
    disabled already does elsewhere in the pipeline. Thread-local, because
    Streamlit serves every browser session from one process.
    """
    _state.enabled = bool(enabled)
    clear_cache()


def _ontology_enabled() -> bool:
    """
    Off unless a caller has asked for it, matching `ols4_enabled` everywhere
    else in the pipeline.

    Defaulting to *on* was the obvious choice and the wrong one. `lineage()` is
    called from three modules and from the test suite, and reaching OLS4 also
    opens its cache — which, for any caller that has not been through
    `session_db.activate()`, means creating a database in the working
    directory. A module answering a question about a string should not decide
    on its own that this is the moment to touch the network and the disk. The
    application turns it on per run, where the user's ontology setting is
    known.
    """
    return getattr(_state, "enabled", False)


def clear_cache() -> None:
    """Forget resolved lineages. For tests, and after a policy change."""
    with _cache_lock:
        _cache.clear()


def _resolve_via_ontology(cell_type: str) -> Optional[str]:
    """
    The lineage CL puts this string in, or None if it cannot say.

    Every failure — ontology disabled, OLS4 unreachable, no confident match, no
    policy ancestor — returns None so the caller falls back. Being offline must
    make this tool less certain, never differently certain.
    """
    if not _ontology_enabled():
        return None
    try:
        from stage2_extraction import ols4_client

        result = ols4_client.search(cell_type, ontologies=("cl",), rows=3)
        best = next(
            (m for m in getattr(result, "matches", []) if m.score >= _MIN_SCORE),
            None,
        )
        if best is None:
            return None

        labels = {(best.label or "").strip().lower()}
        labels |= {
            (term.label or "").strip().lower()
            for term in ols4_client.ancestor_terms(best.curie, "cl")
        }
    except Exception:  # noqa: BLE001 - enrichment, never fatal
        return None

    for ancestor_label, lineage_name in LINEAGE_POLICY:
        if ancestor_label in labels:
            return lineage_name
    return None


_real_resolver = _resolve_via_ontology


def set_resolver(resolver: Optional[Callable[[str], Optional[str]]]) -> None:
    """
    Replace ontology resolution, for tests and for an offline run.

    Kept out of `lineage`'s signature because it is called once per row from
    three modules, and threading a resolver through all of them would put test
    scaffolding into the pipeline. Passing None restores the real one.
    """
    global _resolve_via_ontology  # noqa: PLW0603 - deliberate seam
    _resolve_via_ontology = resolver if resolver is not None else _real_resolver
    clear_cache()


def _resolve_via_patterns(cell_type: str) -> Optional[str]:
    for name, pattern in _FALLBACK_PATTERNS:
        if re.search(pattern, cell_type, re.I):
            return name
    return None


def lineage(cell_type: Optional[str]) -> str:
    """
    Reduce a free-text cell type to one lineage.

    Returns `UNSPECIFIED` when nothing was recorded and `UNRESOLVED` when
    something was recorded that neither the ontology nor the fallback could
    place. Callers must treat those differently: the first is silence, the
    second is a gap in this tool's coverage and belongs in the run's report.
    """
    text = (cell_type or "").strip()
    if not text:
        return UNSPECIFIED

    key = text.lower()
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit

    name = _resolve_via_ontology(text) or _resolve_via_patterns(text) or UNRESOLVED
    with _cache_lock:
        _cache[key] = name
    return name


def suffix_for(lineage_name: str) -> str:
    """The phrase to append when a Key Event is split by lineage."""
    return SUFFIXES.get(lineage_name, f"in {lineage_name}")


def distinct_lineages(cell_types: Optional[Iterable[str]]) -> list[str]:
    """
    The identified lineages in a collection of cell-type strings.

    `UNSPECIFIED` and `UNRESOLVED` are both excluded, for different reasons. A
    row that recorded nothing has not disagreed with one that did. A row this
    tool could not place has not agreed with one either — but a lineage split
    is a claim, and no claim should rest on a string nobody understood. Those
    rows are reported by `unresolved_cell_types` instead of being voted with.
    """
    found = {lineage(c) for c in cell_types or []}
    found.discard(UNSPECIFIED)
    found.discard(UNRESOLVED)
    return sorted(found)


def unresolved_cell_types(cell_types: Optional[Iterable[str]]) -> list[str]:
    """
    The recorded cell types that could not be placed in any lineage.

    A corpus where this is large is one whose Key Events are being merged with
    no cell-type check at all — the state a liver or kidney run was silently in
    before. Surfaced so that it reads as missing coverage rather than as
    agreement.
    """
    return sorted({
        str(c).strip()
        for c in cell_types or []
        if str(c or "").strip() and lineage(c) == UNRESOLVED
    })
