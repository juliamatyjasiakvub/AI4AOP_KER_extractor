from __future__ import annotations

"""
Coarse cell lineage from the free text a paper used.

Papers name cells at whatever resolution suits their argument: "oligodendrocyte
precursor cells", "NG2+ oligodendrocyte progenitor cells", "mESC-derived
oligodendrocyte progenitor cells" and "immature oligodendrocytes (spiking OLs)"
are four strings for one lineage, while "calyx of Held nerve terminal" is a
different lineage entirely.

That difference is the one that matters for a Key Event. Voltage-gated sodium
channel activity in an oligodendrocyte and voltage-gated sodium channel
activity in a presynaptic nerve terminal are not the same event: different
cells, different channels, different consequences, and pooling them produces a
node that averages two unrelated literatures. Distinguishing every string, on
the other hand, would split one lineage into nine nodes and make the map
unreadable.

So the strings are reduced to a handful of lineages, and only a disagreement
at THAT level is treated as two Key Events. Anything finer is a synonym.

Deliberately keyword-based rather than a model call: it runs over thousands of
rows, must give the same answer every time, and the vocabulary is small and
stable.
"""

import re
from typing import Optional

#: Checked in order. The first pattern that matches decides, which is why
#: oligodendroglial comes before neuronal — "oligodendrocyte lineage cells
#: (brainstem, MNTB)" is an oligodendrocyte string that mentions a nucleus,
#: not an axonal one.
_LINEAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "oligodendroglial",
        r"oligodendro|\bOPC\b|\bOPCs\b|\bOL\b|\bOLs\b|pre-?OL|NG2\+?|"
        r"myelinating cell|myelin-forming",
    ),
    ("microglial", r"microglia|macrophage|\bIba1\b"),
    ("astroglial", r"astrocyt|\bGFAP\b"),
    (
        "neuronal / axonal",
        r"\baxon|neuron|calyx|nerve terminal|presynap|postsynap|synapse|"
        r"node of ranvier|heminode|internode|\bMNTB\b|\bsoma\b|dendrit|"
        r"granule cell|pyramidal",
    ),
    ("vascular", r"endotheli|pericyte|blood-brain"),
)

#: How each lineage reads when appended to a Key Event name.
SUFFIXES = {
    "oligodendroglial": "in oligodendrocytes",
    "neuronal / axonal": "in neurons/axons",
    "microglial": "in microglia",
    "astroglial": "in astrocytes",
    "vascular": "in vascular cells",
}

UNSPECIFIED = "unspecified"


def lineage(cell_type: Optional[str]) -> str:
    """
    Reduce a free-text cell type to one lineage.

    Returns `UNSPECIFIED` for an empty or unrecognised string, which is not the
    same as a conflict: a paper that did not localise its finding has not
    disagreed with one that did.
    """
    text = (cell_type or "").strip()
    if not text:
        return UNSPECIFIED
    for name, pattern in _LINEAGE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return name
    return UNSPECIFIED


def suffix_for(lineage_name: str) -> str:
    """The phrase to append when a Key Event is split by lineage."""
    return SUFFIXES.get(lineage_name, f"in {lineage_name}")


def distinct_lineages(cell_types) -> list[str]:
    """
    The identified lineages in a collection of cell-type strings.

    `UNSPECIFIED` is excluded on purpose. Two rows, one saying "oligodendrocyte"
    and one saying nothing, are not evidence of two Key Events.
    """
    found = {lineage(c) for c in cell_types or []}
    found.discard(UNSPECIFIED)
    return sorted(found)


__all__ = ["lineage", "suffix_for", "distinct_lineages", "SUFFIXES", "UNSPECIFIED"]
