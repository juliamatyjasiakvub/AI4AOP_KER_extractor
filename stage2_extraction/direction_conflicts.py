from __future__ import annotations

"""
Key Events whose claims disagree about which way the event moved.

The map draws a "±" on a node when some of its claims recorded an increase and
others a decrease. That mark is correct and was, until now, the end of the
conversation: it stated that the corpus disagreed and offered no way to find
out which paper said what, no way to correct one that had been misread, and no
way to record a judgement. A flag that cannot be cleared stops being read.

Three things genuinely resolve a conflict, and they are different findings
rather than three routes to the same one:

    the extraction misread a paper  →  correct that claim
    two events share one name       →  split them
    the literature really is split  →  rule on it, with a reason

Only the third is stored (in `ke_direction`). The first two are edits to Table
1 and to the canonical grouping, which is where those answers belong.

This module holds the reading — what the claims say, what the curator ruled,
what the map should therefore draw — with no Streamlit in it, because the
question "do these papers agree?" is about the corpus and not about a page.
"""

from collections import Counter
from typing import Any, Optional

import pandas as pd

from stage2_extraction import ke_normalizer, table1_store

#: Prefixes that mark a recorded change as an increase or a decrease.
#:
#: Deliberately prefix matching rather than a word list: extraction writes
#: "reduced", "reduction", "reduced by 40%" and "loss of clustering" for the
#: same finding, and a list would silently drop the ones nobody thought of.
_UP_PREFIXES = ("increas", "elevat", "gain")
_DOWN_PREFIXES = ("decreas", "reduc", "lost", "loss", "abolish", "impair")

CONFLICT = "±"
NAME_MISMATCH = "⚠"


def sign_of(word: Any) -> str:
    """Which way one recorded change points, or '' where it does not say."""
    text = str(word or "").strip().lower()
    if text.startswith(_UP_PREFIXES):
        return "↑"
    if text.startswith(_DOWN_PREFIXES):
        return "↓"
    return ""


def dominant_change(counted: Counter) -> str:
    """
    One arrow for the node face, but only where the papers agree.

    Where they do not, the node says so rather than picking the winner: a Key
    Event that four papers push up and two push down is a finding about the
    corpus, not a tie to be broken by majority.
    """
    if not counted:
        return ""
    up = sum(n for word, n in counted.items() if sign_of(word) == "↑")
    down = sum(n for word, n in counted.items() if sign_of(word) == "↓")
    if up and down:
        return CONFLICT
    if up:
        return "↑"
    if down:
        return "↓"
    return ""


def resolved_change(
    canonical_id: int, derived: str, rulings: dict[int, dict[str, Any]]
) -> str:
    """
    The arrow to draw, after any curator ruling on this Key Event.

    A ruling of 'conflicted' deliberately leaves the "±" in place. It is an
    answer — the curator looked, and judges the disagreement real — not a
    dismissal, and a figure that tidied the mark away on the strength of it
    would be hiding the finding on the authority of the person who confirmed
    it.
    """
    ruling = rulings.get(int(canonical_id))
    if not ruling:
        return derived
    return {"increased": "↑", "decreased": "↓"}.get(
        str(ruling.get("direction")), derived
    )


def claims_for(table1: pd.DataFrame, canonical_id: int) -> pd.DataFrame:
    """
    The individual claims behind a mark, one row each.

    The map can only say that the corpus disagrees. Acting on it needs the
    other half of the sentence — which paper said which — and that was the
    part with nowhere to appear.
    """
    columns = [
        "record_id", "doi", "side", "change", "sign",
        "origin", "measured_as", "context",
    ]
    if table1.empty:
        return pd.DataFrame(columns=columns)

    rows = table1[
        (table1.get("upstream_ke_canonical_id") == canonical_id)
        | (table1.get("downstream_ke_canonical_id") == canonical_id)
    ]

    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        side = (
            "upstream"
            if row.get("upstream_ke_canonical_id") == canonical_id
            else "downstream"
        )
        change = str(row.get(f"{side}_change") or "").strip()
        if not change or change.lower() == "nan":
            continue
        out.append(
            {
                "record_id": int(row["record_id"]),
                "doi": str(row.get("source_doi") or ""),
                "side": side,
                "change": change,
                "sign": sign_of(change),
                "origin": str(row.get("origin") or table1_store.LLM_ORIGIN),
                "measured_as": str(row.get("measured_as") or ""),
                "context": str(row.get("study_context") or ""),
            }
        )
    return pd.DataFrame(out, columns=columns)


def observed_summary(table1: pd.DataFrame, canonical_id: int) -> tuple[Counter, str]:
    """
    What was recorded at one Key Event, counted rather than averaged.

    "decreased in 3, increased in 1" reads as the disagreement it is;
    collapsing it to a single arrow hides it.
    """
    claims = claims_for(table1, canonical_id)
    counted = Counter(str(c).lower() for c in claims["change"])
    if not counted:
        return counted, "no change recorded"
    return counted, ", ".join(f"{word} in {n}" for word, n in counted.most_common())


def flag_for(name: str, derived: str) -> Optional[str]:
    """
    Whether one Key Event is flagged, and which way.

    `±` is the claims disagreeing with each other. `⚠` is the *name* stating
    one direction while the claims report the other — "reduced myelination"
    whose papers all report myelination rising. One of the two is wrong, and
    trusting either one silently is how a sign error survives review.
    """
    if derived == CONFLICT:
        return CONFLICT
    stated = ke_normalizer.polarity(name)
    observed = {"↑": 1, "↓": -1}.get(derived, 0)
    if stated and observed and stated != observed:
        return NAME_MISMATCH
    return None


def find_all(graph) -> list[dict[str, Any]]:
    """
    Every flagged Key Event on a built graph, unresolved ones first.

    Takes the graph rather than the database so it reports on exactly what is
    drawn — a conflict on a Key Event that never reached the map is not
    something a reader of the map needs to act on.
    """
    rulings = table1_store.load_ke_directions()
    found: list[dict[str, Any]] = []

    for name, data in graph.nodes(data=True):
        derived = str(data.get("derived_change") or "")
        mark = flag_for(name, derived)
        if mark is None:
            continue
        canonical_id = int(data.get("canonical_id", 0))
        ruling = rulings.get(canonical_id)
        found.append(
            {
                "name": name,
                "canonical_id": canonical_id,
                "mark": mark,
                "observed": str(data.get("observed") or ""),
                "ruled": str(ruling.get("direction")) if ruling else "",
                "rationale": str(ruling.get("rationale") or "") if ruling else "",
            }
        )

    found.sort(key=lambda f: (bool(f["ruled"]), f["name"]))
    return found
