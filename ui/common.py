from __future__ import annotations

"""
Shared UI furniture: the pipeline cache, formatting, and the section header.

`section_intro` exists because the redesign asks for a short description in
every section telling the user how to work there. Putting it in one function
means the wording is consistent and the workflow position is always drawn the
same way — the user should be able to tell at a glance which of the five steps
they are in and what has to be true before the next one opens.
"""

import getpass
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import streamlit as st

from legal import with_disclaimer
from stage2_extraction import citations
from stage2_extraction.pdf_reader import strip_control_chars


#: The linear workflow, in order. Used to draw the position indicator.
WORKFLOW_STEPS: tuple[str, ...] = (
    "Extract evidence",
    "Normalize and curate",
    "Approve",
    "Synthesize evidence",
    "Final AOP",
)


def section_intro(
    title: str,
    step: Optional[str],
    what: str,
    how: Sequence[str] = (),
    *,
    caution: Optional[str] = None,
) -> None:
    """
    Standard header: where you are, what this section is for, how to work it.

    `what` is one sentence on the section's job. `how` is the sequence of
    actions in order — deliberately imperative and numbered, because the
    complaint the redesign answers is that it was never clear what to do next.
    """
    st.header(title)

    if step in WORKFLOW_STEPS:
        position = WORKFLOW_STEPS.index(step)
        trail = " → ".join(
            f"**{name}**" if i == position else f"<span style='opacity:0.45'>{name}</span>"
            for i, name in enumerate(WORKFLOW_STEPS)
        )
        st.markdown(
            f"<div style='font-size:0.85em;margin-bottom:0.6em'>{trail}</div>",
            unsafe_allow_html=True,
        )

    st.caption(what)

    if how:
        st.markdown(
            "\n".join(f"{i}. {line}" for i, line in enumerate(how, start=1))
        )
    if caution:
        st.info(caution, icon="ℹ️")
    st.divider()


def section_heading(
    title: str,
    what: str,
    help_text: str = "",
    *,
    level: str = "subheader",
) -> None:
    """
    A heading, one line saying what the section does, and a hover explanation.

    Every section now has to carry all three. The complaint this answers is
    that headings named the machinery ("Provenance drawer", "Canonical
    groups") without ever saying what the machinery was for, so the only way
    to find out was to press something and see. `what` is the one-line answer
    and is always visible; `help_text` is the longer one, including how the
    numbers are produced, and hides behind the ? until it is wanted.
    """
    renderer = {"header": st.header, "subheader": st.subheader}.get(level, st.subheader)
    if help_text:
        renderer(title, help=help_text)
    else:
        renderer(title)
    if what:
        st.caption(what)


#: Table 1's reading order: (database column, displayed name).
#:
#: Table 1 was drawn with `SELECT *`, so its columns appeared in the order they
#: were added to the schema over ten migrations. That put `record_id`,
#: `source_doi`, `source_filename`, `source_title`, `extraction_date`, `run_id`,
#: `aop_id` and `aop_status` in front of the biology, and the four columns that
#: say what the claim actually *is* — which way each end moved, and the sign of
#: the link — at positions 41 to 44, off the right-hand edge of the frame.
#:
#: The consequence was not cosmetic. A reader who cannot see `upstream_change`
#: and `downstream_change` sees "Voltage-gated sodium channels → myelination"
#: three times over and has no way to tell that one paper blocked the channel,
#: one activated it, and the third measured expression: three different claims,
#: identical on screen. Every column below was already extracted and stored.
#:
#: The two ends lead, each immediately followed by what the paper did to it,
#: with the relationship's own sign between them — which is the order the claim
#: is read in, and the order a KER is defined in.
TABLE1_READING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("record_id", "Claim"),
    ("upstream_ke_name", "Upstream KE (as written)"),
    ("upstream_change", "Upstream change"),
    ("direction", "Link sign"),
    ("downstream_ke_name", "Downstream KE (as written)"),
    ("downstream_change", "Downstream change"),
    ("ker_name", "Key Event Relationship"),
    ("ker_adjacency", "Adjacency"),
    ("relation_kind", "Relation kind"),
    ("evidence_type", "Evidence type"),
    ("measured_as", "Measured as"),
    ("upstream_cell_type", "Upstream cell type"),
    ("downstream_cell_type", "Downstream cell type"),
    ("upstream_ke_level", "Upstream level"),
    ("downstream_ke_level", "Downstream level"),
    ("chemical_stressor", "Stressor"),
    ("study_design", "Study design"),
    ("taxonomic_applicability", "Species"),
    ("contradicts_ker", "Contradicts"),
    ("extraction_confidence", "Confidence"),
    ("claim_status", "Curation"),
    ("n_verified_spans", "Quotes verified"),
    ("n_evidence_spans", "Quotes"),
    ("source_doi", "DOI"),
)


def directed_claim(row: Any) -> str:
    """
    One claim as a sentence: what moved, which way, and what followed.

    Built from `upstream_change` / `downstream_change`, which are the fields
    that distinguish "we blocked the channel and myelination failed" from "we
    activated it and myelination failed". Both are `Voltage-gated sodium
    channels → myelination` and they are not the same claim.
    """
    def _side(name: Any, change: Any) -> str:
        label = fmt(name, "?")
        moved = fmt(change, "")
        return f"{moved} {label}".strip() if moved else label

    upstream = _side(row.get("upstream_ke_name"), row.get("upstream_change"))
    downstream = _side(row.get("downstream_ke_name"), row.get("downstream_change"))
    sign = str(row.get("direction") or "").strip().lower()
    arrow = {"positive": "→", "negative": "⊣", "none": "⇢", "unclear": "→?"}.get(
        sign, "→"
    )
    return f"{upstream} {arrow} {downstream}"


def table1_reading_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1 with the columns a person reads a KER in, in that order.

    Missing columns are skipped rather than filled, so a database that has not
    yet been migrated shows fewer columns instead of a wall of blanks.
    """
    if df is None or df.empty:
        return df

    out = pd.DataFrame(index=df.index)
    out["Claim (directed)"] = df.apply(directed_claim, axis=1)
    for column, display in TABLE1_READING_COLUMNS:
        if column in df.columns:
            out[display] = df[column]
    if "contradicts_ker" in df.columns:
        out["Contradicts"] = (
            df["contradicts_ker"].fillna(0).astype(bool).map({True: "yes", False: ""})
        )
    return out


def citation_keys(dois: Iterable[Any], *, refresh: bool = False) -> dict[str, str]:
    """
    DOI → "Sanchez et al., 2019a" for every paper in view.

    Cached in session state on top of the SQLite cache underneath, because
    Streamlit reruns the whole script on every widget interaction and even a
    cache hit costs a query per rerun.
    """
    wanted = sorted({str(d).strip().lower() for d in dois if str(d or "").strip()})
    if not wanted:
        return {}

    store: dict[str, str] = st.session_state.setdefault("_citation_keys", {})
    if refresh or any(d not in store for d in wanted):
        store.update(citations.citation_keys(wanted, refresh=refresh))
        st.session_state["_citation_keys"] = store
    return store


def cite(doi: Any, keys: Optional[dict[str, str]] = None) -> str:
    """One paper's display key. Falls back to the DOI when unresolved."""
    return citations.key_for(str(doi or ""), keys or st.session_state.get("_citation_keys", {}))


def count_chain() -> None:
    """
    The corpus counts, each with its unit named.

    These numbers get read as a sequence — 28 becomes 18 — and they are not
    the same kind of thing, so the sequence reads as a loss that never
    happened. A *claim* is one paper's statement that one event leads to
    another; a *Key Event* is one of the two ends. Twenty-eight claims naming
    eighteen distinct events is not eighteen surviving claims, and
    normalisation does not delete rows at all: it only decides which labels
    name the same event.

    Every step is labelled with what it counts, and the arrow between them is
    dropped deliberately — there is no arithmetic connecting these.
    """
    from stage2_extraction import table1_store

    counts = table1_store.corpus_counts()
    if not counts.get("claims"):
        return

    with st.expander("What these numbers count", expanded=False):
        st.caption(
            "Different units. They are not stages of a funnel and one does "
            "not reduce to the next."
        )
        rows = [
            ("Papers", counts["papers"],
             "Distinct source publications that yielded at least one row."),
            ("KER claims (Table 1 rows)", counts["claims"],
             "One paper's statement that one event leads to another. This is "
             "the unit Table 1 is in."),
            ("Key Event mentions", counts["label_occurrences"],
             f"Two per claim, because every claim names an upstream and a "
             f"downstream event — so {counts['claims']} claims give "
             f"{counts['label_occurrences']} mentions. Not a count of events."),
            ("Distinct Key Event labels", counts["distinct_labels"],
             "How many different wordings those mentions used."),
            ("Canonical Key Events", counts["canonical_kes"],
             "Labels judged to name the same event are one canonical event. "
             "This is what becomes a node on the map."),
            ("Relationships on the map", counts["relationships"],
             "Distinct event-to-event links, after several papers' claims "
             "about the same link are consolidated."),
        ]
        for label, value, explanation in rows:
            st.markdown(f"**{value:,} — {label}**")
            st.caption(explanation)

        merged = counts["distinct_labels"] - counts["canonical_kes"]
        if merged > 0:
            st.info(
                f"Normalisation merged {merged} label(s) into existing events.",
                icon="🔗",
            )
        elif counts["distinct_labels"]:
            st.info(
                "Normalisation merged nothing — every distinct label is still "
                "its own Key Event. That is normal when the papers used "
                "consistent wording, and it also means no clustering decision "
                "has been made for you.",
                icon="🔗",
            )


def locked(title: str, reason: str, remedy: str) -> None:
    """
    Draw a section that is not available yet.

    Says what is missing and where to fix it, rather than only refusing. A
    disabled button with no explanation reads as a bug.
    """
    st.header(title)
    st.warning(f"**{reason}**\n\n{remedy}", icon="🔒")


def curator_name() -> str:
    """
    Who is making decisions, taken from the machine rather than asked for.

    This used to be a sidebar box that had to be filled in before any curation
    button would work, on the theory that a curator column full of blanks is
    the same as having no attribution. That reasoning holds for a shared
    instance and not at all for this one, which one person runs on their own
    machine: the answer was the same every time, the box emptied itself on
    every restart, and sixteen buttons stayed disabled until it was retyped.
    The account name is already the answer, so it is read instead of asked.
    """
    try:
        return getpass.getuser().strip()
    except Exception:  # noqa: BLE001 - a nameless account is not an error
        return ""


def require_curator(message: str = "") -> bool:
    """
    Kept so call sites read the same, but no longer a gate.

    Nothing about a name makes a curation decision more or less sound, and
    blocking the work until one is typed only ever delayed the work.
    """
    return True


def fmt(value: Any, dash: str = "—") -> str:
    """Render a possibly-missing value for display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    text = str(value).strip()
    return text if text and text.lower() not in {"none", "nan"} else dash


def has_text(value: Any) -> bool:
    return fmt(value, "") != ""


def csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Disclaimer-stamped CSV, with control characters scrubbed.

    Rows extracted before control characters were stripped on write still carry
    them, and one NUL anywhere makes `to_csv` raise — which surfaces as a
    download button that does nothing.
    """
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].map(strip_control_chars)
    return with_disclaimer(cleaned.to_csv(index=False)).encode("utf-8")


def state_badge(label: str, drifted: bool = False) -> str:
    """A coloured pill for a workflow state."""
    colour = {
        "Raw": "#9aa0a6",
        "Normalization proposed": "#c98a00",
        "Curated": "#1a73e8",
        "Approved": "#1e8e3e",
        "Synthesized": "#6a1b9a",
    }.get(label, "#9aa0a6")
    if drifted:
        colour = "#c5221f"
        label = f"{label} · changed since approval"
    return (
        f"<span style='background:{colour};color:white;border-radius:10px;"
        f"padding:1px 8px;font-size:0.78em;white-space:nowrap'>{label}</span>"
    )


def relationship_badge(relationship: str, label: str) -> str:
    """A coloured pill for a semantic classification."""
    colour = {
        "equivalent": "#1e8e3e",
        "broader_than": "#1a73e8",
        "narrower_than": "#1a73e8",
        "related_but_distinct": "#c98a00",
        "contradictory_or_incompatible": "#c5221f",
        "uncertain": "#9aa0a6",
    }.get(relationship, "#9aa0a6")
    return (
        f"<span style='background:{colour};color:white;border-radius:10px;"
        f"padding:1px 8px;font-size:0.78em;white-space:nowrap'>{label}</span>"
    )


def quote_block(quote: str, citation: str, verified: bool) -> None:
    """One verbatim quotation with its citation and verification status."""
    badge = "✅ verified in source" if verified else "⚠️ not located verbatim"
    st.markdown(
        f"> {quote}  \n"
        f"<span style='opacity:0.65;font-size:0.85em'>{badge} · {citation}</span>",
        unsafe_allow_html=True,
    )


def invalidate_pipeline() -> None:
    st.session_state.pop("_pipeline", None)
