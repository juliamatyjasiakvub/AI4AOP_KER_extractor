from __future__ import annotations

"""
The form for entering or correcting a claim by hand.

One form, three call sites: under a paper's extraction results ("this run
missed something"), in curation ("this pathway is missing a step"), and opened
on an existing row ("this extraction is wrong"). They differ only in what is
filled in when it opens, which is why they are one function and not three.

The design constraint worth stating: the fastest path through this form has to
be the *common* case — a relationship between two events the corpus already
knows about, supported by a sentence in a paper already uploaded. That path is
two dropdowns, one paste and a save. Everything else is behind an expander,
because a form that asks thirty questions to record one relationship is a form
that gets abandoned halfway and leaves the pathway wrong anyway.
"""

from typing import Any, Optional

import streamlit as st

from schemas import (
    DIRECTION_VALUES,
    KE_LEVEL_ORDER,
    KER_ADJACENCY_VALUES,
    RELATION_KIND_VALUES,
)
from stage2_extraction import manual_entry, table1_store
from stage2_extraction import workflow_state as wf
from ui.common import curator_name, invalidate_pipeline

_NEW = "— new event —"

DIRECTION_HELP = {
    "positive": "both moved the same way",
    "negative": "one rose as the other fell",
    "none": "measured, and they did not move together",
    "unclear": "the source does not say",
}


def _event_picker(
    side: str,
    *,
    known: list[str],
    key_prefix: str,
    default_name: str = "",
    default_level: str = "",
) -> tuple[str, str]:
    """
    Name and level for one end of the relationship.

    The picklist is not a convenience. A hand-typed name that differs from the
    corpus by a word creates a second Key Event that looks like the first, and
    the curator pays for it later in the merge queue — so choosing an existing
    event is the default action and typing a new one is the deliberate one.
    """
    label = side.capitalize()
    options = [_NEW] + known
    index = options.index(default_name) if default_name in options else 0

    choice = st.selectbox(
        f"{label} Key Event",
        options,
        index=index,
        key=f"{key_prefix}_{side}_pick",
        help=(
            "Events already in the corpus are listed first. Picking one joins "
            "this claim to that event instead of creating a near-duplicate."
        ),
    )

    if choice == _NEW:
        name = st.text_input(
            f"{label} Key Event name",
            value=default_name if default_name not in known else "",
            key=f"{key_prefix}_{side}_name",
            placeholder="Decreased voltage-gated sodium-channel activity",
        ).strip()
    else:
        name = choice

    suggested = manual_entry.level_for_name(name) or default_level or ""
    levels = list(KE_LEVEL_ORDER)
    level_index = levels.index(suggested) if suggested in levels else 1
    level = st.selectbox(
        f"{label} biological level",
        levels,
        index=level_index,
        key=f"{key_prefix}_{side}_level",
        help=(
            "Already set by the corpus for this event."
            if suggested and name
            else "Decides which column the event is drawn in."
        ),
    )
    return name, level


def _source_picker(
    *,
    key_prefix: str,
    fixed_doi: str = "",
    fixed_filename: str = "",
    fixed_title: str = "",
) -> dict[str, str]:
    """
    Where the claim comes from, or an explicit statement that it has no paper.

    The third option is the one that earns its place. A developer asserting a
    step from the handbook or from unpublished work is doing something
    legitimate and different from quoting a paper, and a form that only accepts
    DOIs would push that assertion into a fake one.
    """
    if fixed_doi:
        st.caption(f"Source: **{fixed_title or fixed_filename or fixed_doi}**")
        return {
            "source_doi": fixed_doi,
            "source_filename": fixed_filename,
            "source_title": fixed_title,
        }

    papers = manual_entry.known_papers()
    labels = ["A paper already in the corpus", "A source not in the corpus",
              "No source — my own assessment"]
    kind = st.radio(
        "Where does this come from?",
        labels,
        key=f"{key_prefix}_source_kind",
    )

    if kind == labels[0]:
        if not papers:
            st.warning("No papers in the corpus yet. Extract one first, or "
                       "choose one of the other two options.")
            return {"source_doi": "", "source_filename": "", "source_title": ""}
        options = {
            f"{p['title'] or p['filename'] or p['doi']}": p for p in papers
        }
        pick = st.selectbox("Paper", list(options), key=f"{key_prefix}_paper")
        chosen = options[pick]
        return {
            "source_doi": chosen["doi"],
            "source_filename": chosen["filename"],
            "source_title": chosen["title"],
        }

    if kind == labels[1]:
        doi = st.text_input(
            "DOI or citation", key=f"{key_prefix}_ext_doi",
            placeholder="10.1016/j.example.2021.01.001",
            help="Recorded as the source. Quotations cannot be verified "
                 "against a paper the tool has not read.",
        ).strip()
        title = st.text_input(
            "Title", key=f"{key_prefix}_ext_title", placeholder="optional"
        ).strip()
        return {"source_doi": doi, "source_filename": "", "source_title": title}

    st.caption(
        "This will be stored as a curator assertion with no source, and drawn "
        "on the map as one."
    )
    return {
        "source_doi": manual_entry.NO_SOURCE_DOI,
        "source_filename": "",
        "source_title": "",
    }


def _quote_box(source: dict[str, str], *, key_prefix: str) -> list[str]:
    """Pasted quotations, checked against the stored text as they are typed."""
    doi = source.get("source_doi") or ""
    if doi == manual_entry.NO_SOURCE_DOI:
        return []

    quote = st.text_area(
        "Supporting quotation",
        key=f"{key_prefix}_quote",
        height=90,
        placeholder="Paste the sentence from the paper that supports this.",
        help=(
            "Checked against the stored text of the paper. A located quotation "
            "makes a hand-entered claim as traceable as an extracted one."
        ),
    ).strip()

    if not quote:
        return []

    located = manual_entry.verify_quote(
        quote,
        source_doi=doi or None,
        source_filename=source.get("source_filename") or None,
    )
    if not located["searched"]:
        st.caption(
            "⚪ The text of this paper is not stored, so the quotation cannot "
            "be checked. It will be saved as unverified — which records that "
            "it was not checked, not that it failed."
        )
    elif located["verified"]:
        where = located["section"] or "the paper"
        page = located["page_start"]
        st.success(
            f"✅ Found in {where}" + (f", p. {page}" if page else "") + "."
        )
    else:
        st.warning(
            f"⚠️ Not found in this paper's stored text "
            f"(best match {located['match_ratio']:.0%}). Check the wording, or "
            f"save it anyway — it will be flagged as unverified."
        )
    return [quote]


def render_form(
    *,
    key_prefix: str,
    record: Optional[dict[str, Any]] = None,
    fixed_doi: str = "",
    fixed_filename: str = "",
    fixed_title: str = "",
    on_saved=None,
) -> None:
    """
    Draw the add/edit form. `record` non-None switches it to editing that row.
    """
    editing = record is not None
    record = record or {}
    known = manual_entry.known_event_names()
    curator = curator_name()

    if editing and record.get("origin") == table1_store.LLM_ORIGIN:
        st.info(
            "This row came from the model. Saving a change marks it as "
            "curator-edited and keeps the original in the record history — "
            "the extraction is not overwritten silently."
        )

    col_up, col_down = st.columns(2)
    with col_up:
        up_name, up_level = _event_picker(
            "upstream", known=known, key_prefix=key_prefix,
            default_name=str(record.get("upstream_ke_name") or ""),
            default_level=str(record.get("upstream_ke_level") or ""),
        )
    with col_down:
        down_name, down_level = _event_picker(
            "downstream", known=known, key_prefix=key_prefix,
            default_name=str(record.get("downstream_ke_name") or ""),
            default_level=str(record.get("downstream_ke_level") or ""),
        )

    col_dir, col_kind, col_adj = st.columns(3)
    with col_dir:
        directions = list(DIRECTION_VALUES)
        current = str(record.get("direction") or "unclear")
        direction = st.selectbox(
            "Direction", directions,
            index=directions.index(current) if current in directions else 3,
            format_func=lambda d: f"{d} — {DIRECTION_HELP[d]}",
            key=f"{key_prefix}_direction",
        )
    with col_kind:
        kinds = list(RELATION_KIND_VALUES)
        current = str(record.get("relation_kind") or "causal")
        relation_kind = st.selectbox(
            "Kind of link", kinds,
            index=kinds.index(current) if current in kinds else 0,
            key=f"{key_prefix}_kind",
            help=(
                "A *marker* link says the downstream event is how the upstream "
                "one was measured. Drawn as a causal step it invents a terminal "
                "node that looks like an adverse outcome."
            ),
        )
    with col_adj:
        adjacencies = list(KER_ADJACENCY_VALUES)
        current = str(record.get("ker_adjacency") or "Adjacent")
        adjacency = st.selectbox(
            "Adjacency", adjacencies,
            index=adjacencies.index(current) if current in adjacencies else 0,
            key=f"{key_prefix}_adjacency",
        )

    if editing:
        source = {
            "source_doi": str(record.get("source_doi") or ""),
            "source_filename": str(record.get("source_filename") or ""),
            "source_title": str(record.get("source_title") or ""),
        }
        shown = (
            source["source_title"]
            or source["source_filename"]
            or source["source_doi"]
        )
        st.caption(
            f"Source: **{shown}** — not editable, because it says where the "
            f"claim came from."
        )
        quotes: list[str] = []
    else:
        source = _source_picker(
            key_prefix=key_prefix, fixed_doi=fixed_doi,
            fixed_filename=fixed_filename, fixed_title=fixed_title,
        )
        quotes = _quote_box(source, key_prefix=key_prefix)

    rationale = st.text_area(
        "Why does this belong in the pathway?" if not editing
        else "What are you correcting, and why?",
        value=str(record.get("entry_rationale") or ""),
        key=f"{key_prefix}_rationale",
        height=70,
        placeholder=(
            "Stated in the Figure 4 legend; the extraction only read the body "
            "text."
        ),
    ).strip()

    with st.expander("More detail — all optional"):
        description = st.text_area(
            "What the relationship says",
            value=str(record.get("ker_description") or ""),
            key=f"{key_prefix}_description", height=70,
        )
        c1, c2 = st.columns(2)
        with c1:
            up_change = st.text_input(
                "What happened upstream",
                value=str(record.get("upstream_change") or ""),
                key=f"{key_prefix}_up_change", placeholder="decreased",
            )
            up_cell = st.text_input(
                "Upstream cell type",
                value=str(record.get("upstream_cell_type") or ""),
                key=f"{key_prefix}_up_cell",
            )
            measured_as = st.text_input(
                "Measured how",
                value=str(record.get("measured_as") or ""),
                key=f"{key_prefix}_measured",
            )
        with c2:
            down_change = st.text_input(
                "What happened downstream",
                value=str(record.get("downstream_change") or ""),
                key=f"{key_prefix}_down_change", placeholder="reduced",
            )
            down_cell = st.text_input(
                "Downstream cell type",
                value=str(record.get("downstream_cell_type") or ""),
                key=f"{key_prefix}_down_cell",
            )
            evidence_types = list(manual_entry.MANUAL_EVIDENCE_TYPES)
            current = str(record.get("evidence_type") or "not_stated")
            evidence_type = st.selectbox(
                "How the link was established", evidence_types,
                index=(evidence_types.index(current)
                       if current in evidence_types else len(evidence_types) - 1),
                key=f"{key_prefix}_evidence_type",
            )
        study_context = st.text_input(
            "Study context",
            value=str(record.get("study_context") or ""),
            key=f"{key_prefix}_context",
            placeholder="developmental; rat; in vivo",
        )
        null_findings = st.text_area(
            "What was measured and did not change",
            value=str(record.get("null_findings") or ""),
            key=f"{key_prefix}_nulls", height=60,
        )
        contradicts = st.checkbox(
            "This claim contradicts the relationship",
            value=bool(record.get("contradicts_ker", False)),
            key=f"{key_prefix}_contradicts",
        )

    values = {
        "upstream_ke_name": up_name,
        "upstream_ke_level": up_level,
        "downstream_ke_name": down_name,
        "downstream_ke_level": down_level,
        "direction": direction,
        "relation_kind": relation_kind,
        "ker_adjacency": adjacency,
        "ker_description": description,
        "upstream_change": up_change,
        "downstream_change": down_change,
        "upstream_cell_type": up_cell,
        "downstream_cell_type": down_cell,
        "measured_as": measured_as,
        "evidence_type": evidence_type,
        "study_context": study_context,
        "null_findings": null_findings,
        "contradicts_ker": contradicts,
        "entry_rationale": rationale,
        **source,
    }

    problems = manual_entry.validate(values)
    if problems:
        st.caption("Still needed: " + " ".join(problems))

    save_label = "Save correction" if editing else "Add this claim"
    if st.button(save_label, type="primary", disabled=bool(problems),
                 key=f"{key_prefix}_save"):
        try:
            if editing:
                result = manual_entry.update_manual_claim(
                    int(record["record_id"]), values,
                    curator=curator, rationale=rationale,
                )
                if not result["changed"]:
                    st.info("Nothing changed.")
                    return
                _restate_dependents(int(record["record_id"]))
                st.success(
                    f"Updated {len(result['changed'])} field(s). The previous "
                    f"version is in the record history."
                )
            else:
                result = manual_entry.save_manual_claim(
                    values, curator=curator, quotes=quotes
                )
                verified = result["n_verified"]
                st.success(
                    f"Added as row {result['record_id']}."
                    + (f" Quotation located in the source."
                       if verified else "")
                    + " Re-run **Propose canonical Key Events** in Normalize "
                      "and curate to fold it into the pathway."
                )
        except manual_entry.ValidationError as exc:
            st.error(str(exc))
            return

        invalidate_pipeline()
        if on_saved is not None:
            on_saved()
        st.rerun()


def _restate_dependents(record_id: int) -> None:
    """
    Push anything derived from a changed row back for review.

    Editing a claim after its relationship was approved leaves an approval
    attached to something nobody approved. The workflow machinery already
    knows how to express that; this only has to tell it the row moved.
    """
    row = table1_store.load_record(record_id)
    if not row:
        return
    up, down = row.get("upstream_ke_canonical_id"), row.get("downstream_ke_canonical_id")
    if up is None or down is None:
        return
    try:
        wf.invalidate_for_ker(f"{int(up)}->{int(down)}",
                              reason="a source claim was edited")
    except Exception:
        # Never let bookkeeping lose the curator's edit — the edit is saved by
        # this point and a missing staleness flag is the smaller failure.
        pass


def add_claim_expander(
    *,
    key_prefix: str,
    fixed_doi: str = "",
    fixed_filename: str = "",
    fixed_title: str = "",
    label: str = "Add a claim this run missed",
    expanded: bool = False,
) -> None:
    """The form folded into an expander, for placing under existing results."""
    with st.expander(label, expanded=expanded):
        st.caption(
            "Stored as a Table 1 row like any extracted claim, marked as "
            "entered by you. It goes through the same normalization, approval "
            "and synthesis as everything else."
        )
        render_form(
            key_prefix=key_prefix,
            fixed_doi=fixed_doi,
            fixed_filename=fixed_filename,
            fixed_title=fixed_title,
        )
