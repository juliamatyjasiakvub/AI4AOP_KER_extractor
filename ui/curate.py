from __future__ import annotations

"""
The single place where normalization, merging and curation happen.

Before the redesign these controls were spread across three tabs: a
normalization tab that proposed merges, an explore tab that let you rename
things, and a review tab with its own accept/reject buttons. Each wrote to the
same tables through a different route, so the answer to "what is this Key Event
and who decided that" depended on which screen you happened to have used.

There is now one workspace, laid out in three columns:

    left    candidate groups the tool has proposed, with checkboxes
    middle  the raw records in the selected group, exactly as extracted
    right   what the canonical result would be

and one set of actions, all of which write a `merge_decision` row. A curator
can merge as equivalent, keep separate, map to a broader ontology concept,
record a biological relationship, reject as not a Key Event, or mark
unresolved. Only the first of those changes what is on the map.
"""

from collections import Counter
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import streamlit as st

from stage2_extraction import (
    canonical_groups as cg,
    curation_store,
    ke_normalizer,
    ols4_client,
    table1_store,
    workflow_state as wf,
)
from schemas import KE_LEVEL_ORDER
from stage2_extraction.semantic_merge import (
    Classification,
    KERecord,
    Relationship,
    classify,
    classify_all,
    is_key_event,
    rank_candidates,
    summarise,
    worst,
)
from ui.common import (
    cite,
    citation_keys,
    count_chain,
    csv_bytes,
    curator_name,
    fmt,
    invalidate_pipeline,
    relationship_badge,
    require_curator,
    section_heading,
    section_intro,
    state_badge,
)
from ui import manual_claim


HOW_TO = (
    "Run **Propose canonical Key Events** to group the raw labels into a "
    "first draft.",
    "Read the **Crosswalk**. One row per wording, saying which Key Event it "
    "went to and which rule put it there. This is the whole of what happens "
    "between Table 1 and the Key Events, and the rows grouped on wording "
    "similarity alone are the ones to check.",
    "**Assign** is where most of the work happens, one row per KER claim. "
    "Confirm each claim against the paper and name the Key Event at each end "
    "— two ends given the same name become one event, the other wording kept "
    "as its synonym. Apply once.",
    "**Key Events** is the roster: one row per event, with the In-AOP tick.",
    "**Decide** is only for pairs you are unsure about. It shows the evidence "
    "behind each record and classifies the pair before allowing a merge, so "
    "it is slower than Assign and safer. **Only pairs classified Equivalent "
    "can be merged.** Skip it if nothing is doubtful.",
    "**Merge history** is a log, not a third way to merge. It holds the "
    "before-and-after of every merge, which is what makes undo possible.",
)


def render(*, ols4_enabled: bool = True, ols4_min_score: float = 0.45) -> None:
    section_intro(
        "Normalize and curate",
        "Normalize and curate",
        "Decide what each extracted Key Event actually is. This is the only "
        "place where records are merged, renamed, mapped or rejected.",
        HOW_TO,
        caution=(
            "Nothing here is applied automatically. The tool proposes and "
            "explains; you decide."
        ),
    )

    table1 = table1_store.load_table1_as_dataframe()
    if table1.empty:
        st.info("No extracted rows yet. Start in **Extract evidence**.")
        return

    _normalization_controls(table1, ols4_enabled, ols4_min_score)

    canonical = table1_store.load_canonical_kes()
    if canonical.empty:
        st.info(
            "No canonical Key Events yet. Run **Propose canonical Key Events** "
            "above to group the raw labels."
        )
        # Not a return. Adding the step the papers missed is *more* likely
        # before normalization has run, not less, and sending the curator away
        # to come back later is how the gap gets forgotten.
        st.divider()
        _fill_a_gap(canonical)
        return

    st.divider()
    _crosswalk_view(table1)

    st.divider()
    _claim_assignment_table(canonical, table1)

    st.divider()
    _event_roster(canonical, table1)

    st.divider()
    _workspace(canonical, table1)

    st.divider()
    _canonical_groups_view()

    st.divider()
    _fill_a_gap(canonical)

    st.divider()
    _decision_log()


# ---------------------------------------------------------------------------
# Filling in what the corpus did not supply
# ---------------------------------------------------------------------------

def _fill_a_gap(canonical: pd.DataFrame) -> None:
    """
    The two things curation can be missing that no merge decision can fix.

    Everything else on this page rearranges what the papers produced. These two
    add something they did not: a link between two events the corpus never
    connected, and an event the corpus never mentioned. Both are assertions and
    are stored as such — but refusing to hold them does not make the pathway
    more evidence-based, it just moves the assertion into a Word document where
    nothing can audit it.
    """
    section_heading(
        "Fill a gap the papers left",
        "For the step you know belongs in the pathway and no paper in this "
        "corpus states.",
        help_text=(
            "**A missing relationship** is stored as a Table 1 row like any "
            "extracted claim, marked as entered by you, and normalised, "
            "approved and synthesised along with everything else.\n\n"
            "**A missing Key Event** is the rarer case — an adverse outcome or "
            "initiating event you know the pathway reaches, where the papers "
            "gathered so far cover only the middle. It survives re-running "
            "normalization, which derived Key Events do not.\n\n"
            "Both are drawn on the final map as curator assertions rather than "
            "as evidence."
        ),
    )

    left, right = st.columns(2)

    with left:
        with st.expander("✚ A relationship the papers do not state"):
            manual_claim.render_form(key_prefix="curate_add_ker")

    with right:
        with st.expander("✚ A Key Event no paper named"):
            st.caption(
                "Use this only for an event with no relationship to record "
                "yet. If you know what it connects to, add the relationship "
                "instead — it carries more and produces the Key Event anyway."
            )
            name = st.text_input(
                "Key Event name", key="curate_new_ke_name",
                placeholder="Auditory hypersensitivity",
            ).strip()
            level = st.selectbox(
                "Biological level", list(KE_LEVEL_ORDER),
                index=len(KE_LEVEL_ORDER) - 2, key="curate_new_ke_level",
            )
            why = st.text_area(
                "Why does this belong in the pathway?",
                key="curate_new_ke_why", height=70,
            ).strip()

            clash = (
                not canonical.empty
                and name
                and (canonical["canonical_name"].astype(str).str.strip().str.casefold()
                     == name.casefold()).any()
            )
            if clash:
                st.warning(
                    f"**{name}** already exists as a canonical Key Event."
                )

            if st.button(
                "Add this Key Event", key="curate_new_ke_save",
                disabled=not (name and why) or bool(clash),
            ):
                table1_store.create_manual_canonical_ke(
                    name, level, curator=curator_name(), rationale=why,
                )
                invalidate_pipeline()
                st.success(
                    f"Added **{name}**. It has no evidence attached, so it "
                    f"will show as unreviewed until you approve it, and it "
                    f"will not appear on the map until something connects to "
                    f"it."
                )
                st.rerun()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalization_controls(
    table1: pd.DataFrame, ols4_enabled: bool, ols4_min_score: float
) -> None:
    """Build canonical Key Events from the raw labels, then classify pairs."""
    section_heading(
        "1 · Propose — a first draft of the grouping",
        "Clusters raw labels that look like the same event. A proposal only: "
        "nothing here is applied until you confirm it in Assign.",
        help_text=(
            "**How the grouping is made.** Labels are compared on their "
            "wording and, where ontology enrichment is on, on the ontology "
            "term each was matched to. Labels scoring above the threshold are "
            "put in the same draft group. No model is asked to judge them — "
            "this is string and ontology similarity, which is why it is a "
            "draft rather than a decision.\n\n"
            "**Clustering threshold** decides only what you are *shown*. "
            "Raise it and you get fewer, tighter groups and more work in "
            "Assign; lower it and you get more speculative groups to reject. "
            "It has no say in what may be merged.\n\n"
            "**Classify candidate merges** is the second, slower pass: it "
            "asks whether each proposed pair is genuinely equivalent or only "
            "related, and is what the Decide step reads."
        ),
    )

    raw_labels = ke_normalizer.collect_raw_kes(table1)
    unique_labels = sorted({label for label, _, _ in raw_labels})
    left, right = st.columns([2, 3])

    with left:
        # "Raw Key Event labels: 56" was counting label *mentions* — two per
        # Table 1 row — so 28 extracted claims reported 56 labels, and the 18
        # canonical events that came out of them looked like a heavy loss.
        # They are three different units and only one of them is labels.
        st.metric(
            "Distinct Key Event labels", len(unique_labels),
            help=(
                f"Different wordings the papers used, counted once each. "
                f"They came from {len(raw_labels)} mentions across the "
                f"extracted claims — every claim names two events, so "
                f"mentions always outnumber labels."
            ),
        )
        threshold = st.slider(
            "Clustering threshold", 0.60, 0.95, 0.86, step=0.01,
            help=(
                "Higher is stricter. This only decides which groups you are "
                "shown; it has no say in whether a group may be merged."
            ),
            key="curate_threshold",
        )

    with right:
        st.write("")
        if st.button("Propose canonical Key Events", type="primary",
                     use_container_width=True, key="curate_normalize"):
            with st.spinner("Clustering labels…"):
                report = ke_normalizer.normalize_table1(
                    table1,
                    threshold=float(threshold),
                    ols4_enabled=bool(ols4_enabled),
                    ols4_min_score=float(ols4_min_score),
                )
            st.session_state["_norm_report"] = report
            st.session_state.pop("_classifications", None)
            invalidate_pipeline()
            for row in table1_store.load_canonical_kes().itertuples():
                if wf.get_state("ke", row.canonical_id) is wf.State.RAW:
                    wf.set_state("ke", row.canonical_id,
                                 wf.State.NORMALIZATION_PROPOSED)
            merged = max(0, report.n_raw - report.n_canonical)
            st.success(
                f"{report.n_canonical} canonical Key Events proposed from "
                f"{report.n_raw} raw labels — "
                + (
                    f"{merged} label(s) grouped into an existing event."
                    if merged
                    else "no labels were grouped; every wording stands as its "
                         "own event."
                )
                + " No Table 1 rows were changed: this decides which labels "
                "name the same event, not which claims survive."
            )
            st.rerun()

        if st.button("Classify candidate merges", use_container_width=True,
                     key="curate_classify"):
            _run_classification(ols4_enabled)
            st.rerun()

    # Before clustering, not after it. The raw labels are already in Table 1
    # the moment extraction finishes, and refusing to show them until Propose
    # has run made the first screen of curation an empty table with an
    # instruction on it — the corpus you are about to make decisions about was
    # the one thing you could not look at.
    _raw_label_view(table1, unique_labels)
    count_chain()

    report = st.session_state.get("_norm_report")
    if report is not None:
        st.caption(
            f"Last run: {report.n_raw} raw labels → {report.n_canonical} "
            f"canonical Key Events."
        )
        if getattr(report, "ontology_error", None):
            st.warning(
                f"Ontology lookup problem — clustering used strings alone. "
                f"{report.ontology_error}"
            )
        _cell_type_conflicts(getattr(report, "cell_type_conflicts", None))
        _context_conflicts(table1)
        _name_proposals(table1)

    classifications = st.session_state.get("_classifications")
    if classifications:
        counts = summarise(classifications)
        cols = st.columns(6)
        for col, (key, label) in zip(
            cols,
            [
                ("equivalent", "Equivalent"),
                ("broader_than", "Broader"),
                ("narrower_than", "Narrower"),
                ("related_but_distinct", "Related"),
                ("contradictory_or_incompatible", "Contradictory"),
                ("uncertain", "Uncertain"),
            ],
        ):
            col.metric(label, counts.get(key, 0))
        st.caption(
            "Only the Equivalent group can be merged. The others are offered "
            "different actions."
        )


def _label_evidence(table1: pd.DataFrame, labels: Iterable[str]) -> dict[str, dict]:
    """
    What the papers actually reported for each raw label.

    A Key Event label names a quantity and stops there. "Voltage-gated sodium
    channels" does not say whether the papers found more of them or fewer, in
    what cell, or measured by what — and those are exactly the questions that
    decide whether two labels are the same event. All of it was extracted and
    is sitting in Table 1 unread: `upstream_change`/`downstream_change`,
    `measured_as`, and the cell type columns.

    Directions are counted, never averaged. A label four papers report rising
    and two report falling is a finding about the corpus, and collapsing it to
    one arrow would hide the disagreement at the exact moment a curator is
    deciding whether to merge it into something.
    """
    out: dict[str, dict] = {}
    if table1 is None or table1.empty:
        return {label: {} for label in labels}

    for label in labels:
        changes: Counter = Counter()
        cells: list[str] = []
        assays: list[str] = []
        levels: Counter = Counter()

        for side in ("upstream", "downstream"):
            rows = table1[table1[f"{side}_ke_name"] == label]
            if rows.empty:
                continue
            for _, row in rows.iterrows():
                change = fmt(row.get(f"{side}_change"), "")
                if change:
                    changes[change.strip().lower()] += 1
                cell = fmt(row.get(f"{side}_cell_type"), "")
                if cell:
                    cells.append(cell.strip())
                assay = fmt(row.get("measured_as"), "")
                if assay:
                    assays.append(assay.strip())
                level = fmt(row.get(f"{side}_ke_level"), "")
                if level:
                    levels[level] += 1

        matched = table1[
            (table1["upstream_ke_name"] == label)
            | (table1["downstream_ke_name"] == label)
        ]
        out[label] = {
            "reported": (
                ", ".join(f"{word} ×{n}" for word, n in changes.most_common(3))
                or "not stated"
            ),
            "measured_as": "; ".join(sorted(set(assays))[:3]) or "—",
            "cells": "; ".join(sorted(set(cells))[:3]) or "—",
            "level": levels.most_common(1)[0][0] if levels else "—",
            "claims": len(matched),
            "papers": int(matched["source_doi"].nunique()) if not matched.empty else 0,
        }
    return out


def _raw_label_view(table1: pd.DataFrame, labels: Sequence[str]) -> None:
    """Every wording the papers used, readable before anything is clustered."""
    if not labels:
        return

    evidence = _label_evidence(table1, labels)
    frame = pd.DataFrame(
        [
            {
                "Raw label": label,
                "Level": evidence[label].get("level", "—"),
                "Reported change": evidence[label].get("reported", "not stated"),
                "Assay (whole claim)": evidence[label].get("measured_as", "—"),
                "Cell type": evidence[label].get("cells", "—"),
                "Claims": evidence[label].get("claims", 0),
                "Papers": evidence[label].get("papers", 0),
            }
            for label in labels
        ]
    ).sort_values(["Papers", "Claims"], ascending=False)

    with st.expander(
        f"Raw Key Event labels as the papers wrote them ({len(labels)})",
        expanded=False,
    ):
        st.caption(
            "Straight from Table 1, before any clustering. **Reported change** "
            "counts how many extracted claims described the label going each "
            "way — it is not averaged, so a label that four papers report "
            "rising and two report falling shows both. **Assay (whole claim)** "
            "is recorded once per relationship, not once per event, so the "
            "assay beside a label may be the measurement at the OTHER end of "
            "the link — open the claim before reading it as how this event "
            "was measured."
        )
        st.dataframe(frame, use_container_width=True, hide_index=True, height=340)
        st.caption(
            "A label with no reported change is one the papers named but never "
            "said which way it went. That is worth knowing before you merge it "
            "into anything."
        )


def _cell_type_conflicts(conflicts: Optional[list]) -> None:
    """
    Warn where one label covers events observed in different cell types.

    Deliberately loud. Every other problem on this page is visible in the
    names — two spellings of one event look like two spellings of one event.
    This one is invisible by construction: the strings are identical, so the
    node looks well supported precisely because two unrelated literatures have
    been pooled into it. Splitting them is a judgement, so the tool says where
    to look and leaves the decision in the table below.
    """
    if not conflicts:
        return

    st.error(
        f"{len(conflicts)} Key Event label(s) cover more than one cell "
        "lineage. Until they are split, evidence from different cells is "
        "pooled on one node — which is how two mechanisms become one pathway.",
        icon="🧬",
    )

    canonical = table1_store.load_canonical_kes()
    ids_by_name = {str(r["canonical_name"]): int(r["canonical_id"])
                   for _, r in canonical.iterrows()}

    for conflict in conflicts[:10]:
        label = conflict["label"]
        with st.container(border=True):
            st.markdown(f"**{label}** — {conflict['n_rows']} row(s)")
            for name in conflict.get("lineages", []):
                examples = "; ".join(conflict.get("examples", {}).get(name, []))
                st.markdown(f"- *{name}* — {examples}")

            canonical_id = ids_by_name.get(label)
            if canonical_id is None:
                st.caption("Run **Propose** first so this label has a record to split.")
                continue

            if st.button(
                f"Split into {len(conflict['lineages'])} Key Events",
                key=f"split_lineage_{canonical_id}",
                type="primary",
                help=(
                    "Creates one Key Event per lineage and repoints each row "
                    "at the one matching its own cell type. Rows that never "
                    "stated a cell type stay where they are."
                ),
            ):
                try:
                    result = cg.split_by_cell_lineage(
                        canonical_id,
                        curator=curator_name(),
                        rationale=(
                            "Same molecule in different cells is not the same "
                            "Key Event."
                        ),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    _after_decision(
                        f"Split into {len(result['lineages'])}: "
                        + ", ".join(result["lineages"])
                        + f". {result['rows_moved']} row(s) repointed"
                        + (
                            f", {result['rows_left_unassigned']} left on the "
                            f"original because no cell type was stated."
                            if result["rows_left_unassigned"] else "."
                        )
                    )


def _context_conflicts(table1: pd.DataFrame) -> None:
    """
    Key Events built from incompatible study models.

    Unlike a cell-lineage clash there is no automatic split here, because the
    right answer is usually a judgement: an injury model and a developmental
    one may describe the same event or two different ones, and only a reader
    of both papers can say which.
    """
    conflicts = ke_normalizer.find_context_conflicts(table1)
    if not conflicts:
        return

    st.warning(
        f"{len(conflicts)} Key Event(s) pool evidence from different study "
        "models. Findings from an injury model and from normal development "
        "can describe different biology under the same name.",
        icon="🧪",
    )
    for conflict in conflicts[:8]:
        st.markdown(
            f"- **{conflict['label']}** — {conflict['n_rows']} row(s): "
            + " · ".join(conflict["contexts"])
        )


def _name_proposals(table1: pd.DataFrame) -> None:
    """Offer the precise name each Key Event's own rows already support."""
    proposals = ke_normalizer.propose_specific_names(table1)
    if not proposals:
        return

    with st.expander(f"Suggested precise names ({len(proposals)})"):
        st.caption(
            "A targeted run names every event after your question, which is "
            "what lets papers join up — and leaves the map saying "
            "\"sodium channel activity\" when the evidence is about one "
            "isoform in one cell type. These names are assembled from the "
            "direction, isoform and cell lineage already on the rows. "
            "Nothing is renamed unless you say so."
        )
        canonical = table1_store.load_canonical_kes()
        ids = {str(r["canonical_name"]): int(r["canonical_id"])
               for _, r in canonical.iterrows()}

        for i, proposal in enumerate(proposals[:12]):
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"**{proposal['label']}**  \n→ *{proposal['suggested']}*"
            )
            canonical_id = ids.get(proposal["label"])
            if canonical_id is None:
                continue
            if c2.button("Rename", key=f"rename_{canonical_id}_{i}",
                         use_container_width=True):
                table1_store.rename_canonical_ke(
                    canonical_id, proposal["suggested"]
                )
                affected = wf.invalidate_for_ke(
                    canonical_id, reason="renamed from its own evidence"
                )
                _after_decision(
                    f"Renamed. {affected} synthesis(es) marked stale."
                    if affected else "Renamed."
                )


def _run_classification(ols4_enabled: bool) -> None:
    """Classify every near-miss pair and cache the verdicts."""
    canonical = table1_store.load_canonical_kes()
    if canonical.empty:
        st.warning("Propose canonical Key Events first.")
        return

    records = [KERecord.from_row(row) for _, row in canonical.iterrows()]
    ancestors_of = ols4_client.ancestor_lookup(enabled=bool(ols4_enabled))

    with st.spinner("Checking each candidate pair…"):
        results = classify_all(records, ancestors_of=ancestors_of)

    already = cg.decided_pairs()

    def undecided(c: Classification) -> bool:
        try:
            return frozenset({int(c.source), int(c.target)}) not in already
        except (TypeError, ValueError):
            return True

    results = [c for c in results if undecided(c)]
    st.session_state["_classifications"] = rank_candidates(results)
    st.session_state["_ke_records"] = {r.key: r for r in records}


# ---------------------------------------------------------------------------
# The crosswalk: how Table 1 became the canonical Key Events
# ---------------------------------------------------------------------------

def _crosswalk_view(table1: pd.DataFrame) -> None:
    """
    One row per raw label: where it went, and on whose authority.

    This section exists because the step it documents was invisible. Table 1 is
    one row per claim in the paper's own words; the canonical Key Events are the
    nodes those claims resolve to. Between them the tool made one decision per
    label, and the only account of it on screen was a pair of totals — "18
    canonical Key Events proposed from 31 raw labels" — which does not say which
    labels moved, where they went, or why. Read as a sequence, "27 claims → 18
    Key Events" looks like nine findings were thrown away. Nothing was: the two
    numbers count different things, and the arithmetic that connects them is
    below.
    """
    crosswalk = table1_store.load_alias_crosswalk()

    section_heading(
        "1b · Crosswalk — which wording became which Key Event",
        "Every label the papers used, the Key Event it now belongs to, and the "
        "rule that put it there. Read this before Assign.",
        help_text=(
            "**Why grouped** names the rule, in the order the rules are "
            "applied and in descending order of authority:\n\n"
            "1. *Same AOP-Wiki Key Event id* — both extractions carried the "
            "same identifier. The strongest basis there is.\n"
            "2. *Same ontology term* — both labels resolved to one OLS4 CURIE "
            "at a match score of 0.75 or better.\n"
            "3. *Identical after normalisation* — the same string once case, "
            "plurals, punctuation and direction wording are regularised.\n"
            "4. *Same content words, different order* — an anagram of content "
            "words, at the same biological level.\n"
            "5. *Lexical similarity above threshold* — the weakest basis, and "
            "the only one the clustering slider affects. These are the rows to "
            "read first.\n\n"
            "A polarity guard applies throughout: two labels that disagree on "
            "direction cannot be grouped by ontology term or by similarity, so "
            "“increased apoptosis” and “decreased apoptosis” stay apart even "
            "when everything else about them matches.\n\n"
            "**Role** distinguishes the wording the event is named after from "
            "the wordings folded into it as synonyms. Nothing is deleted: every "
            "synonym stays attached to its event and every claim keeps its own "
            "words in Table 1."
        ),
    )

    if crosswalk.empty:
        st.info(
            "No labels are grouped yet. Run **Propose canonical Key Events** "
            "above."
        )
        return

    # The arithmetic, written out. Each step names its unit, because the whole
    # confusion this section answers comes from reading counts of different
    # things as one shrinking number.
    n_claims = int(len(table1))
    n_mentions = n_claims * 2
    n_labels = int(crosswalk["raw_label"].nunique())
    n_events = int(crosswalk["canonical_name"].nunique())
    n_folded = n_labels - n_events

    st.markdown(
        f"**{n_claims} KER claims** × 2 ends = **{n_mentions} Key Event "
        f"mentions**, written in **{n_labels} different wordings**, grouped "
        f"into **{n_events} canonical Key Events**."
    )
    st.caption(
        f"{n_folded} wording(s) were folded into an event named by another "
        f"wording. No claim was dropped and no Table 1 row was edited — "
        f"Table 1 still has {n_claims} rows. Grouping decides which labels "
        f"name the same event; it never decides which claims survive."
    )

    evidence = _label_evidence(table1, crosswalk["raw_label"].tolist())

    basis_labels = getattr(table1_store, "ALIAS_BASIS_LABELS", {})
    frame = pd.DataFrame(
        [
            {
                "Raw label (as written)": row["raw_label"],
                "Role": "names the event" if row["is_event_name"] else "synonym",
                "→ Key Event": row["canonical_name"],
                "Level": row.get("level") or "—",
                "Why grouped": basis_labels.get(
                    str(row.get("merge_basis") or ""),
                    "Recorded before this was tracked — re-run Propose to fill it in",
                ),
                "On what evidence": fmt(row.get("merge_detail"), "—"),
                "Reported change": evidence.get(row["raw_label"], {}).get(
                    "reported", "not stated"
                ),
                "Claims": evidence.get(row["raw_label"], {}).get("claims", 0),
                "Papers": evidence.get(row["raw_label"], {}).get("papers", 0),
            }
            for _, row in crosswalk.iterrows()
        ]
    )

    # Weak groupings first. A curator has finite attention and the rows that
    # deserve it are the ones held together by nothing but string similarity.
    _priority = {
        "Lexical similarity above threshold": 0,
        "Same content words, different order": 1,
        "Same ontology term": 2,
        "Identical after normalisation": 3,
        "Same AOP-Wiki Key Event id": 4,
        "Curator assigned it": 5,
    }
    frame["_order"] = frame["Why grouped"].map(lambda w: _priority.get(w, 6))
    frame = frame.sort_values(
        ["_order", "→ Key Event", "Role"], ascending=[True, True, True]
    ).drop(columns=["_order"])

    st.dataframe(frame, use_container_width=True, hide_index=True, height=360)

    weakest = int((frame["Why grouped"] == "Lexical similarity above threshold").sum())
    if weakest:
        st.warning(
            f"{weakest} label(s) were grouped on wording similarity alone — no "
            f"shared identifier and no shared ontology term. Those are the "
            f"groupings most likely to be wrong, and the clustering threshold "
            f"is the only thing holding them together.",
            icon="⚠️",
        )

    st.download_button(
        "Download crosswalk CSV",
        csv_bytes(frame),
        "table1_to_key_event_crosswalk.csv",
        "text/csv",
        key="curate_crosswalk_csv",
    )

    # Same information the other way round, for the reader who wants to see one
    # event and everything that folded into it rather than one label and where
    # it went.
    folded = (
        crosswalk.groupby("canonical_name")["raw_label"].count().sort_values(
            ascending=False
        )
    )
    multi = [name for name, count in folded.items() if count > 1]
    if multi:
        with st.expander(
            f"Grouped by Key Event — {len(multi)} event(s) hold more than one wording",
            expanded=False,
        ):
            for name in multi:
                members = crosswalk[crosswalk["canonical_name"] == name]
                st.markdown(f"**{name}**")
                for _, row in members.iterrows():
                    mark = "⭐" if row["is_event_name"] else "↳"
                    st.caption(
                        f"{mark} {row['raw_label']} — "
                        f"{basis_labels.get(str(row.get('merge_basis') or ''), 'basis not recorded')}"
                        f". {fmt(row.get('merge_detail'), '')}"
                    )


# ---------------------------------------------------------------------------
# Assign: one row per claim
# ---------------------------------------------------------------------------

def _claim_assignment_table(canonical: pd.DataFrame, table1: pd.DataFrame) -> None:
    """
    One row per KER claim, with both ends nameable and the paper's words kept.

    The previous version of this step was one row per raw *label*. That is the
    wrong unit and it was the source of the complaint that this screen made no
    sense. A label is half a claim: "Voltage-gated sodium channels" appearing on
    twelve rows says nothing about what those twelve papers did to the channel or
    what followed, and one paper can perturb the same channel three ways and get
    three different downstream events. Rolling that up to "12 claims, 11 papers"
    is exactly the information a curator needs and does not have.

    So the grid is one row per claim, in the paper's own words, with the
    upstream and downstream ends named independently. Grouping still happens by
    naming: give two ends the same Key Event and they become one event. The
    difference is that the curator is now looking at the claim while deciding.
    """
    section_heading(
        "2 · Assign — confirm each claim and name both of its ends",
        "One row per KER claim, exactly as Table 1 holds it. Name the Key Event "
        "at each end, tick the claims you have checked, untick the ones that do "
        "not belong.",
        help_text=(
            "**Each row is one relationship one paper reported** — the same "
            "unit as Table 1, so the two screens line up row for row and the "
            "claim number matches.\n\n"
            "**Upstream KE** and **Downstream KE** are editable dropdowns. Give "
            "two ends the same name and they become one Key Event, both "
            "wordings kept as synonyms. Pick a label's own wording to stand it "
            "up as its own event. The list is a dropdown rather than free text "
            "because a name typed with one character different would silently "
            "create a second Key Event nobody meant — to introduce a genuinely "
            "new name, use **Rename a Key Event** below.\n\n"
            "**As written** columns and **change** columns are never editable. "
            "They are the paper's words, and this screen has no business "
            "altering them; correct an extraction in step 1 instead.\n\n"
            "**Checked** means you have read this row against the paper. It is "
            "a record of your attention, not a judgement about the biology.\n\n"
            "**Keep** unticked removes the claim from the synthesis and the "
            "map. The row stays in Table 1 with its quotations — nothing is "
            "deleted — but it stops contributing evidence.\n\n"
            "**One wording can be two Key Events, and this is where you say "
            "so.** The unit of assignment is the row, not the wording. Eleven "
            "claims say “voltage-gated sodium channels”; if one blocked the "
            "channel in an oligodendrocyte and another activated it in an axon, "
            "send those two rows to two different events and each row keeps its "
            "own. The wording stays attached to both as a synonym, because both "
            "papers wrote it. A note appears when this happens, so it cannot be "
            "done by accident without being told."
        ),
    )

    aliases = table1_store.load_alias_map()
    if not aliases:
        st.info("No raw labels are assigned yet. Run **Propose** above first.")
        return

    names = {
        int(r["canonical_id"]): str(r["canonical_name"])
        for _, r in canonical.iterrows()
    }

    def event_for_end(claim: pd.Series, side: str) -> str:
        """
        The Key Event *this row's* end points at.

        Read from the row's own `{side}_ke_canonical_id` before the alias map,
        and that order matters. The alias map is one entry per wording, so for a
        wording that has been split by cell type it can only hold one of the
        events and reading it first would show every one of the eleven
        sodium-channel claims under whichever event happened to be written last
        — quietly discarding the split and then, on Apply, destroying it.
        """
        canonical_id = claim.get(f"{side}_ke_canonical_id")
        if pd.notna(canonical_id) and int(canonical_id) in names:
            return names[int(canonical_id)]
        text = str(claim.get(f"{side}_ke_name") or "").strip()
        fallback = aliases.get(text)
        return names.get(fallback, text) if fallback is not None else text

    keys = citation_keys(table1["source_doi"].tolist())
    claim_state = curation_store.curation_map("claim")

    def state_for(record_id: Any) -> tuple[bool, bool]:
        """(keep, checked) for one claim, defaulting to kept and unchecked."""
        status = str(
            (claim_state.get(str(int(record_id))) or {}).get("status") or "unreviewed"
        )
        if status == "rejected":
            return False, False
        return True, status == "accepted"

    rows = []
    for _, claim in table1.iterrows():
        keep, checked = state_for(claim["record_id"])
        rows.append(
            {
                "Claim": int(claim["record_id"]),
                "Cite": cite(claim.get("source_doi"), keys),
                "Upstream, as written": fmt(claim.get("upstream_ke_name"), ""),
                "Upstream change": fmt(claim.get("upstream_change"), "not stated"),
                "Upstream KE": event_for_end(claim, "upstream"),
                "Sign": fmt(claim.get("direction"), "—"),
                "Downstream, as written": fmt(claim.get("downstream_ke_name"), ""),
                "Downstream change": fmt(claim.get("downstream_change"), "not stated"),
                "Downstream KE": event_for_end(claim, "downstream"),
                "Relationship, as written": fmt(claim.get("ker_name"), "—"),
                "Checked": checked,
                "Keep": keep,
            }
        )

    current = pd.DataFrame(rows)
    if current.empty:
        st.info("No claims to assign.")
        return

    # Every existing event name and every raw label, so a label folded into
    # another event can always be pulled back out under its own wording.
    options = sorted({*names.values(), *aliases.keys()})

    st.caption(
        "\U0001f4a1 Double-click an **Upstream KE** or **Downstream KE** cell to "
        "change it · the **as written** and **change** columns are the paper's "
        "words and stay fixed · nothing is saved until you press Apply."
    )

    _fixed = {
        "Claim": ("Claim", "small", "The Table 1 row this is. Same number in step 1."),
        "Cite": ("Cite", "small", "First author and year."),
        "Upstream, as written": (
            "Upstream, as written", "large",
            "The upstream event in the paper's own words. Never edited here.",
        ),
        "Upstream change": (
            "Upstream change", "medium",
            "What this paper did to the upstream event, or found it doing. "
            "This is what separates two claims that name the same pair of "
            "events — blocking a channel and activating it are not one claim.",
        ),
        "Sign": (
            "Sign", "small",
            "How the two ends moved together in this experiment: positive, "
            "negative, none or unclear. A property of the relationship, not of "
            "either event.",
        ),
        "Downstream, as written": (
            "Downstream, as written", "large",
            "The downstream event in the paper's own words. Never edited here.",
        ),
        "Downstream change": (
            "Downstream change", "medium",
            "What was observed at the downstream event.",
        ),
        "Relationship, as written": (
            "Relationship, as written", "large",
            "The relationship as the extraction phrased it, kept verbatim.",
        ),
    }

    column_config: dict[str, Any] = {
        name: st.column_config.TextColumn(
            title, disabled=True, width=width, help=help_text
        )
        for name, (title, width, help_text) in _fixed.items()
    }
    column_config["Claim"] = st.column_config.NumberColumn(
        "Claim", disabled=True, width="small",
        help="The Table 1 row this is. Same number in step 1.",
    )
    column_config["Upstream KE"] = st.column_config.SelectboxColumn(
        "Upstream KE", options=options, required=True, width="large",
        help=(
            "The canonical Key Event this end belongs to. Two ends given the "
            "same name become one event."
        ),
    )
    column_config["Downstream KE"] = st.column_config.SelectboxColumn(
        "Downstream KE", options=options, required=True, width="large",
        help=(
            "The canonical Key Event this end belongs to. Two ends given the "
            "same name become one event."
        ),
    )
    column_config["Checked"] = st.column_config.CheckboxColumn(
        "Checked",
        help="You have read this row against the paper.",
    )
    column_config["Keep"] = st.column_config.CheckboxColumn(
        "Keep",
        help=(
            "Unticked claims stay in Table 1 with their quotations but stop "
            "contributing to the synthesis and the map."
        ),
    )

    edited = st.data_editor(
        current,
        use_container_width=True,
        hide_index=True,
        height=420,
        key="curate_claim_editor",
        column_order=[
            "Claim", "Cite",
            "Upstream, as written", "Upstream change", "Upstream KE",
            "Sign",
            "Downstream, as written", "Downstream change", "Downstream KE",
            "Relationship, as written",
            "Checked", "Keep",
        ],
        column_config=column_config,
    )

    # --- What the edits imply ---------------------------------------------
    #
    # Two things are being decided at once and they are stored in two places,
    # which is why this is not a single dict.
    #
    #   1. Which Key Events exist and which wordings name them. That is the
    #      alias level, and `apply_assignments` owns it.
    #   2. Which Key Event each individual claim's end points at. That is the
    #      row level, stored on the Table 1 row, and no label-keyed structure
    #      can express it.
    #
    # The second is what this grid is for. A wording sent to two different
    # events on two different rows is not an error to refuse — it is the
    # sodium-channel case, one wording covering a channel blocked in an
    # oligodendrocyte and activated in an axon — so both events are created and
    # each row is then pointed at its own.
    pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    per_claim: dict[int, dict[str, str]] = {}
    sent_to: dict[str, set[str]] = {}

    def add_pair(label: str, event: str) -> None:
        if not label or not event or (label, event) in seen_pairs:
            return
        seen_pairs.add((label, event))
        pairs.append((label, event))

    for _, row in edited.iterrows():
        record_id = int(row["Claim"])
        for written, target, side in (
            ("Upstream, as written", "Upstream KE", "upstream"),
            ("Downstream, as written", "Downstream KE", "downstream"),
        ):
            label = str(row[written]).strip()
            event = str(row[target]).strip()
            if not label or not event:
                continue
            add_pair(label, event)
            sent_to.setdefault(label, set()).add(event)
            per_claim.setdefault(record_id, {})[side] = event

    # Wordings on no displayed row keep the event they already have. Omitting
    # them would not merely lose the mapping: `apply_assignments` rebuilds the
    # Key Events from these pairs alone, so an event named by no pair is deleted.
    #
    # Only wordings the grid did not speak for. Carrying over the stored event
    # of a wording the curator has just moved would re-create the event they
    # moved it out of, with the wording attached to both.
    crosswalk = table1_store.load_alias_crosswalk()
    if not crosswalk.empty:
        for _, row in crosswalk.iterrows():
            label = str(row["raw_label"])
            if label not in sent_to:
                add_pair(label, str(row["canonical_name"]))

    split_labels = {
        label: events for label, events in sent_to.items() if len(events) > 1
    }

    n_events = len({event for _, event in pairs})
    n_checked = int(edited["Checked"].sum())
    n_dropped = int((~edited["Keep"]).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims", len(edited))
    c2.metric("Key Events", n_events,
              help="Distinct event names across both ends of every claim.")
    c3.metric("Checked", n_checked)
    c4.metric("Not kept", n_dropped)
    st.caption(
        f"{len(sent_to)} wording(s) across {n_events} Key Event(s). Claims are "
        f"never merged: {len(edited)} claims in, "
        f"{len(edited) - n_dropped} contributing."
    )

    if split_labels:
        st.info(
            "**One wording, more than one Key Event.** These wordings point at "
            "different events on different claims, which is kept as-is — each "
            "claim keeps its own end:\n\n"
            + "\n".join(
                f"- “{label}” → " + ", ".join(f"**{e}**" for e in sorted(events))
                for label, events in sorted(split_labels.items())
            )
            + "\n\nThis is what a cell-type split looks like once it is made, "
            "and it is the right answer when one wording really covers two "
            "events. If it was not deliberate, set the affected rows back to "
            "one name before applying.",
            icon="⚖️",
        )

    changed = not edited.equals(current)
    if not changed:
        st.caption("No changes to apply.")

    if not require_curator():
        return

    # No rationale box. This is a bulk pass over dozens of rows, so one
    # free-text sentence cannot explain any particular one; the grouping itself
    # is recorded in full, and the doubtful cases that need an argument are made
    # in **Decide**, which keeps its rationale box.
    if st.button(
        "Apply assignments", type="primary", disabled=not changed,
        key="curate_claim_apply",
    ):
        excluded = _excluded_event_names(canonical)
        try:
            result = cg.apply_assignments(
                pairs,
                excluded=excluded,
                curator=curator_name(),
                rationale="Assigned per claim in the Assign grid.",
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            # Now the row level. `apply_assignments` back-fills Table 1 by
            # label, so at this point every claim sharing a wording points at
            # the same event; this repoints each row at what its own line in the
            # grid said, which is the only step that preserves a split.
            ids_by_name = table1_store.canonical_ids_by_name()
            repoint = {
                record_id: (
                    ids_by_name.get(ends.get("upstream", "")),
                    ids_by_name.get(ends.get("downstream", "")),
                )
                for record_id, ends in per_claim.items()
            }
            table1_store.set_claim_canonical_ends(repoint)
            table1_store.recount_canonical_source_rows()

            # Claim-level state is stored separately from the grouping: it is a
            # judgement about one paper's row, not about a Key Event's identity,
            # and a re-run of Propose must not wipe it.
            for _, row in edited.iterrows():
                keep = bool(row["Keep"])
                checked = bool(row["Checked"])
                curation_store.set_curation(
                    "claim",
                    str(int(row["Claim"])),
                    status=(
                        "rejected" if not keep
                        else "accepted" if checked
                        else "unreviewed"
                    ),
                    curator=curator_name(),
                )
            _after_decision(
                f"{result['n_labels']} wording(s) assigned to "
                f"{result['n_events']} Key Event(s); "
                f"{len(repoint)} claim(s) repointed. "
                f"{n_checked} claim(s) marked checked, "
                f"{n_dropped} not kept."
            )


def _excluded_event_names(canonical: pd.DataFrame) -> list[str]:
    """Key Events currently marked off the map, read from their stored status."""
    if canonical is None or canonical.empty:
        return []
    return sorted(
        {
            str(row["canonical_name"])
            for _, row in canonical.iterrows()
            if str(row.get("curation_status") or "unreviewed") == "rejected"
        }
    )


# ---------------------------------------------------------------------------
# The event roster: one row per Key Event, with the In-AOP tick
# ---------------------------------------------------------------------------

def _event_roster(canonical: pd.DataFrame, table1: pd.DataFrame) -> None:
    """
    One row per canonical Key Event: its synonyms, its evidence, and whether
    it is on the map.

    Keeping a Key Event off the map is a decision about the event, not about
    any one claim, so it belongs on a table whose unit is the event. It used to
    live on the per-raw-label grid, where unticking one of four wordings for the
    same event raised a question the screen could not answer.
    """
    section_heading(
        "3 · Key Events — what is on the map",
        "One row per canonical Key Event, with every wording folded into it. "
        "Untick **In AOP** to keep an event off the map without deleting it.",
        help_text=(
            "**Reported change** counts how many claims described this event "
            "going each way, and is never averaged. “decreased ×4, increased "
            "×2” means the corpus disagrees, which is a finding — and it is "
            "also the signal to check whether those papers used different "
            "agents or different concentrations before treating it as a "
            "contradiction.\n\n"
            "**Claims** counts extracted rows naming this event at either end; "
            "**Papers** counts separate studies. An event resting on one claim "
            "from one paper is not the same kind of node as one resting on six.\n\n"
            "**In AOP** unticked keeps the event and all its evidence in the "
            "database and off the map. Nothing is deleted."
        ),
    )

    if canonical.empty:
        st.info("No canonical Key Events yet.")
        return

    aliases = table1_store.load_alias_crosswalk()
    by_event: dict[str, list[str]] = {}
    if not aliases.empty:
        for name, group in aliases.groupby("canonical_name"):
            by_event[str(name)] = group["raw_label"].astype(str).tolist()

    evidence = _label_evidence(
        table1,
        sorted({label for labels in by_event.values() for label in labels}),
    )

    def rollup(event: str, key: str) -> int:
        return sum(
            int(evidence.get(label, {}).get(key, 0) or 0)
            for label in by_event.get(event, [])
        )

    def changes(event: str) -> str:
        counted: Counter = Counter()
        for label in by_event.get(event, []):
            reported = str(evidence.get(label, {}).get("reported") or "")
            for piece in reported.split(","):
                piece = piece.strip()
                if not piece or piece == "not stated":
                    continue
                word, _, times = piece.rpartition("×")
                try:
                    counted[word.strip()] += int(times)
                except ValueError:
                    counted[piece] += 1
        return ", ".join(f"{w} ×{n}" for w, n in counted.most_common(3)) or "not stated"

    current = pd.DataFrame(
        [
            {
                "Key Event": str(row["canonical_name"]),
                "Level": str(row.get("level") or "—"),
                "In AOP": str(row.get("curation_status") or "unreviewed") != "rejected",
                "Wordings": len(by_event.get(str(row["canonical_name"]), [])) or 1,
                "Reported change": changes(str(row["canonical_name"])),
                "Claims": rollup(str(row["canonical_name"]), "claims"),
                "Papers": rollup(str(row["canonical_name"]), "papers"),
                "Synonyms": "; ".join(
                    label
                    for label in by_event.get(str(row["canonical_name"]), [])
                    if label.strip().casefold()
                    != str(row["canonical_name"]).strip().casefold()
                ) or "—",
            }
            for _, row in canonical.iterrows()
        ]
    ).sort_values(["Papers", "Claims"], ascending=False)

    edited = st.data_editor(
        current,
        use_container_width=True,
        hide_index=True,
        height=320,
        key="curate_roster_editor",
        column_config={
            "Key Event": st.column_config.TextColumn(
                "Key Event", disabled=True, width="large",
                help="Rename below the table, not here.",
            ),
            "Level": st.column_config.TextColumn("Level", disabled=True, width="small"),
            "In AOP": st.column_config.CheckboxColumn(
                "In AOP",
                help="Unticked events keep their evidence and stay off the map.",
            ),
            "Wordings": st.column_config.NumberColumn(
                "Wordings", disabled=True, width="small",
                help="How many raw labels are folded into this event.",
            ),
            "Reported change": st.column_config.TextColumn(
                "Reported change", disabled=True, width="medium",
                help="Counted across every claim naming this event. Never averaged.",
            ),
            "Claims": st.column_config.NumberColumn("Claims", disabled=True, width="small"),
            "Papers": st.column_config.NumberColumn("Papers", disabled=True, width="small"),
            "Synonyms": st.column_config.TextColumn(
                "Synonyms", disabled=True, width="large",
                help="Every other wording the papers used for this event.",
            ),
        },
    )

    off_map = sorted(edited.loc[~edited["In AOP"], "Key Event"].astype(str))
    was_off = sorted(current.loc[~current["In AOP"], "Key Event"].astype(str))

    if off_map != was_off and require_curator():
        if st.button("Apply In-AOP changes", type="primary", key="curate_roster_apply"):
            status_by_name = {
                str(row["canonical_name"]): str(row["canonical_id"])
                for _, row in canonical.iterrows()
            }
            for name, canonical_id in status_by_name.items():
                table1_store.set_canonical_ke_status(
                    int(canonical_id),
                    "rejected" if name in off_map else "accepted",
                )
            _after_decision(
                f"{len(off_map)} Key Event(s) kept off the map, "
                f"{len(edited) - len(off_map)} on it."
            )

    if off_map:
        with st.expander(f"Kept off the map ({len(off_map)})"):
            for name in off_map:
                st.markdown(f"- {name}")

    st.divider()
    _rename_control(canonical)


def _rename_control(canonical: pd.DataFrame) -> None:
    """
    Give a Key Event a name none of the papers used.

    The grid's dropdown can only assign a label to a name that already exists,
    which is right for assignment — free text there would turn one mistyped
    character into a second Key Event nobody meant — and useless when the name
    you want is one no paper wrote. "Voltage-gated sodium channels" is what
    eleven papers called it; "Decreased NaV1.6 activity at heminodes" is what
    it is. That name has to be typed somewhere.

    Renaming acts on the canonical record, so every raw label assigned to it
    follows, and every original wording is still kept as an alias.
    """
    names = {str(r["canonical_name"]): int(r["canonical_id"])
             for _, r in canonical.iterrows()}
    if not names:
        return

    with st.expander("Rename a Key Event", expanded=False):
        st.caption(
            "Renaming changes the canonical record. Every raw label assigned "
            "to it keeps pointing at it, and every paper's original wording is "
            "still kept as a synonym — nothing the papers said is lost."
        )
        target = st.selectbox(
            "Key Event to rename", sorted(names), key="curate_rename_target"
        )
        new_name = st.text_input(
            "New name",
            value=target,
            key="curate_rename_value",
            help=(
                "Say which way it goes and what was measured where the papers "
                "support it — a name that reads as a measurable change is the "
                "one an AOP wants."
            ),
        )

        clean = new_name.strip()
        collision = clean in names and names[clean] != names[target]
        if collision:
            st.warning(
                f"“{clean}” is already another Key Event. Renaming to it would "
                "not merge them — use the grid above to assign both labels to "
                "the same event instead."
            )

        if st.button(
            "Rename", type="primary", key="curate_rename_apply",
            disabled=not clean or clean == target or collision,
        ):
            canonical_id = names[target]
            table1_store.rename_canonical_ke(canonical_id, clean)
            affected = wf.invalidate_for_ke(canonical_id, reason="renamed in curation")
            _after_decision(
                f"Renamed to “{clean}”."
                + (f" {affected} synthesis(es) marked stale." if affected else "")
            )


# ---------------------------------------------------------------------------
# The three-column workspace
# ---------------------------------------------------------------------------

def _workspace(canonical: pd.DataFrame, table1: pd.DataFrame) -> None:
    section_heading(
        "4 · Decide the doubtful cases",
        "Optional. Use it only for pairs you could not settle in Assign — it "
        "asks one question at a time and refuses merges it cannot justify.",
        help_text=(
            "**What it adds over Assign.** Assign takes your word for it: set "
            "two labels to the same name and they merge. Here the pair is "
            "first classified — *equivalent*, *broader than*, *narrower "
            "than*, *related but distinct*, *contradictory* — and **only a "
            "pair classified Equivalent can be merged as the same event**. "
            "That guard exists because merging a specific event into a broader "
            "one silently turns a finding about NaV1.2 into a finding about "
            "sodium channels in general.\n\n"
            "**If you want that pooling anyway**, use *Collapse into the "
            "broader Key Event*. It does the same thing mechanically, and is "
            "equally reversible, but is logged as a coarsening rather than an "
            "equivalence — so the record shows the subtype evidence was pooled "
            "deliberately rather than found to be the same thing. "
            "Contradictory pairs stay blocked either way: that is not a "
            "question of grain.\n\n"
            "**What happens on merge.** The chosen survivor keeps its record; "
            "the others' raw labels move across as its synonyms, and every "
            "Table 1 row that pointed at them is repointed at the survivor. "
            "The state before and after is saved, which is what lets **Merge "
            "history** undo it exactly.\n\n"
            "**The three columns.** Left: pairs the classifier flagged, worst "
            "first. Middle: the actual extracted rows behind each record, so "
            "you can read what the papers said before deciding. Right: what "
            "the merged record would look like.\n\n"
            "If nothing looks doubtful, skip this step."
        ),
    )

    classifications: list[Classification] = st.session_state.get("_classifications") or []
    records: dict[str, KERecord] = st.session_state.get("_ke_records") or {
        str(r["canonical_id"]): KERecord.from_row(r) for _, r in canonical.iterrows()
    }

    left, middle, right = st.columns([1.1, 1.5, 1.4], gap="medium")

    with left:
        selected_ids = _candidate_column(classifications, canonical, records)

    with middle:
        _raw_records_column(selected_ids, canonical, table1)

    with right:
        _canonical_result_column(selected_ids, canonical, records, classifications)


def _candidate_column(
    classifications: list[Classification],
    canonical: pd.DataFrame,
    records: dict[str, KERecord],
) -> list[int]:
    """Left column: proposed groups with checkboxes, plus a free selection."""
    st.markdown("##### Candidate groups")
    st.caption("Tick the records you want to act on together.")

    selected: list[int] = []
    names = {int(r["canonical_id"]): str(r["canonical_name"])
             for _, r in canonical.iterrows()}

    if not classifications:
        st.caption(
            "No classified candidates. Run **Classify candidate merges** "
            "above, or pick records manually below."
        )
    else:
        show_only_mergeable = st.checkbox(
            "Show only mergeable candidates", value=False, key="curate_only_merge",
            help="Pairs classified Equivalent. The rest still need a decision.",
        )
        shown = [c for c in classifications if c.mergeable] if show_only_mergeable \
            else classifications

        for i, c in enumerate(shown[:60]):
            try:
                a_id, b_id = int(c.source), int(c.target)
            except (TypeError, ValueError):
                continue
            a_name = names.get(a_id, c.source)
            b_name = names.get(b_id, c.target)

            with st.container(border=True):
                st.markdown(
                    relationship_badge(c.relationship.value, c.relationship.label)
                    + f" <span style='opacity:0.6;font-size:0.78em'>"
                      f"similarity {c.similarity:.2f}</span>",
                    unsafe_allow_html=True,
                )
                picked = st.checkbox(
                    f"{a_name}  ·  {b_name}",
                    key=f"cand_{i}_{a_id}_{b_id}",
                    disabled=False,
                )
                if picked:
                    selected.extend([a_id, b_id])
                    st.session_state["_active_classification"] = c

    with st.expander("Select records manually"):
        options = sorted(names.items(), key=lambda kv: kv[1])
        manual = st.multiselect(
            "Canonical Key Events",
            options=[k for k, _ in options],
            format_func=lambda k: names.get(k, str(k)),
            key="curate_manual_select",
            help=(
                "Use this to act on a group the tool did not propose. A merge "
                "made this way is recorded as a manual override."
            ),
        )
        selected.extend(int(m) for m in manual)

    return list(dict.fromkeys(selected))


def _raw_records_column(
    selected_ids: list[int], canonical: pd.DataFrame, table1: pd.DataFrame
) -> None:
    """Middle column: the raw records behind the selection, unedited."""
    st.markdown("##### Raw records")
    st.caption(
        "The wording each paper used, with the quotations behind it. Nothing "
        "here is normalised."
    )

    if not selected_ids:
        st.info("Select a candidate group to see the records behind it.")
        return

    aliases = table1_store.load_alias_map()
    for canonical_id in selected_ids:
        row = canonical[canonical["canonical_id"] == canonical_id]
        if row.empty:
            continue
        row = row.iloc[0]
        labels = [label for label, cid in aliases.items() if cid == canonical_id]

        with st.container(border=True):
            st.markdown(f"**{row['canonical_name']}**  ·  `#{canonical_id}`")
            status = wf.get_status("ke", str(canonical_id))
            st.markdown(
                state_badge(status.effective_state.label, status.drifted),
                unsafe_allow_html=True,
            )

            ok, why = is_key_event(str(row["canonical_name"]))
            if not ok:
                st.error(why, icon="🚫")

            rows = table1[
                (table1["upstream_ke_canonical_id"] == canonical_id)
                | (table1["downstream_ke_canonical_id"] == canonical_id)
            ]

            st.caption(
                f"{row['level']} level · {len(labels)} raw wording(s) · "
                f"{len(rows)} claim(s) · "
                f"{rows['source_doi'].nunique() if not rows.empty else 0} paper(s)"
            )

            if labels:
                st.markdown("*Raw wordings*")
                for label in sorted(labels):
                    st.markdown(f"- {label}")

            if not rows.empty:
                with st.expander(f"Source papers and quotations ({len(rows)})"):
                    _record_evidence(rows, canonical_id)


def _record_evidence(rows: pd.DataFrame, canonical_id: int) -> None:
    """Papers, direction, level and participants for one canonical record."""
    keys = citation_keys(rows["source_doi"].tolist())
    for _, r in rows.head(12).iterrows():
        side = ("upstream" if r["upstream_ke_canonical_id"] == canonical_id
                else "downstream")
        other = (r["downstream_ke_name"] if side == "upstream"
                 else r["upstream_ke_name"])
        st.markdown(
            f"**{cite(r['source_doi'], keys)}** — appears as the *{side}* "
            f"event of “{fmt(r['ker_name'])}”, opposite {fmt(other)}."
        )
        st.caption(
            f"Level {fmt(r[f'{side}_ke_level'])} · "
            f"{fmt(r['study_design'])} · {fmt(r['taxonomic_applicability'])} · "
            f"{fmt(r['sex_applicability'])} · {fmt(r['life_stage_applicability'])}"
            + (" · **contradicts the relationship**"
               if bool(r["contradicts_ker"]) else "")
        )

        spans = table1_store.load_evidence_spans([int(r["record_id"])])
        if not spans.empty:
            for _, span in spans.head(3).iterrows():
                st.markdown(
                    f"> {span['quote']}  \n"
                    f"<span style='opacity:0.6;font-size:0.8em'>"
                    f"{span['citation']}</span>",
                    unsafe_allow_html=True,
                )
        st.markdown("---")


def _canonical_result_column(
    selected_ids: list[int],
    canonical: pd.DataFrame,
    records: dict[str, KERecord],
    classifications: list[Classification],
) -> None:
    """Right column: what the result would be, and the actions available."""
    st.markdown("##### Canonical result")

    if not selected_ids:
        st.info("Nothing selected.")
        return

    if len(selected_ids) == 1:
        _single_record_actions(selected_ids[0], canonical)
        return

    verdict = _verdict_for(selected_ids, records, classifications)

    with st.container(border=True):
        st.markdown(
            relationship_badge(verdict.relationship.value, verdict.relationship.label),
            unsafe_allow_html=True,
        )
        st.write(verdict.explanation)

        with st.expander("How this was decided", expanded=not verdict.mergeable):
            for check in verdict.checks:
                st.markdown(f"{check.icon} **{check.name}** — {check.detail}")

    try:
        preview = cg.preview_merge(selected_ids)
    except ValueError as exc:
        st.error(str(exc))
        return

    _preview_panel(preview)
    _action_panel(selected_ids, verdict, preview)


def _verdict_for(
    selected_ids: list[int],
    records: dict[str, KERecord],
    classifications: list[Classification],
) -> Classification:
    """
    The classification governing a selection.

    For a pair, the cached verdict if there is one. For a larger group, the
    *worst* pairwise verdict: a group is only equivalent if every member is
    equivalent to every other, so one contradiction anywhere sinks the merge.
    """
    active = st.session_state.get("_active_classification")
    if (
        active is not None
        and len(selected_ids) == 2
        and {str(active.source), str(active.target)} == {str(i) for i in selected_ids}
    ):
        return active

    lookup = {str(k): v for k, v in records.items()}
    pairs: list[Classification] = []
    for i in range(len(selected_ids)):
        for j in range(i + 1, len(selected_ids)):
            a = lookup.get(str(selected_ids[i]))
            b = lookup.get(str(selected_ids[j]))
            if a and b:
                pairs.append(classify(a, b))

    if not pairs:
        return Classification(
            source=str(selected_ids[0]),
            target=str(selected_ids[-1]),
            relationship=Relationship.UNCERTAIN,
            explanation="These records could not be compared.",
        )

    return worst(pairs)


def _preview_panel(preview: cg.MergePreview) -> None:
    """Everything the merge would change, before it changes."""
    st.markdown("**If merged**")
    st.markdown(
        f"Survivor: **{preview.survivor_name}** (`#{preview.survivor_id}`), "
        f"absorbing {len(preview.absorbed_ids)} record(s)."
    )

    bullets = [
        f"{len(preview.aliases_moving)} alias(es) move across",
        f"{preview.evidence_reassigned} evidence row(s) reassigned",
    ]
    if preview.kers_consolidated:
        bullets.append(f"{len(preview.kers_consolidated)} KER consolidation(s)")
    st.markdown("\n".join(f"- {b}" for b in bullets))

    if preview.aliases_moving:
        with st.expander(f"Aliases that will move ({len(preview.aliases_moving)})"):
            for alias in preview.aliases_moving:
                st.markdown(f"- {alias}")

    for note in preview.kers_consolidated:
        st.caption(f"• {note}")

    for problem in preview.blocking:
        st.error(problem, icon="🚫")
    for warning in preview.warnings:
        st.warning(warning, icon="⚠️")
    for touched in preview.approved_records_touched:
        st.warning(
            f"{touched} — merging will retract that approval and mark anything "
            f"synthesised from it stale.",
            icon="↩️",
        )


def _action_panel(
    selected_ids: list[int], verdict: Classification, preview: cg.MergePreview
) -> None:
    """The six actions. Merge is disabled unless the verdict permits it."""
    st.markdown("**Action**")

    if not require_curator():
        return

    rationale = st.text_area(
        "Rationale",
        key=f"curate_rationale_{'_'.join(map(str, selected_ids))}",
        placeholder="Why this decision? Recorded with your name.",
        height=80,
    )
    curator = curator_name()

    merge_blocked = (not verdict.mergeable) or bool(preview.blocking)
    merge_help = (
        f"Blocked: classified “{verdict.relationship.label}”. Only Equivalent "
        f"records may be merged as the same event. To pool them at a coarser "
        f"grain instead, use **Collapse into the broader Key Event**."
        if not verdict.mergeable
        else ("Blocked: " + "; ".join(preview.blocking))
        if preview.blocking
        else "Fold these records into one canonical Key Event."
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Merge as equivalent", type="primary", disabled=merge_blocked,
                     use_container_width=True, help=merge_help,
                     key=f"act_merge_{selected_ids}"):
            try:
                result = cg.merge_as_equivalent(
                    selected_ids,
                    classification=verdict,
                    curator=curator,
                    rationale=rationale or None,
                )
            except cg.MergeRefused as exc:
                st.error(str(exc))
            else:
                _after_decision(
                    f"Merged into #{result['survivor_id']}. "
                    f"{result['aliases_moved']} alias(es) moved; "
                    f"{result['syntheses_invalidated']} synthesis(es) marked stale."
                )

        _collapse_action(selected_ids, verdict, preview, curator, rationale)

        if st.button("Keep separate", use_container_width=True,
                     key=f"act_sep_{selected_ids}",
                     help="Records the pair as not duplicates so it stops being suggested."):
            cg.keep_separate(selected_ids, classification=verdict,
                             curator=curator, rationale=rationale or None)
            _after_decision("Recorded as separate records.")

        if st.button("Mark unresolved", use_container_width=True,
                     key=f"act_unres_{selected_ids}",
                     help="Park the group without deciding. It stays visible."):
            cg.mark_unresolved(selected_ids, classification=verdict,
                               curator=curator, rationale=rationale or None)
            _after_decision("Marked unresolved.")

    with c2:
        _broader_mapping_action(selected_ids, curator, rationale)
        _relation_action(selected_ids, curator, rationale)
        _reject_action(selected_ids, curator, rationale)


def _collapse_action(
    selected_ids: list[int],
    verdict: Classification,
    preview: cg.MergePreview,
    curator: str,
    rationale: str,
) -> None:
    """
    Pool a subtype into the class, on purpose and on the record.

    "Merge as equivalent" is blocked for a broader/narrower pair, and rightly:
    it asserts the two are the same event. But a curator working at the class
    level is not claiming that — they are saying the subtype distinction is
    not one this AOP turns on. That is a legitimate decision the tool had no
    way to express, leaving "Map to a broader concept", which keeps both
    records and so does not do what was asked.
    """
    with st.popover("Collapse into the broader Key Event", use_container_width=True):
        st.caption(
            "Pools these records into one at a coarser grain. Mechanically "
            "the same as a merge — the absorbed names survive as synonyms and "
            "it can be undone — but recorded as a **coarsening**, not as an "
            "equivalence, so the log shows the subtype evidence was pooled "
            "rather than found to be the same thing."
        )
        st.warning(
            "Evidence about the specific event will read as evidence about "
            "the broader one from here on. Use this only where that is the "
            "claim you want to make.",
            icon="⚠️",
        )

        options = [preview.survivor_id] + list(preview.absorbed_ids)
        names = {}
        try:
            canonical = table1_store.load_canonical_kes()
            names = {int(r["canonical_id"]): str(r["canonical_name"])
                     for _, r in canonical.iterrows()}
        except Exception:  # noqa: BLE001
            names = {}

        # Which record is the broader one IS the decision, so it is asked
        # rather than inferred. `preview_merge` picks a survivor by evidence
        # volume, and the best-evidenced record is often the narrow one.
        survivor = st.selectbox(
            "Which is the broader Key Event? It is the one that survives.",
            options=options,
            format_func=lambda i: f"{names.get(i, i)} (#{i})",
            key=f"collapse_survivor_{selected_ids}",
        )

        blocked = bool(preview.blocking) or (
            verdict.relationship is Relationship.CONTRADICTORY
        )
        if verdict.relationship is Relationship.CONTRADICTORY:
            st.error(
                "Blocked: these records are classified contradictory. That is "
                "not a difference of grain — pooling them would discard one of "
                "two opposite findings."
            )
        elif preview.blocking:
            st.error("Blocked: " + "; ".join(preview.blocking))

        if st.button(
            "Collapse into this record", type="primary", disabled=blocked,
            use_container_width=True, key=f"act_collapse_{selected_ids}",
        ):
            try:
                result = cg.collapse_into_broader(
                    selected_ids,
                    survivor_id=int(survivor),
                    classification=verdict,
                    curator=curator,
                    rationale=rationale or None,
                )
            except (cg.MergeRefused, ValueError) as exc:
                st.error(str(exc))
            else:
                _after_decision(
                    f"Collapsed into #{result['survivor_id']}. "
                    f"{result['aliases_moved']} alias(es) moved; "
                    f"{result['syntheses_invalidated']} synthesis(es) marked stale."
                )


def _broader_mapping_action(
    selected_ids: list[int], curator: str, rationale: str
) -> None:
    """Attach an ontology parent without collapsing the record."""
    with st.popover("Map to a broader concept", use_container_width=True):
        st.caption(
            "Keeps the specific Key Event and attaches a parent term. Use this "
            "when one record is a *kind of* the other — evidence about a "
            "subtype must not be pooled with evidence about the whole class."
        )
        target = st.selectbox(
            "Key Event to map",
            options=selected_ids,
            key=f"map_target_{selected_ids}",
        )
        query = st.text_input(
            "Search for a broader term", key=f"map_query_{selected_ids}"
        )
        if query and st.button("Search OLS4", key=f"map_search_{selected_ids}"):
            st.session_state[f"_map_hits_{target}"] = ols4_client.parents_for_mapping(query)

        hits = st.session_state.get(f"_map_hits_{target}") or []
        if hits:
            choice = st.selectbox(
                "Broader term",
                options=list(range(len(hits))),
                format_func=lambda i: f"{hits[i].curie} — {hits[i].label} ({hits[i].ontology})",
                key=f"map_choice_{selected_ids}",
            )
            if st.button("Attach as broader concept", type="primary",
                         key=f"map_apply_{selected_ids}"):
                match = hits[choice]
                cg.map_to_broader(
                    int(target), curie=match.curie, label=match.label,
                    iri=match.iri, source=match.ontology, score=match.score,
                    curator=curator, rationale=rationale or None,
                )
                _after_decision(f"{match.curie} attached as a broader concept.")


def _relation_action(selected_ids: list[int], curator: str, rationale: str) -> None:
    """Record that two distinct Key Events are biologically related."""
    with st.popover("Record a biological relationship", use_container_width=True):
        st.caption(
            "For records that are genuinely different but connected — one "
            "upstream of the other, part of the other, or a marker for it."
        )
        if len(selected_ids) < 2:
            st.info("Select two records.")
            return
        source = st.selectbox("From", selected_ids, key=f"rel_src_{selected_ids}")
        target = st.selectbox(
            "To", [i for i in selected_ids if i != source],
            key=f"rel_tgt_{selected_ids}",
        )
        relation = st.selectbox(
            "Relationship", cg.RELATION_TYPES,
            format_func=lambda r: r.replace("_", " "),
            key=f"rel_type_{selected_ids}",
        )
        if st.button("Record relationship", type="primary",
                     key=f"rel_apply_{selected_ids}"):
            cg.record_relation(int(source), int(target), relation,
                               curator=curator, rationale=rationale or None)
            _after_decision(f"Recorded: {source} {relation.replace('_', ' ')} {target}.")


def _reject_action(selected_ids: list[int], curator: str, rationale: str) -> None:
    """Reject a record as not being a Key Event."""
    with st.popover("Reject as not a Key Event", use_container_width=True):
        st.caption(
            "For study observations (“no change in X”) and bare entities, "
            "which belong in the evidence rather than on the map."
        )
        target = st.selectbox(
            "Record", selected_ids, key=f"rej_target_{selected_ids}"
        )
        if st.button("Reject", type="primary", key=f"rej_apply_{selected_ids}"):
            cg.reject_not_ke(int(target), curator=curator,
                             rationale=rationale or None)
            _after_decision(f"#{target} rejected as not a Key Event.")


def _single_record_actions(canonical_id: int, canonical: pd.DataFrame) -> None:
    """Rename, re-level and re-annotate a single record."""
    row = canonical[canonical["canonical_id"] == canonical_id]
    if row.empty:
        st.info("Record not found.")
        return
    row = row.iloc[0]

    status = wf.get_status("ke", str(canonical_id))
    st.markdown(state_badge(status.effective_state.label, status.drifted),
                unsafe_allow_html=True)

    if not require_curator():
        return

    with st.form(f"single_{canonical_id}"):
        name = st.text_input("Canonical name", value=str(row["canonical_name"]))
        from schemas import KE_LEVEL_ORDER
        level = st.selectbox(
            "Biological level", KE_LEVEL_ORDER,
            index=(KE_LEVEL_ORDER.index(str(row["level"]))
                   if str(row["level"]) in KE_LEVEL_ORDER else 0),
        )
        if st.form_submit_button("Save", type="primary"):
            changed = False
            if name.strip() and name.strip() != str(row["canonical_name"]):
                table1_store.rename_canonical_ke(int(canonical_id), name.strip())
                changed = True
            if level != str(row["level"]):
                table1_store.set_canonical_ke_level(int(canonical_id), level)
                changed = True
            if changed:
                affected = wf.invalidate_for_ke(
                    int(canonical_id), reason="edited in curation"
                )
                _after_decision(
                    f"Saved. {affected} synthesis(es) marked stale."
                    if affected else "Saved."
                )
            else:
                st.info("Nothing changed.")

    mappings = cg.ontology_mappings(int(canonical_id))
    if not mappings.empty:
        st.markdown("**Broader concepts**")
        for _, m in mappings.iterrows():
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"`{m['curie']}` {fmt(m['label'])} — *{m['relation']}*")
            if c2.button("Remove", key=f"unmap_{m['mapping_id']}"):
                cg.remove_mapping(int(m["mapping_id"]))
                st.rerun()


def _after_decision(message: str) -> None:
    """Clear caches, tell the user, and redraw."""
    invalidate_pipeline()
    st.session_state.pop("_classifications", None)
    st.session_state.pop("_active_classification", None)
    st.success(message)
    st.rerun()


# ---------------------------------------------------------------------------
# Merge history — the log, and the way back
# ---------------------------------------------------------------------------

def _canonical_groups_view() -> None:
    """Every completed merge, its provenance, and the way back."""
    section_heading(
        "5 · Merge history",
        "A log of every merge already made, not another way to make one. "
        "Each entry can be undone, and single labels split back out.",
        help_text=(
            "**Why this exists.** A merge destroys information: two records "
            "become one, and afterwards nothing in the table shows that it "
            "ever happened. So before each merge the tool saves a snapshot of "
            "the records, their aliases and every Table 1 row that pointed at "
            "them. **Undo** replays that snapshot, restoring the original "
            "records under their original ids — so a layout, an export or a "
            "note in your manuscript that referred to one still resolves.\n\n"
            "**Split** is narrower: it pulls a single raw label back out into "
            "a Key Event of its own, leaving the rest of the merge intact.\n\n"
            "Each entry records who merged what, what the classifier said at "
            "the time, and the reason given. A merge the classifier did not "
            "vouch for is marked **manual override**."
        ),
    )

    include_reverted = st.checkbox("Include undone merges", value=False,
                                   key="cg_include_reverted")
    groups = cg.canonical_groups(include_reverted=include_reverted)

    if groups.empty:
        st.info("No merges recorded yet.")
        return

    st.caption(f"{len(groups)} merge(s).")

    for _, row in groups.iterrows():
        # The kind of decision goes in the title, not inside the expander. A
        # coarsening and an equivalence merge leave identical database state,
        # so the log is the only place the difference survives — and a
        # difference you have to open something to see is one nobody sees.
        kind = "🝖 coarsened" if row.get("action") == "collapse_broader" else "≡ equivalent"
        title = (
            f"{'↩️ ' if row['reverted'] else ''}{row['canonical_ke']} "
            f"· {kind} · {row['n_claims']} claim(s) · {row['n_publications']} paper(s)"
        )
        with st.expander(title):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Canonical Key Event**")
                st.write(f"{row['canonical_ke']} ({row['level']})")
                st.markdown("**Original aliases**")
                st.write(row["original_aliases"] or "—")
                st.markdown("**Ontology term**")
                st.write(row["ontology_term"] or "—")
                st.markdown("**Broader concepts**")
                st.write(row["broader_concepts"] or "—")

            with c2:
                st.markdown("**Merge method**")
                st.write(f"{row['merge_method']} · classified "
                         f"{row['classification'] or 'unclassified'}")
                st.markdown("**Curator**")
                st.write(row["curator"] or "—")
                st.markdown("**Rationale**")
                st.write(row["rationale"] or "—")
                st.markdown("**Date**")
                st.write(row["date"])
                st.markdown("**Workflow state**")
                st.markdown(state_badge(row["workflow_state"]),
                            unsafe_allow_html=True)

            st.markdown("**Source publications**")
            st.write(row["source_publications"] or "—")

            if row["explanation"]:
                st.markdown("**Why this was classified as it was**")
                st.info(row["explanation"])

            _before_after(int(row["decision_id"]))

            if not row["reverted"]:
                _undo_and_split(int(row["decision_id"]), row)
            else:
                st.caption(f"Undone {row['reverted_at']}.")


def _before_after(decision_id: int) -> None:
    """The state either side of the merge."""
    detail = cg.group_detail(decision_id)
    if not detail:
        return
    with st.expander("Before and after"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before**")
            before = detail["before"].get("ke_canonical", [])
            for record in before:
                st.markdown(
                    f"- `#{record.get('canonical_id')}` "
                    f"{record.get('canonical_name')} "
                    f"({record.get('level')}, {record.get('n_source_rows')} rows)"
                )
        with c2:
            st.markdown("**After**")
            after = detail["after"].get("ke_canonical", [])
            for record in after:
                st.markdown(
                    f"- `#{record.get('canonical_id')}` "
                    f"{record.get('canonical_name')} "
                    f"({record.get('level')}, {record.get('n_source_rows')} rows)"
                )


def _undo_and_split(decision_id: int, row: pd.Series) -> None:
    """Reverse the whole merge, or pull one alias out of it."""
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Undo this merge", key=f"undo_{decision_id}"):
            try:
                result = cg.undo(decision_id, curator=curator_name() or None)
            except ValueError as exc:
                st.error(str(exc))
            else:
                _after_decision(
                    f"Merge undone; {len(result['restored_ids'])} record(s) restored."
                )

    with c2:
        with st.popover("Split an alias out"):
            aliases = [a.strip() for a in str(row["original_aliases"]).split(";") if a.strip()]
            if not aliases:
                st.info("No aliases recorded.")
            else:
                alias = st.selectbox("Alias", aliases, key=f"split_alias_{decision_id}")
                if st.button("Split into its own Key Event", type="primary",
                             key=f"split_apply_{decision_id}"):
                    detail = cg.group_detail(decision_id)
                    survivor = detail["decision"]["survivor_id"]
                    new_id = cg.split_alias(
                        int(survivor), alias, curator=curator_name() or None
                    )
                    _after_decision(f"“{alias}” split out as #{new_id}.")


def _decision_log() -> None:
    """The full curation trail."""
    with st.expander("Curation decision log"):
        log = cg.decision_log()
        if log.empty:
            st.info("No decisions recorded yet.")
            return
        st.dataframe(
            log[
                ["created_at", "action_label", "relationship", "member_ids",
                 "curator", "curator_rationale", "method", "reverted"]
            ],
            use_container_width=True,
            hide_index=True,
            height=300,
        )
