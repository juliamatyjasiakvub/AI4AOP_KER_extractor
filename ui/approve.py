from __future__ import annotations

"""
Sign-off. The step that unlocks synthesis.

Approval is the point of the whole linear order. Everything downstream —
the weight-of-evidence text, the confidence bands, the graph — reads as
settled, and none of it should be produced over Key Events nobody has looked
at. This section is where a curator says "these are right", and it is the only
thing that opens the next two.

It also shows what has drifted: records approved earlier and edited since. A
stale approval is worse than no approval, because the state column still says
"approved" while the thing approved has changed underneath it.
"""

from typing import Optional

import pandas as pd
import streamlit as st

from stage2_extraction import causal_layout, table1_store, workflow_state as wf
from ui.common import (
    curator_name,
    fmt,
    require_curator,
    section_heading,
    section_intro,
    state_badge,
)


HOW_TO = (
    "Review the Key Events below. Anything still **Raw** or **Normalization "
    "proposed** has not been curated — go back and decide it first.",
    "Mark the records you have checked as **Curated**, then **Approve** them.",
    "Approve the relationships you intend to synthesise, in the second table.",
    "Say which events start and end the pathway under **Pathway endpoints**.",
    "Once everything a KER depends on is approved, **Synthesize evidence** "
    "opens for it.",
)


def render() -> None:
    section_intro(
        "Approve",
        "Approve",
        "Sign off the curated Key Events and relationships. Nothing is "
        "synthesised and nothing is drawn until this is done.",
        HOW_TO,
        caution=(
            "Editing an approved record afterwards retracts its approval, "
            "marks anything built on it stale, and requires regeneration."
        ),
    )

    canonical = table1_store.load_canonical_kes()
    if canonical.empty:
        st.info("No canonical Key Events yet. Start in **Normalize and curate**.")
        return

    _overview()
    st.divider()
    _drift_warnings()
    _key_events()
    st.divider()
    _relationships()
    st.divider()
    _roles_editor(canonical)
    st.divider()
    _audit_trail()


# ---------------------------------------------------------------------------
# Pathway endpoints
# ---------------------------------------------------------------------------

def _declared_roles() -> dict[str, str]:
    """Curator-assigned MIE / KE / AO labels, keyed by canonical name."""
    with table1_store.connect() as conn:
        rows = conn.execute(
            "SELECT k.canonical_name, r.role FROM ke_role r "
            "JOIN ke_canonical k ON k.canonical_id = r.canonical_id "
            "WHERE r.approved = 1"
        ).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}


def _roles_editor(canonical: pd.DataFrame) -> None:
    """
    Which events start and end the pathway.

    This sat under the finished map, which is the wrong place twice over: it
    is a curation decision, not a drawing option, and the map is downstream of
    it — the picture cannot be right until the roles are, so being asked about
    them after looking at the picture is backwards.
    """
    section_heading(
        "Pathway endpoints — what starts and ends this AOP",
        "Say which Key Event the stressor acts on first, and which one (if "
        "any) is the harm. Everything else stays an intermediate Key Event.",
        help_text=(
            "**Nothing is assigned a role unless you assign it.** Every Key "
            "Event is drawn as a Key Event until you say otherwise. The map "
            "used to guess the initiating event — whatever had nothing "
            "upstream of it, at a molecular level — but both halves of that "
            "describe the corpus rather than the biology. Nothing upstream "
            "means no paper *you have collected* reported an earlier step, "
            "which is the normal state of a corpus assembled around the "
            "middle of a pathway.\n\n"
            "**Why the Adverse Outcome matters more than the others.** "
            "Calling something an adverse outcome is a regulatory claim: it "
            "asserts that this endpoint is a harm worth regulating, not "
            "merely a change that was measured. So nothing is labelled one "
            "unless you say so. Most mechanistic corpora never reach an "
            "adverse outcome, and a pathway fragment with no AO is a normal, "
            "publishable result — not an error to be fixed by promoting the "
            "last event you happen to have.\n\n"
            "**Suggestions below are proposals only.** They come from the "
            "graph and the biological levels, and nothing is applied until "
            "you approve it."
        ),
    )

    graph = _role_graph(canonical)
    has_graph = graph.number_of_nodes() > 0

    # Offered, not applied. Nothing here takes effect until it is approved.
    left, right = st.columns(2)

    with left:
        st.markdown("**Where the pathway might start**")
        mie_options = causal_layout.mie_candidates(graph) if has_graph else []
        plausible_mie = [c for c in mie_options if c.get("plausible")]
        if plausible_mie:
            for c in plausible_mie:
                st.markdown(f"- **{c['ke_name']}** — {c['why']}")
        elif mie_options:
            st.caption(
                "Nothing molecular sits at the head of this graph. The first "
                "step is probably in a paper you have not collected."
            )
        else:
            st.caption("Every Key Event has something upstream of it.")

    with right:
        st.markdown("**Where the pathway might end**")
        candidates = causal_layout.ao_candidates(graph) if has_graph else []
        plausible = [c for c in candidates if c.get("plausible")]
        if plausible:
            for c in plausible:
                st.markdown(f"- **{c['ke_name']}** — {c['why']}")
        else:
            st.caption(
                "No organism-level endpoint in this corpus, so there is "
                "probably no adverse outcome to declare. A pathway fragment "
                "is a normal result."
            )

    st.caption(
        "Suggestions only, from the shape of the graph. Until you assign one, "
        "every event on the map is drawn as an ordinary Key Event."
    )

    declared = _declared_roles()
    if declared:
        with st.expander(f"Roles already set ({len(declared)})"):
            for name, role in sorted(declared.items()):
                st.markdown(f"- **{name}** — {role}")

    ids = {str(r["canonical_name"]): int(r["canonical_id"])
           for _, r in canonical.iterrows()}
    if not ids:
        st.info("No canonical Key Events to assign roles to yet.")
        return

    node = st.selectbox("Key Event", sorted(ids), key="role_node")
    current = declared.get(node, "")
    role = st.radio(
        "Role",
        ["MIE", "KE", "AO"],
        horizontal=True,
        index=["MIE", "KE", "AO"].index(current) if current in ("MIE", "KE", "AO") else 1,
        key="role_choice",
        captions=[
            "The event the stressor acts on directly.",
            "An intermediate measurable change.",
            "A harm of regulatory concern.",
        ],
    )

    # No rationale box. It was a free-text field on a three-way choice whose
    # whole content is the choice, and what it collected was the word
    # "endpoint". The decision, who made it and when are all recorded anyway.
    if st.button("Approve this assignment", type="primary", key="role_save"):
        canonical_id = ids.get(node)
        if canonical_id is None:
            st.error("That Key Event is not in the canonical table.")
            return
        with table1_store.connect() as conn:
            conn.execute(
                "INSERT INTO ke_role (canonical_id, role, curator, rationale, "
                "approved, updated_at) VALUES (?, ?, ?, NULL, 1, datetime('now')) "
                "ON CONFLICT(canonical_id) DO UPDATE SET role = excluded.role, "
                "curator = excluded.curator, approved = 1, "
                "updated_at = excluded.updated_at",
                (canonical_id, role, curator_name() or None),
            )
            conn.commit()
        st.success(f"{node} assigned as {role}.")
        st.rerun()


def _role_graph(canonical: pd.DataFrame):
    """
    The relationship graph, purely so the AO suggestions have something to read.

    Built from every linked Table 1 row rather than only the approved ones:
    the suggestions are a prompt to think, and withholding them until sign-off
    is complete would make them useless at the point they are needed.
    """
    import networkx as nx

    graph = nx.DiGraph()
    table1 = table1_store.load_table1_as_dataframe()
    if table1.empty or canonical.empty:
        return graph

    names = {int(r["canonical_id"]): str(r["canonical_name"])
             for _, r in canonical.iterrows()}
    levels = {int(r["canonical_id"]): str(r["level"])
              for _, r in canonical.iterrows()}

    linked = table1.dropna(
        subset=["upstream_ke_canonical_id", "downstream_ke_canonical_id"]
    )
    for _, row in linked.iterrows():
        up_id = int(row["upstream_ke_canonical_id"])
        down_id = int(row["downstream_ke_canonical_id"])
        if up_id == down_id:
            continue
        for canonical_id in (up_id, down_id):
            name = names.get(canonical_id)
            if name and name not in graph:
                graph.add_node(name, level=levels.get(canonical_id, "Molecular"))
        if names.get(up_id) and names.get(down_id):
            graph.add_edge(names[up_id], names[down_id])
    return graph


def _overview() -> None:
    counts = wf.counts("ke")
    cols = st.columns(5)
    for col, state in zip(cols, wf.State):
        col.metric(state.label, counts.get(state.value, 0))


def _drift_warnings() -> None:
    """Records whose approval no longer matches their content."""
    frame = wf.state_frame("ke")
    if frame.empty:
        return
    drifted = frame[frame["drifted"]]
    if drifted.empty:
        return

    st.error(
        f"{len(drifted)} approved Key Event(s) have changed since sign-off and "
        f"need re-approval. Anything synthesised from them is marked stale.",
        icon="⚠️",
    )
    for _, row in drifted.iterrows():
        st.caption(f"• {row['name']} — approved by {fmt(row['approved_by'])} "
                   f"on {fmt(row['approved_at'])}")


def _key_events() -> None:
    st.subheader("Key Events")

    frame = wf.state_frame("ke")
    if frame.empty:
        st.info("Nothing to approve.")
        return

    show = st.radio(
        "Show",
        ["Needing attention", "All"],
        horizontal=True,
        key="approve_ke_filter",
        help="“Needing attention” hides records that are already approved and unchanged.",
    )
    view = frame if show == "All" else frame[
        (frame["state"] != wf.State.APPROVED.value)
        | (frame["drifted"])
    ]

    if view.empty:
        st.success("Every Key Event is approved and unchanged.")
        return

    if not require_curator():
        return

    st.caption(f"{len(view)} record(s).")
    selected = _selection_table(view, key_prefix="ke")

    _bulk_actions("ke", selected)


def _relationships() -> None:
    st.subheader("Key Event Relationships")
    st.caption(
        "A relationship can only be approved once both of its Key Events are. "
        "Approving it is what makes its evidence page available."
    )

    table1 = table1_store.load_table1_as_dataframe()
    if table1.empty:
        st.info("No relationships extracted yet.")
        return

    canonical = table1_store.load_canonical_kes()
    names = {int(r["canonical_id"]): str(r["canonical_name"])
             for _, r in canonical.iterrows()}

    pairs = (
        table1.dropna(subset=["upstream_ke_canonical_id", "downstream_ke_canonical_id"])
        .groupby(["upstream_ke_canonical_id", "downstream_ke_canonical_id"])
        .agg(n_claims=("record_id", "count"),
             n_papers=("source_doi", "nunique"))
        .reset_index()
    )
    if pairs.empty:
        st.info(
            "No relationships link two canonical Key Events yet. Normalize "
            "the Key Events first."
        )
        return

    rows = []
    for _, p in pairs.iterrows():
        up_id, down_id = int(p["upstream_ke_canonical_id"]), int(p["downstream_ke_canonical_id"])
        key = f"{up_id}->{down_id}"
        status = wf.get_status("ker", key)
        up_ok = wf.get_status("ke", str(up_id)).is_approved
        down_ok = wf.get_status("ke", str(down_id)).is_approved
        rows.append(
            {
                "target_key": key,
                "name": f"{names.get(up_id, up_id)} → {names.get(down_id, down_id)}",
                "level": "",
                "state": status.effective_state.value,
                "state_label": status.effective_state.label,
                "drifted": status.drifted,
                "approved_by": status.approved_by,
                "approved_at": status.approved_at,
                "endpoints_approved": up_ok and down_ok,
                "n_claims": int(p["n_claims"]),
                "n_papers": int(p["n_papers"]),
            }
        )

    frame = pd.DataFrame(rows)
    blocked = frame[~frame["endpoints_approved"]]
    if not blocked.empty:
        st.caption(
            f"{len(blocked)} relationship(s) are waiting on their Key Events "
            f"being approved."
        )

    if not require_curator():
        return

    selected = _selection_table(frame, key_prefix="ker", show_endpoints=True)
    _bulk_actions("ker", selected)


def _selection_table(
    frame: pd.DataFrame, *, key_prefix: str, show_endpoints: bool = False
) -> list[str]:
    """Rows with checkboxes and a state badge. Returns the selected keys."""
    selected: list[str] = []

    # Ticking twenty boxes by hand, and re-ticking them if anything resets the
    # page, is the slowest possible way to say "all of them".
    actionable = [
        str(r["target_key"]) for _, r in frame.iterrows()
        if not (show_endpoints and not r.get("endpoints_approved", True))
    ]
    a1, a2, _ = st.columns([1, 1, 4])
    if a1.button("Select all", key=f"{key_prefix}_all",
                 use_container_width=True):
        for key in actionable:
            st.session_state[f"{key_prefix}_pick_{key}"] = True
        st.rerun()
    if a2.button("Clear", key=f"{key_prefix}_none", use_container_width=True):
        for key in actionable:
            st.session_state[f"{key_prefix}_pick_{key}"] = False
        st.rerun()

    for _, row in frame.iterrows():
        key = str(row["target_key"])
        blocked = show_endpoints and not row.get("endpoints_approved", True)

        c1, c2, c3 = st.columns([0.5, 6, 3])
        with c1:
            if st.checkbox("", key=f"{key_prefix}_pick_{key}",
                           disabled=blocked, label_visibility="collapsed"):
                selected.append(key)
        with c2:
            st.markdown(f"**{row['name']}**")
            detail = []
            if row.get("level"):
                detail.append(str(row["level"]))
            if "n_claims" in row:
                detail.append(f"{row['n_claims']} claim(s)")
            if "n_papers" in row:
                detail.append(f"{row['n_papers']} paper(s)")
            if row.get("approved_by"):
                detail.append(f"approved by {row['approved_by']} on {row['approved_at']}")
            if blocked:
                detail.append("**waiting on its Key Events**")
            if detail:
                st.caption(" · ".join(detail))
        with c3:
            st.markdown(
                state_badge(row["state_label"], bool(row["drifted"])),
                unsafe_allow_html=True,
            )

    return selected


def _bulk_actions(target_type: str, selected: list[str]) -> None:
    """
    Move a selection through the workflow.

    The two-step curate-then-approve sequence exists so that on a shared
    instance nobody signs off by accident. On a single-user machine it is a
    trap: **Approve** silently skips everything not yet curated, reports
    "0 record(s) were skipped… nothing changed", and leaves no clue that a
    different button had to be pressed first. So the primary action now does
    both steps in one call, and the individual steps stay available for anyone
    who wants them.
    """
    if not selected:
        st.caption("Select records to act on them, or use **Select all** above.")
        return

    st.markdown(f"**{len(selected)} selected**")
    note = st.text_input(
        "Note (optional)", key=f"approve_note_{target_type}",
        placeholder="Recorded against every record in this action.",
    )
    curator = curator_name()
    targets = [(target_type, k) for k in selected]

    c1, c2 = st.columns([2, 1])

    with c1:
        if st.button("Curate and approve", type="primary",
                     use_container_width=True,
                     key=f"btn_signoff_{target_type}",
                     help="Marks these curated and signs them off in one step."):
            curated = wf.bulk_set(
                targets, wf.State.CURATED, curator=curator, note=note or None
            )
            approved = wf.bulk_set(
                targets, wf.State.APPROVED, curator=curator, note=note or None
            )
            _report(
                {
                    "changed": approved["changed"],
                    "skipped": approved["skipped"],
                },
                context=(
                    f"{curated['changed']} marked curated, "
                    f"{approved['changed']} approved."
                ),
            )

    with c2:
        if st.button("Retract approval", use_container_width=True,
                     key=f"btn_retract_{target_type}",
                     help="Pulls sign-off back and invalidates anything built on it."):
            _report(
                wf.bulk_set(
                    targets, wf.State.CURATED,
                    curator=curator, note=note or "Approval retracted",
                )
            )

    with st.expander("One step at a time"):
        s1, s2 = st.columns(2)
        with s1:
            if st.button("Mark curated only", use_container_width=True,
                         key=f"btn_curated_{target_type}"):
                _report(wf.bulk_set(
                    targets, wf.State.CURATED, curator=curator, note=note or None
                ))
        with s2:
            if st.button("Approve only", use_container_width=True,
                         key=f"btn_approve_{target_type}",
                         help="Fails on anything not already curated."):
                result = wf.bulk_set(
                    targets, wf.State.APPROVED, curator=curator, note=note or None
                )
                if result["skipped"]:
                    st.warning(
                        f"{result['skipped']} record(s) skipped — not curated "
                        f"yet. Use **Curate and approve** instead."
                    )
                _report(result)


def _report(result: dict, context: str = "") -> None:
    if result["changed"]:
        st.success(f"{result['changed']} record(s) updated. {context}".strip())
        st.rerun()
    elif result.get("skipped"):
        st.warning(
            f"Nothing changed — all {result['skipped']} record(s) were "
            f"refused. {context}".strip()
        )
    else:
        # "Nothing changed" on its own tells a user nothing about why. The
        # usual reason is that the records were already in that state.
        st.info(
            "Nothing changed — these records were already in that state. "
            f"{context}".strip()
        )


def _audit_trail() -> None:
    with st.expander("Approval history"):
        log = wf.approval_log()
        if log.empty:
            st.info("No transitions recorded yet.")
            return
        st.dataframe(log, use_container_width=True, hide_index=True, height=300)
