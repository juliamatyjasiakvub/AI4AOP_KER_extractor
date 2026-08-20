from __future__ import annotations

"""
The final AOP: an approved snapshot, arranged by causal order.

Two changes from the old map.

It shows only what has been approved. The previous version drew whatever was
in Table 2, so a graph could contain Key Events nobody had looked at, drawn
with the same confidence as the curated ones. A picture is the most quotable
artefact the tool produces and it should not contain anything provisional.

And it is arranged left to right by causal order rather than by biological
level, with horizontal position recomputed on every render. Only vertical
nudges are saved, so no stored layout can put a Key Event in a causal column
the evidence does not support.

Every visual encoding used is explained in the legend, which is generated from
the same constants that drive the drawing — so an encoding cannot be added to
the picture without appearing in the legend.
"""

import json
from typing import Any, Optional

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from legal import with_disclaimer
from stage2_extraction import (
    causal_layout,
    direction_conflicts,
    ke_normalizer,
    stage2_report,
    table1_store,
    table2_synthesis,
    workflow_state as wf,
)
from ui.common import (
    citation_keys,
    cite,
    csv_bytes,
    curator_name,
    fmt,
    section_heading,
    section_intro,
)
from ui import manual_claim


HOW_TO = (
    "The map shows only approved Key Events and relationships. Anything "
    "missing is waiting for sign-off in **Approve**.",
    "Horizontal position is causal order and is recomputed every time — it "
    "cannot be dragged.",
    "Vertical position is yours: nudge nodes to untangle the picture and save "
    "the arrangement.",
    "**Freeze snapshot** stores the graph as it stands, so a figure you cite "
    "does not change under you.",
    "Which events are the MIE and the Adverse Outcome is decided in "
    "**Approve**, not here.",
)

#: Every visual encoding the map uses. The legend is rendered from this, so a
#: new encoding has to be declared here to exist at all.
ENCODINGS: dict[str, list[tuple[str, str, str]]] = {
    "Coloured stripe down the left of a box": [
        ("Purple", "Molecular Initiating Event",
         "The event a stressor triggers directly. The start of the pathway."),
        ("Blue", "Key Event",
         "An intermediate measurable change on the way to the outcome."),
        ("Red", "Adverse Outcome",
         "The endpoint of regulatory concern. Only appears if you declared "
         "one — most mechanistic corpora have none."),
        ("Grey", "Marker",
         "A readout measured alongside the pathway rather than a step in it. "
         "Kept with its evidence, but it is not a Key Event and never an "
         "adverse outcome."),
    ],
    "Box corners": [
        ("Square-ish", "Molecular Initiating Event", "Barely rounded corners."),
        ("Rounded", "Key Event or marker", "The default box."),
        ("Pill-shaped", "Adverse Outcome", "Fully rounded ends."),
    ],
    "Box outline": [
        ("Solid green", "Approved", "Signed off, and unchanged since."),
        ("Dashed amber, with ⚠", "Approved but edited since",
         "The picture may no longer match what was approved. Re-approve it."),
    ],
    "Arrow before the name": [
        ("↑ / ↓", "Direction the papers report",
         "Shown only where the name does not already say it — a node called "
         "“reduced myelination” gets no arrow, because “↓ reduced” reads as "
         "the opposite of the finding."),
        ("±", "The claims disagree on direction",
         "Some claims attached to this Key Event recorded an increase and "
         "others a decrease. Not a defect in the map — a finding about the "
         "corpus. **Direction conflicts**, below the map, shows which claim "
         "said what and offers the three ways out: correct the claim that was "
         "misread, split an event that is really two, or rule on a "
         "disagreement that is genuine."),
        ("⚠", "Name and measurements disagree",
         "The name says one direction and the papers report the other. Either "
         "the name or the extraction is wrong; both are worth checking. Also "
         "listed under **Direction conflicts**."),
    ],
    "Small text along the bottom of a box": [
        ("MIE · Mol · 4p", "Role · biological level · papers",
         "The role, the level of organisation abbreviated (Mol, Cell, Tis, "
         "Org, Ind, Pop), and how many separate papers contributed to it."),
    ],
    "Arrow colour": [
        ("Green", "Supported", "Every contributing study supports the relationship."),
        ("Amber", "Mixed", "The studies disagree with each other."),
        ("Red", "Contradicted", "The weight of the studies argues against it."),
        ("Violet, with an “asserted” badge", "Curator-asserted",
         "No paper in this corpus states this relationship. A curator entered "
         "it, and the rationale they gave is recorded with the claim. The "
         "other three colours report what studies found; this one has no "
         "studies to report."),
    ],
    "Arrow style": [
        ("Solid", "Adjacent", "Nothing known lies between the two Key Events."),
        ("Dashed", "Non-adjacent", "Other Key Events lie between them."),
    ],
    "Arrow direction": [
        ("→", "Causal direction",
         "Points from the upstream event to the downstream one."),
    ],
    "Column the box sits in": [
        ("MIE → Early → Intermediate → Late → AO", "Causal order",
         "Computed from the shape of the graph, not from biological level. "
         "It is recomputed every time and cannot be dragged or saved."),
    ],
}

EDGE_COLOURS = {"supporting": "#1e8e3e", "mixed": "#c98a00", "contradictory": "#c5221f"}
#: An edge no paper in the corpus states, drawn in the same violet the tool
#: uses nowhere else. It is deliberately not green: the evidence colours answer
#: "what do the studies say", and for this edge there are no studies to ask.
ASSERTED_EDGE_COLOUR = "#7b3fb5"
#: Markers are grey on purpose. They are measurements attached to the pathway
#: rather than steps in it, and colouring them like Key Events was half of why
#: five readouts were reading as five adverse outcomes.
ROLE_COLOURS = {
    "MIE": "#6a1b9a", "KE": "#1a73e8", "AO": "#c5221f", "marker": "#9aa0a6",
}
LEVEL_ABBR = {
    "MIE": "MIE", "Molecular": "Mol", "Cellular": "Cell", "Tissue": "Tis",
    "Organ": "Org", "Individual": "Ind", "Population": "Pop",
}


def render() -> None:
    section_intro(
        "Final AOP",
        "Final AOP",
        "The approved pathway. Only Key Events and relationships that have "
        "been signed off appear here.",
        HOW_TO,
    )

    graph, excluded = _approved_graph()

    if graph.number_of_nodes() == 0:
        st.warning(
            "Nothing is approved yet, so there is no pathway to draw. Approve "
            "Key Events and relationships in **Approve**.",
            icon="🔒",
        )
        if excluded:
            st.caption(f"{len(excluded)} unapproved relationship(s) are being held back.")
        return

    if excluded:
        st.info(
            f"{len(excluded)} relationship(s) are not shown because they are "
            f"not approved.",
            icon="ℹ️",
        )
        with st.expander("Which ones"):
            for label in excluded:
                st.markdown(f"- {label}")

    _snapshot_banner()

    placements = causal_layout.layout(
        graph,
        y_offsets=_load_offsets(),
        declared_roles=_declared_roles(),
    )

    problems = causal_layout.validate(placements, graph)
    for problem in problems:
        st.error(problem, icon="↩️")

    _statistics(graph, placements)

    # Before the picture, not after it. A drawn graph is persuasive whether or
    # not it is an AOP, and the answer to "is this one" should arrive first.
    try:
        _completeness_banner(table2_synthesis.compute_table2(
            table1_store.load_table1_as_dataframe()
        ))
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not assess AOP completeness: {exc}")

    _canvas(graph, placements)
    _legend()
    st.divider()
    _direction_conflicts(graph)
    st.divider()
    st.caption(
        "Every event is drawn as a Key Event unless you declare it the MIE or "
        "the Adverse Outcome, in **Approve**, under *Pathway endpoints* — that "
        "declaration is what fixes the leftmost and rightmost columns here."
    )
    st.divider()
    _layout_controls(placements)
    _snapshot_controls(graph, placements)
    _exports(graph, placements)


# ---------------------------------------------------------------------------
# Building the approved graph
# ---------------------------------------------------------------------------

def _node_evidence(table1, canonical_id: int) -> dict:
    """
    What the papers actually observed at one Key Event.

    A node labelled "Voltage-gated sodium channel activity" states a molecule
    and nothing else — not whether it went up or down, in what cell, or on
    how many papers' say-so. All of that is in Table 1 and none of it reached
    the picture. It is collected here so the node can carry it, both in its
    label and on hover.
    """
    rows = table1[
        (table1.get("upstream_ke_canonical_id") == canonical_id)
        | (table1.get("downstream_ke_canonical_id") == canonical_id)
    ]
    if rows.empty:
        return {"n_papers": 0, "n_claims": 0, "observed": "", "cells": "", "changes": ""}

    cells: list[str] = []
    for _, row in rows.iterrows():
        side = "upstream" if row.get("upstream_ke_canonical_id") == canonical_id else "downstream"
        value = row.get(f"{side}_cell_type")
        if value is not None and str(value).strip() and str(value) != "nan":
            cells.append(str(value).strip().lower())

    counted, observed = direction_conflicts.observed_summary(table1, canonical_id)

    return {
        "n_papers": int(rows["source_doi"].nunique()),
        "n_claims": len(rows),
        "observed": observed,
        "changes": direction_conflicts.dominant_change(counted),
        "cells": "; ".join(sorted(set(cells))[:4]),
    }


def _approved_graph() -> tuple[nx.DiGraph, list[str]]:
    """
    Assemble the graph from approved records only.

    Both endpoints and the relationship itself have to be approved. A KER whose
    downstream Key Event is unapproved is not drawn as a dangling edge — it is
    left out entirely, because a half-drawn relationship reads as a complete
    one.
    """
    table1 = table1_store.load_table1_as_dataframe()
    canonical = table1_store.load_canonical_kes()
    graph = nx.DiGraph()
    excluded: list[str] = []

    if table1.empty or canonical.empty:
        return graph, excluded

    names = {int(r["canonical_id"]): str(r["canonical_name"])
             for _, r in canonical.iterrows()}
    levels = {int(r["canonical_id"]): str(r["level"])
              for _, r in canonical.iterrows()}

    linked = table1.dropna(
        subset=["upstream_ke_canonical_id", "downstream_ke_canonical_id"]
    )

    # Resolved once for the whole map. Every edge needs the same lookup and
    # Streamlit reruns this function on every interaction.
    citation_key_map = citation_keys(linked["source_doi"].tolist())
    rulings = table1_store.load_ke_directions()

    for (up, down), rows in linked.groupby(
        ["upstream_ke_canonical_id", "downstream_ke_canonical_id"]
    ):
        up_id, down_id = int(up), int(down)
        if up_id == down_id:
            continue
        key = f"{up_id}->{down_id}"
        label = f"{names.get(up_id, up_id)} → {names.get(down_id, down_id)}"

        if not wf.gate(ke_ids=[up_id, down_id], ker_keys=[key]).allowed:
            excluded.append(label)
            continue

        for canonical_id, side in ((up_id, "upstream"), (down_id, "downstream")):
            name = names.get(canonical_id, str(canonical_id))
            if name not in graph:
                status = wf.get_status("ke", str(canonical_id))
                evidence = _node_evidence(table1, canonical_id)
                ruling = rulings.get(canonical_id)
                graph.add_node(
                    name,
                    canonical_id=canonical_id,
                    level=levels.get(canonical_id, "Molecular"),
                    approved=status.is_approved,
                    drifted=status.drifted,
                    **{
                        **evidence,
                        # What the claims say, what the curator ruled, and the
                        # arrow that results. All three are kept: a node whose
                        # "±" was resolved to "↓" by judgement should still be
                        # able to say that the corpus itself was split.
                        "derived_change": evidence["changes"],
                        "changes": direction_conflicts.resolved_change(
                            canonical_id, evidence["changes"], rulings
                        ),
                        "ruling": str(ruling.get("direction")) if ruling else "",
                        "ruling_rationale": (
                            str(ruling.get("rationale") or "") if ruling else ""
                        ),
                    },
                )

        n_total = len(rows)
        n_contra = int(rows["contradicts_ker"].astype(bool).sum())
        verdict = (
            "supporting" if n_contra == 0
            else "contradictory" if n_contra > n_total / 2
            else "mixed"
        )
        adjacency = (
            "Adjacent" if (rows["ker_adjacency"] == "Adjacent").any()
            else "Non-adjacent"
        )
        # Named, not counted. "3 paper(s)" invites the question the tooltip
        # then cannot answer; the citation keys answer it in the same space.
        edge_dois = sorted({str(d) for d in rows["source_doi"] if str(d).strip()})
        edge_papers = ", ".join(cite(d, citation_key_map) for d in edge_dois[:6])
        if len(edge_dois) > 6:
            edge_papers += f", and {len(edge_dois) - 6} more"

        # Whether a person put this edge here, and whether anything else did.
        #
        # The figure is the most quotable thing this tool produces, and an
        # edge a curator asserted is a different kind of claim from an edge
        # three papers support. Drawing them identically would let an
        # assertion leave the tool looking like evidence — which is the exact
        # failure the whole approval workflow exists to prevent, arriving
        # through the one door that had no check on it.
        origins = (
            rows["origin"].fillna(table1_store.LLM_ORIGIN).astype(str)
            if "origin" in rows.columns
            else pd.Series([table1_store.LLM_ORIGIN] * len(rows))
        )
        n_curator = int(origins.isin(table1_store.CURATOR_ORIGINS).sum())
        n_extracted = int(len(rows) - n_curator)

        graph.add_edge(
            names.get(up_id, str(up_id)),
            names.get(down_id, str(down_id)),
            ker_key=key,
            n_papers=int(rows["source_doi"].nunique()),
            n_claims=n_total,
            n_contradicting=n_contra,
            papers=edge_papers,
            verdict=verdict,
            adjacency=adjacency,
            # One paper calling this a measurement is enough. Nobody writes
            # "we assayed X by staining for Y" as a causal claim, so the
            # marker vote is the informed one.
            relation_kind=(
                "marker"
                if "relation_kind" in rows.columns
                and (rows["relation_kind"].astype(str) == "marker").any()
                else "causal"
            ),
            n_curator_claims=n_curator,
            n_extracted_claims=n_extracted,
            # An edge with no extracted claim behind it rests entirely on a
            # curator's judgement. One with both is evidence a curator also
            # vouched for, which is not the same thing and is not drawn as if
            # it were.
            asserted=bool(n_curator and not n_extracted),
        )

    return graph, excluded


def _direction_conflicts(graph: nx.DiGraph) -> None:
    """
    Every "±" and "⚠" on the map, with the claims behind it and a way out.

    The map could state a direction conflict and nothing more. A curator who
    saw "±" learned that the corpus disagreed and had no way to find out which
    paper said what, no way to correct the one that was misread, and no way to
    record a judgement — so the mark stayed on the figure permanently and
    stopped meaning anything.

    Three things genuinely resolve one, and the panel offers all three because
    they are different findings, not three ways of doing the same thing:

        the extraction misread a paper   → correct that claim
        two events share one name        → split them
        the literature really is split   → rule on it, with a reason

    Only the third is stored here. The first two are edits to Table 1, which
    is where the answer belongs.
    """
    section_heading(
        "Direction conflicts",
        "Key Events whose claims disagree about which way the event moved.",
        help_text=(
            "**±** means some claims attached to this Key Event recorded an "
            "increase and others a decrease.\n\n"
            "**⚠** means the event's *name* states one direction and its "
            "claims report the other — “reduced myelination” whose papers all "
            "report myelination rising. One of the two is wrong.\n\n"
            "Neither is a defect in the map. Both are findings about the "
            "corpus, and both need a person to decide what they mean."
        ),
    )

    table1 = table1_store.load_table1_as_dataframe()
    rulings = table1_store.load_ke_directions()
    flagged = direction_conflicts.find_all(graph)

    if not flagged:
        st.success(
            "No direction conflicts. Every Key Event's claims agree on which "
            "way it moved, and every name matches its measurements."
        )
        return

    unresolved = [f for f in flagged if not f["ruled"]]
    st.caption(
        f"{len(flagged)} Key Event(s) flagged, {len(unresolved)} not yet "
        f"looked at."
    )

    labels = {
        f"{f['mark']} {f['name']}" + ("  ·  ruled" if f["ruled"] else ""): f
        for f in flagged
    }
    pick = st.selectbox("Key Event", list(labels), key="conflict_pick")
    entry = labels[pick]
    name, mark = entry["name"], entry["mark"]
    canonical_id = entry["canonical_id"]

    if mark == direction_conflicts.NAME_MISMATCH:
        st.warning(
            f"**{name}** is named as though it goes one way, and its claims "
            f"report the other. Either the name is wrong — fix it in "
            f"**Normalize and curate** — or a claim is, in which case correct "
            f"it below."
        )

    claims = direction_conflicts.claims_for(table1, canonical_id)
    if claims.empty:
        st.info("No per-claim changes were recorded for this event.")
        return

    keys = citation_keys(claims["doi"].tolist())
    shown = claims.assign(
        Paper=claims["doi"].map(lambda d: cite(d, keys)),
        Recorded=claims["sign"] + " " + claims["change"],
    )[["record_id", "Paper", "Recorded", "side", "measured_as", "context", "origin"]]
    shown.columns = [
        "Row", "Paper", "Recorded", "Side", "Measured as", "Context", "Origin",
    ]
    st.dataframe(shown, use_container_width=True, hide_index=True)

    st.markdown("**How do you want to resolve it?**")
    fix, split, rule = st.tabs(
        ["A claim is wrong", "These are two events", "The split is real"]
    )

    with fix:
        st.caption(
            "The commonest cause. One paper was misread — a rescue "
            "experiment scored as a decrease, a control group taken for the "
            "treated one — and correcting that row removes the conflict at "
            "its source."
        )
        row_ids = list(claims["record_id"])
        target = st.selectbox(
            "Row to correct", row_ids,
            format_func=lambda r: (
                f"{r} — {cite(claims[claims['record_id'] == r].iloc[0]['doi'], keys)}"
                f" · {claims[claims['record_id'] == r].iloc[0]['change']}"
            ),
            key="conflict_fix_row",
        )
        manual_claim.render_form(
            key_prefix=f"conflict_edit_{target}",
            record=table1_store.load_record(int(target)),
        )

    with split:
        st.caption(
            "The other common cause, and the one worth suspecting when the "
            "papers are all sound: “sodium-channel activity” rising in one "
            "model and falling in another is often two events sharing a name "
            "that is too general to tell them apart. Splitting is done in "
            "**Normalize and curate**, where the raw wordings and the merge "
            "history are visible."
        )
        st.markdown(
            "- Open **2 · Normalize and curate**\n"
            "- Find this event under **Canonical groups**\n"
            "- Split the wordings that describe the opposite direction into "
            "their own Key Event"
        )

    with rule:
        st.caption(
            "For when the papers are right and they disagree. Say what the "
            "AOP asserts and why — or record that the conflict is real and "
            "should stay on the figure."
        )
        current = rulings.get(canonical_id, {})
        options = list(table1_store.KE_DIRECTIONS)
        choice = st.radio(
            "This Key Event",
            options,
            index=(
                options.index(str(current.get("direction")))
                if str(current.get("direction")) in options else 2
            ),
            key="conflict_rule",
            captions=[
                "The AOP asserts it increases; the map draws ↑.",
                "The AOP asserts it decreases; the map draws ↓.",
                "The disagreement is genuine; the map keeps drawing ±.",
            ],
        )
        why = st.text_area(
            "Why?",
            value=str(current.get("rationale") or ""),
            key="conflict_rule_why", height=70,
            placeholder=(
                "The two increases are from the injury model; the "
                "developmental papers all report a decrease, and this AOP is "
                "developmental."
            ),
        ).strip()

        cols = st.columns([2, 1])
        with cols[0]:
            if st.button("Record this ruling", type="primary",
                         disabled=not why, key="conflict_rule_save"):
                table1_store.set_ke_direction(
                    canonical_id, choice,
                    curator=curator_name(), rationale=why,
                )
                # The map is downstream of what was approved, so a ruling that
                # changes the arrow changes what the figure asserts.
                wf.invalidate_for_ke(
                    canonical_id, reason="direction ruled by the curator"
                )
                st.success("Recorded.")
                st.rerun()
        with cols[1]:
            if current and st.button("Withdraw", key="conflict_rule_clear"):
                table1_store.clear_ke_direction(canonical_id)
                st.rerun()


def _declared_roles() -> dict[str, str]:
    """Curator-assigned MIE / AO labels, keyed by canonical name."""
    with table1_store.connect() as conn:
        rows = conn.execute(
            "SELECT k.canonical_name, r.role FROM ke_role r "
            "JOIN ke_canonical k ON k.canonical_id = r.canonical_id "
            "WHERE r.approved = 1"
        ).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _statistics(graph: nx.DiGraph, placements: dict) -> None:
    roles = [p.role for p in placements.values()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Key Events", graph.number_of_nodes())
    c2.metric("Relationships", graph.number_of_edges())
    c3.metric("MIEs", roles.count("MIE"))
    c4.metric("Adverse Outcomes", roles.count("AO"))


#: Pan-and-zoom shell for the map.
#:
#: Deliberately dependency-free: a CDN import would make the one view a
#: curator most needs fail on a machine without internet, which is exactly the
#: machine this tool is meant to run on.
_PAN_ZOOM_HTML = """
<style>
  html, body { margin:0; height:100%; overflow:hidden; }
  #frame {
    position:relative; width:100%; height:100%; overflow:hidden;
    background:#fbfbfd; border:1px solid #e0e0e0; border-radius:8px;
    cursor:grab;
  }
  #frame.dragging { cursor:grabbing; }
  #hud {
    position:absolute; right:10px; top:10px; z-index:5;
    font:11px system-ui,sans-serif; color:#5f6368;
    background:rgba(255,255,255,.92); border:1px solid #e0e0e0;
    border-radius:6px; padding:4px 8px;
  }
  #hud button {
    font:11px system-ui,sans-serif; margin-left:6px; cursor:pointer;
    border:1px solid #dadce0; background:#fff; border-radius:4px; padding:2px 6px;
  }
</style>
<div id="frame">
  <div id="hud"><span id="pct">100%</span><button id="fit">Fit</button></div>
  __SVG__
</div>
<script>
(function () {
  const frame = document.getElementById('frame');
  const svg   = document.getElementById('aopmap');
  const pct   = document.getElementById('pct');
  let scale = 1, tx = 0, ty = 0;

  const apply = () => {
    svg.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    pct.textContent = Math.round(scale * 100) + '%';
  };

  const fit = () => {
    const w = svg.width.baseVal.value, h = svg.height.baseVal.value;
    scale = Math.min(frame.clientWidth / w, frame.clientHeight / h) * 0.98;
    tx = (frame.clientWidth  - w * scale) / 2;
    ty = (frame.clientHeight - h * scale) / 2;
    apply();
  };

  // Drag to pan. Buttons and links inside the SVG keep working because a
  // drag only starts on the background, and a click that never moved is
  // not treated as a drag at all.
  let dragging = false, sx = 0, sy = 0;
  frame.addEventListener('mousedown', e => {
    dragging = true; sx = e.clientX - tx; sy = e.clientY - ty;
    frame.classList.add('dragging'); e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    tx = e.clientX - sx; ty = e.clientY - sy; apply();
  });
  window.addEventListener('mouseup', () => {
    dragging = false; frame.classList.remove('dragging');
  });

  // Zoom about the pointer, so the thing under the cursor stays under it.
  // Zooming about the centre makes reading a corner an exercise in chasing
  // it back into view.
  frame.addEventListener('wheel', e => {
    e.preventDefault();
    const r = frame.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const next = Math.min(8, Math.max(0.1, scale * factor));
    tx = mx - (mx - tx) * (next / scale);
    ty = my - (my - ty) * (next / scale);
    scale = next; apply();
  }, { passive: false });

  frame.addEventListener('dblclick', fit);
  document.getElementById('fit').addEventListener('click', fit);
  window.addEventListener('resize', fit);
  fit();
})();
</script>
"""


def _canvas(graph: nx.DiGraph, placements: dict) -> None:
    """Render the graph as inline SVG."""
    if not placements:
        return

    pad = 90
    width = max(p.x for p in placements.values()) + 300 + pad
    height = max(p.y for p in placements.values()) + 180 + pad

    st.caption(
        "**Drag** to move around · **scroll wheel** to zoom where the pointer "
        "is · **double-click** to fit the whole map · **hover any box or "
        "arrow** to see what it means and which papers it rests on."
    )

    parts: list[str] = [
        f'<svg id="aopmap" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'style="display:block;transform-origin:0 0" '
        f'xmlns="http://www.w3.org/2000/svg">',
        "<defs>",
    ]
    for name, colour in {**EDGE_COLOURS, "asserted": ASSERTED_EDGE_COLOUR}.items():
        parts.append(
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="10" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker>'
        )
    parts.append("</defs>")

    # Column headers.
    for _, band, x in causal_layout.column_headers(placements):
        parts.append(
            f'<text x="{x + 100:.0f}" y="30" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="#5f6368" '
            f'letter-spacing="0.5">{band.upper()}</text>'
        )

    # Tall enough for three wrapped lines plus the role badge. Key Event names
    # in an AOP are descriptive by convention — "dispersed voltage-gated
    # sodium channel clustering at heminodes" is a normal one — so the box is
    # sized for the names the field actually uses.
    node_w, node_h = 200, 80
    top = 55

    # Edges first, so nodes sit above them.
    for source, target, data in graph.edges(data=True):
        if source not in placements or target not in placements:
            continue
        a, b = placements[source], placements[target]
        is_asserted = bool(data.get("asserted"))
        colour = (
            ASSERTED_EDGE_COLOUR if is_asserted
            else EDGE_COLOURS.get(data.get("verdict", "supporting"), "#5f6368")
        )
        dash = ' stroke-dasharray="7,5"' if data.get("adjacency") == "Non-adjacent" else ""
        x1, y1 = a.x + node_w, a.y + top + node_h / 2
        x2, y2 = b.x, b.y + top + node_h / 2
        mid = (x1 + x2) / 2
        # The <title> has to be a CHILD of the shape it describes. It was
        # being appended next to the path instead, which is why hovering an
        # edge did nothing: an SVG <title> with no parent element describes
        # nothing and is not rendered as a tooltip.
        # The tooltip has to explain itself, not just report numbers. "3
        # paper(s)" next to "Mixed" tells a reader nothing about what mixed
        # means or which three papers, and hovering is the only place there is
        # room to say so.
        verdict = data.get("verdict", "supporting")
        verdict_line = {
            "supporting": "Supported — every contributing study agrees.",
            "mixed": "Mixed — the studies disagree with each other.",
            "contradictory": "Contradicted — the weight of studies argues against it.",
        }.get(verdict, "")
        adjacency = data.get("adjacency", "Adjacent")
        adjacency_line = (
            "Adjacent: nothing known lies between these two events."
            if adjacency != "Non-adjacent"
            else "Non-adjacent: other Key Events lie between these two."
        )
        asserted = bool(data.get("asserted"))
        n_curator = int(data.get("n_curator_claims", 0) or 0)
        provenance_lines = (
            [
                "This arrow was entered by a curator. No paper in this",
                "corpus states it — it rests on the curator's rationale,",
                "which is recorded with the claim in step 1.",
            ]
            if asserted
            else [
                "This arrow is a relationship the papers reported, not an",
                "inference drawn by the tool.",
            ]
            + (
                [f"A curator added {n_curator} of its claims by hand."]
                if n_curator
                else []
            )
        )
        tooltip_lines = [
            f"{source}  →  {target}",
            "",
            *provenance_lines,
            "",
            verdict_line,
            adjacency_line,
            "",
            f"Built from {data.get('n_claims', 0)} claim(s) across "
            f"{data.get('n_papers', 0)} source(s).",
        ]
        if data.get("n_contradicting"):
            tooltip_lines.append(
                f"{data['n_contradicting']} of them argue AGAINST the link."
            )
        if data.get("papers"):
            tooltip_lines += ["", "Papers: " + str(data["papers"])]
        tooltip_lines += [
            "",
            "Open step 1 to read the quotations behind each claim.",
        ]
        tooltip = _escape("\n".join(tooltip_lines))
        arrow_head = "asserted" if is_asserted else data.get("verdict", "supporting")
        # Colour carries the distinction on screen; a badge carries it into a
        # screenshot pasted into a slide deck, which is where a figure does
        # most of its travelling and where no tooltip survives.
        badge = (
            f'<g><rect x="{mid - 26:.0f}" y="{(y1 + y2) / 2 - 9:.0f}" '
            f'width="52" height="16" rx="8" fill="#f3e8ff" '
            f'stroke="{ASSERTED_EDGE_COLOUR}" stroke-width="1"/>'
            f'<text x="{mid:.0f}" y="{(y1 + y2) / 2 + 3:.0f}" font-size="10" '
            f'text-anchor="middle" fill="{ASSERTED_EDGE_COLOUR}" '
            f'font-family="Helvetica, Arial, sans-serif">asserted</text></g>'
            if is_asserted
            else ""
        )
        parts.append(
            f'<g><path d="M {x1:.0f} {y1:.0f} C {mid:.0f} {y1:.0f}, '
            f'{mid:.0f} {y2:.0f}, {x2:.0f} {y2:.0f}" fill="none" '
            f'stroke="{colour}" stroke-width="2"{dash} '
            f'marker-end="url(#arrow-{arrow_head})"/>'
            f"<title>{tooltip}</title>{badge}</g>"
        )

    for name, placement in placements.items():
        data = graph.nodes[name]
        colour = ROLE_COLOURS.get(placement.role, "#1a73e8")
        drifted = bool(data.get("drifted"))
        border = "#c98a00" if drifted else "#1e8e3e"
        border_dash = ' stroke-dasharray="5,4"' if drifted else ""
        radius = {"MIE": 4, "KE": 8, "AO": 28, "marker": 8}.get(placement.role, 8)
        y = placement.y + top

        # Everything the node knows, on hover. A name alone cannot say which
        # way the event went, in what cell, or on how many papers' evidence,
        # and those are the first three questions anyone asks of a box.
        role_line = {
            "MIE": "Molecular Initiating Event — what the stressor does first.",
            "KE": "Key Event — a measurable change on the way to the outcome.",
            "AO": "Adverse Outcome — the endpoint of regulatory concern.",
            "marker": (
                "Marker — measured alongside the pathway, not a step in it. "
                "It is not a Key Event and never an adverse outcome."
            ),
        }.get(placement.role, "")
        hover = [
            name,
            "",
            role_line,
            f"Biological level: {data.get('level', 'unknown')}.",
            "",
        ]
        if data.get("n_papers"):
            hover.append(
                f"Named by {data.get('n_claims', 0)} extracted claim(s) across "
                f"{data['n_papers']} paper(s)."
            )
        else:
            hover.append("No extracted claims are attached to this event.")
        if data.get("observed"):
            # Counted, not averaged: "decreased in 3, increased in 1" is a
            # finding about the corpus and collapsing it to one arrow hides it.
            hover.append(
                f"Direction reported: {data['observed']} "
                "(counted across claims, not averaged)."
            )
        if data.get("derived_change") == "±" and not data.get("ruling"):
            hover += [
                "",
                "± the claims disagree about which way this event moved.",
                "Open “Direction conflicts” below the map to see which claim",
                "said what, and to correct, split or rule on it.",
            ]
        elif data.get("ruling"):
            ruled = {
                "increased": "increased",
                "decreased": "decreased",
                "conflicted": "genuinely split",
            }.get(str(data["ruling"]), str(data["ruling"]))
            hover += [
                "",
                f"The curator ruled this event {ruled}, against a corpus that "
                f"reported both.",
            ]
            if data.get("ruling_rationale"):
                hover.append(f"Reason: {data['ruling_rationale']}")
        if data.get("cells"):
            hover.append(f"Measured in: {data['cells']}.")
        if data.get("aliases"):
            hover += [
                "",
                f"Papers wrote this event {len(data['aliases'])} different "
                "way(s); the wordings were merged in step 2.",
            ]
        if data.get("drifted"):
            hover += [
                "",
                "⚠ Edited since it was approved, so this picture may not "
                "match what was signed off. Re-approve it in step 3.",
            ]
        hover += ["", "Horizontal position is causal order and is recomputed;",
                  "vertical position is yours to set."]

        parts.append("<g>")
        parts.append(
            f'<rect x="{placement.x:.0f}" y="{y:.0f}" width="{node_w}" '
            f'height="{node_h}" rx="{radius}" fill="white" stroke="{border}" '
            f'stroke-width="2.5"{border_dash}/>'
        )
        parts.append(
            f'<rect x="{placement.x:.0f}" y="{y:.0f}" width="6" '
            f'height="{node_h}" rx="3" fill="{colour}"/>'
        )
        # The arrow belongs in the label, not only in the tooltip: a reader
        # scanning the map should not have to hover every box to find out
        # which direction the corpus reports.
        # An arrow is only worth adding to a name that does not already say
        # which way the event went. Half these names do — "decreased myelin
        # basic protein expression", "reduced presynaptic excitability" — and
        # prefixing those produced "↓ decreased …", a double negative that
        # reads as the opposite of the finding.
        #
        # Where the name and the measurements disagree, that is worth a mark
        # of its own: a node called "reduced X" whose papers all report X
        # rising is either misnamed or misextracted, and silently trusting
        # either one is how a sign error survives review.
        arrow = data.get("changes") or ""
        stated = ke_normalizer.polarity(name)
        observed = {"↑": 1, "↓": -1}.get(arrow, 0)
        if stated and observed and stated != observed:
            arrow = "⚠"
        elif stated:
            arrow = ""
        display = f"{arrow} {name}".strip()
        # Three lines, and an ellipsis when even three will not hold it. The
        # old two-line limit cut long names off mid-word with no sign that
        # anything was missing — "dispersed voltage-gated sodium channel
        # clustering at heminodes" simply ended at "sodium". The full name is
        # always on hover.
        lines = _wrap(display, 26)
        if len(lines) > 3:
            lines = lines[:3]
            lines[2] = lines[2][:23].rstrip() + "…"
        for i, line in enumerate(lines):
            parts.append(
                f'<text x="{placement.x + 16:.0f}" y="{y + 20 + i * 15:.0f}" '
                f'font-size="12.5" fill="#202124">{_escape(line)}</text>'
            )
        badge = LEVEL_ABBR.get(str(data.get("level")), str(data.get("level"))[:4])
        papers = f" · {data['n_papers']}p" if data.get("n_papers") else ""
        parts.append(
            f'<text x="{placement.x + 16:.0f}" y="{y + node_h - 9:.0f}" '
            f'font-size="10.5" fill="{colour}" font-weight="600">'
            f'{placement.role} · {badge}{papers}</text>'
        )
        parts.append(f"<title>{_escape(chr(10).join(hover))}</title>")
        parts.append("</g>")
        if drifted:
            parts.append(
                f'<text x="{placement.x + node_w - 16:.0f}" y="{y + 20:.0f}" '
                f'text-anchor="end" font-size="13">⚠</text>'
            )

    parts.append("</svg>")
    svg = "".join(parts)

    # Rendered through a component rather than `st.markdown`, because Streamlit
    # strips <script> out of markdown and pan/zoom needs one. A scrollbar is
    # not a substitute: scrolling moves the whole canvas, whereas reading a
    # crowded region means magnifying that region and leaving the rest alone.
    components.html(_PAN_ZOOM_HTML.replace("__SVG__", svg), height=720)

    st.download_button(
        "Download this map as SVG",
        svg.encode("utf-8"),
        file_name="aop_map.svg",
        mime="image/svg+xml",
        key="map_svg_download",
        help="Opens at any size in a browser, and is the right format for a figure.",
    )


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = str(text).split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def _escape(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _legend() -> None:
    """
    Generated from `ENCODINGS`, so it cannot fall out of step with the drawing.
    """
    section_heading(
        "Legend",
        "What every colour, shape and line style on the map means.",
        help_text=(
            "This list is generated from the same table the map is drawn "
            "from, so an encoding cannot appear in the picture without "
            "appearing here.\n\n"
            "Nothing on the map is decorative. If two boxes look different, "
            "the difference is listed below and means something."
        ),
        level="subheader",
    )
    for group, entries in ENCODINGS.items():
        st.markdown(f"**{group}**")
        for symbol, meaning, explanation in entries:
            st.markdown(
                f"- `{symbol}` — **{meaning}**: {explanation}"
            )


# ---------------------------------------------------------------------------
# Roles, layout, snapshots
# ---------------------------------------------------------------------------

def _completeness_banner(table2_df) -> None:
    """
    Say plainly whether this is an AOP or a fragment of one.

    A graph with nodes and edges looks finished. Whether it has a molecular
    initiating event and an adverse outcome with a path between them is a
    different question, and one the picture cannot answer by being drawn.
    """
    verdict = table2_synthesis.aop_completeness(table2_df)
    if verdict.get("is_aop"):
        st.success(verdict["reason"], icon="✅")
    else:
        st.warning(verdict.get("reason", ""), icon="🧩")

    roles = verdict.get("roles")
    if roles is None or roles.empty:
        return

    markers = verdict.get("markers") or []
    if markers:
        st.info(
            "Measured as readouts rather than events in the pathway: "
            + ", ".join(markers)
            + ". These are kept with their evidence but are not Key Events, "
            "and none of them is an adverse outcome.",
            icon="🔬",
        )

    with st.expander("Proposed roles and why", expanded=not verdict.get("is_aop")):
        st.caption(
            "A proposal from the graph and the biological levels, not a "
            "decision. Approve the ones you agree with in **Approve**, under "
            "*Pathway endpoints*."
        )
        st.dataframe(roles, use_container_width=True, hide_index=True)


# The MIE / Adverse Outcome editor used to live here, under the finished map.
# It has moved to **Approve**: declaring an event an adverse outcome is a
# regulatory claim and a curation decision, and the map is drawn from it — so
# being asked about it after looking at the picture was backwards. What the
# roles do to the layout is unchanged; see `_declared_roles` above, which reads
# what Approve wrote.


def _load_offsets(layout_name: str = "default") -> dict[str, float]:
    with table1_store.connect() as conn:
        rows = conn.execute(
            "SELECT node_key, y FROM layout_offset WHERE layout_name = ?",
            (layout_name,),
        ).fetchall()
    return {str(r[0]): float(r[1]) for r in rows}


def _layout_controls(placements: dict) -> None:
    """Vertical nudges only — there is deliberately no horizontal control."""
    st.subheader("Vertical arrangement")
    st.caption(
        "Nudge nodes up and down to untangle the picture. Horizontal position "
        "is causal order and is recomputed every time, so it cannot be saved."
    )

    node = st.selectbox("Key Event", sorted(placements), key="layout_node")
    placement = placements[node]
    st.caption(
        f"Currently in the **{placement.band}** column at causal depth "
        f"{placement.depth}. That position is derived from the graph."
    )

    new_y = st.slider(
        "Vertical position", 0.0, 1400.0, float(placement.y), step=10.0,
        key="layout_y",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save this position", key="layout_save"):
            with table1_store.connect() as conn:
                conn.execute(
                    "INSERT INTO layout_offset (layout_name, node_key, y, updated_at) "
                    "VALUES ('default', ?, ?, datetime('now')) "
                    "ON CONFLICT(layout_name, node_key) DO UPDATE SET "
                    "y = excluded.y, updated_at = excluded.updated_at",
                    (node, float(new_y)),
                )
                conn.commit()
            st.success("Saved.")
            st.rerun()
    with c2:
        if st.button("Reset all vertical positions", key="layout_reset"):
            with table1_store.connect() as conn:
                conn.execute("DELETE FROM layout_offset WHERE layout_name = 'default'")
                conn.commit()
            st.success("Reset.")
            st.rerun()


def _snapshot_banner() -> None:
    with table1_store.connect() as conn:
        row = conn.execute(
            "SELECT name, created_at, stale, stale_reason FROM aop_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return
    if row["stale"]:
        st.warning(
            f"The frozen snapshot “{row['name']}” ({row['created_at']}) is out "
            f"of date — {fmt(row['stale_reason'])}. Freeze a new one once you "
            f"have re-approved.",
            icon="♻️",
        )
    else:
        st.caption(f"Frozen snapshot: **{row['name']}** ({row['created_at']}).")


def _snapshot_controls(graph: nx.DiGraph, placements: dict) -> None:
    st.subheader("Snapshot")
    st.caption(
        "Freezing stores the graph as it stands now. A figure you cite should "
        "not change because new rows arrived afterwards."
    )
    name = st.text_input("Snapshot name", value="", key="snap_name",
                         placeholder="e.g. figure-2-draft")
    if st.button("Freeze snapshot", type="primary", key="snap_save",
                 disabled=not name.strip()):
        payload = _payload(graph, placements)
        with table1_store.connect() as conn:
            conn.execute(
                "INSERT INTO aop_snapshot (name, payload, content_hash, "
                "created_by, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
                (name.strip(), json.dumps(payload),
                 wf.content_hash(payload), curator_name() or None),
            )
            conn.commit()
        st.success(f"Snapshot “{name.strip()}” frozen.")
        st.rerun()


def _payload(graph: nx.DiGraph, placements: dict) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "name": name,
                "level": graph.nodes[name].get("level"),
                "role": placements[name].role,
                "band": placements[name].band,
                "column": placements[name].column,
                "depth": placements[name].depth,
            }
            for name in graph.nodes
            if name in placements
        ],
        "edges": [
            {
                "from": s, "to": t,
                "verdict": d.get("verdict"),
                "adjacency": d.get("adjacency"),
                "n_papers": d.get("n_papers"),
                # Travels with the export, or the distinction survives only on
                # screen and the CSV a reviewer receives cannot tell an
                # asserted edge from an evidenced one.
                "asserted": bool(d.get("asserted")),
                "n_curator_claims": d.get("n_curator_claims", 0),
                "n_extracted_claims": d.get("n_extracted_claims", 0),
            }
            for s, t, d in graph.edges(data=True)
        ],
    }


def _exports(graph: nx.DiGraph, placements: dict) -> None:
    payload = _payload(graph, placements)
    nodes = pd.DataFrame(payload["nodes"])
    edges = pd.DataFrame(payload["edges"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Nodes (CSV)", csv_bytes(nodes), "aop_nodes.csv",
                           "text/csv", use_container_width=True)
    with c2:
        st.download_button("Edges (CSV)", csv_bytes(edges), "aop_edges.csv",
                           "text/csv", use_container_width=True)
    with c3:
        st.download_button(
            "Graph (JSON)", json.dumps(payload, indent=2).encode("utf-8"),
            "aop_graph.json", "application/json", use_container_width=True,
        )

    # The three exports above describe the figure. None of them says how it
    # was arrived at — which papers, which wordings were folded together and
    # on whose authority, who approved what, what is still unresolved. That is
    # what a reviewer asks for, and until now it lived in nine tables with no
    # way out except the database file itself.
    st.divider()
    st.markdown("**Full Stage 2 record**")
    st.caption(
        "Everything the extraction produced and every decision made over it, "
        "in one document: corpus and run conditions, what was extracted and "
        "how much of it was verified, the crosswalk from raw wording to Key "
        "Events, curator decisions with their rationales, the approval trail, "
        "the relationships and their syntheses, and what remains outstanding."
    )
    try:
        record = stage2_report.build_stage2_report()
    except Exception as exc:  # a failed export must not take the map down
        st.warning(f"The Stage 2 record could not be assembled: {exc}")
        return

    markdown = stage2_report.report_markdown(record)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Stage 2 record (Markdown)",
            with_disclaimer(markdown).encode("utf-8"),
            "stage2_record.md", "text/markdown", use_container_width=True,
        )
    with d2:
        st.download_button(
            "Decisions and rationales (CSV)",
            with_disclaimer(stage2_report.report_csv(record)).encode("utf-8"),
            "stage2_decisions.csv", "text/csv", use_container_width=True,
        )

    with st.expander("Preview the Stage 2 record", expanded=False):
        st.markdown(markdown)
