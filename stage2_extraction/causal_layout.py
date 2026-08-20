from __future__ import annotations

"""
Causal-order layout for the approved AOP.

The old layout put a node's horizontal position on its biological level and
then let saved coordinates override both axes. Two things went wrong with that.

First, biological level is not causal order. A tissue-level event can be
upstream or downstream of a cellular one, and lanes drawn by level implied an
ordering the evidence had not established.

Second, a saved layout could move a node into the wrong column. Once a curator
dragged a late Key Event to the left of the MIE, the picture said something
false and stayed saying it — and nothing in the file recorded that the position
was a preference rather than a claim.

Here, horizontal position is *computed* from causal depth on every render and
is not persistable. Only the vertical offset is the curator's. Dragging a node
sideways is therefore not a thing the interface can do, which is the point:
the left-to-right axis is evidence, not preference.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import networkx as nx


LANE_WIDTH = 340
ROW_HEIGHT = 130

#: The bands a Key Event can occupy, left to right.
BANDS: tuple[str, ...] = ("MIE", "Early", "Intermediate", "Late", "AO")


@dataclass
class Placement:
    """Where one node sits and why."""

    node: str
    depth: int          # causal depth, 0 = molecular initiating event
    column: int         # the column actually drawn in
    band: str
    x: float
    y: float
    role: str           # "MIE" | "KE" | "AO"
    y_from_saved: bool = False


def causal_depth(graph: nx.DiGraph) -> dict[str, int]:
    """
    Longest-path depth of every node from any source.

    Longest rather than shortest path: a Key Event reachable from the MIE by
    both a two-step and a five-step route belongs after everything on the long
    route, or the picture shows an event preceding something it depends on.

    Cycles are broken by ignoring the edge that closes them. Feedback loops are
    real biology, but a cyclic graph has no causal order to draw, and dropping
    the back-edge for layout purposes keeps the loop visible as an edge while
    leaving the ordering well defined.

    The edge dropped is specifically the *last* one in the cycle as depth-first
    search found it — the back-edge, the one that returns to a node already on
    the search stack. Dropping any other edge of the cycle would break the
    forward chain instead: in a → b → c → a, removing a → b leaves b → c → a
    and the layout then draws the initiating event last.
    """
    if graph.number_of_nodes() == 0:
        return {}

    working = graph.copy()
    guard = working.number_of_edges() + 1
    while guard > 0:
        guard -= 1
        try:
            cycle = nx.find_cycle(working, orientation="original")
        except nx.NetworkXNoCycle:
            break
        edges = [(u, v) for u, v, *_ in cycle]
        if not edges:
            break
        working.remove_edge(*edges[-1])

    depth: dict[str, int] = {}
    for node in nx.topological_sort(working):
        preds = list(working.predecessors(node))
        depth[node] = 0 if not preds else max(depth[p] for p in preds) + 1

    for node in graph.nodes:
        depth.setdefault(node, 0)
    return depth


#: Levels at which an event can plausibly be an adverse outcome. An AO is a
#: consequence for the whole organism, so tissue-level findings are excluded
#: deliberately: "myelination of axons" and "decreased myelin basic protein
#: expression" are things measured in tissue, not harms an animal suffers.
_AO_LEVELS = frozenset({"Organ", "Individual", "Population"})

#: Levels at which an event can plausibly initiate a pathway.
#:
#: Retained for `mie_candidates`, which offers suggestions. It is deliberately
#: no longer consulted by `assign_roles` — see the note there on why position
#: plus level is not enough to name a molecular initiating event.
_MIE_LEVELS = frozenset({"MIE", "Molecular"})


def assign_roles(
    graph: nx.DiGraph,
    *,
    declared: Optional[dict[str, str]] = None,
    marker_nodes: Optional[set] = None,
) -> dict[str, str]:
    """
    Which nodes are the MIE, which are the AO, which are ordinary Key Events.

    A curator's declared assignment always wins. Where none exists, position
    in the graph is a starting point and not the answer.

    Position alone was the answer once, and it produced five adverse outcomes
    in one AOP: myelination of axons, CC1+ oligodendrocyte density, myelin
    basic protein expression, motor evoked potential amplitude and auditory
    hypersensitivity. Only the last is a harm. The rest are terminal for a
    dull reason — they are the last thing the papers measured — and calling
    that an adverse outcome states something about the corpus while appearing
    to state something about the biology.

    So **no node is ever inferred to be an adverse outcome**. An AO is a claim
    that harm occurred and that this pathway is why, and no arrangement of
    arrows can establish that. Most mechanistic papers never reach one, and a
    corpus of them legitimately has none — a fragment with no AO is an
    ordinary and honest result, not a gap to be filled by whichever node
    happens to be last. `ao_candidates` will suggest where one might go; only
    a curator's declaration puts it on the map.

    **The same now holds for the molecular initiating event.** It used to be
    inferred: nothing upstream of it, molecular level, therefore the MIE. Both
    halves of that are facts about the corpus rather than about the biology.
    Nothing upstream means no paper *in this collection* reported a step
    before it — which is the normal state of a corpus assembled around the
    middle of a pathway — and a molecular level says what kind of event it is,
    not that a stressor acts on it directly. Naming an MIE asserts that the
    chain starts here and that a stressor is what starts it. That is a
    curator's claim, and inferring it dressed a gap in the literature as a
    finding, in the leftmost column, in the tool's most quotable output.

    Everything is therefore a Key Event unless a curator declares otherwise.
    `mie_candidates` suggests where one might go, the same way `ao_candidates`
    does for the other end.

    A node reached only by *marker* links is still labelled a marker, since
    that follows from what the extraction recorded rather than from position.
    """
    declared = declared or {}
    marker_nodes = marker_nodes or set()
    roles: dict[str, str] = {}

    for node in graph.nodes:
        if node in declared:
            roles[node] = declared[node]
            continue

        if node in marker_nodes and graph.out_degree(node) == 0:
            roles[node] = "marker"
        else:
            roles[node] = "KE"

    return roles


def mie_candidates(graph: nx.DiGraph) -> list[dict]:
    """
    Where a molecular initiating event could sit, for a curator to consider.

    The counterpart of `ao_candidates`, and offered on the same terms: a list,
    with the reason for each, and no effect on the map until someone declares
    one. Ranked so that a molecular-level event with nothing upstream — the
    shape an MIE usually has — comes first, while being explicit that the
    shape is not the evidence.
    """
    out: list[dict] = []
    for node in graph.nodes:
        if graph.in_degree(node) != 0:
            continue
        level = str(graph.nodes[node].get("level") or "")
        plausible = level in _MIE_LEVELS
        out.append(
            {
                "ke_name": node,
                "level": level,
                "plausible": plausible,
                "why": (
                    f"Nothing in this corpus reports a step before it, and it "
                    f"sits at the {level.lower()} level. That is the shape an "
                    f"initiating event has — but it is also the shape of an "
                    f"event whose upstream paper you have not collected."
                    if plausible
                    else f"Nothing in this corpus reports a step before it, "
                         f"though at the {level.lower()} level it is more "
                         f"likely to be missing an upstream step than to be "
                         f"the start of the pathway."
                ),
            }
        )
    out.sort(key=lambda c: (not c["plausible"], c["ke_name"]))
    return out


def ao_candidates(
    graph: nx.DiGraph, *, marker_nodes: Optional[set] = None
) -> list[dict]:
    """
    Where an adverse outcome could plausibly sit, for a curator to consider.

    A suggestion, offered once, in a list — not a label painted on the map.
    Ranked so the organism-level findings come first, because that is where a
    harm is normally described.
    """
    marker_nodes = marker_nodes or set()
    out: list[dict] = []
    for node in graph.nodes:
        if graph.out_degree(node) or node in marker_nodes or not graph.in_degree(node):
            continue
        level = str(graph.nodes[node].get("level") or "")
        out.append(
            {
                "ke_name": node,
                "level": level,
                "plausible": level in _AO_LEVELS,
                "why": (
                    f"Terminal in the corpus at {level} level."
                    if level in _AO_LEVELS
                    else f"Terminal, but {level} level — more likely the last "
                         f"thing measured than a harm."
                ),
            }
        )
    return sorted(out, key=lambda c: (not c["plausible"], c["ke_name"]))


def band_for(depth: int, max_depth: int, role: str) -> str:
    """Which of the five bands a node falls in."""
    if role == "MIE":
        return "MIE"
    if role == "AO":
        return "AO"
    if max_depth <= 1:
        return "Intermediate"
    # Spread the intermediate nodes across early / intermediate / late.
    fraction = depth / max_depth
    if fraction <= 0.34:
        return "Early"
    if fraction <= 0.67:
        return "Intermediate"
    return "Late"


def layout(
    graph: nx.DiGraph,
    *,
    y_offsets: Optional[dict[str, float]] = None,
    declared_roles: Optional[dict[str, str]] = None,
    lane_width: int = LANE_WIDTH,
    row_height: int = ROW_HEIGHT,
) -> dict[str, Placement]:
    """
    Place every node.

    `y_offsets` are the curator's saved vertical nudges. There is deliberately
    no `x_offsets`: the horizontal axis is derived from the graph on every
    render, so a saved layout cannot put a node in a causal column the evidence
    does not support.
    """
    y_offsets = y_offsets or {}
    if graph.number_of_nodes() == 0:
        return {}

    depths = causal_depth(graph)
    # A node every one of whose incoming links is a marker link is a readout.
    marker_nodes = {
        node for node in graph.nodes
        if graph.in_degree(node) > 0
        and all(
            str(graph.edges[u, node].get("relation_kind") or "causal") == "marker"
            for u in graph.predecessors(node)
        )
    }
    roles = assign_roles(
        graph, declared=declared_roles, marker_nodes=marker_nodes
    )
    max_depth = max(depths.values()) if depths else 0

    # MIEs are pinned to column 0 and AOs to the last column, so the two ends
    # of the pathway read as the ends even when the graph is shallow.
    #
    # Column 0 is reserved for the MIE only when there is one. Roles are no
    # longer inferred, so most maps have none — and reserving the column
    # regardless left every such map with an empty leftmost column and its
    # first real event drawn under the "Early" header, which reads as a
    # missing initiating event rather than an undeclared one.
    has_mie = any(role == "MIE" for role in roles.values())
    first_column = 1 if has_mie else 0

    columns: dict[str, int] = {}
    for node, depth in depths.items():
        if roles[node] == "MIE":
            columns[node] = 0
        elif roles[node] == "AO":
            columns[node] = max(max_depth, 1)
        else:
            columns[node] = min(max(depth, first_column), max(max_depth, 1))

    by_column: dict[int, list[str]] = {}
    for node, column in columns.items():
        by_column.setdefault(column, []).append(node)

    # Order within a column by barycentre of already-placed predecessors, which
    # keeps edges short without moving anything between columns.
    order: dict[str, int] = {}
    for column in sorted(by_column):
        nodes = by_column[column]
        if column == 0:
            nodes.sort()
        else:
            def barycentre(n: str) -> tuple[float, str]:
                preds = [order[p] for p in graph.predecessors(n) if p in order]
                return (sum(preds) / len(preds) if preds else 1e6, n)
            nodes.sort(key=barycentre)
        for row, node in enumerate(nodes):
            order[node] = row

    tallest = max((len(nodes) for nodes in by_column.values()), default=1)

    placements: dict[str, Placement] = {}
    for column in sorted(by_column):
        nodes = by_column[column]
        centring = (tallest - len(nodes)) / 2.0
        for row, node in enumerate(nodes):
            saved_y = y_offsets.get(node)
            y = (
                float(saved_y)
                if saved_y is not None
                else float((row + centring) * row_height)
            )
            placements[node] = Placement(
                node=node,
                depth=depths[node],
                column=column,
                band=band_for(depths[node], max_depth, roles[node]),
                x=float(column * lane_width),
                y=y,
                role=roles[node],
                y_from_saved=saved_y is not None,
            )

    return placements


def column_headers(placements: dict[str, Placement]) -> list[tuple[int, str, float]]:
    """Band labels for each occupied column, for the header row of the canvas."""
    seen: dict[int, tuple[str, float]] = {}
    for placement in placements.values():
        if placement.column not in seen:
            seen[placement.column] = (placement.band, placement.x)
    return [(column, band, x) for column, (band, x) in sorted(seen.items())]


def validate(placements: dict[str, Placement], graph: nx.DiGraph) -> list[str]:
    """
    Check that the drawing does not contradict the graph.

    Run after layout as a self-check. Every edge should point rightwards or
    stay in the same column; an edge pointing left means the picture is making
    a causal claim backwards, which is the failure the redesign is guarding
    against.
    """
    problems: list[str] = []
    for source, target in graph.edges:
        if source not in placements or target not in placements:
            continue
        if placements[target].column < placements[source].column:
            problems.append(
                f"“{source}” is drawn to the right of “{target}” despite "
                f"pointing at it."
            )
    return problems


__all__ = [
    "BANDS",
    "LANE_WIDTH",
    "ROW_HEIGHT",
    "Placement",
    "causal_depth",
    "assign_roles",
    "band_for",
    "layout",
    "column_headers",
    "validate",
]
