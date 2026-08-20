from __future__ import annotations

"""
Layered AOP pathway visualization.

The previous visualizer used a force-directed layout, which is the wrong tool
for an AOP. A force layout places nodes wherever the physics settles, so the
biological level of organisation — the single most important axis of an AOP —
was invisible, and the map re-flowed differently on every page load.

This module lays the pathway out on a fixed grid instead:

    MIE → Molecular → Cellular → Tissue → Organ → Individual → Population
     x=0     x=1         x=2       x=3      x=4       x=5         x=6

Left-to-right position now *means* something. Within each lane, nodes are
ordered by iterated barycentre sweeps, the standard Sugiyama-style heuristic
for reducing edge crossings. The result is deterministic: the same graph always
produces the same picture, which is what makes a curated layout worth saving.

Readability features
--------------------
* Adjacent-only by default — non-adjacent and long-range edges are hidden until
  asked for, because they are what turns an AOP map into spaghetti.
* Overview versus detail — the principal backbone is shown first; peripheral
  mechanisms are one toggle away.
* Saved coordinates win — a node the curator has placed stays where it was put.
"""

import json
import math
from typing import Any, Iterable, Optional, Sequence

import networkx as nx
import pandas as pd

from schemas import KE_LEVEL_ORDER, level_index

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

KE_LEVEL_COLORS: dict[str, str] = {
    "MIE": "#e74c3c",
    "Molecular": "#3498db",
    "Cellular": "#2ecc71",
    "Tissue": "#f39c12",
    "Organ": "#9b59b6",
    "Individual": "#1abc9c",
    "Population": "#34495e",
}

#: Softer versions of the level colours, used for the lane background bands.
KE_LEVEL_BANDS: dict[str, str] = {
    "MIE": "#fdeceb",
    "Molecular": "#ebf3fb",
    "Cellular": "#eafaf0",
    "Tissue": "#fef5e7",
    "Organ": "#f5eef8",
    "Individual": "#e8f8f5",
    "Population": "#eceff1",
}

EDGE_COLORS = {
    "supported": "#27ae60",
    "contradicted": "#e74c3c",
    "mixed": "#f39c12",
    "rejected": "#bdc3c7",
}

LANE_WIDTH = 340
ROW_HEIGHT = 130


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_pathway_graph(
    table2_df: pd.DataFrame,
    *,
    physio_links: Optional[dict[int, list[dict]]] = None,
    canonical_lookup: Optional[dict[str, dict]] = None,
) -> nx.DiGraph:
    """
    Build a directed graph from Table 2.

    Nodes are Key Events carrying their level, ontology annotation and any
    physiological-map links. Edges are consolidated KERs carrying the full
    evidence payload needed by the edge-level evidence panel, so the panel can
    render without another database round-trip.
    """
    G = nx.DiGraph()
    if table2_df is None or table2_df.empty:
        return G

    physio_links = physio_links or {}
    canonical_lookup = canonical_lookup or {}

    # --- Nodes -------------------------------------------------------------
    node_levels: dict[str, list[str]] = {}
    node_canonical_ids: dict[str, Any] = {}

    for _, row in table2_df.iterrows():
        for name_col, level_col, id_col in (
            ("upstream_ke_name", "upstream_ke_level", "upstream_ke_canonical_id"),
            ("downstream_ke_name", "downstream_ke_level", "downstream_ke_canonical_id"),
        ):
            name = row.get(name_col)
            if pd.isna(name) or not str(name).strip():
                continue
            key = str(name).strip()
            level = str(row.get(level_col) or "Molecular").strip()
            node_levels.setdefault(key, []).append(level)
            canonical_id = row.get(id_col)
            if pd.notna(canonical_id) and key not in node_canonical_ids:
                try:
                    node_canonical_ids[key] = int(float(canonical_id))
                except (TypeError, ValueError):
                    pass

    for name, levels in node_levels.items():
        # Ties break upstream, so a KE never drifts rightward between sessions.
        level = min(set(levels), key=lambda lv: (-levels.count(lv), level_index(lv)))
        meta = canonical_lookup.get(name, {})
        canonical_id = node_canonical_ids.get(name)
        links = physio_links.get(canonical_id, []) if canonical_id is not None else []

        G.add_node(
            name,
            label=name,
            level=level,
            lane=level_index(level),
            color=KE_LEVEL_COLORS.get(level, "#95a5a6"),
            canonical_id=canonical_id,
            ontology_curie=meta.get("ontology_curie"),
            ontology_label=meta.get("ontology_label"),
            ontology_iri=meta.get("ontology_iri"),
            ontology_score=meta.get("ontology_score", 0.0),
            aliases=meta.get("aliases", ""),
            n_aliases=meta.get("n_aliases", 0),
            curation_status=meta.get("curation_status", "unreviewed"),
            physio_links=links,
        )

    # --- Edges -------------------------------------------------------------
    for _, row in table2_df.iterrows():
        up = str(row.get("upstream_ke_name", "")).strip()
        down = str(row.get("downstream_ke_name", "")).strip()
        if not up or not down or up not in G.nodes or down not in G.nodes:
            continue
        if up == down:
            continue  # self-loops carry no pathway information

        n_support = _as_int(row.get("n_papers_supporting"))
        n_contra = _as_int(row.get("n_papers_contradicting"))

        if n_support > 0 and n_contra == 0:
            state = "supported"
        elif n_contra > 0 and n_support == 0:
            state = "contradicted"
        else:
            state = "mixed"

        curation_status = str(row.get("curation_status") or "unreviewed")
        adjacency = str(row.get("ker_adjacency") or "Adjacent")
        lane_gap = abs(G.nodes[down]["lane"] - G.nodes[up]["lane"])

        # An edge whose papers disagree about the sign is drawn as contested
        # regardless of how many papers it has. Counting those papers as
        # support is what let a conflicted edge look like a strong one.
        direction = str(row.get("direction") or "unclear")
        if bool(row.get("sign_conflict")):
            state = "mixed"

        G.add_edge(
            up,
            down,
            ker_key=row.get("ker_key"),
            ker_name=row.get("ker_name") or f"{up} leads to {down}",
            ker_description=row.get("ker_description"),
            adjacency=adjacency,
            is_adjacent=(adjacency == "Adjacent" and lane_gap <= 1),
            lane_gap=lane_gap,
            is_backward=G.nodes[down]["lane"] < G.nodes[up]["lane"],
            state=state,
            color=EDGE_COLORS["rejected"] if curation_status == "rejected" else EDGE_COLORS[state],
            width=min(9.0, max(1.6, 1.6 + n_support * 0.7)),
            n_supporting=n_support,
            n_contradicting=n_contra,
            n_papers=_as_int(row.get("n_papers_total")),
            confidence_score=_as_float(row.get("confidence_score")),
            confidence_band=row.get("confidence_band") or "Low",
            uncertainty_level=row.get("uncertainty_level") or "Low",
            aop_status=row.get("aop_status") or "novel",
            aop_id=row.get("aop_id"),
            curation_status=curation_status,
            n_evidence_spans=_as_int(row.get("n_evidence_spans")),
            n_verified_spans=_as_int(row.get("n_verified_spans")),
            all_source_dois=row.get("all_source_dois"),
            taxa=row.get("all_taxa"),
            direction=direction,
            sign_conflict=bool(row.get("sign_conflict")),
            n_positive=_as_int(row.get("n_positive")),
            n_negative=_as_int(row.get("n_negative")),
            cell_types=row.get("cell_types") or "",
            title=_edge_tooltip(row, n_support, n_contra),
        )

    return G


def _as_int(value) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


#: How each aggregated sign reads on an edge label. "Conflicting" is spelled
#: out rather than shown as a symbol because it is the one a reader must not
#: skim past.
_DIRECTION_LABELS = {
    "positive": "same direction (↑↑ / ↓↓)",
    "negative": "opposite directions (↑↓)",
    "conflicting": "CONFLICTING — papers disagree on the sign",
    "none": "no coupling observed",
    "unclear": "direction not stated",
}


def _edge_tooltip(row: pd.Series, n_support: int, n_contra: int) -> str:
    bits = [str(row.get("ker_name") or "")]
    bits.append(f"Supporting: {n_support}   Contradicting: {n_contra}")

    direction = str(row.get("direction") or "unclear")
    label = _DIRECTION_LABELS.get(direction, direction)
    if direction == "conflicting":
        bits.append(
            f"Sign: {label} "
            f"({_as_int(row.get('n_positive'))} positive, "
            f"{_as_int(row.get('n_negative'))} negative)"
        )
    else:
        bits.append(f"Sign: {label}")

    cells = str(row.get("cell_types") or "").strip()
    if cells:
        bits.append(f"Cell types: {cells}")
    band = row.get("confidence_band")
    score = row.get("confidence_score")
    if band is not None:
        bits.append(f"Confidence: {band} ({score})")
    spans = _as_int(row.get("n_evidence_spans"))
    verified = _as_int(row.get("n_verified_spans"))
    if spans:
        bits.append(f"Quotations: {verified}/{spans} verified")
    if row.get("aop_status") == "existing":
        bits.append(f"AOP-Wiki: {row.get('aop_id') or 'listed'}")
    return "\n".join(b for b in bits if b)


# ---------------------------------------------------------------------------
# Layered layout
# ---------------------------------------------------------------------------

def _initial_order(graph: nx.DiGraph) -> dict[int, list[str]]:
    """Group nodes into lanes, seeded in a stable order."""
    lanes: dict[int, list[str]] = {}
    for node, attrs in graph.nodes(data=True):
        lanes.setdefault(attrs.get("lane", 1), []).append(node)
    for lane in lanes.values():
        lane.sort()
    return lanes


def _barycentre_sweep(
    graph: nx.DiGraph,
    lanes: dict[int, list[str]],
    *,
    iterations: int = 6,
) -> dict[int, list[str]]:
    """
    Reduce edge crossings by repeated barycentre ordering.

    Each node is repositioned to the average position of its neighbours in the
    adjacent lane, alternating forward and backward passes. This is the classic
    Sugiyama layered-graph heuristic; it is not optimal (crossing minimisation
    is NP-hard) but it removes the great majority of avoidable crossings for a
    fraction of the cost.
    """
    lane_keys = sorted(lanes)
    if len(lane_keys) < 2:
        return lanes

    positions = {
        node: idx for lane in lane_keys for idx, node in enumerate(lanes[lane])
    }

    for iteration in range(iterations):
        forward = iteration % 2 == 0
        ordered_lanes = lane_keys[1:] if forward else list(reversed(lane_keys[:-1]))

        for lane_key in ordered_lanes:
            def barycentre(node: str) -> float:
                neighbours = (
                    list(graph.predecessors(node)) if forward else list(graph.successors(node))
                )
                relevant = [
                    positions[n]
                    for n in neighbours
                    if n in positions
                    and graph.nodes[n].get("lane") == (lane_key - 1 if forward else lane_key + 1)
                ]
                if not relevant:
                    return float(positions.get(node, 0))
                return sum(relevant) / len(relevant)

            lanes[lane_key].sort(key=lambda n: (barycentre(n), n))
            for idx, node in enumerate(lanes[lane_key]):
                positions[node] = idx

    return lanes


def compute_layered_positions(
    graph: nx.DiGraph,
    *,
    saved_positions: Optional[dict[str, dict]] = None,
    lane_width: int = LANE_WIDTH,
    row_height: int = ROW_HEIGHT,
    respect_saved: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Assign an (x, y) to every node.

    x is fixed by biological level, so the lanes read MIE→Population from left
    to right. y comes from crossing-reduced ordering within the lane, centred
    vertically so lanes of different sizes stay visually balanced.

    Saved coordinates take precedence when `respect_saved` is True, which is
    what makes a curated layout persistent: only genuinely new nodes are
    positioned automatically, and everything the curator placed stays put.
    """
    saved_positions = saved_positions or {}

    lanes = _barycentre_sweep(graph, _initial_order(graph))
    max_rows = max((len(nodes) for nodes in lanes.values()), default=1)

    positions: dict[str, dict[str, float]] = {}

    for lane_key in sorted(lanes):
        nodes = lanes[lane_key]
        # Centre this lane against the tallest lane.
        offset = (max_rows - len(nodes)) / 2.0
        for row, node in enumerate(nodes):
            saved = saved_positions.get(node) if respect_saved else None
            if saved and saved.get("x") is not None and saved.get("y") is not None:
                positions[node] = {
                    "x": float(saved["x"]),
                    "y": float(saved["y"]),
                    "pinned": bool(saved.get("pinned", False)),
                    "from_saved": True,
                }
            else:
                positions[node] = {
                    "x": float(lane_key * lane_width),
                    "y": float((row + offset) * row_height),
                    "pinned": False,
                    "from_saved": False,
                }

    return positions


# ---------------------------------------------------------------------------
# Overview / detail and adjacency filtering
# ---------------------------------------------------------------------------

def find_backbone(
    graph: nx.DiGraph,
    *,
    max_paths: int = 3,
) -> tuple[set[str], set[tuple[str, str]]]:
    """
    Identify the principal AOP backbone.

    The backbone is the highest-weight chain (or chains) running from an
    upstream source toward the most downstream sink, scoring each candidate
    path by the evidence behind its edges and how far right it travels. This is
    what the overview should show first: the spine of the pathway, not every
    peripheral mechanism the corpus happens to mention.

    Returns (backbone_nodes, backbone_edges).
    """
    if graph.number_of_nodes() == 0:
        return set(), set()

    # Work on a DAG view: dropping backward edges keeps path enumeration finite
    # even when the corpus contains feedback loops.
    dag = nx.DiGraph()
    dag.add_nodes_from(graph.nodes(data=True))
    for u, v, attrs in graph.edges(data=True):
        if not attrs.get("is_backward"):
            dag.add_edge(u, v, **attrs)

    while not nx.is_directed_acyclic_graph(dag):
        try:
            cycle = nx.find_cycle(dag, orientation="original")
        except nx.NetworkXNoCycle:
            break
        weakest = min(
            cycle,
            key=lambda e: dag.edges[e[0], e[1]].get("confidence_score", 0.0),
        )
        dag.remove_edge(weakest[0], weakest[1])

    sources = [n for n in dag.nodes if dag.in_degree(n) == 0]
    sinks = [n for n in dag.nodes if dag.out_degree(n) == 0]
    if not sources or not sinks:
        return set(graph.nodes), set(graph.edges)

    def edge_weight(u: str, v: str) -> float:
        attrs = dag.edges[u, v]
        support = attrs.get("n_supporting", 0)
        confidence = attrs.get("confidence_score", 0.0)
        penalty = 0.5 if attrs.get("adjacency") == "Non-adjacent" else 0.0
        return 1.0 + confidence + math.log1p(support) - penalty

    scored_paths: list[tuple[float, list[str]]] = []
    for source in sources:
        # Longest-path-style DP over the DAG from this source.
        best_score: dict[str, float] = {source: 0.0}
        best_prev: dict[str, Optional[str]] = {source: None}
        for node in nx.topological_sort(dag):
            if node not in best_score:
                continue
            for successor in dag.successors(node):
                candidate = best_score[node] + edge_weight(node, successor)
                if candidate > best_score.get(successor, float("-inf")):
                    best_score[successor] = candidate
                    best_prev[successor] = node

        for sink in sinks:
            if sink not in best_score or sink == source:
                continue
            path: list[str] = []
            cursor: Optional[str] = sink
            while cursor is not None:
                path.append(cursor)
                cursor = best_prev.get(cursor)
            path.reverse()
            if len(path) < 2:
                continue
            # Reward paths that traverse more biological levels.
            span = dag.nodes[sink].get("lane", 0) - dag.nodes[source].get("lane", 0)
            scored_paths.append((best_score[sink] + 0.75 * span, path))

    if not scored_paths:
        return set(graph.nodes), set(graph.edges)

    scored_paths.sort(key=lambda item: item[0], reverse=True)

    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for _, path in scored_paths[:max_paths]:
        nodes.update(path)
        edges.update(zip(path, path[1:]))

    return nodes, edges


def filter_graph(
    graph: nx.DiGraph,
    *,
    adjacent_only: bool = True,
    max_lane_gap: int = 1,
    overview_only: bool = False,
    min_confidence: float = 0.0,
    min_supporting_papers: int = 0,
    include_contradicted: bool = True,
    hide_rejected: bool = True,
    hide_isolated: bool = True,
) -> tuple[nx.DiGraph, dict[str, int]]:
    """
    Produce the subgraph to display, plus a count of what was hidden.

    `adjacent_only` is the default because hiding non-adjacent and long-range
    relationships is by far the most effective single measure against edge
    crossings: those edges span multiple lanes and cut across every node in
    between. They remain in the underlying data and one toggle away.

    Returns (filtered_graph, hidden_counts).
    """
    hidden = {
        "non_adjacent": 0,
        "long_range": 0,
        "backward": 0,
        "low_confidence": 0,
        "few_papers": 0,
        "contradicted": 0,
        "rejected": 0,
        "off_backbone": 0,
        "isolated_nodes": 0,
    }

    backbone_nodes, backbone_edges = (
        find_backbone(graph) if overview_only else (set(graph.nodes), set(graph.edges))
    )

    out = nx.DiGraph()
    out.add_nodes_from(graph.nodes(data=True))

    for u, v, attrs in graph.edges(data=True):
        if hide_rejected and attrs.get("curation_status") == "rejected":
            hidden["rejected"] += 1
            continue
        if overview_only and (u, v) not in backbone_edges:
            hidden["off_backbone"] += 1
            continue
        if adjacent_only and attrs.get("adjacency") == "Non-adjacent":
            hidden["non_adjacent"] += 1
            continue
        if adjacent_only and attrs.get("lane_gap", 0) > max_lane_gap:
            hidden["long_range"] += 1
            continue
        if adjacent_only and attrs.get("is_backward"):
            hidden["backward"] += 1
            continue
        if attrs.get("confidence_score", 0.0) < min_confidence:
            hidden["low_confidence"] += 1
            continue
        if attrs.get("n_supporting", 0) < min_supporting_papers:
            hidden["few_papers"] += 1
            continue
        if not include_contradicted and attrs.get("state") == "contradicted":
            hidden["contradicted"] += 1
            continue
        out.add_edge(u, v, **attrs)

    if overview_only:
        for node in list(out.nodes):
            if node not in backbone_nodes:
                out.remove_node(node)

    if hide_isolated:
        isolated = [n for n in out.nodes if out.degree(n) == 0]
        hidden["isolated_nodes"] = len(isolated)
        out.remove_nodes_from(isolated)

    return out, hidden


def expandable_neighbourhood(
    full_graph: nx.DiGraph,
    visible_graph: nx.DiGraph,
) -> dict[str, dict[str, int]]:
    """
    For each visible node, count what is hidden immediately around it.

    This drives the "+3" affordance on a node: the curator can see at a glance
    which parts of the overview have detail folded underneath them.
    """
    out: dict[str, dict[str, int]] = {}
    for node in visible_graph.nodes:
        if node not in full_graph:
            continue
        hidden_in = sum(
            1 for p in full_graph.predecessors(node) if not visible_graph.has_edge(p, node)
        )
        hidden_out = sum(
            1 for s in full_graph.successors(node) if not visible_graph.has_edge(node, s)
        )
        if hidden_in or hidden_out:
            out[node] = {
                "hidden_upstream": hidden_in,
                "hidden_downstream": hidden_out,
                "total": hidden_in + hidden_out,
            }
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _wrap_label(text: str, width: int = 22, max_lines: int = 4) -> str:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > sum(len(l) for l in lines):
        lines[-1] = lines[-1][: width - 1] + "…"
    return "\n".join(lines) if lines else str(text)


def render_interactive_graph(
    graph: nx.DiGraph,
    *,
    positions: Optional[dict[str, dict[str, float]]] = None,
    height: int = 820,
    show_lanes: bool = True,
    expandable: Optional[dict[str, dict[str, int]]] = None,
    physics: bool = False,
) -> str:
    """
    Render the graph as a self-contained interactive HTML document.

    Physics is off by default: positions are computed server-side so the layout
    is reproducible and can be saved. Dragging a node still works — the layout
    panel exports the new coordinates so they can be persisted.
    """
    if graph.number_of_nodes() == 0:
        return (
            "<p style='font-family:system-ui;padding:1rem;color:#555'>"
            "Nothing to display. Either no KERs match the current filters, or "
            "Table 1 is still empty.</p>"
        )

    positions = positions or compute_layered_positions(graph)
    expandable = expandable or {}

    nodes: list[dict] = []
    for node_id, attrs in graph.nodes(data=True):
        pos = positions.get(node_id, {"x": 0.0, "y": 0.0, "pinned": False})
        extra = expandable.get(node_id)
        badge = f"  ⊕{extra['total']}" if extra else ""

        tooltip_lines = [str(node_id), f"Level: {attrs.get('level', 'Unknown')}"]
        if attrs.get("ontology_curie"):
            tooltip_lines.append(
                f"Ontology: {attrs.get('ontology_label')} ({attrs['ontology_curie']})"
            )
        if attrs.get("n_aliases"):
            tooltip_lines.append(f"Merged from {attrs['n_aliases']} raw label(s)")
        if extra:
            tooltip_lines.append(
                f"Hidden: {extra['hidden_upstream']} upstream, "
                f"{extra['hidden_downstream']} downstream"
            )
        if attrs.get("physio_links"):
            tooltip_lines.append(
                "Physiological map: "
                + ", ".join(l["entity_label"] for l in attrs["physio_links"][:2])
            )

        status = attrs.get("curation_status", "unreviewed")
        border = {"accepted": "#1e8449", "rejected": "#922b21"}.get(status, "#2c3e50")

        nodes.append(
            {
                "id": str(node_id),
                "label": _wrap_label(node_id) + badge,
                "title": "\n".join(tooltip_lines),
                "x": pos["x"],
                "y": pos["y"],
                "fixed": {"x": bool(pos.get("pinned")), "y": bool(pos.get("pinned"))},
                "level": attrs.get("lane", 1),
                "shape": "box",
                "color": {
                    "background": attrs.get("color", "#95a5a6"),
                    "border": border,
                    "highlight": {"background": "#ffffff", "border": "#f1c40f"},
                },
                "borderWidth": 3 if status != "unreviewed" else 1.5,
                "borderWidthSelected": 4,
                "font": {"size": 13, "color": "#1b2631", "face": "system-ui", "multi": False},
                "margin": {"top": 10, "bottom": 10, "left": 12, "right": 12},
                "widthConstraint": {"minimum": 150, "maximum": 210},
                "meta": {
                    "level": attrs.get("level"),
                    "ontology_curie": attrs.get("ontology_curie"),
                    "ontology_label": attrs.get("ontology_label"),
                    "aliases": attrs.get("aliases"),
                    "physio": attrs.get("physio_links", []),
                    "curation_status": status,
                },
            }
        )

    edges: list[dict] = []
    for source, target, attrs in graph.edges(data=True):
        dashed = attrs.get("adjacency") == "Non-adjacent" or attrs.get("is_backward")
        edges.append(
            {
                "id": str(attrs.get("ker_key") or f"{source}->{target}"),
                "from": str(source),
                "to": str(target),
                "title": attrs.get("title", ""),
                "color": {
                    "color": attrs.get("color", "#999999"),
                    "highlight": "#f1c40f",
                    "opacity": 0.55 if dashed else 0.9,
                },
                "width": attrs.get("width", 2),
                "dashes": bool(dashed),
                "smooth": {"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.45},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
                "meta": {
                    "ker_key": attrs.get("ker_key"),
                    "ker_name": attrs.get("ker_name"),
                    "description": attrs.get("ker_description"),
                    "adjacency": attrs.get("adjacency"),
                    "n_supporting": attrs.get("n_supporting", 0),
                    "n_contradicting": attrs.get("n_contradicting", 0),
                    "confidence_band": attrs.get("confidence_band"),
                    "confidence_score": attrs.get("confidence_score"),
                    "uncertainty": attrs.get("uncertainty_level"),
                    "aop_status": attrs.get("aop_status"),
                    "aop_id": attrs.get("aop_id"),
                    "spans": attrs.get("n_evidence_spans", 0),
                    "verified": attrs.get("n_verified_spans", 0),
                    "dois": attrs.get("all_source_dois"),
                    "taxa": attrs.get("taxa"),
                },
            }
        )

    lanes_present = sorted({attrs.get("lane", 1) for _, attrs in graph.nodes(data=True)})
    lane_meta = [
        {
            "index": lane,
            "name": KE_LEVEL_ORDER[lane] if lane < len(KE_LEVEL_ORDER) else "Other",
            "color": KE_LEVEL_BANDS.get(
                KE_LEVEL_ORDER[lane] if lane < len(KE_LEVEL_ORDER) else "", "#f4f6f6"
            ),
            "x": lane * LANE_WIDTH,
        }
        for lane in lanes_present
    ]

    return _HTML_TEMPLATE.format(
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
        lanes_json=json.dumps(lane_meta if show_lanes else []),
        lane_width=LANE_WIDTH,
        physics_json=json.dumps(bool(physics)),
        height=height,
    )


_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body {{ margin:0; padding:0; height:100%; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
    #wrap {{ position:relative; width:100%; height:{height}px; }}
    #network {{ width:100%; height:100%; background:#ffffff; border:1px solid #dfe4e8; border-radius:6px; }}
    #panel {{
      position:absolute; top:12px; right:12px; width:310px; max-height:calc(100% - 90px);
      overflow-y:auto; background:rgba(255,255,255,.97); border:1px solid #d5dbdb;
      border-radius:8px; padding:12px 14px; font-size:12.5px; line-height:1.45;
      box-shadow:0 2px 12px rgba(0,0,0,.10); display:none;
    }}
    #panel h4 {{ margin:0 0 8px; font-size:13.5px; color:#1b2631; }}
    #panel .row {{ display:flex; justify-content:space-between; gap:10px; padding:3px 0; border-bottom:1px solid #f0f3f4; }}
    #panel .row span:first-child {{ color:#5d6d7e; }}
    #panel .row span:last-child {{ text-align:right; font-weight:600; color:#212f3d; }}
    #panel .desc {{ margin-top:8px; color:#34495e; font-style:italic; }}
    #panel .close {{ position:absolute; top:8px; right:10px; cursor:pointer; color:#85929e; font-size:16px; }}
    #toolbar {{
      position:absolute; bottom:12px; left:12px; background:rgba(255,255,255,.96);
      border:1px solid #d5dbdb; border-radius:8px; padding:8px 10px; font-size:12px;
      display:flex; gap:8px; align-items:center; box-shadow:0 2px 8px rgba(0,0,0,.08);
    }}
    #toolbar button {{
      font:inherit; padding:5px 10px; border:1px solid #aeb6bf; background:#fff;
      border-radius:5px; cursor:pointer; color:#1b2631;
    }}
    #toolbar button:hover {{ background:#eaf2f8; border-color:#5499c7; }}
    #legend {{
      position:absolute; bottom:12px; right:12px; background:rgba(255,255,255,.96);
      border:1px solid #d5dbdb; border-radius:8px; padding:8px 12px; font-size:11.5px;
      box-shadow:0 2px 8px rgba(0,0,0,.08);
    }}
    #legend div {{ display:flex; align-items:center; gap:6px; padding:1.5px 0; }}
    #legend i {{ width:22px; height:3px; display:inline-block; border-radius:2px; }}
    #export {{
      position:absolute; top:12px; left:12px; width:330px; display:none;
      background:rgba(255,255,255,.98); border:1px solid #d5dbdb; border-radius:8px;
      padding:10px 12px; font-size:12px; box-shadow:0 2px 12px rgba(0,0,0,.10);
    }}
    #export textarea {{ width:100%; height:120px; font-family:ui-monospace,Menlo,monospace; font-size:10.5px; }}
    .hint {{ color:#7f8c8d; font-size:11px; margin-top:4px; }}
  </style>
</head>
<body>
<div id="wrap">
  <div id="network"></div>

  <div id="panel">
    <span class="close" onclick="document.getElementById('panel').style.display='none'">&times;</span>
    <div id="panelBody"></div>
  </div>

  <div id="export">
    <b>Saved node positions</b>
    <div class="hint">Copy this and paste it into the "Persist curated layout" box below the map.</div>
    <textarea id="exportText" readonly></textarea>
    <div style="margin-top:6px; display:flex; gap:6px;">
      <button onclick="copyPositions()">Copy</button>
      <button onclick="document.getElementById('export').style.display='none'">Close</button>
    </div>
  </div>

  <div id="toolbar">
    <button onclick="fitView()">Fit</button>
    <button onclick="showPositions()">Export positions</button>
    <button onclick="togglePhysics()">Toggle physics</button>
    <span id="status" style="color:#7f8c8d"></span>
  </div>

  <div id="legend">
    <div><i style="background:#27ae60"></i> supported</div>
    <div><i style="background:#f39c12"></i> mixed evidence</div>
    <div><i style="background:#e74c3c"></i> contradicted</div>
    <div><i style="background:#95a5a6; border-top:2px dashed #95a5a6; height:0"></i> non-adjacent</div>
  </div>
</div>

<script>
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var lanes = {lanes_json};
  var laneWidth = {lane_width};
  var physicsOn = {physics_json};

  var container = document.getElementById('network');
  var data = {{ nodes: nodes, edges: edges }};

  var options = {{
    physics: physicsOn ? {{
      enabled: true,
      barnesHut: {{ gravitationalConstant: -8000, springLength: 180, avoidOverlap: 0.4 }},
      stabilization: {{ iterations: 150 }}
    }} : {{ enabled: false }},
    interaction: {{
      hover: true, navigationButtons: true, keyboard: true,
      tooltipDelay: 150, multiselect: false, dragNodes: true
    }},
    nodes: {{ shadow: {{ enabled: true, size: 5, x: 1, y: 2, color: 'rgba(0,0,0,.12)' }} }},
    edges: {{ selectionWidth: 2.5, hoverWidth: 1.4 }}
  }};

  var network = new vis.Network(container, data, options);

  // Lane bands and headers, drawn behind the graph so the biological level
  // axis is readable at a glance.
  network.on('beforeDrawing', function (ctx) {{
    if (!lanes.length) return;
    var bbox = network.getViewPosition();
    var scale = network.getScale();
    var h = container.clientHeight / scale;
    var top = bbox.y - h;
    lanes.forEach(function (lane) {{
      ctx.fillStyle = lane.color;
      ctx.fillRect(lane.x - laneWidth / 2, top, laneWidth, h * 2);
      ctx.save();
      ctx.fillStyle = '#5d6d7e';
      ctx.font = '600 15px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(lane.name.toUpperCase(), lane.x, top + 26);
      ctx.restore();
    }});
  }});

  function fmt(v) {{ return (v === null || v === undefined || v === '') ? '—' : v; }}

  function rowHtml(k, v) {{
    return '<div class="row"><span>' + k + '</span><span>' + fmt(v) + '</span></div>';
  }}

  network.on('selectEdge', function (params) {{
    if (!params.edges.length) return;
    var e = edges.get(params.edges[0]);
    if (!e || !e.meta) return;
    var m = e.meta;
    var html = '<h4>' + fmt(m.ker_name) + '</h4>';
    html += rowHtml('Confidence', fmt(m.confidence_band) + ' (' + fmt(m.confidence_score) + ')');
    html += rowHtml('Supporting papers', m.n_supporting);
    html += rowHtml('Contradicting papers', m.n_contradicting);
    html += rowHtml('Uncertainty', m.uncertainty);
    html += rowHtml('Adjacency', m.adjacency);
    html += rowHtml('Quotations verified', m.verified + ' / ' + m.spans);
    html += rowHtml('AOP-Wiki', m.aop_status === 'existing' ? fmt(m.aop_id) : 'novel');
    html += rowHtml('Taxa', m.taxa);
    if (m.description) html += '<div class="desc">' + m.description + '</div>';
    if (m.dois) html += '<div class="desc">Sources: ' + m.dois + '</div>';
    html += '<div class="hint">Select this KER in the evidence panel below the map to read its quotations.</div>';
    document.getElementById('panelBody').innerHTML = html;
    document.getElementById('panel').style.display = 'block';
  }});

  network.on('selectNode', function (params) {{
    if (!params.nodes.length) return;
    var n = nodes.get(params.nodes[0]);
    if (!n || !n.meta) return;
    var m = n.meta;
    var html = '<h4>' + n.id + '</h4>';
    html += rowHtml('Level', m.level);
    html += rowHtml('Curation', m.curation_status);
    if (m.ontology_curie) html += rowHtml('Ontology', m.ontology_label + ' (' + m.ontology_curie + ')');
    if (m.aliases) html += '<div class="desc">Merged labels: ' + m.aliases + '</div>';
    if (m.physio && m.physio.length) {{
      html += '<div class="desc">Physiological map:<br>';
      m.physio.forEach(function (p) {{
        html += '<a href="' + p.url + '" target="_blank" rel="noopener">' +
                p.provider_label + ': ' + p.entity_label + '</a><br>';
      }});
      html += '</div>';
    }}
    document.getElementById('panelBody').innerHTML = html;
    document.getElementById('panel').style.display = 'block';
  }});

  network.on('deselectEdge', function () {{ document.getElementById('panel').style.display = 'none'; }});
  network.on('deselectNode', function () {{ document.getElementById('panel').style.display = 'none'; }});

  function currentPositions() {{
    var raw = network.getPositions();
    return Object.keys(raw).map(function (id) {{
      var n = nodes.get(id);
      return {{
        node_key: id,
        x: Math.round(raw[id].x),
        y: Math.round(raw[id].y),
        lane: n && n.meta ? n.meta.level : null,
        pinned: true
      }};
    }});
  }}

  function showPositions() {{
    document.getElementById('exportText').value =
      JSON.stringify({{ positions: currentPositions() }}, null, 1);
    document.getElementById('export').style.display = 'block';
  }}

  function copyPositions() {{
    var ta = document.getElementById('exportText');
    ta.select();
    try {{
      document.execCommand('copy');
      document.getElementById('status').textContent = 'positions copied';
      setTimeout(function () {{ document.getElementById('status').textContent = ''; }}, 2500);
    }} catch (err) {{
      document.getElementById('status').textContent = 'press Ctrl+C to copy';
    }}
  }}

  function fitView() {{ network.fit({{ animation: {{ duration: 300 }} }}); }}

  function togglePhysics() {{
    physicsOn = !physicsOn;
    network.setOptions({{ physics: {{ enabled: physicsOn }} }});
    document.getElementById('status').textContent = physicsOn ? 'physics on' : 'physics off';
  }}

  network.once('afterDrawing', function () {{ network.fit({{ animation: false }}); }});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Pathway analysis
# ---------------------------------------------------------------------------

def _papers_on(graph: nx.DiGraph, u: str, v: str) -> set[str]:
    """The set of source DOIs behind one edge."""
    raw = graph.edges[u, v].get("all_source_dois") or ""
    return {p.strip() for p in str(raw).split(";") if p.strip()}


def chain_provenance(graph: nx.DiGraph, path: Sequence[str]) -> dict[str, Any]:
    """
    Describe how much of a chain any single paper actually saw.

    A path through a graph is not a finding. Where two consecutive edges come
    from disjoint sets of papers, the junction between them is something this
    tool inferred by matching two labels, not something a study observed — and
    if the two studies used different cell types, the inference is very likely
    wrong. Reporting such a path as a "mechanistic chain" is how a corpus on
    oligodendroglial Nav1.2 and one on microglial sodium current became a
    single pathway running from inflammation into myelination.

    Returns the junction nodes that no paper spans, the cell types on each
    side of them, and whether any single paper covers the whole chain.
    """
    hops = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    if not hops:
        return {"inferred_junctions": [], "spanning_papers": set(), "evidenced": True}

    per_hop = [_papers_on(graph, u, v) for u, v in hops]

    inferred: list[dict[str, Any]] = []
    for i in range(len(hops) - 1):
        shared = per_hop[i] & per_hop[i + 1]
        if shared:
            continue
        node = hops[i][1]
        inferred.append(
            {
                "node": node,
                "incoming": f"{hops[i][0]} → {hops[i][1]}",
                "outgoing": f"{hops[i + 1][0]} → {hops[i + 1][1]}",
                "incoming_papers": sorted(per_hop[i]),
                "outgoing_papers": sorted(per_hop[i + 1]),
                "incoming_cells": graph.edges[hops[i]].get("cell_types") or "",
                "outgoing_cells": graph.edges[hops[i + 1]].get("cell_types") or "",
            }
        )

    spanning = set.intersection(*per_hop) if per_hop else set()
    return {
        "inferred_junctions": inferred,
        "spanning_papers": spanning,
        "evidenced": not inferred,
    }


def get_pathway_chains(
    graph: nx.DiGraph,
    max_length: int = 8,
    limit: int = 50,
    *,
    evidenced_only: bool = True,
) -> list[list[str]]:
    """
    Extract mechanistic chains through the graph, longest first.

    Backward edges are removed first so enumeration terminates on corpora that
    contain feedback loops.

    By default only chains whose every junction is spanned by at least one
    paper are returned. `evidenced_only=False` restores the old behaviour of
    enumerating every simple path, which is useful for asking "what might
    connect" but must not be presented as what the literature says — use
    `pathway_chains_with_provenance` if you need both.
    """
    if graph.number_of_nodes() == 0:
        return []

    dag = nx.DiGraph()
    dag.add_nodes_from(graph.nodes(data=True))
    for u, v, attrs in graph.edges(data=True):
        if not attrs.get("is_backward"):
            dag.add_edge(u, v)

    while not nx.is_directed_acyclic_graph(dag):
        try:
            cycle = nx.find_cycle(dag, orientation="original")
        except nx.NetworkXNoCycle:
            break
        dag.remove_edge(cycle[-1][0], cycle[-1][1])

    sources = [n for n in dag.nodes if dag.in_degree(n) == 0]
    sinks = [n for n in dag.nodes if dag.out_degree(n) == 0]

    paths: list[tuple[str, ...]] = []
    for source in sources:
        for sink in sinks:
            if source == sink:
                continue
            try:
                for path in nx.all_simple_paths(dag, source, sink, cutoff=max_length):
                    paths.append(tuple(path))
                    if len(paths) >= limit * 4:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if len(paths) >= limit * 4:
            break

    unique = sorted(set(paths), key=len, reverse=True)
    chains = [list(p) for p in unique]
    if evidenced_only:
        chains = [c for c in chains if chain_provenance(graph, c)["evidenced"]]
    return chains[:limit]


def pathway_chains_with_provenance(
    graph: nx.DiGraph, max_length: int = 8, limit: int = 50
) -> list[dict[str, Any]]:
    """
    Every enumerated chain, each labelled with what actually supports it.

    The UI needs both halves: the chains a paper vouches for end to end, and
    the ones the graph merely permits. Returning them together, sorted so the
    evidenced ones come first, means a spliced path can still be explored —
    it just cannot be mistaken for a result.
    """
    out: list[dict[str, Any]] = []
    for chain in get_pathway_chains(
        graph, max_length=max_length, limit=limit * 4, evidenced_only=False
    ):
        provenance = chain_provenance(graph, chain)
        out.append(
            {
                "chain": chain,
                "n_hops": len(chain) - 1,
                "evidenced": provenance["evidenced"],
                "spanning_papers": sorted(provenance["spanning_papers"]),
                "inferred_junctions": provenance["inferred_junctions"],
            }
        )
    out.sort(key=lambda c: (not c["evidenced"], -c["n_hops"]))
    return out[:limit]


def graph_statistics(graph: nx.DiGraph) -> dict[str, Any]:
    """Structural summary of the displayed graph."""
    if graph.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0, "lanes": {}, "n_sources": 0, "n_sinks": 0}

    lanes: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        level = attrs.get("level", "Unknown")
        lanes[level] = lanes.get(level, 0) + 1

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "lanes": {lv: lanes.get(lv, 0) for lv in KE_LEVEL_ORDER if lanes.get(lv)},
        "n_sources": sum(1 for n in graph.nodes if graph.in_degree(n) == 0),
        "n_sinks": sum(1 for n in graph.nodes if graph.out_degree(n) == 0),
        "n_adjacent_edges": sum(
            1 for _, _, a in graph.edges(data=True) if a.get("adjacency") == "Adjacent"
        ),
        "n_backward_edges": sum(
            1 for _, _, a in graph.edges(data=True) if a.get("is_backward")
        ),
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def export_graph_json(
    graph: nx.DiGraph,
    table2_df: Optional[pd.DataFrame] = None,
    positions: Optional[dict[str, dict[str, float]]] = None,
) -> str:
    """Complete graph, layout and Table 2 payload as JSON."""
    positions = positions or {}

    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        pos = positions.get(node_id, {})
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "level": attrs.get("level", "Unknown"),
                "lane": attrs.get("lane"),
                "x": pos.get("x"),
                "y": pos.get("y"),
                "pinned": pos.get("pinned", False),
                "canonical_id": attrs.get("canonical_id"),
                "ontology_curie": attrs.get("ontology_curie"),
                "ontology_iri": attrs.get("ontology_iri"),
                "aliases": attrs.get("aliases"),
                "curation_status": attrs.get("curation_status"),
                "physio_links": attrs.get("physio_links", []),
            }
        )

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append(
            {
                "source": source,
                "target": target,
                "ker_key": attrs.get("ker_key"),
                "label": attrs.get("ker_name", ""),
                "adjacency": attrs.get("adjacency"),
                "n_supporting": attrs.get("n_supporting", 0),
                "n_contradicting": attrs.get("n_contradicting", 0),
                "confidence_score": attrs.get("confidence_score"),
                "confidence_band": attrs.get("confidence_band"),
                "uncertainty_level": attrs.get("uncertainty_level"),
                "aop_status": attrs.get("aop_status"),
                "n_evidence_spans": attrs.get("n_evidence_spans", 0),
                "n_verified_spans": attrs.get("n_verified_spans", 0),
                "curation_status": attrs.get("curation_status"),
            }
        )

    payload = {
        "format": "ai4aop-pathway/2",
        "lane_order": list(KE_LEVEL_ORDER),
        "graph_metadata": graph_statistics(graph),
        "nodes": nodes,
        "edges": edges,
        "table2_data": table2_df.to_dict("records") if table2_df is not None and not table2_df.empty else [],
    }
    return json.dumps(payload, indent=2, default=str)


def export_graph_csv(
    graph: nx.DiGraph,
    table2_df: Optional[pd.DataFrame] = None,
    positions: Optional[dict[str, dict[str, float]]] = None,
) -> tuple[str, str]:
    """Nodes and edges as two CSV strings."""
    positions = positions or {}

    nodes_data = []
    for node_id, attrs in graph.nodes(data=True):
        pos = positions.get(node_id, {})
        nodes_data.append(
            {
                "ke_name": node_id,
                "level": attrs.get("level", "Unknown"),
                "lane": attrs.get("lane"),
                "x": pos.get("x"),
                "y": pos.get("y"),
                "canonical_id": attrs.get("canonical_id"),
                "ontology_curie": attrs.get("ontology_curie"),
                "ontology_label": attrs.get("ontology_label"),
                "aliases": attrs.get("aliases"),
                "curation_status": attrs.get("curation_status"),
            }
        )

    edges_data = []
    for source, target, attrs in graph.edges(data=True):
        edges_data.append(
            {
                "upstream_ke": source,
                "downstream_ke": target,
                "ker_key": attrs.get("ker_key"),
                "ker_name": attrs.get("ker_name", ""),
                "adjacency": attrs.get("adjacency"),
                "n_supporting_papers": attrs.get("n_supporting", 0),
                "n_contradicting_papers": attrs.get("n_contradicting", 0),
                "confidence_score": attrs.get("confidence_score"),
                "confidence_band": attrs.get("confidence_band"),
                "uncertainty_level": attrs.get("uncertainty_level"),
                "aop_status": attrs.get("aop_status"),
                "n_evidence_spans": attrs.get("n_evidence_spans", 0),
                "n_verified_spans": attrs.get("n_verified_spans", 0),
                "source_dois": attrs.get("all_source_dois"),
                "curation_status": attrs.get("curation_status"),
            }
        )

    return (
        pd.DataFrame(nodes_data).to_csv(index=False),
        pd.DataFrame(edges_data).to_csv(index=False),
    )


def export_graph_png(
    graph: nx.DiGraph,
    positions: Optional[dict[str, dict[str, float]]] = None,
    *,
    width: int = 26,
    height: int = 16,
    dpi: int = 120,
) -> bytes:
    """
    Render the layered layout to PNG, preserving lane positions.

    Unlike the old spring-layout export, this produces the same picture as the
    interactive view, so a figure taken for a report matches what the curator
    approved on screen.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        from io import BytesIO
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for PNG export. Install with: pip install matplotlib"
        ) from exc

    if graph.number_of_nodes() == 0:
        raise ValueError("No graph data to export")

    positions = positions or compute_layered_positions(graph)
    # Canvas y grows downward; matplotlib grows upward.
    pos = {n: (p["x"], -p["y"]) for n, p in positions.items() if n in graph.nodes}

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_facecolor("white")
    ax.axis("off")

    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]

    # Lane bands + headers.
    lanes = sorted({attrs.get("lane", 1) for _, attrs in graph.nodes(data=True)})
    for lane in lanes:
        name = KE_LEVEL_ORDER[lane] if lane < len(KE_LEVEL_ORDER) else "Other"
        x = lane * LANE_WIDTH
        ax.axvspan(
            x - LANE_WIDTH / 2, x + LANE_WIDTH / 2,
            color=KE_LEVEL_BANDS.get(name, "#f4f6f6"), zorder=0,
        )
        ax.text(
            x, max(ys) + ROW_HEIGHT * 1.1, name.upper(),
            ha="center", va="bottom", fontsize=13, weight="bold", color="#5d6d7e", zorder=4,
        )

    for source, target, attrs in graph.edges(data=True):
        if source not in pos or target not in pos:
            continue
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        dashed = attrs.get("adjacency") == "Non-adjacent" or attrs.get("is_backward")
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>", mutation_scale=14,
                linewidth=max(0.7, attrs.get("width", 2) * 0.45),
                color=attrs.get("color", "#95a5a6"),
                linestyle="--" if dashed else "-",
                alpha=0.45 if dashed else 0.8,
                connectionstyle="arc3,rad=0.12",
                shrinkA=42, shrinkB=42, zorder=1,
            )
        )

    for node, (x, y) in pos.items():
        attrs = graph.nodes[node]
        label = str(node)
        wrapped = _wrap_label(label, width=20, max_lines=4)
        ax.text(
            x, y, wrapped,
            ha="center", va="center", fontsize=7.5, weight="bold", color="#1b2631",
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor=attrs.get("color", "#95a5a6"),
                edgecolor="#2c3e50", linewidth=0.9, alpha=0.95,
            ),
            zorder=3,
        )

    margin_x, margin_y = LANE_WIDTH * 0.7, ROW_HEIGHT * 1.6
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y * 1.4)

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.3)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


__all__ = [
    "KE_LEVEL_COLORS",
    "build_pathway_graph",
    "compute_layered_positions",
    "find_backbone",
    "filter_graph",
    "expandable_neighbourhood",
    "render_interactive_graph",
    "get_pathway_chains",
    "pathway_chains_with_provenance",
    "chain_provenance",
    "graph_statistics",
    "export_graph_json",
    "export_graph_csv",
    "export_graph_png",
]
